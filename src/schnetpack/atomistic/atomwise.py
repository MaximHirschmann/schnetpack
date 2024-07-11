from typing import Sequence, Union, Callable, Dict, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

import schnetpack as spk
import schnetpack.nn as snn
import schnetpack.properties as properties

__all__ = ["Atomwise", "DipoleMoment", "Polarizability", 
           "Hessian", "Hessian2", "Hessian3", "Hessian4", "Hessian5", "Hessian6", 
           "NewtonStep", "BestDirection", "Forces2", "HessianDiagonal"]


class Atomwise(nn.Module):
    """
    Predicts atom-wise contributions and accumulates global prediction, e.g. for the energy.

    If `aggregation_mode` is None, only the per-atom predictions will be returned.
    """

    def __init__(
        self,
        n_in: int,
        n_out: int = 1,
        n_hidden: Optional[Union[int, Sequence[int]]] = None,
        n_layers: int = 2,
        activation: Callable = F.silu,
        aggregation_mode: str = "sum",
        output_key: str = "y",
        per_atom_output_key: Optional[str] = None,
    ):
        """
        Args:
            n_in: input dimension of representation
            n_out: output dimension of target property (default: 1)
            n_hidden: size of hidden layers.
                If an integer, same number of node is used for all hidden layers resulting
                in a rectangular network.
                If None, the number of neurons is divided by two after each layer starting
                n_in resulting in a pyramidal network.
            n_layers: number of layers.
            aggregation_mode: one of {sum, avg} (default: sum)
            output_key: the key under which the result will be stored
            per_atom_output_key: If not None, the key under which the per-atom result will be stored
        """
        super(Atomwise, self).__init__()
        self.output_key = output_key
        self.model_outputs = [output_key]
        self.per_atom_output_key = per_atom_output_key
        if self.per_atom_output_key is not None:
            self.model_outputs.append(self.per_atom_output_key)
        self.n_out = n_out

        if aggregation_mode is None and self.per_atom_output_key is None:
            raise ValueError(
                "If `aggregation_mode` is None, `per_atom_output_key` needs to be set,"
                + " since no accumulated output will be returned!"
            )

        self.outnet = spk.nn.build_mlp(
            n_in=n_in,
            n_out=n_out,
            n_hidden=n_hidden,
            n_layers=n_layers,
            activation=activation,
        )
        self.aggregation_mode = aggregation_mode

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # predict atomwise contributions
        y = self.outnet(inputs["scalar_representation"])

        # accumulate the per-atom output if necessary
        if self.per_atom_output_key is not None:
            inputs[self.per_atom_output_key] = y

        # aggregate
        if self.aggregation_mode is not None:
            idx_m = inputs[properties.idx_m]
            maxm = int(idx_m[-1]) + 1
            y = snn.scatter_add(y, idx_m, dim_size=maxm)
            y = torch.squeeze(y, -1)

            if self.aggregation_mode == "avg":
                y = y / inputs[properties.n_atoms]

        inputs[self.output_key] = y
        return inputs


class DipoleMoment(nn.Module):
    """
    Predicts dipole moments from latent partial charges and (optionally) local, atomic dipoles.
    The latter requires a representation supplying (equivariant) vector features.

    References:

    .. [#painn1] Schütt, Unke, Gastegger.
       Equivariant message passing for the prediction of tensorial properties and molecular spectra.
       ICML 2021, http://proceedings.mlr.press/v139/schutt21a.html
    .. [#irspec] Gastegger, Behler, Marquetand.
       Machine learning molecular dynamics for the simulation of infrared spectra.
       Chemical science 8.10 (2017): 6924-6935.
    .. [#dipole] Veit et al.
       Predicting molecular dipole moments by combining atomic partial charges and atomic dipoles.
       The Journal of Chemical Physics 153.2 (2020): 024113.
    """

    def __init__(
        self,
        n_in: int,
        n_hidden: Optional[Union[int, Sequence[int]]] = None,
        n_layers: int = 2,
        activation: Callable = F.silu,
        predict_magnitude: bool = False,
        return_charges: bool = False,
        dipole_key: str = properties.dipole_moment,
        charges_key: str = properties.partial_charges,
        correct_charges: bool = True,
        use_vector_representation: bool = False,
    ):
        """
        Args:
            n_in: input dimension of representation
            n_hidden: size of hidden layers.
                If an integer, same number of node is used for all hidden layers
                resulting in a rectangular network.
                If None, the number of neurons is divided by two after each layer
                starting n_in resulting in a pyramidal network.
            n_layers: number of layers.
            activation: activation function
            predict_magnitude: If true, calculate magnitude of dipole
            return_charges: If true, return latent partial charges
            dipole_key: the key under which the dipoles will be stored
            charges_key: the key under which partial charges will be stored
            correct_charges: If true, forces the sum of partial charges to be the total
                charge, if provided, and zero otherwise.
            use_vector_representation: If true, use vector representation to predict
                local, atomic dipoles.
        """
        super().__init__()

        self.dipole_key = dipole_key
        self.charges_key = charges_key
        self.return_charges = return_charges
        self.model_outputs = [dipole_key]
        if self.return_charges:
            self.model_outputs.append(charges_key)

        self.predict_magnitude = predict_magnitude
        self.use_vector_representation = use_vector_representation
        self.correct_charges = correct_charges

        if use_vector_representation:
            self.outnet = spk.nn.build_gated_equivariant_mlp(
                n_in=n_in,
                n_out=1,
                n_hidden=n_hidden,
                n_layers=n_layers,
                activation=activation,
                sactivation=activation,
            )
        else:
            self.outnet = spk.nn.build_mlp(
                n_in=n_in,
                n_out=1,
                n_hidden=n_hidden,
                n_layers=n_layers,
                activation=activation,
            )

    def forward(self, inputs):
        positions = inputs[properties.R]
        l0 = inputs["scalar_representation"]
        natoms = inputs[properties.n_atoms]
        idx_m = inputs[properties.idx_m]
        maxm = int(idx_m[-1]) + 1

        if self.use_vector_representation:
            l1 = inputs["vector_representation"]
            charges, atomic_dipoles = self.outnet((l0, l1))
            atomic_dipoles = torch.squeeze(atomic_dipoles, -1)
        else:
            charges = self.outnet(l0)
            atomic_dipoles = 0.0

        if self.correct_charges:
            sum_charge = snn.scatter_add(charges, idx_m, dim_size=maxm)

            if properties.total_charge in inputs:
                total_charge = inputs[properties.total_charge][:, None]
            else:
                total_charge = torch.zeros_like(sum_charge)

            charge_correction = (total_charge - sum_charge) / natoms.unsqueeze(-1)
            charge_correction = charge_correction[idx_m]
            charges = charges + charge_correction

        if self.return_charges:
            inputs[self.charges_key] = charges

        y = positions * charges
        if self.use_vector_representation:
            y = y + atomic_dipoles

        # sum over atoms
        y = snn.scatter_add(y, idx_m, dim_size=maxm)

        if self.predict_magnitude:
            y = torch.norm(y, dim=1, keepdim=False)

        inputs[self.dipole_key] = y
        return inputs


class Polarizability(nn.Module):
    """
    Predicts polarizability tensor using tensor rank factorization.
    This requires an equivariant representation, e.g. PaiNN, that provides both scalar and vectorial features.

    References:

    .. [#painn1a] Schütt, Unke, Gastegger:
       Equivariant message passing for the prediction of tensorial properties and molecular spectra.
       ICML 2021, http://proceedings.mlr.press/v139/schutt21a.html
    """

    def __init__(
        self,
        n_in: int,
        n_hidden: Optional[Union[int, Sequence[int]]] = None,
        n_layers: int = 2,
        activation: Callable = F.silu,
        polarizability_key: str = properties.polarizability,
    ):
        """
        Args:
            n_in: input dimension of representation
            n_hidden: size of hidden layers.
                If an integer, same number of node is used for all hidden layers resulting
                in a rectangular network.
                If None, the number of neurons is divided by two after each layer starting
                n_in resulting in a pyramidal network.
            n_layers: number of layers.
            activation: activation function
            polarizability_key: the key under which the predicted polarizability will be stored
        """
        super(Polarizability, self).__init__()
        self.n_in = n_in
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.polarizability_key = polarizability_key
        self.model_outputs = [polarizability_key]

        self.outnet = spk.nn.build_gated_equivariant_mlp(
            n_in=n_in,
            n_out=1,
            n_hidden=n_hidden,
            n_layers=n_layers,
            activation=activation,
            sactivation=activation,
        )

        self.requires_dr = False
        self.requires_stress = False

    def forward(self, inputs):
        positions = inputs[properties.R]
        l0 = inputs["scalar_representation"]
        l1 = inputs["vector_representation"]
        dim = l1.shape[-2]

        l0, l1 = self.outnet((l0, l1))

        # isotropic on diagonal
        alpha = l0[..., 0:1]
        size = list(alpha.shape)
        size[-1] = dim
        alpha = alpha.expand(*size)
        alpha = torch.diag_embed(alpha)

        # add anisotropic components
        mur = l1[..., None, 0] * positions[..., None, :]
        alpha_c = mur + mur.transpose(-2, -1)
        alpha = alpha + alpha_c

        # sum over atoms
        idx_m = inputs[properties.idx_m]
        maxm = int(idx_m[-1]) + 1
        alpha = snn.scatter_add(alpha, idx_m, dim_size=maxm)

        inputs[self.polarizability_key] = alpha
        return inputs



class Hessian(nn.Module):
    """
    Using combination of l0, l1 features to build 27 x 27 matrix and 
    then run one big linear layer.
    """
    def __init__(
        self,
        n_in: int,
        n_hidden: Optional[Union[int, Sequence[int]]] = None,
        n_layers: int = 2,
        activation: Callable = F.silu,
        hessian_key: str = properties.hessian,
        final_linear: bool = False,
    ):
        super(Hessian, self).__init__()
        self.n_in = n_in
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.hessian_key = hessian_key
        self.model_outputs = [hessian_key]
        self.final_linear = final_linear
        self.final_linear_layer = nn.Linear(27 * 27, 27 * 27)
        
        self.outnet = spk.nn.build_gated_equivariant_mlp(
            n_in=n_in,
            n_out=1,
            n_hidden=n_hidden,
            n_layers=n_layers,
            activation=activation,
            sactivation=activation,
        )
        
    def forward(self, inputs):
        positions = inputs[properties.R] # 90 x 3
        l0 = inputs["scalar_representation"] # 90 x 30
        l1 = inputs["vector_representation"] # 90 x 3 x 30

        l0, l1 = self.outnet((l0, l1)) # 90 x 1, 90 x 3 x 1

        # isotropic on diagonal
        scalar_features = l0[..., 0:1] # 90 x 1
        size = list(scalar_features.shape)
        size[-1] = 27
        scalar_features = scalar_features.expand(*size) # 90 x 27
        scalar_features = torch.diag_embed(scalar_features) # 90 x 27 x 27

        # anisotropic components
        mur = l1[..., None, 0] * positions[..., None, :] # 90 x 3 x 1  *  90 x 1 x 3 = 90 x 3 x 3
        mur_1 = mur.view(-1, 1, 9) # 90 x 1 x 9
        mur_2 = mur.view(-1, 9, 1) # 90 x 9 x 1
        
        temp1 = (mur_1 * l1[..., None, 0]).view(-1, 27, 1) # 90 x 27 x 1
        temp2 = (mur_2 * positions[..., None, :]).view(-1, 1, 27) # 90 x 1 x 27
        
        temp3 = temp2 * temp1 + scalar_features # 90 x 27 x 27
        
        idx_m = inputs[properties.idx_m]
        maxm = int(idx_m[-1]) + 1
        hessian = snn.scatter_add(temp3, idx_m, dim_size=maxm) # 10 x 27 x 27
        
        if self.final_linear:
            hessian = hessian.view(-1, 27 * 27)
            hessian = self.final_linear_layer(hessian)
            hessian = hessian.view(-1, 27, 27)
        
        # shape batch_size x 27 x 27 to (batch_size * 27) x 27
        hessian = hessian.view(-1, 27)
        
        inputs[self.hessian_key] = hessian
        return inputs


class Hessian2(nn.Module):
    """
    Applies linear layers to scalar, vector and positions features to get multiple 27 x 27 matrices
    and then sum them up.
    """
    def __init__(
        self,
        n_in: int,
        n_hidden: Optional[Union[int, Sequence[int]]] = None,
        n_layers: int = 2,
        activation: Callable = F.silu,
        hessian_key: str = properties.hessian,
    ):
        super(Hessian2, self).__init__()
        self.n_in = n_in
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.hessian_key = hessian_key
        self.model_outputs = [hessian_key]

        self.outnet = spk.nn.build_gated_equivariant_mlp(
            n_in=n_in,
            n_out=1,
            n_hidden=n_hidden,
            n_layers=n_layers,
            activation=activation,
            sactivation=activation,
        )
        
        self.fnn_s = nn.Sequential(
            nn.Linear(1, 30),
            nn.SiLU(),
            nn.Linear(30, 27 * 27)
        )
        
        self.fnn_v = nn.Sequential(
            nn.Linear(3, 30),
            nn.SiLU(),
            nn.Linear(30, 27 * 27)
        )
        
        self.fnn_p = nn.Sequential(
            nn.Linear(3, 30),
            nn.SiLU(),
            nn.Linear(30, 27 * 27)
        )
    
    def forward(self, inputs):
        positions = inputs[properties.R] # 90 x 3
        l0 = inputs["scalar_representation"] # 90 x 30
        l1 = inputs["vector_representation"] # 90 x 3 x 30

        l0, l1 = self.outnet((l0, l1)) # 90 x 1, 90 x 3 x 1

        p = self.fnn_p(positions)
        s = self.fnn_s(l0)
        l1 = l1.squeeze(-1)
        v = self.fnn_v(l1)
        
        hessian = p + s + v
        hessian = hessian.view(-1, 27, 27)
        hessian = hessian + hessian.transpose(-2, -1) - torch.diag_embed(torch.diagonal(hessian, dim1=-2, dim2=-1))
        
        idx_m = inputs[properties.idx_m]
        maxm = int(idx_m[-1]) + 1
        hessian = snn.scatter_add(hessian, idx_m, dim_size=maxm) # 10 x 27 x 27
        
        hessian = hessian.view(-1, 27)
        
        inputs[self.hessian_key] = hessian
        return inputs


class Hessian3(nn.Module):
    """
    Creates 3x3 mini hessians for each pair of atoms by applying multiple linear layers to l0, l1, s features
    and then cocatenates them to get 27 x 27 hessian.
    """
    def __init__(
        self,
        n_in: int,
        n_hidden: Optional[Union[int, Sequence[int]]] = None,
        n_layers: int = 2,
        activation: Callable = F.silu,
        hessian_key: str = properties.hessian,
    ):
        super(Hessian3, self).__init__()
        self.n_in = n_in
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.hessian_key = hessian_key
        self.model_outputs = [hessian_key]
    
        self.outnet = spk.nn.build_gated_equivariant_mlp(
            n_in=n_in,
            n_out=1,
            n_hidden=n_hidden,
            n_layers=n_layers,
            activation=activation,
            sactivation=activation,
        )
        
        self.fnn_v_v = nn.Sequential(
            nn.Linear(9, 30),
            nn.SiLU(),
            nn.Linear(30, 9)
        )
        
        self.fnn_v_r = nn.Sequential(
            nn.Linear(9, 30),
            nn.SiLU(),
            nn.Linear(30, 9)
        )
        
        self.fnn_s = nn.Sequential(
            nn.Linear(2, 30),
            nn.SiLU(),
            nn.Linear(30, 9)
        )
        
        self.fnn_h = nn.Sequential(
            nn.Linear(9, 30),
            nn.SiLU(),
            nn.Linear(30, 9)
        )
        
    def forward(self, inputs):
        positions = inputs[properties.R] # 90 x 3
        l0 = inputs["scalar_representation"] # 90 x 30
        l1 = inputs["vector_representation"] # 90 x 3 x 30

        l0, l1 = self.outnet((l0, l1)) # 90 x 1, 90 x 3 x 1
        
        l1 = l1.squeeze(-1) # 90 x 3
        
        v_v = torch.einsum('ik,jl->ijkl', l1, l1) # 90 x 90 x 3 x 3
        v_r = torch.einsum('ik,jl->ijkl', l1, positions) # 90 x 90 x 3 x 3
        
        n_atoms = inputs[properties.n_atoms]
        hessians: List[torch.Tensor] = []
        
        indices = torch.cumsum(n_atoms, dim=0) - n_atoms
        for start_idx, n_atom in zip(indices, n_atoms):
            end_idx = start_idx + n_atom
            v_v_i = v_v[start_idx:end_idx, start_idx:end_idx] # n_atom x n_atom x 3 x 3
            v_r_i = v_r[start_idx:end_idx, start_idx:end_idx] # n_atom x n_atom x 3 x 3
            
            v_v_i = self.fnn_v_v(v_v_i.contiguous().view(-1, 9)).view(n_atom, n_atom, 3, 3) # n_atom x n_atom x 3 x 3
            v_r_i = self.fnn_v_r(v_r_i.contiguous().view(-1, 9)).view(n_atom, n_atom, 3, 3) # n_atom x n_atom x 3 x 3
            
            i, j = torch.meshgrid(torch.arange(n_atom), torch.arange(n_atom), indexing='ij')
            pairs = torch.stack((i.flatten(), j.flatten()), dim=1)
            s_i = l0[start_idx:end_idx].squeeze(-1) # n_atom
            s_i = s_i[pairs] # (n_atom * n_atom) x 2
            s_i = self.fnn_s(s_i).view(n_atom, n_atom, 3, 3) # n_atom x n_atom x 3 x 3
            
            mini_hessian = v_v_i + v_r_i + s_i # n_atom x n_atom x 3 x 3
            mini_hessian = self.fnn_h(mini_hessian.contiguous().view(-1, 9)).view(n_atom, n_atom, 3, 3) # n_atom x n_atom x 3 x 3
            mini_hessian = mini_hessian.permute(0, 2, 1, 3).reshape(n_atom * 3, n_atom * 3) # (3 * n_atom) x (3 * n_atom)
            
            hessians.append(mini_hessian)
            
        # ethanol specific reshape
        hessians = torch.stack(hessians, dim = 0).view(-1, 27)
        
        inputs[self.hessian_key] = hessians
        return inputs
    
class Hessian4(nn.Module):
    """
    calculate 3x3 mini hessians using one mlp which takes in l0, l1, positions and
    then concatenate them to get 27x27 hessian.
    """
    def __init__(
        self,
        n_in: int,
        n_hidden: Optional[Union[int, Sequence[int]]] = None,
        n_layers: int = 2,
        activation: Callable = F.silu,
        hessian_key: str = properties.hessian,
    ):
        super(Hessian4, self).__init__()
        self.n_in = n_in
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.hessian_key = hessian_key
        self.model_outputs = [hessian_key]
    
        self.outnet = spk.nn.build_gated_equivariant_mlp(
            n_in=n_in,
            n_out=1,
            n_hidden=n_hidden,
            n_layers=n_layers,
            activation=activation,
            sactivation=activation,
        )
        
        self.fnn = nn.Sequential(
            nn.Linear(3 + 3 + 3 + 3 + 1 + 1, 30),
            nn.SiLU(),
            nn.Linear(30, 9)
        )
        
    def forward(self, inputs):
        positions = inputs[properties.R] # 90 x 3
        l0 = inputs["scalar_representation"] # 90 x 30
        l1 = inputs["vector_representation"] # 90 x 3 x 30

        l0, l1 = self.outnet((l0, l1)) # 90 x 1, 90 x 3 x 1
        
        l1 = l1.squeeze(-1)
        
        n_atoms = inputs[properties.n_atoms]
        hessians: List[torch.Tensor] = []
        
        indices = torch.cumsum(n_atoms, dim=0) - n_atoms
        for start_idx, n_atom in zip(indices, n_atoms):
            end_idx = start_idx + n_atom
            
            i, j = torch.meshgrid(torch.arange(n_atom), torch.arange(n_atom), indexing='ij')
            pairs = torch.stack((i.flatten(), j.flatten()), dim=1)
            
            s_i = l0[start_idx:end_idx].squeeze(-1) # n_atom
            s_i = s_i[pairs] # (n_atom * n_atom) x 2
            
            v_i = l1[start_idx:end_idx] # n_atom x 3
            v_i = v_i[pairs] # (n_atom * n_atom) x 2 x 3 
            v_i = v_i.view(-1, 6) # (n_atom * n_atom) x 6
            
            r_i = positions[start_idx:end_idx] # n_atom x 3
            r_i = r_i[pairs] # (n_atom * n_atom) x 2 x 3
            r_i = r_i.view(-1, 6) # (n_atom * n_atom) x 6
            
            # concatenate
            mini_hessian = torch.cat((s_i, v_i, r_i), dim=1) # (n_atom * n_atom) x 14
            mini_hessian = self.fnn(mini_hessian).view(n_atom, n_atom, 3, 3) # n_atom x n_atom x 3 x 3
            mini_hessian = mini_hessian.permute(0, 2, 1, 3).reshape(n_atom * 3, n_atom * 3) # (3 * n_atom) x (3 * n_atom)
            
            hessians.append(mini_hessian)
            
        # ethanol specific reshape
        hessians = torch.stack(hessians, dim = 0).view(-1, 27)
        
        inputs[self.hessian_key] = hessians
        return inputs
    

class Hessian5(nn.Module):
    """
    """
    def __init__(
        self,
        n_in: int,
        n_hidden: Optional[Union[int, Sequence[int]]] = None,
        n_layers: int = 2,
        activation: Callable = F.silu,
        hessian_key: str = properties.hessian,
    ):
        super(Hessian5, self).__init__()
        self.n_in = n_in
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.hessian_key = hessian_key
        self.model_outputs = [hessian_key]
    
        self.outnet = spk.nn.build_gated_equivariant_mlp(
            n_in=n_in,
            n_out=1,
            n_hidden=n_hidden,
            n_layers=n_layers,
            activation=activation,
            sactivation=activation,
        )
        self.fnn_v_v = nn.Sequential(
            nn.Linear(9, 30),
            nn.SiLU(),
            nn.Linear(30, 9)
        )
        
        self.fnn_v_r = nn.Sequential(
            nn.Linear(9, 30),
            nn.SiLU(),
            nn.Linear(30, 9)
        )
        
        self.fnn_s = nn.Sequential(
            nn.Linear(2, 30),
            nn.SiLU(),
            nn.Linear(30, 9)
        )
        
        self.fnn_h = nn.Sequential(
            nn.Linear(9, 30),
            nn.SiLU(),
            nn.Linear(30, 9)
        )
        
        
    def forward(self, inputs):
        positions = inputs[properties.R] # 90 x 3
        l0 = inputs["scalar_representation"] # 90 x 30
        l1 = inputs["vector_representation"] # 90 x 3 x 30

        l0, l1 = self.outnet((l0, l1)) # 90 x 1, 90 x 3 x 1
        
        l1 = l1.squeeze(-1) # 90 x 3
        
        n_atoms = inputs[properties.n_atoms]
        hessians: List[torch.Tensor] = []
        
        indices = torch.cumsum(n_atoms, dim=0) - n_atoms
        for start_idx, n_atom in zip(indices, n_atoms):
            end_idx = start_idx + n_atom
            
            v_v_i = torch.einsum('ik,jl->ijkl', l1[start_idx:end_idx], l1[start_idx:end_idx]) # n_atom x n_atom x 3 x 3
            v_r_i = torch.einsum('ik,jl->ijkl', l1[start_idx:end_idx], positions[start_idx:end_idx]) # n_atom x n_atom x 3 x 3
            
            v_v_i = self.fnn_v_v(v_v_i.contiguous().view(-1, 9)).view(n_atom, n_atom, 3, 3) # n_atom x n_atom x 3 x 3
            v_r_i = self.fnn_v_r(v_r_i.contiguous().view(-1, 9)).view(n_atom, n_atom, 3, 3) # n_atom x n_atom x 3 x 3
            
            i, j = torch.meshgrid(torch.arange(n_atom), torch.arange(n_atom), indexing='ij')
            pairs = torch.stack((i.flatten(), j.flatten()), dim=1)
            s_i = l0[start_idx:end_idx].squeeze(-1) # n_atom
            s_i = s_i[pairs] # (n_atom * n_atom) x 2
            s_i = self.fnn_s(s_i).view(n_atom, n_atom, 3, 3) # n_atom x n_atom x 3 x 3
            
            mini_hessian = v_v_i + v_r_i + s_i # n_atom x n_atom x 3 x 3
            mini_hessian = self.fnn_h(mini_hessian.contiguous().view(-1, 9)).view(n_atom, n_atom, 3, 3) # n_atom x n_atom x 3 x 3
            mini_hessian = mini_hessian.permute(0, 2, 1, 3).reshape(n_atom * 3, n_atom * 3) # (3 * n_atom) x (3 * n_atom)
            
            hessians.append(mini_hessian)
            
        # ethanol specific reshape
        hessians = torch.stack(hessians, dim = 0).view(-1, 27)
        
        inputs[self.hessian_key] = hessians
        return inputs
    
class Hessian6(nn.Module):
    def __init__(
        self,
        n_in: int,
        n_hidden: Optional[Union[int, Sequence[int]]] = None,
        n_layers: int = 2,
        activation: Callable = F.silu,
        hessian_key: str = properties.hessian,
    ):
        super(Hessian6, self).__init__()
        self.n_in = n_in
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.hessian_key = hessian_key
        self.model_outputs = [hessian_key]
    
        self.outnet = spk.nn.build_gated_equivariant_mlp(
            n_in=n_in,
            n_out=1,
            n_hidden=n_hidden,
            n_layers=n_layers,
            activation=activation,
            sactivation=activation,
        )
        self.fnn_v_v = nn.Sequential(
            nn.Linear(9, 30),
            nn.SiLU(),
            nn.Linear(30, 9)
        )
        
        self.fnn_v_r = nn.Sequential(
            nn.Linear(9, 30),
            nn.SiLU(),
            nn.Linear(30, 9)
        )
        
        self.fnn_s = nn.Sequential(
            nn.Linear(2, 30),
            nn.SiLU(),
            nn.Linear(30, 9)
        )
        
        self.fnn_h = nn.Sequential(
            nn.Linear(9, 30),
            nn.SiLU(),
            nn.Linear(30, 9)
        )
        
        self.fnn_final = nn.Sequential(
            nn.Linear(27 * 27, 27 * 27)
        )
        
        
    def forward(self, inputs):
        positions = inputs[properties.R] # 90 x 3
        l0 = inputs["scalar_representation"] # 90 x 30
        l1 = inputs["vector_representation"] # 90 x 3 x 30

        l0, l1 = self.outnet((l0, l1)) # 90 x 1, 90 x 3 x 1
        
        l1 = l1.squeeze(-1) # 90 x 3
        
        n_atoms = inputs[properties.n_atoms]
        hessians: List[torch.Tensor] = []
        
        indices = torch.cumsum(n_atoms, dim=0) - n_atoms
        for start_idx, n_atom in zip(indices, n_atoms):
            end_idx = start_idx + n_atom
            
            v_v_i = torch.einsum('ik,jl->ijkl', l1[start_idx:end_idx], l1[start_idx:end_idx]) # n_atom x n_atom x 3 x 3
            v_r_i = torch.einsum('ik,jl->ijkl', l1[start_idx:end_idx], positions[start_idx:end_idx]) # n_atom x n_atom x 3 x 3
            
            v_v_i = self.fnn_v_v(v_v_i.contiguous().view(-1, 9)).view(n_atom, n_atom, 3, 3) # n_atom x n_atom x 3 x 3
            v_r_i = self.fnn_v_r(v_r_i.contiguous().view(-1, 9)).view(n_atom, n_atom, 3, 3) # n_atom x n_atom x 3 x 3
            
            i, j = torch.meshgrid(torch.arange(n_atom), torch.arange(n_atom), indexing='ij')
            pairs = torch.stack((i.flatten(), j.flatten()), dim=1)
            s_i = l0[start_idx:end_idx].squeeze(-1) # n_atom
            s_i = s_i[pairs] # (n_atom * n_atom) x 2
            s_i = self.fnn_s(s_i).view(n_atom, n_atom, 3, 3) # n_atom x n_atom x 3 x 3
            
            mini_hessian = v_v_i + v_r_i + s_i # n_atom x n_atom x 3 x 3
            mini_hessian = self.fnn_h(mini_hessian.contiguous().view(-1, 9)).view(n_atom, n_atom, 3, 3) # n_atom x n_atom x 3 x 3
            mini_hessian = mini_hessian.permute(0, 2, 1, 3).reshape(n_atom * 3, n_atom * 3) # (3 * n_atom) x (3 * n_atom)
            
            mini_hessian = self.fnn_final(mini_hessian.view(-1, 27 * 27)).view(n_atom * 3, n_atom * 3)
            
            hessians.append(mini_hessian)
            
        # ethanol specific reshape
        hessians = torch.stack(hessians, dim = 0).view(-1, 27)
        
        inputs[self.hessian_key] = hessians
        return inputs
    

class NewtonStep(nn.Module):
    def __init__(
        self,
        n_in: int,
        n_hidden: Optional[Union[int, Sequence[int]]] = None,
        n_layers: int = 2,
        activation: Callable = F.silu,
        newton_step_key: str = properties.newton_step,
    ):
        super(NewtonStep, self).__init__()
        self.n_in = n_in
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.newton_step_key = newton_step_key
        self.model_outputs = [newton_step_key]
    
        self.outnet = spk.nn.build_gated_equivariant_mlp(
            n_in=n_in,
            n_out=1,
            n_hidden=n_hidden,
            n_layers=n_layers,
            activation=activation,
            sactivation=activation,
        )
        
        self.dnn = nn.Sequential(
            nn.Linear(27, 30),
            nn.SiLU(),
            nn.Linear(30, 27)
        )
        
    def forward(self, inputs):
        positions = inputs[properties.R] # 90 x 3
        l0 = inputs["scalar_representation"] # 90 x 30
        l1 = inputs["vector_representation"] # 90 x 3 x 30

        l0, l1 = self.outnet((l0, l1)) # 90 x 1, 90 x 3 x 1
        
        l1 = l1.squeeze(-1) # 90 x 3
        
        # newton = self.dnn(l1.view(10, 9, 3).view(-1, 27)).view(-1, 3) # 90 x 3
        
        inputs[self.newton_step_key] = l1
        return inputs


class BestDirection(nn.Module):
    def __init__(
        self,
        n_in: int,
        n_hidden: Optional[Union[int, Sequence[int]]] = None,
        n_layers: int = 2,
        activation: Callable = F.silu,
        best_direction_key: str = "best_direction",
    ):
        super(BestDirection, self).__init__()
        self.n_in = n_in
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.best_direction_key = best_direction_key
        self.model_outputs = [best_direction_key]
    
        self.outnet = spk.nn.build_gated_equivariant_mlp(
            n_in=n_in,
            n_out=1,
            n_hidden=n_hidden,
            n_layers=n_layers,
            activation=activation,
            sactivation=activation,
        )
        
        # self.dnn = nn.Sequential(
        #     nn.Linear(27, 30),
        #     nn.SiLU(),
        #     nn.Linear(30, 27)
        # )
        
    def forward(self, inputs):
        positions = inputs[properties.R] # 90 x 3
        l0 = inputs["scalar_representation"] # 90 x 30
        l1 = inputs["vector_representation"] # 90 x 3 x 30

        l0, l1 = self.outnet((l0, l1)) # 90 x 1, 90 x 3 x 1
        
        l1 = l1.squeeze(-1) # 90 x 3
        
        
        #newton = self.dnn(l1.view(10, 9, 3).view(-1, 27)).view(-1, 3) # 90 x 3
        
        inputs[self.best_direction_key] = l1
        return inputs


class Forces2(nn.Module): 
    def __init__(
        self,
        n_in: int,
        n_hidden: Optional[Union[int, Sequence[int]]] = None,
        n_layers: int = 2,
        activation: Callable = F.silu,
        forces_copy_key: str = "forces_copy",
    ):
        super(Forces2, self).__init__()
        self.n_in = n_in
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.forces_copy_key = forces_copy_key
        self.model_outputs = [forces_copy_key]
    
        self.outnet = spk.nn.build_gated_equivariant_mlp(
            n_in=n_in,
            n_out=1,
            n_hidden=n_hidden,
            n_layers=n_layers,
            activation=activation,
            sactivation=activation,
        )
        
    def forward(self, inputs):
        positions = inputs[properties.R] # 90 x 3
        l0 = inputs["scalar_representation"] # 90 x 30
        l1 = inputs["vector_representation"] # 90 x 3 x 30

        l0, l1 = self.outnet((l0, l1)) # 90 x 1, 90 x 3 x 1
        
        l1 = l1.squeeze(-1) # 90 x 3
        
        inputs[self.forces_copy_key] = l1
        return inputs
    

class HessianDiagonal(nn.Module):
    def __init__(
        self,
        n_in: int,
        n_hidden: Optional[Union[int, Sequence[int]]] = None,
        n_layers: int = 2,
        activation: Callable = F.silu,
        diagonal_key: str = "original_diagonal",
    ):
        super(HessianDiagonal, self).__init__()
        self.n_in = n_in
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.diagonal_key = diagonal_key
        self.model_outputs = [diagonal_key]
    
        self.outnet = spk.nn.build_gated_equivariant_mlp(
            n_in=n_in,
            n_out=1,
            n_hidden=n_hidden,
            n_layers=n_layers,
            activation=activation,
            sactivation=activation,
        )
        
    def forward(self, inputs):
        positions = inputs[properties.R] # 90 x 3
        l0 = inputs["scalar_representation"] # 90 x 30
        l1 = inputs["vector_representation"] # 90 x 3 x 30

        l0, l1 = self.outnet((l0, l1)) # 90 x 1, 90 x 3 x 1
        
        l1 = l1.squeeze(-1) # 90 x 3
        l1 = 1e3 * l1
        
        idx_m = inputs[properties.idx_m]
        maxm = int(idx_m[-1]) + 1
        diagonals = []
        last = -1
        for i, idx in enumerate(idx_m):
            if idx == last:
                diagonals[-1] = torch.cat([diagonals[-1], l1[i]], dim=0)
            else:
                diagonals.append(l1[i])
                last = idx
        diagonals = torch.stack(diagonals, dim=0) # 10 x 27

        
        inputs[self.diagonal_key] = diagonals.flatten() # 10*27
        return inputs