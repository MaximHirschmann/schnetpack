from typing import Sequence, Union, Callable, Dict, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

import schnetpack as spk
import schnetpack.nn as snn
import schnetpack.properties as properties

__all__ = ["Atomwise", "DipoleMoment", "Polarizability", 
           "HessianKronecker", "HessianDUUT",
           "NewtonStep", "HessianDiagonalL0", "HessianDiagonalL1"]


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
    
class HessianKronecker(nn.Module):
    def __init__(
        self,
        n_in: int,
        n_hidden: Optional[Union[int, Sequence[int]]] = None,
        n_layers: int = 2,
        activation: Callable = F.silu,
        hessian_key: str = properties.hessian,
    ):
        super(HessianKronecker, self).__init__()
        self.n_in = n_in
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.hessian_key = hessian_key
        self.model_outputs = [hessian_key]
    
        self.n_out = 20 # has to be even
        self.outnet_l1 = spk.nn.build_gated_equivariant_mlp(
            n_in=n_in,
            n_out=self.n_out,
            n_hidden=n_hidden,
            n_layers=n_layers,
            activation=activation,
            sactivation=activation,
        )
        
        
    def forward(self, inputs):
        positions = inputs[properties.R] # 90 x 3
        l0 = inputs["scalar_representation"] # 90 x 30
        l1 = inputs["vector_representation"] # 90 x 3 x 30

        _, l1_out = self.outnet_l1((l0, l1)) # 90 x n_out, 90 x 3 x n_out
        
        n_atoms = inputs[properties.n_atoms]
        hessians: List[torch.Tensor] = []
        
        indices = torch.cumsum(n_atoms, dim=0) - n_atoms
        for start_idx, n_atom in zip(indices, n_atoms):
            end_idx = start_idx + n_atom

            l1_atom = l1_out[start_idx:end_idx] # n_atom x 3 x n_out
        
            kronecker_products = [
                torch.einsum('ik,jl->ijkl', l1_atom[:, :, i], l1_atom[:, :, i+1]).permute(0, 2, 1, 3).reshape(n_atom * 3, n_atom * 3)
                for i in range(0, self.n_out, 2)
            ] # (n_out * 3, n_out * 3) x n_out//2
            
            hessian = torch.sum(torch.stack(kronecker_products), dim=0) # n_out * 3 x n_out * 3
            
            # make symmetric
            hessian = (hessian + hessian.T) / 2
            
            # hessian = hessian + self.offset_matrix
            hessians.append(hessian)

        hessians = torch.stack(hessians, dim = 0).view(-1, 27) 

        inputs[self.hessian_key] = hessians
        return inputs
    
    def plot_kronecker_products(self, inputs):
        import matplotlib.pyplot as plt
        import numpy as np
        
        def plot_hessian(hessian, ax):
            import numpy as np
            import matplotlib.pyplot as plt
            from mpl_toolkits.axes_grid1 import make_axes_locatable

            # Convert hessian to numpy array if needed
            if type(hessian) is not np.ndarray:
                hessian = hessian.cpu().detach().numpy()

            # Plot the hessian
            im = ax.imshow(hessian, cmap="viridis")

            # Adding a fine grid every 3 cells
            ax.set_xticks(np.arange(-0.5, 27, 3), minor=True)
            ax.set_yticks(np.arange(-0.5, 27, 3), minor=True)
            ax.grid(which="minor", color="w", linestyle="-", linewidth=0.5)

            # Remove default ticks
            ax.set_xticks([])
            ax.set_yticks([])

            # Adding sublabels and main labels
            main_labels = [r"$C$", r"$C$", r"$O$", r"$H$", r"$H$", r"$H$", r"$H$", r"$H$", r"$H$"]
            for i, label in enumerate(main_labels):
                ax.text(i * 3 + 1, -1.5, label, ha="center", va="center", fontsize=10, color="black", fontweight="bold")
                ax.text(-1.5, i * 3 + 1, label, ha="center", va="center", fontsize=10, color="black", fontweight="bold")

            # Create a colorbar with controlled size
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)  # Adjust size and padding
            plt.colorbar(im, cax=cax, orientation="vertical")

        l0 = inputs["scalar_representation"] # 90 x 30
        l1 = inputs["vector_representation"] # 90 x 3 x 30

        l0, l1 = self.outnet_l1((l0, l1)) # 90 x n_out, 90 x 3 x n_out

        n_atoms = inputs[properties.n_atoms]
        hessians: List[torch.Tensor] = []

        indices = torch.cumsum(n_atoms, dim=0) - n_atoms
        for start_idx, n_atom in zip(indices, n_atoms):
            end_idx = start_idx + n_atom

            l1_atom = l1[start_idx:end_idx] # n_atom x 3 x n_out

            kronecker_products = [
                torch.einsum('ik,jl->ijkl', l1_atom[:, :, i], l1_atom[:, :, i+1]).permute(0, 2, 1, 3).reshape(n_atom * 3, n_atom * 3)
                for i in range(0, self.n_out, 2)
            ]
            
            fig, axs = plt.subplots(2, 5, figsize=(15, 15))
            for i, ax in enumerate(axs.flat):
                #im = ax.imshow(kronecker_products[i].detach().cpu().numpy())
                kronecker_product = kronecker_products[i].detach().cpu().numpy()
                plot_hessian(kronecker_product, ax)
            
            # Adjust spacing between subplots and margins
            plt.subplots_adjust(
                left=0.05,  # Space on the left
                right=0.95,  # Space on the right
                top=0.95,  # Space at the top
                bottom=0.6,  # Space at the bottom
                hspace=0.1,  # Vertical space between rows
                wspace=0.3,  # Horizontal space between columns
            )

            # plt.tight_layout()  # Ensures everything fits nicely
            plt.savefig(f"kronecker_products_{start_idx}.svg", bbox_inches="tight")  # Export without excessive whitespace
            plt.show()
                        
# D + UU^T
class HessianDUUT(nn.Module):
    def __init__(
        self,
        n_in: int,
        n_hidden: Optional[Union[int, Sequence[int]]] = None,
        n_layers: int = 2,
        activation: Callable = F.silu,
        hessian_key: str = properties.hessian,
    ):
        super(HessianDUUT, self).__init__()
        self.U_features = 6
        self.outnet_l1 = spk.nn.build_gated_equivariant_mlp(
            n_in=n_in,
            n_out=self.U_features,
            n_hidden=n_hidden,
            n_layers=n_layers,
            activation=activation,
            sactivation=activation,
        )
        
        self.outnet_l0 = spk.nn.build_gated_equivariant_mlp(
            n_in=n_in,
            n_out=3,
            n_hidden=n_hidden,
            n_layers=n_layers,
            activation=activation,
            sactivation=activation,
        ) 
        
        self.hessian_key = hessian_key
        self.model_outputs = [hessian_key]
        
    def forward(self, inputs):
        atomic_numbers = inputs[properties.Z]
        positions = inputs[properties.R] # 90 x 3
        l0 = inputs["scalar_representation"] # 90 x F
        l1 = inputs["vector_representation"] # 90 x 3 x F

        l0_outnet, _ = self.outnet_l0((l0, l1)) # 90 x 1, 90 x 3 x 1
        _, l1_outnet = self.outnet_l1((l0, l1)) # 90 x F, 90 x 3 x F
        
        n_atoms = inputs[properties.n_atoms]
        hessians: List[torch.Tensor] = []
        
        indices = torch.cumsum(n_atoms, dim=0) - n_atoms
        for start_idx, n_atom in zip(indices, n_atoms):
            end_idx = start_idx + n_atom
            
            l0_atom = l0_outnet[start_idx:end_idx].reshape(n_atom * 3) # n_atom * 3
            l1_atom = l1_outnet[start_idx:end_idx].reshape(n_atom * 3, self.U_features) # (n_atom * 3) x U_features
            
            # represent hessian as D + UU^T
            D = torch.diag_embed(l0_atom.flatten()) # (n_atom * 3) x (n_atom * 3)
            UUT = l1_atom @ l1_atom.T # (n_atom * 3) x (n_atom * 3)
            
            hessian = D + UUT
            
            hessians.append(hessian)
            
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
        l0 = inputs["scalar_representation"] # 90 x 30
        l1 = inputs["vector_representation"] # 90 x 3 x 30

        l0, l1 = self.outnet((l0, l1)) # 90 x 1, 90 x 3 x 1
        
        l1 = l1.squeeze(-1) # 90 x 3
        
        inputs[self.newton_step_key] = l1
        return inputs

class HessianDiagonalL0(nn.Module):
    def __init__(
        self,
        n_in: int,
        n_hidden: Optional[Union[int, Sequence[int]]] = None,
        n_layers: int = 2,
        activation: Callable = F.silu,
        diagonal_key: str = "diagonal",
    ):
        super(HessianDiagonalL0, self).__init__()
        self.diagonal_key = diagonal_key
        self.model_outputs = [diagonal_key]
    
        self.outnet = spk.nn.build_gated_equivariant_mlp(
            n_in=n_in,
            n_out=3,
            n_hidden=n_hidden,
            n_layers=n_layers,
            activation=activation,
            sactivation=activation,
        )
        
    def forward(self, inputs):
        l0 = inputs["scalar_representation"] # 90 x 30
        l1 = inputs["vector_representation"] # 90 x 3 x 30

        l0, l1 = self.outnet((l0, l1)) # 90 x 3, 90 x 3 x 3
        
        inputs[self.diagonal_key] = l0.flatten() # 10*27
        return inputs
    
class HessianDiagonalL1(nn.Module):
    def __init__(
        self,
        n_in: int,
        n_hidden: Optional[Union[int, Sequence[int]]] = None,
        n_layers: int = 2,
        activation: Callable = F.silu,
        diagonal_key: str = "diagonal",
    ):
        super(HessianDiagonalL1, self).__init__()
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
        
        self.dnn = nn.Sequential(
            nn.Linear(3, 10),
            nn.SiLU(),
            nn.Linear(10, 3)
        )
        
    def forward(self, inputs):
        # l0 = inputs["scalar_representation"] # 90 x 30
        # l1 = inputs["vector_representation"] # 90 x 3 x 30

        # l0, l1 = self.outnet((l0, l1)) # 90 x 1, 90 x 3 x 1
        
        # scalar = torch.sum(torch.abs(l0))
        # l1 = torch.abs(l1)
        # l1 = scalar * l1
        # #elementwise = l0.unsqueeze(-1) * l1 # 90 x 3 x 1
        
        # inputs[self.diagonal_key] = l1.squeeze(-1).flatten() # 10*27
        # return inputs

        positions = inputs[properties.R] # 90 x 3
        l0 = inputs["scalar_representation"] # 90 x 30
        l1 = inputs["vector_representation"] # 90 x 3 x 30

        l0, l1 = self.outnet((l0, l1)) # 90 x 1, 90 x 3 x 1
        
        l1 = l1.squeeze(-1) # 90 x 3
        
        idx_m = inputs[properties.idx_m]
        diagonals = []
        last = -1
        for i, idx in enumerate(idx_m):
            # concatted = torch.cat([l1[i], positions[i], l0[i]], dim=0)
            values = self.dnn(l1[i])
            if idx == last:
                diagonals[-1] = torch.cat([diagonals[-1], values], dim=0)
            else:
                diagonals.append(values)
                last = idx
        diagonals = torch.stack(diagonals, dim=0) # 10 x 27
        
        inputs[self.diagonal_key] = diagonals.flatten() # 10*27
        return inputs