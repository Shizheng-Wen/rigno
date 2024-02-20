"""A library of auxliary functions and classes."""

from typing import Sequence, Callable, Tuple

import jax.numpy as jnp
import jax.tree_util as tree
import flax.linen as nn

def concatenate_args(args, kwargs, axis: int = -1):
  combined_args = tree.tree_flatten(args)[0] + tree.tree_flatten(kwargs)[0]
  concat_args = jnp.concatenate(combined_args, axis=axis)
  return concat_args

class MLP(nn.Module):
  """
  Multi-layer perceptron with optional layer norm on the last layer.
  Activation is applied on all layers except the last one.
  Multiple inputs are concatenated before being fed to the MLP.
  """

  layer_sizes: Sequence[int]
  activation: Callable
  use_layer_norm: bool = False
  concatenate_axis: int = -1

  def setup(self):
    self.layers = [nn.Dense(features) for features in self.layer_sizes]
    self.layernorm = nn.LayerNorm(
      reduction_axes=-1,
      feature_axes=-1,
      use_scale=True,
      use_bias=True,
    ) if self.use_layer_norm else None

  def __call__(self, *args, **kwargs):
    x = concatenate_args(args=args, kwargs=kwargs, axis=self.concatenate_axis)
    for layer in self.layers[:-1]:
      x = layer(x)
      x = self.activation(x)
    x = self.layers[-1](x)
    if self.layernorm:
      x = self.layernorm(x)
    return x

def grid_mesh_connectivity_fixed_dx(
        x: jnp.ndarray, n_cover: int, n_overlap: int,
        dx: float, minx: float, maxx: float,
) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]]:
    """
    1D with periodic boundary conditions.

    The last point must be excluded in x.
    Shape of x has to be a power of 2.
    """

    zeta_grid = jnp.array((x - minx) / (maxx - minx) * 2 - 1)
    dzeta = (dx / (maxx - minx)) * 2
    zeta_mesh = (n_cover - 1) / 2 * dzeta + jnp.arange(
        start=-1., stop=(1. - (n_cover-n_overlap)*dzeta/2), step=(n_cover-n_overlap)*dzeta)

    indices_grid = jnp.array([
        jnp.mod(jnp.arange(0, n_cover) + (n_cover - n_overlap) * k, zeta_grid.shape[0])
        for k in range(zeta_mesh.shape[0])
    ]).ravel()
    indices_mesh = jnp.arange(0, zeta_mesh.shape[0]).repeat(n_cover)

    return (indices_grid, indices_mesh), (zeta_grid, zeta_mesh)
