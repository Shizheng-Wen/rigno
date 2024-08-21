"""Utility functions for reading the datasets."""

import h5py
from dataclasses import dataclass
from pathlib import Path
from typing import Union, Sequence, NamedTuple

import flax.typing
import jax
import jax.lax
import jax.numpy as jnp
import numpy as np

from rigno.utils import Array, shuffle_arrays


@dataclass
class Metadata:
  periodic: bool
  data_group: str
  coeff_group: str
  domain_t: tuple[int, int]
  domain_x: tuple[Sequence[int], Sequence[int]]
  active_variables: Sequence[int]
  target_variables: Sequence[int]
  signed: dict[str, Union[bool, Sequence[bool]]]
  names: dict[str, Sequence[str]]

  @property
  def stats_target_variables(self) -> dict[str, np.array]:
    _stats = {
      'mean': np.array(self.stats['mean']).reshape(1, 1, 1, 1, -1)[..., self.target_variables],
      'std': np.array(self.stats['std']).reshape(1, 1, 1, 1, -1)[..., self.target_variables],
    }
    return _stats

ACTIVE_VARS_NS = [0, 1]
ACTIVE_VARS_CE = [0, 1, 2, 3]
ACTIVE_VARS_GCE = [0, 1, 2, 3, 5]
ACTIVE_VARS_RD = [0]
ACTIVE_VARS_WE = [0]
ACTIVE_VARS_PE = [0]

TARGET_VARS_NS = [0, 1]
TARGET_VARS_CE = [1, 2]
TARGET_VARS_GCE = [1, 2]
TARGET_VARS_RD = [0]
TARGET_VARS_WE = [0]
TARGET_VARS_PE = [0]

SIGNED_NS = {'u': [True, True], 'c': None}
SIGNED_CE = {'u': [False, True, True, False, False], 'c': None}
SIGNED_GCE = {'u': [False, True, True, False, False, False], 'c': None}
SIGNED_RD = {'u': [True], 'c': None}
SIGNED_WE = {'u': [True], 'c': [False]}
SIGNED_PE = {'u': [True], 'c': [True]}

NAMES_NS = {'u': ['$v_x$', '$v_y$'], 'c': None}
NAMES_CE = {'u': ['$\\rho$', '$v_x$', '$v_y$', '$p$'], 'c': None}
NAMES_GCE = {'u': ['$\\rho$', '$v_x$', '$v_y$', '$p$', 'E', '$\\phi$'], 'c': None}
NAMES_RD = {'u': ['$u$'], 'c': None}
NAMES_WE = {'u': ['$u$'], 'c': ['$c$']}
NAMES_PE = {'u': ['$u$'], 'c': ['$f$']}

DATASET_METADATA = {
  # incompressible_fluids: [velocity, velocity]
    'velocity_0': Metadata(
    periodic=True,
    data_group='velocity',
    coeff_group=None,
    domain_t=(0, 1),
    domain_x=([0, 0], [1, 1]),
    active_variables=ACTIVE_VARS_NS,
    target_variables=TARGET_VARS_NS,
    signed=SIGNED_NS,
    names=NAMES_NS,
  ),
  'incompressible_fluids/brownian_bridge': Metadata(
    periodic=True,
    data_group='velocity',
    coeff_group=None,
    domain_t=(0, 1),
    domain_x=([0, 0], [1, 1]),
    active_variables=ACTIVE_VARS_NS,
    target_variables=TARGET_VARS_NS,
    signed=SIGNED_NS,
    names=NAMES_NS,
  ),
  'incompressible_fluids/gaussians': Metadata(
    periodic=True,
    data_group='velocity',
    coeff_group=None,
    domain_t=(0, 1),
    domain_x=([0, 0], [1, 1]),
    active_variables=ACTIVE_VARS_NS,
    target_variables=TARGET_VARS_NS,
    signed=SIGNED_NS,
    names=NAMES_NS,
  ),
  'incompressible_fluids/pwc': Metadata(
    periodic=True,
    data_group='velocity',
    coeff_group=None,
    domain_t=(0, 1),
    domain_x=([0, 0], [1, 1]),
    active_variables=ACTIVE_VARS_NS,
    target_variables=TARGET_VARS_NS,
    signed=SIGNED_NS,
    names=NAMES_NS,
  ),
  'incompressible_fluids/shear_layer': Metadata(
    periodic=True,
    data_group='velocity',
    coeff_group=None,
    domain_t=(0, 1),
    domain_x=([0, 0], [1, 1]),
    active_variables=ACTIVE_VARS_NS,
    target_variables=TARGET_VARS_NS,
    signed=SIGNED_NS,
    names=NAMES_NS,
  ),
  'incompressible_fluids/sines': Metadata(
    periodic=True,
    data_group='velocity',
    coeff_group=None,
    domain_t=(0, 1),
    domain_x=([0, 0], [1, 1]),
    active_variables=ACTIVE_VARS_NS,
    target_variables=TARGET_VARS_NS,
    signed=SIGNED_NS,
    names=NAMES_NS,
  ),
  'incompressible_fluids/vortex_sheet': Metadata(
    periodic=True,
    data_group='velocity',
    coeff_group=None,
    domain_t=(0, 1),
    domain_x=([0, 0], [1, 1]),
    active_variables=ACTIVE_VARS_NS,
    target_variables=TARGET_VARS_NS,
    signed=SIGNED_NS,
    names=NAMES_NS,
  ),
  # compressible_flow: [density, velocity, velocity, pressure, energy]
  'compressible_flow/cloudshock': Metadata(
    periodic=True,
    data_group='data',
    coeff_group=None,
    domain_t=(0, 1),
    domain_x=([0, 0], [1, 1]),
    active_variables=ACTIVE_VARS_CE,
    target_variables=TARGET_VARS_CE,
    signed=SIGNED_CE,
    names=NAMES_CE,
  ),
  'compressible_flow/gauss': Metadata(
    periodic=True,
    data_group='data',
    coeff_group=None,
    domain_t=(0, 1),
    domain_x=([0, 0], [1, 1]),
    active_variables=ACTIVE_VARS_CE,
    target_variables=TARGET_VARS_CE,
    signed=SIGNED_CE,
    names=NAMES_CE,
  ),
  'compressible_flow/kh': Metadata(
    periodic=True,
    data_group='data',
    coeff_group=None,
    domain_t=(0, 1),
    domain_x=([0, 0], [1, 1]),
    active_variables=ACTIVE_VARS_CE,
    target_variables=TARGET_VARS_CE,
    signed=SIGNED_CE,
    names=NAMES_CE,
  ),
  'compressible_flow/richtmyer_meshkov': Metadata(
    periodic=True,
    data_group='solution',
    coeff_group=None,
    domain_t=(0, 2),
    domain_x=([0, 0], [1, 1]),
    active_variables=ACTIVE_VARS_CE,
    target_variables=TARGET_VARS_CE,
    signed=SIGNED_CE,
    names=NAMES_CE,
  ),
  'compressible_flow/riemann': Metadata(
    periodic=True,
    data_group='data',
    coeff_group=None,
    domain_t=(0, 1),
    domain_x=([0, 0], [1, 1]),
    active_variables=ACTIVE_VARS_CE,
    target_variables=TARGET_VARS_CE,
    signed=SIGNED_CE,
    names=NAMES_CE,
  ),
  'compressible_flow/riemann_curved': Metadata(
    periodic=True,
    data_group='data',
    coeff_group=None,
    domain_t=(0, 1),
    domain_x=([0, 0], [1, 1]),
    active_variables=ACTIVE_VARS_CE,
    target_variables=TARGET_VARS_CE,
    signed=SIGNED_CE,
    names=NAMES_CE,
  ),
  'compressible_flow/riemann_kh': Metadata(
    periodic=True,
    data_group='data',
    coeff_group=None,
    domain_t=(0, 1),
    domain_x=([0, 0], [1, 1]),
    active_variables=ACTIVE_VARS_CE,
    target_variables=TARGET_VARS_CE,
    signed=SIGNED_CE,
    names=NAMES_CE,
  ),
  'compressible_flow/gravity/blast': Metadata(
    periodic=True,
    data_group='solution',
    coeff_group=None,
    domain_t=(0, 1),
    domain_x=([0, 0], [1, 1]),
    # CHECK: Where is the gravitational field?
    active_variables=ACTIVE_VARS_CE,
    target_variables=TARGET_VARS_CE,
    signed=SIGNED_CE,
    names=NAMES_CE,
  ),
  'compressible_flow/gravity/rayleigh_taylor': Metadata(
    periodic=True,
    data_group='solution',
    coeff_group=None,
    domain_t=(0, 5),
    domain_x=([0, 0], [1, 1]),
    active_variables=ACTIVE_VARS_GCE,
    target_variables=TARGET_VARS_GCE,
    signed=SIGNED_GCE,
    names=NAMES_GCE,
  ),
  # reaction_diffusion
  'reaction_diffusion/allen_cahn': Metadata(
    periodic=False,
    data_group='solution',
    coeff_group=None,
    domain_t=(0, 0.0002),
    domain_x=([0, 0], [1, 1]),
    active_variables=ACTIVE_VARS_RD,
    target_variables=TARGET_VARS_RD,
    signed=SIGNED_RD,
    names=NAMES_RD,
  ),
  # wave_equation
  'wave_equation/seismic_20step': Metadata(
    periodic=False,
    data_group='solution',
    coeff_group='c',
    domain_t=(0, 1),
    domain_x=([0, 0], [1, 1]),
    active_variables=ACTIVE_VARS_WE,
    target_variables=TARGET_VARS_WE,
    signed=SIGNED_WE,
    names=NAMES_WE,
  ),
  'wave_equation/gaussians_15step': Metadata(
    periodic=False,
    data_group='solution',
    coeff_group='c',
    domain_t=(0, 1),
    domain_x=([0, 0], [1, 1]),
    active_variables=ACTIVE_VARS_WE,
    target_variables=TARGET_VARS_WE,
    signed=SIGNED_WE,
    names=NAMES_WE,
  ),
  # poisson_equation
  'poisson_equation/sines': Metadata(
    periodic=False,
    data_group='solution',
    coeff_group='source',
    domain_t=None,
    domain_x=([0, 0], [1, 1]),
    active_variables=ACTIVE_VARS_PE,
    target_variables=TARGET_VARS_PE,
    signed=SIGNED_PE,
    names=NAMES_PE,
  ),
  'poisson_equation/chebyshev': Metadata(
    periodic=False,
    data_group='solution',
    coeff_group='source',
    domain_t=None,
    domain_x=([0, 0], [1, 1]),
    active_variables=ACTIVE_VARS_PE,
    target_variables=TARGET_VARS_PE,
    signed=SIGNED_PE,
    names=NAMES_PE,
  ),
  'poisson_equation/pwc': Metadata(
    periodic=False,
    data_group='solution',
    coeff_group='source',
    domain_t=None,
    domain_x=([0, 0], [1, 1]),
    active_variables=ACTIVE_VARS_PE,
    target_variables=TARGET_VARS_PE,
    signed=SIGNED_PE,
    names=NAMES_PE,
  ),
  'poisson_equation/gaussians': Metadata(
    periodic=False,
    data_group='solution',
    coeff_group='source',
    domain_t=None,
    domain_x=([0, 0], [1, 1]),
    active_variables=ACTIVE_VARS_PE,
    target_variables=TARGET_VARS_PE,
    signed=SIGNED_PE,
    names=NAMES_PE,
  ),
}

class Batch(NamedTuple):
  u: Array
  c: Union[Array, None]
  t: Union[Array, None]
  x: Array

  @property
  def shape(self) -> tuple:
    return self.u.shape

  def unravel(self) -> tuple:
    return (self.u, self.c, self.x, self.t)

  @property
  def _x(self) -> Array:
    return self.x[0, 0]

  def __len__(self) -> int:
    return self.shape[0]

class Dataset:

  def __init__(self,
    datadir: str,
    datapath: str,
    include_passive_variables: bool = False,
    concatenate_coeffs: bool = False,
    time_cutoff_idx: int = None,
    time_downsample_factor: int = 1,
    space_downsample_factor: int = 1,
    unstructured: bool = False,
    n_train: int = 0,
    n_valid: int = 0,
    n_test: int = 0,
    preload: bool = False,
    key: flax.typing.PRNGKey = None,
  ):

    # Set attributes
    self.key = key if (key is not None) else jax.random.PRNGKey(0)
    self.metadata = DATASET_METADATA[datapath]
    self.preload = preload
    self.concatenate_coeffs = concatenate_coeffs
    self.time_cutoff_idx = time_cutoff_idx
    self.time_downsample_factor = time_downsample_factor
    self.unstructured = unstructured
    self.space_downsample_factor = space_downsample_factor

    # Modify metadata
    if not include_passive_variables:
      self.metadata.names['u'] = [self.metadata.names['u'][v] for v in self.metadata.active_variables]
      self.metadata.signed['u'] = [self.metadata.signed['u'][v] for v in self.metadata.active_variables]
    if self.concatenate_coeffs and self.metadata.coeff_group:
      self.metadata.names['u'] += self.metadata.names['c']
      self.metadata.signed['u'] += self.metadata.signed['c']

    # Set data attributes
    self.data_group = self.metadata.data_group
    self.coeff_group = self.metadata.coeff_group
    self.reader = h5py.File(Path(datadir) / f'{datapath}.nc', 'r')
    self.idx_vars = (None if include_passive_variables
      else self.metadata.active_variables)
    self.data = None
    self.coeff = None
    self.length = ((n_train + n_valid + n_test) if self.preload
      else self.reader[self.data_group].shape[0])
    self.sample = self._fetch(0)
    self.shape = self.sample.shape
    if self.time_dependent:
      self.dt = (self.sample.t[0, 1] - self.sample.t[0, 0]).item() # NOTE: Assuming fix dt

    # Check sample dimensions
    for arr in self.sample.unravel():
      if arr is not None:
        assert arr.ndim == 4

    # Split the dataset
    assert (n_train + n_valid + n_test) <= self.length
    self.nums = {'train': n_train, 'valid': n_valid, 'test': n_test}
    self.idx_modes = {
      # First n_train samples
      'train': jax.random.permutation(self.key, n_train),
      # First n_valid samples after the training samples
      'valid': np.arange(n_train, (n_train + n_valid)),
      # Last n_test samples
      'test': np.arange((self.length - n_test), self.length),
    }

    # Instantiate the dataset stats
    self.stats = {
      'u': {'mean': None, 'std': None},
      'c': {'mean': None, 'std': None},
      'x': {
        'min': np.array(self.metadata.domain_x[0]).reshape(1, 1, 1, -1),
        'max': np.array(self.metadata.domain_x[1]).reshape(1, 1, 1, -1),
      },
      't': {
        'min': np.array(self.metadata.domain_t[0]).reshape(1, 1, 1, 1)
          if self.time_dependent else None,
        'max': np.array(self.metadata.domain_t[1]).reshape(1, 1, 1, 1)
          if self.time_dependent else None,
      },
      'res': {'mean': None, 'std': None},
      'der': {'mean': None, 'std': None},
    }

    # Load the data
    if self.preload:
      _len_dataset = self.reader[self.data_group].shape[0]
      train_data = self.reader[self.data_group][np.arange(n_train)]
      valid_data = self.reader[self.data_group][np.arange(n_train, (n_train + n_valid))]
      test_data = self.reader[self.data_group][np.arange((_len_dataset - n_test), (_len_dataset))]
      self.data = np.concatenate([train_data, valid_data, test_data], axis=0)
      if self.coeff_group is not None:
        train_coeff = self.reader[self.coeff_group][np.arange(n_train)]
        valid_coeff = self.reader[self.coeff_group][np.arange(n_train, (n_train + n_valid))]
        test_coeff = self.reader[self.coeff_group][np.arange((_len_dataset - n_test), (_len_dataset))]
        self.coeff = np.concatenate([train_coeff, valid_coeff, test_coeff], axis=0)

  @property
  def time_dependent(self):
    return self.metadata.domain_t is not None

  def compute_stats(self, residual_steps: int = 0) -> None:

    # Check inputs
    assert residual_steps >= 0
    assert residual_steps < self.shape[1]

    # Get all trajectories
    batch = self.train(np.arange(self.nums['train']))
    u, c, _, t = batch.unravel()

    # Compute statistics of solutions and coefficients
    self.stats['u']['mean'] = np.mean(u, axis=(0, 1, 2), keepdims=True)
    self.stats['u']['std'] = np.std(u, axis=(0, 1, 2), keepdims=True)
    if c is not None:
      self.stats['c']['mean'] = np.mean(c, axis=(0, 1, 2), keepdims=True)
      self.stats['c']['std'] = np.std(c, axis=(0, 1, 2), keepdims=True)

    # Compute statistics of the residuals and time derivatives
    if self.time_dependent:
      _get_res = lambda s, trj: (trj[:, (s):] - trj[:, :-(s)])
      residuals = []
      derivatives = []
      for s in range(1, residual_steps+1):
        res = _get_res(s, u)
        tau = _get_res(s, t)
        residuals.append(res)
        derivatives.append(res / tau)
      residuals = np.concatenate(residuals, axis=1)
      derivatives = np.concatenate(derivatives, axis=1)
      self.stats['res']['mean'] = np.mean(residuals, axis=(0, 1, 2), keepdims=True)
      self.stats['res']['std'] = np.std(residuals, axis=(0, 1, 2), keepdims=True)
      self.stats['der']['mean'] = np.mean(derivatives, axis=(0, 1, 2), keepdims=True)
      self.stats['der']['std'] = np.std(derivatives, axis=(0, 1, 2), keepdims=True)

  def _fetch(self, idx: Union[int, Sequence]) -> Batch:
    """Fetches a sample from the dataset, given its global index."""

    # Check inputs
    if isinstance(idx, int):
      idx = [idx]

    # Get trajectories
    if self.data is not None:
      u = self.data[np.sort(idx)]
    else:
      u = self.reader[self.data_group][np.sort(idx)]

    # Move axes
    if len(u.shape) == 5:  # NOTE: Multi-variable datasets
      u = np.moveaxis(u, source=(2, 3, 4), destination=(4, 2, 3))
    elif len(u.shape) == 4:  # NOTE: Single-variable datasets
      u = np.expand_dims(u, axis=-1)
    elif len(u.shape) == 3:  # NOTE: Single-variable time-independent datasets
      u = np.expand_dims(u, axis=(1, -1))

    # Select variables
    if self.idx_vars is not None:
      u = u[..., self.idx_vars]

    # Read coefficients
    if self.coeff_group is not None:
      # Get the coefficients
      if self.coeff is not None:
        c = self.coeff[np.sort(idx)]
      else:
        c = self.reader[self.coeff_group][np.sort(idx)]
      c = np.expand_dims(c, axis=(1, 4))
      c = np.tile(c, reps=(1, u.shape[1], 1, 1, 1))
    else:
      c = None

    # Downsample the spatial grid
    if not self.unstructured:
      u = u[:, :, ::self.space_downsample_factor, ::self.space_downsample_factor]
      if c is not None:
        c = c[:, :, ::self.space_downsample_factor, ::self.space_downsample_factor]

    # Define spatial coordinates
    _xv = np.linspace(self.metadata.domain_x[0][0], self.metadata.domain_x[1][0], u.shape[2], endpoint=(not self.metadata.periodic))
    _yv = np.linspace(self.metadata.domain_x[0][1], self.metadata.domain_x[1][1], u.shape[3], endpoint=(not self.metadata.periodic))
    _x, _y = np.meshgrid(_xv, _yv)
    # Align the dimensions
    _x = _x.reshape(1, 1, -1, 1)
    _y = _y.reshape(1, 1, -1, 1)
    # Concatenate the coordinates
    x = np.concatenate([_x, _y], axis=3)
    # Repeat along sample and time axes
    x = np.tile(x, reps=(u.shape[0], u.shape[1], 1, 1))
    # Flatten the trajectory
    u = u.reshape(u.shape[0], u.shape[1], (u.shape[2] * u.shape[3]), -1)
    if c is not None:
      c = c.reshape(u.shape[0], u.shape[1], (u.shape[2] * u.shape[3]), -1)

    # Define temporal coordinates
    if self.metadata.domain_t is not None:
      t = np.linspace(*self.metadata.domain_t, u.shape[1], endpoint=True)
      t = t.reshape(1, -1, 1, 1)
      # Repeat along sample trajectory
      t = np.tile(t, reps=(u.shape[0], 1, 1, 1))
    else:
      t = None

    # Cut the time axis
    if self.time_dependent and self.time_cutoff_idx:
      u = u[:, :self.time_cutoff_idx]
      if c is not None: c = c[:, :self.time_cutoff_idx]
      t = t[:, :self.time_cutoff_idx]
      x = x[:, :self.time_cutoff_idx]

    # Downsample in the time axis
    if self.time_dependent and self.time_downsample_factor > 1:
      u = u[:, ::self.time_downsample_factor]
      if c is not None: c = c[:, ::self.time_downsample_factor]
      if t is not None: t = t[:, ::self.time_downsample_factor]
      x = x[:, ::self.time_downsample_factor]

    # Downsample the space coordinates randomly
    if self.unstructured:
      permutation = jax.random.permutation(self.key, u.shape[2])
      # NOTE: Same discretization for all snapshots
      u = u[:, :, permutation]
      c = c[:, :, permutation] if (c is not None) else None
      x = x[:, :, permutation]

      size = int(u.shape[2] / (self.space_downsample_factor ** 2))
      u = u[:, :, :size]
      c = c[:, :, :size] if (c is not None) else None
      x = x[:, :, :size]

    if self.concatenate_coeffs and (c is not None):
      u = np.concatenate([u, c], axis=-1)
      c = None

    batch = Batch(u=u, c=c, t=t, x=x)

    return batch

  def _fetch_mode(self, idx: Union[int, Sequence], mode: str):
    # Check inputs
    if isinstance(idx, int):
      idx = [idx]
    # Set mode index
    assert all([i < len(self.idx_modes[mode]) for i in idx])
    _idx = self.idx_modes[mode][np.array(idx)]

    return self._fetch(_idx)

  def train(self, idx: Union[int, Sequence]):
    return self._fetch_mode(idx, mode='train')

  def valid(self, idx: Union[int, Sequence]):
    return self._fetch_mode(idx, mode='valid')

  def test(self, idx: Union[int, Sequence]):
    return self._fetch_mode(idx, mode='test')

  def batches(self, mode: str, batch_size: int, key: flax.typing.PRNGKey = None):
    assert batch_size > 0
    assert batch_size <= self.nums[mode]

    if key is not None:
      _idx_mode_permuted = jax.random.permutation(key, np.arange(self.nums[mode]))
    else:
      _idx_mode_permuted = jnp.arange(self.nums[mode])

    len_dividable = self.nums[mode] - (self.nums[mode] % batch_size)
    for idx in np.split(_idx_mode_permuted[:len_dividable], len_dividable // batch_size):
      batch = self._fetch_mode(idx, mode)
      yield batch

    if (self.nums[mode] % batch_size):
      idx = _idx_mode_permuted[len_dividable:]
      batch = self._fetch_mode(idx, mode)
      yield batch

  def __len__(self):
    return self.length
