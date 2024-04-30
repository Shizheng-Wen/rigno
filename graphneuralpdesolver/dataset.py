"""Utility functions for reading the datasets."""

from pathlib import Path
import h5py
import numpy as np
from typing import Any, Union, Sequence

import numpy as np
import jax
import jax.lax
import jax.numpy as jnp
import flax.typing

from graphneuralpdesolver.utils import Array
from graphneuralpdesolver.utils import normalize
from graphneuralpdesolver.models.utils import compute_derivatives


DATAGROUP = {
  'incompressible_fluids': 'velocity',
  'compressible_flow': 'data',
  'compressible_flow/gravity': 'solution',
  'reaction_diffusion': 'solution',
  'wave_equation': 'solution',
}

class Dataset:

  def __init__(self, key: flax.typing.PRNGKey,
      datadir: str, subpath: str, name: str,
      n_train: int, n_valid: int, n_test: int,
      idx_vars: Union[int, Sequence] = None,
      preload: bool = False,
      cutoff: int = None,
      downsample_factor: int = 1,
    ):

    # Set attributes
    self.datagroup = DATAGROUP[subpath]
    if name == 'richtmyer_meshkov': self.datagroup = 'solution'
    self.reader = h5py.File(Path(datadir) / subpath / f'{name}.nc', 'r')
    self.idx_vars = [idx_vars] if isinstance(idx_vars, int) else idx_vars
    self.preload = preload
    self.data = None
    if self.preload:
      self.length = n_train+n_valid+n_test
    else:
      self.length = self.reader[self.datagroup].shape[0]
    self.cutoff = cutoff if (cutoff is not None) else (self._fetch(0, raw=True)[0].shape[1])
    self.downsample_factor = downsample_factor
    self.sample = self._fetch(0)
    self.shape = self.sample[0].shape

    # Split the dataset
    assert (n_train+n_valid+n_test) <= self.length
    self.nums = {'train': n_train, 'valid': n_valid, 'test': n_test}
    random_permutation = jax.random.permutation(key, self.length)
    self.idx_modes = {
      'train': random_permutation[:n_train],
      'valid': random_permutation[n_train:(n_train+n_valid)],
      'test': random_permutation[(n_train+n_valid):(n_train+n_valid+n_test)],
    }

    # Instantiate the dataset stats
    self.stats = {
      'trj': {'mean': None, 'std': None},
      'der': {'mean': None, 'std': None},
      'res': {'mean': None, 'std': None},
      'time': {'max': self.shape[1]},
    }

    if self.preload:
      self.data = self.reader[self.datagroup][np.arange(self.length)]

  def compute_stats(self,
      axes: Sequence[int] = (0,),
      derivs_degree: int = 0,
      residual_steps: int = 0,
      skip_residual_steps: int = 1,
    ) -> None:

    # Check inputs
    assert residual_steps >= 0
    assert residual_steps < self.shape[1]

    # Get all trajectories
    trj, _ = self.train(np.arange(self.nums['train']))

    # Compute statistics of the solutions
    self.stats['trj']['mean'] = np.mean(trj, axis=axes, keepdims=True)
    self.stats['trj']['std'] = np.std(trj, axis=axes, keepdims=True)

    # Compute statistics of the derivatives
    if derivs_degree > 0:
      trj_nrm = normalize(trj, shift=self.stats['trj']['mean'], scale=self.stats['trj']['std'])
      trj_nrm_der = compute_derivatives(trj_nrm, degree=derivs_degree)
      self.stats['der']['mean'] = np.mean(trj_nrm_der, axis=axes, keepdims=True)
      self.stats['der']['std'] = np.std(trj_nrm_der, axis=axes, keepdims=True)

    # Compute statistics of the residuals
    # TRY: Compute statistics of residuals of normalized trajectories
    _get_res = lambda s, trj: trj[:, (s):] - trj[:, :-(s)]
    self.stats['res']['mean'] = []
    self.stats['res']['std'] = []
    for s in range(1, residual_steps+1):
      if (s % skip_residual_steps):
        self.stats['res']['mean'].append(np.zeros(shape=(1, *self.shape[1:])))
        self.stats['res']['std'].append(np.ones(shape=(1, *self.shape[1:])))
      res = _get_res(s, trj)
      res_mean = np.mean(res, axis=axes, keepdims=True)
      res_std = np.std(res, axis=axes, keepdims=True)
      # Fill the time axis so that all stats have the same shape
      if 1 not in axes:
        fill_shape = [1 if (ax in axes) else self.shape[ax] for ax in range(len(self.shape))]
        fill_shape[1] = s
        res_mean = np.concatenate([res_mean, np.zeros(shape=fill_shape)], axis=1)
        res_std = np.concatenate([res_std, np.ones(shape=fill_shape)], axis=1)
      self.stats['res']['mean'].append(res_mean)
      self.stats['res']['std'].append(res_std)

    # Repeat along the time axis
    if 1 in axes:
      reps = (1, self.shape[1], 1, 1, 1)
      self.stats['trj']['mean'] = np.tile(self.stats['trj']['mean'], reps=reps)
      self.stats['trj']['std'] = np.tile(self.stats['trj']['std'], reps=reps)
      if derivs_degree > 0:
        self.stats['der']['mean'] = np.tile(self.stats['der']['mean'], reps=reps)
        self.stats['der']['std'] = np.tile(self.stats['der']['std'], reps=reps)
      self.stats['res']['mean'] = [np.tile(stat, reps=reps) for stat in self.stats['res']['mean']]
      self.stats['res']['std'] = [np.tile(stat, reps=reps) for stat in self.stats['res']['std']]

  def _fetch(self, idx: Union[int, Sequence], raw: bool = False):
    """Fetches a sample from the dataset, given its global index."""

    # Check inputs
    if isinstance(idx, int):
      idx = [idx]

    # Get trajectories
    if self.data is not None:
      traj = self.data[np.sort(idx)]
    else:
      traj = self.reader[self.datagroup][np.sort(idx)]
    # Move axes
    if len(traj.shape) == 5:
      traj = np.moveaxis(traj, source=(2, 3, 4), destination=(4, 2, 3))
    elif len(traj.shape) == 4:
      traj = np.expand_dims(traj, axis=-1)
    # Set equation parameters
    spec = None

    # Select variables
    if self.idx_vars is not None:
      traj = traj[..., self.idx_vars]

    # Downsample and cut the trajectories
    if not raw:
      traj = traj[:, :(self.cutoff):self.downsample_factor]

    return traj, spec

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


# NOTE: 1D
NX_SUPER_RESOLUTION = 256
NT_SUPER_RESOLUTION = 256

# NOTE: 1D
def read_datasets(dir: Union[Path, str], pde_type: str, experiment: str, nx: int,
                  downsample_x: bool = True,
                  modes: Sequence[str] = ['train', 'valid', 'test'],
                ) -> dict[str, dict[str, Any]]:
  """Reads a dataset from its source file and prepares the shapes and specifications."""

  dir = Path(dir)
  datasets = {
    mode: _read_dataset_attributes(
      h5group=h5py.File(dir / f'{pde_type}_{mode}_{experiment}.h5')[mode],
      nx=(NX_SUPER_RESOLUTION if downsample_x else nx),
      nt=NT_SUPER_RESOLUTION,
    )
    for mode in modes
  }

  if downsample_x:
    ratio = NX_SUPER_RESOLUTION // nx
    for dataset in datasets.values():
      dataset['trajectories'] = downsample(dataset['trajectories'], ratio=ratio, axis=2)
      dataset['x'] = downsample(dataset['x'], ratio=ratio, axis=0)
      dataset['dx'] = dataset['dx'] * ratio

  return datasets

# NOTE: 1D
def _read_dataset_attributes(h5group: h5py.Group, nx: int, nt: int) -> dict[str, Any]:
  """Prepares the shapes and puts together the specifications of a dataset."""

  resolution = f'pde_{nt}-{nx}'
  trajectories = h5group[resolution][:]
  if trajectories.ndim == 3:
    trajectories = trajectories[:, None]
  dataset = dict(
    trajectories = np.moveaxis(trajectories, (1, 2, 3), (3, 1, 2)),
    x = h5group[resolution].attrs['x'],
    # dx = h5group[resolution].attrs['dx'].item(),
    tmin = h5group[resolution].attrs['tmin'].item(),
    tmax = h5group[resolution].attrs['tmax'].item(),
    dt = h5group[resolution].attrs['dt'].item(),
    # nx = h5group[resolution].attrs['nx'].item(),
    # nt = h5group[resolution].attrs['nt'].item(),
  )
  dataset['dx'] = (dataset['x'][1] - dataset['x'][0]).item()  # CHECK: Why is it different from data['dx']?
  dataset['range_x'] = (np.min(dataset['x']), np.max(dataset['x']))

  return dataset

# NOTE: 1D
def downsample_convolution(trajectories: Array, ratio: int) -> Array:
  trj_padded = np.concatenate([trajectories[:, :, -(ratio//2+1):-1], trajectories, trajectories[:, :, :(ratio//2)]], axis=2)
  kernel = np.array([1/(ratio+1)]*(ratio+1)).reshape(1, 1, ratio+1, 1)
  trj_downsampled = jax.lax.conv_general_dilated(
    lhs=trj_padded,
    rhs=kernel,
    window_strides=(1,ratio),
    padding='VALID',
    dimension_numbers=jax.lax.conv_dimension_numbers(
      trajectories.shape, kernel.shape, ('NHWC', 'OHWI', 'NHWC')),
  )

  return trj_downsampled

# NOTE: 1D
def downsample(arr: Array, ratio: int, axis: int = 0) -> Array:
  slc = [slice(None)] * len(arr.shape)
  slc[axis] = slice(None, None, ratio)
  return arr[tuple(slc)]
