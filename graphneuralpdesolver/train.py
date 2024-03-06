from datetime import datetime
import functools
from time import time
from typing import Tuple, Any, Mapping, Sequence, Union
import json
from dataclasses import dataclass

from absl import app, flags, logging
import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax.linen as nn
import flax.typing
from flax.training import orbax_utils
from flax.training.train_state import TrainState
import orbax.checkpoint

from graphneuralpdesolver.experiments import DIR_EXPERIMENTS
from graphneuralpdesolver.autoregressive import AutoregressivePredictor
from graphneuralpdesolver.dataset import read_datasets, shuffle_arrays, normalize, unnormalize
from graphneuralpdesolver.models.graphneuralpdesolver import GraphNeuralPDESolver, AbstractOperator, DummyOperator
from graphneuralpdesolver.utils import disable_logging, Array
from graphneuralpdesolver.metrics import mse, rel_l2_error, rel_l1_error


SEED = 43

FLAGS = flags.FLAGS
flags.DEFINE_string(name='datadir', default=None, required=True,
  help='Path of the folder containing the datasets'
)
flags.DEFINE_string(name='params', default=None, required=False,
  help='Path of the previous experiment containing the initial parameters'
)
flags.DEFINE_integer(name='resolution', default=128, required=False,
  help='Resolution of the physical discretization'
)
flags.DEFINE_string(name='experiment', default=None, required=True,
  help='Name of the experiment: {"E1", "E2", "E3", "WE1", "WE2", "WE3"'
)
flags.DEFINE_integer(name='batch_size', default=4, required=False,
  help='Size of a batch of training samples'
)
flags.DEFINE_integer(name='epochs', default=20, required=False,
  help='Number of training epochs'
)
flags.DEFINE_float(name='lr', default=1e-04, required=False,
  help='Training learning rate'
)
flags.DEFINE_float(name='lr_decay', default=None, required=False,
  help='The minimum learning rate decay in the cosine scheduler'
)
flags.DEFINE_integer(name='latent_size', default=128, required=False,
  help='Size of latent node and edge features'
)
flags.DEFINE_integer(name='unroll_steps', default=1, required=False,
  help='Number of steps for getting a noisy input and applying the model autoregressively'
)
flags.DEFINE_integer(name='direct_steps', default=1, required=False,
  help='Maximum number of time steps between input/output pairs during training'
)
flags.DEFINE_bool(name='verbose', default=False, required=False,
  help='If passed, training reports for batches are printed'
)
flags.DEFINE_bool(name='debug', default=False, required=False,
  help='If passed, the code is launched only for debugging purposes.'
)

PDETYPE = {
  'E1': 'CE',
  'E2': 'CE',
  'E3': 'CE',
  'WE1': 'WE',
  'WE2': 'WE',
  'WE3': 'WE',
}

@dataclass
class EvalMetrics:
  error_l1: Sequence[Tuple[int, Sequence[float]]] = None
  error_l2: Sequence[Tuple[int, Sequence[float]]] = None

DIR = DIR_EXPERIMENTS / datetime.now().strftime('%Y%m%d-%H%M%S.%f')

def train(model: nn.Module, dataset_trn: Mapping[str, Array], dataset_val: dict[str, Array],
          epochs: int, key: flax.typing.PRNGKey, params: flax.typing.Collection = None) -> TrainState:
  """Trains a model and returns the state."""

  # Set constants
  num_samples_trn = dataset_trn['trajectories'].shape[0]
  num_times = dataset_trn['trajectories'].shape[1]
  num_grid_points = dataset_trn['trajectories'].shape[2]
  batch_size = FLAGS.batch_size
  unroll_offset = FLAGS.unroll_steps * FLAGS.direct_steps
  assert num_samples_trn % batch_size == 0

  # Store the initial time
  time_int_pre = time()

  # Normalize the train dataset
  dataset_trn['trajectories_nrm'], stats_trn = normalize(dataset_trn['trajectories'])

  # Initialzize the model or use the loaded parameters
  if params:
    variables = {'params': params}
  else:
    subkey, key = jax.random.split(key)
    sample_input_u = dataset_trn['trajectories_nrm'][:batch_size, :1]
    sample_input_specs = dataset_trn['specs'][:batch_size]
    sample_dt = jnp.array([1.])  # Single float dt for a batch
    variables = jax.jit(model.init)(subkey, specs=sample_input_specs, u_inp=sample_input_u, dt=sample_dt)

  # Calculate the total number of parameters
  n_model_parameters = np.sum(
  jax.tree_util.tree_flatten(
    jax.tree_map(
      lambda x: np.prod(x.shape).item(),
      variables['params']
    ))[0]
  ).item()
  logging.info(f'Total number of trainable paramters: {n_model_parameters}')

  # Define the permissible lead times and number of batches
  lead_times = jnp.arange(unroll_offset, num_times - FLAGS.direct_steps)
  num_batches = num_samples_trn // batch_size
  num_lead_times = num_times - unroll_offset - FLAGS.direct_steps

  # Set up the optimization components
  criterion_loss = mse
  lr = optax.cosine_decay_schedule(
    init_value=FLAGS.lr,
    decay_steps=(FLAGS.epochs * num_batches),
    alpha=FLAGS.lr_decay,
  ) if FLAGS.lr_decay else FLAGS.lr
  tx = optax.inject_hyperparams(optax.adamw)(learning_rate=lr, weight_decay=1e-8)
  state = TrainState.create(apply_fn=model.apply, params=variables['params'], tx=tx)

  # Define the autoregressive predictor
  predictor_full = AutoregressivePredictor(operator=model, full_rollout=True)
  predictor_skip = AutoregressivePredictor(operator=model, full_rollout=False)

  def compute_loss(params: flax.typing.Collection, specs: Array,
                   u_inp_lagged: Array, dt: Array, u_out: Array, num_steps_autoreg: int) -> Array:
    """Computes the prediction of the model and returns its loss."""

    variables = {'params': params}
    # Apply autoregressive steps
    _, u_inp = predictor_skip(
      variables=variables,
      specs=specs,
      u_inp=u_inp_lagged,
      num_steps=(num_steps_autoreg * FLAGS.direct_steps),
      num_steps_direct=FLAGS.direct_steps,
    )
    # Get rollouts using the input from above
    rollout, u_next = predictor_full(
      variables=variables,
      specs=specs,
      u_inp=u_inp,
      num_steps=FLAGS.direct_steps,
      num_steps_direct=FLAGS.direct_steps,
    )
    # Get the corresponding output prediction based on dt
    rollout_extended = jnp.concatenate([rollout, u_next], axis=1)
    u_out_pred = rollout_extended[np.arange(u_out.shape[0]), dt][:, None, :, :]

    return criterion_loss(u_out_pred, u_out)

  def get_noisy_input(params: flax.typing.Collection, specs: Array,
                      u_inp_lagged: Array, num_steps_autoreg: int) -> Array:
    """Apply the model to the lagged input to get a noisy input."""

    variables = {'params': params}
    _, u_inp_noisy = predictor_skip(
      variables=variables,
      specs=specs,
      u_inp=u_inp_lagged,
      num_steps=(num_steps_autoreg * FLAGS.direct_steps),
      num_steps_direct=FLAGS.direct_steps,
    )

    return u_inp_noisy

  def get_loss_and_grads(params: flax.typing.Collection, specs: Array,
                         u_lag: Array, u_out: Array, dt: Array) -> Tuple[Array, Any]:
    """
    Computes the loss and the gradients of the loss w.r.t the parameters.
    """

    # Split the unrolling steps randomly to cut the gradients along the way
    # MODIFY: Change to JAX-generated random number (reproducability)
    noise_steps = np.random.choice(FLAGS.unroll_steps+1)
    grads_steps = FLAGS.unroll_steps - noise_steps

    # Get noisy input
    u_inp = get_noisy_input(
      params, specs, u_lag, num_steps_autoreg=noise_steps)
    # Use noisy input and compute gradients
    loss, grads = jax.value_and_grad(compute_loss)(
      params, specs, u_inp, dt, u_out, num_steps_autoreg=grads_steps)

    return loss, grads

  @jax.jit
  def train_one_batch(
    state: TrainState, batch: Tuple[Array, Array],
    key: flax.typing.PRNGKey = None) -> Tuple[TrainState, Array]:
    """TODO: WRITE."""

    trajectory, specs = batch

    # Get input output pairs for all lead times
    u_lag_batch = jax.vmap(
        lambda lt: jax.lax.dynamic_slice_in_dim(
          operand=trajectory,
          start_index=(lt-unroll_offset), slice_size=1, axis=1)
      )(lead_times)
    u_out_batch = jax.vmap(
        lambda lt: jax.lax.dynamic_slice_in_dim(
          operand=trajectory,
          start_index=(lt+1), slice_size=FLAGS.direct_steps, axis=1)
      )(lead_times)
    specs_batch = (specs[None, :, :]
      .repeat(repeats=num_lead_times, axis=0)
    )
    dt_batch = ((1 + jnp.arange(FLAGS.direct_steps))[None, None, :]
      .repeat(repeats=num_lead_times, axis=0)
      .repeat(repeats=batch_size, axis=1)
    )

    # Concatenate lead times along the batch axis
    u_lag_batch = u_lag_batch.reshape(
        (batch_size * num_lead_times), 1, num_grid_points, -1)
    u_out_batch = u_out_batch.reshape(
        (batch_size * num_lead_times), FLAGS.direct_steps, num_grid_points, -1)
    specs_batch = specs_batch.reshape(
        (batch_size * num_lead_times), -1)
    dt_batch = dt_batch.reshape(
        (batch_size * num_lead_times), FLAGS.direct_steps)
    # Concatenate the outputs along the batch axis
    u_lag_batch = (u_lag_batch.repeat(repeats=FLAGS.direct_steps, axis=1)
      .reshape((batch_size * num_lead_times * FLAGS.direct_steps), 1, num_grid_points, -1))
    u_out_batch = (u_out_batch
      .reshape((batch_size * num_lead_times * FLAGS.direct_steps), 1, num_grid_points, -1))
    specs_batch = (specs_batch.repeat(repeats=FLAGS.direct_steps, axis=1)
      .reshape((batch_size * num_lead_times * FLAGS.direct_steps), -1))
    dt_batch = (dt_batch
      .reshape((batch_size * num_lead_times * FLAGS.direct_steps)))

    # Shuffle the input/outputs along the batch axis
    if key is not None:
      specs_batch, u_lag_batch, u_out_batch, dt_batch = shuffle_arrays(
        key, [specs_batch, u_lag_batch, u_out_batch, dt_batch])
    # Calculate loss and grads and update state
    loss, grads = get_loss_and_grads(
        params=state.params,
        specs=specs_batch,
        u_lag=u_lag_batch,
        u_out=u_out_batch,
        dt=dt_batch,
    )

    state = state.apply_gradients(grads=grads)

    return state, loss

  def train_one_epoch(state: TrainState, key: flax.typing.PRNGKey) -> Tuple[TrainState, Array]:
    """Updates the state based on accumulated losses and gradients."""

    # Shuffle and split to batches
    subkey, key = jax.random.split(key)
    trajectories, specs = shuffle_arrays(
      subkey, [dataset_trn['trajectories_nrm'], dataset_trn['specs']])
    batches = (
      jnp.split(trajectories, num_batches),
      jnp.split(specs, num_batches)
    )

    # Loop over the batches
    loss_epoch = 0.
    for idx, batch in enumerate(zip(*batches)):
      begin_batch = time()
      subkey, key = jax.random.split(key)
      state, loss = train_one_batch(state, batch, subkey)
      loss_epoch += loss * batch_size / num_samples_trn
      time_batch = time() - begin_batch

      if FLAGS.verbose and not (idx % (num_batches // 5)):
        logging.info('\t'.join([
          f'\t',
          f'BTCH: {idx+1:04d}/{num_batches:04d}',
          f'TIME: {time_batch:06.1f}s',
          f'LOSS: {loss:.2e}',
        ]))

    return state, loss_epoch

  @functools.partial(jax.jit, static_argnames=('num_steps',))
  def predict_trajectory(
      state: TrainState, specs: Array, input: Array, num_steps: int,
      stats_input: Tuple[Array, Array], stats_target: Tuple[Array, Array],
    ) -> Array:
    """
    Normalizes the input and predicts the trajectories autoregressively.
    The input dataset must be raw (not normalized).
    """

    # Normalize the input
    input, _ = normalize(input, stats=stats_input)
    # Get normalized predictions
    variables = {'params': state.params}
    rollout, _ = predictor_full(
      variables=variables,
      specs=specs,
      u_inp=input,
      num_steps=num_steps,
      num_steps_direct=FLAGS.direct_steps,
    )
    # Denormalize the predictions
    rollout = unnormalize(rollout, stats=stats_target)

    return rollout

  def evaluate(state: TrainState, dataset: Array,
        parts: Union[Sequence[int], int] = 1) -> EvalMetrics:
      """Evaluates the model on a dataset based on multiple trajectory lengths."""

      # Initialize the containers
      if isinstance(parts, int):
        parts = [parts]
      error_l1 = {p: [] for p in parts}
      error_l2 = {p: [] for p in parts}

      for p in parts:
        for idx_sub_trajectory in np.split(np.arange(num_times), p):
          # Get the input/target time indices
          idx_input = idx_sub_trajectory[:1]
          idx_target = idx_sub_trajectory[:]
          # Split the dataset along the time axis
          specs = dataset['specs']
          input = dataset['trajectories'][:, idx_input]
          target = dataset['trajectories'][:, idx_target]
          # Get predictions and target
          pred = predict_trajectory(
            state=state,
            specs=specs,
            input=input,
            num_steps=target.shape[1],
            stats_input=tuple([s[:, idx_input] for s in stats_trn]),
            stats_target=tuple([s[:, idx_target] for s in stats_trn]),
          )
          # Compute and store metrics
          error_l1_per_var = rel_l1_error(pred, target)
          error_l2_per_var = rel_l2_error(pred, target)
          error_l1[p].append(jnp.sqrt(jnp.mean(jnp.power(error_l1_per_var, 2))).item())
          error_l2[p].append(jnp.sqrt(jnp.mean(jnp.power(error_l2_per_var, 2))).item())

      # Build the metrics object
      metrics = EvalMetrics(
        error_l1=[(p, errors) for p, errors in error_l1.items()],
        error_l2=[(p, errors) for p, errors in error_l2.items()],
      )

      return metrics

  # Set the evaluation partitions
  eval_parts = [1, 4, 8, 16]
  assert all([(num_times / p) >= FLAGS.direct_steps for p in eval_parts])
  # Evaluate before training
  metrics_trn = evaluate(state=state, dataset=dataset_trn, parts=eval_parts)
  metrics_val = evaluate(state=state, dataset=dataset_val, parts=eval_parts)

  # Report the initial evaluations
  time_tot_pre = time() - time_int_pre
  logging.info('\t'.join([
    f'EPCH: {0 : 04d}/{epochs : 04d}',
    f'TIME: {time_tot_pre : 06.1f}s',
    f'LR: {state.opt_state.hyperparams["learning_rate"].item() : .2e}',
    f'RMSE: {0. : .2e}',
    f'L2/{eval_parts[0]}: {metrics_val.error_l2[0][1][0] * 100 : .2f}%',
    f'L2/{eval_parts[1]}: {metrics_val.error_l2[1][1][0] * 100 : .2f}%',
  ]))

  # Set up the checkpoint manager
  with disable_logging(level=logging.FATAL):
    (DIR / 'metrics').mkdir()
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    checkpointer_options = orbax.checkpoint.CheckpointManagerOptions(
      max_to_keep=1,
      keep_period=None,
      best_fn=(lambda metrics: metrics['loss']),
      best_mode='min',
      create=True,)
    checkpointer_save_args = orbax_utils.save_args_from_target(target={'state': state})
    checkpoint_manager = orbax.checkpoint.CheckpointManager(
      (DIR / 'checkpoints'), checkpointer, checkpointer_options)

  for epoch in range(1, epochs+1):
    # Store the initial time
    time_int = time()

    # Train one epoch
    subkey, key = jax.random.split(key)
    state, loss = train_one_epoch(state, key=subkey)

    # Evaluate
    metrics_trn = evaluate(state=state, dataset=dataset_trn, parts=eval_parts)
    metrics_val = evaluate(state=state, dataset=dataset_val, parts=eval_parts)

    # Log the results
    time_tot = time() - time_int
    logging.info('\t'.join([
      f'EPCH: {epoch : 04d}/{epochs : 04d}',
      f'TIME: {time_tot : 06.1f}s',
      f'LR: {state.opt_state.hyperparams["learning_rate"].item() : .2e}',
      f'RMSE: {np.sqrt(loss).item() : .2e}',
      f'L2/{eval_parts[0]}: {metrics_val.error_l2[0][1][0] * 100 : .2f}%',
      f'L2/{eval_parts[1]}: {metrics_val.error_l2[1][1][0] * 100 : .2f}%',
    ]))

    with disable_logging(level=logging.FATAL):
      checkpoint_metrics = {
        'loss': loss.item(),
        'train': {
          'l1': metrics_trn.error_l1,
          'l2': metrics_trn.error_l2
        },
        'valid': {
          'l1': metrics_val.error_l1,
          'l2': metrics_val.error_l2
        },
      }
      # Store the state and the metrics
      checkpoint_manager.save(
        step=epoch,
        items={'state': state,},
        metrics=checkpoint_metrics,
        save_kwargs={'save_args': checkpointer_save_args}
      )
      with open(DIR / 'metrics' / f'{str(epoch)}.json', 'w') as f:
        json.dump(checkpoint_metrics, f)

  return state

def get_model(model_configs: Mapping[str, Any]) -> AbstractOperator:
  model = GraphNeuralPDESolver(
    **model_configs,
  )

  return model

def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # Check the available devices
  with disable_logging():
    process_index = jax.process_index()
    process_count = jax.process_count()
    local_devices = jax.local_devices()
  logging.info('JAX host: %d / %d', process_index, process_count)
  logging.info('JAX local devices: %r', local_devices)
  # We only support single-host training.
  assert process_count == 1

  # Read the datasets
  experiment = FLAGS.experiment
  datasets = read_datasets(
    dir=FLAGS.datadir, pde_type=PDETYPE[experiment],
    experiment=experiment, nx=FLAGS.resolution, downsample_x=True)
  assert np.all(datasets['test']['dt'] == datasets['valid']['dt'])
  assert np.all(datasets['test']['dt'] == datasets['train']['dt'])
  assert np.all(datasets['test']['x'] == datasets['valid']['x'])
  assert np.all(datasets['test']['x'] == datasets['train']['x'])
  domain = {
    't': {
      'delta': datasets['test']['dt'],
      'range': (datasets['test']['tmin'], datasets['test']['tmax']),
    },
    'x': {
      'delta': datasets['test']['dx'],
      'range': datasets['test']['range_x']
    }
  }
  datasets = jax.tree_map(jax.device_put, datasets)
  for space_dim in domain.keys():
    if 'grid' in domain[space_dim]:
      domain[space_dim]['grid'] = jax.device_put(domain[space_dim]['grid'])

  # Check the array devices
  assert jax.devices()[0] in datasets['train']['trajectories'].devices()
  assert jax.devices()[0] in datasets['train']['specs'].devices()
  assert jax.devices()[0] in datasets['valid']['trajectories'].devices()
  assert jax.devices()[0] in datasets['valid']['specs'].devices()

  # Read the checkpoint
  if FLAGS.params:
    DIR_OLD_EXPERIMENT = DIR_EXPERIMENTS / FLAGS.params
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    step = orbax.checkpoint.CheckpointManager(DIR_OLD_EXPERIMENT / 'checkpoints', orbax_checkpointer).latest_step()
    ckpt = orbax_checkpointer.restore(directory=(DIR_OLD_EXPERIMENT / 'checkpoints' / str(step) / 'default'))
    state = ckpt['state']
    with open(DIR_OLD_EXPERIMENT / 'configs.json', 'rb') as f:
      model_kwargs = json.load(f)['model_configs']
  else:
    state = None
    model_kwargs = None

  # Get the model
  if not model_kwargs:
    model_kwargs = dict(
      domain=domain,
      num_outputs=datasets['valid']['trajectories'].shape[3],
      latent_size=(2 if FLAGS.debug else FLAGS.latent_size),
      time_conditioned=True,
    )
  model = get_model(model_kwargs)

  # Store the configurations
  DIR.mkdir()
  flags = {f: FLAGS.get_flag_value(f, default=None) for f in FLAGS}
  with open(DIR / 'configs.json', 'w') as f:
    json.dump(fp=f,
      obj={'flags': flags, 'model_configs': model.configs},
      indent=2,
    )

  # Train the model
  key = jax.random.PRNGKey(SEED)
  state = train(
    model=model,
    dataset_trn=(datasets['valid'] if FLAGS.debug else datasets['train']),
    dataset_val=datasets['valid'],
    epochs=FLAGS.epochs,
    key=key,
    params=(state['params'] if state else None),
  )

if __name__ == '__main__':
  logging.set_verbosity('info')
  app.run(main)
