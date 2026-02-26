from typing import Callable
from time import time

from flax.training import orbax_utils
import jax
import orbax.checkpoint

from pobax.config import Hyperparams
from pobax.utils.file_system import get_results_path


def vmap_and_train(args: Hyperparams,
                   train_fn: Callable,
                   hparams: dict,
                   rng: jax.random.PRNGKey):
    rngs = jax.random.split(rng, args.n_seeds)

    vmap_seeds_train_fn = jax.vmap(train_fn, in_axes=[None, 0])
    vmap_train_fn = jax.vmap(vmap_seeds_train_fn, in_axes=[0, None])
    train_jit = jax.jit(vmap_train_fn)

    t = time()

    out = jax.block_until_ready(train_jit(hparams, rngs))

    new_t = time()
    total_runtime = new_t - t
    print('Total runtime:', total_runtime)

    # our final_eval_metric returns max_num_steps.
    # we can filter that down by the max episode length amongst the runs.
    final_eval = out['final_eval_metric']

    final_train_state = out['runner_state'][0]
    if not args.save_runner_state:
        del out['runner_state']

    # Pull intermediate checkpoints out of out before saving — we write them
    # as separate Orbax dirs rather than a single batched array.
    ckpt_train_states = out.pop('ckpt_train_states', None)

    results_path = get_results_path(args, return_npy=False)  # returns a results directory

    all_results = {
        'swept_hparams': hparams,
        'out': out,
        'args': args.as_dict(),
        'total_runtime': total_runtime,
        'final_train_state': final_train_state,
        'final_eval': final_eval
    }

    # Convert to numpy
    all_results = jax.device_get(all_results)

    # Save all results with Orbax
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(all_results)

    print(f"Saving results to {results_path}")
    orbax_checkpointer.save(results_path, all_results, save_args=save_args)

    # Save each intermediate checkpoint as its own subdirectory: checkpoint_0, checkpoint_1, ...
    # Shape of each leaf in ckpt_train_states: [n_hparams, n_seeds, num_checkpoints, *leaf_shape]
    if ckpt_train_states is not None:
        ckpt_train_states = jax.device_get(ckpt_train_states)
        num_ckpts = getattr(args, 'num_checkpoints', 0)
        for i in range(num_ckpts):
            ckpt_i = jax.tree.map(lambda x, idx=i: x[:, :, idx], ckpt_train_states)
            ckpt_path = results_path / f"checkpoint_{i}"
            orbax_checkpointer.save(ckpt_path, ckpt_i, save_args=orbax_utils.save_args_from_target(ckpt_i))
        print(f"Saved {num_ckpts} checkpoints to {results_path}/checkpoint_{{0..{num_ckpts - 1}}}")

    print("Done.")
