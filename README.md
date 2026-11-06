# fleet-ops-gym

A small reinforcement-learning environment for satellite fleet operations, meant to run
entirely in the browser on [jax-js](https://github.com/ekzhang/jax-js) (WebGPU / Wasm).

Twelve imaging satellites (3 orbital planes x 4) share four ground stations. Every minute a
policy decides, for each satellite, whether to **image**, **downlink** or **idle/charge**,
subject to battery, onboard storage, eclipse cycles and ground-station contention (a station
serves one satellite per step). Reward is the amount of data delivered to the ground, minus a
small penalty for running a battery flat.

## Run it

Requires Node 18 or newer.

```
npm install
npm run dev          # the app, with live training in a web worker
npm test             # simulator, jax-js parity and PPO tests
npm run train -- --iters 150 --envs 16 --out public/pretrained.json
```

Other scripts: `npm run build`, `npm run preview`, `npm run typecheck`, `npm run baselines`,
`npm run bench`.

The app loads `public/pretrained.json` (the policy trained below) so the replay works at once.
"Train" continues from the loaded weights; "Start over" trains from random weights.

## Results

Mean over 32 held-out seeds (5000 to 5031). Training episodes use a different seed range.

| Policy | Data delivered (buffers) | Reward |
| --- | --- | --- |
| Random | 8.9 | 89 |
| Greedy (default thresholds) | 21.0 | 210 |
| Tuned greedy (thresholds swept on other seeds) | 23.1 | 231 |
| **Learned policy, PPO, 150 iterations** | **23.8** | **237** |

The learned policy beats the tuned greedy by **+5.4 reward (standard error 0.5, paired over the
same 32 seeds)**, which is about 2%. The effect is real but modest, and the evidence is one
training run (seed 0) on one fleet layout. Worth knowing:

- Training took about 8 minutes on Wasm in Node. Learning is steady rather than sudden: the
  policy reaches the tuned greedy's level after about 50 iterations and keeps edging past it.
- A shorter 60-iteration run was statistically tied with the tuned greedy (-0.8 +/- 1.1).
- In this environment a failed downlink costs nothing, so the policy uses "downlink" as a free
  idle. That is why the map shows many hollow diamonds ("downlink not granted").
- Per-seed scores vary a lot (standard deviation about 52), so always compare policies on the
  same seeds, as `train.ts` does, rather than comparing means from different runs.

## Structure

| Path | What it is |
| --- | --- |
| `src/sim/` | Reference simulator in plain TypeScript: the ground truth. Tested. |
| `src/jax/env.ts` | The same environment vectorized over `[batch, satellites]` and jit-compiled with jax-js. |
| `src/jax/parity.test.ts` | Runs both side by side on random actions: max difference 7e-6, zero mismatches. |
| `src/rl/network.ts` | Actor-critic MLP (2 x 64, tanh) shared by all satellites. |
| `src/rl/ppo.ts` | Rollouts through the jax-js env, GAE, clipped PPO with `@jax-js/optax` Adam. |
| `src/rl/evaluate.ts` | Scores any policy on the reference simulator; records episodes for the UI. |
| `src/ui/` | The app: worker, ground-track map, charts. |

Every satellite is an agent with the same weights, so one batch holds `envs x 12` agents.
There is no `scan` in this version of jax-js, so rollouts are a JavaScript loop over a jitted
step.

## Current Limitations

- Orbits are circular, the sun direction is fixed and ground contact is a simple cone.
- Satellites do not see each other, so contention for a station is handled only implicitly.
- Cross-origin isolation headers are set in `vite.config.ts` so Wasm can use threads. If you
  deploy elsewhere, serve `Cross-Origin-Opener-Policy: same-origin` and
  `Cross-Origin-Embedder-Policy: require-corp`, or Wasm runs single-threaded.

## jax-js notes (0.1.27)

- **No `scan` or `while_loop`.** Rollouts are a JS loop calling the jitted step.
- **Ownership.** Functions consume their array arguments; pass `x.ref` for every use except the
  last. `optimizer.init(params)` also takes ownership of `params`.
- **Tree types need `type`, not `interface`.** An `interface` has no implicit index signature,
  so it is rejected as a `JsTree`.
- **`slice`:** `[]` keeps an axis, `null` adds one, a number indexes and removes it.
- **A worker resolves relative URLs against its own script**, not the page. Pass it an absolute
  URL for any file it fetches.
- **jax-js assumes a modern runtime.** It reads `Float16Array`, `navigator`, and calls
  `Array.prototype.toSpliced` / `toReversed`. `src/jax/polyfill.ts` shims these for Node
  scripts (Node 18 to 23); browsers need none of it.


## License

This project is released under the [MIT License](LICENSE).
