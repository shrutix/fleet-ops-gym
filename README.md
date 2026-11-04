# fleet-ops-gym

A small reinforcement-learning environment for satellite fleet operations, meant to run
entirely in the browser on [jax-js](https://github.com/ekzhang/jax-js) (WebGPU / Wasm).

A constellation of 12 imaging satellites (3 planes x 4) must decide, every minute, whether to
**image**, **downlink**, or **idle/charge**, subject to battery, onboard storage, eclipse
cycles, and ground-station contention (each station serves one satellite per step). Reward is
the amount of data actually delivered to the ground, minus a penalty for brownouts.

## Status

| Piece | State |
| --- | --- |
| Reference simulator (plain TypeScript, `src/sim`) | Done, tested |
| Greedy and random baselines | Done |
| Vectorized jax-js environment (`src/jax/env.ts`, `jit` over `[batch, sats]`) | Done, matches the reference |
| PPO trainer (`@jax-js/optax`) | Not started |
| Browser UI (ground tracks, reward curves, WebGPU) | Not started |

The reference simulator is deliberately scalar and loop-based. `src/jax/parity.test.ts` runs it
side by side with the jax-js environment (3 seeds, 360 steps, identical random actions) and
compares observations, rewards, battery, storage and delivered data. Current result: max
absolute difference 7e-6, zero mismatches.

## Run it

Requires Node 18 or newer. Tests, baselines and the benchmark run through `tsx`, and were
checked on Node 18.20.8 and Node 22.23.4. Older Node versions need a few Node-only shims for
jax-js, which `src/jax/polyfill.ts` provides (see below).

```
git clone <https://github.com/shrutix/fleet-ops-gym.git>
cd fleet-ops-gym

npm install
npm test            # reference simulator tests + jax-js parity test
npm run baselines   # random vs greedy returns
npm run bench       # vectorized env throughput on the Wasm backend
```

Baselines (16 episodes, 360 steps each):

```
random   delivered=8.81   reward=88.0   brownout-steps=2.9
greedy   delivered=20.43  reward=204.3  brownout-steps=0.0
```

A sweep over the greedy thresholds found settings about 10% better (delivered 22.5), so a
learned policy has to beat that, not just the default.

Throughput of the jit-compiled step on Wasm in Node (float32, single run, no WebGPU available
in Node): about 14-16k environment-steps per second, scaling linearly with batch size.
WebGPU numbers have not been measured; run `npm run bench`-style code in a browser to get them.

## jax-js notes (version 0.1.27)

Things that cost time and are worth knowing:

- **No `scan` or `while_loop`.** Rollouts are a JS loop calling the jitted step function.
- **Ownership.** Functions consume their array arguments; pass `x.ref` for every use except the
  last. This includes `optimizer.init(params)`, which takes ownership of `params`.
- **`slice`:** `[]` keeps an axis (like `:`), `null` adds a new axis (like `np.newaxis`), a number
  indexes and removes the axis. `slice(null, null, 2)` is not "the third column".
- **`np.allclose` returns a plain boolean**, not an array.
- **jax-js assumes a modern JS runtime.** It reads `Float16Array` (float64 arrays throw
  `Float16Array is not defined` before Node 24), `navigator` (missing before Node 21), and calls
  `Array.prototype.toSpliced` / `toReversed` (missing before Node 20). `src/jax/polyfill.ts`
  shims exactly these for Node scripts and tests; browsers need none of it. float32, which is
  what you want on WebGPU anyway, is unaffected by the `Float16Array` gap.

## Known limitations

- Orbits are circular and the sun direction is fixed; ground contact is a simple cone.
- Imaging-limited regime: simple threshold rules are already close to saturated, so the real
  test is whether lookahead features (`demand_ahead`, `contact_ahead`) give PPO an edge.
