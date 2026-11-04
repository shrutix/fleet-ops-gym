import "./polyfill.ts";
import { defaultDevice, init, numpy as np, blockUntilReady } from "@jax-js/jax";
import { makeJaxEnv } from "./env.ts";
import { NUM_SATS, DT } from "../sim/constants.ts";

await init();
defaultDevice("wasm");
const dtype = np.float32;
const { step } = makeJaxEnv(dtype);

for (const B of [64, 256, 1024]) {
    const u0 = np.array(Array.from({ length: B }, (_, b) => Array.from({ length: NUM_SATS }, (_, i) => (b * 0.37 + i) % 6.28)), { dtype });
    const theta0 = np.array(Array.from({ length: B }, (_, b) => (b * 0.11) % 6.28), { dtype });
    let battery = np.full([B, NUM_SATS], 0.8, { dtype });
    let storage = np.full([B, NUM_SATS], 0.02, { dtype });
    const acts = np.array(Array.from({ length: B }, (_, b) => Array.from({ length: NUM_SATS }, (_, i) => (b + i) % 3)), { dtype: np.int32 });

    const run = (n: number, t0: number) => {
        for (let k = 0; k < n; k++) {
            const out = step(battery, storage, u0.ref, theta0.ref, np.array((t0 + k) * DT, { dtype }), acts.ref);
            battery = out.battery; storage = out.storage;
            out.obs.dispose(); out.reward.dispose(); out.delivered.dispose(); out.brownouts.dispose();
        }
    };
    run(3, 0); // warm-up / compile
    await blockUntilReady(battery.ref);
    const N = 40;
    const t = performance.now();
    run(N, 3);
    await blockUntilReady(battery.ref);
    const ms = performance.now() - t;
    console.log(`B=${String(B).padStart(5)}  ${(ms / N).toFixed(1)} ms/step  ${((B * NUM_SATS * N) / (ms / 1000) / 1e3).toFixed(0)}k sat-steps/s  ${((B * N) / (ms / 1000)).toFixed(0)} env-steps/s`);
    battery.dispose(); storage.dispose(); u0.dispose(); theta0.dispose(); acts.dispose();
}
