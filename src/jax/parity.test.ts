import assert from "node:assert/strict";
import { test } from "node:test";

import "./polyfill.ts";
import { defaultDevice, init, numpy as np } from "@jax-js/jax";

import { DT, EPISODE_STEPS, NUM_ACTIONS, NUM_SATS, OBS_DIM } from "../sim/constants.ts";
import { FleetEnv, makeRng } from "../sim/env.ts";
import { makeJaxEnv } from "./env.ts";

const SEEDS = [11, 12, 13];
const TOL = 1e-4;

interface Tally {
    maxDiff: number;
    bad: number;
    firstBad: string | null;
}

function compare(tally: Tally, label: string, step: number, got: ArrayLike<number>, want: ArrayLike<number>[]) {
    const per = want[0].length;
    for (let b = 0; b < want.length; b++) {
        for (let j = 0; j < per; j++) {
            const diff = Math.abs(got[b * per + j] - want[b][j]);
            tally.maxDiff = Math.max(tally.maxDiff, diff);
            if (!(diff <= TOL)) {
                tally.bad++;
                tally.firstBad ??= `${label} step=${step} env=${b} index=${j} got=${got[b * per + j]} want=${want[b][j]}`;
            }
        }
    }
}

test("vectorized jax-js env matches the reference simulator", async () => {
    await init();
    defaultDevice("wasm");
    const dtype = np.float64;
    const { step, observe } = makeJaxEnv(dtype);

    const refs = SEEDS.map((s) => new FleetEnv(s));
    const refObs = refs.map((e) => e.reset());
    const snaps = refs.map((e) => e.snapshot());

    let battery = np.array(snaps.map((s) => s.battery), { dtype });
    let storage = np.array(snaps.map((s) => s.storage), { dtype });
    const u0 = np.array(snaps.map((s) => s.u0), { dtype });
    const theta0 = np.array(snaps.map((s) => s.theta0), { dtype });

    const tally: Tally = { maxDiff: 0, bad: 0, firstBad: null };

    // Initial observation.
    const obs0 = observe(battery.ref, storage.ref, u0.ref, theta0.ref, np.array(0, { dtype }));
    compare(tally, "initial obs", 0, await obs0.data(), refObs);

    const rngs = SEEDS.map((s) => makeRng(s + 1000));
    let totalDelivered = 0;

    for (let k = 0; k < EPISODE_STEPS; k++) {
        const actionRows = rngs.map(() =>
            Array.from({ length: NUM_SATS }, (_, i) => Math.floor(rngs[0]() * NUM_ACTIONS) + 0 * i),
        );
        const refRes = refs.map((e, b) => e.step(actionRows[b]));

        const out = step(
            battery,
            storage,
            u0.ref,
            theta0.ref,
            np.array(k * DT, { dtype }),
            np.array(actionRows, { dtype: np.int32 }),
        );
        battery = out.battery;
        storage = out.storage;

        compare(tally, "obs", k, await out.obs.data(), refRes.map((r) => r.obs));
        compare(tally, "reward", k, await out.reward.data(), refRes.map((r) => r.reward));
        compare(tally, "delivered", k, await out.delivered.data(), refRes.map((r) => [r.delivered]));
        compare(tally, "battery", k, await battery.ref.data(), refs.map((e) => e.battery));
        compare(tally, "storage", k, await storage.ref.data(), refs.map((e) => e.storage));
        out.brownouts.dispose();
        totalDelivered += refRes.reduce((a, r) => a + r.delivered, 0);
    }

    battery.dispose();
    storage.dispose();
    u0.dispose();
    theta0.dispose();

    console.log(`parity: max abs diff ${tally.maxDiff.toExponential(2)}, mismatches ${tally.bad}`);
    assert.ok(totalDelivered > 0, "test is vacuous: nothing was ever delivered");
    assert.equal(tally.bad, 0, `first mismatch: ${tally.firstBad}`);
    assert.equal(OBS_DIM, 10);
});
