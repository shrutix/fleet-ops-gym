import assert from "node:assert/strict";
import { test } from "node:test";

import { EPISODE_STEPS, NUM_SATS, OBS_DIM, ACTION_IDLE } from "./constants.ts";
import { FleetEnv, makeRng } from "./env.ts";
import { greedyPolicy, makeRandomPolicy, runEpisode } from "./policies.ts";

function meanDelivered(make: (seed: number) => ReturnType<typeof makeRandomPolicy>, n = 8) {
    let sum = 0;
    for (let seed = 0; seed < n; seed++) {
        sum += runEpisode(new FleetEnv(seed), make(seed)).delivered;
    }
    return sum / n;
}

test("same seed gives identical episodes", () => {
    const a = runEpisode(new FleetEnv(7), greedyPolicy);
    const b = runEpisode(new FleetEnv(7), greedyPolicy);
    assert.deepEqual(a, b);
});

test("different seeds give different episodes", () => {
    const a = runEpisode(new FleetEnv(1), greedyPolicy);
    const b = runEpisode(new FleetEnv(2), greedyPolicy);
    assert.notEqual(a.totalReward, b.totalReward);
});

test("state stays in range and observations are finite", () => {
    const env = new FleetEnv(3);
    const rng = makeRng(99);
    const policy = makeRandomPolicy(rng);
    let obs = env.reset();
    let steps = 0;
    for (; ;) {
        assert.equal(obs.length, NUM_SATS * OBS_DIM);
        for (const v of obs) assert.ok(Number.isFinite(v));
        for (let i = 0; i < NUM_SATS; i++) {
            assert.ok(env.battery[i] >= 0 && env.battery[i] <= 1);
            assert.ok(env.storage[i] >= 0 && env.storage[i] <= 1);
        }
        const res = env.step(policy(obs));
        obs = res.obs;
        steps++;
        if (res.done) break;
    }
    assert.equal(steps, EPISODE_STEPS);
});

test("doing nothing never earns positive reward", () => {
    const idle = () => new Uint8Array(NUM_SATS).fill(ACTION_IDLE);
    const stats = runEpisode(new FleetEnv(5), idle);
    assert.ok(stats.totalReward <= 0);
    assert.equal(stats.delivered, 0);
});

test("greedy baseline clearly beats a random policy", () => {
    const random = meanDelivered((seed) => makeRandomPolicy(makeRng(seed + 1000)));
    const greedy = meanDelivered(() => greedyPolicy);
    assert.ok(greedy > 0, "greedy should deliver some data");
    assert.ok(greedy > 1.5 * random, `greedy ${greedy} vs random ${random}`);
});
