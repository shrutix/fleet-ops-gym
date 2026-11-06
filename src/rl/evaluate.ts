// Evaluation and episode recording on the reference simulator. The reference simulator is
// the ground truth (the jax-js env is tested to match it), so every policy, learned or
// hand-written, is scored by the same code on the same seeds.

import { NUM_SATS, OBS_DIM } from "../sim/constants.ts";
import { FleetEnv } from "../sim/env.ts";
import type { Policy } from "../sim/policies.ts";

/** Chooses an action for every satellite in every environment: obs [E*N*OBS] -> [E*N]. */
export type BatchPolicy = (obs: Float32Array, numEnvs: number) => Promise<Uint8Array>;

export interface EvalResult {
    reward: number; // mean episode reward, summed over satellites
    delivered: number; // mean storage units delivered per episode
    brownouts: number; // mean satellite-steps spent in safe mode
    rewardStd: number; // standard deviation of episode reward across seeds
    episodes: number;
    /** Reward of every episode, in seed order, so two policies can be compared pairwise. */
    episodeRewards: number[];
}

/** Mean and standard error of (a - b) over episodes that share seeds. */
export function pairedDifference(a: EvalResult, b: EvalResult): { mean: number; stdError: number } {
    const diffs = a.episodeRewards.map((r, i) => r - b.episodeRewards[i]);
    const n = diffs.length;
    const mean = diffs.reduce((x, y) => x + y, 0) / n;
    const variance = diffs.reduce((s, d) => s + (d - mean) ** 2, 0) / (n - 1);
    return { mean, stdError: Math.sqrt(variance / n) };
}

function summarize(rewards: number[], delivered: number[], brownouts: number[]): EvalResult {
    const n = rewards.length;
    const mean = (a: number[]) => a.reduce((x, y) => x + y, 0) / n;
    const mu = mean(rewards);
    const variance = rewards.reduce((s, r) => s + (r - mu) ** 2, 0) / n;
    return {
        reward: mu,
        delivered: mean(delivered),
        brownouts: mean(brownouts),
        rewardStd: Math.sqrt(variance),
        episodes: n,
        episodeRewards: rewards.slice(),
    };
}

/** Score a hand-written (synchronous) policy on a set of seeds. */
export function evaluateSync(policy: Policy, seeds: number[]): EvalResult {
    const rewards: number[] = [];
    const delivered: number[] = [];
    const brownouts: number[] = [];
    for (const seed of seeds) {
        const env = new FleetEnv(seed);
        let obs = env.reset();
        let total = 0;
        for (; ;) {
            const res = env.step(policy(obs));
            for (const r of res.reward) total += r;
            obs = res.obs;
            if (res.done) break;
        }
        rewards.push(total);
        delivered.push(env.totalDelivered);
        brownouts.push(env.totalBrownouts);
    }
    return summarize(rewards, delivered, brownouts);
}

/** Score a batched, possibly asynchronous (GPU) policy; all seeds run in lockstep. */
export async function evaluateBatch(policy: BatchPolicy, seeds: number[]): Promise<EvalResult> {
    const envs = seeds.map((s) => new FleetEnv(s));
    const E = envs.length;
    const obsBuf = new Float32Array(E * NUM_SATS * OBS_DIM);
    envs.forEach((e, k) => obsBuf.set(e.reset(), k * NUM_SATS * OBS_DIM));
    const totals = new Array<number>(E).fill(0);
    let done = false;
    while (!done) {
        const actions = await policy(obsBuf, E);
        for (let k = 0; k < E; k++) {
            const res = envs[k].step(actions.subarray(k * NUM_SATS, (k + 1) * NUM_SATS));
            for (const r of res.reward) totals[k] += r;
            obsBuf.set(res.obs, k * NUM_SATS * OBS_DIM);
            done = res.done;
        }
    }
    return summarize(
        totals,
        envs.map((e) => e.totalDelivered),
        envs.map((e) => e.totalBrownouts),
    );
}

// ---- recording for the UI ---------------------------------------------------------------

/** Values stored per satellite per frame. */
export const FRAME_FIELDS = ["lat", "lon", "battery", "storage", "status", "sunlit", "demand"] as const;
export const FRAME_STRIDE = FRAME_FIELDS.length;

export interface Recording {
    seed: number;
    steps: number;
    /** [steps+1][NUM_SATS][FRAME_STRIDE], flattened. Frame k is the state before step k. */
    frames: Float32Array;
    /** Earth rotation angle per frame, radians. */
    theta: Float32Array;
    /** Cumulative reward and delivered data after each frame. */
    cumReward: Float32Array;
    cumDelivered: Float32Array;
    /** Which action each satellite took at each frame (status after the step). */
    totalReward: number;
    totalDelivered: number;
}

/** Run one episode and record everything the visualization needs. */
export async function recordEpisode(
    seed: number,
    policy: BatchPolicy,
): Promise<Recording> {
    const env = new FleetEnv(seed);
    let obs = env.reset();
    const steps = 360;
    const frames = new Float32Array((steps + 1) * NUM_SATS * FRAME_STRIDE);
    const theta = new Float32Array(steps + 1);
    const cumReward = new Float32Array(steps + 1);
    const cumDelivered = new Float32Array(steps + 1);

    const snapshotFrame = (k: number, statuses: Uint8Array | null) => {
        theta[k] = env.thetaAt(k);
        for (let i = 0; i < NUM_SATS; i++) {
            const g = env.geometry(i, k);
            const o = (k * NUM_SATS + i) * FRAME_STRIDE;
            frames[o + 0] = (Math.asin(Math.max(-1, Math.min(1, g.ecef[2]))) * 180) / Math.PI;
            frames[o + 1] = (Math.atan2(g.ecef[1], g.ecef[0]) * 180) / Math.PI;
            frames[o + 2] = env.battery[i];
            frames[o + 3] = env.storage[i];
            frames[o + 4] = statuses ? statuses[i] : 0;
            frames[o + 5] = g.sunlit ? 1 : 0;
            frames[o + 6] = g.demand;
        }
    };

    let total = 0;
    const obsBuf = new Float32Array(NUM_SATS * OBS_DIM);
    for (let k = 0; k < steps; k++) {
        obsBuf.set(obs);
        snapshotFrame(k, null);
        const actions = await policy(obsBuf, 1);
        const res = env.step(actions);
        // The status describes what the satellite did during step k, so store it on frame k.
        for (let i = 0; i < NUM_SATS; i++) {
            frames[(k * NUM_SATS + i) * FRAME_STRIDE + 4] = env.lastStatus[i];
        }
        for (const r of res.reward) total += r;
        cumReward[k + 1] = total;
        cumDelivered[k + 1] = env.totalDelivered;
        obs = res.obs;
    }
    snapshotFrame(steps, null);
    return {
        seed,
        steps,
        frames,
        theta,
        cumReward,
        cumDelivered,
        totalReward: total,
        totalDelivered: env.totalDelivered,
    };
}

/** Adapt a synchronous single-env policy (greedy, random) to the batch interface. */
export function syncToBatch(policy: Policy): BatchPolicy {
    return async (obs, numEnvs) => {
        const out = new Uint8Array(numEnvs * NUM_SATS);
        for (let k = 0; k < numEnvs; k++) {
            out.set(policy(obs.subarray(k * NUM_SATS * OBS_DIM, (k + 1) * NUM_SATS * OBS_DIM)), k * NUM_SATS);
        }
        return out;
    };
}
