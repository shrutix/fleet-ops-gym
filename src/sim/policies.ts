// Hand-written baseline policies. They only see the same per-satellite
// observation vector that the learned policy will see, so the comparison is fair.

import {
    ACTION_DOWNLINK,
    ACTION_IDLE,
    ACTION_IMAGE,
    NUM_ACTIONS,
    NUM_SATS,
    OBS_DIM,
} from "./constants.ts";

export type Policy = (obs: Float32Array) => Uint8Array;

// Observation layout (see FleetEnv.observe).
const BATTERY = 0;
const STORAGE = 1;
const SUNLIT = 2;
const DEMAND = 3;
const IN_CONTACT = 5;

export interface GreedyParams {
    /** Only image when demand at the sub-point exceeds this. */
    minDemand: number;
    /** Only image when the battery is above this (keeps a power reserve). */
    minBattery: number;
    /** Stop imaging once the buffer is fuller than this. */
    maxStorage: number;
}

export const DEFAULT_GREEDY: GreedyParams = { minDemand: 0.35, minBattery: 0.3, maxStorage: 0.85 };

/** Downlink when in contact, image when over demand in sunlight, else charge. */
export function makeGreedyPolicy(params: GreedyParams = DEFAULT_GREEDY): Policy {
    return (obs) => {
        const actions = new Uint8Array(NUM_SATS);
        for (let i = 0; i < NUM_SATS; i++) {
            const o = i * OBS_DIM;
            const battery = obs[o + BATTERY];
            const storage = obs[o + STORAGE];
            if (obs[o + IN_CONTACT] > 0.5 && storage > 0.05 && battery > 0.12) {
                actions[i] = ACTION_DOWNLINK;
            } else if (
                obs[o + SUNLIT] > 0.5 &&
                obs[o + DEMAND] > params.minDemand &&
                storage < params.maxStorage &&
                battery > params.minBattery
            ) {
                actions[i] = ACTION_IMAGE;
            } else {
                actions[i] = ACTION_IDLE;
            }
        }
        return actions;
    };
}

export const greedyPolicy: Policy = makeGreedyPolicy();

export function makeRandomPolicy(rng: () => number): Policy {
    return () => {
        const actions = new Uint8Array(NUM_SATS);
        for (let i = 0; i < NUM_SATS; i++) actions[i] = Math.floor(rng() * NUM_ACTIONS);
        return actions;
    };
}

export interface EpisodeStats {
    totalReward: number;
    delivered: number;
    brownouts: number;
}

import { FleetEnv } from "./env.ts";

export function runEpisode(env: FleetEnv, policy: Policy): EpisodeStats {
    let obs = env.reset();
    let totalReward = 0;
    for (; ;) {
        const { obs: next, reward, done } = env.step(policy(obs));
        for (const r of reward) totalReward += r;
        obs = next;
        if (done) break;
    }
    return { totalReward, delivered: env.totalDelivered, brownouts: env.totalBrownouts };
}
