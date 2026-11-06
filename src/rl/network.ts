// Actor-critic MLP shared by every satellite. Each satellite is an agent that sees its own
// observation vector; all agents use the same weights (parameter sharing).
//
// Ownership: jax-js consumes array arguments, so each parameter is used exactly once per
// forward pass, and callers pass `refTree(params)` when they want to keep their copy.

import { jit, nn, numpy as np, random } from "@jax-js/jax";

import { NUM_ACTIONS, OBS_DIM } from "../sim/constants.ts";

type Arr = np.Array;

// A `type` alias, not an `interface`: jax-js tree types require an implicit index signature.
export type Params = {
    w1: Arr;
    b1: Arr;
    w2: Arr;
    b2: Arr;
    wp: Arr;
    bp: Arr;
    wv: Arr;
    bv: Arr;
};

export type ParamKey = keyof Params;
export const PARAM_KEYS: ParamKey[] = ["w1", "b1", "w2", "b2", "wp", "bp", "wv", "bv"];

/** A new reference to every array, so the original tree stays alive after a consuming call. */
export function refTree(p: Params): Params {
    const out = {} as Params;
    for (const k of PARAM_KEYS) out[k] = p[k].ref;
    return out;
}

export function disposeTree(p: Params): void {
    for (const k of PARAM_KEYS) p[k].dispose();
}

export function initParams(seed: number, hidden = 64): Params {
    let key = random.key(seed);
    const normal = (shape: number[], scale: number): Arr => {
        const ks = random.split(key, 2);
        key = ks.ref.slice(0);
        const sub = ks.slice(1);
        return random.normal(sub, shape).mul(scale);
    };
    const params: Params = {
        w1: normal([OBS_DIM, hidden], Math.sqrt(1 / OBS_DIM)),
        b1: np.zeros([hidden]),
        w2: normal([hidden, hidden], Math.sqrt(1 / hidden)),
        b2: np.zeros([hidden]),
        wp: normal([hidden, NUM_ACTIONS], 0.01), // near-uniform initial policy
        bp: np.zeros([NUM_ACTIONS]),
        wv: normal([hidden, 1], 1),
        bv: np.zeros([1]),
    };
    key.dispose();
    return params;
}

/** Logits [M, A] and state value [M] for observations [M, OBS_DIM]. Consumes params and obs. */
export function forward(p: Params, obs: Arr): { logits: Arr; value: Arr } {
    const m = obs.shape[0];
    const h1 = np.tanh(np.matmul(obs, p.w1).add(p.b1));
    const h2 = np.tanh(np.matmul(h1, p.w2).add(p.b2));
    const logits = np.matmul(h2.ref, p.wp).add(p.bp);
    const value = np.matmul(h2, p.wv).add(p.bv).reshape([m]);
    return { logits, value };
}

/** Sample actions. Returns action [M] (int32), its log-probability [M] and value [M]. */
export const actSample = jit((p: Params, obs: Arr, key: Arr) => {
    const { logits, value } = forward(p, obs);
    const action = random.categorical(key, logits.ref);
    const logp = np.sum(nn.logSoftmax(logits).mul(nn.oneHot(action.ref, NUM_ACTIONS)), 1);
    return { action, logp, value };
});

/** Most likely action [M] (int32). */
export const actGreedy = jit((p: Params, obs: Arr) => {
    const { logits, value } = forward(p, obs);
    value.dispose();
    return np.argmax(logits, 1);
});

// ---- (de)serialization -------------------------------------------------------------------

export interface SerializedParams {
    [key: string]: number[] | number[][];
}

export async function serializeParams(p: Params): Promise<SerializedParams> {
    const out: SerializedParams = {};
    for (const k of PARAM_KEYS) out[k] = (await p[k].ref.jsAsync()) as number[] | number[][];
    return out;
}

export function deserializeParams(s: SerializedParams): Params {
    const out = {} as Params;
    for (const k of PARAM_KEYS) out[k] = np.array(s[k] as number[]);
    return out;
}
