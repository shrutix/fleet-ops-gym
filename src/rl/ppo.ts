// Proximal Policy Optimization on the vectorized jax-js environment.
//
// One iteration = collect B full episodes (EPISODE_STEPS steps each) with the current
// policy, compute GAE advantages, then run several epochs of clipped-PPO minibatch updates.
// Every satellite is an agent sharing one network, so a batch holds B * NUM_SATS agents.
//
// Environment-agnostic: no Node or DOM APIs, so it runs in Node, in a browser worker, on
// Wasm or on WebGPU. All device-to-host reads use the async `data()` for that reason.

import { jit, nn, numpy as np, random, valueAndGrad } from "@jax-js/jax";
import * as optax from "@jax-js/optax";

import { DT, EPISODE_STEPS, NUM_ACTIONS, NUM_SATS, OBS_DIM } from "../sim/constants.ts";
import { FleetEnv, makeRng } from "../sim/env.ts";
import { makeJaxEnv } from "../jax/env.ts";
import type { JaxEnv } from "../jax/env.ts";
import {
    actGreedy,
    actSample,
    disposeTree,
    forward,
    initParams,
    refTree,
} from "./network.ts";
import type { Params } from "./network.ts";
import type { BatchPolicy } from "./evaluate.ts";

type Arr = np.Array;

export interface PPOConfig {
    seed: number;
    /** Parallel environments per rollout. */
    envs: number;
    hidden: number;
    gamma: number;
    lambda: number;
    clip: number;
    lr: number;
    /** Learning rate decays linearly to lr * lrFinalFraction over `totalIterations`. */
    lrFinalFraction: number;
    totalIterations: number;
    epochs: number;
    minibatch: number;
    valueCoef: number;
    entropyCoef: number;
    maxGradNorm: number;
}

export const DEFAULT_CONFIG: PPOConfig = {
    seed: 0,
    envs: 16,
    hidden: 64,
    gamma: 0.995,
    lambda: 0.95,
    clip: 0.2,
    lr: 3e-4,
    lrFinalFraction: 0.1,
    totalIterations: 100,
    epochs: 4,
    minibatch: 4096,
    valueCoef: 0.5,
    entropyCoef: 0.01,
    maxGradNorm: 0.5,
};

export interface IterationMetrics {
    iteration: number;
    /** Mean episode reward (summed over satellites) of the rollouts collected this iteration. */
    episodeReward: number;
    delivered: number;
    brownouts: number;
    policyLoss: number;
    valueLoss: number;
    entropy: number;
    approxKl: number;
    clipFraction: number;
    explainedVariance: number;
    learningRate: number;
    /** Environment steps (one step of one satellite) per second, rollout plus update. */
    agentStepsPerSecond: number;
    rolloutSeconds: number;
    updateSeconds: number;
}

interface Rollout {
    obs: Float32Array; // [T, M, OBS]
    action: Int32Array; // [T, M]
    logp: Float32Array; // [T, M]
    value: Float32Array; // [T + 1, M]
    reward: Float32Array; // [T, M]
    episodeReward: number;
    delivered: number;
    brownouts: number;
}

// ---- the PPO loss, differentiated with respect to the parameters ---------------------------

interface LossBatch {
    obs: Arr;
    action: Arr;
    oldLogp: Arr;
    adv: Arr;
    ret: Arr;
}

function makeLossGrad(cfg: PPOConfig) {
    const loss = (p: Params, b: LossBatch) => {
        const { logits, value } = forward(p, b.obs);
        const logpAll = nn.logSoftmax(logits);
        const probs = np.exp(logpAll.ref);
        const onehot = nn.oneHot(b.action, NUM_ACTIONS);
        const logp = np.sum(logpAll.ref.mul(onehot), 1);
        const entropy = np.sum(probs.mul(logpAll), 1).mean().neg();

        const logRatio = logp.sub(b.oldLogp);
        const ratio = np.exp(logRatio.ref);
        const unclipped = ratio.ref.mul(b.adv.ref);
        const clipped = np.clip(ratio.ref, 1 - cfg.clip, 1 + cfg.clip).mul(b.adv);
        const policyLoss = np.minimum(unclipped, clipped).mean().neg();

        const err = value.ref.sub(b.ret.ref);
        const valueLoss = np.square(err.ref).mean();

        const total = policyLoss.ref
            .add(valueLoss.ref.mul(cfg.valueCoef))
            .sub(entropy.ref.mul(cfg.entropyCoef));

        const approxKl = ratio.ref.sub(1).sub(logRatio).mean();
        const clipFraction = np.greater(np.abs(ratio.sub(1)), cfg.clip).astype(np.float32).mean();
        // Explained variance of the value head: 1 - Var(ret - value) / Var(ret).
        const retMean = b.ret.ref.mean();
        const varRet = np.square(b.ret.sub(retMean)).mean();
        const errMean = err.ref.mean();
        const varErr = np.square(err.sub(errMean)).mean();
        const explainedVariance = np.array(1).sub(varErr.div(varRet.add(1e-8)));
        value.dispose();

        return [total, { policyLoss, valueLoss, entropy, approxKl, clipFraction, explainedVariance }] as [
            Arr,
            Record<string, Arr>,
        ];
    };
    return jit(valueAndGrad(loss, { hasAux: true }));
}

// ---- the trainer ---------------------------------------------------------------------------

export class PPOTrainer {
    readonly cfg: PPOConfig;
    params: Params;
    iteration = 0;

    private env: JaxEnv;
    private opt: optax.GradientTransformation;
    private optState: optax.OptState;
    private lossGrad: ReturnType<typeof makeLossGrad>;
    private rng: () => number;
    private key: Arr;
    private episodeCounter = 0;
    private lastLr = 0;
    private stepCount = 0;
    private totalUpdates = 1;

    private lrAt(count: number): number {
        const frac = Math.min(1, count / this.totalUpdates);
        return this.cfg.lr * (1 - (1 - this.cfg.lrFinalFraction) * frac);
    }

    constructor(cfg: Partial<PPOConfig> = {}, initial?: Params) {
        this.cfg = { ...DEFAULT_CONFIG, ...cfg };
        this.params = initial ?? initParams(this.cfg.seed, this.cfg.hidden);
        this.env = makeJaxEnv(np.float32);
        this.lossGrad = makeLossGrad(this.cfg);
        const samples = EPISODE_STEPS * this.cfg.envs * NUM_SATS;
        const perEpoch = Math.max(1, Math.floor(samples / Math.min(this.cfg.minibatch, samples)));
        this.totalUpdates = Math.max(1, this.cfg.totalIterations * this.cfg.epochs * perEpoch);
        this.opt = optax.chain(
            optax.clipByGlobalNorm(this.cfg.maxGradNorm),
            optax.adam((count: number) => this.lrAt(count)),
        );
        this.optState = this.opt.init(refTree(this.params));
        this.rng = makeRng(this.cfg.seed + 7919);
        this.key = random.key(this.cfg.seed + 1);
    }

    /** Replace the policy weights (for example with a pretrained policy) and reset the optimizer. */
    setParams(params: Params): void {
        disposeTree(this.params);
        this.params = params;
        this.optState = this.opt.init(refTree(this.params));
    }

    dispose(): void {
        disposeTree(this.params);
        this.key.dispose();
    }

    private nextKey(): Arr {
        const ks = random.split(this.key, 2);
        this.key = ks.ref.slice(0);
        return ks.slice(1);
    }

    /** Greedy (argmax) batch policy over the current weights, for evaluation. */
    greedyPolicy(): BatchPolicy {
        return async (obs, numEnvs) => {
            const m = numEnvs * NUM_SATS;
            const out = actGreedy(
                refTree(this.params),
                np.array(obs as Float32Array<ArrayBuffer>).reshape([m, OBS_DIM]),
            );
            const data = await out.data();
            const result = new Uint8Array(m);
            for (let i = 0; i < m; i++) result[i] = data[i];
            return result;
        };
    }

    // ---- rollout --------------------------------------------------------------------------

    private async collect(): Promise<Rollout> {
        const { envs: B } = this.cfg;
        const N = NUM_SATS;
        const M = B * N;
        const T = EPISODE_STEPS;
        const dtype = np.float32;

        // Start every environment from the same reset distribution the reference sim uses.
        const u0 = new Float32Array(M);
        const theta0 = new Float32Array(B);
        const battery0 = new Float32Array(M);
        const storage0 = new Float32Array(M);
        for (let b = 0; b < B; b++) {
            const ref = new FleetEnv(1_000_000 + this.cfg.seed * 100_003 + this.episodeCounter++);
            ref.reset();
            const snap = ref.snapshot();
            theta0[b] = snap.theta0;
            for (let i = 0; i < N; i++) {
                u0[b * N + i] = snap.u0[i];
                battery0[b * N + i] = snap.battery[i];
                storage0[b * N + i] = snap.storage[i];
            }
        }
        const u0A = np.array(u0).reshape([B, N]);
        const theta0A = np.array(theta0);
        let battery = np.array(battery0).reshape([B, N]);
        let storage = np.array(storage0).reshape([B, N]);

        const obsBuf = new Float32Array(T * M * OBS_DIM);
        const actionBuf = new Int32Array(T * M);
        const logpBuf = new Float32Array(T * M);
        const valueBuf = new Float32Array((T + 1) * M);
        const rewardBuf = new Float32Array(T * M);
        let delivered = 0;
        let brownouts = 0;

        let obs = this.env.observe(battery.ref, storage.ref, u0A.ref, theta0A.ref, np.array(0, { dtype }));
        obs = obs.reshape([M, OBS_DIM]);

        for (let t = 0; t < T; t++) {
            obsBuf.set(await obs.ref.data(), t * M * OBS_DIM);
            const { action, logp, value } = actSample(refTree(this.params), obs, this.nextKey());
            actionBuf.set(await action.ref.data(), t * M);
            logpBuf.set(await logp.data(), t * M);
            valueBuf.set(await value.data(), t * M);

            const out = this.env.step(
                battery,
                storage,
                u0A.ref,
                theta0A.ref,
                np.array(t * DT, { dtype }),
                action.reshape([B, N]),
            );
            battery = out.battery;
            storage = out.storage;
            rewardBuf.set(await out.reward.data(), t * M);
            for (const d of await out.delivered.data()) delivered += d;
            for (const x of await out.brownouts.data()) brownouts += x;
            obs = out.obs.reshape([M, OBS_DIM]);
        }

        // Bootstrap value of the final state (the episode is truncated by the time limit, not
        // terminated), using a throwaway sample step.
        const last = actSample(refTree(this.params), obs, this.nextKey());
        last.action.dispose();
        last.logp.dispose();
        valueBuf.set(await last.value.data(), T * M);

        battery.dispose();
        storage.dispose();
        u0A.dispose();
        theta0A.dispose();

        let totalReward = 0;
        for (let i = 0; i < rewardBuf.length; i++) totalReward += rewardBuf[i];
        return {
            obs: obsBuf,
            action: actionBuf,
            logp: logpBuf,
            value: valueBuf,
            reward: rewardBuf,
            episodeReward: totalReward / B,
            delivered: delivered / B,
            brownouts: brownouts / B,
        };
    }

    // ---- GAE ------------------------------------------------------------------------------

    private advantages(r: Rollout): { adv: Float32Array; ret: Float32Array } {
        const M = this.cfg.envs * NUM_SATS;
        const T = EPISODE_STEPS;
        const adv = new Float32Array(T * M);
        const ret = new Float32Array(T * M);
        const { gamma, lambda } = this.cfg;
        for (let j = 0; j < M; j++) {
            let gae = 0;
            for (let t = T - 1; t >= 0; t--) {
                const delta = r.reward[t * M + j] + gamma * r.value[(t + 1) * M + j] - r.value[t * M + j];
                gae = delta + gamma * lambda * gae;
                adv[t * M + j] = gae;
                ret[t * M + j] = gae + r.value[t * M + j];
            }
        }
        return { adv, ret };
    }

    // ---- one full iteration ---------------------------------------------------------------

    async iterate(): Promise<IterationMetrics> {
        const t0 = performance.now();
        const rollout = await this.collect();
        const t1 = performance.now();

        const { adv, ret } = this.advantages(rollout);
        const S = adv.length;

        // Normalize advantages over the whole batch.
        let mean = 0;
        for (let i = 0; i < S; i++) mean += adv[i];
        mean /= S;
        let variance = 0;
        for (let i = 0; i < S; i++) variance += (adv[i] - mean) ** 2;
        const std = Math.sqrt(variance / S) + 1e-8;
        for (let i = 0; i < S; i++) adv[i] = (adv[i] - mean) / std;

        const order = new Uint32Array(S);
        for (let i = 0; i < S; i++) order[i] = i;

        const mb = Math.min(this.cfg.minibatch, S);
        const sums = { policyLoss: 0, valueLoss: 0, entropy: 0, approxKl: 0, clipFraction: 0, explainedVariance: 0 };
        let updates = 0;

        for (let epoch = 0; epoch < this.cfg.epochs; epoch++) {
            for (let i = S - 1; i > 0; i--) {
                const j = Math.floor(this.rng() * (i + 1));
                const tmp = order[i];
                order[i] = order[j];
                order[j] = tmp;
            }
            for (let start = 0; start + mb <= S; start += mb) {
                const obsB = new Float32Array(mb * OBS_DIM);
                const actB = new Int32Array(mb);
                const logpB = new Float32Array(mb);
                const advB = new Float32Array(mb);
                const retB = new Float32Array(mb);
                for (let k = 0; k < mb; k++) {
                    const idx = order[start + k];
                    obsB.set(rollout.obs.subarray(idx * OBS_DIM, (idx + 1) * OBS_DIM), k * OBS_DIM);
                    actB[k] = rollout.action[idx];
                    logpB[k] = rollout.logp[idx];
                    advB[k] = adv[idx];
                    retB[k] = ret[idx];
                }
                const batch: LossBatch = {
                    obs: np.array(obsB).reshape([mb, OBS_DIM]),
                    action: np.array(actB, { dtype: np.int32 }),
                    oldLogp: np.array(logpB),
                    adv: np.array(advB),
                    ret: np.array(retB),
                };
                const [[totalLoss, aux], grads] = this.lossGrad(refTree(this.params), batch) as unknown as [
                    [Arr, Record<string, Arr>],
                    Params,
                ];
                totalLoss.dispose();
                const [updatesTree, newState] = this.opt.update(grads, this.optState);
                this.optState = newState;
                this.params = optax.applyUpdates(this.params, updatesTree);
                this.stepCount++;

                for (const key of Object.keys(sums) as (keyof typeof sums)[]) {
                    sums[key] += (await aux[key].data())[0];
                }
                updates++;
            }
        }
        const t2 = performance.now();

        this.iteration++;
        this.lastLr = this.lrAt(this.stepCount);
        const agentSteps = this.cfg.envs * NUM_SATS * EPISODE_STEPS;
        return {
            iteration: this.iteration,
            episodeReward: rollout.episodeReward,
            delivered: rollout.delivered,
            brownouts: rollout.brownouts,
            policyLoss: sums.policyLoss / updates,
            valueLoss: sums.valueLoss / updates,
            entropy: sums.entropy / updates,
            approxKl: sums.approxKl / updates,
            clipFraction: sums.clipFraction / updates,
            explainedVariance: sums.explainedVariance / updates,
            learningRate: this.lastLr,
            agentStepsPerSecond: agentSteps / ((t2 - t0) / 1000),
            rolloutSeconds: (t1 - t0) / 1000,
            updateSeconds: (t2 - t1) / 1000,
        };
    }
}
