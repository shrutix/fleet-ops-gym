// Training worker: owns the jax-js runtime and the PPO trainer so the page stays responsive.

import { defaultDevice, init } from "@jax-js/jax";

import { EVAL_SEEDS, TUNED_GREEDY } from "../rl/eval-config.ts";
import { evaluateBatch, evaluateSync, recordEpisode, syncToBatch } from "../rl/evaluate.ts";
import { deserializeParams, initParams } from "../rl/network.ts";
import { PPOTrainer } from "../rl/ppo.ts";
import { makeRng } from "../sim/env.ts";
import { DEFAULT_GREEDY, makeGreedyPolicy, makeRandomPolicy } from "../sim/policies.ts";
import type { Baselines, FromWorker, PolicyName, ToWorker } from "./protocol.ts";

// Browsers older than Chrome 135 / Firefox 129 lack Float16Array, which jax-js probes.
const g = globalThis as { Float16Array?: unknown };
if (typeof g.Float16Array === "undefined") g.Float16Array = class Float16Array { };

const scope = globalThis as unknown as {
    postMessage(message: FromWorker, transfer?: Transferable[]): void;
    onmessage: ((e: MessageEvent<ToWorker>) => void) | null;
};
const post = (message: FromWorker, transfer: Transferable[] = []) =>
    scope.postMessage(message, transfer);

const TOTAL_ITERATIONS = 120;
const EVAL_EVERY = 10;

let trainer: PPOTrainer | null = null;
let running = false;

function requireTrainer(): PPOTrainer {
    if (!trainer) throw new Error("Worker is not initialized");
    return trainer;
}

async function handleInit(choice: "auto" | "wasm" | "webgpu") {
    const devices = await init();
    let device = "wasm";
    let note: string | undefined;
    if (choice === "webgpu" || choice === "auto") {
        if (devices.includes("webgpu")) device = "webgpu";
        else if (choice === "webgpu") note = "WebGPU is not available in this browser, so training runs on Wasm.";
    }
    defaultDevice(device as "wasm" | "webgpu");

    trainer = new PPOTrainer({ seed: 0, envs: 16, totalIterations: TOTAL_ITERATIONS });
    const baselines: Baselines = {
        random: evaluateSync(makeRandomPolicy(makeRng(1)), EVAL_SEEDS),
        greedy: evaluateSync(makeGreedyPolicy(DEFAULT_GREEDY), EVAL_SEEDS),
        tuned: evaluateSync(makeGreedyPolicy(TUNED_GREEDY), EVAL_SEEDS),
    };
    post({ type: "ready", devices, device, note, baselines, totalIterations: TOTAL_ITERATIONS });
    post({ type: "state", state: "idle" });
}

async function trainLoop() {
    const t = requireTrainer();
    if (running) return;
    running = true;
    post({ type: "state", state: "training" });
    while (running && t.iteration < t.cfg.totalIterations) {
        const metrics = await t.iterate();
        post({ type: "metrics", metrics });
        const last = t.iteration >= t.cfg.totalIterations;
        if (t.iteration % EVAL_EVERY === 0 || last) {
            const result = await evaluateBatch(t.greedyPolicy(), EVAL_SEEDS);
            post({ type: "eval", iteration: t.iteration, result });
        }
    }
    const finished = t.iteration >= t.cfg.totalIterations;
    running = false;
    post({ type: "state", state: finished ? "done" : "paused" });
}

async function handleRecord(seed: number) {
    const t = requireTrainer();
    const recs: Record<PolicyName, Awaited<ReturnType<typeof recordEpisode>>> = {
        learned: await recordEpisode(seed, t.greedyPolicy()),
        tuned: await recordEpisode(seed, syncToBatch(makeGreedyPolicy(TUNED_GREEDY))),
        greedy: await recordEpisode(seed, syncToBatch(makeGreedyPolicy(DEFAULT_GREEDY))),
    };
    const transfer: Transferable[] = [];
    for (const r of Object.values(recs)) {
        transfer.push(r.frames.buffer, r.theta.buffer, r.cumReward.buffer, r.cumDelivered.buffer);
    }
    post({ type: "recordings", seed, recs }, transfer);
}

async function handlePretrained(url: string) {
    const t = requireTrainer();
    const missing = () =>
        post({
            type: "pretrained",
            ok: false,
            message: "No pretrained policy is bundled with this build, so the replay uses an untrained policy.",
        });
    const response = await fetch(url);
    // Static servers often answer a missing file with the HTML page and a 200, so check the type.
    if (!response.ok || !(response.headers.get("content-type") ?? "").includes("json")) {
        missing();
        return;
    }
    let payload;
    try {
        payload = await response.json();
    } catch {
        missing();
        return;
    }
    t.setParams(deserializeParams(payload.params));
    post({
        type: "pretrained",
        ok: true,
        final: payload.meta?.final,
        iterations: payload.meta?.iterations,
    });
}

scope.onmessage = async (event) => {
    const msg = event.data;
    try {
        switch (msg.type) {
            case "init":
                await handleInit(msg.device);
                break;
            case "train":
                void trainLoop().catch((e) => {
                    running = false;
                    post({ type: "error", message: String(e?.stack ?? e) });
                });
                break;
            case "pause":
                running = false;
                break;
            case "loadPretrained":
                await handlePretrained(msg.url);
                break;
            case "resetPolicy": {
                const t = requireTrainer();
                t.setParams(initParams(Math.floor(Math.random() * 1e6), t.cfg.hidden));
                post({ type: "state", state: "idle" });
                break;
            }
            case "record":
                await handleRecord(msg.seed);
                break;
        }
    } catch (e) {
        running = false;
        post({ type: "error", message: String((e as Error)?.stack ?? e) });
    }
};
