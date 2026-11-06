// Messages between the page and the training worker.

import type { EvalResult, Recording } from "../rl/evaluate.ts";
import type { IterationMetrics } from "../rl/ppo.ts";

export type PolicyName = "learned" | "tuned" | "greedy";
export type DeviceChoice = "auto" | "wasm" | "webgpu";

export interface Baselines {
    random: EvalResult;
    greedy: EvalResult;
    tuned: EvalResult;
}

export type ToWorker =
    | { type: "init"; device: DeviceChoice }
    | { type: "train" }
    | { type: "pause" }
    | { type: "loadPretrained"; url: string }
    | { type: "resetPolicy" }
    | { type: "record"; seed: number };

export type FromWorker =
    | {
        type: "ready";
        devices: string[];
        device: string;
        note?: string;
        baselines: Baselines;
        totalIterations: number;
    }
    | { type: "metrics"; metrics: IterationMetrics }
    | { type: "eval"; iteration: number; result: EvalResult }
    | { type: "state"; state: "idle" | "training" | "paused" | "done" }
    | { type: "pretrained"; ok: boolean; message?: string; final?: EvalResult; iterations?: number }
    | { type: "recordings"; seed: number; recs: Record<PolicyName, Recording> }
    | { type: "error"; message: string };
