// Command-line training: npm run train -- --iters 100 --envs 16 --out public/pretrained.json
//
// Trains on the Wasm backend in Node, prints one line per iteration, evaluates the greedy
// (argmax) policy on held-out seeds, and compares it with the hand-written baselines.

import "../jax/polyfill.ts";
import { writeFileSync } from "node:fs";

import { defaultDevice, init } from "@jax-js/jax";

import { EVAL_SEEDS, TUNED_GREEDY } from "./eval-config.ts";
import { evaluateBatch, evaluateSync, pairedDifference } from "./evaluate.ts";
import { serializeParams } from "./network.ts";
import { PPOTrainer } from "./ppo.ts";
import { makeRandomPolicy, makeGreedyPolicy, DEFAULT_GREEDY } from "../sim/policies.ts";
import { makeRng } from "../sim/env.ts";

function arg(name: string, fallback: number): number {
    const i = process.argv.indexOf(`--${name}`);
    return i >= 0 ? Number(process.argv[i + 1]) : fallback;
}
function argStr(name: string, fallback: string): string {
    const i = process.argv.indexOf(`--${name}`);
    return i >= 0 ? process.argv[i + 1] : fallback;
}


async function main() {
    const iters = arg("iters", 100);
    const envs = arg("envs", 16);
    const seed = arg("seed", 0);
    const evalEvery = arg("eval-every", 10);
    const out = argStr("out", "");

    const devices = await init();
    defaultDevice("wasm");
    console.log(`devices: ${devices.join(", ")} (training on wasm)`);

    const random = evaluateSync(makeRandomPolicy(makeRng(1)), EVAL_SEEDS);
    const greedy = evaluateSync(makeGreedyPolicy(DEFAULT_GREEDY), EVAL_SEEDS);
    const tuned = evaluateSync(makeGreedyPolicy(TUNED_GREEDY), EVAL_SEEDS);
    const fmt = (r: { reward: number; delivered: number; rewardStd: number }) =>
        `reward ${r.reward.toFixed(1)} (std ${r.rewardStd.toFixed(1)}), delivered ${r.delivered.toFixed(2)}`;
    console.log(`baselines on ${EVAL_SEEDS.length} held-out seeds:`);
    console.log(`  random : ${fmt(random)}`);
    console.log(`  greedy : ${fmt(greedy)}`);
    console.log(`  tuned  : ${fmt(tuned)}`);

    const trainer = new PPOTrainer({ seed, envs, totalIterations: iters });
    let best = { reward: -Infinity, iteration: 0 };
    let last = null as Awaited<ReturnType<typeof evaluateBatch>> | null;

    // Checkpoint at every evaluation so an interrupted run still leaves usable weights.
    const checkpoint = async (iteration: number, result: NonNullable<typeof last>) => {
        if (!out) return;
        const payload = {
            meta: {
                iterations: iteration,
                envs,
                seed,
                evalSeeds: `${EVAL_SEEDS[0]}..${EVAL_SEEDS[EVAL_SEEDS.length - 1]}`,
                final: result,
                baselines: { random, greedy, tuned },
                tunedGreedyParams: TUNED_GREEDY,
            },
            params: await serializeParams(trainer.params),
        };
        writeFileSync(out, JSON.stringify(payload));
    };

    for (let it = 1; it <= iters; it++) {
        const m = await trainer.iterate();
        console.log(
            `iter ${String(it).padStart(3)}  reward ${m.episodeReward.toFixed(1).padStart(6)}  ` +
            `delivered ${m.delivered.toFixed(2).padStart(5)}  brownouts ${m.brownouts.toFixed(1).padStart(5)}  ` +
            `H ${m.entropy.toFixed(3)}  kl ${m.approxKl.toFixed(4)}  clip ${m.clipFraction.toFixed(2)}  ` +
            `vloss ${m.valueLoss.toFixed(3)}  ev ${m.explainedVariance.toFixed(2)}  ` +
            `${(m.agentStepsPerSecond / 1000).toFixed(0)}k steps/s (rollout ${m.rolloutSeconds.toFixed(1)}s, update ${m.updateSeconds.toFixed(1)}s)`,
        );
        if (it % evalEvery === 0 || it === iters) {
            last = await evaluateBatch(trainer.greedyPolicy(), EVAL_SEEDS);
            const diff = pairedDifference(last, tuned);
            console.log(
                `  eval (argmax policy, held-out): ${fmt(last)}   ` +
                `vs tuned greedy ${diff.mean >= 0 ? "+" : ""}${diff.mean.toFixed(1)} ± ${diff.stdError.toFixed(1)} (paired, 1 s.e.)`,
            );
            if (last.reward > best.reward) best = { reward: last.reward, iteration: it };
            await checkpoint(it, last);
        }
    }

    if (out && last) console.log(`saved ${out}`);
    console.log(`best eval reward ${best.reward.toFixed(1)} at iteration ${best.iteration}`);
}

main();
