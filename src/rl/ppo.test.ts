import assert from "node:assert/strict";
import { test } from "node:test";

import "../jax/polyfill.ts";
import { defaultDevice, init } from "@jax-js/jax";

import { evaluateBatch } from "./evaluate.ts";
import { deserializeParams, serializeParams } from "./network.ts";
import { PPOTrainer } from "./ppo.ts";

test("one PPO iteration runs, stays finite and changes the policy", async () => {
    await init();
    defaultDevice("wasm");
    const trainer = new PPOTrainer({ seed: 3, envs: 2, minibatch: 1024, epochs: 2, totalIterations: 2 });

    const before = await serializeParams(trainer.params);
    const m = await trainer.iterate();

    for (const [name, value] of Object.entries(m)) {
        assert.ok(Number.isFinite(value), `${name} is not finite: ${value}`);
    }
    // A fresh policy is close to uniform over 3 actions: entropy near ln(3) = 1.0986.
    assert.ok(m.entropy > 1.0 && m.entropy < 1.1, `initial entropy ${m.entropy}`);
    assert.ok(m.agentStepsPerSecond > 0);

    const after = await serializeParams(trainer.params);
    const changed = JSON.stringify(before.w1) !== JSON.stringify(after.w1);
    assert.ok(changed, "parameters did not change after an update");

    // Weights survive a save/load round trip and still produce a working policy.
    const reloaded = deserializeParams(JSON.parse(JSON.stringify(after)));
    trainer.setParams(reloaded);
    const result = await evaluateBatch(trainer.greedyPolicy(), [9000, 9001]);
    assert.ok(Number.isFinite(result.reward));
    assert.equal(result.episodes, 2);
    trainer.dispose();
});
