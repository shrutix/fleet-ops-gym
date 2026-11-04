// Quick baseline report: node src/sim/run.ts
import { FleetEnv, makeRng } from "./env.ts";
import { greedyPolicy, makeRandomPolicy, runEpisode } from "./policies.ts";

const EPISODES = 16;

function report(name: string, makePolicy: (seed: number) => ReturnType<typeof makeRandomPolicy>) {
    let delivered = 0;
    let reward = 0;
    let brownouts = 0;
    for (let seed = 0; seed < EPISODES; seed++) {
        const s = runEpisode(new FleetEnv(seed), makePolicy(seed));
        delivered += s.delivered;
        reward += s.totalReward;
        brownouts += s.brownouts;
    }
    console.log(
        `${name.padEnd(8)} delivered=${(delivered / EPISODES).toFixed(2)}  ` +
        `reward=${(reward / EPISODES).toFixed(1)}  brownout-steps=${(brownouts / EPISODES).toFixed(1)}`,
    );
}

report("random", (seed) => makeRandomPolicy(makeRng(seed + 1000)));
report("greedy", () => greedyPolicy);
