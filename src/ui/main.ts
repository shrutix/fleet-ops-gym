import "@fontsource-variable/archivo/wdth.css";
import "./style.css";

import { FRAME_STRIDE } from "../rl/evaluate.ts";
import type { EvalResult, Recording } from "../rl/evaluate.ts";
import type { IterationMetrics } from "../rl/ppo.ts";
import { NUM_SATS } from "../sim/constants.ts";
import { LineChart } from "./chart.ts";
import type { Series } from "./chart.ts";
import { STATUS_LABEL, drawMap, readTheme, satName } from "./map.ts";
import type { Baselines, DeviceChoice, FromWorker, PolicyName, ToWorker } from "./protocol.ts";

// ---- helpers -----------------------------------------------------------------------------

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const STEPS = 360;

const POLICY_NAMES: Record<PolicyName, string> = {
    learned: "Learned policy",
    tuned: "Tuned greedy",
    greedy: "Greedy",
};
const POLICY_COLOR: Record<PolicyName, string> = {
    learned: "var(--learned)",
    tuned: "var(--tuned)",
    greedy: "var(--greedy)",
};

function niceTicks(max: number, count = 4): number[] {
    const raw = max / count;
    const mag = 10 ** Math.floor(Math.log10(raw));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? raw;
    // Always end on a tick at or above the data, so nothing is drawn outside the plot.
    const ticks: number[] = [];
    for (let v = 0; ; v += step) {
        ticks.push(v);
        if (v >= max - 1e-9) break;
    }
    return ticks;
}

const fmt1 = (n: number) => n.toFixed(1);
const fmt0 = (n: number) => Math.round(n).toString();

// A worker resolves relative URLs against its own script (/assets/...), not the page, so the
// pretrained file's address must be made absolute here.
const PRETRAINED_URL = new URL(`${import.meta.env.BASE_URL}pretrained.json`, document.baseURI).href;

// ---- state -------------------------------------------------------------------------------

interface AppState {
    worker: Worker | null;
    ready: boolean;
    state: "idle" | "training" | "paused" | "done";
    totalIterations: number;
    baselines: Baselines | null;
    metrics: IterationMetrics[];
    evals: { iteration: number; result: EvalResult }[];
    pretrained: { final?: EvalResult; iterations?: number } | null;
    recs: Record<PolicyName, Recording> | null;
    recordPending: boolean;
    recordAgain: boolean;
    seed: number;
    policy: PolicyName;
    frame: number;
    playing: boolean;
}

const S: AppState = {
    worker: null,
    ready: false,
    state: "idle",
    totalIterations: 120,
    baselines: null,
    metrics: [],
    evals: [],
    pretrained: null,
    recs: null,
    recordPending: false,
    recordAgain: false,
    seed: 4242,
    policy: "learned",
    frame: 0,
    playing: false,
};

// ---- worker plumbing ---------------------------------------------------------------------

function send(msg: ToWorker) {
    S.worker?.postMessage(msg);
}

function showError(message: string | null) {
    const box = $("alert");
    box.hidden = message == null;
    box.textContent = message ?? "";
}

function startWorker(device: DeviceChoice) {
    S.worker?.terminate();
    Object.assign(S, {
        ready: false,
        state: "idle",
        metrics: [],
        evals: [],
        pretrained: null,
        recs: null,
        recordPending: false,
        recordAgain: false,
        playing: false,
        frame: 0,
    });
    showError(null);
    $("device-note").textContent = "Starting the compute backend";
    updateControls();
    renderAll();

    const worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
    worker.onmessage = (e: MessageEvent<FromWorker>) => onWorkerMessage(e.data);
    worker.onerror = (e) => showError(`Worker failed: ${e.message}`);
    S.worker = worker;
    send({ type: "init", device });
}

function requestRecord() {
    if (!S.ready) return;
    if (S.recordPending) {
        S.recordAgain = true;
        return;
    }
    S.recordPending = true;
    $("map-status").textContent = "Replaying the episode";
    send({ type: "record", seed: S.seed });
}

function onWorkerMessage(msg: FromWorker) {
    switch (msg.type) {
        case "ready": {
            S.ready = true;
            S.baselines = msg.baselines;
            S.totalIterations = msg.totalIterations;
            const label = msg.device === "webgpu" ? "WebGPU" : "Wasm on the CPU";
            $("device-note").textContent = msg.note ?? `Training runs on ${label}.`;
            $("progress").setAttribute("aria-valuemax", String(msg.totalIterations));
            updateControls();
            send({ type: "loadPretrained", url: PRETRAINED_URL });
            break;
        }
        case "pretrained":
            if (msg.ok) {
                S.pretrained = { final: msg.final, iterations: msg.iterations };
                setSource(`Training continues from the pretrained policy (${msg.iterations ?? "?"} iterations). Choose Start over to train from random weights.`);
            } else {
                setSource("No pretrained policy is bundled with this build. Training starts from random weights.");
            }
            requestRecord();
            renderAll();
            break;
        case "metrics":
            S.metrics.push(msg.metrics);
            renderTraining();
            break;
        case "eval":
            S.evals.push({ iteration: msg.iteration, result: msg.result });
            renderAll();
            requestRecord();
            break;
        case "state":
            S.state = msg.state;
            updateControls();
            break;
        case "recordings":
            S.recordPending = false;
            if (msg.seed === S.seed) {
                S.recs = msg.recs;
                S.frame = Math.min(S.frame, STEPS);
                renderAll();
            }
            if (S.recordAgain) {
                S.recordAgain = false;
                requestRecord();
            } else if (msg.seed !== S.seed) {
                requestRecord();
            }
            break;
        case "error":
            S.recordPending = false;
            showError(msg.message);
            break;
    }
}

// ---- controls ----------------------------------------------------------------------------

function setSource(text: string) {
    $("policy-source").textContent = text;
}

function updateControls() {
    const trainBtn = $<HTMLButtonElement>("train");
    trainBtn.disabled = !S.ready || S.state === "done";
    trainBtn.textContent =
        S.state === "training" ? "Pause" : S.state === "paused" ? "Resume" : S.state === "done" ? "Finished" : "Train";
    $<HTMLButtonElement>("pretrained").disabled = !S.ready || S.state === "training";
    $<HTMLButtonElement>("reset").disabled = !S.ready || S.state === "training";
    const hasRecs = S.recs != null;
    $<HTMLButtonElement>("play").disabled = !hasRecs;
    $<HTMLInputElement>("scrub").disabled = !hasRecs;
    $<HTMLButtonElement>("play").textContent = S.playing ? "Pause" : "Play";
}

$("train").addEventListener("click", () => {
    send({ type: S.state === "training" ? "pause" : "train" });
});
$("pretrained").addEventListener("click", () => {
    S.metrics = [];
    S.evals = [];
    send({ type: "loadPretrained", url: PRETRAINED_URL });
});
$("reset").addEventListener("click", () => {
    setSource("Training starts from random weights.");
    S.metrics = [];
    S.evals = [];
    S.pretrained = null;
    send({ type: "resetPolicy" });
    requestRecord();
    renderAll();
});
$("device").addEventListener("change", (e) => startWorker((e.target as HTMLSelectElement).value as DeviceChoice));

$("policy-tabs").addEventListener("click", (e) => {
    const btn = (e.target as HTMLElement).closest<HTMLButtonElement>("button[data-policy]");
    if (!btn) return;
    S.policy = btn.dataset.policy as PolicyName;
    for (const b of document.querySelectorAll<HTMLButtonElement>("#policy-tabs button")) {
        b.setAttribute("aria-checked", String(b === btn));
    }
    renderAll();
});

function setSeed(seed: number) {
    if (!Number.isFinite(seed) || seed < 0) return;
    S.seed = Math.floor(seed);
    $<HTMLInputElement>("seed").value = String(S.seed);
    S.frame = 0;
    S.playing = false;
    requestRecord();
    updateControls();
}
$("seed").addEventListener("change", (e) => setSeed(Number((e.target as HTMLInputElement).value)));
$("new-seed").addEventListener("click", () => setSeed(Math.floor(Math.random() * 100000)));

$("play").addEventListener("click", () => {
    if (!S.recs) return;
    if (!S.playing && S.frame >= STEPS) S.frame = 0;
    S.playing = !S.playing;
    last = performance.now();
    updateControls();
    if (S.playing) requestAnimationFrame(tick);
});
$("scrub").addEventListener("input", (e) => {
    S.frame = Number((e.target as HTMLInputElement).value);
    renderFrame();
});

// ---- playback ----------------------------------------------------------------------------

let last = 0;
let carry = 0;
function tick(now: number) {
    if (!S.playing) return;
    const speed = Number($<HTMLSelectElement>("speed").value);
    carry += ((now - last) / 1000) * 6 * speed; // 6 simulated minutes per second at 1x
    last = now;
    const whole = Math.floor(carry);
    if (whole > 0) {
        carry -= whole;
        S.frame = Math.min(STEPS, S.frame + whole);
        if (S.frame >= STEPS) {
            S.playing = false;
            updateControls();
        }
        renderFrame();
    }
    if (S.playing) requestAnimationFrame(tick);
}

// ---- rendering: map, fleet strip, replay figures -----------------------------------------

const canvas = $<HTMLCanvasElement>("map");
const ctx2d = canvas.getContext("2d")!;
let theme = readTheme();

function sizeCanvas() {
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
        canvas.width = Math.round(w * dpr);
        canvas.height = Math.round(h * dpr);
    }
    ctx2d.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { w, h };
}

const fleetEl = $("fleet");
for (let i = 0; i < NUM_SATS; i++) {
    const sat = document.createElement("div");
    sat.className = "sat s-0";
    sat.innerHTML = `
    <div class="sat-head"><span>${satName(i)}</span><span class="sat-glyph" aria-hidden="true">○</span></div>
    <div class="sat-label"><span>Battery</span><span class="bv">–</span></div>
    <div class="meter bat"><i></i></div>
    <div class="sat-label"><span>Data</span><span class="dv">–</span></div>
    <div class="meter data"><i></i></div>`;
    fleetEl.append(sat);
}
const GLYPH = ["○", "●", "◆", "◇", "■"];

function renderFrame() {
    const rec = S.recs?.[S.policy];
    $<HTMLInputElement>("scrub").value = String(S.frame);
    $("minute").textContent = `Minute ${S.frame} of ${STEPS}`;
    if (!rec) return;
    const { w, h } = sizeCanvas();
    drawMap(ctx2d, w, h, rec, S.frame, theme);

    const counts = [0, 0, 0, 0, 0];
    for (let i = 0; i < NUM_SATS; i++) {
        const o = (S.frame * NUM_SATS + i) * FRAME_STRIDE;
        const battery = rec.frames[o + 2];
        const storage = rec.frames[o + 3];
        const status = S.frame < STEPS ? rec.frames[o + 4] : 0;
        counts[status]++;
        const card = fleetEl.children[i] as HTMLElement;
        card.className = `sat s-${status}`;
        (card.querySelector(".sat-glyph") as HTMLElement).textContent = GLYPH[status];
        card.title = `${satName(i)}: ${STATUS_LABEL[status]}`;
        (card.querySelector(".bv") as HTMLElement).textContent = `${Math.round(battery * 100)}%`;
        (card.querySelector(".dv") as HTMLElement).textContent = `${Math.round(storage * 100)}%`;
        const bat = card.querySelector(".bat") as HTMLElement;
        bat.classList.toggle("low", battery < 0.15);
        (bat.firstElementChild as HTMLElement).style.width = `${battery * 100}%`;
        ((card.querySelector(".data") as HTMLElement).firstElementChild as HTMLElement).style.width = `${storage * 100}%`;
    }
    const delivered = rec.cumDelivered[S.frame];
    $("map-status").textContent =
        `${POLICY_NAMES[S.policy]}: ${counts[1]} imaging, ${counts[2]} downlinking, ${delivered.toFixed(1)} buffers delivered`;

    // Move the race playhead without rebuilding the data.
    race.update(raceSeries(), raceOptions());
}

// ---- rendering: replay figures -----------------------------------------------------------

const race = new LineChart($("race"), 230, 104);

function raceSeries(): Series[] {
    if (!S.recs) return [];
    return (["learned", "tuned", "greedy"] as PolicyName[]).map((p) => {
        const rec = S.recs![p];
        return {
            id: p,
            label: POLICY_NAMES[p],
            xs: Array.from({ length: STEPS + 1 }, (_, i) => i),
            ys: Array.from(rec.cumDelivered),
            color: POLICY_COLOR[p],
            width: p === S.policy ? 2.8 : 1.8,
            opacity: p === S.policy ? 1 : 0.85,
        };
    });
}

function raceOptions() {
    const max = S.recs ? Math.max(...Object.values(S.recs).map((r) => r.totalDelivered)) * 1.05 : 25;
    const yTicks = niceTicks(Math.max(max, 5), 4);
    return {
        xDomain: [0, STEPS] as [number, number],
        yDomain: [0, yTicks[yTicks.length - 1]] as [number, number],
        xTicks: [0, 90, 180, 270, 360],
        yTicks,
        xLabel: "Minute",
        yLabel: "Buffers delivered",
        formatX: (x: number) => String(x),
        formatY: (y: number) => fmt1(y),
        playhead: S.frame,
        onScrub: (x: number) => {
            S.frame = Math.round(x);
            S.playing = false;
            updateControls();
            renderFrame();
        },
        summary: "Cumulative data delivered over the episode for the learned policy, tuned greedy and greedy.",
    };
}

function renderSeedTable() {
    const body = $("seed-table").querySelector("tbody")!;
    body.innerHTML = "";
    if (!S.recs) return;
    const best = Math.max(...Object.values(S.recs).map((r) => r.totalDelivered));
    for (const p of ["learned", "tuned", "greedy"] as PolicyName[]) {
        const r = S.recs[p];
        const tr = document.createElement("tr");
        if (r.totalDelivered === best) tr.className = "lead";
        tr.innerHTML = `<th scope="row"><span class="swatch" style="background:${POLICY_COLOR[p]}"></span>${POLICY_NAMES[p]}</th>
      <td class="num">${fmt1(r.totalDelivered)}</td><td class="num">${fmt0(r.totalReward)}</td>`;
        body.append(tr);
    }
}

function renderHeldoutTable() {
    const body = $("heldout-table").querySelector("tbody")!;
    body.innerHTML = "";
    if (!S.baselines) return;
    const b = S.baselines;
    const latest = S.evals.length ? S.evals[S.evals.length - 1].result : (S.pretrained?.final ?? null);
    const rows: { name: string; color: string; r: EvalResult | null }[] = [
        { name: POLICY_NAMES.learned, color: POLICY_COLOR.learned, r: latest },
        { name: POLICY_NAMES.tuned, color: POLICY_COLOR.tuned, r: b.tuned },
        { name: POLICY_NAMES.greedy, color: POLICY_COLOR.greedy, r: b.greedy },
        { name: "Random", color: "var(--random)", r: b.random },
    ];
    for (const row of rows) {
        const tr = document.createElement("tr");
        let vs = "–";
        if (row.r && row.name !== POLICY_NAMES.tuned) {
            const d = ((row.r.reward - b.tuned.reward) / b.tuned.reward) * 100;
            const rounded = Math.round(Math.abs(d));
            vs =
                rounded === 0
                    ? "0%"
                    : `<span class="${d >= 0 ? "up" : "down"}">${d >= 0 ? "+" : "−"}${rounded}%</span>`;
        }
        tr.innerHTML = `<th scope="row"><span class="swatch" style="background:${row.color}"></span>${row.name}</th>
      <td class="num">${row.r ? fmt1(row.r.delivered) : "–"}</td>
      <td class="num">${row.r ? fmt0(row.r.reward) : "–"}</td><td class="num">${vs}</td>`;
        body.append(tr);
    }
}

// ---- rendering: training -----------------------------------------------------------------

const curve = new LineChart($("curve"), 280);

function renderTraining() {
    const m = S.metrics[S.metrics.length - 1];
    const done = m?.iteration ?? 0;
    $("v-iter").textContent = `${done} of ${S.totalIterations}`;
    $("v-speed").textContent = m ? `${(m.agentStepsPerSecond / 1000).toFixed(0)}k steps/s` : "–";
    $("v-entropy").textContent = m ? m.entropy.toFixed(2) : "–";
    $("v-kl").textContent = m ? m.approxKl.toFixed(4) : "–";
    ($("progress").firstElementChild as HTMLElement).style.width = `${(done / S.totalIterations) * 100}%`;
    $("progress").setAttribute("aria-valuenow", String(done));

    const b = S.baselines;
    const total = S.totalIterations;
    const ref = (id: string, label: string, y: number, color: string, dash: string, w: number): Series => ({
        id,
        label: `${label} ${fmt0(y)}`,
        xs: [0, total],
        ys: [y, y],
        color,
        dash,
        width: w,
        noTooltip: true,
    });
    const series: Series[] = [];
    if (b) {
        series.push(ref("tuned", "Tuned greedy", b.tuned.reward, "var(--tuned)", "", 1.8));
        series.push(ref("greedy", "Greedy", b.greedy.reward, "var(--greedy)", "7 4", 1.8));
        series.push(ref("random", "Random", b.random.reward, "var(--random)", "2 4", 1.8));
    }
    if (S.pretrained?.final) {
        series.push(
            ref("pretrained", "Pretrained", S.pretrained.final.reward, "var(--learned)", "5 4", 1.6),
        );
    }
    series.push({
        id: "train",
        label: "Training",
        xs: S.metrics.map((x) => x.iteration),
        ys: S.metrics.map((x) => x.episodeReward),
        color: "var(--learned)",
        width: 1.6,
        opacity: 0.4,
    });
    series.push({
        id: "eval",
        label: "Held-out",
        xs: S.evals.map((x) => x.iteration),
        ys: S.evals.map((x) => x.result.reward),
        color: "var(--learned)",
        width: 2.8,
        markers: true,
    });

    const maxY = Math.max(
        260,
        ...(b ? [b.tuned.reward * 1.1] : []),
        ...S.metrics.map((x) => x.episodeReward),
        ...S.evals.map((x) => x.result.reward),
    );
    const yTicks = niceTicks(maxY, 5);
    curve.update(series, {
        xDomain: [0, total],
        yDomain: [0, yTicks[yTicks.length - 1]],
        xTicks: [0, 20, 40, 60, 80, 100, 120].filter((t) => t <= total),
        yTicks,
        xLabel: "Iteration",
        yLabel: "Episode reward",
        formatX: (x) => String(x),
        formatY: (y) => fmt0(y),
        summary: "Episode reward per training iteration for the learned policy, with greedy baselines for comparison.",
    });
}

function renderAll() {
    renderSeedTable();
    renderHeldoutTable();
    renderTraining();
    renderFrame();
    updateControls();
}

// ---- boot --------------------------------------------------------------------------------

window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
    theme = readTheme();
    renderAll();
});
new ResizeObserver(() => renderFrame()).observe(canvas);

renderAll();
startWorker("auto");
