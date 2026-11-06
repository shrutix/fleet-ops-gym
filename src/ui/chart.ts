// A small SVG line chart with direct labels, a hover crosshair with tooltip, and an
// optional scrub handle. Hand-rolled so it ships no dependency.

export interface Series {
    id: string;
    label: string;
    xs: number[];
    ys: number[];
    /** CSS color or var(). */
    color: string;
    width?: number;
    dash?: string;
    opacity?: number;
    /** Draw a dot at every point (for sparse series like periodic evaluations). */
    markers?: boolean;
    /** Skip this series in the tooltip (for reference lines). */
    noTooltip?: boolean;
}

export interface ChartOptions {
    xDomain: [number, number];
    yDomain: [number, number];
    xTicks: number[];
    yTicks: number[];
    xLabel: string;
    yLabel: string;
    formatX: (x: number) => string;
    formatY: (y: number) => string;
    /** Vertical marker (the replay playhead). */
    playhead?: number | null;
    /** Called while the pointer is pressed on the chart, with the data-space x. */
    onScrub?: (x: number) => void;
    summary: string;
}

const NS = "http://www.w3.org/2000/svg";

function el<K extends keyof SVGElementTagNameMap>(
    name: K,
    attrs: Record<string, string | number>,
    text?: string,
): SVGElementTagNameMap[K] {
    const node = document.createElementNS(NS, name);
    for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, String(v));
    if (text !== undefined) node.textContent = text;
    return node;
}

function nearestIndex(xs: number[], x: number): number {
    let best = 0;
    for (let i = 1; i < xs.length; i++) if (Math.abs(xs[i] - x) < Math.abs(xs[best] - x)) best = i;
    return best;
}

export class LineChart {
    private series: Series[] = [];
    private opts!: ChartOptions;
    private hoverX: number | null = null;
    private pressed = false;
    private readonly tip: HTMLDivElement;

    private readonly m: { top: number; right: number; bottom: number; left: number };

    constructor(
        private readonly host: HTMLElement,
        private readonly height: number,
        right = 128,
    ) {
        this.m = { top: 26, right, bottom: 34, left: 46 };
        this.host.classList.add("chart");
        this.tip = document.createElement("div");
        this.tip.className = "tip";
        this.tip.hidden = true;
        this.host.append(this.tip);
        this.host.addEventListener("pointermove", (e) => this.onMove(e));
        this.host.addEventListener("pointerdown", (e) => {
            this.pressed = true;
            this.host.setPointerCapture(e.pointerId);
            this.onMove(e);
        });
        this.host.addEventListener("pointerup", () => (this.pressed = false));
        this.host.addEventListener("pointerleave", () => {
            if (this.pressed) return;
            this.hoverX = null;
            this.render();
        });
        new ResizeObserver(() => this.render()).observe(this.host);
    }

    update(series: Series[], opts: ChartOptions) {
        this.series = series;
        this.opts = opts;
        this.render();
    }

    private geometry() {
        const width = Math.max(240, this.host.clientWidth);
        const o = this.opts;
        const iw = width - this.m.left - this.m.right;
        const ih = this.height - this.m.top - this.m.bottom;
        const x = (v: number) => this.m.left + ((v - o.xDomain[0]) / (o.xDomain[1] - o.xDomain[0])) * iw;
        const y = (v: number) => this.m.top + (1 - (v - o.yDomain[0]) / (o.yDomain[1] - o.yDomain[0])) * ih;
        const invX = (px: number) => o.xDomain[0] + ((px - this.m.left) / iw) * (o.xDomain[1] - o.xDomain[0]);
        return { width, iw, ih, x, y, invX };
    }

    private onMove(e: PointerEvent) {
        if (!this.opts) return;
        const g = this.geometry();
        const rect = this.host.getBoundingClientRect();
        const dataX = g.invX(e.clientX - rect.left);
        const [lo, hi] = this.opts.xDomain;
        this.hoverX = Math.min(hi, Math.max(lo, dataX));
        if (this.pressed && this.opts.onScrub) this.opts.onScrub(this.hoverX);
        this.render();
    }

    render() {
        if (!this.opts) return;
        const o = this.opts;
        const g = this.geometry();
        const svg = el("svg", {
            viewBox: `0 0 ${g.width} ${this.height}`,
            width: g.width,
            height: this.height,
            role: "img",
            "aria-label": o.summary,
        });

        // Grid and axes stay quiet so the data carries the contrast.
        for (const t of o.yTicks) {
            svg.append(el("line", { x1: this.m.left, x2: g.width - this.m.right, y1: g.y(t), y2: g.y(t), class: "grid" }));
            svg.append(el("text", { x: this.m.left - 8, y: g.y(t) + 4, class: "tick", "text-anchor": "end" }, o.formatY(t)));
        }
        for (const t of o.xTicks) {
            svg.append(el("text", { x: g.x(t), y: this.height - 14, class: "tick", "text-anchor": "middle" }, o.formatX(t)));
        }
        svg.append(
            el("line", { x1: this.m.left, x2: g.width - this.m.right, y1: g.y(o.yDomain[0]), y2: g.y(o.yDomain[0]), class: "axis" }),
        );
        svg.append(
            el("text", { x: this.m.left + g.iw / 2, y: this.height - 1, class: "axis-title", "text-anchor": "middle" }, o.xLabel),
        );
        svg.append(el("text", { x: 4, y: 10, class: "axis-title" }, o.yLabel));

        if (o.playhead != null) {
            svg.append(el("line", { x1: g.x(o.playhead), x2: g.x(o.playhead), y1: this.m.top, y2: g.y(o.yDomain[0]), class: "playhead" }));
        }

        for (const s of this.series) {
            if (s.xs.length === 0) continue;
            const d = s.xs.map((xv, i) => `${i === 0 ? "M" : "L"}${g.x(xv).toFixed(1)},${g.y(s.ys[i]).toFixed(1)}`).join("");
            svg.append(
                el("path", {
                    d,
                    fill: "none",
                    stroke: s.color,
                    "stroke-width": s.width ?? 2,
                    "stroke-dasharray": s.dash ?? "",
                    "stroke-linejoin": "round",
                    "stroke-linecap": "round",
                    opacity: s.opacity ?? 1,
                }),
            );
            if (s.markers) {
                s.xs.forEach((xv, i) =>
                    svg.append(el("circle", { cx: g.x(xv), cy: g.y(s.ys[i]), r: 3.5, fill: s.color, class: "dot" })),
                );
            }
        }

        // Direct labels at the right edge, nudged apart so they never overlap.
        const labels = this.series
            .filter((s) => s.xs.length > 0 && s.label)
            .map((s) => ({ s, y: g.y(s.ys[s.ys.length - 1]) }))
            .sort((a, b) => a.y - b.y);
        const gap = 14;
        for (let i = 1; i < labels.length; i++) labels[i].y = Math.max(labels[i].y, labels[i - 1].y + gap);
        for (const l of labels) {
            svg.append(el("text", { x: g.width - this.m.right + 8, y: l.y + 4, class: "direct", fill: l.s.color }, l.s.label));
        }

        // Hover crosshair and focus dots.
        if (this.hoverX != null) {
            const hx = g.x(this.hoverX);
            svg.append(el("line", { x1: hx, x2: hx, y1: this.m.top, y2: g.y(o.yDomain[0]), class: "crosshair" }));
            const rows: string[] = [];
            for (const s of this.series) {
                if (s.noTooltip || s.xs.length === 0) continue;
                const i = nearestIndex(s.xs, this.hoverX);
                svg.append(el("circle", { cx: g.x(s.xs[i]), cy: g.y(s.ys[i]), r: 4, fill: s.color, class: "focus" }));
                rows.push(`<span class="sw" style="background:${s.color}"></span>${s.label}<b>${o.formatY(s.ys[i])}</b>`);
            }
            this.tip.innerHTML = `<div class="tip-head">${o.xLabel} ${o.formatX(Math.round(this.hoverX))}</div>${rows.join("<br>")}`;
            this.tip.hidden = rows.length === 0;
            const left = Math.min(Math.max(8, hx + 12), g.width - 190);
            this.tip.style.left = `${left}px`;
        } else {
            this.tip.hidden = true;
        }

        this.host.querySelector("svg")?.remove();
        this.host.prepend(svg);
    }
}
