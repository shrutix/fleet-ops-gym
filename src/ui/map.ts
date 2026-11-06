// Ground-track map: an equirectangular plot of the fleet at one recorded frame.
// Pure drawing code, no state. Colors come from the page's CSS variables so the map follows
// light and dark mode.

import {
    CONTACT_ANGLE,
    HOTSPOTS,
    HOTSPOT_SIGMA,
    NUM_SATS,
    STATIONS,
    SUN_DIR,
} from "../sim/constants.ts";
import { toEcef } from "../sim/env.ts";
import type { Vec3 } from "../sim/env.ts";
import { FRAME_STRIDE } from "../rl/evaluate.ts";
import type { Recording } from "../rl/evaluate.ts";

export interface Theme {
    sea: string;
    night: string;
    grid: string;
    ink: string;
    muted: string;
    demand: string;
    image: string;
    link: string;
    alert: string;
    idle: string;
}

export function readTheme(): Theme {
    const css = getComputedStyle(document.documentElement);
    const v = (name: string) => css.getPropertyValue(name).trim();
    return {
        sea: v("--map-sea"),
        night: v("--map-night"),
        grid: v("--map-grid"),
        ink: v("--ink"),
        muted: v("--muted"),
        demand: v("--map-demand"),
        image: v("--image"),
        link: v("--link"),
        alert: v("--alert"),
        idle: v("--idle"),
    };
}

export const STATUS_LABEL = ["Idle", "Imaging", "Downlinking", "Downlink not granted", "Safe mode"];

export function satName(i: number): string {
    return `${"ABC"[Math.floor(i / 4)]}${(i % 4) + 1}`;
}

let nightCanvas: HTMLCanvasElement | null = null;

const TRAIL = 48;
const RAD = Math.PI / 180;

function drawMarker(
    ctx: CanvasRenderingContext2D,
    status: number,
    x: number,
    y: number,
    r: number,
    th: Theme,
) {
    ctx.lineWidth = 1.75;
    ctx.beginPath();
    switch (status) {
        case 1: // imaging: filled circle
            ctx.arc(x, y, r, 0, Math.PI * 2);
            ctx.fillStyle = th.image;
            ctx.fill();
            break;
        case 2: // downlinking: filled diamond
            ctx.moveTo(x, y - r * 1.25);
            ctx.lineTo(x + r * 1.25, y);
            ctx.lineTo(x, y + r * 1.25);
            ctx.lineTo(x - r * 1.25, y);
            ctx.closePath();
            ctx.fillStyle = th.link;
            ctx.fill();
            break;
        case 3: // downlink denied: hollow diamond
            ctx.moveTo(x, y - r * 1.25);
            ctx.lineTo(x + r * 1.25, y);
            ctx.lineTo(x, y + r * 1.25);
            ctx.lineTo(x - r * 1.25, y);
            ctx.closePath();
            ctx.strokeStyle = th.link;
            ctx.stroke();
            break;
        case 4: // safe mode: filled square
            ctx.rect(x - r, y - r, r * 2, r * 2);
            ctx.fillStyle = th.alert;
            ctx.fill();
            break;
        default: // idle: hollow circle
            ctx.arc(x, y, r * 0.8, 0, Math.PI * 2);
            ctx.strokeStyle = th.idle;
            ctx.stroke();
    }
}

export function drawMap(
    ctx: CanvasRenderingContext2D,
    w: number,
    h: number,
    rec: Recording,
    frame: number,
    th: Theme,
) {
    const X = (lon: number) => ((lon + 180) / 360) * w;
    const Y = (lat: number) => ((90 - lat) / 180) * h;

    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = th.sea;
    ctx.fillRect(0, 0, w, h);

    // Night side of the Earth at this instant. Each 2 degree cell is one pixel of a small
    // offscreen canvas (so cells never overlap), which is then scaled up with smoothing to give
    // a soft terminator instead of visible tiles.
    const sun = toEcef(SUN_DIR as Vec3, rec.theta[frame]);
    const cols = 180;
    const rows = 90;
    if (!nightCanvas) {
        nightCanvas = document.createElement("canvas");
        nightCanvas.width = cols;
        nightCanvas.height = rows;
    }
    const nctx = nightCanvas.getContext("2d")!;
    nctx.clearRect(0, 0, cols, rows);
    nctx.fillStyle = th.night;
    for (let r = 0; r < rows; r++) {
        const lat = 90 - (r + 0.5) * (180 / rows);
        const cl = Math.cos(lat * RAD);
        const sl = Math.sin(lat * RAD);
        for (let c = 0; c < cols; c++) {
            const lon = (c + 0.5) * (360 / cols) - 180;
            const lr = lon * RAD;
            if (cl * Math.cos(lr) * sun[0] + cl * Math.sin(lr) * sun[1] + sl * sun[2] < 0) {
                nctx.fillRect(c, r, 1, 1);
            }
        }
    }
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(nightCanvas, 0, 0, w, h);

    // Graticule.
    ctx.strokeStyle = th.grid;
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let lon = -150; lon <= 150; lon += 30) {
        ctx.moveTo(X(lon), 0);
        ctx.lineTo(X(lon), h);
    }
    for (let lat = -60; lat <= 60; lat += 30) {
        ctx.moveTo(0, Y(lat));
        ctx.lineTo(w, Y(lat));
    }
    ctx.stroke();

    // Imaging demand.
    const sigmaPx = (HOTSPOT_SIGMA / RAD / 360) * w;
    for (const hs of HOTSPOTS) {
        const cx = X(hs.lonDeg);
        const cy = Y(hs.latDeg);
        const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, sigmaPx * 1.4);
        grad.addColorStop(0, th.demand);
        grad.addColorStop(1, "transparent");
        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.arc(cx, cy, sigmaPx * 1.4, 0, Math.PI * 2);
        ctx.fill();
    }

    // Ground stations and their contact range.
    const small = w < 560;
    ctx.font = `${small ? 10 : 11}px "Archivo Variable", system-ui, sans-serif`;
    ctx.textBaseline = "middle";
    const ringPx = (CONTACT_ANGLE / RAD / 360) * w;
    for (const s of STATIONS) {
        const cx = X(s.lonDeg);
        const cy = Y(s.latDeg);
        ctx.strokeStyle = th.muted;
        ctx.lineWidth = 1;
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.arc(cx, cy, ringPx, 0, Math.PI * 2);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = th.ink;
        ctx.beginPath();
        ctx.moveTo(cx, cy - 5);
        ctx.lineTo(cx + 4.5, cy + 3.5);
        ctx.lineTo(cx - 4.5, cy + 3.5);
        ctx.closePath();
        ctx.fill();
        if (!small) {
            ctx.fillStyle = th.muted;
            ctx.textAlign = "left";
            ctx.fillText(s.name, cx + 9, cy + 1);
        }
    }

    // Trails.
    const first = Math.max(0, frame - TRAIL);
    ctx.lineWidth = 1.25;
    ctx.strokeStyle = th.muted;
    for (let i = 0; i < NUM_SATS; i++) {
        ctx.globalAlpha = 0.55;
        ctx.beginPath();
        let prevLon = 0;
        for (let k = first; k <= frame; k++) {
            const o = (k * NUM_SATS + i) * FRAME_STRIDE;
            const lon = rec.frames[o + 1];
            const x = X(lon);
            const y = Y(rec.frames[o]);
            if (k === first || Math.abs(lon - prevLon) > 180) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
            prevLon = lon;
        }
        ctx.stroke();
    }
    ctx.globalAlpha = 1;

    // Contacts, then satellites on top.
    for (let i = 0; i < NUM_SATS; i++) {
        const o = (frame * NUM_SATS + i) * FRAME_STRIDE;
        if (rec.frames[o + 4] !== 2) continue;
        const lat = rec.frames[o];
        const lon = rec.frames[o + 1];
        let best = STATIONS[0];
        let bestAngle = Infinity;
        for (const s of STATIONS) {
            const a = Math.acos(
                Math.min(
                    1,
                    Math.sin(lat * RAD) * Math.sin(s.latDeg * RAD) +
                    Math.cos(lat * RAD) * Math.cos(s.latDeg * RAD) * Math.cos((lon - s.lonDeg) * RAD),
                ),
            );
            if (a < bestAngle) {
                bestAngle = a;
                best = s;
            }
        }
        if (Math.abs(lon - best.lonDeg) > 180) continue;
        ctx.strokeStyle = th.link;
        ctx.lineWidth = 1.75;
        ctx.beginPath();
        ctx.moveTo(X(lon), Y(lat));
        ctx.lineTo(X(best.lonDeg), Y(best.latDeg));
        ctx.stroke();
    }

    const r = small ? 4 : 5;
    ctx.font = `600 ${small ? 9 : 10}px "Archivo Variable", system-ui, sans-serif`;
    for (let i = 0; i < NUM_SATS; i++) {
        const o = (frame * NUM_SATS + i) * FRAME_STRIDE;
        const x = X(rec.frames[o + 1]);
        const y = Y(rec.frames[o]);
        // Surface-colored halo keeps markers legible over trails and demand glow.
        ctx.beginPath();
        ctx.arc(x, y, r + 3, 0, Math.PI * 2);
        ctx.fillStyle = th.sea;
        ctx.globalAlpha = 0.8;
        ctx.fill();
        ctx.globalAlpha = 1;
        drawMarker(ctx, rec.frames[o + 4], x, y, r, th);
        if (!small) {
            ctx.fillStyle = th.ink;
            ctx.textAlign = "left";
            ctx.fillText(satName(i), x + r + 5, y - r - 2);
        }
    }
}
