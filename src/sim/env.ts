// Plain-TypeScript reference implementation of the satellite fleet environment.
//
// This is deliberately scalar, loop-based code. It is the ground truth that the
// vectorized jax-js version (vmap + jit on WebGPU) must match step for step.

import {
    ACTION_DOWNLINK,
    ACTION_IDLE,
    ACTION_IMAGE,
    BROWNOUT_PENALTY,
    CONTACT_ANGLE,
    DOWNLINK_RATE,
    DT,
    EPISODE_STEPS,
    HOTSPOTS,
    HOTSPOT_SIGMA,
    IMAGE_GAIN,
    INCLINATION,
    LOOKAHEAD_STEPS,
    MEAN_MOTION,
    NUM_PLANES,
    NUM_SATS,
    OBS_DIM,
    OMEGA_EARTH,
    P_BASE,
    P_DOWNLINK,
    P_IMAGE,
    P_SOLAR,
    RAAN,
    REWARD_PER_UNIT,
    SAFE_MODE_BATTERY,
    SATS_PER_PLANE,
    SHADOW_RHO,
    STATIONS,
    SUN_DIR,
} from "./constants.ts";
import type { Site } from "./constants.ts";

export type Vec3 = [number, number, number];

/** Deterministic PRNG (mulberry32). */
export function makeRng(seed: number): () => number {
    let a = seed >>> 0;
    return () => {
        a = (a + 0x6d2b79f5) >>> 0;
        let t = a;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

export function siteVector(site: Site): Vec3 {
    const lat = (site.latDeg * Math.PI) / 180;
    const lon = (site.lonDeg * Math.PI) / 180;
    return [Math.cos(lat) * Math.cos(lon), Math.cos(lat) * Math.sin(lon), Math.sin(lat)];
}

const dot = (a: Vec3, b: Vec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const clamp = (x: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, x));
const centralAngle = (a: Vec3, b: Vec3) => Math.acos(clamp(dot(a, b), -1, 1));

const STATION_VECS = STATIONS.map(siteVector);
const HOTSPOT_VECS = HOTSPOTS.map(siteVector);

/** Unit position of a satellite in the inertial frame. */
export function eciUnit(raan: number, u: number): Vec3 {
    const xo = Math.cos(u);
    const yo = Math.sin(u);
    const y1 = yo * Math.cos(INCLINATION);
    const z1 = yo * Math.sin(INCLINATION);
    const x2 = xo * Math.cos(raan) - y1 * Math.sin(raan);
    const y2 = xo * Math.sin(raan) + y1 * Math.cos(raan);
    return [x2, y2, z1];
}

/** Rotate an inertial vector into the Earth-fixed frame at Earth angle theta. */
export function toEcef(p: Vec3, theta: number): Vec3 {
    const c = Math.cos(theta);
    const s = Math.sin(theta);
    return [p[0] * c + p[1] * s, -p[0] * s + p[1] * c, p[2]];
}

export function inEclipse(p: Vec3): boolean {
    const d = dot(p, SUN_DIR);
    if (d >= 0) return false;
    return Math.sqrt(1 - d * d) < SHADOW_RHO;
}

export function demandAt(ecef: Vec3): number {
    let best = 0;
    for (const h of HOTSPOT_VECS) {
        const a = centralAngle(ecef, h);
        best = Math.max(best, Math.exp(-((a / HOTSPOT_SIGMA) ** 2)));
    }
    return best;
}

export function nearestStationAngle(ecef: Vec3): number {
    let best = Math.PI;
    for (const g of STATION_VECS) best = Math.min(best, centralAngle(ecef, g));
    return best;
}

/** Geometry of one satellite at one instant. */
export interface Geometry {
    ecef: Vec3;
    sunlit: boolean;
    demand: number;
    nearest: number; // central angle to nearest station
}

export interface StepResult {
    obs: Float32Array; // NUM_SATS * OBS_DIM
    reward: Float32Array; // per satellite
    done: boolean;
    delivered: number; // total storage units delivered this step
}

export class FleetEnv {
    readonly numSats = NUM_SATS;
    readonly obsDim = OBS_DIM;

    battery = new Float32Array(NUM_SATS);
    storage = new Float32Array(NUM_SATS);
    step_ = 0;
    totalDelivered = 0;
    totalBrownouts = 0;

    /**
     * What each satellite actually did in the last step (for visualization only):
     * 0 idle, 1 imaging, 2 downlinking, 3 downlink requested but not granted, 4 safe mode.
     */
    lastStatus = new Uint8Array(NUM_SATS);

    private raan = new Float64Array(NUM_SATS);
    private u0 = new Float64Array(NUM_SATS);
    private theta0 = 0;
    private rng: () => number;

    constructor(seed = 0) {
        this.rng = makeRng(seed);
        for (let i = 0; i < NUM_SATS; i++) this.raan[i] = RAAN[i];
    }

    /** Copy of the episode state, used to start the vectorized env from the same point. */
    snapshot() {
        return {
            u0: Array.from(this.u0),
            theta0: this.theta0,
            battery: Array.from(this.battery),
            storage: Array.from(this.storage),
        };
    }

    reset(): Float32Array {
        for (let p = 0; p < NUM_PLANES; p++) {
            const planeOffset = this.rng() * 2 * Math.PI;
            for (let k = 0; k < SATS_PER_PLANE; k++) {
                this.u0[p * SATS_PER_PLANE + k] = planeOffset + (2 * Math.PI * k) / SATS_PER_PLANE;
            }
        }
        this.theta0 = this.rng() * 2 * Math.PI;
        for (let i = 0; i < NUM_SATS; i++) {
            this.battery[i] = 0.5 + 0.5 * this.rng();
            this.storage[i] = 0.05 * this.rng();
        }
        this.step_ = 0;
        this.totalDelivered = 0;
        this.totalBrownouts = 0;
        return this.observe();
    }

    /** Earth rotation angle (radians) at a step index. */
    thetaAt(stepIndex: number): number {
        return this.theta0 + OMEGA_EARTH * stepIndex * DT;
    }

    geometry(i: number, stepIndex: number): Geometry {
        const t = stepIndex * DT;
        const eci = eciUnit(this.raan[i], this.u0[i] + MEAN_MOTION * t);
        const ecef = toEcef(eci, this.theta0 + OMEGA_EARTH * t);
        return {
            ecef,
            sunlit: !inEclipse(eci),
            demand: demandAt(ecef),
            nearest: nearestStationAngle(ecef),
        };
    }

    observe(): Float32Array {
        const out = new Float32Array(NUM_SATS * OBS_DIM);
        for (let i = 0; i < NUM_SATS; i++) {
            const now = this.geometry(i, this.step_);
            const ahead = this.geometry(i, this.step_ + LOOKAHEAD_STEPS);
            const phase = this.u0[i] + MEAN_MOTION * this.step_ * DT;
            const o = i * OBS_DIM;
            out[o + 0] = this.battery[i];
            out[o + 1] = this.storage[i];
            out[o + 2] = now.sunlit ? 1 : 0;
            out[o + 3] = now.demand;
            out[o + 4] = ahead.demand;
            out[o + 5] = now.nearest < CONTACT_ANGLE ? 1 : 0;
            out[o + 6] = now.nearest / Math.PI;
            out[o + 7] = ahead.nearest < CONTACT_ANGLE ? 1 : 0;
            out[o + 8] = Math.cos(phase);
            out[o + 9] = Math.sin(phase);
        }
        return out;
    }

    step(actions: ArrayLike<number>): StepResult {
        if (actions.length !== NUM_SATS) throw new Error(`expected ${NUM_SATS} actions`);
        const geo: Geometry[] = [];
        for (let i = 0; i < NUM_SATS; i++) geo.push(this.geometry(i, this.step_));

        const safe: boolean[] = [];
        for (let i = 0; i < NUM_SATS; i++) safe.push(this.battery[i] < SAFE_MODE_BATTERY);

        // Each station serves at most one satellite per step: the requester in
        // contact with the fullest buffer wins (ties go to the lower index).
        const granted: boolean[] = new Array(NUM_SATS).fill(false);
        for (const gv of STATION_VECS) {
            let best = -1;
            for (let i = 0; i < NUM_SATS; i++) {
                if (actions[i] !== ACTION_DOWNLINK || safe[i] || granted[i]) continue;
                if (centralAngle(geo[i].ecef, gv) >= CONTACT_ANGLE) continue;
                if (best < 0 || this.storage[i] > this.storage[best]) best = i;
            }
            if (best >= 0) granted[best] = true;
        }

        const reward = new Float32Array(NUM_SATS);
        let delivered = 0;
        for (let i = 0; i < NUM_SATS; i++) {
            let power = -P_BASE + (geo[i].sunlit ? P_SOLAR : 0);
            const action = safe[i] ? ACTION_IDLE : actions[i];
            this.lastStatus[i] = safe[i]
                ? 4
                : action === ACTION_IMAGE
                    ? 1
                    : granted[i]
                        ? 2
                        : action === ACTION_DOWNLINK
                            ? 3
                            : 0;

            if (action === ACTION_IMAGE) {
                power -= P_IMAGE;
                if (geo[i].sunlit) {
                    const gain = Math.min(IMAGE_GAIN * geo[i].demand, 1 - this.storage[i]);
                    this.storage[i] += gain;
                }
            } else if (granted[i]) {
                power -= P_DOWNLINK;
                const sent = Math.min(this.storage[i], DOWNLINK_RATE);
                this.storage[i] -= sent;
                reward[i] += sent * REWARD_PER_UNIT;
                delivered += sent;
            }

            this.battery[i] = clamp(this.battery[i] + power, 0, 1);
            if (this.battery[i] < SAFE_MODE_BATTERY) {
                reward[i] -= BROWNOUT_PENALTY;
                this.totalBrownouts++;
            }
        }

        this.step_++;
        this.totalDelivered += delivered;
        return {
            obs: this.observe(),
            reward,
            done: this.step_ >= EPISODE_STEPS,
            delivered,
        };
    }
}