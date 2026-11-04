// Vectorized jax-js implementation of the fleet environment.
//
// Every array has a leading batch axis B (independent environments) and, where it
// applies, a satellite axis N: battery/storage/actions are [B, N]. The step function
// is pure and jit-compiled, so thousands of environments advance in a few kernels.
//
// Ownership: jax-js functions consume their array arguments, so every array is
// passed as `x.ref` except for its last use. This file follows that convention
// everywhere; keep it when editing.

import { jit, numpy as np } from "@jax-js/jax";
import type { DType } from "@jax-js/jax";

import {
    ACTION_DOWNLINK,
    ACTION_IMAGE,
    BROWNOUT_PENALTY,
    CONTACT_ANGLE,
    DOWNLINK_RATE,
    DT,
    HOTSPOTS,
    HOTSPOT_SIGMA,
    IMAGE_GAIN,
    INCLINATION,
    LOOKAHEAD_STEPS,
    MEAN_MOTION,
    NUM_SATS,
    OMEGA_EARTH,
    P_BASE,
    P_DOWNLINK,
    P_IMAGE,
    P_SOLAR,
    RAAN,
    REWARD_PER_UNIT,
    SAFE_MODE_BATTERY,
    SHADOW_RHO,
    STATIONS,
    SUN_DIR,
} from "../sim/constants.ts";
import { siteVector } from "../sim/env.ts";

type Arr = np.Array;

// 3 x K matrices of unit vectors, so one matmul gives all dot products at once.
const transpose3 = (vecs: number[][]) => [0, 1, 2].map((k) => vecs.map((v) => v[k]));
const STATIONS_T = transpose3(STATIONS.map(siteVector));
const HOTSPOTS_T = transpose3(HOTSPOTS.map(siteVector));

interface Geometry {
    u: Arr; // [B, N] argument of latitude (orbit phase)
    angleS: Arr; // [B, N, G] central angle to each ground station
    nearest: Arr; // [B, N] smallest of those
    demand: Arr; // [B, N] imaging demand under the sub-point
    sunlit: Arr; // [B, N] 1 if sunlit else 0
}

/** Geometry at time `t` (seconds, scalar). Consumes u0, theta0 and t. */
function geometry(u0: Arr, theta0: Arr, t: Arr, dtype: DType): Geometry {
    const B = theta0.shape[0];
    const u = u0.add(t.ref.mul(MEAN_MOTION));
    const theta = theta0.add(t.mul(OMEGA_EARTH)).reshape([B, 1]);

    // Orbit plane -> inertial frame.
    const xo = np.cos(u.ref);
    const yo = np.sin(u.ref);
    const y1 = yo.ref.mul(Math.cos(INCLINATION));
    const z1 = yo.mul(Math.sin(INCLINATION));
    const cosR = np.array(RAAN.map(Math.cos), { dtype });
    const sinR = np.array(RAAN.map(Math.sin), { dtype });
    const x2 = xo.ref.mul(cosR.ref).sub(y1.ref.mul(sinR.ref));
    const y2 = xo.mul(sinR).add(y1.mul(cosR));

    // Eclipse: behind the Earth relative to the sun, inside the shadow cylinder.
    const d = x2.ref
        .mul(SUN_DIR[0])
        .add(y2.ref.mul(SUN_DIR[1]))
        .add(z1.ref.mul(SUN_DIR[2]));
    const perp = np.sqrt(np.maximum(np.square(d.ref).neg().add(1), 0));
    const shadow = np.logicalAnd(np.less(d, 0), np.less(perp, SHADOW_RHO));
    const sunlit = np.logicalNot(shadow).astype(dtype);

    // Inertial -> Earth-fixed.
    const c = np.cos(theta.ref);
    const s = np.sin(theta);
    const xe = x2.ref.mul(c.ref).add(y2.ref.mul(s.ref));
    const ye = y2.mul(c).sub(x2.mul(s));
    const pos = np.stack([xe, ye, z1], 2); // [B, N, 3]

    const angleS = np.acos(np.clip(np.matmul(pos.ref, np.array(STATIONS_T, { dtype })), -1, 1));
    const angleH = np.acos(np.clip(np.matmul(pos, np.array(HOTSPOTS_T, { dtype })), -1, 1));
    const demand = np.max(np.exp(np.square(angleH.div(HOTSPOT_SIGMA)).neg()), 2);
    const nearest = np.min(angleS.ref, 2);
    return { u, angleS, nearest, demand, sunlit };
}

/** Observation [B, N, OBS_DIM] at time `t`. Consumes all arguments. */
function observation(battery: Arr, storage: Arr, u0: Arr, theta0: Arr, t: Arr, dtype: DType): Arr {
    const now = geometry(u0.ref, theta0.ref, t.ref, dtype);
    const ahead = geometry(u0, theta0, t.add(LOOKAHEAD_STEPS * DT), dtype);
    now.angleS.dispose();
    ahead.angleS.dispose();
    ahead.sunlit.dispose();
    ahead.u.dispose();

    const contactNow = np.less(now.nearest.ref, CONTACT_ANGLE).astype(dtype);
    const contactAhead = np.less(ahead.nearest, CONTACT_ANGLE).astype(dtype);
    return np.stack(
        [
            battery,
            storage,
            now.sunlit,
            now.demand,
            ahead.demand,
            contactNow,
            now.nearest.div(Math.PI),
            contactAhead,
            np.cos(now.u.ref),
            np.sin(now.u),
        ],
        2,
    );
}

export interface StepOutput {
    battery: Arr; // [B, N]
    storage: Arr; // [B, N]
    reward: Arr; // [B, N]
    delivered: Arr; // [B] storage units delivered this step
    brownouts: Arr; // [B] satellites in safe mode after this step
    obs: Arr; // [B, N, OBS_DIM]
}

function stepImpl(
    battery: Arr,
    storage: Arr,
    u0: Arr,
    theta0: Arr,
    t: Arr,
    actions: Arr,
    dtype: DType,
): StepOutput {
    const [B, N] = battery.shape;
    const geo = geometry(u0.ref, theta0.ref, t.ref, dtype);

    // Safe-mode satellites ignore their commands.
    const notSafe = np.greaterEqual(battery.ref, SAFE_MODE_BATTERY).astype(dtype);
    const wantDownlink = np.equal(actions.ref, ACTION_DOWNLINK).astype(dtype).mul(notSafe.ref);
    const isImage = np.equal(actions, ACTION_IMAGE).astype(dtype).mul(notSafe);

    // Each station serves one satellite per step: the requester in contact with the
    // fullest buffer wins (lowest index on ties). Stations are processed in order and
    // a satellite that already won one is skipped, matching the reference simulator.
    let granted = np.zeros([B, N], { dtype });
    for (let g = 0; g < STATIONS.length; g++) {
        const inContact = np.less(geo.angleS.ref.slice([], [], g), CONTACT_ANGLE).astype(dtype);
        const eligible = wantDownlink.ref.mul(granted.ref.neg().add(1)).mul(inContact);
        const score = eligible.mul(storage.ref.add(1)); // 0 means not eligible
        const best = np.argmax(score.ref, 1); // [B]
        const top = np.max(score, 1); // [B]
        const win = np
            .equal(np.arange(N), np.expandDims(best, 1))
            .astype(dtype)
            .mul(np.expandDims(np.greater(top, 0).astype(dtype), 1));
        granted = granted.add(win);
    }
    geo.angleS.dispose();
    geo.u.dispose();
    geo.nearest.dispose();
    wantDownlink.dispose();

    // Imaging only collects data in sunlight; the satellite pays the power either way.
    const gain = np
        .minimum(geo.demand.mul(IMAGE_GAIN), storage.ref.neg().add(1))
        .mul(isImage.ref)
        .mul(geo.sunlit.ref);
    const sent = np.minimum(storage.ref, DOWNLINK_RATE).mul(granted.ref);
    const power = geo.sunlit
        .mul(P_SOLAR)
        .sub(P_BASE)
        .sub(isImage.mul(P_IMAGE))
        .sub(granted.mul(P_DOWNLINK));

    const newStorage = storage.add(gain).sub(sent.ref);
    const newBattery = np.clip(battery.add(power), 0, 1);
    const brown = np.less(newBattery.ref, SAFE_MODE_BATTERY).astype(dtype);
    const reward = sent.ref.mul(REWARD_PER_UNIT).sub(brown.ref.mul(BROWNOUT_PENALTY));
    const delivered = sent.sum(1);
    const brownouts = brown.sum(1);

    const obs = observation(newBattery.ref, newStorage.ref, u0, theta0, t.add(DT), dtype);
    return { battery: newBattery, storage: newStorage, reward, delivered, brownouts, obs };
}

export interface JaxEnv {
    /** Advance one step. Consumes battery, storage, t and actions; pass `.ref` for u0 and theta0. */
    step: (
        battery: Arr,
        storage: Arr,
        u0: Arr,
        theta0: Arr,
        t: Arr,
        actions: Arr,
    ) => StepOutput;
    /** Observation for a state without stepping. Consumes everything passed. */
    observe: (battery: Arr, storage: Arr, u0: Arr, theta0: Arr, t: Arr) => Arr;
}

/** Build jit-compiled step/observe functions for a dtype (float32 on GPU, float64 for parity tests). */
export function makeJaxEnv(dtype: DType = np.float32): JaxEnv {
    const step = jit((battery: Arr, storage: Arr, u0: Arr, theta0: Arr, t: Arr, actions: Arr) =>
        stepImpl(battery, storage, u0, theta0, t, actions, dtype),
    );
    const observe = jit((battery: Arr, storage: Arr, u0: Arr, theta0: Arr, t: Arr) =>
        observation(battery, storage, u0, theta0, t, dtype),
    );
    return { step, observe } as unknown as JaxEnv;
}

export { NUM_SATS };
