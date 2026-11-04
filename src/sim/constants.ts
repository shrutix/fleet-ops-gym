// Physical and game constants for the fleet-ops simulator.
// Lengths in km, angles in radians, time in seconds unless noted.

export const R_EARTH = 6371;
export const MU_EARTH = 398600.4418; // km^3 / s^2
export const OMEGA_EARTH = 7.2921159e-5; // rad / s

export const ALTITUDE_KM = 500;
export const ORBIT_RADIUS = R_EARTH + ALTITUDE_KM;
export const MEAN_MOTION = Math.sqrt(MU_EARTH / ORBIT_RADIUS ** 3); // rad / s
/** Shadow cylinder radius in units of orbit radius. */
export const SHADOW_RHO = R_EARTH / ORBIT_RADIUS;

// Constellation layout: sun-synchronous-ish planes.
export const NUM_PLANES = 3;
export const SATS_PER_PLANE = 4;
export const NUM_SATS = NUM_PLANES * SATS_PER_PLANE;
export const INCLINATION = (97.4 * Math.PI) / 180;
/** Right ascension of the ascending node for each satellite (shared by all environments). */
export const RAAN: number[] = Array.from(
    { length: NUM_SATS },
    (_, i) => (Math.floor(i / SATS_PER_PLANE) * Math.PI) / NUM_PLANES,
);

// Fixed (inertial) direction to the sun, normalized at load time.
const SUN_RAW: [number, number, number] = [1, 0.2, 0.1];
const SUN_NORM = Math.hypot(...SUN_RAW);
export const SUN_DIR: [number, number, number] = [
    SUN_RAW[0] / SUN_NORM,
    SUN_RAW[1] / SUN_NORM,
    SUN_RAW[2] / SUN_NORM,
];

// Ground stations (lat, lon in degrees). Generic polar / high-latitude sites,
// which see sun-synchronous satellites on most orbits.
export interface Site {
    name: string;
    latDeg: number;
    lonDeg: number;
}

export const STATIONS: Site[] = [
    { name: "Svalbard", latDeg: 78.2, lonDeg: 15.4 },
    { name: "Fairbanks", latDeg: 64.8, lonDeg: -147.7 },
    { name: "Punta Arenas", latDeg: -53.2, lonDeg: -70.9 },
    { name: "Awarua", latDeg: -46.5, lonDeg: 168.4 },
];

// Demand hotspots: regions where imaging is valuable.
export const HOTSPOTS: Site[] = [
    { name: "Amazon basin", latDeg: -3, lonDeg: -60 },
    { name: "US Midwest", latDeg: 41, lonDeg: -93 },
    { name: "Western Europe", latDeg: 48, lonDeg: 5 },
    { name: "Nile delta", latDeg: 30, lonDeg: 31 },
    { name: "Indo-Gangetic plain", latDeg: 26, lonDeg: 80 },
    { name: "Yangtze delta", latDeg: 31, lonDeg: 121 },
    { name: "Southeast Asia", latDeg: 5, lonDeg: 105 },
    { name: "Southern Africa", latDeg: -26, lonDeg: 28 },
];

/** A station can serve a satellite whose sub-point is within this central angle. */
export const CONTACT_ANGLE = 0.25;
/** Width (central angle) of a demand hotspot. */
export const HOTSPOT_SIGMA = 0.4;

// Episode and time stepping.
export const DT = 60; // seconds per step
export const EPISODE_STEPS = 360; // 6 hours, about 3.8 orbits
export const LOOKAHEAD_STEPS = 10;

// Satellite resource model (all fractions of full scale per step).
export const P_SOLAR = 0.012;
export const P_BASE = 0.004;
export const P_IMAGE = 0.012;
export const P_DOWNLINK = 0.01;
export const IMAGE_GAIN = 0.06; // storage added per step at demand = 1
export const DOWNLINK_RATE = 0.08; // storage removed per granted step
export const SAFE_MODE_BATTERY = 0.1; // below this, only charging is allowed
export const BROWNOUT_PENALTY = 0.05;
export const REWARD_PER_UNIT = 10; // reward per unit of storage delivered

export const OBS_DIM = 10;
export const NUM_ACTIONS = 3;
export const ACTION_IDLE = 0;
export const ACTION_IMAGE = 1;
export const ACTION_DOWNLINK = 2;
