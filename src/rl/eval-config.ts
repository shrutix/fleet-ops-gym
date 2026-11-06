// Shared by the CLI trainer and the browser worker so both report comparable numbers.

/** Held-out seeds. Training episodes use a different seed range, so these are never seen. */
export const EVAL_SEEDS = Array.from({ length: 32 }, (_, i) => 5000 + i);

/** Best thresholds found by sweeping the greedy policy (see README). */
export const TUNED_GREEDY = { minDemand: 0.2, minBattery: 0.2, maxStorage: 0.95 };
