// Node-only shims so jax-js (0.1.27) runs on older Node versions. Import this from Node
// scripts and tests only; browsers already provide everything here.
import { cpus } from "node:os";

const g = globalThis as { Float16Array?: unknown; navigator?: unknown };

// Node < 24 has no `Float16Array` global. jax-js evaluates `data instanceof Float16Array`
// while dispatching on typed-array type, so building a float64 array throws
// "Float16Array is not defined" before it reaches the float64 branch. A placeholder class
// makes that check false for every real array. Nothing in this project uses float16.
if (typeof g.Float16Array === "undefined") {
    g.Float16Array = class Float16Array { };
}

// Node < 21 has no `navigator`. jax-js reads `navigator.userAgent` in its kernel optimizer
// and `navigator.hardwareConcurrency` / `navigator.gpu` when choosing backends. There is
// deliberately no `gpu` property, so WebGPU is reported as unavailable (it is, in Node).
if (typeof g.navigator === "undefined") {
    g.navigator = { userAgent: "Node.js", hardwareConcurrency: cpus().length };
}

// Node < 20 lacks the ES2023 non-mutating array methods. jax-js calls `toSpliced` (shape
// tracking) and `toReversed` (wasm codegen). These are the standard semantics: operate on a
// copy and return it. Only the two methods jax-js actually uses are provided.
const arrayProto = Array.prototype as unknown as Record<string, unknown>;
if (typeof arrayProto.toSpliced !== "function") {
    Object.defineProperty(Array.prototype, "toSpliced", {
        configurable: true,
        writable: true,
        value: function toSpliced(this: unknown[], ...args: unknown[]) {
            const copy = this.slice();
            copy.splice(...(args as [number, number]));
            return copy;
        },
    });
}
if (typeof arrayProto.toReversed !== "function") {
    Object.defineProperty(Array.prototype, "toReversed", {
        configurable: true,
        writable: true,
        value: function toReversed(this: unknown[]) {
            return this.slice().reverse();
        },
    });
}

export { };
