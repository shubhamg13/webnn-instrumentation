# MLComputeTiming: engine-defined compute timing for WebNN

> **Companion explainer:** `webnn-ml-resource-counts-explainer.md` covers resource
> accounting (`MLResourceCounts`). It is a separate API with a separate home (WebNN, not the
> Performance Timeline), so it is documented separately.

## Authors

- Shubham Gupta (Huawei)

## Participate

- Discussion: [webnn#963 — Expose engine-defined compute timing for WebNN](https://github.com/webmachinelearning/webnn/issues/963)
- Web Machine Learning Community Group: <https://www.w3.org/groups/cg/webmachinelearning>
- Web Performance WG (entry-type registry): <https://www.w3.org/groups/wg/webperf>

## Introduction

The web Performance Timeline and the specifications that extend it — Navigation Timing,
Resource Timing, User Timing, Paint Timing, Event Timing, Long Tasks — define a rich set of
capabilities for measuring the performance of a page, so that developers can monitor,
understand, and improve user experience. 

WebNN is now eligible to be part of that set: A page can run inference, and the platform offers no way to measure it. 

- WebNN has no performance surface.
- No timing. No attribution.
- A page can time its own `dispatch()` call — but that number is not the computation.
- Folling Measured in Device's face-detection demo (in mobile device on NPU backend):

| measurement | API field | how it was taken | value |
|---|---|---|---|
| Actual graph execution | `computeDuration` | engine timestamps around the backend run | **5.69 ms** |
| Dispatch round trip | `duration` | page timer around `dispatch()` only | **5.85 ms** |
| Page-timed "Inference" | — | `dispatch()` + two awaited `readTensor()` calls (first run) | **209.23 ms** |

Two problems:

1. **The dispatch round trip is longer than the compute.** 5.85 ms vs 5.69 ms — 0.16 ms of async
   channel.
2. **The page-timed number is not interoperable.** 209 ms, because completion requires a readback.
   Different Engines plumb that path differently; the same page measures different quantities.

The demo's 209 ms comes from a Yolo-n8 Face Detection Model using following script logging

```js
const start = performance.now();
context.dispatch(graph.graph, inputs, outputs);

// Read results
const result_regressors = new Float32Array(await context.readTensor(outputTensor_regressors));
const result_classificators = new Float32Array(await context.readTensor(outputTensor_classificators));
console.log(`Inference: ${(performance.now() - start).toFixed(2)}ms (1 run) on ${deviceType.toUpperCase()}`);
```

- It times `dispatch()` plus two awaited `readTensor()` calls.
- Plus whatever the engine defers to first use.
- The page cannot tell which part is compute.
- `dispatch()` has no completion signal: *"callers can await the results of reading back the output tensors."*

## Proposal

- A uniform, **engine-defined** interval, interoperable by construction.
- Home: WebMAchineLearning Performance Timeline.
- Principle: **inherit the plumbing, own the semantics.**

## Scope

This document adds the compute side benchmarking: `MLComputeTiming`
- Registered with the Performance Timeline and populated by the engine.

It covers:

- `ml-compile` and `ml-compute`, carried by `MLComputeTiming`.
- Both WebNN surfaces: the graph API and the Model Loader API.
- Engine-created entries, delivered through the Performance Timeline.

Adjacent questions are separate by design:

| question | where |
|---|---|
| resource accounting | `webnn-ml-resource-counts-explainer.md` |
| device identity (vendor, architecture, model) | `webnn#934` |
| tensor transfers (`writeTensor`, `readTensor`) | deferred; `readTensor` ordering belongs to WebNN |
| per-operator data, tensor shapes | DevTools |
| analysis and benchmarking algorithms | not a platform concern (WebPerf charter) |

## User-Facing Problem

A page cannot answer:

- **How long did the computation take?** It times the call; the bracket is engine-specific.
- **Did it run on the hardware I asked for?** Silent CPU fallback is spec-mandated.

### Motivating scenario

- Face detector: 30 fps on a fast device, 5 fps on a slow one.
- Page wants a smaller model on the slow device.
- It cannot separate "model", "hardware" and "my own input prep" — one conflated number.

### Goals

- One engine-defined interval for compile and one for execution, identical across engines/backends.
- Per-entry attribution: accelerated or not.
- Cheap by construction: one entry per execution, small buffers, no per-op data.
- One definition covering both WebNN surfaces: graph API and Model Loader.

### Non-goals

- Per-operator timing, tensor shapes — DevTools, not the Timeline.
- Device identity (vendor/architecture/model) — `webnn#934`.
- Analysis or benchmarking algorithms — outside the WebPerf charter.
- Resource accounting — the companion explainer.

## User research

### Developer need

- The demo's author had to build the status-quo workaround: time `dispatch()` and await two
  readbacks. The result is 209 ms against 5.7 ms of compute — the page cannot tell the difference.
- The CPU Performance API (WICG) exists for "which ML model to download and run locally"; a device
  tier answers what the device *is*, not what a run *costs*. Its open TAG review asks for
  NPU/GPU-aware data.
- WebNN §5.2 already invites input on the execution-time analysis fingerprinting vector — the WG
  has acknowledged that execution timing is a question it owns.
- No formal user study has been run. The evidence class is implementation experience, measured
  device runs and the WG threads — the same class the CPU Performance API review accepted in place
  of one.

### Prior art

- **Timeline precedents:** User Timing, Event Timing, LCP, Long Tasks, `WICG/delayed-message-timing`.
- **Non-web precedent:** Node `perf_hooks` `'gc'`.
- **Compute Pressure:** nearest compute metric; deliberately reports outside the timeline.
- **CPU Performance API (WICG):** closest neighbour; open TAG review `w3ctag/design-reviews#1198`.
  - Already established: NPU/GPU relevance, bit accounting, third-party context, normative buckets.
- **`AudioPlaybackStats`** (Web Audio 1.1 draft): the closest stats surface in kind.
  - Dedicated object on `AudioContext`, with a normative Risk/Mitigations privacy section.
  - Updates at most once per second; gated on permission or a visible document.
  - The template for rate limiting, if this proposal ever needs it.

### Implementation experience: Servo, across platforms

- Servo targets Linux, macOS, Windows, Android and OpenHarmony.
- Backend per target, one trait:
  - macOS: CoreML.
  - Linux: LiteRT, ONNX Runtime fallback.
  - Windows: ONNX Runtime.
  - OpenHarmony: LiteRT + CANN.
- What an implementation adds:
  - A completion point per execution (`dispatch()` already has one callback per run).
  - A completion point per compile (`build()` / `load()` resolution).
  - Two timestamps per run (submission, completion, backend interval).
- Servo's internal probes stay internal; only the comparable part is standardised.

### Sibling API: Model Loader

- `webmachinelearning/model-loader`: file-format front end to the same execution.
- Its API has **no timing surface**.
- Its privacy section already names the risk:
  - API "execute[s] much faster or slower, depending on the device".
  - That "makes fingerprinting easier".
  - Mitigation: "bucket the API behavior into fewer categories".

## Sample code

```js
// Observer-only entries; performance.getEntries*() does not return them.
const observer = new PerformanceObserver((list) => {
  for (const entry of list.getEntries()) {
    if (entry.entryType === "ml-compile") {
      console.log(`compiled ${entry.name} in ${entry.duration.toFixed(2)} ms`);
    } else {
      console.log(`ran ${entry.name}: ` +
                  `${entry.duration.toFixed(2)} ms total, ` +
                  `${entry.computeDuration.toFixed(2)} ms compute, ` +
                  `accelerated=${entry.accelerated}`);
    }
  }
});
observer.observe({ type: "ml-compile", buffered: true });
observer.observe({ type: "ml-compute", buffered: true });

const context = await navigator.ml.createContext({ deviceType: "npu" });
const builder = new MLGraphBuilder(context);
const input = builder.input("input", { dataType: "float32", shape: [1, 3, 224, 224] });
const output = builder.relu(input);
const graph = await builder.build({ output });            // one ml-compile entry

const inputTensor = await context.createTensor(
  { dataType: "float32", shape: [1, 3, 224, 224], writable: true });
const outputTensor = await context.createTensor(
  { dataType: "float32", shape: [1, 1000], readable: true });
context.writeTensor(inputTensor, inputData);
context.dispatch(graph, { input: inputTensor }, { output: outputTensor });  // one ml-compute entry
const bytes = await context.readTensor(outputTensor);

// Model Loader ships the same entry types:
//   const model = await loader.load(modelBuffer);   // one ml-compile entry
//   await model.compute({ input: inputData });      // one ml-compute entry
```

## Proposed Approach

- One `PerformanceEntry` interface: `MLComputeTiming`.
- Two entry types: `ml-compile`, `ml-compute`.
- Entries are engine-created at spec-defined completion points, never page-created.
- Delivered through the standard Timeline machinery.
- Covers both surfaces: graph API and Model Loader.
- Backend self-description: left to `webnn#934`.

## Detailed Design

### `MLComputeTiming`

```webidl
[Exposed=(Window,Worker)]
interface MLComputeTiming : PerformanceEntry {
  readonly attribute DOMHighResTimeStamp computeDuration;
  readonly attribute boolean accelerated;
  [Default] object toJSON();
};
```

- `name`, `startTime`, `duration`, `entryType` are inherited; not redeclared.

| entryType | created when | maxBufferSize |
|---|---|---|
| `ml-compile` | compilation completes, incl. weight preprocessing (`build()`, `load()`) | 50 |
| `ml-compute` | one graph execution completes (`dispatch()`, `compute()`) | 150 |

- Two types, one interface: different buffering, per-type subscription.
- `availableFromTimeline: false` for both — observer-only, like LCP and Long Tasks.
- `entryType` is the discriminator; no `kind` field.

### Fields

A graph has no intrinsic name. `build(outputs)` takes an ordered record of output name → operand;
the page also chooses input names and operator labels, and `MLGraph` exposes no structure. So
`name` is taken from the graph's first output and is shared by every entry for that graph —
including every `ml-compute` from `dispatch()`, which runs the same graph.

| field | meaning |
|---|---|
| `name` | the graph's name: the first output name in the `build(outputs)` record; shared by its compile and run entries; empty if there is none |
| `startTime` / `duration` | submission → completion, `timeOrigin`-relative; includes queueing, dispatch, transfer |
| `computeDuration` | the engine's compute interval; `≤ duration`; the remainder is not itemised |
| `accelerated` | one bit; without it a 6 ms CPU run and a 6 ms NPU run look identical |

### Creation and delivery

- `ml-compile` queued when the compile promise resolves (`build()`, `load()`).
- Scope includes initialisation (weight preprocessing, possibly OS-cached).
- An engine that defers compile to first execution must still attribute it — stated in the should-add algorithm.
- `ml-compute` queued at execution completion, before/independently of readback — the completion signal the spec lacks.
- Buffered per type up to `maxBufferSize`; delivered to `PerformanceObserver`.
- Entry must redeclare `[Default] object toJSON()`.
- Registry row per type: `entryType`, spec link, requester, `availableFromTimeline`, `maxBufferSize`, should-add algorithm.
- The should-add algorithm hosts the normative definitions of `duration` and `computeDuration`.

### Dependencies on non-stable features

- **`webnn#923`** — compute-policy rename; removes `MLContext.accelerated`.
- **`webnn#934`** — effective compute policies; decides the long-term shape of `accelerated`.
- **Model Loader API** — Tentative Specification in the WG charter; second surface.
- **Timing entry types registry** — WebPerf row for each type.

### Integrating with the WebNN developer story

- No change to `build()`/`dispatch()` semantics.
- No permission, no feature flag required by the spec.
- Pages that never observe pay nothing beyond a small fixed buffer.
- Model Loader gets the same entries by construction.

## Solving Goal 1: a uniform interval

- Both entries start at submission and end at engine completion — same meaning in every engine.
- The same two types cover graph API and Model Loader.
- A benchmark written once compares across engines, backends and APIs.

## Solving Goal 2: attribution

- `accelerated` is per entry, so a silent CPU fallback cannot pass for acceleration.
- Bridges to `#934`: can stay boolean or align with policy strings without changing the entry shape.

## Alternatives considered

- **User Timing (`performance.measure`)** — available today; arbitrary fields.
  - Page-authored: two engines' numbers are comparable only if two authors wrote identical code.
  - Cannot supply `accelerated`.
- **WebGPU timestamp queries** — feature-gated, quantised, need readback; measure inside a pass, not the whole inference.
- **Compute Pressure** — system pressure, outside the timeline by design; fallback plan and rate-limit template.
- **Per-operator timing** — not standardised.
  - Backends fuse, lower and reorder ops; a "node" is not a stable unit across engines, so
    per-node numbers would not be comparable — the property this proposal exists to provide.
  - Volume and fingerprinting: one entry per node sharpens the timing signal.
  - The page-level workaround is to split the graph into subgraphs that share `MLTensor`s; each
    dispatch emits its own `ml-compute` entry. Caveat: graph breaks prevent fusion, so the sum of
    the parts can differ from the whole — compare `computeDuration`, not `duration`.
  - Op-level attribution belongs in the implementation's profiler (DevTools). If a standard
    surface is wanted, it should be a separate, opt-in, quantised proposal.
- **A dedicated `MLContext` timing dictionary** — reimplements observation/buffering/serialisation; misses `PerformanceObserver`.
- **Device identity only** — `#934`'s question; answers "what is this device", not "what did this run cost".

| API | answers | shape |
|---|---|---|
| CPU Performance API (WICG) | what the device *is* | coarse tier, bucketed, static |
| WebNN `#934` effective policies | what the *graph* got | post-compile, on `MLGraph` |
| `MLComputeTiming` | what *this run* did | per-execution entries |

## Alternatives: Previous API shape

- Original sketch (2026-10-09) had three parts: `MLContext.backend` + `ml-compute` + resource counts.
- `MLContext.backend` → dropped; contributed to `#934` instead.
- `ml-compute` → gained `computeDuration` and an `ml-compile` sibling.
- Resource counts → split into the companion explainer (`MLResourceCounts`).
- `graphId` → dropped: the graph's name is the correlator, and `dispatch()` runs the same graph, so
  its entries reuse it.

## Accessibility, Internationalization, Privacy, and Security Considerations

### Accessibility and internationalization

- No UI, no focus, no live region.
- Entries are not exposed to assistive technology; `toJSON()` is scripting data.
- `name` is page-chosen and locale-independent.

### Security considerations

- No new attack surface: reads engine bookkeeping for the page's own objects.
- No parsing of untrusted input (unlike a model loader).
- Entries are scoped to the document that created the graph; no cross-origin channel.

### Privacy considerations

- Marginal disclosure is the core argument:
  - `duration` and `name` are page-equivalent or page-chosen.
  - `computeDuration` is the one genuinely new signal — and where review pressure should land.
- No device identity: nothing reports vendor/architecture/model; one bit sits inside WebGPU's 32-configuration cap.
- Per-document: iframe entries stay in the iframe; nothing aggregates upward.
- Coarse by construction: per-execution, small buffer, no per-op data.
- `computeDuration` mitigation, if demanded: quantisation of the value (`duration` buckets of ≥10% of devices), or drop the field and ship `duration` only.
- Already acknowledged: WebNN §5.2 invites input on execution-time fingerprinting; Model Loader names bucketing as the mitigation.
- Levers, if demanded: rate limiting (Compute Pressure / `AudioPlaybackStats` style) or cross-origin isolation.
- Circuit breaker: if any per-run timing is rejected, follow Compute Pressure and report outside the timeline; the companion counts API stands alone.

## Stakeholder feedback / opposition

- Servo: proposing this design.
- WebML WG: `webnn#934` is the first thread (evidence + bits argument).
- WebPerf WG: registry and charter scope; `@yoavweiss`, `@tunetheweb`.
- TAG: after WG threads; review takes 4–6 weeks.
- Other implementers: no public positions yet; to be recorded here.

## Open questions

1. `name`: the graph's first output name is its name, shared by all its entries. Is that enough, or
   does the group want a page-supplied label (which would need a new `build` surface)?
2. `accelerated`: boolean, or `#934` policy strings? Partial split?
3. Transfers: `ml-transfer` entry, or fix `readTensor` ordering in WebNN first?
4. `ml-compile` in draft 1, or second?
5. `computeDuration`: normative, or `duration`-only fallback?
6. Per-node attribution: split-the-graph only, or a separate opt-in profiling proposal?

## References & acknowledgements

### References

- WebNN spec: `dispatch()`, `build()`, tensor methods, §5.2.
- [webnn#963](https://github.com/webmachinelearning/webnn/issues/963) — the proposal issue.
- **Performance Timeline** (W3C) — the primitives this proposal extends; its introduction is
  the framing this document follows.
- **High Resolution Time** — `DOMHighResTimeStamp` and `Performance.timeOrigin`.
- Timing entry types registry; Navigation Timing; Resource Timing; User Timing; Paint Timing;
  Event Timing; Long Tasks; LCP; Element Timing.
- `webnn`: `#934`, `#900`, `#923`, `#854`, `#836`.
- `webmachinelearning/model-loader` explainer.
- `WICG/delayed-message-timing`; Node `perf_hooks` `'gc'`.
- `WICG/soft-navigations` (explainer structure).
- `w3ctag/design-reviews#1198` (CPU Performance API).
- `AudioPlaybackStats` (Web Audio 1.1 draft) — rate-limited stats precedent; `renderCapacity` — removed after privacy review.
- WebGPU adapter-info cap, timestamp queries, `GPUQueue.onSubmittedWorkDone()`.
- `webmachinelearning/meetings#39`.

### Acknowledgements

- Structure follows the Model Loader and Soft Navigations explainers.
- Reviewers and contributors to be listed as the design is discussed.
