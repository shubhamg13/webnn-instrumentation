# MLResourceCounts: engine-defined resource accounting for WebNN

> **Companion explainer:** `webnn-ml-compute-timing-explainer.md` covers compute timing
> (`MLComputeTiming`). Resource accounting is a separate API with a different home — a WebNN
> context attribute, not a Performance Timeline entry — so it is documented separately.

## Authors

- Shubham Gupta (Huawei)

## Participate

- Issue tracker: to be filed in `webmachinelearning/webnn`
- Web Machine Learning Community Group: <https://www.w3.org/groups/cg/webmachinelearning>

## Introduction

The Performance Timeline measures what a page **does** — navigation, resources, scripts, paint,
events, long tasks. Nothing measures what it **holds**. WebNN's graphs and tensors are
native-memory holders with an explicit `destroy()`; a page cannot see how many remain, and a
correct engine still keeps them until the page calls `destroy()` or the GC collects them. This
document adds the accounting side: `MLContext.resourceCounts`, two integers the page can read
and assert against.

- WebNN has no accounting surface.
- A page creates graphs and tensors that hold native memory; it cannot see how many remain.
- Even a correct engine keeps them until the page calls `destroy()` or the GC collects them.
- The page cannot observe GC, so it cannot tell whether its last `destroy()` worked.
- Three page needs:
  - bound a cache of graphs/tensors;
  - catch its own leaks;
  - assert release in tests.
- Proposal: `MLContext.resourceCounts`, a snapshot of how many graphs and tensors the context
  owns. Counts, not bytes. Deterministic for `destroy()`.

## Scope

This document specifies **resource accounting only**. It covers:

- `MLContext.resourceCounts` and the `MLResourceCounts` dictionary.
- Graphs and tensors owned by a context, counted as script-visible objects.
- Deterministic release for `destroy()` and context teardown.

Adjacent questions are separate by design:

| question | where |
|---|---|
| compute timing | `webnn-ml-compute-timing-explainer.md` |
| device identity (vendor, architecture, model) | `webnn#934` |
| bytes and memory estimates | out of scope: allocator-dependent, fingerprinting-friendly |
| per-object enumeration, sizes | DevTools |

## User-Facing Problem

A page cannot answer:

- **How many resources is this context holding?** Nothing reports it.
- **Did `destroy()` actually release?** The page can only assume.
- **Is my model cache bounded?** It cannot see the footprint it is caching.

### Motivating scenario

- Face detector that rebuilds a graph per navigation.
- One `destroy()` missed → a leaked graph per navigation → no signal until the device OOMs.
- The same page caches two models and wants to bound the cache by count.
- Neither question is answerable today.

### Goals

- Count the graphs and tensors a context owns.
- Be deterministic for the release paths the spec defines (`destroy()`, context teardown).
- Be cheap: two integers, no allocation, no permission.
- Describe only the page's own objects.

### Non-goals

- Bytes — allocator-dependent and fingerprinting-friendly.
- Per-object enumeration or sizes — that is DevTools territory.
- Device identity — `webnn#934`.
- Detecting browser-internal leaks — this is page-facing accounting.
- Deterministic collection — GC is not observable on the web.

## User research

### Developer need

- A page that rebuilds a graph per navigation cannot see a missed `destroy()`; the leak is
  invisible until the device runs out of memory.
- A cache wants a bound: "keep at most N graphs" is not checkable today.
- Test harnesses cannot assert release without parsing implementation logs — our own
  `channel_leak` test had to do exactly that, which is the workaround this API removes.
- No formal user study has been run; the evidence is implementation experience (Servo's counters
  and channel test) and the platform gap.

### Prior art

| precedent | what it exposes | shape |
|---|---|---|
| **`AudioPlaybackStats`** (Web Audio 1.1 draft) | `underrunEvents`, `underrunDuration`, `totalDuration`, latency stats | **closest in kind**: dedicated stats object, own Risk/Mitigations section |
| WebCodecs `decodeQueueSize` / `encodeQueueSize` + `dequeue` | counts of pending work | raw `unsigned long` |
| WebRTC `bufferedAmount` + `bufferedamountlow` | buffered bytes, live | getter + level-triggered event |
| `navigator.storage.estimate()` | usage/quota | async, quantised |
| `performance.measureUserAgentSpecificMemory()` | JS heap usage | async, COI-gated, noisy |
| WebGPU `GPUBuffer.destroy()` | release semantics | not a count; the "destroy is deterministic" precedent |
| `navigator.hardwareConcurrency` | device parallelism | static count |
| **`AudioContext.renderCapacity`** | was shipped 2022 | **removed** — absent from the current spec; the cautionary tale |

- **`AudioPlaybackStats` is the closest precedent in kind**:
  - a dedicated object on `AudioContext`, not a scattered API;
  - its own normative privacy section (Risk: cross-site covert channel; Mitigations).
  - Mitigations: values update at most once per second; gated on `getUserMedia` permission or a
    fully active, visible document.
- **Verdict:** no platform API reports *live object counts*. The closest shapes are counts of
  pending work. Ours is first of its kind; the pitch is that WebNN is the first web API whose
  objects hold large native memory with both `destroy()` and a GC fallback.

## Implementation experience: Servo, across platforms

- Servo targets Linux, macOS, Windows, Android and OpenHarmony.
- Backend per target, one trait: CoreML (macOS), LiteRT + ONNX Runtime (Linux), ONNX Runtime
  (Windows), LiteRT + CANN (OpenHarmony).
- The implementation already keeps these counts to release resources:
  - an id → device-tensor map per context;
  - an id → compiled-graph map per context;
  - a context map in the channel thread.
- Servo's own `[webnn-leak]` probes and `channel_leak.rs` test exercise the same paths; they are
  internal and stay internal. They are the evidence that the counts return to baseline when
  `destroy()` is called.

## Sample code

```js
const context = await navigator.ml.createContext({ deviceType: "npu" });
const graph = await buildGraph(context);                  // graphs: 1
const input = await context.createTensor({ dataType: "float32", shape: [1, 3, 224, 224] });
const output = await context.createTensor({ dataType: "float32", shape: [1, 1000] });

console.log(context.resourceCounts);      // { graphs: 1, tensors: 2 }
output.destroy();
console.log(context.resourceCounts);      // { graphs: 1, tensors: 1 } — updated before destroy() returned
context.destroy();
console.log(context.resourceCounts);      // { graphs: 0, tensors: 0 }
```

## Proposed Approach

Two integers, owned by the page: the graphs and tensors a context holds. The home is a WebNN
context attribute — no Performance Timeline registration, no new global. The principle: explicit
release is deterministic, collection is explicitly out of contract.

The shape:

- A small dictionary of counts: `MLResourceCounts { graphs, tensors }`.
- One attribute on the owning object: `MLContext.resourceCounts`.
- Synchronous getter returning a snapshot.
- Counts only: no bytes, no ids, no per-object data.
- Covers the graph API; Model Loader models, if they carry resources, are an open question.

## Detailed Design

```webidl
dictionary MLResourceCounts {
  unsigned long graphs;
  unsigned long tensors;
};

partial interface MLContext {
  readonly attribute MLResourceCounts resourceCounts;
};
```

### Semantics

- **Normative sentence:** *"`destroy()` removes the resource from this count before the method
  returns."*
- `resourceCounts` is a snapshot; mutating it does not affect the context.
- `graphs` counts `MLGraph` objects owned by this context.
- `tensors` counts `MLTensor` objects owned by this context, including constant tensors.
- A resource counts from the moment its creation promise resolves — the moment the page can hold
  the object; a pending `build()` or `createTensor()` is not counted.
- Only script-visible objects count — not internal temporaries the implementation allocates.
- `MLContext.destroy()` sets both counts to zero.
- Unreachable-but-uncollected objects may still be counted; the count drops when the
  implementation collects them.
- The attribute is available on a live context; after `destroy()`, both counts are zero.

### Release rules

- `MLTensor.destroy()` — decrements `tensors` synchronously.
- `MLGraph.destroy()` — decrements `graphs` synchronously.
- `MLContext.destroy()` — both to zero.
- Repeated `destroy()` — a no-op; a count never goes below zero.
- GC collection — unspecified timing; the spec says only that collected objects stop counting.

### Dependencies on non-stable features

- **WebNN `destroy()` semantics** for `MLTensor`, `MLGraph`, `MLContext` — the API is defined on
  top of them.
- **Model Loader API** — if its model objects hold context resources, whether they appear in the
  counts is an open question.
- **`webnn#923` / `#934`** — no dependency; they concern compute policy.

### Integrating with the WebNN developer story

- No change to any existing method.
- No permission; no feature flag required by the spec.
- Two integers; no allocation on the page side.
- A page that never reads it sees no difference.

## Solving Goal 1: visibility

- The page can see the native footprint of its own context.
- Leaks become visible before the device runs out of memory.
- A cache can be bounded by count.

## Solving Goal 2: testability

- `destroy()` determinism gives WPT an assertable rule in every engine.
- GC paths are probed by harnesses, not promised by the web API.
- The same rule is what Servo's internal `channel_leak` test asserts today.

## Solving Goal 3: accounting without bytes

- Counts are comparable across allocators; bytes are not.
- Counts are the permissive side of the platform's privacy line (`hardwareConcurrency`,
  `decodeQueueSize` are exact; bytes get quantised).

## Alternatives considered

- **Bytes** — un-comparable across allocators; fingerprinting-friendly; benchmark-shaped.
  - Presenting bytes as counts is exactly the mistake `performance.memory` made.
- **Async estimate** (`storage.estimate()`-style) — hides synchronous determinism and adds noise;
  not needed for two integers.
- **Level-triggered event** (`bufferedamountlow`-style) — useful if the goal were backpressure;
  it is not. A getter is smaller and assertable.
- **Per-object enumeration** — a diagnostic API; DevTools, not the platform.
- **`measureUserAgentSpecificMemory()`** — JS heap only; misses native resources; COI-gated.
- **No API ("just call `destroy()`")** — cannot detect a missed call, and GC makes release
  unobservable.
- **A leak-check API on `destroy()`** — mixes a debug surface into the release path; the count is
  simpler and general.

## Alternatives: Previous API shape

- Original sketch (2026-10-09): `MLContext.liveResources`.
- Then `MLLiveResources` (dictionary).
- Now `MLResourceCounts` / `MLContext.resourceCounts`:
  - "Live" promises GC-liveness semantics the spec cannot define.
  - "Counts" matches the platform vocabulary (`decodeQueueSize`, `hardwareConcurrency`) and the
    counts-not-bytes rule.

## Accessibility, Internationalization, Privacy, and Security Considerations

### Accessibility and internationalization

- No UI, no focus, no live region.
- Data for scripting only; no localisation performed.

### Security considerations

- No new attack surface: reads bookkeeping the implementation already keeps.
- No parsing of untrusted input.
- No cross-origin or cross-process channel is introduced.

### Privacy considerations

- Per-context and page-owned: the counts describe only objects the page created.
- The page already knows what it created; the only new fact is whether the engine has released it.
- Contrast with `AudioPlaybackStats`: its covert channel came from a *shared* observable (the
  audio device). Here there is no shared observable; another tab's counts are not visible.
- Residual signal: GC timing — polling could reveal when the engine collected. The spec does not
  define collection timing, and the page can trigger release deterministically with `destroy()`.
- If reviewers demand more, the mitigation template exists (`AudioPlaybackStats`): rate-limit
  updates, or gate on visibility. Cost: it breaks synchronous `destroy()` determinism, so it would
  have to apply only to the collection-driven part — an open question.
- `AudioContext.renderCapacity` was removed after privacy review; its objection concerned two tabs
  observing the *same* transitions. That objection does not apply to a per-context count.

## Stakeholder feedback / opposition

- Servo: proposing this design.
- WebML WG: the first thread; the `destroy()` semantics are already specified there.
- Other implementers: no public positions yet; to be recorded here.

## Open questions

1. Counts of script-visible objects only — confirm.
2. Should constant tensors be counted separately from read/write tensors?
3. Model Loader: do model objects appear in the counts?
4. Getter, level-triggered event, or both?
5. Any rate limiting for the GC-driven component? (Costs determinism.)
6. Post-destroy reads: `resourceCounts` returns zeros on a destroyed context while other
   operations are unavailable. Should it throw for consistency instead?
7. Completeness: are two counts the complete minimum for a first draft, or does the group expect
   bytes, per-type breakdowns or queue depth before review?

## References & acknowledgements

### References

- WebNN spec: `MLContext`, `MLTensor.destroy()`, `MLGraph.destroy()`, `MLContext.destroy()`.
- Performance Timeline (W3C) — measures operations, not holdings; the gap this fills.
- `AudioPlaybackStats` — Web Audio API 1.1 draft (Risk/Mitigations, rate limit, permission gating).
- `AudioContext.renderCapacity` — removed after privacy review.
- WebCodecs `decodeQueueSize` / `encodeQueueSize` / `dequeue`.
- WebRTC `RTCDataChannel.bufferedAmount` / `bufferedamountlow`.
- `navigator.storage.estimate()`; `performance.measureUserAgentSpecificMemory()`.
- WebGPU `GPUBuffer.destroy()`.
- `navigator.hardwareConcurrency`.
- `webmachinelearning/model-loader` explainer.

### Acknowledgements

- Structure follows the Model Loader and Soft Navigations explainers.
- Reviewers and contributors to be listed as the design is discussed.
