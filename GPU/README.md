<!-- doxy
\page refGPU Module 'GPU'
/doxy -->

# GPU

<!-- doxy
This module contains the following submodules:

* \subpage refGPUTrackingStandalone
* \subpage refGPUTrackingDisplayFilterMacros
/doxy -->

## Experimental SOFIE NN clusterizer

Build against standalone SOFIE on its `alice-custom` branch, with the
`alpaka` and `sofieBLAS` dependencies supplied by alidist. A custom ROOT
SOFIE implementation is no longer required:

- `GPUCA_BUILD_SOFIE=ON` enables the SOFIE host integration.
- `GPUCA_BUILD_ORT=ON` (default) retains ORT; set it to `OFF` for a SOFIE-only
  GPUTracking backend. Other O2 components may still require ONNXRuntime.
- SOFIE is disabled by default. This change does not run a compiler at O2 build
  configuration time to generate models.

Select `ml-framework=SOFIE` (the `mlFramework` field in the NN settings), or
`ORT` to retain the existing path. For SOFIE, also set `sofie-architecture`
(`sofieArchitecture`) to the target GPU architecture, for example `sm_80` for
CUDA or `gfx90a` for HIP. `sofie-compiler` (`sofieCompiler`) optionally supplies
an explicit compiler path; otherwise `nvcc`/`hipcc` is found on PATH at runtime.
The compiler and GPU toolkit must be available on the worker at startup.
CUDA requires cuBLAS/cuBLASLt; HIP requires hipBLAS/hipBLASLt. Load the SOFIE
package environment so `SOFIE_ROOT`, `ALPAKA_ROOT`, and `SOFIEBLAS_ROOT` locate
the installed headers when models are compiled at runtime.

Use GPU TPC cluster finding, `nnInferenceDevice=cuda` or `rocm` as appropriate,
local classification/regression ONNX paths with `nnLoadFromCCDB=0`, or
`nnLoadFromCCDB=1` to parse the ONNX buffers fetched by the existing CCDB workflow.
CCDB model slots follow `nnEvalMode` (classification, regression 1, optional regression 2). Set both
`nnInferenceInputDType` and `nnInferenceOutputDType` to matching `FP32` or `FP16`
and use ONNX models of that precision. FP16 inputs, weights, intermediate tensors
and outputs stay in binary16, with FP32 GEMM/reduction accumulation. Dense models
use hipBLAS/rocBLAS on HIP, including gfx906, without requiring hipBLASLt kernels. Unsupported graphs and CPU SOFIE inference fail explicitly.

Each distinct local model path is parsed and compiled once during chain initialization.
For CCDB, initialization is deferred until the first event has loaded its calibration
objects, before NN inference. Identical model buffers share a compiled program.
Later events reuse the sessions while the model bytes are unchanged. When CCDB
provides changed bytes, only new model contents are parsed and compiled, sharing
unchanged programs across lanes. All replacement sessions are prepared and
validated before synchronizing GPU work and replacing the old sessions. A failed
reload propagates an error without replacing the active sessions. O2 sizes the
workspace for the new models before inference.
Missing or empty required buffers fail explicitly. No temporary ONNX files are needed.
Each lane then gets a session bound to its existing native stream. O2 allocates
input, output and intermediate tensor storage through its registered clusterizer
memory. When O2 recycles the scratch arena, sessions rebind non-owning views.
Weights, persistent state, shape metadata and BLAS resources remain session-owned;
they are not uploaded again when scratch memory changes. Sessions persist until a model
reload or chain finalization, with GPU synchronization before unloading their libraries.
Changing local model paths, batch capacity, precision or backend requires reinitializing the chain.

Model layouts vary between exports, including dense and attention/preprocessing
graphs. Check the deployed model input/output dimensions rather than assuming a
particular layer count. The clusterizer's
input-window and index-data settings must reproduce the training layout;
the default window does not have 246 values. `nnClusterizerBatchedMode` sets
the maximum batch; smaller final batches reuse the same code and workspace.

Validation to run after building: standalone SOFIE's `SOFIEWorkspace` and
`SOFIERuntimeWorkspace` tests (see `SOFIE/core/README.md`), then compare SOFIE
against ORT using FP32 and FP16 models and multiple batch sizes. Check numerical
tolerances and downstream cluster decisions. Inference uses SOFIE's operator
implementations and Alpaka/BLAS backends. For CCDB,
check identical bytes in a new buffer, one changed model, changed hidden-layer
sizes, and invalid replacement data. Only changed contents should compile; an
invalid replacement must fail before switching sessions.
No ROOT/O2/SOFIE build or generated-model GPU test was run while implementing
this change. Small direct hipBLAS capability probes are not end-to-end validation.
