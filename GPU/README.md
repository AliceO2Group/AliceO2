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

Build against the ROOT development package containing `TMVA/RGPUModel.hxx`:

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

Use GPU TPC cluster finding, `nnInferenceDevice=cuda` or `rocm` as appropriate,
`nnLoadFromCCDB=0`, and local classification/regression ONNX paths. Set both
`nnInferenceInputDType` and `nnInferenceOutputDType` to `FP32` or `FP16` to match
the model. Mixed input/output precision, unsupported graphs, CPU SOFIE inference
and CCDB model loading are rejected explicitly.

Each distinct model path is parsed and compiled once during chain initialization.
Each lane then gets a session bound to its existing native stream. O2 allocates
input, output, weights and intermediate storage through its registered clusterizer
memory. When O2 recycles the scratch arena, weights are uploaded again on the
lane's stream; this does not recompile the model. Sessions persist until chain
finalization, which synchronizes before unloading the generated libraries.
Changing models, batch capacity or backend requires reinitializing the chain.

The supplied classification/regression networks have six Gemm layers, five Relu
layers, 246 input values and respectively 1/5 output values. The clusterizer's
input-window and index-data settings must reproduce the training layout;
the default window does not have 246 values. `nnClusterizerBatchedMode` sets
the maximum batch; smaller final batches reuse the same code and workspace.

Validation to run after building: ROOT's `TestSofieGPU` and CUDA/HIP
`TestSofieGPUDevice`, then compare SOFIE against ORT using both supplied
precisions and multiple batch sizes. Check numerical tolerances and downstream
cluster decisions; the scalar dense kernels are not performance tuned.
No ROOT/O2 build or GPU test was run while implementing this change.
