The TPC tracking code is not fully deterministic, i.e. running multiple times on the same data set might yield a slightly different number of tracks on the O(per mille) level.
- This comes from concurrency, i.e. when tracks are processed in parallel, the output order might change, which might have small effects on the consecutive steps.
- Also compile options and optimizations play a row, e.g. using ffast-math or fused-multiply-add might slightly change the rounding of floating point, and in rare cases lead to the acceptance or rejection of a track, and thus a different number of tracks.

For debugging, testing, and validation, a deterministic mode is implemented, which should yield 100% reproducible results, on CPU and on GPU and when running multiple times.
It uses a combination of
- Compile time options, e.g. disabling all optimizations that change floating point rounding.
- Run time options, e.g. to use deterministic sorting, and add additional sorting steps after kernels to make the output deterministic, also intermediate outputs.

This is steered by 3 options:
- The `-DGPUCA_DETERMINISTIC_MODE` Cmake setting : Compile-time setting.
- The `--PROCdeterministicGPUReconstruction` command line option / `GPU_proc.deterministicGPUReconstruction` `--configKeyValue` setting : Run time setting.
- The `--RTCdeterministic` command line option / `GPU_proc_rtc.deterministic` `--configKeyValue` setting. (Auto-enabled by the `deterministicGPUReconstruction` setting.) : Compile-time setting for RTC code.

Note that enabling a single setting will not result in fully deterministic behavior! Each setting enables different deterministic aspects!
In order to be fully deterministic, all settings must be enabled, where the RTC setting is automatically enabled if not explicitly disabled.

`GPUCA_DETERMINISTIC_MODE` has multiple levels, which are described here: [FindO2GPU.cmake](https://github.com/AliceO2Group/AliceO2/blob/80a80a17f5a1d9cb77743e2a39b15b653fe1a4f9/dependencies/FindO2GPU.cmake#L72).
- In order to have fully deterministic GPUReconstruction (i.e. all algorithms that come with the GPUTracking library, like TPC tracking), the level `GPUCA_DETERMINISTIC_MODE=GPU` is needed.
- In order to apply it to all of O2, e.g. for ITS tracking, please use `GPUCA_DETERMINISTIC_MODE=WHOLEO2`

Enabling the options is a bit different for O2 and for the standalone benchmark:
- For enabling it in the standalone benchmark, please set GPUCA_DETERMINISTIC_MODE=GPU in [config.cmake](https://github.com/AliceO2Group/AliceO2/blob/dev/GPU/GPUTracking/Standalone/cmake/config.cmake) and use the command line argument `--PROCdeterministicGPUReconstruction 1`.
- For O2, Either add `set(GPUCA_DETERMINISTIC_MODE GPU)` to the beginning of the [GPU CMakeLists.txt](https://github.com/AliceO2Group/AliceO2/blob/dev/GPU/CMakeLists.txt) or add `set(GPUCA_DETERMINISTIC_MODE WHOLEO2)` to the beginning of the [Global CMakeLists.txt](https://github.com/AliceO2Group/AliceO2/blob/dev/CMakeLists.txt), and use the `configKeyValue` `GPU_proc.deterministicGPUReconstruction`. In order to enable this for the Full-System-Test or with [dpl-workflow.sh](https://github.com/AliceO2Group/AliceO2/blob/dev/prodtests/full-system-test/dpl-workflow.sh), please export `CONFIG_EXTRA_PROCESS_o2_gpu_reco_workflow=GPU_proc.deterministicGPUReconstruction=1;`.

With these settings, if one runs multiple times, the number of clusters and number of tracks should be always fully identical.
Note that this yields a significant performance penalty during the processing, therefore the deterministic mode is not compiled in by default, but it must be enabled explicitly and code must be recompiled.

Beyond comparing only the number of clusters and number of tracks, it is also possible to compare intermediate results. To do so, please use the standalone benchmark (either `./ca` or `o2-gpu-standalone-benchmark` binary) with the `--debug 6` option.
It will create a dump container all (most) intermediate results in text form, which can be compared. The output files is called `CPU.out` if using the CPU backend, and `GPU.out` for the GPU backend.
Note that the dump files will be huge and the processing will be slow and consume much more memory than normal with `--debug 6 . It has been tested with datasets containing up to 50 Pb-Pb collisions, and might fail for larger data.
The dump files (if the deterministic mode is used with both compile- and runtime-activation), the files should be 100% identical and can just be compared with `diff`.

By default, the deterministic mode will apply flush-to-zero and denormals-are-zero to denormal floats.
This can be disabled bia `-DDGPUCA_DETERMINISTIC_MODE`.
Note that some GPUs cannot do precise float computation with denormals flushed to zero, while other GPUs do not support denormals at all.
Thus, comparison between CPU and GPU deterministic results might require that this setting is either set or not set.
CPU results for the 2 cases will always differ, since the floating point math will be slightly different.
