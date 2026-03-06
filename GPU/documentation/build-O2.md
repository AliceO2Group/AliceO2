This ticket will serve as documentation how to enable which GPU features and collect related issues.

So far, the following features exist:
 * GPU Tracking with CUDA
 * GPU Tracking with HIP
 * GPU Tracking with OpenCL (>= 2.1)
 * OpenGL visualization of the tracking
 * ITS GPU tracking

GPU support should be detected and enabled automatically.
If you just want to reproduce the GPU build locally without running it, it might be easiest to use the GPU CI container (see below).
The provisioning script of the container also demonstrates which patches need to be applied such that everything works correctly.

In a nutshell, all is steered via CMake variables, and the `ALIBUILD_O2_FORCE_GPU...` environment variables exist to steer what [alidist/o2.sh](https://github.com/alisw/alidist/blob/master/o2.sh) puts as CMake defaults.
We try to run the same CMake GPU detection as in O2 ([FindO2GPU.cmake](https://github.com/AliceO2Group/AliceO2/blob/dev/dependencies/FindO2GPU.cmake)) during the aliBuild `prefer_system_check` ([gpu-system.sh](https://github.com/alisw/alidist/blob/master/gpu-system.sh)), such that all GPU features / versions / architectures can become part of the `gpu-system` version, which avoid inconsistencies between different packages we build.

All is steered via environment variables, which will go into the version and thus the hash:
- `ALIBUILD_O2_FORCE_GPU=...` sets the mode
- `ALIBUILD_O2_FORCE_GPU_CUDA=1` can force-enable (`=1`) or disable (`=0`) backends, even if they were not detected. Same for `..._HIP` and `..._OPENCL`.
- `ALIBUILD_O2_FORCE_GPU_CUDA_ARCH=...` can override the architecture to cross-compile, e.g. `ALIBUILD_O2_FORCE_GPU_CUDA_ARCH="86;89"`. Same for `..._HIP_ARCH`.

Modes for `ALIBUILD_O2_FORCE_GPU`
- `force` / `1` / `ci`: Force that all backends / features are detected, fail if not. GPU architectures are set to the default ones if not specified by environment variables.
CI is currently identical to force, but should allow special behavior when running in the CI.
- `auto`: check for supported system-cmake version, fail if not found. Auto-detect GPU backends / features and architectures. Selected features can be force-enabled on top via env variable. But not selectively disabled. (But one can use the manual mode below.)
- `onthefly`: Don't detect GPUs at alidist levels. gPUs disabled in ONNX. GPUs auto-detcted in O2 CMake during build as before, but this means the O2 build hash does not depend on GPU features, so we also have the same problems as before. This is just a fallback, to allow users to build with GPUs if they don't have a compatible system CMake.
- `fullauto`: Detect supported system-cmake. If found, behave as Auto. If not found behave as OnTheFly.
- `disabled`: Disable all GPU builds. No extra time during aliBuild command.
- `manual`: all GPU builds disabled by default, to  be enabled manually via env variable. No extra time during aliBuild command.

*Additional reasoning for this approach*
Advantages:
- O2 hash and ONNX hash depend on available GPU backends, detected features (like tensorrt for ML) and on the detected GPU architectures and librar versions. I.e. when you plug in a new GPU or update the CUDA version, the O2 hash will change and this will trigger a rebuild. Otherwise, the build could just fail due to stale settings in CMakeCache.
- We can have binary tarballs depending on the enabled backends.
- O2 and ONNX are always in sync.
- Same detection during aliBuild as in O2 CMake.
- One can see enabled GPU features / versions / architectures in the version string of `gpu-system`.

Disadvantages:
- Need system `CMake` >= `3.26` for the detection at aliBuild level.
- `FindO2GPU.cmake` is duplicated in O2 and alidist and must be kept in sync. But at least this is checked and gives an error otherwise.
- Running cmake during the system check takes around 5 sec for every aliBuild command involving O2 or ONNX.

*GPU Tracking with CUDA*
 * The CMake option `-DENABLE_CUDA=ON/OFF/AUTO` steers whether CUDA is forced enabled / unconditionally disabled / auto-detected.
 * The CMake option `-DCUDA_COMPUTETARGET=...` fixes a GPU target, e.g. 61 for PASCAL or 75 for Turing (if unset, it compiles for the lowest supported architecture)
 * CUDA is detected via the CMake language feature, so essentially nvcc must be in the Path.
 * We require CUDA version >= 12.8
 * CMake will report "Building GPUTracking with CUDA support" when enabled.

*GPU Tracking with HIP*
 * HIP must be installed, and CMake must be able to detect HIP via find_package(hip) and enable language(hip).
 * For the minimum ROCm / HIP version, please check [FindO2GPU.cmake](https://github.com/AliceO2Group/AliceO2/blob/dev/dependencies/FindO2GPU.cmake#L287).
 * The CMake option `-DHIP_AMDGPUTARGET=...` / env variable `ALIBUILD_O2_FORCE_GPU_HIP_ARCH=...` forces a GPU target, e.g. gfx906 for MI50 (if unset, it auto-detects the GPU).
 * CMake will report "Building GPUTracking with HIP support" when enabled.
 * It may be that some patches must be applied to ROCm after the installation. You find the details in the provisioning script of the GPU CI container below.

*GPU Tracking with OpenCL (Needs Clang >= 18 for compilation)*
 * Needs OpenCL library with version >= 2.1, detectable via CMake find_package(OpenCL).
 * Needs the SPIR-V LLVM translator together with LLVM to create the SPIR-V binaries, also detectable via CMake.

*OpenGL visualization of TPC tracking*
 * Needs the following libraries (all detectable via CMake find_package): libOpenGL, libGLEW or libGLFW (default), libGLU.
 * OpenGL must be at least version 4.5, but this is not detectable at CMake time. If the supported OpenGL version is below, the display is not/partially built, and not available at runtime. (Whether it is not or partially built depends on whether the maximum OpenGL version supported by GLEW or that of the system runtime in insufficient.)
 * Note: If ROOT does not detect the system GLEW library, ROOT will install its own very outdated GLEW library, which will be insufficient for the display. Since the ROOT include path will come first in the order, this will prevent the display from being built.
 * CMake will report "Building GPU Event Display" when enabled.

*Vulkan visualization*
 * similar to OpenCL visualization, but with Vulkan.

*ITS GPU Tracking*
 * So far supports only CUDA and HIP, support for OpenCL might come.
 * The build is enabled when the "GPU Tracking with CUDA" (as explained above) detects CUDA, same for HIP.
 * CMake will report "Building ITS CUDA tracker" when enabled, same for HIP.

*Using the GPU CI container*
 * Setting up everything locally might be somewhat time-consuming, instead you can use the GPU CI cdocker container.
 * The docker images is `alisw/slc9-gpu-builder`.
 * The container exports the `ALIBUILD_O2_FORCE_GPU=1` env variable, which force-enables all GPU builds.
 * Note that it might not be possible out-of-the-box to run the GPU version from within the container. In case of HIP it should work when you forwards the necessary GPU devices in the container. For CUDA however, you would either need to (in addition to device forwarding) match the system CUDA driver and toolkit installation to the files present in the container, or you need to use the CUDA docker runtime, which is currently not installed in the container.
 * There are currently some patches needed to install all the GPU backends in a proper way and together. Please refer to the container provisioning script [provision.sh](https://github.com/alisw/docks/blob/master/slc9-gpu-builder/provision.sh). If you want to reproduce the installation locally, it is recommended to follow the steps from the script.

*Summary*

If you want to enforce the GPU builds on a system without GPU, please export the following environment variables:
 * `ALIBUILD_O2_FORCE_GPU_CUDA=ON`
 * `ALIBUILD_O2_FORCE_GPU_HIP=ON`
 * `ALIBUILD_O2_FORCE_GPU_OPENCL=ON`
 * `ALIBUILD_O2_FORCE_GPU_CUDA_ARCH=default`
 * `ALIBUILD_O2_FORCE_GPU_HIP_ARCH=default`
