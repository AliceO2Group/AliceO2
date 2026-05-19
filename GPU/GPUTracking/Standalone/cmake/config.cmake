# Copyright 2019-2020 CERN and copyright holders of ALICE O2.
# See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
# All rights not expressly granted are reserved.
#
# This software is distributed under the terms of the GNU General Public
# License v3 (GPL Version 3), copied verbatim in the file "COPYING".
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization
# or submit itself to any jurisdiction.

# This is the configuration file for the standalone build
# Its options do not affect the O2 build !!!!

set(ENABLE_CUDA AUTO)
set(ENABLE_HIP AUTO)
set(ENABLE_OPENCL AUTO)
set(GPUCA_CONFIG_VC 1)
set(GPUCA_CONFIG_FMT 1)
set(GPUCA_CONFIG_ROOT 1)
set(GPUCA_CONFIG_ONNX 0)
set(GPUCA_BUILD_EVENT_DISPLAY 1)              # Enable compilation of event display
set(GPUCA_BUILD_EVENT_DISPLAY_FREETYPE 1)     # Use FreeType library to render fonts for event display
set(GPUCA_BUILD_EVENT_DISPLAY_VULKAN 1)       # Enable Vulkan backend for event display (otherwise only OpenGL / Win32)
set(GPUCA_BUILD_EVENT_DISPLAY_WAYLAND 1)      # Enable native wayland frontend for event display
set(GPUCA_BUILD_EVENT_DISPLAY_QT 1)           # Use QT for Event Display GUI
set(GPUCA_CONFIG_GL3W 0)                      # Use GL3W instead of glew
set(GPUCA_CONFIG_O2 1)                        # Compile for O2 data, 0 for Run 2 data
set(GPUCA_BUILD_DEBUG 0)                      # Enable debug mode (-O0, -ggdb, enable asserts)
set(GPUCA_BUILD_DEBUG_SANITIZE 0)             # Enable undefined behavior and address sanitizers
set(GPUCA_BUILD_DEBUG_HOSTONLY 0)             # Only compile host code in debug mode, GPU code compiled normally
set(GPUCA_DETERMINISTIC_MODE 0)               # OFF / NO_FAST_MATH / OPTO2 / GPU / WHOLEO2
set(GPUCA_DETERMINISTIC_NO_FTZ 0)             # If 1 and deterministic mode active, do not apply flush denormals to zero
#set(GPUCA_CUDA_GCCBIN c++-14)                # Override which GCC to use for CUDA
#set(GPUCA_OPENCL_CLANGBIN clang-20)          # Override which clang to use for OpenCL
set(HIP_AMDGPUTARGET "default")               # Set AMD GPU tragets to compile for: e.g. "gfx906;gfx908;gfx90a"
set(CUDA_COMPUTETARGET "default")             # Set NVIDIA GPU targets to compile for: e.g. "89;120"
#set(GPUCA_CUDA_COMPILE_MODE perkernel)       # Mode to compile kernels for CUDA: onefile / perkernel / rtc
#set(GPUCA_HIP_COMPILE_MODE perkernel)        # Mode to compile kernels for HIP:  onefile / perkernel / rtc
#set(GPUCA_RTC_NO_COMPILED_KERNELS 1)         # Do not compile "perkernel" kernels at compile time, support only RTC
#set(GPUCA_KERNEL_RESOURCE_USAGE_VERBOSE 1)   # Verbose resource usage output during kernel compilation
#set(GPUCA_CONFIG_COMPILER gcc)               # Compiler to use for standalone compilation: gcc / clang
#set(GPUCA_CONFIG_WERROR 1)                   # Enforce Werror
#add_definitions(-DGPUCA_GPU_DEBUG_PRINT)     # Enable LOG(...) macros and GPUInfo(...) etc. in GPU code
#set(GPUCA_OVERRIDE_PARAMETER_FILE "foo.csv") # Override the CSV or JSON file that contains GPU parameters
