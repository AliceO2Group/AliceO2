// Copyright 2019-2025 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUReconstructionMetal.metal

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
// clang-format off

// --- Backend selection -------------------------------------------------------
#define GPUCA_GPUTYPE_METAL 1

// --- Metal stdlib ------------------------------------------------------------
#include <metal_stdlib>
using namespace metal;

// --- OpenCL compatibility shims ---------------------------------------------

// Address space aliases (match OpenCL vernacular used by the project)
#define global   device
#define local    threadgroup
#define constant constant

#ifndef M_PI
#define M_PI 3.1415926535f
#endif

// Disable assertions inside GPU code (same as OpenCL variant)
#ifdef assert
# undef assert
#endif
#define assert(param)

// --- Project headers ---------------------------------------------------------
#if false
#include "GPUCommonDef.h"
#include "GPUCommonTypeTraits.h" // (MSL can't include system C headers inside kernels; these should be GPU-safe)
#include "GPUCommonArray.h"
#include "GPUConstantMem.h"
// FIXME: We need a solution for the generic memory
#include "GPUReconstructionIncludesDeviceAll.h"
#endif

// --- Kernel list expansion ---------------------------------------------------
#define GPUCA_KRNL(...) GPUCA_KRNLGPU(__VA_ARGS__)

// --- Constant memory + global heap plumbing ---------------------------------
// In OpenCL, the kernels used:
//   GPUglobal() char *gpu_mem, GPUconstant() GPUConstantMem* pConstant,
// For Metal we bind them to buffer(0) and buffer(1) respectively.
// NOTE: Metal prefers references for constant buffers; keep a reference here.
#define GPUCA_CONSMEM_PTR \
  device char* gpu_mem               [[buffer(0)]], \
  constant GPUConstantMem& pConstant [[buffer(1)]],
#define GPUCA_CONSMEM (pConstant)

// If your code uses barriers like barrier(CLK_LOCAL_MEM_FENCE) via macros,
// you likely already map them in GPUReconstructionIncludesDeviceAll.h for each backend.
// If not, uncomment the following generic mapping:
// #define barrier(flags) threadgroup_barrier(mem_flags::mem_threadgroup)

// Include the actual kernels
// FIXME: disabled for now. We need to find a sustainable solution to 
// the missing __generic in Metal.
#if 0
#include "GPUReconstructionKernelList.h"
#endif

// Clean up local macro namespace if desired
// #undef GPUCA_KRNL
// #undef GPUCA_CONSMEM_PTR
// #undef GPUCA_CONSMEM
// #undef global
// #undef local
// #undef constant
// #undef private

// clang-format on
#pragma clang diagnostic pop
