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

/// \file GPUReconstructionMETAL.metal

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
// clang-format off

// --- Backend selection -------------------------------------------------------
#define GPUCA_GPUTYPE_METAL 1

// --- Metal stdlib ------------------------------------------------------------
#include <metal_stdlib>
// MSL rejects derived classes outside of this pragma, and the kernels are
// class-based throughout. metal_stdlib itself uses it in 46 paired places.
#pragma METAL internals : enable
using namespace metal;

// --- OpenCL compatibility shims ---------------------------------------------

// Address space aliases (match OpenCL vernacular used by the project); constant
// is spelled the same in MSL
#define global   device
#define local    threadgroup

#ifndef M_PI
#define M_PI 3.1415926535f
#endif

// Disable assertions inside GPU code (same as OpenCL variant)
#ifdef assert
# undef assert
#endif
#define assert(param)

// --- double ------------------------------------------------------------------
// MSL has no double. GPUdoubleBinary64 is IEEE-754 binary64 in software, with the
// same eight bytes in the same order, so the keyword can simply name it and the
// shared code needs no separate spelling. Must come after metal_stdlib, which
// uses the token itself.
#include "GPUCommonDoubleBinary64.h"
#define double o2::gpu::GPUdoubleBinary64
#include "GPUCommonDouble.h"

// --- Project headers ---------------------------------------------------------
#include "GPUCommonDef.h"
#include "GPUCommonTypeTraits.h"
#include "GPUCommonArray.h"

#include "GPUConstantMem.h"
#include "GPUReconstructionIncludesDeviceAll.h"

// --- Kernel list expansion ---------------------------------------------------
#define GPUCA_KRNL(...) GPUCA_KRNLGPU(__VA_ARGS__)

// --- Constant memory + global heap plumbing ---------------------------------
// The heap and the constant memory arrive as buffer(0) and buffer(1). The latter
// is untyped because a buffer of GPUConstantMem, which has base classes, is not
// a valid kernel argument type.
#define GPUCA_CONSMEM_PTR \
  device char* gpu_mem        [[buffer(0)]], \
  device char* pConstantRaw   [[buffer(1)]],
#define GPUCA_CONSMEM (*(device GPUConstantMem*)pConstantRaw)

#include "GPUReconstructionKernelList.h"

// clang-format on
#pragma clang diagnostic pop
