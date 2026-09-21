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

// --- Project headers ---------------------------------------------------------
#include "GPUCommonDef.h"
#include "GPUCommonTypeTraits.h"
#include "GPUCommonArray.h"

// The remaining headers do not compile as MSL yet, but nothing structural is in
// the way: with the pragma above and the untyped constant buffer below, the
// kernel list expands to all 104 entry points, with no derived-class and no
// kernel-argument-type errors left. What fails is the bodies, and it is bulk
// work rather than a missing language feature -- MSL has no double, and every
// namespace-scope constexpr needs GPUglobalconstexpr().
#if 0
#include "GPUConstantMem.h"
#include "GPUReconstructionIncludesDeviceAll.h"
#endif

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

// Every kernel parameter needs an attribute, so the sector index arrives as a
// buffer rather than by value, and the grid dimensions come in at the end, where
// GPUCommonDefAPI.h's get_group_id() and friends pick them up.
#define GPUCA_KRNL_SECTOR_ARG constant int32_t& _iSector_internal [[buffer(2)]]
#define GPUCA_KRNL_GRID_ARGS \
  , uint _metalTgIg   [[threadgroup_position_in_grid]] \
  , uint _metalTiTg   [[thread_position_in_threadgroup]] \
  , uint _metalTPerTg [[threads_per_threadgroup]] \
  , uint _metalTgPerG [[threadgroups_per_grid]]

// Include the actual kernels, once the headers above compile as MSL.
#if 0
#include "GPUReconstructionKernelList.h"
#endif

// clang-format on
#pragma clang diagnostic pop
