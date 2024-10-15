// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file clusterFinderDefs.h
/// \author David Rohr

#ifndef O2_GPU_CLUSTERFINDERDEFS_H
#define O2_GPU_CLUSTERFINDERDEFS_H

#include "GPUDef.h"

/* #define CHARGEMAP_TIME_MAJOR_LAYOUT */
#define CHARGEMAP_TILING_LAYOUT

#define SCRATCH_PAD_SEARCH_N 8
#define SCRATCH_PAD_COUNT_N 16
#if defined(GPUCA_GPUCODE)
#define SCRATCH_PAD_BUILD_N 8
#define SCRATCH_PAD_NOISE_N 8
#else
// Double shared memory on cpu as we can't reuse the memory from other threads
#define SCRATCH_PAD_BUILD_N 16
#define SCRATCH_PAD_NOISE_N 16
#endif

// Padding of 2 and 3 respectively would be enough. But this ensures that
// rows are always aligned along cache lines. Likewise for TPC_PADS_PER_ROW.
#define GPUCF_PADDING_PAD 8
#define GPUCF_PADDING_TIME 4
#define TPC_PADS_PER_ROW 144

#define TPC_ROWS_PER_CRU 18
#define TPC_PADS_PER_ROW_PADDED (TPC_PADS_PER_ROW + GPUCF_PADDING_PAD)
#define TPC_NUM_OF_PADS (GPUCA_ROW_COUNT * TPC_PADS_PER_ROW_PADDED + GPUCF_PADDING_PAD)
#define TPC_PADS_IN_SECTOR 14560
#define TPC_FEC_IDS_IN_SECTOR 23296
#define TPC_MAX_FRAGMENT_LEN_GPU 4000
#define TPC_MAX_FRAGMENT_LEN_HOST 1000
#define TPC_MAX_FRAGMENT_LEN_PADDED(size) ((size) + 2 * GPUCF_PADDING_TIME)

#ifdef GPUCA_GPUCODE
#define CPU_ONLY(x)
#define CPU_PTR(x) nullptr
#else
#define CPU_ONLY(x) x
#define CPU_PTR(x) x
#endif

namespace GPUCA_NAMESPACE::gpu::tpccf
{

using SizeT = size_t;
using TPCTime = int32_t;
using TPCFragmentTime = int16_t;
using Pad = uint8_t;
using GlobalPad = int16_t;
using Row = uint8_t;
using Cru = uint8_t;

using Charge = float;

using Delta = int16_t;
using Delta2 = short2;

using local_id = short2;

} // namespace GPUCA_NAMESPACE::gpu::tpccf

#endif
