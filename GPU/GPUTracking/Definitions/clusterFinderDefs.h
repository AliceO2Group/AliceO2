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

#include "GPUCommonDef.h"
#include "GPUDef.h"
#include "DataFormatsTPC/Constants.h"

namespace o2::gpu::tpccf
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

static GPUglobalconstexpr() int32_t CHARGEMAP_TILING_LAYOUT = 1; // Layout of the charge map: 1 = tiling layout, 0 = linear layout

static GPUglobalconstexpr() uint32_t SCRATCH_PAD_SEARCH_N = 8;
static GPUglobalconstexpr() uint32_t SCRATCH_PAD_COUNT_N = 16;
#if defined(GPUCA_GPUCODE) && !defined(GPUCA_DEVICE_SINGLE_THREAD)
static GPUglobalconstexpr() uint32_t SCRATCH_PAD_BUILD_N = 8;
static GPUglobalconstexpr() uint32_t SCRATCH_PAD_NOISE_N = 8;
#else
// Double shared memory on cpu as we can't reuse the memory from other threads
static GPUglobalconstexpr() uint32_t SCRATCH_PAD_BUILD_N = 16;
static GPUglobalconstexpr() uint32_t SCRATCH_PAD_NOISE_N = 16;
#endif

// Padding of 2 and 3 respectively would be enough. But this ensures that
// rows are always aligned along cache lines. Likewise for TPC_CLUSTERER_ROW_PAD_CAPACITY.
static GPUglobalconstexpr() int32_t GPUCF_PADDING_PAD = 8;
static GPUglobalconstexpr() int32_t GPUCF_PADDING_TIME = 4;
// Largest possible number of pads in a TPC row
static GPUglobalconstexpr() int32_t TPC_CLUSTERER_ROW_PAD_CAPACITY = 144;

// Stride between rows as stored internally by the clusterizer
static GPUglobalconstexpr() int32_t TPC_CLUSTERER_ROW_STRIDE = TPC_CLUSTERER_ROW_PAD_CAPACITY + GPUCF_PADDING_PAD;
// Number of pads in a sector as stored internally by the clusterizer.
// This includes fake pads for constant strides between rows
static GPUglobalconstexpr() int32_t TPC_CLUSTERER_STRIDED_PAD_COUNT = o2::tpc::constants::MAXGLOBALPADROW * TPC_CLUSTERER_ROW_STRIDE + GPUCF_PADDING_PAD;
// Real of number of pads in a sector
static GPUglobalconstexpr() int32_t TPC_REAL_PADS_IN_SECTOR = 14560;
static GPUglobalconstexpr() int32_t TPC_FEC_IDS_IN_SECTOR = 23296;
static GPUglobalconstexpr() int32_t TPC_MAX_FRAGMENT_LEN_GPU = 4000;
static GPUglobalconstexpr() int32_t TPC_MAX_FRAGMENT_LEN_HOST = 1000;
template <typename T>
constexpr GPUhdi() T TPC_MAX_FRAGMENT_LEN_PADDED(T size)
{
  return size + 2 * GPUCF_PADDING_TIME;
}

} // namespace o2::gpu::tpccf

#endif
