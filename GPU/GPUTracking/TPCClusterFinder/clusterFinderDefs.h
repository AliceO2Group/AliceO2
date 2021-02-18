// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file clusterFinderDefs.h
/// \author David Rohr

#ifndef O2_GPU_CLUSTERFINDERDEFS_H
#define O2_GPU_CLUSTERFINDERDEFS_H

#include "GPUDef.h"

#ifndef __OPENCL__
using uchar = unsigned char;
#endif
#ifdef __APPLE__
using ulong = unsigned long;
#endif

#define QMAX_CUTOFF 3
#define QTOT_CUTOFF 0
#define NOISE_SUPPRESSION_MINIMA_EPSILON 10
#define SCRATCH_PAD_WORK_GROUP_SIZE GPUCA_GET_THREAD_COUNT(GPUCA_LB_CLUSTER_FINDER)

#ifdef GPUCA_GPUCODE
/* #define BUILD_CLUSTER_NAIVE */
#define BUILD_CLUSTER_SCRATCH_PAD
#else
/* #define BUILD_CLUSTER_NAIVE */
#define BUILD_CLUSTER_SCRATCH_PAD
#endif
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

#define PADDING_PAD 2
#define PADDING_TIME 3
#define TPC_SECTORS 36
#define TPC_ROWS_PER_CRU 18
#define TPC_NUM_OF_ROWS 152
#define TPC_PADS_PER_ROW 138
#define TPC_PADS_PER_ROW_PADDED (TPC_PADS_PER_ROW + PADDING_PAD)
#define TPC_NUM_OF_PADS (TPC_NUM_OF_ROWS * TPC_PADS_PER_ROW_PADDED + PADDING_PAD)
#define TPC_MAX_FRAGMENT_LEN 4000
#define TPC_MAX_FRAGMENT_LEN_PADDED (TPC_MAX_FRAGMENT_LEN + 2 * PADDING_TIME)

#if 0
#define DBG_PRINT(msg, ...) printf(msg "\n", __VA_ARGS__)
#else
#define DBG_PRINT(msg, ...) static_cast<void>(0)
#endif

#ifdef GPUCA_GPUCODE
#define CPU_ONLY(x) static_cast<void>(0)
#define CPU_PTR(x) nullptr
#else
#define CPU_ONLY(x) x
#define CPU_PTR(x) x
#endif

namespace GPUCA_NAMESPACE
{
namespace gpu
{
namespace tpccf
{

using SizeT = uint;
using TPCTime = int;
using TPCFragmentTime = short;
using Pad = unsigned char;
using GlobalPad = short;
using Row = unsigned char;
using Cru = unsigned char;

using Charge = float;

using Delta = short;
using Delta2 = short2;

using local_id = short2;

GPUconstexpr() float CHARGE_THRESHOLD = 0.f;
GPUconstexpr() float OUTER_CHARGE_THRESHOLD = 0.f;
GPUconstexpr() float QTOT_THRESHOLD = 500.f;
GPUconstexpr() int MIN_SPLIT_NUM = 1;

} // namespace tpccf
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
