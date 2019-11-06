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

#ifndef __OPENCL__
typedef unsigned char uchar;
#endif
#ifdef __APPLE__
typedef unsigned long ulong;
#endif

#ifdef GPUCA_ALIGPUCODE
#define QMAX_CUTOFF 3
#define QTOT_CUTOFF 0
#define NOISE_SUPPRESSION_MINIMA_EPSILON 10
#define SCRATCH_PAD_WORK_GROUP_SIZE GPUCA_THREAD_COUNT_CLUSTERER
#define BUILD_CLUSTER_NAIVE
#ifdef GPUCA_GPUCODE
#define BUILD_CLUSTER_SCRATCH_PAD
#endif
#define DCHARGEMAP_TIME_MAJOR_LAYOUT
#endif

#include "GPUDef.h"
#include "config.h"

#include "shared/ClusterNative.h"
#include "shared/constants.h"
#include "shared/tpc.h"

#define GET_IS_PEAK(val) (val & 0x01)
#define GET_IS_ABOVE_THRESHOLD(val) (val >> 1)


namespace GPUCA_NAMESPACE
{
namespace gpu
{

using PackedCharge = ushort;

struct ClusterAccumulator {
  Charge Q;
  Charge padMean;
  Charge padSigma;
  Charge timeMean;
  Charge timeSigma;
  uchar splitInTime;
  uchar splitInPad;
};

using Delta = short;
using Delta2 = short2;

struct ChargePos {
  GlobalPad gpad;
  Timestamp time;
};

typedef short2 local_id;

struct search_t {
  ChargePos posBcast[SCRATCH_PAD_WORK_GROUP_SIZE];
  PackedCharge buf[SCRATCH_PAD_WORK_GROUP_SIZE * SCRATCH_PAD_SEARCH_N];
};
struct noise_t {
  ChargePos posBcast[SCRATCH_PAD_WORK_GROUP_SIZE];
  PackedCharge buf[SCRATCH_PAD_WORK_GROUP_SIZE * SCRATCH_PAD_NOISE_N];
};
struct count_t {
  ChargePos posBcast1[SCRATCH_PAD_WORK_GROUP_SIZE];
  uchar aboveThresholdBcast[SCRATCH_PAD_WORK_GROUP_SIZE];
  uchar buf[SCRATCH_PAD_WORK_GROUP_SIZE * SCRATCH_PAD_COUNT_N];
};
struct build_t {
  ChargePos posBcast[SCRATCH_PAD_WORK_GROUP_SIZE];
  PackedCharge buf[SCRATCH_PAD_WORK_GROUP_SIZE * SCRATCH_PAD_BUILD_N];
  uchar innerAboveThreshold[SCRATCH_PAD_WORK_GROUP_SIZE];
};

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
