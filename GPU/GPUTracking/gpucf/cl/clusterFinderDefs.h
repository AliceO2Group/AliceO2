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
#define SCRATCH_PAD_WORK_GROUP_SIZE 64
#define BUILD_CLUSTER_NAIVE
#define BUILD_CLUSTER_SCRATCH_PAD
#define DCHARGEMAP_TIME_MAJOR_LAYOUT
#endif

#include "shared/types.h"
#include "config.h"
#include "debug.h"

#include "shared/ClusterNative.h"
#include "shared/constants.h"
#include "shared/tpc.h"

#if defined(DEBUG_ON)
#define IF_DBG_INST if (get_global_id(0) == 8)
#define IF_DBG_GROUP if (get_group_id(0) == 0)
#else
#define IF_DBG_INST if (false)
#define IF_DBG_GROUP if (false)
#endif

#define GET_IS_PEAK(val) (val & 0x01)
#define GET_IS_ABOVE_THRESHOLD(val) (val >> 1)

typedef ushort packed_charge_t;

typedef struct ClusterAccumulator_s {
  charge_t Q;
  charge_t padMean;
  charge_t padSigma;
  charge_t timeMean;
  charge_t timeSigma;
  uchar splitInTime;
  uchar splitInPad;
} ClusterAccumulator;

typedef short delta_t;
typedef short2 delta2_t;

typedef struct ChargePos_s {
  global_pad_t gpad;
  timestamp time;
} ChargePos;

typedef short2 local_id;

struct search_t {
  ChargePos posBcast[SCRATCH_PAD_WORK_GROUP_SIZE];
  packed_charge_t buf[SCRATCH_PAD_WORK_GROUP_SIZE * SCRATCH_PAD_SEARCH_N];
};
struct noise_t {
  ChargePos posBcast[SCRATCH_PAD_WORK_GROUP_SIZE];
  packed_charge_t buf[SCRATCH_PAD_WORK_GROUP_SIZE * SCRATCH_PAD_NOISE_N];
};
struct count_t {
  ChargePos posBcast1[SCRATCH_PAD_WORK_GROUP_SIZE];
  uchar aboveThresholdBcast[SCRATCH_PAD_WORK_GROUP_SIZE];
  uchar buf[SCRATCH_PAD_WORK_GROUP_SIZE * SCRATCH_PAD_COUNT_N];
};
struct build_t {
  ChargePos posBcast[SCRATCH_PAD_WORK_GROUP_SIZE];
  packed_charge_t buf[SCRATCH_PAD_WORK_GROUP_SIZE * SCRATCH_PAD_BUILD_N];
  uchar innerAboveThreshold[SCRATCH_PAD_WORK_GROUP_SIZE];
};

#endif
