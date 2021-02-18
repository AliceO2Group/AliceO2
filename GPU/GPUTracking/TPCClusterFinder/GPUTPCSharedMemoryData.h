// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTPCSharedMemoryData.h
/// \author Felix Weiglhofer

#ifndef O2_GPU_GPU_TPC_SHARED_MEMORY_DATA_H
#define O2_GPU_GPU_TPC_SHARED_MEMORY_DATA_H

#include "cl/clusterFinderDefs.h"
#include "PackedCharge.h"

#ifndef SCRATCH_PAD_WORK_GROUP_SIZE
#error "Work group size not defined"
#endif

namespace GPUCA_NAMESPACE
{
namespace gpu
{

class GPUTPCSharedMemoryData
{

 public:
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
};

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
