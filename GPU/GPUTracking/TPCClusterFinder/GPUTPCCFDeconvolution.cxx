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

/// \file GPUTPCCFDeconvolution.cxx
/// \author Felix Weiglhofer

#include "GPUTPCCFDeconvolution.h"
#include "CfConsts.h"
#include "CfUtils.h"
#include "ChargePos.h"
#include "GPUDefMacros.h"

using namespace GPUCA_NAMESPACE::gpu;
using namespace GPUCA_NAMESPACE::gpu::tpccf;

template <>
GPUdii() void GPUTPCCFDeconvolution::Thread<0>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUSharedMemory& smem, processorType& clusterer)
{
  Array2D<PackedCharge> chargeMap(reinterpret_cast<PackedCharge*>(clusterer.mPchargeMap));
  Array2D<uint8_t> isPeakMap(clusterer.mPpeakMap);
  GPUTPCCFDeconvolution::deconvolutionImpl(get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), smem, isPeakMap, chargeMap, clusterer.mPpositions, clusterer.mPmemory->counters.nPositions);
}

GPUdii() void GPUTPCCFDeconvolution::deconvolutionImpl(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUSharedMemory& smem,
                                                       const Array2D<uint8_t>& peakMap,
                                                       Array2D<PackedCharge>& chargeMap,
                                                       const ChargePos* positions,
                                                       const uint32_t digitnum)
{
  SizeT idx = get_global_id(0);

  bool iamDummy = (idx >= digitnum);
  idx = iamDummy ? digitnum - 1 : idx;

  ChargePos pos = positions[idx];

  bool iamPeak = CfUtils::isPeak(peakMap[pos]);

  int8_t peakCount = (iamPeak) ? 1 : 0;

  uint16_t ll = get_local_id(0);
  uint16_t partId = ll;

  uint16_t in3x3 = 0;
  bool exclude3x3 = iamPeak || !pos.valid();
  partId = CfUtils::partition<SCRATCH_PAD_WORK_GROUP_SIZE>(smem, ll, exclude3x3, SCRATCH_PAD_WORK_GROUP_SIZE, &in3x3);

  if (partId < in3x3) {
    smem.posBcast1[partId] = pos;
  }
  GPUbarrier();

  CfUtils::blockLoad(
    peakMap,
    in3x3,
    SCRATCH_PAD_WORK_GROUP_SIZE,
    ll,
    0,
    8,
    cfconsts::InnerNeighbors,
    smem.posBcast1,
    smem.buf);

  uint8_t aboveThreshold = 0;
  if (partId < in3x3) {
    peakCount = countPeaksInner(partId, smem.buf, &aboveThreshold);
  }

  uint16_t in5x5 = 0;
  partId = CfUtils::partition<SCRATCH_PAD_WORK_GROUP_SIZE>(smem, partId, peakCount > 0 && !exclude3x3, in3x3, &in5x5);

  if (partId < in5x5) {
    smem.posBcast1[partId] = pos;
    smem.aboveThresholdBcast[partId] = aboveThreshold;
  }
  GPUbarrier();

  CfUtils::condBlockLoad<uint8_t, true>(
    peakMap,
    in5x5,
    SCRATCH_PAD_WORK_GROUP_SIZE,
    ll,
    0,
    16,
    cfconsts::OuterNeighbors,
    smem.posBcast1,
    smem.aboveThresholdBcast,
    smem.buf);

  if (partId < in5x5) {
    peakCount = countPeaksOuter(partId, aboveThreshold, smem.buf);
    peakCount *= -1;
  }

  if (iamDummy || !pos.valid()) {
    return;
  }

  bool has3x3 = (peakCount > 0);
  peakCount = CAMath::Abs(int32_t(peakCount));
  bool split = (peakCount > 1);

  peakCount = (peakCount == 0) ? 1 : peakCount;

  PackedCharge charge = chargeMap[pos];
  PackedCharge p(charge.unpack() / peakCount, has3x3, split);

  chargeMap[pos] = p;
}

GPUdi() uint8_t GPUTPCCFDeconvolution::countPeaksInner(
  uint16_t ll,
  const uint8_t* isPeak,
  uint8_t* aboveThreshold)
{
  uint8_t peaks = 0;
  GPUCA_UNROLL(U(), U())
  for (uint8_t i = 0; i < 8; i++) {
    uint8_t p = isPeak[ll * 8 + i];
    peaks += CfUtils::isPeak(p);
    *aboveThreshold |= uint8_t(CfUtils::isAboveThreshold(p)) << i;
  }

  return peaks;
}

GPUdi() uint8_t GPUTPCCFDeconvolution::countPeaksOuter(
  uint16_t ll,
  uint8_t aboveThreshold,
  const uint8_t* isPeak)
{
  uint8_t peaks = 0;
  GPUCA_UNROLL(U(), U())
  for (uint8_t i = 0; i < 16; i++) {
    uint8_t p = isPeak[ll * 16 + i];
    peaks += CfUtils::isPeak(p);
  }

  return peaks;
}
