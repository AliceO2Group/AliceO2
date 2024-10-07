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

/// \file GPUTPCCFNoiseSuppression.cxx
/// \author Felix Weiglhofer

#include "GPUTPCCFNoiseSuppression.h"
#include "Array2D.h"
#include "CfConsts.h"
#include "CfUtils.h"
#include "ChargePos.h"

using namespace GPUCA_NAMESPACE::gpu;
using namespace GPUCA_NAMESPACE::gpu::tpccf;

template <>
GPUdii() void GPUTPCCFNoiseSuppression::Thread<GPUTPCCFNoiseSuppression::noiseSuppression>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUSharedMemory& smem, processorType& clusterer)
{
  Array2D<PackedCharge> chargeMap(reinterpret_cast<PackedCharge*>(clusterer.mPchargeMap));
  Array2D<uint8_t> isPeakMap(clusterer.mPpeakMap);
  noiseSuppressionImpl(get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), smem, clusterer.Param().rec, chargeMap, isPeakMap, clusterer.mPpeakPositions, clusterer.mPmemory->counters.nPeaks, clusterer.mPisPeak);
}

template <>
GPUdii() void GPUTPCCFNoiseSuppression::Thread<GPUTPCCFNoiseSuppression::updatePeaks>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUSharedMemory& smem, processorType& clusterer)
{
  Array2D<uint8_t> isPeakMap(clusterer.mPpeakMap);
  updatePeaksImpl(get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), clusterer.mPpeakPositions, clusterer.mPisPeak, clusterer.mPmemory->counters.nPeaks, isPeakMap);
}

GPUdii() void GPUTPCCFNoiseSuppression::noiseSuppressionImpl(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUSharedMemory& smem,
                                                             const GPUSettingsRec& calibration,
                                                             const Array2D<PackedCharge>& chargeMap,
                                                             const Array2D<uint8_t>& peakMap,
                                                             const ChargePos* peakPositions,
                                                             const uint32_t peaknum,
                                                             uint8_t* isPeakPredicate)
{
  SizeT idx = get_global_id(0);

  ChargePos pos = peakPositions[CAMath::Min(idx, (SizeT)(peaknum - 1))];
  Charge charge = chargeMap[pos].unpack();

  uint64_t minimas, bigger, peaksAround;
  findMinimaAndPeaks(
    chargeMap,
    peakMap,
    calibration,
    charge,
    pos,
    smem.posBcast,
    smem.buf,
    &minimas,
    &bigger,
    &peaksAround);

  peaksAround &= bigger;

  bool keepMe = keepPeak(minimas, peaksAround);

  bool iamDummy = (idx >= peaknum);
  if (iamDummy) {
    return;
  }

  isPeakPredicate[idx] = keepMe;
}

GPUd() void GPUTPCCFNoiseSuppression::updatePeaksImpl(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread,
                                                      const ChargePos* peakPositions,
                                                      const uint8_t* isPeak,
                                                      const uint32_t peakNum,
                                                      Array2D<uint8_t>& peakMap)
{
  SizeT idx = get_global_id(0);

  if (idx >= peakNum) {
    return;
  }

  ChargePos pos = peakPositions[idx];

  uint8_t peak = isPeak[idx];

  peakMap[pos] = 0b10 | peak; // if this positions was marked as peak at some point, then its charge must exceed the charge threshold.
                              // So we can just set the bit and avoid rereading the charge
}

GPUdi() void GPUTPCCFNoiseSuppression::checkForMinima(
  const float q,
  const float epsilon,
  const float epsilonRelative,
  PackedCharge other,
  int32_t pos,
  uint64_t* minimas,
  uint64_t* bigger)
{
  float r = other.unpack();

  uint64_t isMinima = (q - r > epsilon) && (float)CAMath::Abs(q - r) / (float)CAMath::Max(q, r) > epsilonRelative; // TODO: Can we assume q > r and get rid of Max/Abs?
  *minimas |= (isMinima << pos);

  uint64_t lq = (r > q);
  *bigger |= (lq << pos);
}

GPUdi() void GPUTPCCFNoiseSuppression::findMinima(
  const PackedCharge* buf,
  const uint16_t ll,
  const int32_t N,
  int32_t pos,
  const float q,
  const float epsilon,
  const float epsilonRelative,
  uint64_t* minimas,
  uint64_t* bigger)
{
  GPUCA_UNROLL(U(), U())
  for (int32_t i = 0; i < N; i++, pos++) {
    PackedCharge other = buf[N * ll + i];

    checkForMinima(q, epsilon, epsilonRelative, other, pos, minimas, bigger);
  }
}

GPUdi() void GPUTPCCFNoiseSuppression::findPeaks(
  const uint8_t* buf,
  const uint16_t ll,
  const int32_t N,
  int32_t pos,
  uint64_t* peaks)
{
  GPUCA_UNROLL(U(), U())
  for (int32_t i = 0; i < N; i++, pos++) {
    uint64_t p = CfUtils::isPeak(buf[N * ll + i]);

    *peaks |= (p << pos);
  }
}

GPUdi() bool GPUTPCCFNoiseSuppression::keepPeak(
  uint64_t minima,
  uint64_t peaks)
{
  bool keepMe = true;

  GPUCA_UNROLL(U(), U())
  for (int32_t i = 0; i < NOISE_SUPPRESSION_NEIGHBOR_NUM; i++) {
    bool otherPeak = (peaks & (uint64_t(1) << i));
    bool minimaBetween = (minima & cfconsts::NoiseSuppressionMinima[i]);

    keepMe &= (!otherPeak || minimaBetween);
  }

  return keepMe;
}

GPUd() void GPUTPCCFNoiseSuppression::findMinimaAndPeaks(
  const Array2D<PackedCharge>& chargeMap,
  const Array2D<uint8_t>& peakMap,
  const GPUSettingsRec& calibration,
  float q,
  const ChargePos& pos,
  ChargePos* posBcast,
  PackedCharge* buf,
  uint64_t* minimas,
  uint64_t* bigger,
  uint64_t* peaks)
{
  uint16_t ll = get_local_id(0);

  posBcast[ll] = pos;
  GPUbarrier();

  uint16_t wgSizeHalf = (SCRATCH_PAD_WORK_GROUP_SIZE + 1) / 2;

  bool inGroup1 = ll < wgSizeHalf;
  uint16_t llhalf = (inGroup1) ? ll : (ll - wgSizeHalf);

  *minimas = 0;
  *bigger = 0;
  *peaks = 0;

  /**************************************
   * Look for minima
   **************************************/

  CfUtils::blockLoad(
    chargeMap,
    SCRATCH_PAD_WORK_GROUP_SIZE,
    SCRATCH_PAD_WORK_GROUP_SIZE,
    ll,
    16,
    2,
    cfconsts::NoiseSuppressionNeighbors,
    posBcast,
    buf);

  findMinima(
    buf,
    ll,
    2,
    16,
    q,
    calibration.tpc.cfNoiseSuppressionEpsilon,
    calibration.tpc.cfNoiseSuppressionEpsilonRelative / 255.f,
    minimas,
    bigger);

  CfUtils::blockLoad(
    chargeMap,
    wgSizeHalf,
    SCRATCH_PAD_WORK_GROUP_SIZE,
    ll,
    0,
    16,
    cfconsts::NoiseSuppressionNeighbors,
    posBcast,
    buf);

  if (inGroup1) {
    findMinima(
      buf,
      llhalf,
      16,
      0,
      q,
      calibration.tpc.cfNoiseSuppressionEpsilon,
      calibration.tpc.cfNoiseSuppressionEpsilonRelative / 255.f,
      minimas,
      bigger);
  }

  CfUtils::blockLoad(
    chargeMap,
    wgSizeHalf,
    SCRATCH_PAD_WORK_GROUP_SIZE,
    ll,
    18,
    16,
    cfconsts::NoiseSuppressionNeighbors,
    posBcast,
    buf);

  if (inGroup1) {
    findMinima(
      buf,
      llhalf,
      16,
      18,
      q,
      calibration.tpc.cfNoiseSuppressionEpsilon,
      calibration.tpc.cfNoiseSuppressionEpsilonRelative / 255.f,
      minimas,
      bigger);
  }

#if defined(GPUCA_GPUCODE)
  CfUtils::blockLoad(
    chargeMap,
    wgSizeHalf,
    SCRATCH_PAD_WORK_GROUP_SIZE,
    ll,
    0,
    16,
    cfconsts::NoiseSuppressionNeighbors,
    posBcast + wgSizeHalf,
    buf);

  if (!inGroup1) {
    findMinima(
      buf,
      llhalf,
      16,
      0,
      q,
      calibration.tpc.cfNoiseSuppressionEpsilon,
      calibration.tpc.cfNoiseSuppressionEpsilonRelative / 255.f,
      minimas,
      bigger);
  }

  CfUtils::blockLoad(
    chargeMap,
    wgSizeHalf,
    SCRATCH_PAD_WORK_GROUP_SIZE,
    ll,
    18,
    16,
    cfconsts::NoiseSuppressionNeighbors,
    posBcast + wgSizeHalf,
    buf);

  if (!inGroup1) {
    findMinima(
      buf,
      llhalf,
      16,
      18,
      q,
      calibration.tpc.cfNoiseSuppressionEpsilon,
      calibration.tpc.cfNoiseSuppressionEpsilonRelative / 255.f,
      minimas,
      bigger);
  }
#endif

  uint8_t* bufp = (uint8_t*)buf;

  /**************************************
     * Look for peaks
     **************************************/

  CfUtils::blockLoad(
    peakMap,
    SCRATCH_PAD_WORK_GROUP_SIZE,
    SCRATCH_PAD_WORK_GROUP_SIZE,
    ll,
    0,
    16,
    cfconsts::NoiseSuppressionNeighbors,
    posBcast,
    bufp);

  findPeaks(
    bufp,
    ll,
    16,
    0,
    peaks);

  CfUtils::blockLoad(
    peakMap,
    SCRATCH_PAD_WORK_GROUP_SIZE,
    SCRATCH_PAD_WORK_GROUP_SIZE,
    ll,
    18,
    16,
    cfconsts::NoiseSuppressionNeighbors,
    posBcast,
    bufp);

  findPeaks(
    bufp,
    ll,
    16,
    18,
    peaks);
}
