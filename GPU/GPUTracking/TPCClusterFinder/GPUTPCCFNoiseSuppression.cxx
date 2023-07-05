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
GPUdii() void GPUTPCCFNoiseSuppression::Thread<GPUTPCCFNoiseSuppression::noiseSuppression>(int nBlocks, int nThreads, int iBlock, int iThread, GPUSharedMemory& smem, processorType& clusterer)
{
  Array2D<PackedCharge> chargeMap(reinterpret_cast<PackedCharge*>(clusterer.mPchargeMap));
  Array2D<uchar> isPeakMap(clusterer.mPpeakMap);
  noiseSuppressionImpl(get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), smem, clusterer.Param().rec, chargeMap, isPeakMap, clusterer.mPpeakPositions, clusterer.mPmemory->counters.nPeaks, clusterer.mPisPeak);
}

template <>
GPUdii() void GPUTPCCFNoiseSuppression::Thread<GPUTPCCFNoiseSuppression::updatePeaks>(int nBlocks, int nThreads, int iBlock, int iThread, GPUSharedMemory& smem, processorType& clusterer)
{
  Array2D<uchar> isPeakMap(clusterer.mPpeakMap);
  updatePeaksImpl(get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), clusterer.mPpeakPositions, clusterer.mPisPeak, clusterer.mPmemory->counters.nPeaks, isPeakMap);
}

GPUdii() void GPUTPCCFNoiseSuppression::noiseSuppressionImpl(int nBlocks, int nThreads, int iBlock, int iThread, GPUSharedMemory& smem,
                                                             const GPUSettingsRec& calibration,
                                                             const Array2D<PackedCharge>& chargeMap,
                                                             const Array2D<uchar>& peakMap,
                                                             const ChargePos* peakPositions,
                                                             const uint peaknum,
                                                             uchar* isPeakPredicate)
{
  SizeT idx = get_global_id(0);

  ChargePos pos = peakPositions[CAMath::Min(idx, (SizeT)(peaknum - 1))];
  Charge charge = chargeMap[pos].unpack();

  ulong minimas, bigger, peaksAround;
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

GPUd() void GPUTPCCFNoiseSuppression::updatePeaksImpl(int nBlocks, int nThreads, int iBlock, int iThread,
                                                      const ChargePos* peakPositions,
                                                      const uchar* isPeak,
                                                      const uint peakNum,
                                                      Array2D<uchar>& peakMap)
{
  SizeT idx = get_global_id(0);

  if (idx >= peakNum) {
    return;
  }

  ChargePos pos = peakPositions[idx];

  uchar peak = isPeak[idx];

  peakMap[pos] = 0b10 | peak; // if this positions was marked as peak at some point, then its charge must exceed the charge threshold.
                              // So we can just set the bit and avoid rereading the charge
}

GPUdi() void GPUTPCCFNoiseSuppression::checkForMinima(
  const float q,
  const float epsilon,
  const float epsilonRelative,
  PackedCharge other,
  int pos,
  ulong* minimas,
  ulong* bigger)
{
  float r = other.unpack();

  ulong isMinima = (q - r > epsilon) && (float)CAMath::Abs(q - r) / (float)CAMath::Max(q, r) > epsilonRelative; // TODO: Can we assume q > r and get rid of Max/Abs?
  *minimas |= (isMinima << pos);

  ulong lq = (r > q);
  *bigger |= (lq << pos);
}

GPUdi() void GPUTPCCFNoiseSuppression::findMinima(
  const PackedCharge* buf,
  const ushort ll,
  const int N,
  int pos,
  const float q,
  const float epsilon,
  const float epsilonRelative,
  ulong* minimas,
  ulong* bigger)
{
  GPUCA_UNROLL(U(), U())
  for (int i = 0; i < N; i++, pos++) {
    PackedCharge other = buf[N * ll + i];

    checkForMinima(q, epsilon, epsilonRelative, other, pos, minimas, bigger);
  }
}

GPUdi() void GPUTPCCFNoiseSuppression::findPeaks(
  const uchar* buf,
  const ushort ll,
  const int N,
  int pos,
  ulong* peaks)
{
  GPUCA_UNROLL(U(), U())
  for (int i = 0; i < N; i++, pos++) {
    ulong p = CfUtils::isPeak(buf[N * ll + i]);

    *peaks |= (p << pos);
  }
}

GPUdi() bool GPUTPCCFNoiseSuppression::keepPeak(
  ulong minima,
  ulong peaks)
{
  bool keepMe = true;

  GPUCA_UNROLL(U(), U())
  for (int i = 0; i < NOISE_SUPPRESSION_NEIGHBOR_NUM; i++) {
    bool otherPeak = (peaks & (ulong(1) << i));
    bool minimaBetween = (minima & cfconsts::NoiseSuppressionMinima[i]);

    keepMe &= (!otherPeak || minimaBetween);
  }

  return keepMe;
}

GPUd() void GPUTPCCFNoiseSuppression::findMinimaAndPeaks(
  const Array2D<PackedCharge>& chargeMap,
  const Array2D<uchar>& peakMap,
  const GPUSettingsRec& calibration,
  float q,
  const ChargePos& pos,
  ChargePos* posBcast,
  PackedCharge* buf,
  ulong* minimas,
  ulong* bigger,
  ulong* peaks)
{
  ushort ll = get_local_id(0);

  posBcast[ll] = pos;
  GPUbarrier();

  ushort wgSizeHalf = (SCRATCH_PAD_WORK_GROUP_SIZE + 1) / 2;

  bool inGroup1 = ll < wgSizeHalf;
  ushort llhalf = (inGroup1) ? ll : (ll - wgSizeHalf);

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
    calibration.tpc.cfNoiseSuppressionEpsilonRelative / 255.,
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
      calibration.tpc.cfNoiseSuppressionEpsilonRelative / 255.,
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
      calibration.tpc.cfNoiseSuppressionEpsilonRelative / 255.,
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
      calibration.tpc.cfNoiseSuppressionEpsilonRelative / 255.,
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
      calibration.tpc.cfNoiseSuppressionEpsilonRelative / 255.,
      minimas,
      bigger);
  }
#endif

  uchar* bufp = (uchar*)buf;

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
