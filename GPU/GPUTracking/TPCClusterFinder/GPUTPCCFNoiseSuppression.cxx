// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
using namespace GPUCA_NAMESPACE::gpu::deprecated;

template <>
GPUdii() void GPUTPCCFNoiseSuppression::Thread<GPUTPCCFNoiseSuppression::noiseSuppression>(int nBlocks, int nThreads, int iBlock, int iThread, GPUSharedMemory& smem, processorType& clusterer)
{
  Array2D<PackedCharge> chargeMap(reinterpret_cast<PackedCharge*>(clusterer.mPchargeMap));
  Array2D<uchar> isPeakMap(clusterer.mPpeakMap);
  GPUTPCCFNoiseSuppression::noiseSuppressionImpl(get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), smem, chargeMap, isPeakMap, clusterer.mPpeaks, clusterer.mPmemory->counters.nPeaks, clusterer.mPisPeak);
}

template <>
GPUdii() void GPUTPCCFNoiseSuppression::Thread<GPUTPCCFNoiseSuppression::updatePeaks>(int nBlocks, int nThreads, int iBlock, int iThread, GPUSharedMemory& smem, processorType& clusterer)
{
  Array2D<uchar> isPeakMap(clusterer.mPpeakMap);
  GPUTPCCFNoiseSuppression::updatePeaksImpl(get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), clusterer.mPpeaks, clusterer.mPisPeak, clusterer.mPmemory->counters.nPeaks, isPeakMap);
}

GPUd() void GPUTPCCFNoiseSuppression::noiseSuppressionImpl(int nBlocks, int nThreads, int iBlock, int iThread, GPUSharedMemory& smem,
                                                           const Array2D<PackedCharge>& chargeMap,
                                                           const Array2D<uchar>& peakMap,
                                                           const Digit* peaks,
                                                           const uint peaknum,
                                                           uchar* isPeakPredicate)
{
  size_t idx = get_global_id(0);

  Digit myDigit = peaks[CAMath::Min(idx, (size_t)(peaknum - 1))];

  ChargePos pos(myDigit);

  ulong minimas, bigger, peaksAround;

#if defined(BUILD_CLUSTER_SCRATCH_PAD)
  findMinimaAndPeaksScratchpad(
    chargeMap,
    peakMap,
    myDigit.charge,
    pos,
    smem.posBcast,
    smem.buf,
    &minimas,
    &bigger,
    &peaksAround);
#else
  findMinima(
    chargeMap,
    pos,
    myDigit.charge,
    NOISE_SUPPRESSION_MINIMA_EPSILON,
    &minimas,
    &bigger);

  peaksAround = findPeaks(peakMap, pos);
#endif

  peaksAround &= bigger;

  bool keepMe = keepPeak(minimas, peaksAround);

  bool iamDummy = (idx >= peaknum);
  if (iamDummy) {
    return;
  }

  DBG_PRINT("%d: p:%lx, m:%lx, b:%lx.", int(idx), peaksAroundBack, minimas, bigger);

  isPeakPredicate[idx] = keepMe;
}

GPUd() void GPUTPCCFNoiseSuppression::updatePeaksImpl(int nBlocks, int nThreads, int iBlock, int iThread,
                                                      const Digit* peaks,
                                                      const uchar* isPeak,
                                                      const uint peakNum,
                                                      Array2D<uchar>& peakMap)
{
  size_t idx = get_global_id(0);

  if (idx >= peakNum) {
    return;
  }

  Digit myDigit = peaks[idx];

  ChargePos pos(myDigit);

  uchar peak = isPeak[idx];

  peakMap[pos] = (uchar(myDigit.charge > CHARGE_THRESHOLD) << 1) | peak;
}

GPUd() void GPUTPCCFNoiseSuppression::checkForMinima(
  float q,
  float epsilon,
  PackedCharge other,
  int pos,
  ulong* minimas,
  ulong* bigger)
{
  float r = other.unpack();

  ulong isMinima = (q - r > epsilon);
  *minimas |= (isMinima << pos);

  ulong lq = (r > q);
  *bigger |= (lq << pos);
}

GPUd() void GPUTPCCFNoiseSuppression::findMinimaScratchPad(
  const PackedCharge* buf,
  const ushort ll,
  const int N,
  int pos,
  const float q,
  const float epsilon,
  ulong* minimas,
  ulong* bigger)
{
  for (int i = 0; i < N; i++, pos++) {
    PackedCharge other = buf[N * ll + i];

    checkForMinima(q, epsilon, other, pos, minimas, bigger);
  }
}

GPUd() void GPUTPCCFNoiseSuppression::findPeaksScratchPad(
  const uchar* buf,
  const ushort ll,
  const int N,
  int pos,
  ulong* peaks)
{
  for (int i = 0; i < N; i++, pos++) {
    ulong p = GET_IS_PEAK(buf[N * ll + i]);

    *peaks |= (p << pos);
  }
}

GPUd() void GPUTPCCFNoiseSuppression::findMinima(
  const Array2D<PackedCharge>& chargeMap,
  const ChargePos& pos,
  const float q,
  const float epsilon,
  ulong* minimas,
  ulong* bigger)
{
  *minimas = 0;
  *bigger = 0;

  for (int i = 0; i < NOISE_SUPPRESSION_NEIGHBOR_NUM; i++) {
    Delta2 d = CfConsts::NoiseSuppressionNeighbors[i];

    PackedCharge other = chargeMap[pos.delta(d)];

    checkForMinima(q, epsilon, other, i, minimas, bigger);
  }
}

GPUd() ulong GPUTPCCFNoiseSuppression::findPeaks(
  const Array2D<uchar>& peakMap,
  const ChargePos& pos)
{
  ulong peaks = 0;

  DBG_PRINT("Looking around %d, %d", pos.gpad, pos.time);

  for (int i = 0; i < NOISE_SUPPRESSION_NEIGHBOR_NUM; i++) {
    Delta2 d = CfConsts::NoiseSuppressionNeighbors[i];

    uchar p = peakMap[pos.delta(d)];

    DBG_PRINT("%d, %d: %d", d.x, d.y, p);

    peaks |= (ulong(GET_IS_PEAK(p)) << i);
  }

  return peaks;
}

GPUd() bool GPUTPCCFNoiseSuppression::keepPeak(
  ulong minima,
  ulong peaks)
{
  bool keepMe = true;

  for (int i = 0; i < NOISE_SUPPRESSION_NEIGHBOR_NUM; i++) {
    bool otherPeak = (peaks & (ulong(1) << i));
    bool minimaBetween = (minima & CfConsts::NoiseSuppressionMinima[i]);

    keepMe &= (!otherPeak || minimaBetween);
  }

  return keepMe;
}

GPUd() void GPUTPCCFNoiseSuppression::findMinimaAndPeaksScratchpad(
  const Array2D<PackedCharge>& chargeMap,
  const Array2D<uchar>& peakMap,
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
    CfConsts::NoiseSuppressionNeighbors,
    posBcast,
    buf);

  findMinimaScratchPad(
    buf,
    ll,
    2,
    16,
    q,
    NOISE_SUPPRESSION_MINIMA_EPSILON,
    minimas,
    bigger);

  CfUtils::blockLoad(
    chargeMap,
    wgSizeHalf,
    SCRATCH_PAD_WORK_GROUP_SIZE,
    ll,
    0,
    16,
    CfConsts::NoiseSuppressionNeighbors,
    posBcast,
    buf);

  if (inGroup1) {
    findMinimaScratchPad(
      buf,
      llhalf,
      16,
      0,
      q,
      NOISE_SUPPRESSION_MINIMA_EPSILON,
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
    CfConsts::NoiseSuppressionNeighbors,
    posBcast,
    buf);

  if (inGroup1) {
    findMinimaScratchPad(
      buf,
      llhalf,
      16,
      18,
      q,
      NOISE_SUPPRESSION_MINIMA_EPSILON,
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
    CfConsts::NoiseSuppressionNeighbors,
    posBcast + wgSizeHalf,
    buf);

  if (!inGroup1) {
    findMinimaScratchPad(
      buf,
      llhalf,
      16,
      0,
      q,
      NOISE_SUPPRESSION_MINIMA_EPSILON,
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
    CfConsts::NoiseSuppressionNeighbors,
    posBcast + wgSizeHalf,
    buf);

  if (!inGroup1) {
    findMinimaScratchPad(
      buf,
      llhalf,
      16,
      18,
      q,
      NOISE_SUPPRESSION_MINIMA_EPSILON,
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
    CfConsts::NoiseSuppressionNeighbors,
    posBcast,
    bufp);

  findPeaksScratchPad(
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
    CfConsts::NoiseSuppressionNeighbors,
    posBcast,
    bufp);

  findPeaksScratchPad(
    bufp,
    ll,
    16,
    18,
    peaks);
}
