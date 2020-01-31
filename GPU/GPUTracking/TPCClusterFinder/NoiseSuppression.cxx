// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file NoiseSuppression.cxx
/// \author Felix Weiglhofer

#include "NoiseSuppression.h"
#include "Array2D.h"
#include "CfConsts.h"
#include "CfUtils.h"

using namespace GPUCA_NAMESPACE::gpu;
using namespace GPUCA_NAMESPACE::gpu::deprecated;

GPUd() void NoiseSuppression::noiseSuppressionImpl(int nBlocks, int nThreads, int iBlock, int iThread, GPUTPCClusterFinderKernels::GPUTPCSharedMemory& smem,
                                                   const Array2D<PackedCharge>& chargeMap,
                                                   const Array2D<uchar>& peakMap,
                                                   GPUglobalref() const Digit* peaks,
                                                   const uint peaknum,
                                                   GPUglobalref() uchar* isPeakPredicate)
{
  size_t idx = get_global_id(0);

  Digit myDigit = peaks[CAMath::Min(idx, (size_t)(peaknum - 1))];

  GlobalPad gpad = CfUtils::tpcGlobalPadIdx(myDigit.row, myDigit.pad);

  ulong minimas, bigger, peaksAround;

  bool debug = false;

  ChargePos pos(gpad, myDigit.time);

#if defined(BUILD_CLUSTER_SCRATCH_PAD)
  findMinimaAndPeaksScratchpad(
    chargeMap,
    peakMap,
    myDigit.charge,
    pos,
    smem.noise.posBcast,
    smem.noise.buf,
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

  peaksAround = findPeaks(peakMap, pos, debug);
#endif

  ulong peaksAroundBack = peaksAround;
  peaksAround &= bigger;

  bool keepMe = keepPeak(minimas, peaksAround);

  bool iamDummy = (idx >= peaknum);
  if (iamDummy) {
    return;
  }

  if (debug) {
    printf("%d: p:%lx, m:%lx, b:%lx.\n", int(idx), peaksAroundBack, minimas, bigger);
  }

  isPeakPredicate[idx] = keepMe;
}

GPUd() void NoiseSuppression::updatePeaksImpl(int nBlocks, int nThreads, int iBlock, int iThread, GPUTPCClusterFinderKernels::GPUTPCSharedMemory& smem,
                                              GPUglobalref() const Digit* peaks,
                                              GPUglobalref() const uchar* isPeak,
                                              Array2D<uchar>& peakMap)
{
  size_t idx = get_global_id(0);

  Digit myDigit = peaks[idx];
  GlobalPad gpad = CfUtils::tpcGlobalPadIdx(myDigit.row, myDigit.pad);

  ChargePos pos(gpad, myDigit.time);

  uchar peak = isPeak[idx];

  peakMap[pos] = (uchar(myDigit.charge > CHARGE_THRESHOLD) << 1) | peak;
}

GPUd() void NoiseSuppression::checkForMinima(
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

GPUd() void NoiseSuppression::findMinimaScratchPad(
  GPUsharedref() const PackedCharge* buf,
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

GPUd() void NoiseSuppression::findPeaksScratchPad(
  GPUsharedref() const uchar* buf,
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

GPUd() void NoiseSuppression::findMinima(
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

GPUd() ulong NoiseSuppression::findPeaks(
  const Array2D<uchar>& peakMap,
  const ChargePos& pos,
  bool debug)
{
  ulong peaks = 0;
  if (debug) {
    printf("Looking around %d, %d\n", pos.gpad, pos.time);
  }
  for (int i = 0; i < NOISE_SUPPRESSION_NEIGHBOR_NUM; i++) {
    Delta2 d = CfConsts::NoiseSuppressionNeighbors[i];

    uchar p = peakMap[pos.delta(d)];

    if (debug) {
      printf("%d, %d: %d\n", d.x, d.y, p);
    }

    peaks |= (ulong(GET_IS_PEAK(p)) << i);
  }

  return peaks;
}

GPUd() bool NoiseSuppression::keepPeak(
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

GPUd() void NoiseSuppression::findMinimaAndPeaksScratchpad(
  const Array2D<PackedCharge>& chargeMap,
  const Array2D<uchar>& peakMap,
  float q,
  const ChargePos& pos,
  GPUsharedref() ChargePos* posBcast,
  GPUsharedref() PackedCharge* buf,
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

  GPUsharedref() uchar* bufp = (GPUsharedref() uchar*)buf;

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
