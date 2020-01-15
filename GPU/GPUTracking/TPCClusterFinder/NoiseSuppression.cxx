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
                                                   GPUglobalref() const PackedCharge* chargeMap,
                                                   GPUglobalref() const uchar* peakMap,
                                                   GPUglobalref() const Digit* peaks,
                                                   const uint peaknum,
                                                   GPUglobalref() uchar* isPeakPredicate)
{
  size_t idx = get_global_id(0);

  Digit myDigit = peaks[CAMath::Min(idx, (size_t)(peaknum - 1))];

  GlobalPad gpad = Array2D::tpcGlobalPadIdx(myDigit.row, myDigit.pad);

  ulong minimas, bigger, peaksAround;

  bool debug = false;

#if defined(BUILD_CLUSTER_SCRATCH_PAD)
  findMinimaAndPeaksScratchpad(
    chargeMap,
    peakMap,
    myDigit.charge,
    gpad,
    myDigit.time,
    smem.noise.posBcast,
    smem.noise.buf,
    &minimas,
    &bigger,
    &peaksAround);
#else
  findMinima(
    chargeMap,
    gpad,
    myDigit.time,
    myDigit.charge,
    NOISE_SUPPRESSION_MINIMA_EPSILON,
    &minimas,
    &bigger);

  peaksAround = findPeaks(peakMap, gpad, myDigit.time, debug);
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
  /* isPeakPredicate[idx] = keepMe && false; */
  /* isPeakPredicate[idx] = keepMe || true; */
}

GPUd() void NoiseSuppression::updatePeaksImpl(int nBlocks, int nThreads, int iBlock, int iThread, GPUTPCClusterFinderKernels::GPUTPCSharedMemory& smem,
                                              GPUglobalref() const Digit* peaks,
                                              GPUglobalref() const uchar* isPeak,
                                              GPUglobalref() uchar* peakMap)
{
  size_t idx = get_global_id(0);

  Digit myDigit = peaks[idx];
  GlobalPad gpad = Array2D::tpcGlobalPadIdx(myDigit.row, myDigit.pad);

  uchar peak = isPeak[idx];

  IS_PEAK(peakMap, gpad, myDigit.time) =
    ((myDigit.charge > CHARGE_THRESHOLD) << 1) | peak;
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
  GPUglobalref() const PackedCharge* chargeMap,
  const GlobalPad gpad,
  const Timestamp time,
  const float q,
  const float epsilon,
  ulong* minimas,
  ulong* bigger)
{
  *minimas = 0;
  *bigger = 0;

  for (int i = 0; i < NOISE_SUPPRESSION_NEIGHBOR_NUM; i++) {
    Delta2 d = CfConsts::NoiseSuppressionNeighbors[i];
    Delta dp = d.x;
    Delta dt = d.y;

    PackedCharge other = CHARGE(chargeMap, gpad + dp, time + dt);

    checkForMinima(q, epsilon, other, i, minimas, bigger);
  }
}

GPUd() ulong NoiseSuppression::findPeaks(
  GPUglobalref() const uchar* peakMap,
  const GlobalPad gpad,
  const Timestamp time,
  bool debug)
{
  ulong peaks = 0;
  if (debug) {
    printf("Looking around %d, %d\n", gpad, time);
  }
  for (int i = 0; i < NOISE_SUPPRESSION_NEIGHBOR_NUM; i++) {
    Delta2 d = CfConsts::NoiseSuppressionNeighbors[i];
    Delta dp = d.x;
    Delta dt = d.y;

    uchar p = IS_PEAK(peakMap, gpad + dp, time + dt);

    if (debug) {
      printf("%d, %d: %d\n", dp, dt, p);
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
  GPUglobalref() const PackedCharge* chargeMap,
  GPUglobalref() const uchar* peakMap,
  float q,
  GlobalPad gpad,
  Timestamp time,
  GPUsharedref() ChargePos* posBcast,
  GPUsharedref() PackedCharge* buf,
  ulong* minimas,
  ulong* bigger,
  ulong* peaks)
{
  ushort ll = get_local_id(0);

  posBcast[ll] = (ChargePos){gpad, time};
  GPUbarrier();

#if defined(GPUCA_GPUCODE)
  ushort wgSizeHalf = SCRATCH_PAD_WORK_GROUP_SIZE / 2;
#else
  ushort wgSizeHalf = 1;
#endif
  bool inGroup1 = ll < wgSizeHalf;
  ushort llhalf = (inGroup1) ? ll : (ll - wgSizeHalf);

  *minimas = 0;
  *bigger = 0;
  *peaks = 0;

  /**************************************
   * Look for minima
   **************************************/

  CfUtils::fillScratchPad_PackedCharge(
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

  CfUtils::fillScratchPad_PackedCharge(
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

  CfUtils::fillScratchPad_PackedCharge(
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
  CfUtils::fillScratchPad_PackedCharge(
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

  CfUtils::fillScratchPad_PackedCharge(
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

  CfUtils::fillScratchPad_uchar(
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

  CfUtils::fillScratchPad_uchar(
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
