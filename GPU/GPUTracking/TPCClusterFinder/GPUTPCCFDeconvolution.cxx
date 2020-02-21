// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

using namespace GPUCA_NAMESPACE::gpu;
using namespace GPUCA_NAMESPACE::gpu::deprecated;

template <>
GPUdii() void GPUTPCCFDeconvolution::Thread<GPUTPCCFDeconvolution::countPeaks>(int nBlocks, int nThreads, int iBlock, int iThread, GPUSharedMemory& smem, processorType& clusterer)
{
  Array2D<PackedCharge> chargeMap(reinterpret_cast<PackedCharge*>(clusterer.mPchargeMap));
  Array2D<uchar> isPeakMap(clusterer.mPpeakMap);
  GPUTPCCFDeconvolution::countPeaksImpl(get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), smem, isPeakMap, chargeMap, clusterer.mPdigits, clusterer.mPmemory->counters.nDigits);
}

GPUd() void GPUTPCCFDeconvolution::countPeaksImpl(int nBlocks, int nThreads, int iBlock, int iThread, GPUSharedMemory& smem,
                                                  const Array2D<uchar>& peakMap,
                                                  Array2D<PackedCharge>& chargeMap,
                                                  const Digit* digits,
                                                  const uint digitnum)
{
  size_t idx = get_global_id(0);

  bool iamDummy = (idx >= digitnum);
  idx = iamDummy ? digitnum - 1 : idx;

  Digit myDigit = digits[idx];

  ChargePos pos(myDigit);

  bool iamPeak = GET_IS_PEAK(peakMap[pos]);

  char peakCount = (iamPeak) ? 1 : 0;

#if defined(BUILD_CLUSTER_SCRATCH_PAD)
  /* #if defined(BUILD_CLUSTER_SCRATCH_PAD) && defined(GPUCA_GPUCODE) */
  /* #if 0 */
  ushort ll = get_local_id(0);
  ushort partId = ll;

  ushort in3x3 = 0;
  partId = CfUtils::partition(smem, ll, iamPeak, SCRATCH_PAD_WORK_GROUP_SIZE, &in3x3);

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
    CfConsts::InnerNeighbors,
    smem.posBcast1,
    smem.buf);

  uchar aboveThreshold = 0;
  if (partId < in3x3) {
    peakCount = countPeaksScratchpadInner(partId, smem.buf, &aboveThreshold);
  }

  ushort in5x5 = 0;
  partId = CfUtils::partition(smem, partId, peakCount > 0 && !iamPeak, in3x3, &in5x5);

  if (partId < in5x5) {
    smem.posBcast1[partId] = pos;
    smem.aboveThresholdBcast[partId] = aboveThreshold;
  }
  GPUbarrier();

  CfUtils::condBlockLoad<uchar, true>(
    peakMap,
    in5x5,
    SCRATCH_PAD_WORK_GROUP_SIZE,
    ll,
    0,
    16,
    CfConsts::OuterNeighbors,
    smem.posBcast1,
    smem.aboveThresholdBcast,
    smem.buf);

  if (partId < in5x5) {
    peakCount = countPeaksScratchpadOuter(partId, 0, aboveThreshold, smem.buf);
    peakCount *= -1;
  }

#else
  peakCount = countPeaksAroundDigit(pos, peakMap);
  peakCount = iamPeak ? 1 : peakCount;
#endif

  if (iamDummy) {
    return;
  }

  bool has3x3 = (peakCount > 0);
  peakCount = CAMath::Abs(int(peakCount));
  bool split = (peakCount > 1);

  peakCount = (peakCount == 0) ? 1 : peakCount;

  PackedCharge p(myDigit.charge / peakCount, has3x3, split);

  chargeMap[pos] = p;
}

GPUd() char GPUTPCCFDeconvolution::countPeaksAroundDigit(
  const ChargePos& pos,
  const Array2D<uchar>& peakMap)
{
  char peakCount = 0;

  uchar aboveThreshold = 0;
  for (uchar i = 0; i < 8; i++) {
    Delta2 d = CfConsts::InnerNeighbors[i];

    uchar p = peakMap[pos.delta(d)];
    peakCount += GET_IS_PEAK(p);
    aboveThreshold |= GET_IS_ABOVE_THRESHOLD(p) << i;
  }

  if (peakCount > 0) {
    return peakCount;
  }

  for (uchar i = 0; i < 16; i++) {
    Delta2 d = CfConsts::OuterNeighbors[i];

    if (CfUtils::innerAboveThresholdInv(aboveThreshold, i)) {
      peakCount -= GET_IS_PEAK(peakMap[pos.delta(d)]);
    }
  }

  return peakCount;
}

GPUd() char GPUTPCCFDeconvolution::countPeaksScratchpadInner(
  ushort ll,
  const uchar* isPeak,
  uchar* aboveThreshold)
{
  char peaks = 0;
  for (uchar i = 0; i < 8; i++) {
    uchar p = isPeak[ll * 8 + i];
    peaks += GET_IS_PEAK(p);
    *aboveThreshold |= uchar(GET_IS_ABOVE_THRESHOLD(p)) << i;
  }

  return peaks;
}

GPUd() char GPUTPCCFDeconvolution::countPeaksScratchpadOuter(
  ushort ll,
  ushort offset,
  uchar aboveThreshold,
  const uchar* isPeak)
{
  char peaks = 0;
  for (uchar i = 0; i < 16; i++) {
    uchar p = isPeak[ll * 16 + i];
    peaks += GET_IS_PEAK(p);
  }

  return peaks;
}
