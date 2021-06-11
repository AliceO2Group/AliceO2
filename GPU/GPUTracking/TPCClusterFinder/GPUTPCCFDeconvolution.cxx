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
#include "GPUDefMacros.h"

using namespace GPUCA_NAMESPACE::gpu;
using namespace GPUCA_NAMESPACE::gpu::tpccf;

template <>
GPUdii() void GPUTPCCFDeconvolution::Thread<0>(int nBlocks, int nThreads, int iBlock, int iThread, GPUSharedMemory& smem, processorType& clusterer)
{
  Array2D<PackedCharge> chargeMap(reinterpret_cast<PackedCharge*>(clusterer.mPchargeMap));
  Array2D<uchar> isPeakMap(clusterer.mPpeakMap);
  GPUTPCCFDeconvolution::deconvolutionImpl(get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), smem, isPeakMap, chargeMap, clusterer.mPpositions, clusterer.mPmemory->counters.nPositions);
}

GPUdii() void GPUTPCCFDeconvolution::deconvolutionImpl(int nBlocks, int nThreads, int iBlock, int iThread, GPUSharedMemory& smem,
                                                       const Array2D<uchar>& peakMap,
                                                       Array2D<PackedCharge>& chargeMap,
                                                       const ChargePos* positions,
                                                       const uint digitnum)
{
  SizeT idx = get_global_id(0);

  bool iamDummy = (idx >= digitnum);
  idx = iamDummy ? digitnum - 1 : idx;

  ChargePos pos = positions[idx];

  bool iamPeak = CfUtils::isPeak(peakMap[pos]);

  char peakCount = (iamPeak) ? 1 : 0;

  ushort ll = get_local_id(0);
  ushort partId = ll;

  ushort in3x3 = 0;
  partId = CfUtils::partition<SCRATCH_PAD_WORK_GROUP_SIZE>(smem, ll, iamPeak, SCRATCH_PAD_WORK_GROUP_SIZE, &in3x3);

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

  uchar aboveThreshold = 0;
  if (partId < in3x3) {
    peakCount = countPeaksInner(partId, smem.buf, &aboveThreshold);
  }

  ushort in5x5 = 0;
  partId = CfUtils::partition<SCRATCH_PAD_WORK_GROUP_SIZE>(smem, partId, peakCount > 0 && !iamPeak, in3x3, &in5x5);

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
    cfconsts::OuterNeighbors,
    smem.posBcast1,
    smem.aboveThresholdBcast,
    smem.buf);

  if (partId < in5x5) {
    peakCount = countPeaksOuter(partId, aboveThreshold, smem.buf);
    peakCount *= -1;
  }

  if (iamDummy) {
    return;
  }

  bool has3x3 = (peakCount > 0);
  peakCount = CAMath::Abs(int(peakCount));
  bool split = (peakCount > 1);

  peakCount = (peakCount == 0) ? 1 : peakCount;

  PackedCharge charge = chargeMap[pos];
  PackedCharge p(charge.unpack() / peakCount, has3x3, split);

  chargeMap[pos] = p;
}

GPUdi() char GPUTPCCFDeconvolution::countPeaksInner(
  ushort ll,
  const uchar* isPeak,
  uchar* aboveThreshold)
{
  char peaks = 0;
  GPUCA_UNROLL(U(), U())
  for (uchar i = 0; i < 8; i++) {
    uchar p = isPeak[ll * 8 + i];
    peaks += CfUtils::isPeak(p);
    *aboveThreshold |= uchar(CfUtils::isAboveThreshold(p)) << i;
  }

  return peaks;
}

GPUdi() char GPUTPCCFDeconvolution::countPeaksOuter(
  ushort ll,
  uchar aboveThreshold,
  const uchar* isPeak)
{
  char peaks = 0;
  GPUCA_UNROLL(U(), U())
  for (uchar i = 0; i < 16; i++) {
    uchar p = isPeak[ll * 16 + i];
    peaks += CfUtils::isPeak(p);
  }

  return peaks;
}
