// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file PeakFinder.cxx
/// \author Felix Weiglhofer

#include "PeakFinder.h"

#include "Array2D.h"
#include "CfUtils.h"
#include "Digit.h"
#include "PackedCharge.h"

using namespace GPUCA_NAMESPACE::gpu;
using namespace GPUCA_NAMESPACE::gpu::deprecated;

GPUd() bool PeakFinder::isPeakScratchPad(
  GPUTPCClusterFinderKernels::GPUTPCSharedMemory& smem,
  Charge q,
  const ChargePos& pos,
  ushort N,
  const Array2D<o2::gpu::PackedCharge>& chargeMap,
  GPUsharedref() ChargePos* posBcast,
  GPUsharedref() PackedCharge* buf)
{
  ushort ll = get_local_id(0);

  bool belowThreshold = (q <= QMAX_CUTOFF);

  ushort lookForPeaks;
  ushort partId = CfUtils::partition(
    smem,
    ll,
    belowThreshold,
    SCRATCH_PAD_WORK_GROUP_SIZE,
    &lookForPeaks);

  if (partId < lookForPeaks) {
    posBcast[partId] = pos;
  }
  GPUbarrier();

  CfUtils::blockLoad<PackedCharge>(
    chargeMap,
    lookForPeaks,
    SCRATCH_PAD_WORK_GROUP_SIZE,
    ll,
    0,
    N,
    CfConsts::InnerNeighbors,
    posBcast,
    buf);

  if (belowThreshold) {
    return false;
  }

  bool peak = true;
  for (ushort i = 0; i < N; i++) {
    Charge other = buf[N * partId + i].unpack();
    peak &= (q > other) || (CfConsts::InnerTestEq[i] && q == other);
  }

  return peak;
}

GPUd() bool PeakFinder::isPeak(
  Charge myCharge,
  const ChargePos& pos,
  const Array2D<PackedCharge>& chargeMap)
{
  if (myCharge <= QMAX_CUTOFF) {
    return false;
  }

  bool peak = true;

#define CMP_NEIGHBOR(dp, dt, cmpOp)                  \
  do {                                               \
    PackedCharge p = chargeMap[pos.delta({dp, dt})]; \
    const Charge otherCharge = p.unpack();           \
    peak &= (otherCharge cmpOp myCharge);            \
  } while (false)

#define CMP_LT CMP_NEIGHBOR(-1, -1, <=)
#define CMP_T CMP_NEIGHBOR(-1, 0, <=)
#define CMP_RT CMP_NEIGHBOR(-1, 1, <=)

#define CMP_L CMP_NEIGHBOR(0, -1, <=)
#define CMP_R CMP_NEIGHBOR(0, 1, <)

#define CMP_LB CMP_NEIGHBOR(1, -1, <)
#define CMP_B CMP_NEIGHBOR(1, 0, <)
#define CMP_RB CMP_NEIGHBOR(1, 1, <)

#if defined(CHARGEMAP_TILING_LAYOUT)
  CMP_LT;
  CMP_T;
  CMP_RT;
  CMP_R;
  CMP_RB;
  CMP_B;
  CMP_LB;
  CMP_L;
#else
  CMP_LT;
  CMP_T;
  CMP_RT;
  CMP_L;
  CMP_R;
  CMP_LB;
  CMP_B;
  CMP_RB;
#endif

#undef CMP_LT
#undef CMP_T
#undef CMP_RT
#undef CMP_L
#undef CMP_R
#undef CMP_LB
#undef CMP_B
#undef CMP_RB
#undef CMP_NEIGHBOR

  return peak;
}

GPUd() void PeakFinder::findPeaksImpl(int nBlocks, int nThreads, int iBlock, int iThread, GPUTPCClusterFinderKernels::GPUTPCSharedMemory& smem,
                                      const Array2D<PackedCharge>& chargeMap,
                                      GPUglobalref() const Digit* digits,
                                      uint digitnum,
                                      GPUglobalref() uchar* isPeakPredicate,
                                      Array2D<uchar>& peakMap)
{
  size_t idx = get_global_id(0);

  // For certain configurations dummy work items are added, so the total
  // number of work items is dividable by 64.
  // These dummy items also compute the last digit but discard the result.
  Digit myDigit = digits[CAMath::Min(idx, (size_t)(digitnum - 1))];

  const GlobalPad gpad = CfUtils::tpcGlobalPadIdx(myDigit.row, myDigit.pad);

  ChargePos pos(gpad, myDigit.time);

  uchar peak;
#if defined(BUILD_CLUSTER_SCRATCH_PAD)
  peak = isPeakScratchPad(smem, myDigit.charge, pos, SCRATCH_PAD_SEARCH_N, chargeMap, smem.search.posBcast, smem.search.buf);
#else
  peak = isPeak(myDigit.charge, pos, chargeMap);
#endif

  // Exit early if dummy. See comment above.
  bool iamDummy = (idx >= digitnum);
  if (iamDummy) {
    return;
  }

  isPeakPredicate[idx] = peak;

  peakMap[pos] = (uchar(myDigit.charge > CHARGE_THRESHOLD) << 1) | peak;
}
