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

#include "GPUTPCCFPeakFinder.h"

#include "Array2D.h"
#include "CfUtils.h"
#include "PackedCharge.h"
#include "TPCPadGainCalib.h"

using namespace GPUCA_NAMESPACE::gpu;
using namespace GPUCA_NAMESPACE::gpu::tpccf;

template <>
GPUdii() void GPUTPCCFPeakFinder::Thread<0>(int nBlocks, int nThreads, int iBlock, int iThread, GPUSharedMemory& smem, processorType& clusterer)
{
  Array2D<PackedCharge> chargeMap(reinterpret_cast<PackedCharge*>(clusterer.mPchargeMap));
  Array2D<uchar> isPeakMap(clusterer.mPpeakMap);
  findPeaksImpl(get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), smem, chargeMap, clusterer.mPpadHasLostBaseline, clusterer.mPpositions, clusterer.mPmemory->counters.nPositions, clusterer.Param().rec, *clusterer.GetConstantMem()->calibObjects.tpcPadGain, clusterer.mPisPeak, isPeakMap);
}

GPUdii() bool GPUTPCCFPeakFinder::isPeak(
  GPUSharedMemory& smem,
  Charge q,
  const ChargePos& pos,
  ushort N,
  const Array2D<PackedCharge>& chargeMap,
  const GPUSettingsRec& calib,
  ChargePos* posBcast,
  PackedCharge* buf)
{
  ushort ll = get_local_id(0);

  bool belowThreshold = (q <= calib.tpcCFqmaxCutoff);

  ushort lookForPeaks;
  ushort partId = CfUtils::partition<SCRATCH_PAD_WORK_GROUP_SIZE>(
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
    cfconsts::InnerNeighbors,
    posBcast,
    buf);

  if (belowThreshold) {
    return false;
  }

  // Ensure q has the same float->int->float conversion error
  // as values in chargeMap, so identical charges are actually identical
  q = PackedCharge(q).unpack();

  int idx = N * partId;
  bool peak = true;
  peak = peak && buf[idx + 0].unpack() <= q;
  peak = peak && buf[idx + 1].unpack() <= q;
  peak = peak && buf[idx + 2].unpack() <= q;
  peak = peak && buf[idx + 3].unpack() <= q;
  peak = peak && buf[idx + 4].unpack() < q;
  peak = peak && buf[idx + 5].unpack() < q;
  peak = peak && buf[idx + 6].unpack() < q;
  peak = peak && buf[idx + 7].unpack() < q;

  return peak;
}

GPUd() void GPUTPCCFPeakFinder::findPeaksImpl(int nBlocks, int nThreads, int iBlock, int iThread, GPUSharedMemory& smem,
                                              const Array2D<PackedCharge>& chargeMap,
                                              const uchar* padHasLostBaseline,
                                              const ChargePos* positions,
                                              SizeT digitnum,
                                              const GPUSettingsRec& calib,
                                              const TPCPadGainCalib& gainCorrection, // Only used for globalPad() function
                                              uchar* isPeakPredicate,
                                              Array2D<uchar>& peakMap)
{
  SizeT idx = get_global_id(0);

  // For certain configurations dummy work items are added, so the total
  // number of work items is dividable by 64.
  // These dummy items also compute the last digit but discard the result.
  ChargePos pos = positions[CAMath::Min(idx, (SizeT)(digitnum - 1))];
  Charge charge = pos.valid() ? chargeMap[pos].unpack() : Charge(0);

  bool hasLostBaseline = padHasLostBaseline[gainCorrection.globalPad(pos.row(), pos.pad())];
  charge = (hasLostBaseline) ? 0.f : charge;

  uchar peak = isPeak(smem, charge, pos, SCRATCH_PAD_SEARCH_N, chargeMap, calib, smem.posBcast, smem.buf);

  // Exit early if dummy. See comment above.
  bool iamDummy = (idx >= digitnum);
  if (iamDummy) {
    return;
  }

  isPeakPredicate[idx] = peak;

  peakMap[pos] = (uchar(charge > calib.tpcCFinnerThreshold) << 1) | peak;
}
