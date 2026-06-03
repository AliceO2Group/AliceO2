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

/// \file GPUTPCCFCheckPadBaseline.cxx
/// \author Felix Weiglhofer

#include "GPUTPCCFCheckPadBaseline.h"
#include "CfArray2D.h"
#include "PackedCharge.h"
#include "clusterFinderDefs.h"
#include "DataFormatsTPC/ClusterNative.h"

#ifndef GPUCA_GPUCODE
#include "utils/VcShim.h"
#endif

#if 0
#define DPRINT(...) printf(__VA_ARGS__)
#define DPRINTB(...) \
  if (iThread == 0)  \
  printf(__VA_ARGS__)
#define DPRINTB_IF(test, ...) \
  if (iThread == 0 && (test)) \
  printf(__VA_ARGS__)
#else
#define DPRINT(...) ((void)0)
#define DPRINTB(...) ((void)0)
#define DPRINTB_IF(test, ...) ((void)0)
#endif

using namespace o2::gpu;
using namespace o2::gpu::tpccf;

using Kernel = GPUTPCCFCheckPadBaseline;

static GPUdi() HIPTailDescriptor* GetHIPTails(GPUTPCClusterFinder& clusterer, int32_t row)
{
  // HIP TAILS: indexing starts at 1, so 0 index indicates no connection
  return clusterer.mPhipTailsByRow + row * GPUTPCCFHIPClusterizer::MaxHIPTailsPerRow;
}

static GPUdi() Charge UpdateHIPTailFilter(Charge filteredCharge, Charge charge, Charge alpha)
{
  return filteredCharge + alpha * (charge - filteredCharge);
}

static GPUdi() float HIPTailTimeMean(const HIPTailDescriptor& tail)
{
  const float length = tail.tailEnd > tail.tailStart ? float(tail.tailEnd - tail.tailStart) : 1.f;
  return tail.tailStart + 0.5f * (length - 1.f);
}

static GPUdi() float HIPTailTimeVariance(const HIPTailDescriptor& tail)
{
  const float length = tail.tailEnd > tail.tailStart ? float(tail.tailEnd - tail.tailStart) : 1.f;
  return (length * length - 1.f) * (1.f / 12.f);
}

// Collect tails marked for closing across the workgroup using a prefix scan,
// then cooperatively zero the charge map entries for each closed tail.
// Caller must set acc.activeHIPTail.end before calling if the tail is open.
static GPUdi() uint16_t CloseHIPTails(
  Kernel::GPUSharedMemory& smem,
  GPUTPCClusterFinder& clusterer,
  int32_t iThread, int32_t nThreads,
  int16_t iPadHandle,
  CfChargePos basePos,
  CfArray2D<PackedCharge>& chargeMap,
  Kernel::PadChargeAccu& acc,
  bool shouldCloseTail)
{
  const uint32_t row = basePos.row();
  const uint16_t nClosedTails = work_group_count(shouldCloseTail);

  auto* nHIPTails = clusterer.mPnHIPTails;
  auto* hipTails = GetHIPTails(clusterer, row);

  if (nClosedTails > 0) {
    int16_t iClosedTail = work_group_scan_inclusive_add((int16_t)shouldCloseTail) - 1;
    const bool shouldStoreTail = shouldCloseTail && acc.activeHIPTail.Length() > 0;
    uint16_t nStoredTails = work_group_count(shouldStoreTail);
    int16_t iStoredTail = work_group_scan_inclusive_add((int16_t)shouldStoreTail) - 1;

    // Use exactly one atomic add per closing call to reduce differences in
    // tail ordering between runs.
    if (nStoredTails > 0) {
      if (iThread == 0) {
        smem.tailStoreBase = CAMath::AtomicAdd(&nHIPTails[row], (uint32_t)nStoredTails);
      }
      GPUbarrier();
    }
    if (shouldCloseTail) {
      smem.tailsClosedPad[iClosedTail] = iPadHandle;
      smem.tailsClosed[iClosedTail] = acc.activeHIPTail;
      smem.tailsClosedStoreIdx[iClosedTail] = GPUTPCCFHIPTailConnector::MaxHIPTailsPerRow;

      if (shouldStoreTail) {
        const uint32_t idx = smem.tailStoreBase + iStoredTail + 1;
        smem.tailsClosedStoreIdx[iClosedTail] = idx;
        if (idx < GPUTPCCFHIPTailConnector::MaxHIPTailsPerRow) {
          hipTails[idx] = {0, 0, (uint16_t)iPadHandle,
                           (uint16_t)acc.activeHIPTail.start, (uint16_t)acc.activeHIPTail.end,
                           0.f, 0.f};
        }
      }

      acc.tailFilterCharge = 0;
      acc.activeHIPTail.Reset();
    }

    GPUbarrier();
  }

  // TODO: performance improvement -> parallelize this loop across tails
  for (uint16_t iTail = 0; iTail < nClosedTails; iTail++) {
    const auto tailPad = smem.tailsClosedPad[iTail];
    const auto tail = smem.tailsClosed[iTail];
    const uint32_t tailStoreIdx = smem.tailsClosedStoreIdx[iTail];

    Charge qTot = 0.f;
    Charge qMax = 0.f;
    for (uint16_t iTime = iThread; iTime < tail.Length(); iTime += nThreads) {
      const int16_t time = tail.start + iTime;
      auto pos = basePos.delta({tailPad, time});
      const Charge q = chargeMap[pos].unpack();
      qTot += q;
      qMax = CAMath::Max(qMax, q);
      chargeMap[pos] = PackedCharge{0};
    }

    smem.tailQTotScratch[iThread] = qTot;
    smem.tailQMaxScratch[iThread] = qMax;
    GPUbarrier();
    for (uint16_t active = nThreads; active > 1;) {
      const uint16_t stride = (active + 1) / 2;
      if (iThread < active - stride) {
        smem.tailQTotScratch[iThread] += smem.tailQTotScratch[iThread + stride];
        smem.tailQMaxScratch[iThread] = CAMath::Max(smem.tailQMaxScratch[iThread], smem.tailQMaxScratch[iThread + stride]);
      }
      active = stride;
      GPUbarrier();
    }

    if (iThread == 0 && tailStoreIdx < GPUTPCCFHIPTailConnector::MaxHIPTailsPerRow) {
      HIPTailDescriptor& tailDescriptor = hipTails[tailStoreIdx];
      tailDescriptor.qTot = smem.tailQTotScratch[0];
      tailDescriptor.qMax = smem.tailQMaxScratch[0];
    }
  }

  return nClosedTails;
}

template <bool CheckHIPTrigger, bool CheckHIPTailEnd>
static GPUdi() void ScanCachedCharges(Kernel::GPUSharedMemory& smem, uint16_t timeOffset, uint16_t pad, Charge hipTailThreshold, Charge hipTailFilterAlpha, Kernel::PadChargeAccu& acc)
{
  for (int32_t i = 0; i < Kernel::NumOfCachedTBs; i++) {
    const Charge qs = smem.charges[i][pad];
    const int16_t curTB = timeOffset + i;

    acc.totalCharges += qs > 0;
    acc.consecCharges = qs > 0 ? acc.consecCharges + 1 : 0;
    acc.maxConsecCharges = CAMath::Max(acc.consecCharges, acc.maxConsecCharges);
    acc.maxCharge = CAMath::Max<Charge>(qs, acc.maxCharge);

    if (qs >= hipTailThreshold) {
      if (acc.aboveThresholdStart < 0) {
        acc.aboveThresholdStart = curTB;
      }
    } else {
      acc.aboveThresholdStart = -1;
    }

    if constexpr (CheckHIPTrigger) {
      if (acc.HIPtb < 0 && qs >= Charge(Kernel::MaxADC)) {
        acc.HIPtb = acc.aboveThresholdStart; // start of rising edge, not first sat TB
        smem.tails[pad] = {acc.HIPtb, 0};    // Broadcast HIP start TB to neighboring pads / threads
      }
    }

    if constexpr (CheckHIPTailEnd) {
      if (acc.activeHIPTail.IsOpen()) {
        acc.tailFilterCharge = UpdateHIPTailFilter(acc.tailFilterCharge, qs, hipTailFilterAlpha);
        if (acc.tailFilterCharge < hipTailThreshold) {
          acc.activeHIPTail.end = curTB;
        }
      }
    }
  }
}

template <>
GPUd() void GPUTPCCFCheckPadBaseline::Thread<0>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUSharedMemory& smem, processorType& clusterer)
{
#ifdef GPUCA_GPUCODE
  CheckBaselineGPU(nBlocks, nThreads, iBlock, iThread, smem, clusterer);
#else
  CheckBaselineCPU(nBlocks, nThreads, iBlock, iThread, smem, clusterer);
#endif
}

// Charges are stored in a 2D array (pad and time) using a tiling layout.
// Tiles are 8 pads x 4 timebins large stored in time-major layout and make up a single cacheline.
//
// This kernel processes one row per block. Threads cooperatively load chunks
// of 4 consecutive time bins for all pads into shared memory. Thread `i` then processes charges for pad `i` in shared memory.
// Blocks require `nextMultipleOf<64>(138 * 4) = 576` threads to process the largest TPC rows with 138 pads correctly.
GPUd() void GPUTPCCFCheckPadBaseline::CheckBaselineGPU(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUSharedMemory& smem, processorType& clusterer)
{
#ifdef GPUCA_GPUCODE
  static_assert(GPUCA_GET_THREAD_COUNT(GPUCA_LB_GPUTPCCFCheckPadBaseline) == 576);
  if (iBlock >= (int32_t)GPUTPCGeometry::NROWS) {
    return;
  }

  const CfFragment& fragment = clusterer.mPmemory->fragment;
  const bool hipFilterOn = clusterer.Param().rec.tpc.hipTailFilter;
  const Charge hipTailThreshold = clusterer.Param().rec.tpc.hipTailFilterThreshold;
  const Charge hipTailFilterAlpha = clusterer.Param().rec.tpc.hipTailFilterAlpha;
  CfArray2D<PackedCharge> chargeMap(reinterpret_cast<PackedCharge*>(clusterer.mPchargeMap));

  constexpr GPUTPCGeometry geo;

  const auto iRow = iBlock;
  const auto nPads = geo.NPads(iRow);
  const CfChargePos basePos{(Row)iRow, 0, 0};

  PadChargeAccu acc;

  const int16_t iPadOffset = iThread % MaxNPadsPerRow;
  const int16_t iTimeOffset = iThread / MaxNPadsPerRow;
  const int16_t iPadHandle = iThread;
  const bool handlePad = iPadHandle < nPads;

  if (iPadHandle < MaxNPadsPerRow) {
    smem.tails[iPadHandle] = {-1, -1};
  }
  GPUbarrier();

  // Pad filter scans the entire fragments including overlap.
  // Minimal runtime overhead and prevents headaches later on as
  // saturated signal in overlap region can create tails in the next fragment
  // even when cleared in current fragment as they're decoded twice
  const TPCFragmentTime firstTB = 0;
  const TPCFragmentTime lastTB = fragment.length;

  for (uint16_t t = firstTB; t < lastTB; t += NumOfCachedTBs) {

    bool thisThreadHasTrigger = false;
    for (uint16_t tt = 0; tt < NumOfCachedTBs; tt += TimebinsPerCacheline) {
      const TPCFragmentTime iTimeLoad = t + tt + iTimeOffset;

      const CfChargePos pos = basePos.delta({iPadOffset, iTimeLoad});

      const Charge ql = iTimeLoad < lastTB && iPadOffset < nPads ? chargeMap[pos].unpack() : 0;
      smem.charges[tt + iTimeOffset][iPadOffset] = ql;

      thisThreadHasTrigger |= ql >= Charge(MaxADC);
    }

    bool hasHIPTrigger = false;
    if (hipFilterOn) {
      hasHIPTrigger = work_group_any(thisThreadHasTrigger);
    } else {
      // Need a barrier here even if HIP filter is disabled
      GPUbarrier();
    }

    acc.HIPtb = -1;

    if (handlePad) {

      // TODO: is this really necessary?
      // Why is the old version so much slower, when we just add short branches to the loop???
      if (!hasHIPTrigger) [[likely]] {
        if (!acc.activeHIPTail.IsOpen()) {
          ScanCachedCharges<false, false>(smem, t, iPadHandle, hipTailThreshold, hipTailFilterAlpha, acc);
        } else {
          ScanCachedCharges<false, true>(smem, t, iPadHandle, hipTailThreshold, hipTailFilterAlpha, acc);
        }
      } else {
        if (!acc.activeHIPTail.IsOpen()) {
          ScanCachedCharges<true, false>(smem, t, iPadHandle, hipTailThreshold, hipTailFilterAlpha, acc);
        } else {
          ScanCachedCharges<true, true>(smem, t, iPadHandle, hipTailThreshold, hipTailFilterAlpha, acc);
        }
      }
    }

    GPUbarrier();

    if (hasHIPTrigger) [[unlikely]] {

      DPRINTB("%d: Trigger!\n", iBlock);

      if (handlePad && acc.HIPtb < 0) {

        // Search neighboring pads for trigger
        for (int16_t i = -SSClusterPadWidth; i < 0; i++) {
          const auto p = iPadHandle + i;
          if (p > -1) {
            acc.HIPtb = CAMath::Max(smem.tails[p].start, acc.HIPtb);
          }
        }

        for (int16_t i = 1; i <= SSClusterPadWidth; i++) {
          const auto p = iPadHandle + i;
          if (p < MaxNPadsPerRow) {
            acc.HIPtb = CAMath::Max(smem.tails[p].start, acc.HIPtb);
          }
        }
      }

      bool shouldCloseTail = acc.HIPtb > -1 && acc.activeHIPTail.HasValue();
      if (shouldCloseTail && acc.activeHIPTail.IsOpen()) {
        DPRINT("%d: end = %d\n", iThread, acc.HIPtb);
        acc.activeHIPTail.end = acc.HIPtb;
      }

      CloseHIPTails(smem, clusterer, iThread, nThreads, iPadHandle, basePos, chargeMap, acc, shouldCloseTail);

      GPUbarrier();

      if (acc.HIPtb > -1) {
        DPRINT("%d: start = %d\n", iThread, acc.HIPtb);
        acc.activeHIPTail.SetOpen(acc.HIPtb);
        acc.tailFilterCharge = Charge(MaxADC);
      }

      // Clear smem between iterations to prevent stale entries
      if (handlePad) {
        smem.tails[iPadHandle].Reset();
      }

      GPUbarrier();

    } // if (hipTriggerFound)

  } // for (uint16_t t = firstTB; t < lastTB; t += NumOfCachedTBs)

  if (handlePad) {
    updatePadBaseline(basePos.gpad + iPadHandle, clusterer, acc.totalCharges, acc.maxConsecCharges, acc.maxCharge);
  }

  // --- Close remaining tails
  const bool shouldCloseTail = acc.activeHIPTail.HasValue();

  // Call `work_group_any` here, instead of always counting.
  // This is important as `work_group_count` is a lot slower
  // and has a lot of overhead if no HIPs were found.
  if (work_group_any(shouldCloseTail)) {
    if (shouldCloseTail && acc.activeHIPTail.IsOpen()) {
      acc.activeHIPTail.end = lastTB;
    }

    [[maybe_unused]] const uint16_t nClosedTails = CloseHIPTails(smem, clusterer, iThread, nThreads, iPadHandle, basePos, chargeMap, acc, shouldCloseTail);

    DPRINTB_IF(nClosedTails > 0, "%d: Close remaining tails (%d)\n", iBlock, nClosedTails);
  }

#endif
}

GPUd() void GPUTPCCFCheckPadBaseline::CheckBaselineCPU(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUSharedMemory& smem, processorType& clusterer)
{
#ifndef GPUCA_GPUCODE
  const CfFragment& fragment = clusterer.mPmemory->fragment;
  CfArray2D<PackedCharge> chargeMap(reinterpret_cast<PackedCharge*>(clusterer.mPchargeMap));

  CfChargePos basePos(iBlock * PadsPerCacheline, 0);

  constexpr GPUTPCGeometry geo;
  if (basePos.pad() >= geo.NPads(basePos.row())) {
    return;
  }

  constexpr size_t ElemsInTileRow = (size_t)TilingLayout<GridSize<2>>::WidthInTiles * TimebinsPerCacheline * PadsPerCacheline;

  using UShort8 = Vc::fixed_size_simd<uint16_t, PadsPerCacheline>;
  using Charge8 = Vc::fixed_size_simd<float, PadsPerCacheline>;

  UShort8 totalCharges{Vc::Zero};
  UShort8 consecCharges{Vc::Zero};
  UShort8 maxConsecCharges{Vc::Zero};
  Charge8 maxCharge{Vc::Zero};

  tpccf::TPCFragmentTime t = fragment.firstNonOverlapTimeBin();

  // Access packed charges as raw integers. We throw away the PackedCharge type here to simplify vectorization.
  const uint16_t* packedChargeStart = reinterpret_cast<uint16_t*>(&chargeMap[basePos.delta({0, t})]);

  for (; t < fragment.lastNonOverlapTimeBin(); t += TimebinsPerCacheline) {
    for (tpccf::TPCFragmentTime localtime = 0; localtime < TimebinsPerCacheline; localtime++) {
      const UShort8 packedCharges{packedChargeStart + PadsPerCacheline * localtime, Vc::Aligned};
      const UShort8::mask_type isCharge = packedCharges != 0;

      if (isCharge.isNotEmpty()) {
        totalCharges(isCharge)++;
        consecCharges += 1;
        consecCharges(not isCharge) = 0;
        maxConsecCharges = Vc::max(consecCharges, maxConsecCharges);

        // Manually unpack charges to float.
        // Duplicated from PackedCharge::unpack to generate vectorized code:
        //   Charge unpack() const { return Charge(mVal & ChargeMask) / Charge(1 << DecimalBits); }
        // Note that PackedCharge has to cut off the highest 2 bits via ChargeMask as they are used for flags by the cluster finder
        // and are not part of the charge value. We can skip this step because the cluster finder hasn't run yet
        // and thus these bits are guarenteed to be zero.
        const Charge8 unpackedCharges = Charge8(packedCharges) / Charge(1 << PackedCharge::DecimalBits);
        maxCharge = Vc::max(maxCharge, unpackedCharges);
      } else {
        consecCharges = 0;
      }
    }

    packedChargeStart += ElemsInTileRow;
  }

  for (tpccf::Pad localpad = 0; localpad < PadsPerCacheline; localpad++) {
    updatePadBaseline(basePos.gpad + localpad, clusterer, totalCharges[localpad], maxConsecCharges[localpad], maxCharge[localpad]);
  }
#endif
}

GPUd() void GPUTPCCFCheckPadBaseline::updatePadBaseline(int32_t pad, const GPUTPCClusterFinder& clusterer, int32_t totalCharges, int32_t consecCharges, Charge maxCharge)
{
  const CfFragment& fragment = clusterer.mPmemory->fragment;
  const int32_t totalChargesBaseline = clusterer.Param().rec.tpc.maxTimeBinAboveThresholdIn1000Bin * fragment.lengthWithoutOverlap() / 1000;
  const int32_t consecChargesBaseline = clusterer.Param().rec.tpc.maxConsecTimeBinAboveThreshold;
  const uint16_t saturationThreshold = clusterer.Param().rec.tpc.noisyPadSaturationThreshold;
  const bool isNoisy = (!saturationThreshold || maxCharge < saturationThreshold) && ((totalChargesBaseline > 0 && totalCharges >= totalChargesBaseline) || (consecChargesBaseline > 0 && consecCharges >= consecChargesBaseline));

  if (isNoisy) {
    clusterer.mPpadIsNoisy[pad] = true;
  }
}

// ======== HIP Tail Connector Kernel ========

template <>
GPUd() void GPUTPCCFHIPTailConnector::Thread<0>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUSharedMemory& smem, processorType& clusterer)
{
  if (iBlock >= (int32_t)GPUTPCGeometry::NROWS) {
    return;
  }
  const uint32_t row = iBlock;

  const uint32_t nTails = CAMath::Min(clusterer.mPnHIPTails[row], (uint32_t)MaxHIPTailsPerRow - 1);

  // HIP TAILS: indexing starts at 1, so 0 index indicates no connection
  HIPTailDescriptor* tails = GetHIPTails(clusterer, row);

#ifdef GPUCA_DETERMINISTIC_MODE
  // Races in tail comparisons and atomic swap can lead to slightly different clusters.
  // So need a sequential fallback for deterministic mode
  if (iThread > 0) {
    return;
  }
  nThreads = 1;
  GPUCommonAlgorithm::sortInBlock(tails + 1, tails + nTails + 1, [](auto&& t1, auto&& t2) {
    if (t1.pad != t2.pad) {
      return t1.pad < t2.pad;
    }
    return t1.tailStart < t2.tailStart;
  });
#endif

  for (uint32_t iTail = iThread + 1; iTail <= nTails; iTail += nThreads) {
    auto* tail = &tails[iTail];

    // TODO: this is needed because tailStarts may vary due to rising edge
    // Better approach would be to also track the triggered timebin and match that instead
    uint16_t overlapWindowStart = tail->tailStart >= 5 ? tail->tailStart - 5 : 0;
    uint16_t overlapWindowEnd = tail->tailStart + 5;

    for (uint32_t jTail = iTail + 1; jTail <= nTails; jTail++) {
      auto* tailNext = &tails[jTail];
      if (tailNext->iPrev > 0) {
        continue;
      }

      const bool overlapPad = tailNext->pad >= tail->pad - GPUTPCCFCheckPadBaseline::SSClusterPadWidth && tailNext->pad <= tail->pad + GPUTPCCFCheckPadBaseline::SSClusterPadWidth;
      const bool overlapTime = tailNext->tailStart >= overlapWindowStart && tailNext->tailStart < overlapWindowEnd;

      if (overlapPad && overlapTime) {
        if (CAMath::AtomicCAS(&tailNext->iPrev, 0u, iTail)) {
          tail->iNext = jTail;
          break;
        }
      }
    }
  }
}

// ======== HIP Clusterizer Kernel ========

template <>
GPUd() void GPUTPCCFHIPClusterizer::Thread<0>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUSharedMemory& smem, processorType& clusterer)
{
  if (iBlock >= (int32_t)GPUTPCGeometry::NROWS) {
    return;
  }

  const uint32_t row = iBlock;
  uint32_t nTails = clusterer.mPnHIPTails[row];
  nTails = CAMath::Min(nTails, (uint32_t)MaxHIPTailsPerRow - 1);

  HIPTailDescriptor* tails = GetHIPTails(clusterer, row);
  const auto& fragment = clusterer.mPmemory->fragment;

  for (uint32_t iTail = iThread + 1; iTail <= nTails; iTail += nThreads) {

    auto* tail = &tails[iTail];

    if (tail->iPrev != 0) {
      continue;
    }

    float qTot = tail->qTot;
    float qMax = tail->qMax;
    const float firstWeight = tail->qTot;
    const float firstPad = tail->pad;
    const float firstTime = HIPTailTimeMean(*tail);
    float padSum = firstWeight * firstPad;
    float padSqSum = firstWeight * firstPad * firstPad;
    float timeSum = firstWeight * firstTime;

    uint32_t tailStart = tail->tailStart;
    uint32_t tailEnd = tail->tailEnd;

    while (tail->iNext != 0) {

      tail = &tails[tail->iNext];

      const float tailWeight = tail->qTot;
      const float tailPad = tail->pad;
      const float tailTime = HIPTailTimeMean(*tail);
      qMax = CAMath::Max(qMax, tail->qMax);
      qTot += tail->qTot;
      padSum += tailWeight * tailPad;
      padSqSum += tailWeight * tailPad * tailPad;
      timeSum += tailWeight * tailTime;
      tailStart = CAMath::Min<uint32_t>(tailStart, tail->tailStart);
      tailEnd = CAMath::Max<uint32_t>(tailEnd, tail->tailEnd);
    }

    const float weightSum = CAMath::Max(qTot, 1.f);
    float padMean = padSum / weightSum;
    float timeMean = timeSum / weightSum; // TODO: Use timebin of saturated signal instead! Time mean is biased for long tails.
    float padSigma = CAMath::Sqrt(CAMath::Max(0.f, padSqSum / weightSum - padMean * padMean));

    tpc::ClusterNative cn;
    cn.qMax = qMax;
    cn.setSaturatedQtot(qTot);
    cn.setSaturatedTailLength(tailEnd - tailStart);
    float clusterTime = fragment.start + timeMean - clusterer.Param().rec.tpc.clustersShiftTimebinsClusterizer;
    cn.setTimeFlags(clusterTime, 0);
    cn.setPad(padMean);
    cn.setSigmaPad(padSigma);

    if (cn.qMax >= 1023) {
      // Cut off clusters where the tail connection failed for some reason
      // TODO: Deduplicate with GPUTPCCFClusterizer::sortIntoBuckets (can't call cross-kernel).
      // TODO: Add error reporting for row cluster overflow.
      uint32_t index = CAMath::AtomicAdd(&clusterer.mPclusterInRow[row], 1u);
      if (index < clusterer.mNMaxClusterPerRow) {
        clusterer.mPclusterByRow[clusterer.mNMaxClusterPerRow * row + index] = cn;
      }
    }
  }
}
