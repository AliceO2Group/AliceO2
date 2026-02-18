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

// Collect tails marked for closing across the workgroup using a prefix scan,
// then cooperatively zero the charge map entries for each closed tail.
// Caller must set acc.activeHIPTail.end before calling if the tail is open.
static GPUdi() uint16_t CloseHIPTails(
  Kernel::GPUSharedMemory& smem,
  int32_t iThread, int32_t nThreads,
  int16_t iPadHandle,
  CfChargePos basePos,
  CfArray2D<PackedCharge>& chargeMap,
  Kernel::PadChargeAccu& acc,
  bool shouldCloseTail)
{
  uint16_t nClosedTails = work_group_count(shouldCloseTail);

  if (nClosedTails > 0) {
    int16_t iClosedTail = work_group_scan_inclusive_add((int16_t)shouldCloseTail) - 1;
    if (shouldCloseTail) {
      smem.tailsClosedPad[iClosedTail] = iPadHandle;
      smem.tailsClosed[iClosedTail] = acc.activeHIPTail;
      acc.activeHIPTail.Reset();
    }

    GPUbarrier();
  }

  // TODO: performance improvement -> parallelize this loop across tails
  for (uint16_t iTail = 0; iTail < nClosedTails; iTail++) {
    const auto tailPad = smem.tailsClosedPad[iTail];
    const auto tail = smem.tailsClosed[iTail];

    for (uint16_t iTime = iThread; iTime < tail.Length(); iTime += nThreads) {
      chargeMap[basePos.delta({tailPad, int16_t(tail.start + iTime)})] = PackedCharge{0};
    }
  }

  return nClosedTails;
}

template <bool CheckHIPTrigger, bool CheckHIPTailEnd>
static GPUdi() void ScanCachedCharges(Kernel::GPUSharedMemory& smem, uint16_t timeOffset, uint16_t pad, Charge hipTailThreshold, Kernel::PadChargeAccu& acc)
{
  for (int32_t i = 0; i < Kernel::NumOfCachedTBs; i++) {
    const Charge qs = smem.charges[i][pad];
    acc.totalCharges += qs > 0;
    acc.consecCharges = qs > 0 ? acc.consecCharges + 1 : 0;
    acc.maxConsecCharges = CAMath::Max(acc.consecCharges, acc.maxConsecCharges);
    acc.maxCharge = CAMath::Max<Charge>(qs, acc.maxCharge);

    if constexpr (CheckHIPTrigger) {
      if (qs >= Charge(Kernel::MaxADC)) {
        acc.HIPtb = timeOffset + i;
        smem.tails[pad] = {acc.HIPtb, 0}; // Broadcast HIP start TB to neighboring pads / threads
      }
    }

    if constexpr (CheckHIPTailEnd) {
      if (qs < hipTailThreshold) {
        acc.activeHIPTail.end = timeOffset + i;
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
  if (iBlock >= GPUCA_ROW_COUNT) {
    return;
  }

  const CfFragment& fragment = clusterer.mPmemory->fragment;
  const bool hipFilterOn = clusterer.Param().rec.tpc.hipTailFilter;
  const Charge hipTailThreshold = clusterer.Param().rec.tpc.hipTailFilterThreshold;
  CfArray2D<PackedCharge> chargeMap(reinterpret_cast<PackedCharge*>(clusterer.mPchargeMap));

  const auto iRow = iBlock;
  const auto rowinfo = GetRowInfo(iRow);
  const CfChargePos basePos{(Row)iRow, 0, 0};

  PadChargeAccu acc;

  const int16_t iPadOffset = iThread % MaxNPadsPerRow;
  const int16_t iTimeOffset = iThread / MaxNPadsPerRow;
  const int16_t iPadHandle = iThread;
  const bool handlePad = iPadHandle < rowinfo.nPads;

  if (iPadHandle < MaxNPadsPerRow) {
    smem.tails[iPadHandle] = {-1, -1};
  }
  GPUbarrier();

  const auto firstTB = fragment.firstNonOverlapTimeBin();
  const auto lastTB = fragment.lastNonOverlapTimeBin();

  for (uint16_t t = firstTB; t < lastTB; t += NumOfCachedTBs) {

    const TPCFragmentTime iTimeLoad = t + iTimeOffset;

    const CfChargePos pos = basePos.delta({iPadOffset, iTimeLoad});

    const Charge ql = iTimeLoad < lastTB && iPadOffset < rowinfo.nPads ? chargeMap[pos].unpack() : 0;
    smem.charges[iTimeOffset][iPadOffset] = ql;

    const bool hasHIPTrigger = hipFilterOn && work_group_any(ql >= MaxADC);

    acc.HIPtb = -1;

    if (handlePad) {

      // TODO: is this really necessary?
      // Why is the old version so much slower, when we just add short branches to the loop???
      if (!hasHIPTrigger) [[likely]] {
        if (!acc.activeHIPTail.IsOpen()) {
          ScanCachedCharges<false, false>(smem, t, iPadHandle, hipTailThreshold, acc);
        } else {
          ScanCachedCharges<false, true>(smem, t, iPadHandle, hipTailThreshold, acc);
        }
      } else {
        if (!acc.activeHIPTail.IsOpen()) {
          ScanCachedCharges<true, false>(smem, t, iPadHandle, hipTailThreshold, acc);
        } else {
          ScanCachedCharges<true, true>(smem, t, iPadHandle, hipTailThreshold, acc);
        }
      }
    }

    GPUbarrier();

    if (hasHIPTrigger) [[unlikely]] {

      DPRINTB("%d: Trigger!\n", iBlock);

      if (handlePad && acc.HIPtb < 0) {

        // Search neighboring pads for trigger
        for (int16_t i = -3; i < 0; i++) {
          const auto p = iPadHandle + i;
          if (p > -1) {
            acc.HIPtb = CAMath::Max(smem.tails[p].start, acc.HIPtb);
          }
        }

        for (int16_t i = 1; i < 4; i++) {
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

      CloseHIPTails(smem, iThread, nThreads, iPadHandle, basePos, chargeMap, acc, shouldCloseTail);

      GPUbarrier(); // TODO: not needed? Debug only

      if (acc.HIPtb > -1) {
        DPRINT("%d: start = %d\n", iThread, acc.HIPtb);
        acc.activeHIPTail.SetOpen(acc.HIPtb);
      }

      // Clear smem between iterations to prevent stale entries
      if (handlePad) {
        smem.tails[iPadHandle].Reset();
      }

      GPUbarrier(); // TODO: not needed? Debug only

    } // if (hipTriggerFound)

  } // for (uint16_t t = firstTB; t < lastTB; t += NumOfCachedTBs)

  if (handlePad) {
    updatePadBaseline(rowinfo.globalPadOffset + iPadOffset, clusterer, acc.totalCharges, acc.maxConsecCharges, acc.maxCharge);
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

    [[maybe_unused]] const uint16_t nClosedTails = CloseHIPTails(smem, iThread, nThreads, iPadHandle, basePos, chargeMap, acc, shouldCloseTail);

    DPRINTB_IF(nClosedTails > 0, "%d: Close remaining tails (%d)\n", iBlock, nClosedTails);
  }

#endif
}

GPUd() void GPUTPCCFCheckPadBaseline::CheckBaselineCPU(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUSharedMemory& smem, processorType& clusterer)
{
#ifndef GPUCA_GPUCODE
  const CfFragment& fragment = clusterer.mPmemory->fragment;
  CfArray2D<PackedCharge> chargeMap(reinterpret_cast<PackedCharge*>(clusterer.mPchargeMap));

  int32_t basePad = iBlock * PadsPerCacheline;
  int32_t padsPerRow;
  CfChargePos basePos = padToCfChargePos<PadsPerCacheline>(basePad, clusterer, padsPerRow);

  if (!basePos.valid()) {
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
    updatePadBaseline(basePad + localpad, clusterer, totalCharges[localpad], maxConsecCharges[localpad], maxCharge[localpad]);
  }
#endif
}

template <int32_t PadsPerBlock>
GPUd() CfChargePos GPUTPCCFCheckPadBaseline::padToCfChargePos(int32_t& pad, const GPUTPCClusterFinder& clusterer, int32_t& padsPerRow)
{
  constexpr GPUTPCGeometry geo;

  int32_t padOffset = 0;
  for (Row r = 0; r < GPUCA_ROW_COUNT; r++) {
    int32_t npads = geo.NPads(r);
    int32_t padInRow = pad - padOffset;
    if (0 <= padInRow && padInRow < npads) {
      int32_t cachelineOffset = padInRow % PadsPerBlock;
      pad -= cachelineOffset;
      padsPerRow = npads;
      return CfChargePos{r, Pad(padInRow - cachelineOffset), 0};
    }
    padOffset += npads;
  }

  padsPerRow = 0;
  return CfChargePos{0, 0, INVALID_TIME_BIN};
}

GPUd() GPUTPCCFCheckPadBaseline::RowInfo GPUTPCCFCheckPadBaseline::GetRowInfo(int16_t row)
{
  constexpr GPUTPCGeometry geo;

  int16_t padOffset = 0;
  for (int16_t r = 0; r < row; r++) {
    padOffset += geo.NPads(r);
  }

  return RowInfo{padOffset, geo.NPads(row)};
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
