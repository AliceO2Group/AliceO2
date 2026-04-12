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

using namespace o2::gpu;
using namespace o2::gpu::tpccf;

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
  CfArray2D<PackedCharge> chargeMap(reinterpret_cast<PackedCharge*>(clusterer.mPchargeMap));

  constexpr GPUTPCGeometry geo;

  const auto iRow = iBlock;
  const auto nPads = geo.NPads(iRow);
  const CfChargePos basePos{(Row)iRow, 0, 0};

  int32_t totalCharges = 0;
  int32_t consecCharges = 0;
  int32_t maxConsecCharges = 0;
  Charge maxCharge = 0;

  const int16_t iPadOffset = iThread % MaxNPadsPerRow;
  const int16_t iTimeOffset = iThread / MaxNPadsPerRow;
  const int16_t iPadHandle = iThread;
  const bool handlePad = iPadHandle < nPads;

  const auto firstTB = fragment.firstNonOverlapTimeBin();
  const auto lastTB = fragment.lastNonOverlapTimeBin();

  for (auto t = firstTB; t < lastTB; t += NumOfCachedTBs) {

    const TPCFragmentTime iTime = t + iTimeOffset;

    const CfChargePos pos = basePos.delta({iPadOffset, iTime});

    smem.charges[iTimeOffset][iPadOffset] = iTime < lastTB && iPadOffset < nPads ? chargeMap[pos].unpack() : 0;

    GPUbarrier();

    if (handlePad) {
      for (int32_t i = 0; i < NumOfCachedTBs; i++) {
        const Charge q = smem.charges[i][iPadHandle];
        totalCharges += (q > 0);
        consecCharges = (q > 0) ? consecCharges + 1 : 0;
        maxConsecCharges = CAMath::Max(consecCharges, maxConsecCharges);
        maxCharge = CAMath::Max<Charge>(q, maxCharge);
      }
    }

    GPUbarrier();
  }

  if (handlePad) {
    updatePadBaseline(basePos.gpad + iPadHandle, clusterer, totalCharges, maxConsecCharges, maxCharge);
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
