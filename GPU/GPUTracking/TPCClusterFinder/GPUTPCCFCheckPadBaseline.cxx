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

/// \file GPUTPCCFCheckPadBaseline.h
/// \author Felix Weiglhofer

#include "GPUTPCCFCheckPadBaseline.h"
#include "CfArray2D.h"
#include "PackedCharge.h"
#include "GPUTPCGeometry.h"
#include "clusterFinderDefs.h"

#ifndef GPUCA_GPUCODE
#include "utils/VcShim.h"
#endif

using namespace o2::gpu;
using namespace o2::gpu::tpccf;

template <>
GPUd() void GPUTPCCFCheckPadBaseline::Thread<0>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUSharedMemory& smem, processorType& clusterer)
{
  const CfFragment& fragment = clusterer.mPmemory->fragment;
  CfArray2D<PackedCharge> chargeMap(reinterpret_cast<PackedCharge*>(clusterer.mPchargeMap));

  int32_t basePad = iBlock * PadsPerCacheline;
  CfChargePos basePos = padToCfChargePos(basePad, clusterer);

  if (not basePos.valid()) {
    return;
  }

#ifdef GPUCA_GPUCODE
  static_assert(TPC_MAX_FRAGMENT_LEN_GPU % NumOfCachedTimebins == 0);

  int32_t totalCharges = 0;
  int32_t consecCharges = 0;
  int32_t maxConsecCharges = 0;
  Charge maxCharge = 0;

  int16_t localPadId = iThread / NumOfCachedTimebins;
  int16_t localTimeBin = iThread % NumOfCachedTimebins;
  bool handlePad = localTimeBin == 0;

  for (tpccf::TPCFragmentTime t = fragment.firstNonOverlapTimeBin(); t < fragment.lastNonOverlapTimeBin(); t += NumOfCachedTimebins) {
    const CfChargePos pos = basePos.delta({localPadId, int16_t(t + localTimeBin)});
    smem.charges[localPadId][localTimeBin] = (pos.valid()) ? chargeMap[pos].unpack() : 0;
    GPUbarrier();
    if (handlePad) {
      for (int32_t i = 0; i < NumOfCachedTimebins; i++) {
        const Charge q = smem.charges[localPadId][i];
        totalCharges += (q > 0);
        consecCharges = (q > 0) ? consecCharges + 1 : 0;
        maxConsecCharges = CAMath::Max(consecCharges, maxConsecCharges);
        maxCharge = CAMath::Max<Charge>(q, maxCharge);
      }
    }
    GPUbarrier();
  }

  GPUbarrier();

  if (handlePad) {
    updatePadBaseline(basePad + localPadId, clusterer, totalCharges, maxConsecCharges, maxCharge);
  }

#else // CPU CODE

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

GPUd() CfChargePos GPUTPCCFCheckPadBaseline::padToCfChargePos(int32_t& pad, const GPUTPCClusterFinder& clusterer)
{
  constexpr GPUTPCGeometry geo;

  int32_t padOffset = 0;
  for (Row r = 0; r < GPUCA_ROW_COUNT; r++) {
    int32_t npads = geo.NPads(r);
    int32_t padInRow = pad - padOffset;
    if (0 <= padInRow && padInRow < CAMath::nextMultipleOf<PadsPerCacheline, int32_t>(npads)) {
      int32_t cachelineOffset = padInRow % PadsPerCacheline;
      pad -= cachelineOffset;
      return CfChargePos{r, Pad(padInRow - cachelineOffset), 0};
    }
    padOffset += npads;
  }

  return CfChargePos{0, 0, INVALID_TIME_BIN};
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
