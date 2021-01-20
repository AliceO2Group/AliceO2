// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTPCCFCheckPadBaseline.h
/// \author Felix Weiglhofer

#include "GPUTPCCFCheckPadBaseline.h"
#include "Array2D.h"
#include "PackedCharge.h"
#include "clusterFinderDefs.h"

#ifndef GPUCA_GPUCODE
#ifndef GPUCA_NO_VC
#include <Vc/Vc>
#else
#include <array>
#endif
#endif

using namespace GPUCA_NAMESPACE::gpu;
using namespace GPUCA_NAMESPACE::gpu::tpccf;

template <>
GPUd() void GPUTPCCFCheckPadBaseline::Thread<0>(int nBlocks, int nThreads, int iBlock, int iThread, GPUSharedMemory& smem, processorType& clusterer)
{
  const CfFragment& fragment = clusterer.mPmemory->fragment;
  Array2D<PackedCharge> chargeMap(reinterpret_cast<PackedCharge*>(clusterer.mPchargeMap));

  int basePad = iBlock * PadsPerCacheline;
  ChargePos basePos = padToChargePos(basePad, clusterer);

  if (not basePos.valid()) {
    return;
  }

#ifdef GPUCA_GPUCODE
  static_assert(TPC_MAX_FRAGMENT_LEN % NumOfCachedTimebins == 0);

  int totalCharges = 0;
  int consecCharges = 0;
  int maxConsecCharges = 0;

  short localPadId = iThread / NumOfCachedTimebins;
  short localTimeBin = iThread % NumOfCachedTimebins;
  bool handlePad = localTimeBin == 0;

  for (tpccf::TPCFragmentTime t = fragment.firstNonOverlapTimeBin(); t < fragment.lastNonOverlapTimeBin(); t += NumOfCachedTimebins) {
    ChargePos pos = basePos.delta({localPadId, short(t + localTimeBin)});
    smem.charges[localPadId][localTimeBin] = (pos.valid()) ? chargeMap[pos].unpack() : 0;
    GPUbarrier();
    if (handlePad) {
      for (int i = 0; i < NumOfCachedTimebins; i++) {
        Charge q = smem.charges[localPadId][i];
        totalCharges += (q > 0);
        consecCharges = (q > 0) ? consecCharges + 1 : 0;
        maxConsecCharges = CAMath::Max(consecCharges, maxConsecCharges);
      }
    }
  }

  GPUbarrier();

  if (handlePad) {
    updatePadBaseline(basePad + localPadId, clusterer, totalCharges, maxConsecCharges);
  }

#else // CPU CODE

  constexpr size_t ElemsInTileRow = TilingLayout<GridSize<2>>::WidthInTiles * TimebinsPerCacheline * PadsPerCacheline;

#ifndef GPUCA_NO_VC
  using UShort8 = Vc::fixed_size_simd<unsigned short, PadsPerCacheline>;

  UShort8 totalCharges{Vc::Zero};
  UShort8 consecCharges{Vc::Zero};
  UShort8 maxConsecCharges{Vc::Zero};
#else
  std::array<unsigned short, PadsPerCacheline> totalCharges{0};
  std::array<unsigned short, PadsPerCacheline> consecCharges{0};
  std::array<unsigned short, PadsPerCacheline> maxConsecCharges{0};
#endif

  tpccf::TPCFragmentTime t = fragment.firstNonOverlapTimeBin();
  const unsigned short* charge = reinterpret_cast<unsigned short*>(&chargeMap[basePos.delta({0, t})]);

  for (; t < fragment.lastNonOverlapTimeBin(); t += TimebinsPerCacheline) {
    for (tpccf::TPCFragmentTime localtime = 0; localtime < TimebinsPerCacheline; localtime++) {
#ifndef GPUCA_NO_VC
      UShort8 charges{charge + PadsPerCacheline * localtime, Vc::Aligned};

      UShort8::mask_type isCharge = charges != 0;

      if (isCharge.isNotEmpty()) {
        totalCharges(isCharge)++;
        consecCharges += 1;
        consecCharges(not isCharge) = 0;
        maxConsecCharges = Vc::max(consecCharges, maxConsecCharges);
      } else {
        consecCharges = 0;
      }
#else // Vc not available
      for (tpccf::Pad localpad = 0; localpad < PadsPerCacheline; localpad++) {
        bool isCharge = charge[PadsPerCacheline * localtime + localpad] != 0;
        if (isCharge) {
          totalCharges[localpad]++;
          consecCharges[localpad]++;
          maxConsecCharges[localpad] = CAMath::Max(maxConsecCharges[localpad], consecCharges[localpad]);
        } else {
          consecCharges[localpad] = 0;
        }
      }
#endif
    }

    charge += ElemsInTileRow;
  }

  for (tpccf::Pad localpad = 0; localpad < PadsPerCacheline; localpad++) {
    updatePadBaseline(basePad + localpad, clusterer, totalCharges[localpad], maxConsecCharges[localpad]);
  }
#endif
}

GPUd() ChargePos GPUTPCCFCheckPadBaseline::padToChargePos(int& pad, const GPUTPCClusterFinder& clusterer)
{
  const GPUTPCGeometry& geo = clusterer.Param().tpcGeometry;

  int padOffset = 0;
  for (Row r = 0; r < TPC_NUM_OF_ROWS; r++) {
    int npads = geo.NPads(r);
    int padInRow = pad - padOffset;
    if (0 <= padInRow && padInRow < CAMath::nextMultipleOf<PadsPerCacheline, int>(npads)) {
      int cachelineOffset = padInRow % PadsPerCacheline;
      pad -= cachelineOffset;
      return ChargePos{r, Pad(padInRow - cachelineOffset), 0};
    }
    padOffset += npads;
  }

  return ChargePos{0, 0, INVALID_TIME_BIN};
}

GPUd() void GPUTPCCFCheckPadBaseline::updatePadBaseline(int pad, const GPUTPCClusterFinder& clusterer, int totalCharges, int consecCharges)
{
  const CfFragment& fragment = clusterer.mPmemory->fragment;
  int totalChargesBaseline = clusterer.Param().rec.maxTimeBinAboveThresholdIn1000Bin * fragment.lengthWithoutOverlap() / 1000;
  int consecChargesBaseline = clusterer.Param().rec.maxConsecTimeBinAboveThreshold;
  bool hasLostBaseline = (totalChargesBaseline > 0 && totalCharges >= totalChargesBaseline) || (consecChargesBaseline > 0 && consecCharges >= consecChargesBaseline);
  clusterer.mPpadHasLostBaseline[pad] |= hasLostBaseline;
}
