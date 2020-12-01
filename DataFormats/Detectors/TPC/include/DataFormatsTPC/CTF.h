// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   CTF.h
/// \author ruben.shahoyan@cern.ch
/// \brief  Definitions for TPC CTF data

#ifndef O2_TPC_CTF_H
#define O2_TPC_CTF_H

#include "DetectorsCommonDataFormats/EncodedBlocks.h"
#include "DataFormatsTPC/CompressedClusters.h"

namespace o2
{
namespace tpc
{

struct CTFHeader : public CompressedClustersCounters {
  enum : uint32_t { CombinedColumns = 0x1 };
  uint32_t flags = 0;

  ClassDefNV(CTFHeader, 1);
};

/// wrapper for the Entropy-encoded clusters of the TF
struct CTF : public o2::ctf::EncodedBlocks<CTFHeader, 23, uint32_t> {

  using container_t = o2::ctf::EncodedBlocks<CTFHeader, 23, uint32_t>;

  static constexpr size_t N = getNBlocks();
  static constexpr int NBitsQTot = 16;
  static constexpr int NBitsQMax = 10;
  static constexpr int NBitsSigmaPad = 8;
  static constexpr int NBitsSigmaTime = 8;
  static constexpr int NBitsRowDiff = 8;
  static constexpr int NBitsSliceLegDiff = 7;

  enum Slots { BLCqTotA,
               BLCqMaxA, // can be combined with BLCqTotA
               BLCflagsA,
               BLCrowDiffA,
               BLCsliceLegDiffA, // can be combined with BLCrowDiffA
               BLCpadResA,
               BLCtimeResA,
               BLCsigmaPadA,
               BLCsigmaTimeA, // can be combined with  BLCsigmaPadA
               BLCqPtA,
               BLCrowA,
               BLCsliceA,
               BLCtimeA,
               BLCpadA,
               BLCqTotU,
               BLCqMaxU, // can be combined with BLCqTotU
               BLCflagsU,
               BLCpadDiffU,
               BLCtimeDiffU,
               BLCsigmaPadU,
               BLCsigmaTimeU, // can be combined with BLCsigmaPadU
               BLCnTrackClusters,
               BLCnSliceRowClusters };

  ClassDefNV(CTF, 2);
};

} // namespace tpc
} // namespace o2

#endif
