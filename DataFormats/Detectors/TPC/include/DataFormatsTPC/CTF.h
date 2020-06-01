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

/// wrapper for the Entropy-encoded clusters of the TF
struct CTF : public o2::ctf::EncodedBlocks<CompressedClustersCounters, 23, uint32_t> {

  static constexpr size_t N = getNBlocks();
  enum Slots { BLCqTotA,
               BLCqMaxA,
               BLCflagsA,
               BLCrowDiffA,
               BLCsliceLegDiffA,
               BLCpadResA,
               BLCtimeResA,
               BLCsigmaPadA,
               BLCsigmaTimeA,
               BLCqPtA,
               BLCrowA,
               BLCsliceA,
               BLCtimeA,
               BLCpadA,
               BLCqTotU,
               BLCqMaxU,
               BLCflagsU,
               BLCpadDiffU,
               BLCtimeDiffU,
               BLCsigmaPadU,
               BLCsigmaTimeU,
               BLCnTrackClusters,
               BLCnSliceRowClusters };

  ClassDefNV(CTF, 1);
};

} // namespace tpc
} // namespace o2

#endif
