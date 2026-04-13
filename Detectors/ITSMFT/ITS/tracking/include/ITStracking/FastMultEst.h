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

/// \file  FastMultEst.h
/// \brief Fast multiplicity estimator for ITS
/// \author ruben.shahoyan@cern.ch

#ifndef ALICEO2_ITS_FASTMULTEST_
#define ALICEO2_ITS_FASTMULTEST_

#include "ITSMFTReconstruction/ChipMappingITS.h"
#include "DataFormatsITS/Vertex.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITSMFT/PhysTrigger.h"
#include "ITStracking/FastMultEstConfig.h"
#include "ITStracking/ROFLookupTables.h"
#include <gsl/span>
#include <array>

namespace o2::its
{

struct FastMultEst {

  static constexpr int NLayers = o2::itsmft::ChipMappingITS::NLayers;
  using ROFOverlapTableN = ROFOverlapTable<NLayers>;
  using ROFMaskTableN = ROFMaskTable<NLayers>;

  float mult = 0.;             /// estimated signal clusters multiplicity on the selected multiplicity layer
  float noisePerChip = 0.;     /// imposed noise per chip (when enabled by configuration)
  float cov[3] = {0.};         /// retained for compatibility; set to zero in single-layer mode
  float chi2 = 0.;             /// retained for compatibility; set to zero in single-layer mode
  int nLayersUsed = 0;         /// number of layers used by estimator (0/1 in single-layer mode)
  uint32_t lastRandomSeed = 0; /// state of the gRandom before
  FastMultEst();

  static uint32_t getCurrentRandomSeed();
  int selectROFs(const std::array<gsl::span<const o2::itsmft::ROFRecord>, NLayers>& rofs,
                 const std::array<gsl::span<const o2::itsmft::CompClusterExt>, NLayers>& clus,
                 const gsl::span<const o2::itsmft::PhysTrigger> trig,
                 uint32_t firstTForbit,
                 bool doStaggering,
                 const ROFOverlapTableN::View& overlapView,
                 ROFMaskTableN& sel);
  void selectROFsWithVertices(const auto& vertices, const ROFOverlapTableN::View& overlapView, ROFMaskTableN& sel) const
  {
    const auto& multEstConf = FastMultEstConfig::Instance();
    if (!multEstConf.isVtxMultCutRequested()) {
      return;
    }

    for (const auto& vertex : vertices) {
      if (!multEstConf.isPassingVtxMultCut(vertex.getNContributors())) {
        const auto& timestamp{vertex.getTimeStamp()};
        for (int layer = 0; layer < NLayers; ++layer) {
          uint32_t startROF = sel.getLayer(layer).getROF(timestamp.lower());
          uint32_t endROF = sel.getLayer(layer).getROF(timestamp.upper());
          for (uint32_t rof = startROF; rof <= endROF; ++rof) {
            sel.setROFsEnabled(layer, rof, 0);
          }
        }
      }
    }
  }

  int countClustersOnLayer(const gsl::span<const o2::itsmft::CompClusterExt>& clusters) const;
  float process(int nClusters)
  {
    return FastMultEstConfig::Instance().imposeNoisePerChip > 0 ? processNoiseImposed(nClusters) : processNoiseFree(nClusters);
  }
  float processNoiseFree(int nClusters);
  float processNoiseImposed(int nClusters);
  float process(const gsl::span<const o2::itsmft::CompClusterExt>& clusters)
  {
    return process(countClustersOnLayer(clusters));
  }
  static bool sSeedSet;
};

} // namespace o2::its

#endif
