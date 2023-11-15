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

#ifndef ALICEO2_ITS3_FASTMULTEST_
#define ALICEO2_ITS3_FASTMULTEST_

#include "ITSMFTReconstruction/ChipMappingITS.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "DataFormatsITS3/CompCluster.h"
#include <DataFormatsITSMFT/PhysTrigger.h>
#include "ITS3Reconstruction/FastMultEstConfig.h"
#include <gsl/span>
#include <array>

namespace o2
{
namespace its3
{

struct FastMultEst {

  static constexpr int NLayers = o2::itsmft::ChipMappingITS::NLayers;

  float mult = 0.;             /// estimated signal clusters multipliciy at reference (1st?) layer
  float noisePerChip = 0.;     /// estimated or imposed noise per chip
  float cov[3] = {0.};         /// covariance matrix of estimation
  float chi2 = 0.;             /// chi2
  int nLayersUsed = 0;         /// number of layers actually used
  uint32_t lastRandomSeed = 0; /// state of the gRandom before

  std::array<int, NLayers> nClPerLayer{0}; // measured N Cl per layer selectROFs
  FastMultEst();

  static uint32_t getCurrentRandomSeed();
  int selectROFs(const gsl::span<const o2::itsmft::ROFRecord> rofs, const gsl::span<const o2::itsmft::CompClusterExt> clus,
                 const gsl::span<const o2::itsmft::PhysTrigger> trig, std::vector<bool>& sel);

  void fillNClPerLayer(const gsl::span<const o2::itsmft::CompClusterExt>& clusters);
  float process(const std::array<int, NLayers> ncl)
  {
    return FastMultEstConfig::Instance().imposeNoisePerChip > 0 ? processNoiseImposed(ncl) : processNoiseFree(ncl);
  }
  float processNoiseFree(const std::array<int, NLayers> ncl);
  float processNoiseImposed(const std::array<int, NLayers> ncl);
  float process(const gsl::span<const o2::itsmft::CompClusterExt>& clusters)
  {
    fillNClPerLayer(clusters);
    return process(nClPerLayer);
  }
  static bool sSeedSet;

  ClassDefNV(FastMultEst, 1);
};

} // namespace its3
} // namespace o2

#endif
