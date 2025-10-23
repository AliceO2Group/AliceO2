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
///
/// \file IOUtils.cxx
/// \brief
///

#include "ITStracking/IOUtils.h"

#include <gsl/span>
#include <vector>
#include <cstdlib>

#include "ITSBase/GeometryTGeo.h"
#include "ITStracking/TrackingConfigParam.h"
#include "ITSMFTReconstruction/ChipMappingITS.h"

namespace
{
constexpr int PrimaryVertexLayerId{-1};
constexpr int EventLabelsSeparator{-1};
} // namespace

using namespace o2::its;

/// convert compact clusters to 3D spacepoints
void ioutils::convertCompactClusters(gsl::span<const itsmft::CompClusterExt> clusters,
                                     gsl::span<const unsigned char>::iterator& pattIt,
                                     std::vector<o2::BaseCluster<float>>& output,
                                     const itsmft::TopologyDictionary* dict)
{
  static const o2::itsmft::ChipMappingITS chmap;
  GeometryTGeo* geom = GeometryTGeo::Instance();
  bool applyMisalignment = false;
  const auto& conf = TrackerParamConfig::Instance();
  for (int il = 0; il < chmap.NLayers; il++) {
    if (conf.sysErrY2[il] > 0.f || conf.sysErrZ2[il] > 0.f) {
      applyMisalignment = true;
      break;
    }
  }

  for (const auto& c : clusters) {
    float sigmaY2{0}, sigmaZ2{0}, sigmaYZ{0};
    auto locXYZ = extractClusterData(c, pattIt, dict, sigmaY2, sigmaZ2);
    auto& cl3d = output.emplace_back(c.getSensorID(), geom->getMatrixT2L(c.getSensorID()) ^ locXYZ); // local --> tracking
    if (applyMisalignment) {
      auto lrID = chmap.getLayer(c.getSensorID());
      sigmaY2 += conf.sysErrY2[lrID];
      sigmaZ2 += conf.sysErrZ2[lrID];
    }
    cl3d.setErrors(sigmaY2, sigmaZ2, sigmaYZ);
  }
}
