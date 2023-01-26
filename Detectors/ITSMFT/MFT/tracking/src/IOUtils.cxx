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
///

#include "MFTTracking/IOUtils.h"

#include <iostream>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <tuple>
#include <unordered_set>
#include <utility>

#include "MFTBase/GeometryTGeo.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITSMFT/TopologyDictionary.h"
#include "MathUtils/Utils.h"
#include "MathUtils/Cartesian.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "MFTTracking/MFTTrackingParam.h"

namespace o2
{
namespace mft
{

//_________________________________________________________
template <typename T>
int ioutils::loadROFrameData(const o2::itsmft::ROFRecord& rof, ROframe<T>& event, gsl::span<const itsmft::CompClusterExt> clusters, gsl::span<const unsigned char>::iterator& pattIt, const itsmft::TopologyDictionary* dict, const dataformats::MCTruthContainer<MCCompLabel>* mcLabels, const o2::mft::Tracker<T>* tracker, ROFFilter& filter)
{
  event.clear();
  GeometryTGeo* geom = GeometryTGeo::Instance();
  geom->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L, o2::math_utils::TransformType::L2G));
  int clusterId{0};
  auto first = rof.getFirstEntry();
  auto clusters_in_frame = rof.getROFData(clusters);
  auto nClusters = clusters_in_frame.size();

  bool skip_ROF = true;
  auto& trackingParam = MFTTrackingParam::Instance();
  if (filter(rof) && trackingParam.isPassingMultCut(nClusters)) {
    LOG(debug) << " ROF selected! ; nClusters = " << nClusters << " ;   orbit = " << rof.getBCData().orbit << " ; bc = " << rof.getBCData().bc;
    skip_ROF = false;
    event.Reserve(nClusters);
  } else {
    nClusters = 0;
  }

  for (auto& c : clusters_in_frame) {
    auto sensorID = c.getSensorID();
    int layer = geom->getLayer(sensorID);
    auto pattID = c.getPatternID();
    o2::math_utils::Point3D<float> locXYZ;
    float sigmaX2 = ioutils::DefClusError2Row, sigmaY2 = ioutils::DefClusError2Col; // Dummy COG errors (about half pixel size)
    if (pattID != itsmft::CompCluster::InvalidPatternID) {
      sigmaX2 = dict->getErr2X(pattID); // ALPIDE local X coordinate => MFT global X coordinate (ALPIDE rows)
      sigmaY2 = dict->getErr2Z(pattID); // ALPIDE local Z coordinate => MFT global Y coordinate (ALPIDE columns)
      if (!dict->isGroup(pattID)) {
        locXYZ = dict->getClusterCoordinates(c);
      } else {
        o2::itsmft::ClusterPattern patt(pattIt);
        locXYZ = dict->getClusterCoordinates(c, patt);
      }
    } else {
      o2::itsmft::ClusterPattern patt(pattIt);
      locXYZ = dict->getClusterCoordinates(c, patt, false);
    }
    if (skip_ROF) { // Skip filtered-out ROFs after processing pattIt
      clusterId++;
      continue;
    }
    // Transformation to the local --> global
    auto gloXYZ = geom->getMatrixL2G(sensorID) * locXYZ;

    auto clsPoint2D = math_utils::Point2D<Float_t>(gloXYZ.x(), gloXYZ.y());
    Float_t rCoord = clsPoint2D.R();
    Float_t phiCoord = clsPoint2D.Phi();
    o2::math_utils::bringTo02PiGen(phiCoord);
    int rBinIndex = tracker->getRBinIndex(rCoord);
    int phiBinIndex = tracker->getPhiBinIndex(phiCoord);
    int binIndex = tracker->getBinIndex(rBinIndex, phiBinIndex);
    event.addClusterToLayer(layer, gloXYZ.x(), gloXYZ.y(), gloXYZ.z(), phiCoord, rCoord, event.getClustersInLayer(layer).size(), binIndex, sigmaX2, sigmaY2, sensorID);
    if (mcLabels) {
      event.addClusterLabelToLayer(layer, *(mcLabels->getLabels(first + clusterId).begin()));
    }
    event.addClusterExternalIndexToLayer(layer, first + clusterId);
    clusterId++;
  }
  return nClusters;
}

//_________________________________________________________
/// convert compact clusters to 3D spacepoints into std::vector<o2::BaseCluster<float>>
void ioutils::convertCompactClusters(gsl::span<const itsmft::CompClusterExt> clusters,
                                     gsl::span<const unsigned char>::iterator& pattIt,
                                     std::vector<o2::BaseCluster<float>>& output,
                                     const itsmft::TopologyDictionary* dict)
{
  GeometryTGeo* geom = GeometryTGeo::Instance();
  geom->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L, o2::math_utils::TransformType::L2G));

  for (auto& c : clusters) {
    auto chipID = c.getChipID();
    auto pattID = c.getPatternID();
    o2::math_utils::Point3D<float> locXYZ;
    float sigmaX2 = DefClusError2Row, sigmaY2 = DefClusError2Col;
    if (pattID != itsmft::CompCluster::InvalidPatternID) {
      sigmaX2 = dict->getErr2X(pattID); // ALPIDE local Y coordinate => MFT global X coordinate (ALPIDE rows)
      sigmaY2 = dict->getErr2Z(pattID); // ALPIDE local Z coordinate => MFT global Y coordinate (ALPIDE columns)
      if (!dict->isGroup(pattID)) {
        locXYZ = dict->getClusterCoordinates(c);
      } else {
        o2::itsmft::ClusterPattern patt(pattIt);
        locXYZ = dict->getClusterCoordinates(c, patt);
      }
    } else {
      o2::itsmft::ClusterPattern patt(pattIt);
      locXYZ = dict->getClusterCoordinates(c, patt, false);
    }

    // Transformation to the local --> global
    auto gloXYZ = geom->getMatrixL2G(chipID) * locXYZ;

    auto& cl3d = output.emplace_back(c.getSensorID(), gloXYZ); // local --> global
    cl3d.setErrors(sigmaX2, sigmaY2, 0);
  }
}

//_________________________________________________________
template <typename T>
int ioutils::loadROFrameData(const o2::itsmft::ROFRecord& rof, ROframe<T>& event, gsl::span<const itsmft::CompClusterExt> clusters, gsl::span<const unsigned char>::iterator& pattIt, const itsmft::TopologyDictionary* dict, const dataformats::MCTruthContainer<MCCompLabel>* mcLabels, const o2::mft::Tracker<T>* tracker)
{
  ROFFilter noFilter = [](const o2::itsmft::ROFRecord r) { return true; };
  return ioutils::loadROFrameData(rof, event, clusters, pattIt, dict, mcLabels, tracker, noFilter);
}

//_________________________________________________________
template int o2::mft::ioutils::loadROFrameData<o2::mft::TrackLTF>(const o2::itsmft::ROFRecord&, ROframe<o2::mft::TrackLTF>&, gsl::span<const itsmft::CompClusterExt>,
                                                                  gsl::span<const unsigned char>::iterator&, const itsmft::TopologyDictionary*,
                                                                  const dataformats::MCTruthContainer<MCCompLabel>*, const o2::mft::Tracker<o2::mft::TrackLTF>*);

template int o2::mft::ioutils::loadROFrameData<o2::mft::TrackLTFL>(const o2::itsmft::ROFRecord&, ROframe<o2::mft::TrackLTFL>&, gsl::span<const itsmft::CompClusterExt>,
                                                                   gsl::span<const unsigned char>::iterator&, const itsmft::TopologyDictionary*,
                                                                   const dataformats::MCTruthContainer<MCCompLabel>*, const o2::mft::Tracker<o2::mft::TrackLTFL>*);
template int o2::mft::ioutils::loadROFrameData<o2::mft::TrackLTF>(const o2::itsmft::ROFRecord&, ROframe<o2::mft::TrackLTF>&, gsl::span<const itsmft::CompClusterExt>,
                                                                  gsl::span<const unsigned char>::iterator&, const itsmft::TopologyDictionary*,
                                                                  const dataformats::MCTruthContainer<MCCompLabel>*, const o2::mft::Tracker<o2::mft::TrackLTF>*,
                                                                  ROFFilter& filter);

template int o2::mft::ioutils::loadROFrameData<o2::mft::TrackLTFL>(const o2::itsmft::ROFRecord&, ROframe<o2::mft::TrackLTFL>&, gsl::span<const itsmft::CompClusterExt>,
                                                                   gsl::span<const unsigned char>::iterator&, const itsmft::TopologyDictionary*,
                                                                   const dataformats::MCTruthContainer<MCCompLabel>*, const o2::mft::Tracker<o2::mft::TrackLTFL>*,
                                                                   ROFFilter& filter);

} // namespace mft
} // namespace o2
