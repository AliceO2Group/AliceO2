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

#include <iostream>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <tuple>
#include <unordered_set>
#include <utility>

#include "ITSBase/GeometryTGeo.h"
#include "ITStracking/Constants.h"
#include "ITStracking/json.h"
#include "MathUtils/Utils.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "GPUCommonLogger.h"

namespace
{
constexpr int PrimaryVertexLayerId{-1};
constexpr int EventLabelsSeparator{-1};
} // namespace

namespace o2
{
namespace its
{

void to_json(nlohmann::json& j, const TrackingParameters& par);
void from_json(const nlohmann::json& j, TrackingParameters& par);

/// convert compact clusters to 3D spacepoints
void ioutils::convertCompactClusters(gsl::span<const itsmft::CompClusterExt> clusters,
                                     gsl::span<const unsigned char>::iterator& pattIt,
                                     std::vector<o2::BaseCluster<float>>& output,
                                     const itsmft::TopologyDictionary* dict)
{
  GeometryTGeo* geom = GeometryTGeo::Instance();
  bool applyMisalignment = false;
  const auto& conf = TrackerParamConfig::Instance();
  const auto& chmap = getChipMappingITS();
  for (int il = 0; il < chmap.NLayers; il++) {
    if (conf.sysErrY2[il] > 0.f || conf.sysErrZ2[il] > 0.f) {
      applyMisalignment = true;
      break;
    }
  }

  for (auto& c : clusters) {
    float sigmaY2, sigmaZ2, sigmaYZ = 0;
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

void ioutils::loadEventData(ROframe& event, gsl::span<const itsmft::CompClusterExt> clusters,
                            gsl::span<const unsigned char>::iterator& pattIt, const itsmft::TopologyDictionary* dict,
                            const dataformats::MCTruthContainer<MCCompLabel>* clsLabels)
{
  if (clusters.empty()) {
    std::cerr << "Missing clusters." << std::endl;
    return;
  }
  event.clear();
  GeometryTGeo* geom = GeometryTGeo::Instance();
  geom->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L, o2::math_utils::TransformType::L2G));
  int clusterId{0};

  for (auto& c : clusters) {
    int layer = geom->getLayer(c.getSensorID());
    float sigmaY2, sigmaZ2, sigmaYZ = 0;
    auto locXYZ = extractClusterData(c, pattIt, dict, sigmaY2, sigmaZ2);
    auto sensorID = c.getSensorID();
    // Inverse transformation to the local --> tracking
    auto trkXYZ = geom->getMatrixT2L(sensorID) ^ locXYZ;
    // Transformation to the local --> global
    auto gloXYZ = geom->getMatrixL2G(sensorID) * locXYZ;

    event.addTrackingFrameInfoToLayer(layer, gloXYZ.x(), gloXYZ.y(), gloXYZ.z(), trkXYZ.x(), geom->getSensorRefAlpha(sensorID),
                                      std::array<float, 2>{trkXYZ.y(), trkXYZ.z()},
                                      std::array<float, 3>{sigmaY2, sigmaYZ, sigmaZ2});

    /// Rotate to the global frame
    event.addClusterToLayer(layer, gloXYZ.x(), gloXYZ.y(), gloXYZ.z(), event.getClustersOnLayer(layer).size());
    if (clsLabels) {
      // event.addClusterLabelToLayer(layer, *(clsLabels->getLabels(clusterId).begin()));
      event.setMClabelsContainer(clsLabels);
    }
    event.addClusterExternalIndexToLayer(layer, clusterId);
    clusterId++;
  }
}

int ioutils::loadROFrameData(const o2::itsmft::ROFRecord& rof, ROframe& event, gsl::span<const itsmft::CompClusterExt> clusters, gsl::span<const unsigned char>::iterator& pattIt, const itsmft::TopologyDictionary* dict,
                             const dataformats::MCTruthContainer<MCCompLabel>* mcLabels)
{
  event.clear();
  GeometryTGeo* geom = GeometryTGeo::Instance();
  geom->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L, o2::math_utils::TransformType::L2G));
  int clusterId{0};

  auto first = rof.getFirstEntry();
  auto clusters_in_frame = rof.getROFData(clusters);
  for (auto& c : clusters_in_frame) {
    int layer = geom->getLayer(c.getSensorID());
    float sigmaY2, sigmaZ2, sigmaYZ = 0;
    auto locXYZ = extractClusterData(c, pattIt, dict, sigmaY2, sigmaZ2);
    auto sensorID = c.getSensorID();
    // Inverse transformation to the local --> tracking
    auto trkXYZ = geom->getMatrixT2L(sensorID) ^ locXYZ;
    // Transformation to the local --> global
    auto gloXYZ = geom->getMatrixL2G(sensorID) * locXYZ;

    event.addTrackingFrameInfoToLayer(layer, gloXYZ.x(), gloXYZ.y(), gloXYZ.z(), trkXYZ.x(), geom->getSensorRefAlpha(sensorID),
                                      std::array<float, 2>{trkXYZ.y(), trkXYZ.z()},
                                      std::array<float, 3>{sigmaY2, sigmaYZ, sigmaZ2});

    /// Rotate to the global frame
    event.addClusterToLayer(layer, gloXYZ.x(), gloXYZ.y(), gloXYZ.z(), event.getClustersOnLayer(layer).size());
    if (mcLabels) {
      // event.addClusterLabelToLayer(layer, *(mcLabels->getLabels(first + clusterId).begin()));
      event.setMClabelsContainer(mcLabels);
    }
    event.addClusterExternalIndexToLayer(layer, first + clusterId);
    clusterId++;
  }
  return clusters_in_frame.size();
}

std::vector<std::unordered_map<int, Label>> ioutils::loadLabels(const int eventsNum, const std::string& fileName)
{
  std::vector<std::unordered_map<int, Label>> labelsMap{};
  std::unordered_map<int, Label> currentEventLabelsMap{};
  std::ifstream inputStream{};
  std::string line{};
  int monteCarloId{}, pdgCode{}, numberOfClusters{};
  float transverseMomentum{}, phi{}, pseudorapidity{};

  labelsMap.reserve(eventsNum);

  inputStream.open(fileName);
  std::getline(inputStream, line);

  while (std::getline(inputStream, line)) {

    std::istringstream inputStringStream(line);

    if (inputStringStream >> monteCarloId) {

      if (monteCarloId == EventLabelsSeparator) {

        labelsMap.emplace_back(currentEventLabelsMap);
        currentEventLabelsMap.clear();

      } else {

        if (inputStringStream >> transverseMomentum >> phi >> pseudorapidity >> pdgCode >> numberOfClusters) {

          if (std::abs(pdgCode) == constants::pdgcodes::PionCode && numberOfClusters == 7) {

            currentEventLabelsMap.emplace(std::piecewise_construct, std::forward_as_tuple(monteCarloId),
                                          std::forward_as_tuple(monteCarloId, transverseMomentum, phi,
                                                                pseudorapidity, pdgCode, numberOfClusters));
          }
        }
      }
    }
  }

  labelsMap.emplace_back(currentEventLabelsMap);

  return labelsMap;
}

void ioutils::writeRoadsReport(std::ofstream& correctRoadsOutputStream, std::ofstream& duplicateRoadsOutputStream,
                               std::ofstream& fakeRoadsOutputStream, const std::vector<std::vector<Road>>& roads,
                               const std::unordered_map<int, Label>& labelsMap)
{
  const int numVertices{static_cast<int>(roads.size())};
  std::unordered_set<int> foundMonteCarloIds{};

  correctRoadsOutputStream << EventLabelsSeparator << std::endl;
  fakeRoadsOutputStream << EventLabelsSeparator << std::endl;

  for (int iVertex{0}; iVertex < numVertices; ++iVertex) {

    const std::vector<Road>& currentVertexRoads{roads[iVertex]};
    const int numRoads{static_cast<int>(currentVertexRoads.size())};

    for (int iRoad{0}; iRoad < numRoads; ++iRoad) {

      const Road& currentRoad{currentVertexRoads[iRoad]};
      const int currentRoadLabel{currentRoad.getLabel()};

      if (!labelsMap.count(currentRoadLabel)) {

        continue;
      }

      const Label& currentLabel{labelsMap.at(currentRoadLabel)};

      if (currentRoad.isFakeRoad()) {

        fakeRoadsOutputStream << currentLabel << std::endl;

      } else {

        if (foundMonteCarloIds.count(currentLabel.monteCarloId)) {

          duplicateRoadsOutputStream << currentLabel << std::endl;

        } else {

          correctRoadsOutputStream << currentLabel << std::endl;
          foundMonteCarloIds.emplace(currentLabel.monteCarloId);
        }
      }
    }
  }
}

} // namespace its
} // namespace o2
