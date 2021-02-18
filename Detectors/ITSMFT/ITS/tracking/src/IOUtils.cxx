// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITSMFT/TopologyDictionary.h"
#include "ITSBase/GeometryTGeo.h"
#include "ITStracking/Constants.h"
#include "ITStracking/json.h"
#include "MathUtils/Utils.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "Framework/Logger.h"

namespace
{
constexpr int PrimaryVertexLayerId{-1};
constexpr int EventLabelsSeparator{-1};
} // namespace

using o2::its::constants::its::LayersRCoordinate;
using o2::its::constants::its::LayersZCoordinate;

namespace o2
{
namespace its
{

void to_json(nlohmann::json& j, const TrackingParameters& par);
void from_json(const nlohmann::json& j, TrackingParameters& par);
void to_json(nlohmann::json& j, const MemoryParameters& par);
void from_json(const nlohmann::json& j, MemoryParameters& par);

/// convert compact clusters to 3D spacepoints
void ioutils::convertCompactClusters(gsl::span<const itsmft::CompClusterExt> clusters,
                                     gsl::span<const unsigned char>::iterator& pattIt,
                                     std::vector<o2::BaseCluster<float>>& output,
                                     const itsmft::TopologyDictionary& dict)
{
  GeometryTGeo* geom = GeometryTGeo::Instance();
  for (auto& c : clusters) {
    auto pattID = c.getPatternID();
    o2::math_utils::Point3D<float> locXYZ;
    float sigmaY2 = ioutils::DefClusError2Row, sigmaZ2 = ioutils::DefClusError2Col, sigmaYZ = 0; //Dummy COG errors (about half pixel size)
    if (pattID != itsmft::CompCluster::InvalidPatternID) {
      sigmaY2 = dict.getErr2X(pattID);
      sigmaZ2 = dict.getErr2Z(pattID);
      if (!dict.isGroup(pattID)) {
        locXYZ = dict.getClusterCoordinates(c);
      } else {
        o2::itsmft::ClusterPattern patt(pattIt);
        locXYZ = dict.getClusterCoordinates(c, patt);
      }
    } else {
      o2::itsmft::ClusterPattern patt(pattIt);
      locXYZ = dict.getClusterCoordinates(c, patt, false);
    }
    auto& cl3d = output.emplace_back(c.getSensorID(), geom->getMatrixT2L(c.getSensorID()) ^ locXYZ); // local --> tracking
    cl3d.setErrors(sigmaY2, sigmaZ2, sigmaYZ);
  }
}

void ioutils::loadConfigurations(const std::string& fileName)
{
  if (!fileName.empty()) {
    std::ifstream inputStream;
    inputStream.open(fileName);
    nlohmann::json j;
    inputStream >> j;
    static_cast<TrackingParameters&>(Configuration<TrackingParameters>::getInstance()) = j.at("TrackingParameters").get<TrackingParameters>();
    //static_cast<IndexTableParameters&>(Configuration<IndexTableParameters>::getInstance()) = j.at("IndexTableParameters").get<IndexTableParameters>();
  }
}

std::vector<ROframe> ioutils::loadEventData(const std::string& fileName)
{
  std::vector<ROframe> events{};
  std::ifstream inputStream{};
  std::string line{}, unusedVariable{};
  int layerId{}, monteCarlo{};
  int clusterId{EventLabelsSeparator};
  float xCoordinate{}, yCoordinate{}, zCoordinate{}, alphaAngle{};
  float varZ{-1.f}, varY{-1.f};

  inputStream.open(fileName);

  /// THIS IS LEAKING IN THE BACKWARD COMPATIBLE MODE. KEEP IT IN MIND.
  dataformats::MCTruthContainer<MCCompLabel>* mcLabels = nullptr;
  while (std::getline(inputStream, line)) {

    std::istringstream inputStringStream(line);
    if (inputStringStream >> layerId >> xCoordinate >> yCoordinate >> zCoordinate) {

      if (layerId == PrimaryVertexLayerId) {

        if (clusterId != 0) {
          events.emplace_back(events.size());
        }

        events.back().addPrimaryVertex(xCoordinate, yCoordinate, zCoordinate);
        clusterId = 0;

      } else {

        if (inputStringStream >> varY >> varZ >> unusedVariable >> alphaAngle >> monteCarlo) {
          events.back().addClusterToLayer(layerId, xCoordinate, yCoordinate, zCoordinate,
                                          events.back().getClustersOnLayer(layerId).size());
          const float sinAlpha = std::sin(alphaAngle);
          const float cosAlpha = std::cos(alphaAngle);
          const float xTF = xCoordinate * cosAlpha - yCoordinate * sinAlpha;
          const float yTF = xCoordinate * sinAlpha + yCoordinate * cosAlpha;
          events.back().addTrackingFrameInfoToLayer(layerId, xCoordinate, yCoordinate, zCoordinate, xTF, alphaAngle,
                                                    std::array<float, 2>{yTF, zCoordinate}, std::array<float, 3>{varY, 0.f, varZ});
          events.back().addClusterLabelToLayer(layerId, MCCompLabel(monteCarlo));

          ++clusterId;
        }
      }
    }
  }

  return events;
}

void ioutils::loadEventData(ROframe& event, gsl::span<const itsmft::CompClusterExt> clusters,
                            gsl::span<const unsigned char>::iterator& pattIt, const itsmft::TopologyDictionary& dict,
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

    auto pattID = c.getPatternID();
    o2::math_utils::Point3D<float> locXYZ;
    float sigmaY2 = ioutils::DefClusError2Row, sigmaZ2 = ioutils::DefClusError2Col, sigmaYZ = 0; //Dummy COG errors (about half pixel size)
    if (pattID != itsmft::CompCluster::InvalidPatternID) {
      sigmaY2 = dict.getErr2X(pattID);
      sigmaZ2 = dict.getErr2Z(pattID);
      if (!dict.isGroup(pattID)) {
        locXYZ = dict.getClusterCoordinates(c);
      } else {
        o2::itsmft::ClusterPattern patt(pattIt);
        locXYZ = dict.getClusterCoordinates(c, patt);
      }
    } else {
      o2::itsmft::ClusterPattern patt(pattIt);
      locXYZ = dict.getClusterCoordinates(c, patt, false);
    }
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
      event.addClusterLabelToLayer(layer, *(clsLabels->getLabels(clusterId).begin()));
    }
    event.addClusterExternalIndexToLayer(layer, clusterId);
    clusterId++;
  }
}

int ioutils::loadROFrameData(const o2::itsmft::ROFRecord& rof, ROframe& event, gsl::span<const itsmft::CompClusterExt> clusters, gsl::span<const unsigned char>::iterator& pattIt, const itsmft::TopologyDictionary& dict,
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

    auto pattID = c.getPatternID();
    o2::math_utils::Point3D<float> locXYZ;
    float sigmaY2 = ioutils::DefClusError2Row, sigmaZ2 = ioutils::DefClusError2Col, sigmaYZ = 0; //Dummy COG errors (about half pixel size)
    if (pattID != itsmft::CompCluster::InvalidPatternID) {
      sigmaY2 = dict.getErr2X(pattID);
      sigmaZ2 = dict.getErr2Z(pattID);
      if (!dict.isGroup(pattID)) {
        locXYZ = dict.getClusterCoordinates(c);
      } else {
        o2::itsmft::ClusterPattern patt(pattIt);
        locXYZ = dict.getClusterCoordinates(c, patt);
      }
    } else {
      o2::itsmft::ClusterPattern patt(pattIt);
      locXYZ = dict.getClusterCoordinates(c, patt, false);
    }
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
      event.addClusterLabelToLayer(layer, *(mcLabels->getLabels(first + clusterId).begin()));
    }
    event.addClusterExternalIndexToLayer(layer, first + clusterId);
    clusterId++;
  }
  return clusters_in_frame.size();
}

void ioutils::generateSimpleData(ROframe& event, const int phiDivs, const int zDivs = 1)
{
  const float angleOffset = constants::math::TwoPi / static_cast<float>(phiDivs);
  // Maximum z allowed on innermost layer should be: ~9,75
  const float zOffsetFirstLayer = (zDivs == 1) ? 0 : 1.5 * (LayersZCoordinate()[6] * LayersRCoordinate()[0]) / (LayersRCoordinate()[6] * (static_cast<float>(zDivs) - 1));
  std::vector<float> x, y;
  std::array<std::vector<float>, 7> z;
  for (size_t j{0}; j < zDivs; ++j) {
    for (size_t i{0}; i < phiDivs; ++i) {
      x.emplace_back(cos(i * angleOffset + 0.001)); // put an epsilon to move from periods (e.g. 20 clusters vs 20 cells)
      y.emplace_back(sin(i * angleOffset + 0.001));
      const float zFirstLayer{-static_cast<float>((zDivs - 1.) / 2.) * zOffsetFirstLayer + zOffsetFirstLayer * static_cast<float>(j)};
      z[0].emplace_back(zFirstLayer);
      for (size_t iLayer{1}; iLayer < constants::its::LayersNumber; ++iLayer) {
        z[iLayer].emplace_back(zFirstLayer * LayersRCoordinate()[iLayer] / LayersRCoordinate()[0]);
      }
    }
  }

  for (int iLayer{0}; iLayer < constants::its::LayersNumber; ++iLayer) {
    for (int i = 0; i < phiDivs * zDivs; i++) {
      o2::MCCompLabel label{i, 0, 0, false};
      event.addClusterLabelToLayer(iLayer, label);                                                                              //last argument : label, goes into mClustersLabel
      event.addClusterToLayer(iLayer, LayersRCoordinate()[iLayer] * x[i], LayersRCoordinate()[iLayer] * y[i], z[iLayer][i], i); //uses 1st constructor for clusters
    }
  }
}

std::vector<std::unordered_map<int, Label>> ioutils::loadLabels(const int eventsNum, const std::string& fileName)
{
  std::vector<std::unordered_map<int, Label>> labelsMap{};
  std::unordered_map<int, Label> currentEventLabelsMap{};
  std::ifstream inputStream{};
  std::string line{};
  int monteCarloId{}, pdgCode{}, numberOfClusters{};
  float transverseMomentum{}, phiCoordinate{}, pseudorapidity{};

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

        if (inputStringStream >> transverseMomentum >> phiCoordinate >> pseudorapidity >> pdgCode >> numberOfClusters) {

          if (std::abs(pdgCode) == constants::pdgcodes::PionCode && numberOfClusters == 7) {

            currentEventLabelsMap.emplace(std::piecewise_construct, std::forward_as_tuple(monteCarloId),
                                          std::forward_as_tuple(monteCarloId, transverseMomentum, phiCoordinate,
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

void to_json(nlohmann::json& j, const TrackingParameters& par)
{
  std::array<float, constants::its::TrackletsPerRoad> tmpTrackletMaxDeltaZ;
  std::copy(par.TrackletMaxDeltaZ, par.TrackletMaxDeltaZ + tmpTrackletMaxDeltaZ.size(), tmpTrackletMaxDeltaZ.begin());
  std::array<float, constants::its::CellsPerRoad> tmpCellMaxDCA;
  std::copy(par.CellMaxDCA, par.CellMaxDCA + tmpCellMaxDCA.size(), tmpCellMaxDCA.begin());
  std::array<float, constants::its::CellsPerRoad> tmpCellMaxDeltaZ;
  std::copy(par.CellMaxDeltaZ, par.CellMaxDeltaZ + tmpCellMaxDeltaZ.size(), tmpCellMaxDeltaZ.begin());
  std::array<float, constants::its::CellsPerRoad - 1> tmpNeighbourMaxDeltaCurvature;
  std::copy(par.NeighbourMaxDeltaCurvature, par.NeighbourMaxDeltaCurvature + tmpNeighbourMaxDeltaCurvature.size(), tmpNeighbourMaxDeltaCurvature.begin());
  std::array<float, constants::its::CellsPerRoad - 1> tmpNeighbourMaxDeltaN;
  std::copy(par.NeighbourMaxDeltaN, par.NeighbourMaxDeltaN + tmpNeighbourMaxDeltaN.size(), tmpNeighbourMaxDeltaN.begin());
  j = nlohmann::json{
    {"ClusterSharing", par.ClusterSharing},
    {"MinTrackLength", par.MinTrackLength},
    {"TrackletMaxDeltaPhi", par.TrackletMaxDeltaPhi},
    {"TrackletMaxDeltaZ", tmpTrackletMaxDeltaZ},
    {"CellMaxDeltaTanLambda", par.CellMaxDeltaTanLambda},
    {"CellMaxDCA", tmpCellMaxDCA},
    {"CellMaxDeltaPhi", par.CellMaxDeltaPhi},
    {"CellMaxDeltaZ", tmpCellMaxDeltaZ},
    {"NeighbourMaxDeltaCurvature", tmpNeighbourMaxDeltaCurvature},
    {"NeighbourMaxDeltaN", tmpNeighbourMaxDeltaN}};
}

void from_json(const nlohmann::json& j, TrackingParameters& par)
{
  par.ClusterSharing = j.at("ClusterSharing").get<int>();
  par.MinTrackLength = j.at("MinTrackLength").get<int>();
  par.TrackletMaxDeltaPhi = j.at("TrackletMaxDeltaPhi").get<float>();
  par.CellMaxDeltaTanLambda = j.at("CellMaxDeltaTanLambda").get<float>();
  par.CellMaxDeltaPhi = j.at("CellMaxDeltaPhi").get<float>();
  auto tmpTrackletMaxDeltaZ = j.at("TrackletMaxDeltaZ").get<std::array<float, constants::its::TrackletsPerRoad>>();
  std::copy(tmpTrackletMaxDeltaZ.begin(), tmpTrackletMaxDeltaZ.end(), par.TrackletMaxDeltaZ);
  auto tmpCellMaxDCA = j.at("CellMaxDCA").get<std::array<float, constants::its::CellsPerRoad>>();
  std::copy(tmpCellMaxDCA.begin(), tmpCellMaxDCA.end(), par.CellMaxDCA);
  auto tmpCellMaxDeltaZ = j.at("CellMaxDeltaZ").get<std::array<float, constants::its::CellsPerRoad>>();
  std::copy(tmpCellMaxDCA.begin(), tmpCellMaxDeltaZ.end(), par.CellMaxDeltaZ);
  auto tmpNeighbourMaxDeltaCurvature = j.at("NeighbourMaxDeltaCurvature").get<std::array<float, constants::its::CellsPerRoad - 1>>();
  std::copy(tmpNeighbourMaxDeltaCurvature.begin(), tmpNeighbourMaxDeltaCurvature.end(), par.NeighbourMaxDeltaCurvature);
  auto tmpNeighbourMaxDeltaN = j.at("NeighbourMaxDeltaN").get<std::array<float, constants::its::CellsPerRoad - 1>>();
  std::copy(tmpNeighbourMaxDeltaN.begin(), tmpNeighbourMaxDeltaN.end(), par.NeighbourMaxDeltaN);
}

void to_json(nlohmann::json& j, const MemoryParameters& par)
{
  std::array<float, constants::its::CellsPerRoad> tmpCellsMemoryCoefficients;
  std::copy(par.CellsMemoryCoefficients, par.CellsMemoryCoefficients + tmpCellsMemoryCoefficients.size(), tmpCellsMemoryCoefficients.begin());
  std::array<float, constants::its::TrackletsPerRoad> tmpTrackletsMemoryCoefficients;
  std::copy(par.TrackletsMemoryCoefficients, par.TrackletsMemoryCoefficients + tmpTrackletsMemoryCoefficients.size(), tmpTrackletsMemoryCoefficients.begin());
  j = nlohmann::json{
    {"MemoryOffset", par.MemoryOffset},
    {"CellsMemoryCoefficients", tmpCellsMemoryCoefficients},
    {"TrackletsMemoryCoefficients", tmpTrackletsMemoryCoefficients}};
}

void from_json(const nlohmann::json& j, MemoryParameters& par)
{
  par.MemoryOffset = j.at("MemoryOffset").get<int>();
  auto tmpCellsMemoryCoefficients = j.at("CellsMemoryCoefficients").get<std::array<float, constants::its::CellsPerRoad>>();
  std::copy(tmpCellsMemoryCoefficients.begin(), tmpCellsMemoryCoefficients.end(), par.CellsMemoryCoefficients);
  auto tmpTrackletsMemoryCoefficients = j.at("TrackletsMemoryCoefficients").get<std::array<float, constants::its::TrackletsPerRoad>>();
  std::copy(tmpTrackletsMemoryCoefficients.begin(), tmpTrackletsMemoryCoefficients.end(), par.TrackletsMemoryCoefficients);
}

} // namespace its
} // namespace o2
