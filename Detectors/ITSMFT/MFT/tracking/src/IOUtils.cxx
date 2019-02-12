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
///

#include "MFTTracking/IOUtils.h"

#include <iostream>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <tuple>
#include <unordered_set>
#include <utility>

#include "DataFormatsITSMFT/Cluster.h"
#include "MFTBase/GeometryTGeo.h"
#include "MathUtils/Utils.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"

namespace o2
{
namespace MFT
{

using Constants::IndexTable::getBinIndex;
using Constants::IndexTable::getPhiBinIndex;
using Constants::IndexTable::getRBinIndex;
using MathUtils::calculatePhiCoordinate;
using MathUtils::calculateRCoordinate;
using MathUtils::getNormalizedPhiCoordinate;

Int_t IOUtils::loadROFrameData(std::uint32_t roFrame, ROframe& event, const std::vector<ITSMFT::Cluster>* clusters,
                               const dataformats::MCTruthContainer<MCCompLabel>* mcLabels)
{
  if (!clusters) {
    std::cerr << "Missing clusters." << std::endl;
    return -1;
  }
  event.clear();
  GeometryTGeo* geom = GeometryTGeo::Instance();
  geom->fillMatrixCache(utils::bit2Mask(TransformType::T2G));
  Int_t clusterId{ 0 };
  Int_t nused = 0;
  for (auto& c : *clusters) {
    if (c.getROFrame() == roFrame) {
      Int_t layer = geom->getLayer(c.getSensorID());

      /// Rotate to the global frame
      auto xyz = c.getXYZGlo(*geom);
      auto rCoord = calculateRCoordinate(xyz.x(), xyz.y());
      auto phiCoord = getNormalizedPhiCoordinate(calculatePhiCoordinate(xyz.x(), xyz.y()));
      auto rBinIndex = getRBinIndex(rCoord);
      auto phiBinIndex = getPhiBinIndex(phiCoord);
      auto binIndex = getBinIndex(rBinIndex, phiBinIndex);
      event.addClusterToLayer(layer, xyz.x(), xyz.y(), xyz.z(), phiCoord, rCoord, event.getClustersInLayer(layer).size(), binIndex);
      event.addClusterLabelToLayer(layer, *(mcLabels->getLabels(clusterId).begin()));
      event.addClusterExternalIndexToLayer(layer, clusterId);
      nused++;
    }
    clusterId++;
  }
  return nused;
}

void IOUtils::loadEventData(ROframe& event, const std::vector<ITSMFT::Cluster>* clusters,
                            const dataformats::MCTruthContainer<MCCompLabel>* mcLabels)
{
  if (!clusters) {
    std::cerr << "Missing clusters." << std::endl;
    return;
  }
  event.clear();
  GeometryTGeo* geom = GeometryTGeo::Instance();
  geom->fillMatrixCache(utils::bit2Mask(TransformType::T2G));
  Int_t clusterId{ 0 };

  for (auto& c : *clusters) {
    Int_t layer = geom->getLayer(c.getSensorID());

    /// Rotate to the global frame
    auto xyz = c.getXYZGlo(*geom);
    event.addClusterToLayer(layer, xyz.x(), xyz.y(), xyz.z(), event.getClustersInLayer(layer).size());
    event.addClusterLabelToLayer(layer, *(mcLabels->getLabels(clusterId).begin()));
    event.addClusterExternalIndexToLayer(layer, clusterId);
    clusterId++;
  }
}

} // namespace MFT
} // namespace o2
