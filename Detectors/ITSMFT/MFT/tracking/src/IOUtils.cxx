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

#include "MFTBase/GeometryTGeo.h"

#include "DataFormatsITSMFT/Cluster.h"
#include "MathUtils/Utils.h"
#include "MathUtils/Cartesian2D.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"

namespace o2
{
namespace MFT
{

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
      auto clsPoint2D = Point2D<Float_t>(xyz.x(), xyz.y());
      Float_t rCoord = clsPoint2D.R();
      Float_t phiCoord = clsPoint2D.Phi();
      o2::utils::BringTo02PiGen(phiCoord);
      Int_t rBinIndex = Constants::IndexTable::getRBinIndex(rCoord);
      Int_t phiBinIndex = Constants::IndexTable::getPhiBinIndex(phiCoord);
      Int_t binIndex = Constants::IndexTable::getBinIndex(rBinIndex, phiBinIndex);
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
