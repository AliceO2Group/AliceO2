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
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITSMFT/TopologyDictionary.h"
#include "MathUtils/Utils.h"
#include "MathUtils/Cartesian2D.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"

namespace o2
{
namespace mft
{

int ioutils::loadROFrameData(const o2::itsmft::ROFRecord& rof, ROframe& event, gsl::span<const itsmft::CompClusterExt> clusters, gsl::span<const unsigned char>::iterator& pattIt, const itsmft::TopologyDictionary& dict,
                             const dataformats::MCTruthContainer<MCCompLabel>* mcLabels)
{
  event.clear();
  GeometryTGeo* geom = GeometryTGeo::Instance();
  geom->fillMatrixCache(utils::bit2Mask(TransformType::T2L, TransformType::L2G));
  int clusterId{0};
  auto first = rof.getFirstEntry();
  auto clusters_in_frame = rof.getROFData(clusters);
  for (auto& c : clusters_in_frame) {
    int layer = geom->getLayer(c.getSensorID());
    auto pattID = c.getPatternID();
    Point3D<float> locXYZ;
    float sigmaX2 = ioutils::DefClusError2Row, sigmaY2 = ioutils::DefClusError2Col; //Dummy COG errors (about half pixel size)
    if (pattID != itsmft::CompCluster::InvalidPatternID) {
      sigmaX2 = dict.getErr2X(pattID); // ALPIDE local X coordinate => MFT global X coordinate (ALPIDE rows)
      sigmaY2 = dict.getErr2Z(pattID); // ALPIDE local Z coordinate => MFT global Y coordinate (ALPIDE columns)
      if (!dict.isGroup(pattID)) {
        locXYZ = dict.getClusterCoordinates(c);
      } else {
        o2::itsmft::ClusterPattern patt(pattIt);
        locXYZ = dict.getClusterCoordinates(c, patt);
      }
    } else {
      o2::itsmft::ClusterPattern patt(pattIt);
      locXYZ = dict.getClusterCoordinates(c, patt);
    }
    auto sensorID = c.getSensorID();
    // Transformation to the local --> global
    auto gloXYZ = geom->getMatrixL2G(sensorID) * locXYZ;

    auto clsPoint2D = Point2D<Float_t>(gloXYZ.x(), gloXYZ.y());
    Float_t rCoord = clsPoint2D.R();
    Float_t phiCoord = clsPoint2D.Phi();
    o2::utils::BringTo02PiGen(phiCoord);
    int rBinIndex = constants::index_table::getRBinIndex(rCoord);
    int phiBinIndex = constants::index_table::getPhiBinIndex(phiCoord);
    int binIndex = constants::index_table::getBinIndex(rBinIndex, phiBinIndex);

    event.addClusterToLayer(layer, gloXYZ.x(), gloXYZ.y(), gloXYZ.z(), phiCoord, rCoord, event.getClustersInLayer(layer).size(), binIndex, sigmaX2, sigmaY2, sensorID);
    if (mcLabels) {
      event.addClusterLabelToLayer(layer, *(mcLabels->getLabels(first + clusterId).begin()));
    }
    event.addClusterExternalIndexToLayer(layer, first + clusterId);
    clusterId++;
  }
  return clusters_in_frame.size();
}

} // namespace mft
} // namespace o2
