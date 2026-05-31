// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
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

#include <algorithm>
#include <cmath>
#include <format>
#include <limits>

#include "ITStracking/TrackerTraitsMC.h"
#include "ITStracking/TuneExt.h"
#include "CommonUtils/TreeStreamRedirector.h"
#include "CommonConstants/MathConstants.h"
#include "Steer/MCKinematicsReader.h"

#include "Framework/Logger.h"

namespace o2::its
{

static o2::utils::TreeStreamRedirector* gDBG{nullptr};
static o2::steer::MCKinematicsReader* gMCReader{nullptr};

template <int NLayers>
void TrackerTraitsMC<NLayers>::computeLayerTracklets(const int iteration, int iVertex)
{
  if (!gDBG) {
    gDBG = new o2::utils::TreeStreamRedirector("its_tune.root");
  }
  if (!gMCReader) {
    gMCReader = new o2::steer::MCKinematicsReader("collisioncontext.root");
  }

  // Create all tracklets we find in this iteration and dump them.
  const std::string treeName = std::format("trklt_{}", iteration);
  TrackerTraits<NLayers>::computeLayerTracklets(iteration, iVertex);
  this->createTrackletMC();
  const auto topology = this->mTimeFrame->getTrackingTopologyView();
  for (int transitionId{0}; transitionId < topology.nTransitions; ++transitionId) {
    const auto& transition = topology.getTransition(transitionId);
    for (int iTrklt{0}; iTrklt < this->mTimeFrame->getTracklets()[transitionId].size(); ++iTrklt) {
      const auto& lbl = this->mTimeFrame->getTrackletsLabel(transitionId)[iTrklt];
      const auto trklt = this->mTimeFrame->getTracklets()[transitionId][iTrklt];
      const auto& firstCluster = this->mTimeFrame->getClusters()[transition.fromLayer][trklt.firstClusterIndex];
      const auto& secondCluster = this->mTimeFrame->getClusters()[transition.toLayer][trklt.secondClusterIndex];
      const float deltaPhi = std::abs(firstCluster.phi - secondCluster.phi);
      TrackletMC trkltMC{
        .tgl = trklt.tanLambda,
        .phi = trklt.phi,
        .rIn = firstCluster.radius,
        .zIn = firstCluster.zCoordinate,
        .phiIn = firstCluster.phi,
        .rOut = secondCluster.radius,
        .zOut = secondCluster.zCoordinate,
        .phiOut = secondCluster.phi,
        .dr = secondCluster.radius - firstCluster.radius,
        .dz = secondCluster.zCoordinate - firstCluster.zCoordinate,
        .dPhi = std::min(deltaPhi, static_cast<float>(o2::constants::math::TwoPI) - deltaPhi),
        .ok = lbl.isValid(),
      };
      float dcaXY = std::numeric_limits<float>::max(), dcaZ = std::numeric_limits<float>::max();
      if (lbl.isValid()) {
        const auto& eve = gMCReader->getMCEventHeader(lbl.getSourceID(), lbl.getEventID());
        const float dx = secondCluster.xCoordinate - firstCluster.xCoordinate;
        const float dy = secondCluster.yCoordinate - firstCluster.yCoordinate;
        const float dz = secondCluster.zCoordinate - firstCluster.zCoordinate;
        trkltMC.tglEvent = (firstCluster.zCoordinate - eve.GetZ()) / firstCluster.radius;
        trkltMC.deltaZEvent = std::abs((trkltMC.tglEvent * (secondCluster.radius - firstCluster.radius)) + firstCluster.zCoordinate - secondCluster.zCoordinate);
        const float dxy2 = math_utils::hypot(dx, dy);
        if (dxy2 > constants::Tolerance) {
          const float t = ((eve.GetX() - firstCluster.xCoordinate) * dx + (eve.GetY() - firstCluster.yCoordinate) * dy) / dxy2;
          const float xAtDCA = firstCluster.xCoordinate + t * dx;
          const float yAtDCA = firstCluster.yCoordinate + t * dy;
          const float zAtDCA = firstCluster.zCoordinate + t * dz;
          const float curDCAx = xAtDCA - eve.GetX();
          const float curDCAy = yAtDCA - eve.GetY();
          dcaXY = math_utils::hypot(curDCAx, curDCAy);
          dcaZ = zAtDCA - eve.GetZ();
          trkltMC.dXY = dcaXY;
          trkltMC.dZ = dcaZ;
        }
        const auto* mcTrk = gMCReader->getTrack(lbl);
        if (mcTrk) {
          trkltMC.prim = mcTrk->isPrimary();
        }
      }
      (*gDBG) << treeName.c_str()
              << "from=" << transition.fromLayer
              << "to=" << transition.toLayer
              << "trklt=" << trkltMC
              << "\n";
    }
  }
}

template <int NLayers>
void TrackerTraitsMC<NLayers>::computeLayerCells(const int iteration)
{
  TrackerTraits<NLayers>::computeLayerCells(iteration);
}

template <int NLayers>
void TrackerTraitsMC<NLayers>::findCellsNeighbours(const int iteration)
{
  TrackerTraits<NLayers>::findCellsNeighbours(iteration);
}

template <int NLayers>
void TrackerTraitsMC<NLayers>::findRoads(const int iteration)
{
  TrackerTraits<NLayers>::findRoads(iteration);

  if (this->mTrkParams.size() - 1 == iteration) {
    gDBG->Close();
  }
}

template class TrackerTraitsMC<7>;

} // namespace o2::its
