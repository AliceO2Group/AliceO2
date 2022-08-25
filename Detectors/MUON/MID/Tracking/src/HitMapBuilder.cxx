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

/// \file   MID/Tracking/src/HitMapBuilder.cxx
/// \brief  Utility to build the MID track hit maps
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   10 December 2021
#include "MIDTracking/HitMapBuilder.h"
namespace o2
{
namespace mid
{

HitMapBuilder::HitMapBuilder(const GeometryTransformer& geoTrans) : mMapping(), mHitFinder(geoTrans) {}

bool HitMapBuilder::crossCommonElement(const std::vector<int>& fired, const std::vector<int>& nonFired) const
{
  // First check that all elements in fired are the same
  auto refIt = fired.begin();
  for (auto it = fired.begin() + 1, end = fired.end(); it != end; ++it) {
    if (*it != *refIt) {
      return false;
    }
  }

  // If there is no element in nonFired, then return true
  if (nonFired.empty()) {
    return true;
  }

  // Check that at least one element in nonFired matches the reference elements
  for (auto it = nonFired.begin(), end = nonFired.end(); it != end; ++it) {
    if (*it == *refIt) {
      return true;
    }
  }

  return false;
}

int HitMapBuilder::getEffFlag(const std::vector<int>& firedFEEIdMT11, const std::vector<int>& nonFiredFEEIdMT11) const
{
  std::vector<int> firedRPCLines, nonFiredRPCLines;
  for (auto& feeId : firedFEEIdMT11) {
    firedRPCLines.emplace_back(detparams::getDEIdFromFEEId(feeId));
  }
  for (auto& feeId : nonFiredFEEIdMT11) {
    nonFiredRPCLines.emplace_back(detparams::getDEIdFromFEEId(feeId));
  }

  if (crossCommonElement(firedRPCLines, nonFiredRPCLines)) {
    if (crossCommonElement(firedFEEIdMT11, nonFiredFEEIdMT11)) {
      return 3;
    }
    return 2;
  }
  return 1;
}

int HitMapBuilder::getFEEIdMT11(double xp, double yp, uint8_t deId) const
{
  auto stripIndex = mMapping.stripByPosition(xp, yp, 0, deId, false);
  if (stripIndex.isValid()) {
    auto deIdMT11 = detparams::getDEId(detparams::isRightSide(deId), 0, detparams::getRPCLine(deId));
    return detparams::getUniqueFEEId(deIdMT11, stripIndex.column, stripIndex.line);
  }
  return -1;
}

void HitMapBuilder::buildTrackInfo(Track& track, gsl::span<const Cluster> clusters) const
{
  std::vector<int> firedFEEIdMT11, nonFiredFEEIdMT11;
  bool outsideAcceptance = false;
  for (int ich = 0; ich < 4; ++ich) {
    auto icl = track.getClusterMatchedUnchecked(ich);
    if (icl >= 0) {
      auto& cl = clusters[icl];
      firedFEEIdMT11.emplace_back(getFEEIdMT11(cl.xCoor, cl.yCoor, cl.deId));
      // auto localPt = mHitFinder.getGeometryTransformer().globalToLocal(cl.deId, cl.xCoor, cl.yCoor, cl.zCoor);
      for (int icath = 0; icath < 2; ++icath) {
        if (cl.isFired(icath)) {
          track.setFiredChamber(ich, icath);
        }
      }
    } else {
      auto impactPts = mHitFinder.getLocalPositions(track, ich);
      for (auto& impactPt : impactPts) {
        auto feeIdMT11 = getFEEIdMT11(impactPt.xCoor, impactPt.yCoor, impactPt.deId);
        if (feeIdMT11 >= 0) {
          nonFiredFEEIdMT11.emplace_back(feeIdMT11);
        } else {
          outsideAcceptance = true;
        }
      }
    }
  }
  track.setFiredFEEId(firedFEEIdMT11.front());
  int effFlag = outsideAcceptance ? 0 : getEffFlag(firedFEEIdMT11, nonFiredFEEIdMT11);
  track.setEfficiencyFlag(effFlag);
}

void HitMapBuilder::process(std::vector<Track>& tracks, gsl::span<const Cluster> clusters) const
{
  for (auto& track : tracks) {
    buildTrackInfo(track, clusters);
  }
}

} // namespace mid
} // namespace o2
