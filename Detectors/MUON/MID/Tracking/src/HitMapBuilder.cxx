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

#include <array>
namespace o2
{
namespace mid
{

HitMapBuilder::HitMapBuilder(const GeometryTransformer& geoTrans) : mMapping(), mHitFinder(geoTrans) {}

void HitMapBuilder::setMaskedChannels(const std::vector<ColumnData>& maskedChannels)
{
  mMaskedChannels.clear();
  std::array<int, 2> nLines{4, 1};
  for (auto& mask : maskedChannels) {
    for (int icath = 0; icath < 2; ++icath) {
      for (int iline = 0; iline < nLines[icath]; ++iline) {
        auto pat = mask.getPattern(icath, iline);
        for (int istrip = 0; istrip < detparams::NStripsBP; ++istrip) {
          if (pat & (1 << istrip)) {
            auto area = mMapping.stripByLocation(istrip, icath, iline, mask.columnId, mask.deId);
            mMaskedChannels[mask.deId].emplace_back(area);
          }
        }
      }
    }
  }
}

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

bool HitMapBuilder::matchesMaskedChannel(const Cluster& cl) const
{
  auto found = mMaskedChannels.find(cl.deId);
  if (found == mMaskedChannels.end()) {
    return false;
  }

  double nSigmas = 4.;

  for (auto& area : found->second) {
    if (std::abs(cl.xCoor - area.getCenterX()) < nSigmas * cl.getEX() + area.getHalfSizeX() &&
        std::abs(cl.yCoor - area.getCenterY()) < nSigmas * cl.getEY() + area.getHalfSizeY()) {
      return true;
    }
  }
  return false;
}

void HitMapBuilder::buildTrackInfo(Track& track, gsl::span<const Cluster> clusters) const
{
  std::vector<int> firedFEEIdMT11, nonFiredFEEIdMT11;
  bool badForEff = false;
  for (int ich = 0; ich < 4; ++ich) {
    auto icl = track.getClusterMatchedUnchecked(ich);
    if (icl >= 0) {
      auto& cl = clusters[icl];
      firedFEEIdMT11.emplace_back(getFEEIdMT11(cl.xCoor, cl.yCoor, cl.deId));
      for (int icath = 0; icath < 2; ++icath) {
        if (cl.isFired(icath)) {
          track.setFiredChamber(ich, icath);
        }
      }
    } else {
      auto impactPts = mHitFinder.getLocalPositions(track, ich, true);
      for (auto& impactPt : impactPts) {
        auto feeIdMT11 = getFEEIdMT11(impactPt.xCoor, impactPt.yCoor, impactPt.deId);
        if (feeIdMT11 >= 0) {
          nonFiredFEEIdMT11.emplace_back(feeIdMT11);
          if (matchesMaskedChannel(impactPt)) {
            badForEff = true;
          }
        } else {
          badForEff = true;
        }
      }
    }
  }
  track.setFiredFEEId(firedFEEIdMT11.front());
  int effFlag = badForEff ? 0 : getEffFlag(firedFEEIdMT11, nonFiredFEEIdMT11);
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
