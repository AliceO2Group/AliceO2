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

/// @file   AlgTrcDbg.h
/// @author ruben.shahoyan@cern.ch

#ifndef ALGTRCDBG_H
#define ALGTRCDBG_H

#include "Align/AlignmentTrack.h"
#include "Align/AlgPntDbg.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "ReconstructionDataFormats/Track.h"

namespace o2
{
namespace align
{

struct AlgTrcDbg : public o2::track::TrackParCov {
  AlgTrcDbg(const AlignmentTrack* trc) { setTrackParam(trc); }
  AlgTrcDbg() = default;
  ~AlgTrcDbg() = default;
  AlgTrcDbg(const AlgTrcDbg&) = default;
  AlgTrcDbg& operator=(const AlgTrcDbg&) = default;

  bool setTrackParam(const AlignmentTrack* trc)
  {
    if (!trc) {
      return false;
    }
    setX(trc->getX());
    setY(trc->getAlpha());
    for (int i = 0; i < 5; i++) {
      setParam(trc->getParam(i), i);
    }
    for (int i = 0; i < 15; i++) {
      setCov(trc->getCov()[i], i);
    }
    mPoints.clear();
    for (int i = 0; i < trc->getNPoints(); i++) {
      const auto* tpoint = trc->getPoint(i);
      if (tpoint->containsMeasurement()) {
        auto& pnt = mPoints.emplace_back(tpoint);
        pnt.mYRes = trc->getResidual(0, i);
        pnt.mZRes = trc->getResidual(1, i);
      }
    }
    setX(trc->getX());
    setY(trc->getAlpha());
    for (int i = 0; i < 5; i++) {
      setParam(trc->getParam(i), i);
    }
    for (int i = 0; i < 15; i++) {
      setCov(trc->getCov()[i], i);
    }
    for (int i = 0; i < trc->getNPoints(); i++) {
      const auto* tpoint = trc->getPoint(i);
      if (tpoint->containsMeasurement()) {
        auto& pnt = mPoints.emplace_back(tpoint);
        pnt.mYRes = trc->getResidual(0, i);
        pnt.mZRes = trc->getResidual(1, i);
      }
    }
    mGID.clear();
    mGIDCosmUp.clear();
    return true;
  }

  auto getNPoints() const { return mPoints.size(); }
  bool isCosmic() const { return mGIDCosmUp.isSourceSet(); }

  std::vector<AlgPntDbg> mPoints;
  o2::dataformats::GlobalTrackID mGID{};
  o2::dataformats::GlobalTrackID mGIDCosmUp{}; // GID of upper leg in case of cosmic
  //
  ClassDefNV(AlgTrcDbg, 1);
};

} // namespace align
} // namespace o2
#endif
