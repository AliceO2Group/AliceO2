// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Track.cxx
/// \brief

#include "ITSReconstruction/CA/Track.h"
#include <cmath>

ClassImp(o2::ITS::CA::TrackObject)

  namespace o2
{
  namespace ITS
  {
  namespace CA
  {

  Track::Track(const Base::Track::TrackParCov& param, float chi2, const std::array<int, 7>& clusters)
    : mParam{ param }, mChi2{ chi2 }, mClusters{ clusters }
  {
  }

  TrackObject::TrackObject(const Track& track) : TObject{}, mTrack{ track } {}

  TrackObject::~TrackObject() {}
  } // namespace CA
  } // namespace ITS
}