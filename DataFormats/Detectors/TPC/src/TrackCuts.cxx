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

#include <fairlogger/Logger.h>

#include "DataFormatsTPC/TrackCuts.h"
#include "DataFormatsTPC/TrackTPC.h"

ClassImp(o2::tpc::TrackCuts);

using namespace o2::tpc;

TrackCuts::TrackCuts(float PMin, float PMax, float NClusMin, float dEdxMin, float dEdxMax)
  : mPMin(PMin),
    mPMax(PMax),
    mNClusMin(NClusMin),
    mdEdxMin(dEdxMin),
    mdEdxMax(dEdxMax)
{
}

//______________________________________________________________________________
bool TrackCuts::goodTrack(o2::tpc::TrackTPC const& track)
{
  const auto p = track.getP();
  const auto nClusters = track.getNClusterReferences();
  const auto dEdx = track.getdEdx().dEdxTotTPC;

  if (p > mPMax) {
    return false;
  }
  if (p < mPMin) {
    return false;
  }
  if (nClusters < mNClusMin) {
    return false;
  }
  if (dEdx > mdEdxMax) {
    return false;
  }
  if (dEdx < mdEdxMin) {
    return false;
  }
  return true;
}
