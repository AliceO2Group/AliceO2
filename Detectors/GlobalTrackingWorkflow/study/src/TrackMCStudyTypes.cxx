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

#include "GlobalTrackingStudy/TrackMCStudyTypes.h"

namespace o2::trackstudy
{

RecTrack TrackFamily::dummyRecTrack;

// get longest number of clusters on consecutive layers
int MCTrackInfo::getNITSClusCont() const
{
  if (nITSCl < 2) {
    return nITSCl;
  }
  int longest = 0, current = 0;
  for (int i = 0; i < 7; i++) {
    if (pattITSCl & (0x1 << i)) {
      longest = ++current;
    } else {
      current = 0;
    }
  }
  return longest;
}

// check how many clusters ITS-TPC afterburner could see (consecutively occupied layers starting from the last one)
int MCTrackInfo::getNITSClusForAB() const
{
  int ncl = 0;
  if (nITSCl) {
    for (int i = 6; i > 2; i--) {
      if (pattITSCl & (0x1 << i)) {
        ncl++;
      } else {
        break;
      }
    }
  }
  return ncl;
}

// lowest ITS layer with cluster
int MCTrackInfo::getLowestITSLayer() const
{
  if (nITSCl) {
    for (int i = 0; i < 7; i++) {
      if (pattITSCl & (0x1 << i)) {
        return i;
      }
    }
  }
  return -1;
}

// highest ITS layer with cluster
int MCTrackInfo::getHighestITSLayer() const
{
  if (nITSCl) {
    for (int i = 7; i--;) {
      if (pattITSCl & (0x1 << i)) {
        return i;
      }
    }
  }
  return -1;
}

} // namespace o2::trackstudy
