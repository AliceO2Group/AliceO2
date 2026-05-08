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

#ifndef O2_CHECK_RESID_TYPES_H
#define O2_CHECK_RESID_TYPES_H

#include "ReconstructionDataFormats/Track.h"

namespace o2::checkresid
{
struct Point {
  float dy = 0.f;
  float dz = 0.f;
  float sig2y = 0.f;
  float sig2z = 0.f;
  float phi = 0.f;
  float z = 0.f;
  int16_t sens = -1;
  int8_t lr = -1; // -1 = vtx
  ClassDefNV(Point, 1)
};

struct Track {
  o2::dataformats::GlobalTrackID gid{};
  o2::track::TrackPar track;
  o2::track::TrackParCov trIBOut;
  o2::track::TrackParCov trOBInw;
  std::vector<Point> points;
  ClassDefNV(Track, 1)
};

} // namespace o2::checkresid

#endif
