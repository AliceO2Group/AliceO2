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

#ifndef O2_ITS3_ALIGNMENT_TYPES_H
#define O2_ITS3_ALIGNMENT_TYPES_H

#include <string>
#include "ReconstructionDataFormats/Track.h"
#include "DataFormatsITS/TrackITS.h"

namespace o2::its3::align
{

struct Measurement final {
  double dy = 0.f;
  double dz = 0.f;
  double sig2y = 0.f;
  double sig2z = 0.f;
  double phi = 0.f;
  double z = 0.f;
  ClassDefNV(Measurement, 1)
};

struct FrameInfoExt final {
  int16_t sens = -1;
  int8_t lr = -1; // -1 = vtx
  double x{-999.f};
  double alpha{-999.f};
  std::array<double, 2> positionTrackingFrame = {999., 999.};
  std::array<double, 3> covarianceTrackingFrame = {999., 999., 999.};

  std::string asString() const;

  ClassDefNV(FrameInfoExt, 1)
};

struct FitInfo final {
  float chi2Ndf{-1}; // Chi2/Ndf of track refit
  float chi2{-1};    // Chi2
  int ndf{-1};       // ndf
  ClassDefNV(FitInfo, 1)
};

struct Track {
  o2::its::TrackITS its;           // original ITS track
  o2::track::TrackParCovD track;   // track at innermost update point, refitted from outwards seed
  FitInfo kfFit;                   // kf fit information
  FitInfo gblFit;                  // gbl fit information
  std::vector<Measurement> points; // measurment point
  std::vector<FrameInfoExt> info;  // frame info
  ClassDefNV(Track, 1)
};

} // namespace o2::its3::align

#endif
