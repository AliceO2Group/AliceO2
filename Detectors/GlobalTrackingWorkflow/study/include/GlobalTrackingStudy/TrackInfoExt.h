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

// class for extended Track info (for debugging)

#ifndef ALICEO2_TRINFOEXT_H
#define ALICEO2_TRINFOEXT_H

#include "ReconstructionDataFormats/MatchInfoTOF.h"
#include "ReconstructionDataFormats/DCA.h"
#include "ReconstructionDataFormats/VtxTrackIndex.h"
#include "ReconstructionDataFormats/Track.h"

namespace o2
{
namespace dataformats
{

struct TrackInfoExt {
  o2::track::TrackParCov track;
  DCA dca{};
  VtxTrackIndex gid;
  MatchInfoTOF infoTOF;
  float ttime = 0;
  float ttimeE = 0;
  float xmin = 0;
  float chi2ITSTPC = 0.f;
  float q2ptITS = 0.f;
  float q2ptTPC = 0.f;
  float q2ptITSTPC = 0.f;
  float q2ptITSTPCTRD = 0.f;
  int nClTPC = 0;
  int nClITS = 0;
  int pattITS = 0;
  ClassDefNV(TrackInfoExt, 1);
};

} // namespace dataformats
} // namespace o2

#endif
