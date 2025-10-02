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

#ifndef ALICEO2_PARTICLEINFO_EXT_H
#define ALICEO2_PARTICLEINFO_EXT_H

#include "ReconstructionDataFormats/Track.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "SimulationDataFormat/MCTrack.h"

namespace o2::its3::study
{

struct ParticleInfoExt {
  // cluster info
  uint8_t clusters{0};
  uint8_t fakeClusters{0};
  // reco info
  uint8_t isReco{0};
  uint8_t isFake{0};
  // matching info
  uint8_t recoTracks;
  uint8_t fakeTracks;
  // reco track
  track::TrackParCov recoTrack;
  // mc info
  MCTrack mcTrack;

  ClassDefNV(ParticleInfoExt, 1);
};

} // namespace o2::its3::study

#endif
