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

#ifndef O2_TRACKING_STUDY_CONFIG_H
#define O2_TRACKING_STUDY_CONFIG_H
#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"

namespace o2::trackstudy
{
struct TrackMCStudyConfig : o2::conf::ConfigurableParamHelper<TrackMCStudyConfig> {
  float minPt = 0.05;
  float maxTgl = 1.5;
  float minPtMC = 0.05;
  float maxTglMC = 1.5;
  float maxRMC = 33.;
  float maxPosTglMC = 2.;
  float maxPVZOffset = 15.;
  float decayMotherMaxT = 1.0f; // max TOF in ns for mother particles to study
  bool requireITSorTPCTrackRefs = true;
  int decayPDG[5] = {310, 3122, 411, 421, -1}; // decays to study, must end by -1
  O2ParamDef(TrackMCStudyConfig, "trmcconf");
};
} // namespace o2::trackstudy

#endif
