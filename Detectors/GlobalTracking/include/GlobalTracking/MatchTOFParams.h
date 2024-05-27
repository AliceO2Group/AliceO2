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

/// \author fnoferin@cern.ch

#ifndef ALICEO2_MATCHTOF_PARAMS_H
#define ALICEO2_MATCHTOF_PARAMS_H

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"

namespace o2
{
namespace globaltracking
{

struct MatchTOFParams : public o2::conf::ConfigurableParamHelper<MatchTOFParams> {
  float calibMaxChi2 = 3.0;       // max value of Chi2 allowed for using matched tracks in calibration
  float nsigmaTimeCut = 4.;       // number of sigmas for non-TPC track time resolution to consider
  float maxResX = 1.;             // max value of track resolution (X dir) used in TOF matching (truncation to that in case it is larger)
  float maxResZ = 1.;             // max value of track resolution (Z dir) used in TOF matching (truncation to that in case it is larger)
  float maxChi2 = 10.;            // max value of Chi2 accepted for matching to TOF
  bool applyPIDcutTPConly = true; // apply PID cut on TPC only tracks

  O2ParamDef(MatchTOFParams, "MatchTOF");
};

} // namespace globaltracking
} // end namespace o2

#endif
