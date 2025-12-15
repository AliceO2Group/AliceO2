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

#include "DetectorsBase/Propagator.h"

namespace o2::its3::study
{

struct ITS3TrackingStudyParam : o2::conf::ConfigurableParamHelper<ITS3TrackingStudyParam> {
  /// general track selection
  float maxChi2{36};
  float maxEta{1.0};
  float minPt{0.1};
  float maxPt{1e2};
  /// PV selection
  int minPVCont{5};
  /// ITS track selection
  int minITSCls{7};
  bool refitITS{true}; // refit ITS track including the PV
  /// TPC track selection
  int minTPCCls{110};

  // propagator
  o2::base::PropagatorImpl<float>::MatCorrType CorrType = o2::base::PropagatorImpl<float>::MatCorrType::USEMatCorrLUT;

  /// studies
  bool doDCA = true;
  bool doDCARefit = true;
  bool doPull = true;
  bool doMC = false;
  O2ParamDef(ITS3TrackingStudyParam, "ITS3TrackingStudyParam");
};

} // namespace o2::its3::study

#endif
