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

/// \author Chiara.Zampolli@cern.ch

#ifndef ALICEO2_ITSTPCMATCHINGQC_PARAMS_H
#define ALICEO2_ITSTPCMATCHINGQC_PARAMS_H

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"

namespace o2
{
namespace gloqc
{

// There are configurable params for TPC-ITS matching
struct ITSTPCMatchingQCParams : public o2::conf::ConfigurableParamHelper<ITSTPCMatchingQCParams> {

  float minPtITSCut = 0.f;
  float etaITSCut = 1e10f;
  int32_t minNITSClustersCut = 0;
  int32_t maxChi2PerClusterITS = 100000;
  float minPtTPCCut = 0.1f;
  float etaTPCCut = 0.9f;
  int32_t minNTPCClustersCut = 60;
  float minDCACut = 100.f;
  float minDCACutY = 10.f;
  float minPtCut = 0.f;
  float maxPtCut = 1e10f;
  float etaCut = 1.e10f;
  float cutK0Mass = 0.05f;
  float maxEtaK0 = 0.8f;

  O2ParamDef(ITSTPCMatchingQCParams, "ITSTPCMatchingQC");
};

} // namespace gloqc
} // end namespace o2

#endif
