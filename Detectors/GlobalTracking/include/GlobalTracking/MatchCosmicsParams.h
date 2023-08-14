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

/// \author ruben.shahoyan@cern.ch

#ifndef ALICEO2_MATCHCOSMICS_PARAMS_H
#define ALICEO2_MATCHCOSMICS_PARAMS_H

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"
#include "DetectorsBase/Propagator.h"

namespace o2
{
namespace globaltracking
{

struct MatchCosmicsParams : public o2::conf::ConfigurableParamHelper<MatchCosmicsParams> {
  float systSigma2[o2::track::kNParams] = {0.01f, 0.01f, 1e-4f, 1e-4f, 0.f}; // extra error to be added at legs comparison
  float crudeNSigma2Cut[o2::track::kNParams] = {49.f, 49.f, 49.f, 49.f, 49.f};
  float crudeChi2Cut = 999.;
  float timeToleranceMUS = 0.;
  float maxStep = 10.;
  float maxSnp = 0.99;
  float minSeedPt = 0.10;  // use only tracks above this pT (scaled with field)
  float nSigmaTError = 4.; // number of sigmas on track time error for matching (except for TPC which provides an interval)
  bool allowTPCOnly = true;
  o2::base::Propagator::MatCorrType matCorr = o2::base::Propagator::MatCorrType::USEMatCorrLUT;

  O2ParamDef(MatchCosmicsParams, "cosmicsMatch");
};

} // namespace globaltracking
} // end namespace o2

#endif
