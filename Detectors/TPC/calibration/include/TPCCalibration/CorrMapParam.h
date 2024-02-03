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

/// \file CorrMapParam.h
/// \brief Implementation of the parameter class for the CorrectionMapsLoader options
/// \author ruben.shahoyan@cern.ch

#ifndef ALICEO2_TPC_CORRMAP_PARAM_H_
#define ALICEO2_TPC_CORRMAP_PARAM_H_

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"

namespace o2
{
namespace tpc
{

struct CorrMapParam : public o2::conf::ConfigurableParamHelper<CorrMapParam> {
  float lumiInst = 0.;       // override CTP instantaneous lumi (if > 0)
  float lumiMean = 0.;       // override TPC corr.map mean lumi (if > 0), disable corrections if < 0
  float lumiMeanRef = 0.;    // override TPC corr.mapRef mean lumi (if > 0)"
  float lumiInstFactor = 1.; // scaling to apply to instantaneous lumi from CTP (but not to IDC scaler)
  int ctpLumiSource = 0;     // CTP lumi source: 0 = LumiInfo.getLumi(), 1 = LumiInfo.getLumiAlt()

  O2ParamDef(CorrMapParam, "TPCCorrMap");
};
} // namespace tpc

namespace framework
{
template <typename T>
struct is_messageable;
template <>
struct is_messageable<o2::tpc::CorrMapParam> : std::true_type {
};
} // namespace framework

} // namespace o2

#endif // ALICEO2_TPC_CORRMAP_PARAM_H_
