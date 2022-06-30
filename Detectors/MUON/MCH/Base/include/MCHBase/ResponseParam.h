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

/// \file ResponseParam.h
/// \brief Configurable parameters for MCH charge induction and signal generation
/// \author Philippe Pillot, Subatech

#ifndef O2_MCH_RESPONSEPARAM_H_
#define O2_MCH_RESPONSEPARAM_H_

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"

namespace o2
{
namespace mch
{

/// Configurable parameters for MCH clustering
struct ResponseParam : public o2::conf::ConfigurableParamHelper<ResponseParam> {

  float pitchSt1 = 0.21;    ///< anode-cathode pitch (cm) for station 1
  float pitchSt2345 = 0.25; ///< anode-cathode pitch (cm) for station 2 to 5

  float mathiesonSqrtKx3St1 = 0.7000;    ///< Mathieson parameter sqrt(K3) in x direction for station 1
  float mathiesonSqrtKx3St2345 = 0.7131; ///< Mathieson parameter sqrt(K3) in x direction for station 2 to 5

  float mathiesonSqrtKy3St1 = 0.7550;    ///< Mathieson parameter sqrt(K3) in y direction for station 1
  float mathiesonSqrtKy3St2345 = 0.7642; ///< Mathieson parameter sqrt(K3) in y direction for station 2 to 5

  float chargeSlopeSt1 = 25.;    ///< charge slope used in E to charge conversion for station 1
  float chargeSlopeSt2345 = 10.; ///< charge slope used in E to charge conversion for station 2 to 5

  float chargeSpreadSt1 = 0.144;   ///< width of the charge distribution for station 1
  float chargeSpreadSt2345 = 0.18; ///< width of the charge distribution for station 2 to 5

  float chargeSigmaIntegration = 10.; ///< number of sigmas used for charge distribution

  float chargeCorrelation = 0.11; ///< amplitude of charge correlation between cathodes (= RMS of ln(q1/q2))

  float chargeThreshold = 1.e-4; ///< minimum fraction of charge added to a pad

  O2ParamDef(ResponseParam, "MCHResponse");
};

} // namespace mch
} // end namespace o2

#endif // O2_MCH_RESPONSEPARAM_H_
