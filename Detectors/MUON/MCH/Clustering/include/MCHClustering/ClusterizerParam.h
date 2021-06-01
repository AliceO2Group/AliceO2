// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file ClusterizerParam.h
/// \brief Configurable parameters for MCH clustering
/// \author Philippe Pillot, Subatech

#ifndef ALICEO2_MCH_CLUSTERIZERPARAM_H_
#define ALICEO2_MCH_CLUSTERIZERPARAM_H_

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"

namespace o2
{
namespace mch
{

/// Configurable parameters for MCH clustering
struct ClusterizerParam : public o2::conf::ConfigurableParamHelper<ClusterizerParam> {

  double lowestPadCharge = 4.f * 0.22875f; ///< minimum charge of a pad

  double pitchSt1 = 0.21;    ///< anode-cathode pitch (cm) for station 1
  double pitchSt2345 = 0.25; ///< anode-cathode pitch (cm) for station 2 to 5

  double mathiesonSqrtKx3St1 = 0.7000;    ///< Mathieson parameter sqrt(K3) in x direction for station 1
  double mathiesonSqrtKx3St2345 = 0.7131; ///< Mathieson parameter sqrt(K3) in x direction for station 2 to 5

  double mathiesonSqrtKy3St1 = 0.7550;    ///< Mathieson parameter sqrt(K3) in y direction for station 1
  double mathiesonSqrtKy3St2345 = 0.7642; ///< Mathieson parameter sqrt(K3) in y direction for station 2 to 5

  double defaultClusterResolution = 0.2; ///< default cluster resolution (cm)
  double badClusterResolution = 10.;     ///< bad (e.g. mono-cathode) cluster resolution (cm)

  O2ParamDef(ClusterizerParam, "MCHClustering");
};

} // namespace mch
} // end namespace o2

#endif // ALICEO2_MCH_CLUSTERIZERPARAM_H_
