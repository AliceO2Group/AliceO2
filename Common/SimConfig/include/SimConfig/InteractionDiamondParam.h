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

/// \author R+Preghenella - October 2018

#ifndef ALICEO2_EVENTGEN_INTERACTIONDIAMONDPARAM_H_
#define ALICEO2_EVENTGEN_INTERACTIONDIAMONDPARAM_H_

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"

namespace o2
{
namespace eventgen
{

/// enumerating the possible vertex smearing settings
enum class EVertexDistribution {
  kGaus = 0, /* ordinary Gaus */
  kFlat = 1  /* flat */
};

/**
 ** a parameter class/struct to keep the settings of
 ** the interaction diamond (position and width) and
 ** allow the user to modify them
 **/
struct InteractionDiamondParam : public o2::conf::ConfigurableParamHelper<InteractionDiamondParam> {
  double position[3] = {0., 0., 0.};
  double width[3] = {0.01, 0.01, 0.01};
  double slopeX = 0.; // z-dependent x pos (see MeanVertexObject)
  double slopeY = 0.; // z-dependent y pos (see MeanVertexObject)
  EVertexDistribution distribution = EVertexDistribution::kGaus;
  O2ParamDef(InteractionDiamondParam, "Diamond");
};

} // end namespace eventgen
} // end namespace o2

#endif // ALICEO2_EVENTGEN_INTERACTIONDIAMONDPARAM_H_
