// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \author R+Preghenella - October 2018

#ifndef ALICEO2_EVENTGEN_INTERACTIONDIAMONDPARAM_H_
#define ALICEO2_EVENTGEN_INTERACTIONDIAMONDPARAM_H_

#include "SimConfig/ConfigurableParam.h"
#include "SimConfig/ConfigurableParamHelper.h"

namespace o2
{
namespace eventgen
{

/**
 ** a parameter class/struct to keep the settings of
 ** the interaction diamond (position and width) and 
 ** allow the user to modify them 
 **/

struct InteractionDiamondParam : public o2::conf::ConfigurableParamHelper<InteractionDiamondParam> {
  double position[3] = {0., 0., 0.};
  double width[3] = {0., 0., 0.};
  O2ParamDef(InteractionDiamondParam, "Diamond");
};

} // end namespace eventgen
} // end namespace o2

#endif // ALICEO2_EVENTGEN_INTERACTIONDIAMONDPARAM_H_
