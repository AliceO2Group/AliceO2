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

#ifndef ALICEO2_FT0_INTERACTIONTAG_H
#define ALICEO2_FT0_INTERACTIONTAG_H

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"
#include "DataFormatsFT0/RecPoints.h"
#include "CommonConstants/LHCConstants.h"

namespace o2
{
namespace ft0
{

// These are configurable params for FT0 selection as interaction tag
struct InteractionTag : public o2::conf::ConfigurableParamHelper<InteractionTag> {
  int minAmplitudeAC = 2; ///< use only FT0 triggers with high enough amplitude
  int minAmplitudeA = 1;
  int minAmplitudeC = 1;
  bool isSelected(const RecPoints& rp) const
  {
    return rp.getTrigger().getVertex() && rp.getTrigger().getAmplA() >= minAmplitudeA && rp.getTrigger().getAmplC() >= minAmplitudeC && (rp.getTrigger().getAmplA() + rp.getTrigger().getAmplC()) > minAmplitudeAC;
  }

  O2ParamDef(InteractionTag, "ft0tag");
};

} // namespace ft0
} // end namespace o2

#endif
