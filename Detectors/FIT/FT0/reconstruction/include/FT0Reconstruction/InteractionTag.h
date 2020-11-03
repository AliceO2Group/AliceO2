// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
  int minAmplitudeAC = 20; ///< use only FT0 triggers with high enough amplitude

  bool isSelected(const RecPoints& rp) const
  {
    return rp.isValidTime(RecPoints::TimeMean) && (rp.getTrigger().amplA + rp.getTrigger().amplC) > minAmplitudeAC;
  }

  float getInteractionTimeNS(const RecPoints& rp, const o2::InteractionRecord& refIR) const
  {
    return rp.getInteractionRecord().differenceInBCNS(refIR); // RS FIXME do we want use precise MeanTime?
  }

  O2ParamDef(InteractionTag, "ft0tag");
};

} // namespace ft0
} // end namespace o2

#endif
