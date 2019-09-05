// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \author R+Preghenella - January 2020

#ifndef ALICEO2_EVENTGEN_TRIGGERPARTICLEPARAM_H_
#define ALICEO2_EVENTGEN_TRIGGERPARTICLEPARAM_H_

#include "SimConfig/ConfigurableParam.h"
#include "SimConfig/ConfigurableParamHelper.h"

namespace o2
{
namespace eventgen
{

/**
 ** a parameter class/struct to keep the settings of
 ** the event-generator trigger particle and 
 ** allow the user to modify them 
 **/
struct TriggerParticleParam : public o2::conf::ConfigurableParamHelper<TriggerParticleParam> {
  int pdg = 0;
  double ptMin = 0.;
  double ptMax = 1.e6;
  double etaMin = -1.e6;
  double etaMax = 1.e6;
  double phiMin = -1.e6;
  double phiMax = 1.e6;
  double yMin = -1.e6;
  double yMax = 1.e6;
  O2ParamDef(TriggerParticleParam, "TriggerParticle");
};

} // end namespace eventgen
} // end namespace o2

#endif // ALICEO2_EVENTGEN_TRIGGERPARTICLEPARAM_H_
