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

/// \author R+Preghenella - September 2019

#ifndef ALICEO2_EVENTGEN_TRIGGERPARTICLE_H_
#define ALICEO2_EVENTGEN_TRIGGERPARTICLE_H_

#include "Generators/Trigger.h"
#include "Generators/TriggerParticleParam.h"
#include <fairlogger/Logger.h>

namespace o2
{
namespace eventgen
{

Trigger TriggerParticle(const TriggerParticleParam& param);

} // namespace eventgen
} // namespace o2

#endif /* ALICEO2_EVENTGEN_TRIGGERPARTICLE_H_ */
