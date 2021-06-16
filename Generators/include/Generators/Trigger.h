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

/// \author R+Preghenella - August 2017

#ifndef ALICEO2_EVENTGEN_TRIGGER_H_
#define ALICEO2_EVENTGEN_TRIGGER_H_

#include <functional>
#include "TParticle.h"

namespace o2
{
namespace eventgen
{

typedef std::function<bool(const std::vector<TParticle>&)> Trigger;
typedef std::function<bool(void*, std::string)> DeepTrigger;

} // namespace eventgen
} // namespace o2

#endif /* ALICEO2_EVENTGEN_TRIGGER_H_ */
