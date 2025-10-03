// Copyright 2019-2025 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "InspectorHelpers.h"

namespace o2::framework
{
char const* InspectorHelpers::getLifeTimeStr(Lifetime lifetime)
{
  switch (lifetime) {
    case Lifetime::Timeframe:
      return "Timeframe";
    case Lifetime::Condition:
      return "Condition";
    case Lifetime::Sporadic:
      return "Sporadic";
    case Lifetime::Transient:
      return "Transient";
    case Lifetime::Timer:
      return "Timer";
    case Lifetime::Enumeration:
      return "Enumeration";
    case Lifetime::Signal:
      return "Signal";
    case Lifetime::Optional:
      return "Optional";
    case Lifetime::OutOfBand:
      return "OutOfBand";
  }
  return "none";
};
} // namespace o2::framework
