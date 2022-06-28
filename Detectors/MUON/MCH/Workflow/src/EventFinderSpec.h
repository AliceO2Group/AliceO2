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

/// \file EventFinderSpec.h
/// \brief Definition of a data processor to group MCH digits based on MID information
///
/// \author Philippe Pillot, Subatech

#ifndef O2_MCH_EVENTFINDERSPEC_H_
#define O2_MCH_EVENTFINDERSPEC_H_

#include "Framework/DataProcessorSpec.h"

namespace o2
{
namespace mch
{

framework::DataProcessorSpec getEventFinderSpec(
  bool useMC,
  std::string_view specName = "mch-event-finder",
  std::string_view inputDigitDataDescription = "F-DIGITS",
  std::string_view outputDigitDataDescription = "E-F-DIGITS",
  std::string_view inputDigitRofDataDescription = "F-DIGITROFS",
  std::string_view outputDigitRofDataDescription = "E-F-DIGITROFS",
  std::string_view inputDigitLabelDataDescription = "F-DIGITLABELS",
  std::string_view outputDigitLabelDataDescription = "E-F-DIGITLABELS");

} // namespace mch
} // namespace o2

#endif // O2_MCH_EVENTFINDERSPEC_H_
