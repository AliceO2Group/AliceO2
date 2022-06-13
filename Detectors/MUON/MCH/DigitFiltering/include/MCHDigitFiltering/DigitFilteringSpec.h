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

#ifndef O2_MCH_DIGIT_FILTERING_SPEC_H
#define O2_MCH_DIGIT_FILTERING_SPEC_H

#include "Framework/DataProcessorSpec.h"

namespace o2::mch
{

o2::framework::DataProcessorSpec
  getDigitFilteringSpec(
    bool useMC,
    std::string_view specName = "mch-digit-filtering",
    std::string_view inputDigitDataDescription = "DIGITS",
    std::string_view outputDigitDataDescription = "F-DIGITS",
    std::string_view inputDigitRofDataDescription = "DIGITROFS",
    std::string_view outputDigitRofDataDescription = "F-DIGITROFS",
    std::string_view inputDigitLabelDataDescription = "DIGITLABELS",
    std::string_view outputDigitLabelDataDescription = "F-DIGITLABELS");

}
#endif
