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

#ifndef O2_DCS_TEST_WORKFLOW_DATA_REPLAY_SPEC_H
#define O2_DCS_TEST_WORKFLOW_DATA_REPLAY_SPEC_H

#include "DetectorsDCS/DCSDataPointHint.h"
#include "Framework/DataProcessorSpec.h"
#include <variant>
#include <string>
#include <vector>
#include <cstdint>

namespace o2::dcs::test
{

o2::framework::DataProcessorSpec getDCSDataReplaySpec(std::vector<HintType> hints = {},
                                                      const char* detName = "TPC");

} // namespace o2::dcs::test

#endif
