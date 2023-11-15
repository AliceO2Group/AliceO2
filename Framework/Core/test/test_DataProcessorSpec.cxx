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

#include <catch_amalgamated.hpp>
#include "Framework/DataProcessorSpec.h"
#include "Framework/DataProcessorSpecHelpers.h"
#include "Framework/ConfigParamSpec.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"

TEST_CASE("TestDataProcessorSpecHelpers")
{
  using namespace o2::framework;
  DataProcessorSpec spec{.name = "test",
                         .algorithm = AlgorithmSpec{[](ProcessingContext& ctx) {}},
                         .options = {ConfigParamSpec{
                           "channel-config",
                           VariantType::String,
                           "name=foo,type=sub,method=connect,address=tcp://localhost:5450,rateLogging=1",
                           {"Out-of-band channel config"}}},
                         .labels = {DataProcessorLabel{"label"},
                                    DataProcessorLabel{"label3"}}};

  REQUIRE(spec.labels.size() == 2);
  REQUIRE(DataProcessorSpecHelpers::hasLabel(spec, "label") == true);
  REQUIRE(DataProcessorSpecHelpers::hasLabel(spec, "label2") == false);
  REQUIRE(DataProcessorSpecHelpers::hasLabel(spec, "label3") == true);
}

#pragma diagnostic pop
