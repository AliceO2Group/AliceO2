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

#define BOOST_TEST_MODULE Test Framework DataProcessorSpec
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include "Framework/DataProcessorSpec.h"
#include "Framework/DataProcessorSpecHelpers.h"
#include "Framework/ConfigParamSpec.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"

BOOST_AUTO_TEST_CASE(TestServiceRegistry)
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

  BOOST_CHECK_EQUAL(spec.labels.size(), 2);
  BOOST_CHECK_EQUAL(DataProcessorSpecHelpers::hasLabel(spec, "label"), true);
  BOOST_CHECK_EQUAL(DataProcessorSpecHelpers::hasLabel(spec, "label2"), false);
  BOOST_CHECK_EQUAL(DataProcessorSpecHelpers::hasLabel(spec, "label3"), true);
}

#pragma diagnostic pop
