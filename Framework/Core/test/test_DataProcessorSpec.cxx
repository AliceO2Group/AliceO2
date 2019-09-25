// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test Framework DataProcessorSpec
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include "Framework/DataProcessorSpec.h"

BOOST_AUTO_TEST_CASE(TestServiceRegistry)
{
  using namespace o2::framework;
  DataProcessorSpec spec{"test",
                         {},
                         {},
                         AlgorithmSpec{[](ProcessingContext& ctx) {}},
                         {ConfigParamSpec{
                           "channel-config",
                           VariantType::String,
                           "name=foo,type=sub,method=connect,address=tcp://localhost:5450,rateLogging=1",
                           {"Out-of-band channel config"}}},
                         {},
                         {DataProcessorLabel{"label"}}};

  BOOST_CHECK_EQUAL(spec.labels.size(), 1);
}
