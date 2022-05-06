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

#define BOOST_TEST_MODULE Test Framework DataDescriptorQueryBuilder
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "Framework/DataDescriptorMatcher.h"
#include "Framework/DataDescriptorQueryBuilder.h"
#include "Framework/DataSpecUtils.h"
#include "Framework/Logger.h"
#include "Framework/InputSpec.h"
#include "Headers/Stack.h"

#include <boost/test/unit_test.hpp>
#include <variant>

using namespace o2::framework;
using namespace o2::header;
using namespace o2::framework::data_matcher;

BOOST_AUTO_TEST_CASE(TestQueryBuilder)
{
  VariableContext context;
  DataHeader header0{"CLUSTERS", "TPC", 1};
  DataHeader header1{"CLUSTERS", "TPC", 2};
  DataHeader header2{"CLUSTERS", "ITS", 1};
  DataHeader header3{"TRACKLET", "TPC", 3};

  auto ispecs = DataDescriptorQueryBuilder::parse("A:TPC/CLUSTERS/1;B:ITS/CLUSTERS/1");
  BOOST_REQUIRE(ispecs.size() == 2);
  BOOST_CHECK(DataSpecUtils::match(ispecs[0], header0) == true);
  BOOST_CHECK(DataSpecUtils::match(ispecs[0], header1) == false);
  BOOST_CHECK(DataSpecUtils::match(ispecs[1], header2) == true);

  ispecs = DataDescriptorQueryBuilder::parse("A:TPC/CLUSTERS/!1;B:ITS/CLUSTERS/1");
  BOOST_REQUIRE(ispecs.size() == 2);
  BOOST_CHECK(DataSpecUtils::match(ispecs[0], header0) == false);
  BOOST_CHECK(DataSpecUtils::match(ispecs[0], header1) == true);
  BOOST_CHECK(DataSpecUtils::match(ispecs[1], header2) == true);
}
