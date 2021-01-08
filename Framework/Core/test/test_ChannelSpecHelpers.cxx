// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test Framework ChannelSpecHelpers
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include "../src/ChannelSpecHelpers.h"

using namespace o2::framework;

BOOST_AUTO_TEST_CASE(TestChannelMethod)
{
  std::ostringstream oss;
  oss << ChannelSpecHelpers::methodAsString(ChannelMethod::Bind)
      << ChannelSpecHelpers::methodAsString(ChannelMethod::Connect);

  BOOST_REQUIRE_EQUAL(oss.str(), "bindconnect");
  std::ostringstream oss2;
}

BOOST_AUTO_TEST_CASE(TestChannelType)
{
  std::ostringstream oss;
  oss << ChannelSpecHelpers::typeAsString(ChannelType::Pub)
      << ChannelSpecHelpers::typeAsString(ChannelType::Sub)
      << ChannelSpecHelpers::typeAsString(ChannelType::Push)
      << ChannelSpecHelpers::typeAsString(ChannelType::Pull);

  BOOST_REQUIRE_EQUAL(oss.str(), "pubsubpushpull");
}
