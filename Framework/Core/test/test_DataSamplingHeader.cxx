// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#define BOOST_TEST_MODULE Test Framework DataSamplingHeader
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>

#include "Framework/DataSamplingHeader.h"
#include "Headers/Stack.h"
#include "Headers/DataHeader.h"

using namespace o2::framework;
using namespace o2::header;

BOOST_AUTO_TEST_CASE(DataSamplingHeaderDefault)
{
  DataSamplingHeader header;

  BOOST_CHECK_EQUAL(header.sampleTimeUs, 0);
  BOOST_CHECK_EQUAL(header.totalAcceptedMessages, 0);
  BOOST_CHECK_EQUAL(header.totalEvaluatedMessages, 0);
  BOOST_CHECK_EQUAL(strcmp(header.deviceID.str, ""), 0);
}

BOOST_AUTO_TEST_CASE(DataSamplingHeaderInit)
{
  DataSamplingHeader header{123, 456, 789, "abc"};

  BOOST_CHECK_EQUAL(header.sampleTimeUs, 123);
  BOOST_CHECK_EQUAL(header.totalAcceptedMessages, 456);
  BOOST_CHECK_EQUAL(header.totalEvaluatedMessages, 789);
  BOOST_CHECK_EQUAL(strcmp(header.deviceID.str, "abc"), 0);
}

BOOST_AUTO_TEST_CASE(DataSamplingHeaderCopy)
{
  DataSamplingHeader header{123, 456, 789, "abc"};
  DataSamplingHeader copy(header);

  BOOST_CHECK_EQUAL(copy.sampleTimeUs, 123);
  BOOST_CHECK_EQUAL(copy.totalAcceptedMessages, 456);
  BOOST_CHECK_EQUAL(copy.totalEvaluatedMessages, 789);
  BOOST_CHECK_EQUAL(strcmp(copy.deviceID.str, "abc"), 0);
}

BOOST_AUTO_TEST_CASE(DataSamplingHeaderAssignement)
{
  DataSamplingHeader first{123, 456, 789, "abc"};
  DataSamplingHeader second;
  second = first;

  BOOST_CHECK_EQUAL(first.sampleTimeUs, 123);
  BOOST_CHECK_EQUAL(first.totalAcceptedMessages, 456);
  BOOST_CHECK_EQUAL(first.totalEvaluatedMessages, 789);
  BOOST_CHECK_EQUAL(strcmp(first.deviceID.str, "abc"), 0);

  BOOST_CHECK_EQUAL(second.sampleTimeUs, 123);
  BOOST_CHECK_EQUAL(second.totalAcceptedMessages, 456);
  BOOST_CHECK_EQUAL(second.totalEvaluatedMessages, 789);
  BOOST_CHECK_EQUAL(strcmp(second.deviceID.str, "abc"), 0);
}

BOOST_AUTO_TEST_CASE(DataSamplingHeaderOnStack)
{
  DataSamplingHeader header{123, 456, 789, "abc"};
  Stack headerStack{header};

  const auto* dsHeaderFromStack = get<DataSamplingHeader*>(headerStack.data());
  BOOST_REQUIRE_NE(dsHeaderFromStack, nullptr);

  BOOST_CHECK_EQUAL(dsHeaderFromStack->sampleTimeUs, 123);
  BOOST_CHECK_EQUAL(dsHeaderFromStack->totalAcceptedMessages, 456);
  BOOST_CHECK_EQUAL(dsHeaderFromStack->totalEvaluatedMessages, 789);
  BOOST_CHECK_EQUAL(strcmp(dsHeaderFromStack->deviceID.str, "abc"), 0);
}