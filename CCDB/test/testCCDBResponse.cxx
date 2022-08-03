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

#define BOOST_TEST_MODULE CCDB
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "testCCDBResponseResources.h"
#include <CCDB/CCDBResponse.h>
#include <boost/test/unit_test.hpp>
#include <string>

using namespace o2::ccdb;

BOOST_AUTO_TEST_CASE(TestCCDBResponseParse)
{
  std::string* responseString = new std::string(secondFullResponse);
  CCDBResponse ccdbResponse(*responseString);
  BOOST_CHECK(ccdbResponse.objectNum == 4);
  BOOST_CHECK("407f3a65-4c7b-11ec-8cf8-200114580202" == ccdbResponse.getStringAttribute(0, "id"));
  BOOST_CHECK("e5183d1a-4c7a-11ec-9d71-7f000001aa8b" == ccdbResponse.getStringAttribute(1, "id"));
  BOOST_CHECK("52d3f61a-4c6b-11ec-a98e-7f000001aa8b" == ccdbResponse.getStringAttribute(2, "id"));
  BOOST_CHECK("99d3f61a-4c6b-11ec-a98e-7f000001aa8b" == ccdbResponse.getStringAttribute(3, "id"));
}

BOOST_AUTO_TEST_CASE(TestCCDBResponseBrowse)
{
  std::string* firstResponseString = new std::string(firstFullResponse);
  std::string* secondResponseString = new std::string(secondFullResponse);
  CCDBResponse firstCcdbResponse(*firstResponseString);
  CCDBResponse secondCcdbResponse(*secondResponseString);

  firstCcdbResponse.browseAndMerge(&secondCcdbResponse);
  BOOST_CHECK(firstCcdbResponse.objectNum == 4);
  BOOST_CHECK("407f3a65-4c7b-11ec-8cf8-200114580202" == firstCcdbResponse.getStringAttribute(0, "id"));
  BOOST_CHECK("52d3f61a-4c6b-11ec-a98e-7f000001aa8b" == firstCcdbResponse.getStringAttribute(1, "id"));
  BOOST_CHECK("e5183d1a-4c7a-11ec-9d71-7f000001aa8b" == firstCcdbResponse.getStringAttribute(2, "id"));
  BOOST_CHECK("99d3f61a-4c6b-11ec-a98e-7f000001aa8b" == firstCcdbResponse.getStringAttribute(3, "id"));
}

BOOST_AUTO_TEST_CASE(TestCCDBResponseLatest)
{
  std::string* firstResponseString = new std::string(firstFullResponse);
  std::string* secondResponseString = new std::string(secondFullResponse);
  CCDBResponse firstCcdbResponse(*firstResponseString);
  CCDBResponse secondCcdbResponse(*secondResponseString);

  firstCcdbResponse.latestAndMerge(&secondCcdbResponse);
  BOOST_CHECK(firstCcdbResponse.objectNum == 3);
  BOOST_CHECK("407f3a65-4c7b-11ec-8cf8-200114580202" == firstCcdbResponse.getStringAttribute(0, "id"));
  BOOST_CHECK("52d3f61a-4c6b-11ec-a98e-7f000001aa8b" == firstCcdbResponse.getStringAttribute(1, "id"));
  BOOST_CHECK("99d3f61a-4c6b-11ec-a98e-7f000001aa8b" == firstCcdbResponse.getStringAttribute(2, "id"));
}
