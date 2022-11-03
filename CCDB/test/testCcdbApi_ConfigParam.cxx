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

///
/// \file   testCcdbApi_ConfigParam.cxx
/// \author Sandro Wenzel
///

#define BOOST_TEST_MODULE CCDB
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "CCDB/CcdbApi.h"
#include "CommonUtils/ConfigurableParam.h"
#include "DetectorsVertexing/PVertexerParams.h"
#include "CCDB/CCDBTimeStampUtils.h"
#include <boost/test/unit_test.hpp>
#include <filesystem>
#include <cstdio>
#include <cassert>
#include <iostream>
#include <cstdio>
#include <curl/curl.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <chrono>
#include <CommonUtils/StringUtils.h>
#include <sys/types.h>
#include <unistd.h>

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/foreach.hpp>
#include <boost/optional/optional.hpp>

using namespace std;
using namespace o2::ccdb;
namespace utf = boost::unit_test;
namespace tt = boost::test_tools;

static string ccdbUrl;
static string basePath;
bool hostReachable = false;

/**
 * Global fixture, ie general setup and teardown
 */
struct Fixture {
  Fixture()
  {
    CcdbApi api;
    ccdbUrl = "http://ccdb-test.cern.ch:8080";
    api.init(ccdbUrl);
    cout << "ccdb url: " << ccdbUrl << endl;
    hostReachable = api.isHostReachable();
    char hostname[_POSIX_HOST_NAME_MAX];
    gethostname(hostname, _POSIX_HOST_NAME_MAX);
    basePath = string("Test/") + hostname + "/pid" + getpid() + "/";
    cout << "Path we will use in this test suite : " + basePath << endl;
  }
  ~Fixture()
  {
    if (hostReachable) {
      CcdbApi api;
      map<string, string> metadata;
      api.init(ccdbUrl);
      api.truncate(basePath + "*");
      cout << "Test data truncated (" << basePath << ")" << endl;
    }
  }
};
BOOST_GLOBAL_FIXTURE(Fixture);

/**
 * Just an accessor to the hostReachable variable to be used to determine whether tests can be ran or not.
 */
struct if_reachable {
  tt::assertion_result operator()(utf::test_unit_id)
  {
    return hostReachable;
  }
};

/**
 * Fixture for the tests, i.e. code is ran in every test that uses it, i.e. it is like a setup and teardown for tests.
 */
struct test_fixture {
  test_fixture()
  {
    api.init(ccdbUrl);
    metadata["Hello"] = "World";
    std::cout << "*** " << boost::unit_test::framework::current_test_case().p_name << " ***" << std::endl;
  }
  ~test_fixture() = default;

  CcdbApi api;
  map<string, string> metadata;
};

BOOST_AUTO_TEST_CASE(testConfigParamRetrieval, *utf::precondition(if_reachable()))
{
  test_fixture f;

  // We'd like to demonstrate the following:
  // GIVEN:
  // - user modifies field in a config param from command line (RT)
  // - user fetches config param from CCDB
  //
  // WE'D LIKE TO ARRIVE AT A STATE with
  // - the returned object from CCDB gets syncs with the config param registry
  // - everything is consistent

  // fetch the default instance
  auto& p1 = o2::vertexing::PVertexerParams::Instance();

  // update the config system with some runtime keys
  o2::conf::ConfigurableParam::updateFromString("pvertexer.dbscanDeltaT=-3.");

  std::map<std::string, std::string> headers;
  std::map<std::string, std::string> meta;
  long from = o2::ccdb::getCurrentTimestamp();
  auto* object = f.api.retrieveFromTFileAny<o2::vertexing::PVertexerParams>("GLO/Config/PVertexer", meta, from + 1, &headers);
  BOOST_CHECK(object != nullptr);
  BOOST_CHECK(object->getMemberProvenance("dbscanDeltaT") == o2::conf::ConfigurableParam::EParamProvenance::kRT);
  BOOST_CHECK(object->getMemberProvenance("useMeanVertexConstraint") == o2::conf::ConfigurableParam::EParamProvenance::kCCDB);
  BOOST_CHECK(p1.getMemberProvenance("dbscanDeltaT") == o2::conf::ConfigurableParam::EParamProvenance::kRT);
  BOOST_CHECK(p1.getMemberProvenance("useMeanVertexConstraint") == o2::conf::ConfigurableParam::EParamProvenance::kCCDB);
  BOOST_CHECK(object == &p1);
}
