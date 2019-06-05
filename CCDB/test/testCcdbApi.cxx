// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

///
/// \file   testCcdbApi.cxx
/// \author Barthelemy von Haller
///

#define BOOST_TEST_MODULE CCDB
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "CCDB/CcdbApi.h"
#include <boost/test/unit_test.hpp>
#include <cassert>
#include <iostream>
#include <cstdio>
#include <curl/curl.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <TH1F.h>
#include <chrono>
#include <CommonUtils/StringUtils.h>

using namespace std;
using namespace o2::ccdb;
namespace utf = boost::unit_test;
namespace tt = boost::test_tools;

static string ccdbUrl = "http://ccdb-test.cern.ch:8080";
bool hostReachable = false;

/**
 * Global fixture, ie general setup and teardown
 */
struct Fixture {
  Fixture()
  {
    CcdbApi api;
    api.init(ccdbUrl);
    hostReachable = api.isHostReachable();
    cout << "Is host reachable ? --> " << hostReachable << endl;
  }
  ~Fixture() = default;
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
    std::cout << "*** " << boost::unit_test::framework::current_test_case().p_name << " ***" << std::endl;
  }
  ~test_fixture() = default;

  CcdbApi api;
  map<string, string> metadata;
};

long getFutureTimestamp(int secondsInFuture)
{
  std::chrono::seconds sec(secondsInFuture);
  auto future = std::chrono::system_clock::now() + sec;
  auto future_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(future);
  auto epoch = future_ms.time_since_epoch();
  auto value = std::chrono::duration_cast<std::chrono::milliseconds>(epoch);
  return value.count();
}

long getCurrentTimestamp()
{
  auto now = std::chrono::system_clock::now();
  auto now_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(now);
  auto epoch = now_ms.time_since_epoch();
  auto value = std::chrono::duration_cast<std::chrono::milliseconds>(epoch);
  return value.count();
}

BOOST_AUTO_TEST_CASE(store_test, *utf::precondition(if_reachable()))
{
  test_fixture f;

  auto h1 = new TH1F("object1", "object1", 100, 0, 99);
  f.api.store(h1, "Test/Detector", f.metadata);
}

BOOST_AUTO_TEST_CASE(retrieve_test, *utf::precondition(if_reachable()))
{
  test_fixture f;

  auto h1 = f.api.retrieve("Test/Detector", f.metadata);
  BOOST_CHECK(h1 != nullptr);
  if (h1 != nullptr) {
    BOOST_CHECK_EQUAL(h1->GetName(), "object1");
  }

  auto h2 = f.api.retrieve("asdf/asdf", f.metadata);
  BOOST_CHECK(h2 == nullptr);
}

BOOST_AUTO_TEST_CASE(truncate_test, *utf::precondition(if_reachable()))
{
  test_fixture f;

  auto h1 = f.api.retrieve("Test/Detector", f.metadata);
  BOOST_CHECK(h1 != nullptr);
  f.api.truncate("Test/Detector");
  h1 = f.api.retrieve("Test/Detector", f.metadata);
  BOOST_CHECK(h1 == nullptr);
}

BOOST_AUTO_TEST_CASE(delete_test, *utf::precondition(if_reachable()))
{
  test_fixture f;

  auto h1 = new TH1F("object1", "object1", 100, 0, 99);
  long from = getCurrentTimestamp();
  long to = getFutureTimestamp(60 * 60 * 24 * 365 * 10);
  f.api.store(h1, "Test/Detector", f.metadata, from, to); // test with explicit dates
  auto h2 = f.api.retrieve("Test/Detector", f.metadata);
  BOOST_CHECK(h2 != nullptr);
  f.api.deleteObject("Test/Detector");
  h2 = f.api.retrieve("Test/Detector", f.metadata);
  BOOST_CHECK(h2 == nullptr);
}

void countItems(const string& s, int& countObjects, int& countSubfolders)
{
  countObjects = 0;
  countSubfolders = 0;
  std::stringstream ss(s);
  std::string line;
  bool subfolderMode = false;
  while (std::getline(ss, line, '\n')) {
    o2::utils::ltrim(line);
    o2::utils::rtrim(line);
    if (line.length() == 0) {
      continue;
    }

    if (line.find("subfolders") != std::string::npos) {
      subfolderMode = true;
      continue;
    }

    if (subfolderMode) {
      countSubfolders++;
      if (line.find(']') != std::string::npos) {
        break;
      }
    }

    if (line.find(R"("path")") == 0) {
      countObjects++;
    }
  }
}

BOOST_AUTO_TEST_CASE(list_test, *utf::precondition(if_reachable()))
{
  test_fixture f;

  // test non-empty top dir
  string s = f.api.list("", "application/json"); // top dir
  long nbLines = std::count(s.begin(), s.end(), '\n') + 1;
  BOOST_CHECK(nbLines > 5);

  // test empty dir
  f.api.truncate("Test/Detector*");
  s = f.api.list("Test/Detector", false, "application/json");
  int countObjects = 0;
  int countSubfolders = 0;
  countItems(s, countObjects, countSubfolders);
  BOOST_CHECK_EQUAL(countObjects, 0);

  // more complex tree
  auto h1 = new TH1F("object1", "object1", 100, 0, 99);
  cout << "storing object 1 in Test" << endl;
  f.api.store(h1, "Test", f.metadata);
  cout << "storing object 2 in Test/Detector" << endl;
  f.api.store(h1, "Test/Detector", f.metadata);
  cout << "storing object 3 in Test/Detector" << endl;
  f.api.store(h1, "Test/Detector", f.metadata);
  cout << "storing object 4 in Test/Detector" << endl;
  f.api.store(h1, "Test/Detector", f.metadata);
  cout << "storing object 5 in Test/Detector/Sub/abc" << endl;
  f.api.store(h1, "Test/Detector/Sub/abc", f.metadata);

  s = f.api.list("Test/Detector", false, "application/json");
  countItems(s, countObjects, countSubfolders);
  BOOST_CHECK_EQUAL(countObjects, 3);
  BOOST_CHECK_EQUAL(countSubfolders, 1);

  s = f.api.list("Test/Detector*", false, "application/json");
  countItems(s, countObjects, countSubfolders);
  BOOST_CHECK_EQUAL(countObjects, 4);
  BOOST_CHECK_EQUAL(countSubfolders, 0);

  s = f.api.list("Test/Detector", true, "application/json");
  countItems(s, countObjects, countSubfolders);
  BOOST_CHECK_EQUAL(countObjects, 1);
}
