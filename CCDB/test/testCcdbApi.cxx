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
#include <TMessage.h>
#include <TStreamerInfo.h>
#include <TGraph.h>
#include <TTree.h>

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

BOOST_AUTO_TEST_CASE(storeTMemFile_test, *utf::precondition(if_reachable()))
{
  test_fixture f;

  auto h1 = new TH1F("th1name", "th1name", 100, 0, 99);
  h1->FillRandom("gaus", 10000);
  BOOST_CHECK_EQUAL(h1->ClassName(), "TH1F");
  f.api.storeAsTFile(h1, "Test/th1", f.metadata);

  auto graph = new TGraph(10);
  graph->SetPoint(0, 2, 3);
  f.api.storeAsTFile(graph, "Test/graph", f.metadata);

  auto tree = new TTree("mytree", "mytree");
  int a = 4;
  tree->Branch("det", &a, "a/I");
  tree->Fill();
  f.api.storeAsTFile(tree, "Test/tree", f.metadata);
}

BOOST_AUTO_TEST_CASE(retrieveTMemFile_test, *utf::precondition(if_reachable()))
{
  test_fixture f;

  TObject* obj = f.api.retrieveFromTFile("Test/th1", f.metadata);
  BOOST_CHECK_NE(obj, nullptr);
  BOOST_CHECK_EQUAL(obj->ClassName(), "TH1F");
  auto h1 = dynamic_cast<TH1F*>(obj);
  BOOST_CHECK_NE(h1, nullptr);
  BOOST_CHECK_EQUAL(obj->GetName(), "th1name");
  delete obj;

  obj = f.api.retrieveFromTFile("Test/graph", f.metadata);
  BOOST_CHECK_NE(obj, nullptr);
  BOOST_CHECK_EQUAL(obj->ClassName(), "TGraph");
  auto graph = dynamic_cast<TGraph*>(obj);
  BOOST_CHECK_NE(graph, nullptr);
  double x, y;
  int ret = graph->GetPoint(0, x, y);
  BOOST_CHECK_EQUAL(ret, 0);
  BOOST_CHECK_EQUAL(x, 2);
  BOOST_CHECK_EQUAL(graph->GetN(), 10);
  delete graph;

  obj = f.api.retrieveFromTFile("Test/tree", f.metadata);
  BOOST_CHECK_NE(obj, nullptr);
  BOOST_CHECK_EQUAL(obj->ClassName(), "TTree");
  auto tree = dynamic_cast<TTree*>(obj);
  BOOST_CHECK_NE(tree, nullptr);
  BOOST_CHECK_EQUAL(tree->GetName(), "mytree");
  delete obj;

  // wrong url
  obj = f.api.retrieveFromTFile("Wrong/wrong", f.metadata);
  BOOST_CHECK_EQUAL(obj, nullptr);
}

BOOST_AUTO_TEST_CASE(store_test, *utf::precondition(if_reachable()))
{
  test_fixture f;

  auto h1 = new TH1F("object1", "object1", 100, 0, 99);
  h1->FillRandom("gaus", 10000);
  f.api.store(h1, "Test/Detector", f.metadata, -1, -1, true);

  auto h2 = new TH1F("object2", "object2", 100, 0, 99);
  h2->FillRandom("gaus", 10000);
  f.api.store(h2, "Test/Detector", f.metadata, -1, -1, true);
}

BOOST_AUTO_TEST_CASE(retrieve_wrong_type, *utf::precondition(if_reachable())) // Test/Detector is not stored as a TFile
{
  test_fixture f;

  TObject* obj = f.api.retrieveFromTFile("Test/Detector", f.metadata);
  BOOST_CHECK_EQUAL(obj, nullptr);
}

BOOST_AUTO_TEST_CASE(retrieve_test, *utf::precondition(if_reachable()))
{
  test_fixture f;

  auto h1 = new TH1F("object1", "object1", 100, 0, 99);
  h1->FillRandom("gaus", 10000);
  f.api.store(h1, "Test/Detector", f.metadata, -1, -1, true);

  auto h2 = f.api.retrieve("Test/Detector", f.metadata);
  BOOST_CHECK(h2 != nullptr);
  if (h2 != nullptr) {
    BOOST_CHECK_EQUAL(h2->GetName(), "object1");
  }

  auto h3 = f.api.retrieve("asdf/asdf", f.metadata);
  BOOST_CHECK(h3 == nullptr);
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
