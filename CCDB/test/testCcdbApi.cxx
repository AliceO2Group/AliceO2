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
#include "CCDB/IdPath.h"    // just as test object
#include "CommonUtils/RootChain.h" // just as test object
#include "CCDB/CCDBTimeStampUtils.h"
#include <boost/test/unit_test.hpp>
#include <boost/filesystem.hpp>
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
#include <TString.h>
#include <sys/types.h>
#include <unistd.h>

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
    cout << "Is host reachable ? --> " << hostReachable << endl;
    basePath = string("Test/pid") + getpid() + "/";
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
    std::cout << "*** " << boost::unit_test::framework::current_test_case().p_name << " ***" << std::endl;
  }
  ~test_fixture() = default;

  CcdbApi api;
  map<string, string> metadata;
};

BOOST_AUTO_TEST_CASE(storeTMemFile_test, *utf::precondition(if_reachable()))
{
  test_fixture f;

  TH1F h1("th1name", "th1name", 100, 0, 99);
  h1.FillRandom("gaus", 10000);
  BOOST_CHECK_EQUAL(h1.ClassName(), "TH1F");
  cout << basePath + "th1" << endl;
  f.api.storeAsTFile(&h1, basePath + "th1", f.metadata);

  TGraph graph(10);
  graph.SetPoint(0, 2, 3);
  f.api.storeAsTFile(&graph, basePath + "graph", f.metadata);

  TTree tree("mytree", "mytree");
  int a = 4;
  tree.Branch("det", &a, "a/I");
  tree.Fill();
  f.api.storeAsTFile(&tree, basePath + "tree", f.metadata);
}

BOOST_AUTO_TEST_CASE(store_retrieve_TMemFile_templated_test, *utf::precondition(if_reachable()))
{
  test_fixture f;

  // try to store a user defined class
  // since we don't depend on anything, we are putting an object known to CCDB
  o2::ccdb::IdPath path;
  path.setPath("HelloWorld");

  f.api.storeAsTFileAny(&path, basePath + "CCDBPath", f.metadata);

  // try to retrieve strongly typed user defined class
  // since we don't depend on anything, we are using an object known to CCDB
  o2::ccdb::IdPath* path2 = nullptr;

  path2 = f.api.retrieveFromTFileAny<o2::ccdb::IdPath>(basePath + "CCDBPath", f.metadata);
  BOOST_CHECK_NE(path2, nullptr);

  // check some non-trivial data content
  BOOST_CHECK(path2 && path2->getPathString().CompareTo("HelloWorld") == 0);

  // try to query with different type and verify that we get nullptr
  BOOST_CHECK(f.api.retrieveFromTFileAny<o2::utils::RootChain>(basePath + "CCDBPath", f.metadata) == nullptr);

  //-----------------------------------------------------------------------------------------------
  // test if writing/reading complicated objects like TTree works (because of particular ownership)
  // ----------------------------------------------------------------------------------------------
  auto tree = new TTree("tree123", "tree123");
  int a = 4;
  tree->Branch("det", &a, "a/I");
  tree->Fill();
  f.api.storeAsTFileAny(tree, basePath + "tree2", f.metadata);
  delete tree;

  // read back
  tree = f.api.retrieveFromTFileAny<TTree>(basePath + "tree2", f.metadata);
  BOOST_CHECK(tree != nullptr);
  BOOST_CHECK(tree != nullptr && std::strcmp(tree->GetName(), "tree123") == 0);
  BOOST_CHECK(tree != nullptr && tree->GetEntries() == 1);

  // ---------------------------
  // test the snapshot mechanism
  // ---------------------------
  // a) create a local snapshot of the Test folder
  auto ph = boost::filesystem::unique_path();
  boost::filesystem::create_directories(ph);
  f.api.snapshot("Test", ph.string(), o2::ccdb::getCurrentTimestamp());
  std::cout << "Creating snapshot at " << ph.string() << "\n";

  // b) init a new instance from the snapshot and query something from it
  o2::ccdb::CcdbApi snapshot;
  snapshot.init("file://" + ph.string());

  // c) query from the snapshot
  BOOST_CHECK(snapshot.retrieveFromTFileAny<o2::ccdb::IdPath>(basePath + "CCDBPath", f.metadata) != nullptr);

  {
    auto tree = snapshot.retrieveFromTFileAny<TTree>(basePath + "tree2", f.metadata);
    BOOST_CHECK(tree != nullptr);
    BOOST_CHECK(tree != nullptr && std::strcmp(tree->GetName(), "tree123") == 0);
    BOOST_CHECK(tree != nullptr && tree->GetEntries() == 1);
  }

  // d) cleanup local snapshot
  if (boost::filesystem::exists(ph)) {
    boost::filesystem::remove_all(ph);
  }
}

/// A test verifying that the DB responds the correct result for given timestamps
BOOST_AUTO_TEST_CASE(timestamptest, *utf::precondition(if_reachable()))
{
  test_fixture f;

  // try to store a user defined class
  // since we don't depend on anything, we are putting an object known to CCDB
  o2::ccdb::IdPath path;
  path.setPath("HelloWorld");

  const long timestamp = 1000;             // inclusive start of validity
  const long endvalidity = timestamp + 10; // exclusive end of validitiy
  f.api.storeAsTFileAny(&path, basePath + "CCDBPathUnitTest", f.metadata, timestamp, endvalidity);

  // try to retrieve strongly typed user defined class
  // since we don't depend on anything, we are using an object known to CCDB
  o2::ccdb::IdPath* path2 = nullptr;

  path2 = f.api.retrieveFromTFileAny<o2::ccdb::IdPath>(basePath + "CCDBPathUnitTest", f.metadata, timestamp);
  BOOST_CHECK_NE(path2, nullptr);

  // check that we get something for the whole time range
  for (int t = timestamp; t < endvalidity; ++t) {
    auto p = f.api.retrieveFromTFileAny<o2::ccdb::IdPath>(basePath + "CCDBPathUnitTest", f.metadata, t);
    BOOST_CHECK_NE(p, nullptr);
  }

  // check that answer is null for anything outside
  auto plower = f.api.retrieveFromTFileAny<o2::ccdb::IdPath>(basePath + "CCDBPathUnitTest", f.metadata, timestamp - 1);
  BOOST_CHECK(plower == nullptr);

  auto pupper = f.api.retrieveFromTFileAny<o2::ccdb::IdPath>(basePath + "CCDBPathUnitTest", f.metadata, endvalidity);
  BOOST_CHECK(pupper == nullptr);
}

BOOST_AUTO_TEST_CASE(retrieveTMemFile_test, *utf::precondition(if_reachable()))
{
  test_fixture f;

  TObject* obj = f.api.retrieveFromTFile(basePath + "th1", f.metadata);
  BOOST_CHECK_NE(obj, nullptr);
  BOOST_CHECK_EQUAL(obj->ClassName(), "TH1F");
  auto h1 = dynamic_cast<TH1F*>(obj);
  BOOST_CHECK_NE(h1, nullptr);
  BOOST_CHECK_EQUAL(obj->GetName(), "th1name");
  delete obj;

  obj = f.api.retrieveFromTFile(basePath + "graph", f.metadata);
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

  obj = f.api.retrieveFromTFile(basePath + "tree", f.metadata);
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

BOOST_AUTO_TEST_CASE(truncate_test, *utf::precondition(if_reachable()))
{
  test_fixture f;

  TH1F h("object1", "object1", 100, 0, 99);
  f.api.storeAsTFile(&h, basePath + "Detector", f.metadata); // test with explicit dates
  auto h1 = f.api.retrieveFromTFile(basePath + "Detector", f.metadata);
  BOOST_CHECK(h1 != nullptr);
  f.api.truncate(basePath + "Detector");
  h1 = f.api.retrieveFromTFile(basePath + "Detector", f.metadata);
  BOOST_CHECK(h1 == nullptr);
}

BOOST_AUTO_TEST_CASE(delete_test, *utf::precondition(if_reachable()))
{
  test_fixture f;

  TH1F h1("object1", "object1", 100, 0, 99);
  long from = o2::ccdb::getCurrentTimestamp();
  long to = o2::ccdb::getFutureTimestamp(60 * 60 * 24 * 365 * 10);
  f.api.storeAsTFile(&h1, basePath + "Detector", f.metadata, from, to); // test with explicit dates
  auto h2 = f.api.retrieveFromTFile(basePath + "Detector", f.metadata);
  BOOST_CHECK(h2 != nullptr);
  f.api.deleteObject(basePath + "Detector");
  h2 = f.api.retrieveFromTFile(basePath + "Detector", f.metadata);
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
  f.api.truncate(basePath + "Detector*");
  s = f.api.list(basePath + "Detector", false, "application/json");
  int countObjects = 0;
  int countSubfolders = 0;
  countItems(s, countObjects, countSubfolders);
  BOOST_CHECK_EQUAL(countObjects, 0);

  // more complex tree
  TH1F h1("object1", "object1", 100, 0, 99);
  cout << "storing object 1 in Test" << endl;
  f.api.storeAsTFile(&h1, "Test", f.metadata);
  cout << "storing object 2 in Test/Detector" << endl;
  f.api.storeAsTFile(&h1, basePath + "Detector", f.metadata);
  cout << "storing object 3 in Test/Detector" << endl;
  f.api.storeAsTFile(&h1, basePath + "Detector", f.metadata);
  cout << "storing object 4 in Test/Detector" << endl;
  f.api.storeAsTFile(&h1, basePath + "Detector", f.metadata);
  cout << "storing object 5 in Test/Detector/Sub/abc" << endl;
  f.api.storeAsTFile(&h1, basePath + "Detector/Sub/abc", f.metadata);

  s = f.api.list(basePath + "Detector", false, "application/json");
  countItems(s, countObjects, countSubfolders);
  BOOST_CHECK_EQUAL(countObjects, 3);
  BOOST_CHECK_EQUAL(countSubfolders, 1);

  s = f.api.list(basePath + "Detector*", false, "application/json");
  countItems(s, countObjects, countSubfolders);
  BOOST_CHECK_EQUAL(countObjects, 4);
  BOOST_CHECK_EQUAL(countSubfolders, 0);

  s = f.api.list(basePath + "Detector", true, "application/json");
  countItems(s, countObjects, countSubfolders);
  BOOST_CHECK_EQUAL(countObjects, 1);
}
