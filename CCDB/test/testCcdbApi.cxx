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
/// \file   testCcdbApi.cxx
/// \author Barthelemy von Haller
///

#define BOOST_BIND_GLOBAL_PLACEHOLDERS
#define BOOST_TEST_MODULE CCDB
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "CCDB/CcdbApi.h"
#include "CCDB/IdPath.h"    // just as test object
#include "CommonUtils/RootChain.h" // just as test object
#include "CCDB/CCDBTimeStampUtils.h"
#include <boost/test/unit_test.hpp>
#include <filesystem>
#include <iostream>
#include <TH1F.h>
#include <chrono>
#include <CommonUtils/StringUtils.h>
#include <TStreamerInfo.h>
#include <TGraph.h>
#include <TTree.h>
#include <TString.h>
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
    cout << "Is host reachable ? --> " << hostReachable << endl;
    char hostname[_POSIX_HOST_NAME_MAX];
    gethostname(hostname, _POSIX_HOST_NAME_MAX);
    basePath = string("Test/TestCcdbApi/") + hostname + "/pid" + getpid() + "/";
    // Replace dashes by underscores to avoid problems in the creation of local directories
    std::replace(basePath.begin(), basePath.end(), '-','_');
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

  // try to get the headers back and to find the metadata
  map<string, string> md;
  path2 = f.api.retrieveFromTFileAny<o2::ccdb::IdPath>(basePath + "CCDBPath", f.metadata, -1, &md);
  BOOST_CHECK_EQUAL(md.count("Hello"), 1);
  BOOST_CHECK_EQUAL(md["Hello"], "World");

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
  // std::filesystem does not yet provide boost::filesystem::unique_path() equivalent, and usin tmpnam generate a warning
  auto ph = o2::utils::Str::create_unique_path(std::filesystem::temp_directory_path().native());
  std::filesystem::create_directories(ph);
  std::cout << "Creating snapshot at " << ph << "\n";
  f.api.snapshot(basePath, ph, o2::ccdb::getCurrentTimestamp());
  std::cout << "Creating snapshot at " << ph << "\n";

  // b) init a new instance from the snapshot and query something from it
  o2::ccdb::CcdbApi snapshot;
  snapshot.init(o2::utils::Str::concat_string("file://", ph));

  // c) query from the snapshot
  BOOST_CHECK(snapshot.retrieveFromTFileAny<o2::ccdb::IdPath>(basePath + "CCDBPath", f.metadata) != nullptr);

  {
    auto tree = snapshot.retrieveFromTFileAny<TTree>(basePath + "tree2", f.metadata);
    BOOST_CHECK(tree != nullptr);
    BOOST_CHECK(tree != nullptr && std::strcmp(tree->GetName(), "tree123") == 0);
    BOOST_CHECK(tree != nullptr && tree->GetEntries() == 1);
  }

  // d) cleanup local snapshot
  if (std::filesystem::exists(ph)) {
    std::filesystem::remove_all(ph);
  }
}

BOOST_AUTO_TEST_CASE(store_max_size_test, *utf::precondition(if_reachable()))
{
  test_fixture f;

  // try to store a user defined class
  // since we don't depend on anything, we are putting an object known to CCDB
  o2::ccdb::IdPath path;
  path.setPath("HelloWorld");

  int result = f.api.storeAsTFileAny(&path, basePath + "CCDBPath", f.metadata); // ok
  BOOST_CHECK_EQUAL(result, 0);
  result = f.api.storeAsTFileAny(&path, basePath + "CCDBPath", f.metadata, -1, -1, 1 /* bytes */); // we know this will fail
  BOOST_CHECK_EQUAL(result, -1);
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

BOOST_AUTO_TEST_CASE(retrieveTemplatedWithHeaders, *utf::precondition(if_reachable()))
{
  test_fixture f;

  // first store something
  o2::ccdb::IdPath path;
  path.setPath("HelloWorld");
  long from = o2::ccdb::getCurrentTimestamp();
  long to = o2::ccdb::getFutureTimestamp(60 * 60 * 24 * 365 * 10);
  f.api.storeAsTFileAny(&path, basePath + "CCDBPathUnitTest", f.metadata, from, to);

  // then try to retrieve, including the headers
  std::map<std::string, std::string> headers;
  std::map<std::string, std::string> meta;
  cout << "basePath + \"CCDBPathUnitTest\" : " << basePath + "CCDBPathUnitTest" << endl;
  cout << "from+1 : " << from + 1 << endl;
  auto* object = f.api.retrieveFromTFileAny<o2::ccdb::IdPath>(basePath + "CCDBPathUnitTest", meta, from + 1, &headers);
  BOOST_CHECK(headers.count("Hello") == 1);
  BOOST_CHECK(headers["Hello"] == "World");
}

BOOST_AUTO_TEST_CASE(retrieveTMemFile_test, *utf::precondition(if_reachable()))
{
  test_fixture f;

  TObject* obj = f.api.retrieveFromTFileAny<TObject>(basePath + "th1", f.metadata);
  BOOST_CHECK_NE(obj, nullptr);
  BOOST_CHECK_EQUAL(obj->ClassName(), "TH1F");
  auto h1 = dynamic_cast<TH1F*>(obj);
  BOOST_CHECK_NE(h1, nullptr);
  BOOST_CHECK_EQUAL(obj->GetName(), "th1name");
  delete obj;

  obj = f.api.retrieveFromTFileAny<TObject>(basePath + "graph", f.metadata);
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

  std::map<std::string, std::string> headers;
  obj = f.api.retrieveFromTFileAny<TObject>(basePath + "tree", f.metadata, -1, &headers);
  BOOST_CHECK_NE(obj, nullptr);
  BOOST_CHECK_EQUAL(obj->ClassName(), "TTree");
  auto tree = dynamic_cast<TTree*>(obj);
  BOOST_CHECK_NE(tree, nullptr);
  BOOST_CHECK_EQUAL(tree->GetName(), "mytree");
  delete obj;
  // make sure we got the correct metadata
  BOOST_CHECK(headers.count("Hello") == 1);
  BOOST_CHECK_EQUAL(headers["Hello"], "World");

  // wrong url
  obj = f.api.retrieveFromTFileAny<TObject>("Wrong/wrong", f.metadata);
  BOOST_CHECK_EQUAL(obj, nullptr);
}

BOOST_AUTO_TEST_CASE(truncate_test, *utf::precondition(if_reachable()))
{
  test_fixture f;

  TH1F h("object1", "object1", 100, 0, 99);
  f.api.storeAsTFile(&h, basePath + "Detector", f.metadata); // test with explicit dates
  auto h1 = f.api.retrieveFromTFileAny<TH1F>(basePath + "Detector", f.metadata);
  BOOST_CHECK(h1 != nullptr);
  f.api.truncate(basePath + "Detector");
  h1 = f.api.retrieveFromTFileAny<TH1F>(basePath + "Detector", f.metadata);
  BOOST_CHECK(h1 == nullptr);
}

BOOST_AUTO_TEST_CASE(delete_test, *utf::precondition(if_reachable()))
{
  test_fixture f;

  TH1F h1("object1", "object1", 100, 0, 99);
  long from = o2::ccdb::getCurrentTimestamp();
  long to = o2::ccdb::getFutureTimestamp(60 * 60 * 24 * 365 * 10);
  f.api.storeAsTFile(&h1, basePath + "Detector", f.metadata, from, to); // test with explicit dates
  auto h2 = f.api.retrieveFromTFileAny<TH1F>(basePath + "Detector", f.metadata);
  BOOST_CHECK(h2 != nullptr);
  f.api.deleteObject(basePath + "Detector");
  h2 = f.api.retrieveFromTFileAny<TH1F>(basePath + "Detector", f.metadata);
  BOOST_CHECK(h2 == nullptr);
}

void countItems(const string& s, int& countObjects, int& countSubfolders)
{
  countObjects = 0;
  countSubfolders = 0;
  std::stringstream ss(s);

  boost::property_tree::ptree pt;
  boost::property_tree::read_json(ss, pt);

  if (pt.count("objects") > 0) {
    countObjects = pt.get_child("objects").size();
  }

  if (pt.count("subfolders") > 0) {
    countSubfolders = pt.get_child("subfolders").size();
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

BOOST_AUTO_TEST_CASE(TestHeaderParsing)
{
  std::vector<std::string> headers = {
    "HTTP/1.1 200",
    "Content-Location: /download/6dcb77c0-ca56-11e9-a807-200114580202",
    "Date: Thu, 26 Sep 2019 08:09:20 GMT",
    "Valid-Until: 1567771311999",
    "Valid-From: 1567080816927",
    "InitialValidityLimit: 1598616816927",
    "Created: 1567080816956",
    "ETag: \"6dcb77c0-ca56-11e9-a807-200114580202\"",
    "Last-Modified: Thu, 29 Aug 2019 12:13:36 GMT",
    "UpdatedFrom: 2001:1458:202:28:0:0:100:35",
    "partName: send",
    "Content-Disposition: inline;filename=\"o2::dataformats::CalibLHCphaseTOF_1567080816916.root\"",
    "Accept-Ranges: bytes",
    "Content-MD5: 9481c9d036660f80e21dae5943c2096f",
    "Content-Type: application/octet-stream",
    "Content-Length: 2097152"};
  std::vector<std::string> results;
  std::string etag;
  CcdbApi::parseCCDBHeaders(headers, results, etag);
  BOOST_CHECK_EQUAL(etag, "\"6dcb77c0-ca56-11e9-a807-200114580202\"");
  BOOST_REQUIRE_EQUAL(results.size(), 1);
  BOOST_CHECK_EQUAL(results[0], "/download/6dcb77c0-ca56-11e9-a807-200114580202");
}

BOOST_AUTO_TEST_CASE(TestFetchingHeaders, *utf::precondition(if_reachable()))
{
  // first store the object
  string objectPath = basePath + "objectETag";
  test_fixture f;
  TH1F h1("objectETag", "objectETag", 100, 0, 99);
  f.api.storeAsTFile(&h1, objectPath, f.metadata);

  // then get the headers
  std::string etag;
  std::vector<std::string> headers;
  std::vector<std::string> pfns;
  string path = objectPath + "/" + std::to_string(getCurrentTimestamp());
  auto updated = CcdbApi::getCCDBEntryHeaders("http://ccdb-test.cern.ch:8080/" + path, etag, headers);
  BOOST_CHECK_EQUAL(updated, true);
  BOOST_REQUIRE(headers.size() != 0);
  CcdbApi::parseCCDBHeaders(headers, pfns, etag);
  BOOST_REQUIRE(etag != "");
  BOOST_REQUIRE(pfns.size());
  updated = CcdbApi::getCCDBEntryHeaders("http://ccdb-test.cern.ch:8080/" + path, etag, headers);
  BOOST_CHECK_EQUAL(updated, false);
}

BOOST_AUTO_TEST_CASE(TestRetrieveHeaders, *utf::precondition(if_reachable()))
{
  test_fixture f;

  TH1F h1("object1", "object1", 100, 0, 99);
  cout << "storing object 1 in " << basePath << "Test" << endl;
  map<string, string> metadata;
  metadata["custom"] = "whatever";
  f.api.storeAsTFile(&h1, basePath + "Test", metadata);

  std::map<std::string, std::string> headers = f.api.retrieveHeaders(basePath + "Test", metadata);
  BOOST_CHECK_NE(headers.size(), 0);
  std::string h = headers["custom"];
  BOOST_CHECK_EQUAL(h, "whatever");

  int i = 0;
  for (auto h : headers) {
    cout << i++ << " : " << h.first << " -> " << h.second << endl;
  }

  headers = f.api.retrieveHeaders(basePath + "Test", metadata);
  BOOST_CHECK_NE(headers.size(), 0);
  h = headers["custom"];
  BOOST_CHECK_EQUAL(h, "whatever");

  metadata["custom"] = "something";
  headers = f.api.retrieveHeaders(basePath + "Test", metadata);

  i = 0;
  for (auto h : headers) {
    cout << i++ << " : " << h.first << " -> " << h.second << endl;
  }
  BOOST_CHECK_EQUAL(headers.size(), 0);
}

BOOST_AUTO_TEST_CASE(TestUpdateMetadata, *utf::precondition(if_reachable()))
{
  test_fixture f;

  // upload an object
  TH1F h1("object1", "object1", 100, 0, 99);
  cout << "storing object 1 in " << basePath << "Test" << endl;
  map<string, string> metadata;
  metadata["custom"] = "whatever";
  metadata["id"] = "first";
  f.api.storeAsTFile(&h1, basePath + "Test", metadata);

  // retrieve the headers just to be sure
  std::map<std::string, std::string> headers = f.api.retrieveHeaders(basePath + "Test", metadata);
  BOOST_CHECK(headers.count("custom") > 0);
  BOOST_CHECK(headers.at("custom") == "whatever");
  string firstID = headers.at("ETag");
  firstID.erase(std::remove(firstID.begin(), firstID.end(), '"'), firstID.end());

  map<string, string> newMetadata;
  newMetadata["custom"] = "somethingelse";

  // update the metadata and check
  f.api.updateMetadata(basePath + "Test", newMetadata, o2::ccdb::getCurrentTimestamp());
  headers = f.api.retrieveHeaders(basePath + "Test", newMetadata);
  BOOST_CHECK(headers.count("custom") > 0);
  BOOST_CHECK(headers.at("custom") == "somethingelse");

  // add a second object
  cout << "storing object 2 in " << basePath << "Test" << endl;
  metadata.clear();
  metadata["custom"] = "whatever";
  metadata["id"] = "second";
  f.api.storeAsTFile(&h1, basePath + "Test", metadata);

  // get id
  cout << "get id" << endl;
  headers = f.api.retrieveHeaders(basePath + "Test", metadata);
  string secondID = headers.at("ETag");
  secondID.erase(std::remove(secondID.begin(), secondID.end(), '"'), secondID.end());

  // update the metadata by id
  cout << "update the metadata by id" << endl;
  newMetadata.clear();
  newMetadata["custom"] = "first";
  f.api.updateMetadata(basePath + "Test", newMetadata, o2::ccdb::getCurrentTimestamp(), firstID);
  newMetadata.clear();
  newMetadata["custom"] = "second";
  f.api.updateMetadata(basePath + "Test", newMetadata, o2::ccdb::getCurrentTimestamp(), secondID);

  // check
  metadata.clear();
  metadata["id"] = "first";
  headers = f.api.retrieveHeaders(basePath + "Test", metadata);
  BOOST_CHECK(headers.count("custom") > 0);
  BOOST_CHECK(headers.at("custom") == "first");
  metadata.clear();
  metadata["id"] = "second";
  headers = f.api.retrieveHeaders(basePath + "Test", metadata);
  BOOST_CHECK(headers.count("custom") > 0);
  BOOST_CHECK(headers.at("custom") == "second");
}

BOOST_AUTO_TEST_CASE(multi_host_test)
{
  CcdbApi api;
  api.init("http://bogus-host.cern.ch,http://ccdb-test.cern.ch:8080");
  std::map<std::string, std::string> metadata;
  std::map<std::string, std::string> headers;
  o2::pmr::vector<char> dst;
  std::string url = "Analysis/ALICE3/Centrality";
  api.loadFileToMemory(dst, url, metadata, 1645780010602, &headers, "", "", "", true);
  BOOST_CHECK(dst.size() != 0);
}

BOOST_AUTO_TEST_CASE(vectored)
{
  CcdbApi api;
  api.init("http://ccdb-test.cern.ch:8080");

  int TEST_SAMPLE_SIZE = 5;
  std::vector<o2::pmr::vector<char>> dests(TEST_SAMPLE_SIZE);
  std::vector<std::map<std::string, std::string>> metadatas(TEST_SAMPLE_SIZE);
  std::vector<std::map<std::string, std::string>> headers(TEST_SAMPLE_SIZE);

  std::vector<CcdbApi::RequestContext> contexts;
  for (int i = 0; i < TEST_SAMPLE_SIZE; i++) {
    contexts.push_back(CcdbApi::RequestContext(dests.at(i), metadatas.at(i), headers.at(i)));
    contexts.at(i).path = "Analysis/ALICE3/Centrality";
    contexts.at(i).timestamp = 1645780010602;
    contexts.at(i).considerSnapshot = true;
  }

  api.vectoredLoadFileToMemory(contexts);

  for (auto context : contexts) {
    BOOST_CHECK(context.dest.size() != 0);
  }
}