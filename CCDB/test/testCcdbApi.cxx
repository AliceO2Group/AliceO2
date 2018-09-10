///
/// \file   testCcdbApi.cxx
/// \author Barthelemy von Haller
///

#include "CCDB/CcdbApi.h"

#define BOOST_TEST_MODULE Quality test
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include <cassert>
#include <iostream>
#include "curl/curl.h"

#include <stdio.h>
#include <curl/curl.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <TH1F.h>

using namespace std;
using namespace o2::ccdb;

struct test_fixture
{
    test_fixture()
    {
      api.init("http://ccdb-test.cern.ch:8080");
    }

    ~test_fixture()
    {
    }

    CcdbApi api;
    map<string, string> metadata;
};

BOOST_AUTO_TEST_CASE(store_test)
{
  test_fixture f;

  auto h1 = new TH1F("object1", "object1", 100, 0, 99);
  f.api.store(h1, "Test/Detector", f.metadata);
}

BOOST_AUTO_TEST_CASE(retrieve_test)
{
  test_fixture f;

  auto h1 = f.api.retrieve("Test/Detector", f.metadata);
  BOOST_CHECK(h1 != nullptr);
  BOOST_CHECK_EQUAL(h1->GetName(), "object1");

  auto h2 = f.api.retrieve("asdf/asdf", f.metadata);
  BOOST_CHECK_EQUAL(h2, nullptr);
}

BOOST_AUTO_TEST_CASE(truncate_test)
{
  test_fixture f;

  auto h1 = f.api.retrieve("Test/Detector", f.metadata);
  BOOST_CHECK(h1 != nullptr);
  f.api.truncate("Test/Detector");
  h1 = f.api.retrieve("Test/Detector", f.metadata);
  BOOST_CHECK(h1 == nullptr);
}

BOOST_AUTO_TEST_CASE(delete_test)
{
  test_fixture f;

  auto h1 = new TH1F("object1", "object1", 100, 0, 99);
  f.api.store(h1, "Test/Detector", f.metadata);
  auto h2 = f.api.retrieve("Test/Detector", f.metadata);
  BOOST_CHECK(h2 != nullptr);
  f.api.deleteObject("Test/Detector");
  h2 = f.api.retrieve("Test/Detector", f.metadata);
  BOOST_CHECK(h2 == nullptr);
}


/// trim from start (in place)
/// https://stackoverflow.com/questions/216823/whats-the-best-way-to-trim-stdstring
static inline void ltrim(std::string &s)
{
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](int ch) {
    return !std::isspace(ch);
  }));
}

/// trim from end (in place)
/// https://stackoverflow.com/questions/216823/whats-the-best-way-to-trim-stdstring
static inline void rtrim(std::string &s)
{
  s.erase(std::find_if(s.rbegin(), s.rend(), [](int ch) {
    return !std::isspace(ch);
  }).base(), s.end());
}

void countItems(const string &s, int &countObjects, int &countSubfolders)
{
  countObjects = 0;
  countSubfolders = 0;
  std::stringstream ss(s);
  std::string line;
  bool subfolderMode = false;
  while (std::getline(ss, line, '\n')) {
    ltrim(line);
    rtrim(line);
    if (line.length() == 0) {
      continue;
    }

    if (line.find("subfolders") != std::string::npos) {
      subfolderMode = true;
      continue;
    }

    if (subfolderMode) {
      if (line.find(']') == 0) {
        break;
      } else {
        countSubfolders++;
      }
    }

    if (line.find("\"path\"") == 0) {
      countObjects++;
    }
  }
}

BOOST_AUTO_TEST_CASE(list_test)
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
  f.api.store(h1, "Test", f.metadata);
  f.api.store(h1, "Test/Detector", f.metadata);
  f.api.store(h1, "Test/Detector", f.metadata);
  f.api.store(h1, "Test/Detector", f.metadata);
  f.api.store(h1, "Test/Detector/Sub/abc", f.metadata);

  s = f.api.list("Test/Detector", false, "application/json");
  countItems(s, countObjects, countSubfolders);
  BOOST_CHECK_EQUAL(countObjects, 3);
//  BOOST_CHECK_EQUAL(countSubfolders, 1);

  s = f.api.list("Test/Detector*", false, "application/json");
  countItems(s, countObjects, countSubfolders);
  cout << "s : " << s << endl;
  BOOST_CHECK_EQUAL(countObjects, 4);
//  BOOST_CHECK_EQUAL(countSubfolders, 0);

  s = f.api.list("Test/Detector", true, "application/json");
  countItems(s, countObjects, countSubfolders);
  BOOST_CHECK_EQUAL(countObjects, 1);
}
