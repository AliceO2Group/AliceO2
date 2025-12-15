// Copyright 2019-2025 CERN and copyright holders of ALICE O2.
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
/// \file   testCcdbApiHeaders.cxx
/// \brief  Test BasicCCDBManager header/metadata information functionality with caching
/// \author martin.oines.eide@cern.ch

#define BOOST_TEST_MODULE CCDB
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <set>
#include "CCDB/BasicCCDBManager.h"
#include "CCDB/CCDBTimeStampUtils.h"
#include "CCDB/CcdbApi.h"
#include <boost/test/unit_test.hpp>

static std::string basePath;
// std::string ccdbUrl = "http://localhost:8080";
std::string ccdbUrl = "http://ccdb-test.cern.ch:8080";
bool hostReachable = false;

/**
 * Global fixture, ie general setup and teardown
 * Copied from testBasicCCDBManager.cxx
 */
struct Fixture {
  Fixture()
  {
    auto& ccdbManager = o2::ccdb::BasicCCDBManager::instance();
    if (std::getenv("ALICEO2_CCDB_HOST")) {
      ccdbUrl = std::string(std::getenv("ALICEO2_CCDB_HOST"));
    }
    ccdbManager.setURL(ccdbUrl);
    hostReachable = ccdbManager.getCCDBAccessor().isHostReachable();
    char hostname[_POSIX_HOST_NAME_MAX];
    gethostname(hostname, _POSIX_HOST_NAME_MAX);
    basePath = std::string("Users/m/meide/Tests/") + hostname + "/pid-" + getpid() + "/BasicCCDBManager/";

    LOG(info) << "Path we will use in this test suite : " + basePath << std::endl;
    LOG(info) << "ccdb url: " << ccdbUrl << std::endl;
    LOG(info) << "Is host reachable ? --> " << hostReachable << std::endl;
  }
  ~Fixture()
  {
    if (hostReachable) {
      o2::ccdb::BasicCCDBManager::instance().getCCDBAccessor().truncate(basePath + "*"); // This deletes the data after test is run, disable if you want to inspect the data
      LOG(info) << "Test data truncated/deleted (" << basePath << ")" << std::endl;
    }
  }
};
BOOST_GLOBAL_FIXTURE(Fixture);
/**
 * Just an accessor to the hostReachable variable to be used to determine whether tests can be ran or not.
 * Copied from testCcdbApi.cxx
 */
struct if_reachable {
  boost::test_tools::assertion_result operator()(boost::unit_test::test_unit_id)
  {
    return hostReachable;
  }
};

// Only compare known and stable keys (avoid volatile ones like Date)
static const std::set<std::string> sStableKeys = {
  "ETag",
  "Valid-From",
  "Valid-Until",
  "Created",
  "Last-Modified",
  "Content-Disposition",
  "Content-Location",
  "path",
  "partName",
  "Content-MD5",
  "Hello" // TODO find other headers to compare to
};

// Test that we get back the same header header keys as we put in (for stable keys)

BOOST_AUTO_TEST_CASE(testCachedHeaders, *boost::unit_test::precondition(if_reachable()))
{
  /// ━━━━━━━ ARRANGE ━━━━━━━━━
  // First store objects to test with
  auto& ccdbManager = o2::ccdb::BasicCCDBManager::instance();
  std::string pathA = basePath + "CachingA";
  std::string pathB = basePath + "CachingB";
  std::string pathC = basePath + "CachingC";
  std::string ccdbObjO = "testObjectO";
  std::string ccdbObjN = "testObjectN";
  std::string ccdbObjX = "testObjectX";
  std::map<std::string, std::string> md = {
    {"Hello", "World"},
    {"Key1", "Value1"},
    {"Key2", "Value2"},
  };
  long start = 1000, stop = 3000;
  ccdbManager.getCCDBAccessor().storeAsTFileAny<std::string>(&ccdbObjO, pathA, md, start, stop);
  ccdbManager.getCCDBAccessor().storeAsTFileAny<std::string>(&ccdbObjN, pathB, md, start, stop);
  ccdbManager.getCCDBAccessor().storeAsTFileAny<std::string>(&ccdbObjX, pathC, md, start, stop);
  // initilize the BasicCCDBManager
  ccdbManager.clearCache();
  ccdbManager.setCaching(true); // This is what we want to test.

  /// ━━━━━━━━━━━ ACT ━━━━━━━━━━━━
  // Plan: get one object, then another, then the first again and check the headers are the same
  std::map<std::string, std::string> headers1, headers2, headers3;

  auto* obj1 = ccdbManager.getForTimeStamp<std::string>(pathA, (start + stop) / 2, &headers1);
  auto* obj2 = ccdbManager.getForTimeStamp<std::string>(pathB, (start + stop) / 2, &headers2);
  auto* obj3 = ccdbManager.getForTimeStamp<std::string>(pathA, (start + stop) / 2, &headers3); // Should lead to a cache hit!

  /// ━━━━━━━━━━━ ASSERT ━━━━━━━━━━━━
  /// Check that we got something
  BOOST_REQUIRE(obj1 != nullptr);
  BOOST_REQUIRE(obj2 != nullptr);
  BOOST_REQUIRE(obj3 != nullptr);

  LOG(debug) << "obj1: " << *obj1;
  LOG(debug) << "obj2: " << *obj2;
  LOG(debug) << "obj3: " << *obj3;

  // Sanity check
  /// Check that the objects are correct
  BOOST_TEST(*obj1 == ccdbObjO);
  BOOST_TEST(*obj3 == ccdbObjO);
  BOOST_TEST(obj3 == obj1); // should be the same object in memory since it is cached

  BOOST_TEST(obj2 != obj1);

  (*obj1) = "ModifiedObject";
  BOOST_TEST(*obj1 == "ModifiedObject");
  BOOST_TEST(*obj3 == "ModifiedObject"); // obj3 and obj1 are the same object in memory

  // Check that the headers are the same for the two retrievals of the same object
  BOOST_REQUIRE(headers1.size() != 0);
  BOOST_REQUIRE(headers3.size() != 0);

  LOG(debug) << "Headers1 size: " << headers1.size();
  for (const auto& h : headers1) {
    LOG(debug) << "  " << h.first << " -> " << h.second;
  }
  LOG(debug) << "Headers3 size: " << headers3.size();
  for (const auto& h : headers3) {
    LOG(debug) << "  " << h.first << " -> " << h.second;
  }

  for (const auto& stableKey : sStableKeys) {
    LOG(info) << "Checking key: " << stableKey;

    BOOST_REQUIRE(headers1.count(stableKey) > 0);
    BOOST_REQUIRE(headers3.count(stableKey) > 0);
    BOOST_TEST(headers1.at(stableKey) == headers3.at(stableKey));
  }
  BOOST_TEST(headers1 != headers2, "The headers for different objects should be different");

  // Test that we can  change the map and the two headers are not affected
  headers1["NewKey"] = "NewValue";
  headers3["NewKey"] = "DifferentValue";
  BOOST_TEST(headers1["NewKey"] != headers3["NewKey"]); // This tests that we have a deep copy of the headers
}

BOOST_AUTO_TEST_CASE(testNonCachedHeaders, *boost::unit_test::precondition(if_reachable()))
{
  /// ━━━━━━━ ARRANGE ━━━━━━━━━
  // First store objects to test with
  auto& ccdbManager = o2::ccdb::BasicCCDBManager::instance();
  std::string pathA = basePath + "NonCachingA";
  std::string pathB = basePath + "NonCachingB";
  std::string ccdbObjO = "testObjectO";
  std::string ccdbObjN = "testObjectN";
  std::map<std::string, std::string> md = {
    {"Hello", "World"},
    {"Key1", "Value1"},
    {"Key2", "Value2"},
  };
  long start = 1000, stop = 2000;
  ccdbManager.getCCDBAccessor().storeAsTFileAny(&ccdbObjO, pathA, md, start, stop);
  ccdbManager.getCCDBAccessor().storeAsTFileAny(&ccdbObjN, pathB, md, start, stop);
  // initilize the BasicCCDBManager
  ccdbManager.clearCache();
  ccdbManager.setCaching(false); // This is what we want to test, no caching

  /// ━━━━━━━━━━━ ACT ━━━━━━━━━━━━
  // Plan: get one object, then another, then the first again. Then check that the contents is the same but not the object in memory
  std::map<std::string, std::string> headers1, headers2, headers3;

  auto* obj1 = ccdbManager.getForTimeStamp<std::string>(pathA, (start + stop) / 2, &headers1);
  auto* obj2 = ccdbManager.getForTimeStamp<std::string>(pathB, (start + stop) / 2, &headers2);
  auto* obj3 = ccdbManager.getForTimeStamp<std::string>(pathA, (start + stop) / 2, &headers3); // Should not be cached since explicitly disabled

  ccdbManager.setCaching(true); // Restore default state
  /// ━━━━━━━━━━━ ASSERT ━━━━━━━━━━━
  /// Check that we got something
  BOOST_REQUIRE(obj1 != nullptr);
  BOOST_REQUIRE(obj2 != nullptr);
  BOOST_REQUIRE(obj3 != nullptr);

  LOG(debug) << "obj1: " << *obj1;
  LOG(debug) << "obj2: " << *obj2;
  LOG(debug) << "obj3: " << *obj3;

  // Sanity check
  /// Check that the objects are correct
  BOOST_TEST(*obj1 == ccdbObjO);
  BOOST_TEST(*obj3 == ccdbObjO);
  BOOST_TEST(obj2 != obj1);
  BOOST_TEST(obj3 != obj1); // should NOT be the same object in memory
  (*obj1) = "ModifiedObject";
  BOOST_TEST(*obj1 == "ModifiedObject");
  BOOST_TEST(*obj3 != "ModifiedObject"); // obj3 and obj1 are NOT the same object in memory

  BOOST_TEST(headers1.size() == headers3.size());

  // Remove the date header since it may be different even for the same object since we might have asked in different seconds
  headers1.erase("Date");
  headers3.erase("Date");
  BOOST_TEST(headers1 == headers3, "The headers for the same object should be the same even if not cached");

  BOOST_TEST(headers1 != headers2, "The headers for different objects should be different");
  BOOST_TEST(headers1.size() != 0);
  BOOST_TEST(headers3.size() != 0);
  BOOST_TEST(headers2.size() != 0);
  BOOST_TEST(headers1 != headers2, "The headers for different objects should be different");

  // cleanup
  delete obj1;
  delete obj2;
  delete obj3;
}

BOOST_AUTO_TEST_CASE(CacheFirstRetrievalAndHeadersPersistence, *boost::unit_test::precondition(if_reachable()))
{
  /// ━━━━━━━ ARRANGE ━━━━━━━━━
  auto& mgr = o2::ccdb::BasicCCDBManager::instance();
  // Prepare two validity slots for same path to test ETag change later
  std::string path = basePath + "ObjA";
  std::string objV1 = "ObjectVersion1";
  std::string objV2 = "ObjectVersion2";
  std::map<std::string, std::string> meta1{
    {"UserKey1", "UValue1"},
    {"UserKey2", "UValue2"}};
  long v1start = 10'000;
  long v1stop = 20'000;
  long v2start = v1stop; // contiguous slot
  long v2stop = v2start + (v1stop - v1start);
  long mid1 = (v1start + v1stop) / 2;
  // Store 2 versions
  mgr.getCCDBAccessor().storeAsTFileAny(&objV1, path, meta1, v1start, v1stop);
  mgr.getCCDBAccessor().storeAsTFileAny(&objV2, path, meta1, v2start, v2stop);

  mgr.clearCache();
  mgr.setCaching(true);
  mgr.setFatalWhenNull(true);
  mgr.setTimestamp(mid1);

  /// ━━━━━━━ACT━━━━━━━━━
  std::map<std::string, std::string> headers1, headers2, headers4, headers5;

  // 1) First retrieval WITH headers inside 1st slot
  auto* p1 = mgr.getForTimeStamp<std::string>(path, mid1, &headers1);
  size_t fetchedSizeAfterFirst = mgr.getFetchedSize();
  // 2) Second retrieval (cache hit)
  auto* p2 = mgr.getForTimeStamp<std::string>(path, mid1, &headers2);
  size_t fetchedSizeAfterSecond = mgr.getFetchedSize();
  // 3) Third retrieval (cache hit) WITHOUT passing headers
  auto* p3 = mgr.getForTimeStamp<std::string>(path, mid1);
  // 4) Fourth retrieval with headers again -> should still produce same headers
  auto* p4 = mgr.getForTimeStamp<std::string>(path, mid1, &headers4);
  // 5) Fifth retrieval with headers again to check persistence
  auto* p5 = mgr.getForTimeStamp<std::string>(path, mid1, &headers5);

  mgr.setFatalWhenNull(false); // restore default

  /// ━━━━━━━ASSERT━━━━━━━━━

  BOOST_TEST(p1 != nullptr);
  BOOST_TEST(*p1 == objV1);

  BOOST_TEST(headers1.count("UserKey1") == 1);
  BOOST_TEST(headers1.count("UserKey2") == 1);
  BOOST_TEST(headers1["UserKey1"] == "UValue1");
  BOOST_TEST(headers1["UserKey2"] == "UValue2");
  BOOST_TEST(headers1.count("Valid-From") == 1);
  BOOST_TEST(headers1.count("Valid-Until") == 1);
  BOOST_TEST(headers1.count("ETag") == 1);

  /*  Need to manually amend the headers1 to have cache valid until for comparison sake,
   *  the header is not set in the first request.
   *  It is only set if the internal cache of CCDB has seen this object before, apperently.
   *  This will never happen in this test since it was just created and not asked for before.
   */
  headers1["Cache-Valid-Until"] = std::to_string(v1stop);

  /* In rare cases the header date might be different, if the second has ticked over between the requests
   */
  headers1.erase("Date");
  headers2.erase("Date");
  headers4.erase("Date");
  headers5.erase("Date");

  BOOST_TEST(p2 == p1);                                        // same pointer for cached scenario
  BOOST_TEST(headers2 == headers1);                            // identical header map
  BOOST_TEST(fetchedSizeAfterSecond == fetchedSizeAfterFirst); // no new fetch

  BOOST_TEST(p3 == p1);

  BOOST_TEST(p4 == p1);
  BOOST_TEST(headers4 == headers1);

  // Mutate the returned header map locally and ensure it does not corrupt internal cache
  headers4["UserKey1"] = "Tampered";
  BOOST_TEST(p5 == p1);
  BOOST_TEST(headers5["UserKey1"] == "UValue1"); // internal unchanged
}

BOOST_AUTO_TEST_CASE(FailedFetchDoesNotGiveMetadata, *boost::unit_test::precondition(if_reachable()))
{

  /// ━━━━━━━ ARRANGE ━━━━━━━━━
  auto& mgr = o2::ccdb::BasicCCDBManager::instance();
  std::string path = basePath + "FailThenRecover";
  std::string content = "ContentX";
  std::map<std::string, std::string> meta{{"Alpha", "Beta"}};
  long s = 300'000, e = 310'000;
  mgr.getCCDBAccessor().storeAsTFileAny(&content, path, meta, s, e);
  mgr.clearCache();
  mgr.setCaching(true);
  mgr.setFatalWhenNull(false);

  /// ━━━━━━━ ACT ━━━━━━━━━
  // Intentionally pick a timestamp outside validity to fail first
  long badTS = s - 1000;
  long goodTS = (s + e) / 2;
  std::map<std::string, std::string> hFail, hGood;
  auto* badObj = mgr.getForTimeStamp<std::string>(path, badTS, &hFail);
  auto* goodObj = mgr.getForTimeStamp<std::string>(path, goodTS, &hGood);

  /// ━━━━━━━ ASSERT ━━━━━━━━━
  BOOST_TEST(!hFail.empty());           // Should have some headers
  BOOST_TEST(hFail["Alpha"] != "Beta"); // But not the metadata
  BOOST_TEST(hGood.count("Alpha") == 1);
  BOOST_TEST(hGood["Alpha"] == "Beta");

  mgr.setFatalWhenNull(true);
}

BOOST_AUTO_TEST_CASE(FirstCallWithoutHeadersThenWithHeaders, *boost::unit_test::precondition(if_reachable()))
{

  auto& mgr = o2::ccdb::BasicCCDBManager::instance();
  std::string path = basePath + "LateHeaders";
  std::string body = "Late";
  std::map<std::string, std::string> meta{{"LateKey", "LateVal"}};
  long s = 400'000, e = 410'000;
  mgr.getCCDBAccessor().storeAsTFileAny(&body, path, meta, s, e);

  mgr.clearCache();
  mgr.setCaching(true);
  long ts = (s + e) / 2;

  // 1) First call with nullptr headers
  auto* first = mgr.getForTimeStamp<std::string>(path, ts);
  BOOST_TEST(first != nullptr);
  BOOST_TEST(*first == body);

  // 2) Second call asking for headers - should return the full set
  std::map<std::string, std::string> h2;
  auto* second = mgr.getForTimeStamp<std::string>(path, ts, &h2);
  BOOST_TEST(second == first);
  BOOST_TEST(h2.count("LateKey") == 1);
  BOOST_TEST(h2["LateKey"] == "LateVal");
  BOOST_TEST(h2.count("Valid-From") == 1);
  BOOST_TEST(h2.count("Valid-Until") == 1);
}

BOOST_AUTO_TEST_CASE(HeadersAreStableAcrossMultipleHits, *boost::unit_test::precondition(if_reachable()))
{

  auto& mgr = o2::ccdb::BasicCCDBManager::instance();
  std::string path = basePath + "StableHeaders";
  std::string body = "Stable";
  std::map<std::string, std::string> meta{{"HK", "HV"}};
  long s = 500'000, e = 510'000;
  mgr.getCCDBAccessor().storeAsTFileAny(&body, path, meta, s, e);

  mgr.clearCache();
  mgr.setCaching(true);
  long ts = (s + e) / 2;

  std::map<std::string, std::string> h1;
  auto* o1 = mgr.getForTimeStamp<std::string>(path, ts, &h1);
  BOOST_TEST(o1 != nullptr);
  BOOST_TEST(h1.count("HK") == 1);

  std::string etag = h1["ETag"];
  for (int i = 0; i < 15; ++i) {
    std::map<std::string, std::string> hi;
    auto* oi = mgr.getForTimeStamp<std::string>(path, ts, &hi);
    BOOST_TEST(oi == o1);
    BOOST_TEST(hi.count("HK") == 1);
    BOOST_TEST(hi["ETag"] == etag);
  }
}
