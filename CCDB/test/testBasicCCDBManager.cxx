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
/// \file   testBasicCCDBManager.cxx
/// \brief  Test BasicCCDBManager and caching functionality
/// \author ruben.shahoyan@cern.ch
///

#define BOOST_TEST_MODULE CCDB
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "CCDB/CcdbApi.h"
#include "CCDB/BasicCCDBManager.h"
#include "Framework/Logger.h"
#include <boost/test/unit_test.hpp>

using namespace o2::ccdb;

BOOST_AUTO_TEST_CASE(TestBasicCCDBManager)
{
  CcdbApi api;
  const std::string uri = "http://ccdb-test.cern.ch:8080";
  api.init(uri);
  if (!api.isHostReachable()) {
    LOG(WARNING) << "Host " << uri << " is not reacheable, abandoning the test";
    return;
  }
  //
  std::string pathA = "Test/CachingA";
  std::string pathB = "Test/CachingB";
  std::string ccdbObjO = "testObjectO";
  std::string ccdbObjN = "testObjectN";
  std::map<std::string, std::string> md;
  long start = 1000, stop = 2000;
  api.storeAsTFileAny(&ccdbObjO, pathA, md, start, stop);
  api.storeAsTFileAny(&ccdbObjN, pathA, md, stop, stop + (stop - start)); // extra slot
  api.storeAsTFileAny(&ccdbObjO, pathB, md, start, stop);

  // test reading
  auto& cdb = o2::ccdb::BasicCCDBManager::instance();
  cdb.setURL(uri);
  cdb.setTimestamp((start + stop) / 2);
  cdb.setCachingEnabled(true);

  auto* objA = cdb.get<std::string>(pathA); // will be loaded from scratch and fill the cache
  LOG(INFO) << "1st reading of A: " << *objA;
  BOOST_CHECK(objA && (*objA) == ccdbObjO); // make sure correct object is loaded

  auto* objB = cdb.get<std::string>(pathB); // will be loaded from scratch and fill the cache
  BOOST_CHECK(objB && (*objB) == ccdbObjO); // make sure correct object is loaded

  std::string hack = "Cached";
  (*objA) = hack;
  (*objB) = hack;
  objA = cdb.get<std::string>(pathA); // should get already cached and hacked object
  LOG(INFO) << "Reading of cached and modified A: " << *objA;
  BOOST_CHECK(objA && (*objA) == hack); // make sure correct object is loaded

  // now check wrong object reading, 0 will be returned and cache will be cleaned
  objA = cdb.getForTimeStamp<std::string>(pathA, start - (stop - start) / 2); // wrong time
  LOG(INFO) << "Read for wrong time, expect null: " << objA;
  BOOST_CHECK(objA == nullptr);
  objA = cdb.get<std::string>(pathA); // cache again
  LOG(INFO) << "Reading of A from scratch after error: " << *objA;
  BOOST_CHECK(objA && (*objA) != hack); // make sure we did not get cached object
  (*objA) = hack;

  // read object from another time slot
  objA = cdb.getForTimeStamp<std::string>(pathA, stop + (stop - start) / 2); // will be loaded from scratch and fill the cache
  LOG(INFO) << "Reading of A for different time slost, expect non-cached object: " << *objA;
  BOOST_CHECK(objA && (*objA) == ccdbObjN); // make sure correct object is loaded

  // clear specific object cache
  cdb.clearCache(pathA);
  objA = cdb.get<std::string>(pathA); // will be loaded from scratch and fill the cache
  LOG(INFO) << "Reading of A after cleaning its cache, expect non-cached object: " << *objA;
  BOOST_CHECK(objA && (*objA) == ccdbObjO); // make sure correct object is loaded
  (*objA) = hack;
  objA = cdb.get<std::string>(pathA); // should get already cached and hacked object
  LOG(INFO) << "Reading same A, expect cached and modified value: " << *objA;
  BOOST_CHECK(objA && (*objA) == hack); // make sure correct object is loaded

  objB = cdb.get<std::string>(pathB); // should get already cached and hacked object, since is was not reset
  LOG(INFO) << "Reading B, expect cached since only A cache was cleaned: " << *objB;
  BOOST_CHECK(objB && (*objB) == hack); // make sure correct object is loaded

  // clear all caches
  cdb.clearCache();
  objB = cdb.get<std::string>(pathB); // will be loaded from scratch and fill the cache
  LOG(INFO) << "Reading B after cleaning cache completely: " << *objB;
  BOOST_CHECK(objB && (*objB) == ccdbObjO); // make sure correct object is loaded

  // get object in TimeMachine mode in the past
  cdb.setCreatedNotAfter(1);          // set upper object validity
  objA = cdb.get<std::string>(pathA); // should not be loaded
  BOOST_CHECK(!objA);                 // make sure correct object is not loaded
  cdb.resetCreatedNotAfter();         // resetting upper validity limit

  // disable cache at all (will also clean it)
  cdb.setCachingEnabled(false);
  objA = cdb.get<std::string>(pathA); // will be loaded from scratch, w/o filling the cache
  LOG(INFO) << "Reading A after disabling the cache: " << *objA;
  BOOST_CHECK(objA && (*objA) == ccdbObjO); // make sure correct object is loaded
  (*objA) = hack;
  objA = cdb.get<std::string>(pathA); // will be loaded from scratch
  LOG(INFO) << "Reading A again, it should not be cached: " << *objA;
  BOOST_CHECK(objA && (*objA) != hack); // make sure correct object is loaded
}
