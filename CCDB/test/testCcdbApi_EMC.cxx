// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// Ccdb unit tests focusing on EMC setup
///
/// \author Sandro Wenzel
///

#define BOOST_TEST_MODULE CCDB
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "CCDB/CcdbApi.h"
#include <TH1.h>
#include <boost/test/unit_test.hpp>
#include <iostream>

using namespace std;
using namespace o2::ccdb;
namespace utf = boost::unit_test;
namespace tt = boost::test_tools;

static string ccdbUrl;
bool hostReachable = false;

/**
 * Global fixture, ie general setup and teardown
 */
struct Fixture {
  Fixture()
  {
    CcdbApi api;
    ccdbUrl = "http://emcccdb-test.cern.ch:8080";
    api.init(ccdbUrl);
    cout << "ccdb url: " << ccdbUrl << endl;
    hostReachable = api.isHostReachable();
    cout << "Is host reachable ? --> " << hostReachable << endl;
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

// handle the case where the object comes from alien and redirect does not work with curl
BOOST_AUTO_TEST_CASE(retrieveTemplated_ALIEN, *utf::precondition(if_reachable()))
{
  test_fixture f;

  // try to retrieve an object from the production instance, including the headers
  std::map<std::string, std::string> headers;
  std::map<std::string, std::string> meta;

  std::string path("/qc/EMC/MO/CellTask/digitOccupancyEMC/1606419105647");
  {
    auto* object = f.api.retrieveFromTFileAny<TH1>(path, meta, -1, &headers);
    BOOST_CHECK(object != nullptr);
    LOG(INFO) << headers["Content-Location"];
    if (object) {
      BOOST_CHECK(headers.size() > 0);
      LOG(INFO) << "Histo name " << object->GetName();
      LOG(INFO) << "Number of bins " << object->GetNbinsX() << " Mean " << object->GetMean();
    }
  }

  // it should also work without headers of course
  {
    auto* object = f.api.retrieveFromTFileAny<TH1>(path, meta);
    BOOST_CHECK(object != nullptr);
  }
}
