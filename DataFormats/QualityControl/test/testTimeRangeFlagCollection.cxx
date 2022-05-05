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

#define BOOST_TEST_MODULE Test quality_control TimeRangeFlagCollection class
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

// boost includes
#include <boost/test/unit_test.hpp>
// STL
#include <sstream>
// o2 includes
#include "DataFormatsQualityControl/TimeRangeFlagCollection.h"
#include "DataFormatsQualityControl/TimeRangeFlag.h"

using namespace o2::quality_control;

BOOST_AUTO_TEST_CASE(test_TimeRangeFlagCollection_Methods)
{
  TimeRangeFlag trf1{12, 34, FlagReasonFactory::BadTracking(), "comment", "source"};
  TimeRangeFlag trf2{10, 34, FlagReasonFactory::BadTracking(), "comment", "source"};

  TimeRangeFlagCollection trfc1{"Raw data checks", "TOF", {10, 20000}, 12345, "LHC22k5", "passMC", "qc_mc"};
  trfc1.insert(trf1); // by copy
  trfc1.insert(trf2);
  trfc1.insert({50, 77, FlagReasonFactory::Invalid()}); // by move
  BOOST_CHECK_EQUAL(trfc1.size(), 3);
  BOOST_CHECK_EQUAL(trfc1.getName(), "Raw data checks");
  BOOST_CHECK_EQUAL(trfc1.getDetector(), "TOF");
  BOOST_CHECK_EQUAL(trfc1.getStart(), 10);
  BOOST_CHECK_EQUAL(trfc1.getEnd(), 20000);
  BOOST_CHECK_EQUAL(trfc1.getRunNumber(), 12345);
  BOOST_CHECK_EQUAL(trfc1.getPeriodName(), "LHC22k5");
  BOOST_CHECK_EQUAL(trfc1.getPassName(), "passMC");
  BOOST_CHECK_EQUAL(trfc1.getProvenance(), "qc_mc");

  TimeRangeFlagCollection trfc2{"Reco checks", "TOF"};
  trfc2.insert({50, 77, FlagReasonFactory::Invalid()}); // this is a duplicate to an entry in trfc1
  trfc2.insert({51, 77, FlagReasonFactory::Invalid()});
  trfc2.insert({1234, 3434, FlagReasonFactory::LimitedAcceptance()});
  trfc2.insert({50, 77, FlagReasonFactory::LimitedAcceptance()});
  BOOST_CHECK_EQUAL(trfc2.size(), 4);

  // Try merging. Duplicate entries should be left in the 'other' objects.
  // Notice that we merge the two partial TRFCs into the third, which covers all cases
  TimeRangeFlagCollection trfc3{"ALL", "TOF"};
  trfc3.merge(trfc1);
  trfc3.merge(trfc2);
  BOOST_CHECK_EQUAL(trfc1.size(), 0);
  BOOST_CHECK_EQUAL(trfc2.size(), 1);
  BOOST_CHECK_EQUAL(trfc3.size(), 6);

  // Try const merging. It should copy the elements and keep the 'other' intact.
  TimeRangeFlagCollection trfc4{"ALL", "TOF"};
  const auto& constTrfc3 = trfc3;
  trfc4.merge(constTrfc3);
  BOOST_CHECK_EQUAL(trfc3.size(), 6);
  BOOST_CHECK_EQUAL(trfc4.size(), 6);

  // Try merging different detectors - it should throw.
  TimeRangeFlagCollection trfc5{"ALL", "TPC"};
  BOOST_CHECK_THROW(trfc5.merge(trfc3), std::runtime_error);
  BOOST_CHECK_THROW(trfc5.merge(constTrfc3), std::runtime_error);

  // try printing
  std::cout << trfc3 << std::endl;

  // iterating
  for (const auto& trf : trfc3) {
    (void)trf;
  }
}

BOOST_AUTO_TEST_CASE(test_TimeRangeFlagCollection_IO)
{
  {
    TimeRangeFlagCollection trfc1{"xyz", "TST"};

    std::stringstream store;
    trfc1.streamTo(store);

    TimeRangeFlagCollection trfc2{"xyz", "TST"};
    trfc2.streamFrom(store);

    BOOST_CHECK_EQUAL(trfc2.size(), 0);
  }
  {
    TimeRangeFlagCollection trfc1{"xyz", "TST"};
    trfc1.insert({50, 77, FlagReasonFactory::Invalid(), "a comment", "a source"});
    trfc1.insert({51, 77, FlagReasonFactory::Invalid()});
    trfc1.insert({1234, 3434, FlagReasonFactory::LimitedAcceptance()});
    trfc1.insert({50, 77, FlagReasonFactory::LimitedAcceptance()});
    trfc1.insert({43434, 63421, FlagReasonFactory::NotBadFlagExample()});

    std::stringstream store;
    trfc1.streamTo(store);

    TimeRangeFlagCollection trfc2{"xyz", "TST"};
    trfc2.streamFrom(store);

    BOOST_REQUIRE_EQUAL(trfc1.size(), trfc2.size());
    for (auto it1 = trfc1.begin(), it2 = trfc2.begin(); it1 != trfc1.end() && it2 != trfc2.end(); ++it1, ++it2) {
      BOOST_CHECK_EQUAL(*it1, *it2);
    }
  }
  {
    std::stringstream store;
    store << "start,end,flag_id,invalid,header,format\n";
    store << R"(123,345,11,"fdsa",1,"comment","source")";
    TimeRangeFlagCollection trfc1{"A", "TST"};
    BOOST_CHECK_THROW(trfc1.streamFrom(store), std::runtime_error);
  }
  {
    std::stringstream store;
    store << "start,end,flag_id,flag_name,flag_bad,comment,source\n";
    store << R"(123,345,11,"fdsa",1,"comment","source","toomanycolumns")" << '\n';
    store << R"(123,345,11,"fdsa",1)" << '\n';
    store << R"(123,,11,"fdsa",1,"comment","source")" << '\n';
    store << R"(,345,11,"fdsa",1,"comment","source")" << '\n';
    store << R"(123,345,,"fdsa",1,"comment","source")" << '\n';
    store << R"(123,345,11,"",1,"comment","source")" << '\n';
    store << R"(123,345,11,"fdsa",,"comment","source")" << '\n';
    TimeRangeFlagCollection trfc1{"A", "TST"};
    BOOST_CHECK_NO_THROW(trfc1.streamFrom(store));
    BOOST_CHECK_EQUAL(trfc1.size(), 0);
  }
}
