// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test quality_control TimeRangeFlagCollection class
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

// boost includes
#include <boost/test/unit_test.hpp>

// o2 includes
#include "DataFormatsQualityControl/TimeRangeFlagCollection.h"
#include "DataFormatsQualityControl/TimeRangeFlag.h"

using namespace o2::quality_control;

BOOST_AUTO_TEST_CASE(test_TimeRangeFlagCollection)
{
  TimeRangeFlag trf1{12, 34, FlagReasonFactory::ProcessingError(), "comment", "source"};
  TimeRangeFlag trf2{10, 34, FlagReasonFactory::ProcessingError(), "comment", "source"};

  TimeRangeFlagCollection trfc1{"Raw data checks", "TOF"};
  trfc1.insert(trf1); // by copy
  trfc1.insert(trf2);
  trfc1.insert({50, 77, FlagReasonFactory::Invalid()}); // by move
  BOOST_CHECK_EQUAL(trfc1.size(), 3);

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