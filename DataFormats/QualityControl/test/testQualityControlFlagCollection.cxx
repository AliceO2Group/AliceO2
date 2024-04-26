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

#define BOOST_TEST_MODULE Test quality_control QualityControlFlagCollection class
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

// boost includes
#include <boost/test/unit_test.hpp>
// STL
#include <sstream>
// o2 includes
#include "DataFormatsQualityControl/QualityControlFlagCollection.h"
#include "DataFormatsQualityControl/QualityControlFlag.h"
#include "DataFormatsQualityControl/FlagTypeFactory.h"

using namespace o2::quality_control;

BOOST_AUTO_TEST_CASE(test_QualityControlFlagCollection_Methods)
{
  QualityControlFlag flag1{12, 34, FlagTypeFactory::BadTracking(), "comment", "source"};
  QualityControlFlag flag2{10, 34, FlagTypeFactory::BadTracking(), "comment", "source"};

  QualityControlFlagCollection qcfc1{"Raw data checks", "TOF", {10, 20000}, 12345, "LHC22k5", "passMC", "qc_mc"};
  qcfc1.insert(flag1); // by copy
  qcfc1.insert(flag2);
  qcfc1.insert({50, 77, FlagTypeFactory::Invalid()}); // by move
  BOOST_CHECK_EQUAL(qcfc1.size(), 3);
  BOOST_CHECK_EQUAL(qcfc1.getName(), "Raw data checks");
  BOOST_CHECK_EQUAL(qcfc1.getDetector(), "TOF");
  BOOST_CHECK_EQUAL(qcfc1.getStart(), 10);
  BOOST_CHECK_EQUAL(qcfc1.getEnd(), 20000);
  BOOST_CHECK_EQUAL(qcfc1.getRunNumber(), 12345);
  BOOST_CHECK_EQUAL(qcfc1.getPeriodName(), "LHC22k5");
  BOOST_CHECK_EQUAL(qcfc1.getPassName(), "passMC");
  BOOST_CHECK_EQUAL(qcfc1.getProvenance(), "qc_mc");

  QualityControlFlagCollection qcfc2{"Reco checks", "TOF"};
  qcfc2.insert({50, 77, FlagTypeFactory::Invalid()}); // this is a duplicate to an entry in qcfc1
  qcfc2.insert({51, 77, FlagTypeFactory::Invalid()});
  qcfc2.insert({1234, 3434, FlagTypeFactory::LimitedAcceptance()});
  qcfc2.insert({50, 77, FlagTypeFactory::LimitedAcceptance()});
  BOOST_CHECK_EQUAL(qcfc2.size(), 4);

  // Try merging. Duplicate entries should be left in the 'other' objects.
  // Notice that we merge the two partial TRFCs into the third, which covers all cases
  QualityControlFlagCollection qcfc3{"ALL", "TOF"};
  qcfc3.merge(qcfc1);
  qcfc3.merge(qcfc2);
  BOOST_CHECK_EQUAL(qcfc1.size(), 0);
  BOOST_CHECK_EQUAL(qcfc2.size(), 1);
  BOOST_CHECK_EQUAL(qcfc3.size(), 6);

  // Try const merging. It should copy the elements and keep the 'other' intact.
  QualityControlFlagCollection qcfc4{"ALL", "TOF"};
  const auto& constTrfc3 = qcfc3;
  qcfc4.merge(constTrfc3);
  BOOST_CHECK_EQUAL(qcfc3.size(), 6);
  BOOST_CHECK_EQUAL(qcfc4.size(), 6);

  // Try merging different detectors - it should throw.
  QualityControlFlagCollection qcfc5{"ALL", "TPC"};
  BOOST_CHECK_THROW(qcfc5.merge(qcfc3), std::runtime_error);
  BOOST_CHECK_THROW(qcfc5.merge(constTrfc3), std::runtime_error);

  // try printing
  std::cout << qcfc3 << std::endl;

  // iterating
  for (const auto& flag : qcfc3) {
    (void)flag;
  }
}

BOOST_AUTO_TEST_CASE(test_QualityControlFlagCollection_IO)
{
  {
    QualityControlFlagCollection qcfc1{"xyz", "TST"};

    std::stringstream store;
    qcfc1.streamTo(store);

    QualityControlFlagCollection qcfc2{"xyz", "TST"};
    qcfc2.streamFrom(store);

    BOOST_CHECK_EQUAL(qcfc2.size(), 0);
  }
  {
    QualityControlFlagCollection qcfc1{"xyz", "TST"};
    qcfc1.insert({50, 77, FlagTypeFactory::Invalid(), "a comment", "a source"});
    qcfc1.insert({51, 77, FlagTypeFactory::Invalid()});
    qcfc1.insert({1234, 3434, FlagTypeFactory::LimitedAcceptance()});
    qcfc1.insert({50, 77, FlagTypeFactory::LimitedAcceptance()});
    qcfc1.insert({43434, 63421, FlagTypeFactory::NotBadFlagExample()});

    std::stringstream store;
    qcfc1.streamTo(store);

    QualityControlFlagCollection qcfc2{"xyz", "TST"};
    qcfc2.streamFrom(store);

    BOOST_REQUIRE_EQUAL(qcfc1.size(), qcfc2.size());
    for (auto it1 = qcfc1.begin(), it2 = qcfc2.begin(); it1 != qcfc1.end() && it2 != qcfc2.end(); ++it1, ++it2) {
      BOOST_CHECK_EQUAL(*it1, *it2);
    }
  }
  {
    std::stringstream store;
    store << "start,end,flag_id,invalid,header,format\n";
    store << R"(123,345,11,"fdsa",1,"comment","source")";
    QualityControlFlagCollection qcfc1{"A", "TST"};
    BOOST_CHECK_THROW(qcfc1.streamFrom(store), std::runtime_error);
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
    QualityControlFlagCollection qcfc1{"A", "TST"};
    BOOST_CHECK_NO_THROW(qcfc1.streamFrom(store));
    BOOST_CHECK_EQUAL(qcfc1.size(), 0);
  }
}
