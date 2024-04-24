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

#define BOOST_TEST_MODULE Test quality_control QualityControlFlag class
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

// boost includes
#include <boost/test/unit_test.hpp>

// o2 includes
#include "DataFormatsQualityControl/QualityControlFlag.h"

using namespace o2::quality_control;

BOOST_AUTO_TEST_CASE(test_QualityControlFlag)
{
  QualityControlFlag qcFlag1{12, 34, FlagTypeFactory::BadTracking(), "comment", "source"};

  BOOST_CHECK_EQUAL(qcFlag1.getStart(), 12);
  BOOST_CHECK_EQUAL(qcFlag1.getEnd(), 34);
  BOOST_CHECK_EQUAL(qcFlag1.getFlag(), FlagTypeFactory::BadTracking());
  BOOST_CHECK_EQUAL(qcFlag1.getComment(), "comment");
  BOOST_CHECK_EQUAL(qcFlag1.getSource(), "source");

  BOOST_CHECK_THROW((QualityControlFlag{12, 0, FlagTypeFactory::BadTracking()}), std::runtime_error);

  QualityControlFlag qcFlag2{10, 34, FlagTypeFactory::BadTracking(), "comment", "source"};

  BOOST_CHECK(qcFlag1 > qcFlag2);
  BOOST_CHECK(!(qcFlag1 < qcFlag2));
}