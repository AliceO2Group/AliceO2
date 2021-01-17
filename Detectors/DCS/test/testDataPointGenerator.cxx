// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test DCS DataPointGenerator
#define BOOST_TEST_MAIN

#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>
#include <iostream>
#include "DetectorsDCS/DataPointGenerator.h"
#include <algorithm>

BOOST_AUTO_TEST_CASE(GenerateDouble)
{
  double fmin = 1620.0;
  double fmax = 1710.5;
  auto fbi = o2::dcs::generateRandomDataPoints({"TST/SECTOR[00..06]/CRATE[0..3]/voltage"}, fmin, fmax, "2022-November-18 12:34:56");

  BOOST_CHECK_EQUAL(fbi.size(), 28);

  for (auto dp : fbi) {
    BOOST_CHECK_EQUAL(dp.id.get_type(), o2::dcs::DeliveryType::RAW_DOUBLE);
    double value = o2::dcs::getValue<double>(dp);
    BOOST_CHECK(value >= fmin && value <= fmax);
  }
}

BOOST_AUTO_TEST_CASE(GenerateInt)
{
  uint32_t imin = 0;
  uint32_t imax = 3;
  auto fbi = o2::dcs::generateRandomDataPoints({"TST/SECTOR[00..06]/CRATE[0..3]/current"}, imin, imax, "2022-November-18 12:34:56");

  BOOST_CHECK_EQUAL(fbi.size(), 28);

  for (auto dp : fbi) {
    BOOST_CHECK_EQUAL(dp.id.get_type(), o2::dcs::DeliveryType::RAW_UINT);
    double value = o2::dcs::getValue<uint32_t>(dp);
    BOOST_CHECK(value >= imin && value <= imax);
    BOOST_CHECK_THROW(o2::dcs::getValue<double>(dp), std::runtime_error);
    BOOST_CHECK_THROW(o2::dcs::getValue<int32_t>(dp), std::runtime_error);
  }
}

BOOST_AUTO_TEST_CASE(GenerateBool)
{
  auto fbi = o2::dcs::generateRandomDataPoints<bool>({"TST/SECTOR[00..06]/status"}, 0, 1, "2022-November-18 12:34:56");

  BOOST_CHECK_EQUAL(fbi.size(), 7);

  for (auto dp : fbi) {
    BOOST_CHECK_EQUAL(dp.id.get_type(), o2::dcs::DeliveryType::RAW_BOOL);
    BOOST_CHECK_NO_THROW(o2::dcs::getValue<bool>(dp));
    BOOST_CHECK_THROW(o2::dcs::getValue<int>(dp), std::runtime_error);
  }
}

BOOST_AUTO_TEST_CASE(GenerateString)
{
  auto fbi = o2::dcs::generateRandomDataPoints<std::string>({"TST/SECTOR[00..06]/name"}, "123", "1234567", "2022-November-18 12:34:56");

  BOOST_CHECK_EQUAL(fbi.size(), 7);

  for (auto dp : fbi) {
    BOOST_CHECK_EQUAL(dp.id.get_type(), o2::dcs::DeliveryType::RAW_STRING);
    BOOST_CHECK_NO_THROW(o2::dcs::getValue<std::string>(dp));
    BOOST_CHECK_THROW(o2::dcs::getValue<int>(dp), std::runtime_error);
    auto value = o2::dcs::getValue<std::string>(dp);
    BOOST_CHECK(value.size() >= 3);
    BOOST_CHECK(value.size() <= 7);
  }
}
