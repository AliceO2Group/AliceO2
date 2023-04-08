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

#include <boost/test/tools/old/interface.hpp>
#include <type_traits>
#define BOOST_TEST_MODULE MCH ChannelCode
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>
#include <boost/test/data/monomorphic.hpp>

#include "MCHGlobalMapping/ChannelCode.h"

BOOST_AUTO_TEST_CASE(CtorShowThrowForInvalidDeId)
{
  BOOST_CHECK_THROW(o2::mch::ChannelCode(42, 0), std::runtime_error);
}

BOOST_AUTO_TEST_CASE(CtorShowThrowForInvalidDePadIndex)
{
  BOOST_CHECK_THROW(o2::mch::ChannelCode(1002, 7616), std::runtime_error);
}

BOOST_AUTO_TEST_CASE(CtorShowThrowForInvalidSolarId)
{
  BOOST_CHECK_THROW(o2::mch::ChannelCode(0, 0, 0), std::runtime_error);
}

BOOST_AUTO_TEST_CASE(Ctor1)
{
  // same pad expressed two different ways
  std::array<o2::mch::ChannelCode, 2> ids = {
    o2::mch::ChannelCode(1002, 7615),
    o2::mch::ChannelCode(557, 7, 60)};

  for (const auto& id : ids) {
    BOOST_CHECK_EQUAL(id.isValid(), true);
    BOOST_CHECK_EQUAL(id.getDeId(), 1002);
    BOOST_CHECK_EQUAL(id.getDsId(), 1361);
    BOOST_CHECK_EQUAL(id.getChannel(), 60);
    BOOST_CHECK_EQUAL(id.getSolarId(), 557);
    BOOST_CHECK_EQUAL(id.getElinkId(), 7);
    BOOST_CHECK_EQUAL(id.getDePadIndex(), 7615);
  }
}

BOOST_AUTO_TEST_CASE(Ctor2)
{
  // same pad expressed two different ways
  std::array<o2::mch::ChannelCode, 2> ids = {
    o2::mch::ChannelCode(100, 28626),
    o2::mch::ChannelCode(325, 39, 63)};

  for (const auto& id : ids) {
    BOOST_CHECK_EQUAL(id.isValid(), true);
    BOOST_CHECK_EQUAL(id.getDeId(), 100);
    BOOST_CHECK_EQUAL(id.getDsId(), 1267);
    BOOST_CHECK_EQUAL(id.getChannel(), 63);
    BOOST_CHECK_EQUAL(id.getSolarId(), 325);
    BOOST_CHECK_EQUAL(id.getElinkId(), 39);
    BOOST_CHECK_EQUAL(id.getDePadIndex(), 28626);
  }
}

BOOST_AUTO_TEST_CASE(ChannelCodeCanBeUsedAsMapKey)
{
  std::map<o2::mch::ChannelCode, int> maptest;
  o2::mch::ChannelCode cc(1002, 7615);
  maptest[cc] = 42;
  BOOST_CHECK_EQUAL(maptest[cc], 42);
}

BOOST_AUTO_TEST_CASE(DefaultConstructorYieldInvalidValue)
{
  o2::mch::ChannelCode cc;
  BOOST_CHECK_EQUAL(cc.isValid(), false);
}
