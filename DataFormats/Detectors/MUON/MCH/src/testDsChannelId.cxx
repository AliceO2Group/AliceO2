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
#define BOOST_TEST_MODULE MCH DsChannelId
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>
#include <boost/test/data/monomorphic.hpp>

#include "DataFormatsMCH/DsChannelId.h"
#include "DataFormatsMCH/DsChannelDetId.h"
#include "Framework/TypeTraits.h"

BOOST_AUTO_TEST_CASE(DsChannelIdIsTriviallyCopyable)
{
  BOOST_CHECK_EQUAL(std::is_trivially_copyable<o2::mch::DsChannelId>::value, true);
}

BOOST_AUTO_TEST_CASE(DsChannelIdIsMessageable)
{
  BOOST_CHECK_EQUAL(o2::framework::is_messageable<o2::mch::DsChannelId>::value, true);
}

BOOST_AUTO_TEST_CASE(DsChannelDetIdEncode)
{
  o2::mch::DsChannelDetId id(1025, 1361, 63);
  BOOST_CHECK_EQUAL(id.getDeId(), 1025);
  BOOST_CHECK_EQUAL(id.getDsId(), 1361);
  BOOST_CHECK_EQUAL(id.getChannel(), 63);
}
