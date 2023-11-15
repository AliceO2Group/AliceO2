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
#define BOOST_TEST_MODULE MCH StatusMap
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>
#include <boost/test/data/monomorphic.hpp>

#include "MCHGlobalMapping/ChannelCode.h"
#include "MCHStatus/StatusMap.h"
#include <fmt/format.h>

namespace bdata = boost::unit_test::data;

BOOST_AUTO_TEST_CASE(CtorBuildAnEmptyMap)
{
  o2::mch::StatusMap statusMap;

  BOOST_CHECK_EQUAL(statusMap.empty(), true);
}

std::vector<o2::mch::DsChannelId> chid{{42, 17, 63},
                                       {320, 34, 1}};

std::vector<o2::mch::ChannelCode> cc{
  {302, 20117}, // same as chid[0]
  {100, 15665}, // same as chid[1]
  {1025, 8},
  {515, 1863}};

BOOST_AUTO_TEST_CASE(ClearShouldGiveEmptyMap)
{
  o2::mch::StatusMap statusMap;
  statusMap.add(cc, o2::mch::StatusMap::kBadPedestal);
  statusMap.clear();
  BOOST_CHECK_EQUAL(statusMap.empty(), true);
}

BOOST_AUTO_TEST_CASE(AddChannelIdWithInvalidMaskShouldThrow)
{
  o2::mch::StatusMap statusMap;
  BOOST_CHECK_THROW(statusMap.add(chid, 4), std::runtime_error);
}

BOOST_AUTO_TEST_CASE(AddChannelCodeWithInvalidMaskShouldThrow)
{
  o2::mch::StatusMap statusMap;
  BOOST_CHECK_THROW(statusMap.add(cc, 4), std::runtime_error);
}

std::array<uint32_t, 3> maskList = {
  o2::mch::StatusMap::kBadPedestal,
  o2::mch::StatusMap::kRejectList,
  o2::mch::StatusMap::kRejectList + o2::mch::StatusMap::kBadPedestal};

BOOST_DATA_TEST_CASE(CheckAddedChannelsGetTheRightMask, bdata::xrange(maskList.size() + 1), maskIndex)
{
  o2::mch::StatusMap statusMap;
  auto mask = maskList[maskIndex];
  statusMap.add(cc, mask);
  for (const auto& status : statusMap) {
    BOOST_CHECK_EQUAL(status.second, mask);
  }
}

BOOST_AUTO_TEST_CASE(ApplyMaskShouldReturnASubsetDependingOnMask)
{
  o2::mch::StatusMap statusMap;
  statusMap.add(cc, o2::mch::StatusMap::kBadPedestal);
  statusMap.add(chid, o2::mch::StatusMap::kRejectList);
  auto badPed = applyMask(statusMap, o2::mch::StatusMap::kBadPedestal);
  auto rejectList = applyMask(statusMap, o2::mch::StatusMap::kRejectList);
  auto all = applyMask(statusMap, o2::mch::StatusMap::kBadPedestal | o2::mch::StatusMap::kRejectList);
  BOOST_CHECK_EQUAL(badPed.size(), cc.size());
  BOOST_CHECK_EQUAL(rejectList.size(), chid.size());
  BOOST_CHECK_EQUAL(all.size(), 4);
}
