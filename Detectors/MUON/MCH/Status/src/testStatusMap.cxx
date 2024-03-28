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
#include "MCHGlobalMapping/DsIndex.h"
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

o2::mch::DsIndex ds = 246; // 56 pads, contains chid[1]

uint16_t de = 1025; // 6976 pads, contains cc[2]

uint32_t badMask = 1 << 3;

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
  BOOST_CHECK_THROW(statusMap.add(chid, badMask), std::runtime_error);
}

BOOST_AUTO_TEST_CASE(AddChannelCodeWithInvalidMaskShouldThrow)
{
  o2::mch::StatusMap statusMap;
  BOOST_CHECK_THROW(statusMap.add(cc, badMask), std::runtime_error);
}

BOOST_AUTO_TEST_CASE(AddDSWithInvalidMaskShouldThrow)
{
  o2::mch::StatusMap statusMap;
  BOOST_CHECK_THROW(statusMap.addDS(0, badMask), std::runtime_error);
}

BOOST_AUTO_TEST_CASE(AddDEWithInvalidMaskShouldThrow)
{
  o2::mch::StatusMap statusMap;
  BOOST_CHECK_THROW(statusMap.addDE(100, badMask), std::runtime_error);
}

std::array<uint32_t, 5> maskList = {
  o2::mch::StatusMap::kBadPedestal,
  o2::mch::StatusMap::kRejectList,
  o2::mch::StatusMap::kBadHV,
  o2::mch::StatusMap::kBadPedestal | o2::mch::StatusMap::kBadHV,
  o2::mch::StatusMap::kRejectList | o2::mch::StatusMap::kBadPedestal | o2::mch::StatusMap::kBadHV};

BOOST_DATA_TEST_CASE(CheckAddedChannelsGetTheRightMask, bdata::xrange(maskList.size()), maskIndex)
{
  o2::mch::StatusMap statusMap;
  auto mask = maskList[maskIndex];
  statusMap.add(cc, mask);
  for (const auto& status : statusMap) {
    BOOST_CHECK_EQUAL(status.second, mask);
  }
}

BOOST_AUTO_TEST_CASE(CheckChannelStatusCombination)
{
  auto size = [](const o2::mch::StatusMap& statusMap) {
    int n = 0;
    for (const auto& status : statusMap) {
      ++n;
    }
    return n;
  };
  o2::mch::StatusMap statusMap;
  statusMap.add(cc, o2::mch::StatusMap::kBadPedestal);
  BOOST_CHECK_EQUAL(size(statusMap), 4);
  statusMap.add(chid, o2::mch::StatusMap::kRejectList);
  BOOST_CHECK_EQUAL(size(statusMap), 4);
  statusMap.addDS(ds, o2::mch::StatusMap::kBadHV);
  BOOST_CHECK_EQUAL(size(statusMap), 59);
  statusMap.addDE(de, o2::mch::StatusMap::kBadHV);
  BOOST_CHECK_EQUAL(size(statusMap), 7034);
  BOOST_CHECK_EQUAL(statusMap.status(cc[0]), o2::mch::StatusMap::kBadPedestal | o2::mch::StatusMap::kRejectList);
  BOOST_CHECK_EQUAL(statusMap.status(cc[1]), o2::mch::StatusMap::kBadPedestal | o2::mch::StatusMap::kRejectList | o2::mch::StatusMap::kBadHV);
  BOOST_CHECK_EQUAL(statusMap.status(cc[2]), o2::mch::StatusMap::kBadPedestal | o2::mch::StatusMap::kBadHV);
  BOOST_CHECK_EQUAL(statusMap.status(cc[3]), o2::mch::StatusMap::kBadPedestal);
}

BOOST_AUTO_TEST_CASE(ApplyMaskShouldReturnASubsetDependingOnMask)
{
  auto size = [](const std::map<int, std::vector<int>>& badChannels) {
    int n = 0;
    for (const auto& channels : badChannels) {
      n += channels.second.size();
    }
    return n;
  };
  o2::mch::StatusMap statusMap;
  statusMap.add(cc, o2::mch::StatusMap::kBadPedestal);
  statusMap.add(chid, o2::mch::StatusMap::kRejectList);
  statusMap.addDS(ds, o2::mch::StatusMap::kBadHV);
  statusMap.addDE(de, o2::mch::StatusMap::kBadHV);
  auto badPed = applyMask(statusMap, o2::mch::StatusMap::kBadPedestal);
  auto rejectList = applyMask(statusMap, o2::mch::StatusMap::kRejectList);
  auto badHV = applyMask(statusMap, o2::mch::StatusMap::kBadHV);
  auto badHVOrRL = applyMask(statusMap, o2::mch::StatusMap::kBadHV | o2::mch::StatusMap::kRejectList);
  auto any = applyMask(statusMap, o2::mch::StatusMap::kBadPedestal | o2::mch::StatusMap::kRejectList | o2::mch::StatusMap::kBadHV);
  BOOST_CHECK_EQUAL(badPed.size(), 4);
  BOOST_CHECK_EQUAL(size(badPed), 4);
  BOOST_CHECK_EQUAL(badPed[1025][0], 8);
  BOOST_CHECK_EQUAL(rejectList.size(), 2);
  BOOST_CHECK_EQUAL(size(rejectList), 2);
  BOOST_CHECK_EQUAL(rejectList[100][0], 15665);
  BOOST_CHECK_EQUAL(badHV.size(), 2);
  BOOST_CHECK_EQUAL(size(badHV), 7032);
  BOOST_CHECK_EQUAL(badHV[1025].size(), 6976);
  BOOST_CHECK_EQUAL(badHVOrRL.size(), 3);
  BOOST_CHECK_EQUAL(size(badHVOrRL), 7033);
  BOOST_CHECK_EQUAL(badHVOrRL[515].size(), 0);
  BOOST_CHECK_EQUAL(any.size(), 4);
  BOOST_CHECK_EQUAL(size(any), 7034);
  BOOST_CHECK_EQUAL(any[302][0], 20117);
}
