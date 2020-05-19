// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test MCHRaw GBTEncoder
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include "BareElinkEncoder.h"
#include "UserLogicElinkEncoder.h"
#include "BareElinkEncoderMerger.h"
#include "UserLogicElinkEncoderMerger.h"
#include <array>
#include <array>
#include <fmt/printf.h>
#include <boost/mpl/list.hpp>
#include "GBTEncoder.h"

using namespace o2::mch::raw;

BOOST_AUTO_TEST_SUITE(o2_mch_raw)

BOOST_AUTO_TEST_SUITE(gbtencoder)

typedef boost::mpl::list<BareFormat, UserLogicFormat> testTypes;

template <typename FORMAT, typename CHARGESUM>
float expectedSize();

template <>
float expectedSize<BareFormat, ChargeSumMode>()
{
  return 4 * 640;
}

template <>
float expectedSize<UserLogicFormat, ChargeSumMode>()
{
  return 96;
}

BOOST_AUTO_TEST_CASE_TEMPLATE(GBTEncoderAddFewChannels, T, testTypes)
{
  GBTEncoder<T, ChargeSumMode>::forceNoPhase = true;
  GBTEncoder<T, ChargeSumMode> enc(0);
  uint32_t bx(0);
  uint16_t ts(0);
  int elinkGroupId = 0;
  int elinkIndexInGroup = 0;
  enc.addChannelData(elinkGroupId, elinkIndexInGroup, 0, {SampaCluster(ts, 0, 10)});
  enc.addChannelData(elinkGroupId, elinkIndexInGroup, 31, {SampaCluster(ts, 0, 160)});
  elinkIndexInGroup = 3;
  enc.addChannelData(elinkGroupId, elinkIndexInGroup, 3, {SampaCluster(ts, 0, 13)});
  enc.addChannelData(elinkGroupId, elinkIndexInGroup, 13, {SampaCluster(ts, 0, 133)});
  enc.addChannelData(elinkGroupId, elinkIndexInGroup, 23, {SampaCluster(ts, 0, 163)});
  BOOST_CHECK_THROW(enc.addChannelData(8, 0, 0, {SampaCluster(ts, 0, 10)}), std::invalid_argument);
  std::vector<std::byte> buffer;
  enc.moveToBuffer(buffer);
  float e = expectedSize<T, ChargeSumMode>();
  BOOST_CHECK_EQUAL(buffer.size(), e);
}

template <typename FORMAT, typename CHARGESUM>
float expectedMaxSize();

template <>
float expectedMaxSize<BareFormat, ChargeSumMode>()
{
  return 4 * 11620;
}

template <>
float expectedMaxSize<UserLogicFormat, ChargeSumMode>()
{
  return 944;
}

BOOST_AUTO_TEST_CASE_TEMPLATE(GBTEncoderAdd64Channels, T, testTypes)
{
  GBTEncoder<T, ChargeSumMode>::forceNoPhase = true;
  GBTEncoder<T, ChargeSumMode> enc(0);
  std::vector<std::byte> buffer;
  enc.moveToBuffer(buffer);
  uint32_t bx(0);
  uint16_t ts(0);
  int elinkGroupId = 0;
  for (int i = 0; i < 64; i++) {
    enc.addChannelData(elinkGroupId, 0, i, {SampaCluster(ts, 0, i * 10)});
  }
  enc.moveToBuffer(buffer);
  //  impl::dumpBuffer<T>(buffer);
  float e = expectedMaxSize<T, ChargeSumMode>();
  BOOST_CHECK_EQUAL(buffer.size(), e);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(GBTEncoderMoveToBufferClearsTheInternalBuffer, T, testTypes)
{
  GBTEncoder<T, ChargeSumMode> enc(0);
  enc.addChannelData(0, 0, 0, {SampaCluster(0, 0, 10)});
  std::vector<std::byte> buffer;
  size_t n = enc.moveToBuffer(buffer);
  BOOST_CHECK_GE(n, 0);
  n = enc.moveToBuffer(buffer);
  BOOST_CHECK_EQUAL(n, 0);
}

BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()
