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

#define BOOST_TEST_MODULE Test MCHRaw DigitsTimeComputation
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include "MCHRawDecoder/DataDecoder.h"

using namespace o2::mch::raw;

BOOST_AUTO_TEST_SUITE(o2_mch_raw)

BOOST_AUTO_TEST_SUITE(digitstimecomputation)

static const int32_t BCINORBIT = 3564;
static const int32_t BCROLLOVER = (1 << 20);
using RawDigit = DataDecoder::RawDigit;

BOOST_AUTO_TEST_CASE(TimeDiffSameOrbitNoRollover)
{
  uint32_t orbit1 = 1;
  uint32_t orbit2 = 1;
  uint32_t bc1 = 0;
  uint32_t bc2 = BCINORBIT - 10;
  auto diff = DataDecoder::getDigitTimeHBPackets(orbit1, bc1, orbit2, bc2);
  BOOST_CHECK_EQUAL(diff, bc2);
}

BOOST_AUTO_TEST_CASE(TimeDiffSameOrbitWithRollover)
{
  uint32_t orbit1 = 1;
  uint32_t orbit2 = 1;
  uint32_t bc1 = BCROLLOVER - 10;
  uint32_t bc2 = 10;
  auto diff = DataDecoder::getDigitTimeHBPackets(orbit1, bc1, orbit2, bc2);
  BOOST_CHECK_EQUAL(diff, 20);
}

BOOST_AUTO_TEST_CASE(TimeDiffSameOrbitWithRollover2)
{
  uint32_t orbit1 = 1;
  uint32_t orbit2 = 1;
  uint32_t bc1 = 10;
  uint32_t bc2 = BCROLLOVER - 10;
  auto diff = DataDecoder::getDigitTimeHBPackets(orbit1, bc1, orbit2, bc2);
  BOOST_CHECK_EQUAL(diff, -20);
}

static std::vector<RawDigit> makeDigitsVector(uint32_t sampaTime, uint32_t bunchCrossing, uint32_t orbit)
{
  RawDigit digit;
  digit.digit = o2::mch::Digit(100, 10, 1000, 0x7FFFFFFF, 10);
  digit.info.chip = 1;
  digit.info.ds = 2;
  digit.info.solar = 80;
  digit.info.sampaTime = sampaTime;
  digit.info.bunchCrossing = bunchCrossing;
  digit.info.orbit = orbit;

  std::vector<RawDigit> digits;
  digits.push_back(digit);
  return digits;
}

static std::vector<RawDigit> processDigits(const std::vector<RawDigit>& digits, uint32_t tfOrbit, uint32_t tfBunchCrossing)
{
  SampaChannelHandler channelHandler;
  RdhHandler rdhHandler;

  bool ds2manu = false;
  bool mDebug = true;
  bool mCheckROFs = false;
  bool mDummyROFs = true;
  bool useDummyElecMap = false;
  DataDecoder decoder{channelHandler, rdhHandler, "", "", ds2manu, mDebug, useDummyElecMap};

  decoder.setFirstOrbitInTF(tfOrbit);
  decoder.setDigits(digits);

  for (auto& digit : digits) {
    auto& info = digit.info;
    auto chipId = DataDecoder::getChipId(info.solar, info.ds, info.chip);
    decoder.updateTimeFrameStartRecord(chipId, tfOrbit, tfBunchCrossing);
  }
  decoder.computeDigitsTime();

  auto& digitsOut = decoder.getDigits();
  return digitsOut;
}

BOOST_AUTO_TEST_CASE(ComputeDigitsTime)
{
  uint32_t sampaTime = 10;
  uint32_t bunchCrossing = BCINORBIT - 100;
  uint32_t orbit = 1;

  std::vector<RawDigit> digits = makeDigitsVector(sampaTime, bunchCrossing, orbit);

  uint32_t tfOrbit = 1;
  uint32_t tfBunchCrossing = 0;

  auto digitsOut = processDigits(digits, tfOrbit, tfBunchCrossing);

  int32_t digitTime = static_cast<int32_t>(bunchCrossing) + static_cast<int32_t>(sampaTime * 4) -
                      static_cast<int32_t>(tfBunchCrossing);

  BOOST_CHECK_EQUAL(digitsOut[0].getTime(), digitTime);
}

BOOST_AUTO_TEST_CASE(ComputeDigitsTimeWithRollover)
{
  uint32_t sampaTime = 10;
  uint32_t bunchCrossing = 100;
  uint32_t orbit = 1;

  std::vector<RawDigit> digits = makeDigitsVector(sampaTime, bunchCrossing, orbit);

  uint32_t tfOrbit = 1;
  uint32_t tfBunchCrossing = BCROLLOVER - 100;

  auto digitsOut = processDigits(digits, tfOrbit, tfBunchCrossing);

  int32_t digitTime = static_cast<int32_t>(bunchCrossing) + static_cast<int32_t>(sampaTime * 4) -
                      static_cast<int32_t>(tfBunchCrossing) + BCROLLOVER;

  BOOST_CHECK_EQUAL(digitsOut[0].getTime(), digitTime);
}

BOOST_AUTO_TEST_CASE(ComputeDigitsTimeWithRollover2)
{
  uint32_t sampaTime = 10;
  uint32_t bunchCrossing = BCROLLOVER - 100;
  uint32_t orbit = 1;

  std::vector<RawDigit> digits = makeDigitsVector(sampaTime, bunchCrossing, orbit);

  uint32_t tfOrbit = 1;
  uint32_t tfBunchCrossing = 100;

  auto digitsOut = processDigits(digits, tfOrbit, tfBunchCrossing);

  int32_t digitTime = static_cast<int32_t>(bunchCrossing) + static_cast<int32_t>(sampaTime * 4) -
                      static_cast<int32_t>(tfBunchCrossing) - BCROLLOVER;

  BOOST_CHECK_EQUAL(digitsOut[0].getTime(), digitTime);
}

BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()
