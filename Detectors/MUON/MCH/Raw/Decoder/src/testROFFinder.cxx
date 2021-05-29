// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test MCHRaw ROFFinder
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include "MCHRawDecoder/ROFFinder.h"

using namespace o2::mch::raw;

BOOST_AUTO_TEST_SUITE(o2_mch_raw)

BOOST_AUTO_TEST_SUITE(roffinder)

static const int32_t BCINORBIT = 3564;
static const int32_t BCROLLOVER = (1 << 20);
using RawDigit = DataDecoder::RawDigit;
using RawDigitVector = DataDecoder::RawDigitVector;

static RawDigit makeDigit(int ds, uint32_t tfTime, uint32_t orbit)
{
  RawDigit digit;
  digit.digit = o2::mch::Digit(100, 10, 1000, 0x7FFFFFFF, 10);
  digit.info.chip = 1;
  digit.info.ds = ds;
  digit.info.solar = 80;
  digit.info.tfTime = tfTime;
  digit.info.orbit = orbit;
  digit.info.sampaTime = 0;
  digit.info.bunchCrossing = 0;

  return digit;
}

static RawDigitVector makeDigitsVector(uint32_t tfTime1, uint32_t orbit1, uint32_t tfTime2, uint32_t orbit2)
{
  RawDigit digit1 = makeDigit(1, tfTime1, orbit1);
  RawDigit digit2 = makeDigit(2, tfTime2, orbit2);

  RawDigitVector digits;
  digits.push_back(digit1);
  digits.push_back(digit2);
  return digits;
}

//----------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(TwoDigitsInOneROF)
{
  uint32_t tfTime1 = 100;
  uint32_t orbit1 = 1;
  uint32_t tfTime2 = tfTime1;
  uint32_t orbit2 = orbit1;

  auto digits = makeDigitsVector(tfTime1, orbit1, tfTime2, orbit2);

  ROFFinder rofFinder(digits, orbit1);
  rofFinder.process();

  const auto& rofDigits = rofFinder.getOrderedDigits();
  const auto& rofRecords = rofFinder.getROFRecords();

  BOOST_CHECK_EQUAL(rofDigits.size(), 2);
  BOOST_CHECK_EQUAL(rofRecords.size(), 1);

  BOOST_CHECK_EQUAL(rofRecords[0].getFirstIdx(), 0);
  BOOST_CHECK_EQUAL(rofRecords[0].getNEntries(), 2);

  BOOST_CHECK_EQUAL(rofRecords[0].getBCData(), rofFinder.digitTime2IR(digits[0]));

  BOOST_CHECK_EQUAL(rofFinder.isDigitsTimeAligned(), true);
}

//----------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(TwoDigitsInOneROFUnaligned)
{
  uint32_t tfTime1 = 100;
  uint32_t orbit1 = 1;
  uint32_t tfTime2 = tfTime1 + 1;
  uint32_t orbit2 = orbit1;

  auto digits = makeDigitsVector(tfTime1, orbit1, tfTime2, orbit2);

  ROFFinder rofFinder(digits, orbit1);
  rofFinder.process();

  const auto& rofDigits = rofFinder.getOrderedDigits();
  const auto& rofRecords = rofFinder.getROFRecords();

  BOOST_CHECK_EQUAL(rofDigits.size(), 2);
  BOOST_CHECK_EQUAL(rofRecords.size(), 1);

  BOOST_CHECK_EQUAL(rofRecords[0].getFirstIdx(), 0);
  BOOST_CHECK_EQUAL(rofRecords[0].getNEntries(), 2);

  BOOST_CHECK_EQUAL(rofRecords[0].getBCData(), rofFinder.digitTime2IR(digits[0]));

  BOOST_CHECK_EQUAL(rofFinder.isDigitsTimeAligned(), false);
}

//----------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(TwoDigitsInOneROFsConsecutiveOrbits)
{
  uint32_t tfTime1 = 100;
  uint32_t orbit1 = 1;
  uint32_t tfTime2 = tfTime1;
  uint32_t orbit2 = orbit1 + 1;

  auto digits = makeDigitsVector(tfTime1, orbit1, tfTime2, orbit2);

  ROFFinder rofFinder(digits, orbit1);
  rofFinder.process();

  const auto& rofDigits = rofFinder.getOrderedDigits();
  const auto& rofRecords = rofFinder.getROFRecords();

  BOOST_CHECK_EQUAL(rofDigits.size(), 2);
  BOOST_CHECK_EQUAL(rofRecords.size(), 1);

  BOOST_CHECK_EQUAL(rofRecords[0].getFirstIdx(), 0);
  BOOST_CHECK_EQUAL(rofRecords[0].getNEntries(), 2);

  BOOST_CHECK_EQUAL(rofRecords[0].getBCData(), rofFinder.digitTime2IR(digits[0]));
}

//----------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(TwoDigitsInTwoROFs)
{
  uint32_t tfTime1 = 100;
  uint32_t orbit1 = 1;
  uint32_t tfTime2 = tfTime1 + 4;
  uint32_t orbit2 = orbit1;

  auto digits = makeDigitsVector(tfTime1, orbit1, tfTime2, orbit2);

  ROFFinder rofFinder(digits, orbit1);
  rofFinder.process();

  const auto& rofDigits = rofFinder.getOrderedDigits();
  const auto& rofRecords = rofFinder.getROFRecords();

  BOOST_CHECK_EQUAL(rofDigits.size(), 2);
  BOOST_CHECK_EQUAL(rofRecords.size(), 2);

  BOOST_CHECK_EQUAL(rofRecords[0].getFirstIdx(), 0);
  BOOST_CHECK_EQUAL(rofRecords[0].getNEntries(), 1);
  BOOST_CHECK_EQUAL(rofRecords[0].getBCData(), rofFinder.digitTime2IR(digits[0]));

  BOOST_CHECK_EQUAL(rofRecords[1].getFirstIdx(), 1);
  BOOST_CHECK_EQUAL(rofRecords[1].getNEntries(), 1);
  BOOST_CHECK_EQUAL(rofRecords[1].getBCData(), rofFinder.digitTime2IR(digits[1]));

  const auto rofDigit1 = rofFinder.getOrderedDigit(rofRecords[0].getFirstIdx());
  const auto rofDigit2 = rofFinder.getOrderedDigit(rofRecords[1].getFirstIdx());

  BOOST_CHECK_EQUAL(rofDigit1.has_value(), true);
  BOOST_CHECK_EQUAL(rofDigit2.has_value(), true);

  BOOST_CHECK_EQUAL(digits[0], rofDigit1.value());
  BOOST_CHECK_EQUAL(digits[1], rofDigit2.value());

  BOOST_CHECK_EQUAL(rofFinder.isRofTimeMonotonic(), true);
  BOOST_CHECK_EQUAL(rofFinder.isDigitsTimeAligned(), true);
}

//----------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(TwoDigitsInTwoROFsConsecutiveOrbits)
{
  uint32_t tfTime1 = 100;
  uint32_t orbit1 = 1;
  uint32_t tfTime2 = tfTime1 - 4;
  uint32_t orbit2 = orbit1 + 1;

  auto digits = makeDigitsVector(tfTime1, orbit1, tfTime2, orbit2);

  ROFFinder rofFinder(digits, orbit1);
  rofFinder.process();

  const auto& rofDigits = rofFinder.getOrderedDigits();
  const auto& rofRecords = rofFinder.getROFRecords();

  BOOST_CHECK_EQUAL(rofDigits.size(), 2);
  BOOST_CHECK_EQUAL(rofRecords.size(), 2);

  BOOST_CHECK_EQUAL(rofRecords[0].getFirstIdx(), 0);
  BOOST_CHECK_EQUAL(rofRecords[0].getNEntries(), 1);
  BOOST_CHECK_EQUAL(rofRecords[0].getBCData(), rofFinder.digitTime2IR(digits[1]));

  BOOST_CHECK_EQUAL(rofRecords[1].getFirstIdx(), 1);
  BOOST_CHECK_EQUAL(rofRecords[1].getNEntries(), 1);
  BOOST_CHECK_EQUAL(rofRecords[1].getBCData(), rofFinder.digitTime2IR(digits[0]));

  const auto rofDigit1 = rofFinder.getOrderedDigit(rofRecords[0].getFirstIdx());
  const auto rofDigit2 = rofFinder.getOrderedDigit(rofRecords[1].getFirstIdx());

  BOOST_CHECK_EQUAL(rofDigit1.has_value(), true);
  BOOST_CHECK_EQUAL(rofDigit2.has_value(), true);

  BOOST_CHECK_EQUAL(digits[1], rofDigit1.value());
  BOOST_CHECK_EQUAL(digits[0], rofDigit2.value());

  BOOST_CHECK_EQUAL(rofFinder.isRofTimeMonotonic(), true);
  BOOST_CHECK_EQUAL(rofFinder.isDigitsTimeAligned(), true);
}

BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()
