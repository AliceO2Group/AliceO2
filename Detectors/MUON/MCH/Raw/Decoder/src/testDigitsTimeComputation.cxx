// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

BOOST_AUTO_TEST_CASE(TimeDiffSameOrbitNoRollover)
{
  uint32_t orbit1 = 1;
  uint32_t orbit2 = 1;
  uint32_t bc1 = 0;
  uint32_t bc2 = BCINORBIT - 10;
  auto diff = DataDecoder::digitsTimeDiff(orbit1, bc1, orbit2, bc2);
  BOOST_CHECK_EQUAL(diff, bc2);
}

BOOST_AUTO_TEST_CASE(TimeDiffSameOrbitWithRollover)
{
  uint32_t orbit1 = 1;
  uint32_t orbit2 = 1;
  uint32_t bc1 = BCROLLOVER - 10;
  uint32_t bc2 = 10;
  auto diff = DataDecoder::digitsTimeDiff(orbit1, bc1, orbit2, bc2);
  BOOST_CHECK_EQUAL(diff, 20);
}

BOOST_AUTO_TEST_CASE(TimeDiffSameOrbitWithRollover2)
{
  uint32_t orbit1 = 1;
  uint32_t orbit2 = 1;
  uint32_t bc1 = 10;
  uint32_t bc2 = BCROLLOVER - 10;
  auto diff = DataDecoder::digitsTimeDiff(orbit1, bc1, orbit2, bc2);
  BOOST_CHECK_EQUAL(diff, -20);
}

static std::vector<o2::mch::Digit> makeDigitsVector()
{
  std::vector<o2::mch::Digit> digits;
  digits.emplace_back(100, 10, 1000, 0x7FFFFFFF, 10);
  return digits;
}

static std::vector<DataDecoder::SampaInfo> makeInfoVector(uint32_t sampaTime, uint32_t bunchCrossing, uint32_t orbit)
{
  DataDecoder::SampaInfo sampaId;
  std::vector<DataDecoder::SampaInfo> sampaInfos;
  sampaId.chip = 1;
  sampaId.ds = 2;
  sampaId.solar = 80;
  sampaId.sampaTime = sampaTime;
  sampaId.bunchCrossing = bunchCrossing;
  sampaId.orbit = orbit;
  sampaInfos.push_back(sampaId);

  return sampaInfos;
}

BOOST_AUTO_TEST_CASE(ComputeDigitsTime)
{
  uint32_t sampaTime = 10;
  uint32_t bunchCrossing = BCINORBIT - 100;
  uint32_t orbit = 1;

  std::vector<o2::mch::Digit> digits = makeDigitsVector();
  std::vector<DataDecoder::SampaInfo> sampaInfos = makeInfoVector(sampaTime, bunchCrossing, orbit);

  uint32_t tfOrbit = 1;
  uint32_t tfBunchCrossing = 0;

  std::unordered_map<uint32_t, std::optional<DataDecoder::SampaTimeFrameStart>> sampaTimeFrameStarts;
  sampaTimeFrameStarts[sampaInfos[0].id].emplace(tfOrbit, tfBunchCrossing);

  DataDecoder::computeDigitsTime_(digits, sampaInfos, sampaTimeFrameStarts, false);

  int32_t digitTime = static_cast<int32_t>(bunchCrossing) + static_cast<int32_t>(sampaTime * 4) -
      static_cast<int32_t>(tfBunchCrossing);

  BOOST_CHECK_EQUAL(digits[0].getTime(), digitTime);
}

BOOST_AUTO_TEST_CASE(ComputeDigitsTimeWithRollover)
{
  uint32_t sampaTime = 10;
  uint32_t bunchCrossing = 100;
  uint32_t orbit = 1;

  std::vector<o2::mch::Digit> digits = makeDigitsVector();
  std::vector<DataDecoder::SampaInfo> sampaInfos = makeInfoVector(sampaTime, bunchCrossing, orbit);

  uint32_t tfOrbit = 1;
  uint32_t tfBunchCrossing = BCROLLOVER - 100;

  std::unordered_map<uint32_t, std::optional<DataDecoder::SampaTimeFrameStart>> sampaTimeFrameStarts;
  sampaTimeFrameStarts[sampaInfos[0].id].emplace(tfOrbit, tfBunchCrossing);

  DataDecoder::computeDigitsTime_(digits, sampaInfos, sampaTimeFrameStarts, false);

  int32_t digitTime = static_cast<int32_t>(bunchCrossing) + static_cast<int32_t>(sampaTime * 4) -
      static_cast<int32_t>(tfBunchCrossing) + BCROLLOVER;

  BOOST_CHECK_EQUAL(digits[0].getTime(), digitTime);
}

BOOST_AUTO_TEST_CASE(ComputeDigitsTimeWithRollover2)
{
  uint32_t sampaTime = 10;
  uint32_t bunchCrossing = BCROLLOVER - 100;
  uint32_t orbit = 1;

  std::vector<o2::mch::Digit> digits = makeDigitsVector();
  std::vector<DataDecoder::SampaInfo> sampaInfos = makeInfoVector(sampaTime, bunchCrossing, orbit);

  uint32_t tfOrbit = 1;
  uint32_t tfBunchCrossing = 100;

  std::unordered_map<uint32_t, std::optional<DataDecoder::SampaTimeFrameStart>> sampaTimeFrameStarts;
  sampaTimeFrameStarts[sampaInfos[0].id].emplace(tfOrbit, tfBunchCrossing);

  DataDecoder::computeDigitsTime_(digits, sampaInfos, sampaTimeFrameStarts, false);

  int32_t digitTime = static_cast<int32_t>(bunchCrossing) + static_cast<int32_t>(sampaTime * 4) -
      static_cast<int32_t>(tfBunchCrossing) - BCROLLOVER;

  BOOST_CHECK_EQUAL(digits[0].getTime(), digitTime);
}

BOOST_AUTO_TEST_CASE(ComputeDigitsTimeNoTFStart)
{
  uint32_t sampaTime = 10;
  uint32_t bunchCrossing = BCINORBIT - 100;
  uint32_t orbit = 1;

  std::vector<o2::mch::Digit> digits = makeDigitsVector();
  std::vector<DataDecoder::SampaInfo> sampaInfos = makeInfoVector(sampaTime, bunchCrossing, orbit);

  uint32_t tfOrbit = 1;
  uint32_t tfBunchCrossing = 0;

  std::unordered_map<uint32_t, std::optional<DataDecoder::SampaTimeFrameStart>> sampaTimeFrameStarts;
  sampaTimeFrameStarts[sampaInfos[0].id + 1].emplace(tfOrbit, tfBunchCrossing);

  DataDecoder::computeDigitsTime_(digits, sampaInfos, sampaTimeFrameStarts, false);

  int32_t digitTime = static_cast<int32_t>(bunchCrossing) + static_cast<int32_t>(sampaTime * 4) -
      static_cast<int32_t>(tfBunchCrossing);

  BOOST_CHECK_EQUAL(digits[0].getTime(), 0xFFFFFFFF);
}

BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()
