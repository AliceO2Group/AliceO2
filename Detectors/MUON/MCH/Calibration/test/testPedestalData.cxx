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

#define BOOST_TEST_MODULE test mch calibration pedestal data
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include "MCHCalibration/PedestalData.h"
#include "MCHCalibration/PedestalDigit.h"
#include "MCHConstants/DetectionElements.h"
#include "MCHMappingInterface/Segmentation.h"
#include "MCHRawElecMap/Mapper.h"
#include <random>
#include <vector>

std::vector<uint16_t> samples(int n)
{
  std::vector<uint16_t> v(n);
  std::uniform_int_distribution<int> distribution(0, 1024);
  std::mt19937 generator(std::random_device{}());
  std::generate(v.begin(), v.end(), [&distribution, &generator] {
    return distribution(generator);
  });
  return v;
}

using o2::mch::calibration::PedestalData;
using o2::mch::calibration::PedestalDigit;

struct PedestalDigits {
  PedestalDigits()
  {
    // non existing DS : that channel should not appear when iterating on pedestal data
    digits.emplace_back(721, 23, 13, 1234, 43, samples(13));

    // solarId 721 groupId 5 indexId 1
    // S721-J5-DS1 (elinkId 26) [ FEE30-LINK1 ] DE708-DS102
    digits.emplace_back(721, 26, 16, 2345, 46, samples(16));

    // solarId 328 groupId 5 indexId 2
    // S328-J5-DS2 (elinkId 27) [ FEE32-LINK6 ] DE714-DS108 channel 18
    digits.emplace_back(328, 27, 18, 3456, 48, samples(18));
  }
  std::vector<PedestalDigit> digits;
};

BOOST_FIXTURE_TEST_SUITE(PedestalDataIterator, PedestalDigits)

BOOST_AUTO_TEST_CASE(TestIteratorOnCompletePedestalData)
{
  // build a vector of all possible digits
  std::vector<PedestalDigit> allDigits;

  auto det2elec = o2::mch::raw::createDet2ElecMapper<o2::mch::raw::ElectronicMapperGenerated>();

  for (auto deId : o2::mch::constants::deIdsForAllMCH) {
    const auto& seg = o2::mch::mapping::segmentation(deId);
    seg.forEachPad([&](int padID) {
      if (seg.isValid(padID)) {
        auto dsId = seg.padDualSampaId(padID);
        auto ch = seg.padDualSampaChannel(padID);
        o2::mch::raw::DsDetId det(deId, dsId);
        auto elec = det2elec(det).value();
        auto solarId = elec.solarId();
        allDigits.emplace_back(solarId, elec.elinkId(), ch, 42, 42, samples(15));
      }
    });
  }

  PedestalData pd;
  pd.fill(allDigits);

  BOOST_REQUIRE(allDigits.size() == 1063528);
  int n{0};
  for (const auto& ped : pd) {
    ++n;
  }
  BOOST_TEST(n == allDigits.size());
}

BOOST_AUTO_TEST_CASE(TestIteratorEquality)
{
  PedestalData pd;
  pd.fill(digits);
  auto it1 = pd.begin();
  auto it2 = pd.begin();
  BOOST_TEST((it1 == it2));
}

BOOST_AUTO_TEST_CASE(TestIteratorInequality)
{
  PedestalData pd;
  pd.fill(digits);
  auto it1 = pd.begin();
  auto it2 = pd.end();
  BOOST_TEST((it1 != it2));
}

BOOST_AUTO_TEST_CASE(TestIteratorPreIncrementable)
{
  PedestalData pd;
  pd.fill(digits);
  int n{0};
  for (auto rec : pd) {
    n++;
  }
  BOOST_TEST(n == 2768);
  // 2768 = 1856 pads in solar 328 + 721 pads in solar 721
  // Note that solar 328 has 29 dual sampas
  // solar 721 has 15 dual sampas
  // But 2768 < (29+15)*64 (=2816) because not all pads are valid ones.
}

BOOST_AUTO_TEST_CASE(TestIteratorAllReturnedPadAreValidByConstruction)
{
  PedestalData pd;
  pd.fill(digits);
  auto n = std::count_if(pd.begin(), pd.end(), [](o2::mch::calibration::PedestalChannel& c) {
    return c.isValid();
  });
  auto n1 = std::distance(pd.begin(), pd.end());
  BOOST_TEST(n == 2768);
  BOOST_CHECK(n == n1);
}

BOOST_AUTO_TEST_CASE(TestIteratorInCountIfAlgorithm)
{
  PedestalData pd;
  pd.fill(digits);
  auto n = std::count_if(pd.begin(), pd.end(), [](o2::mch::calibration::PedestalChannel& c) {
    return c.mEntries > 0;
  });
  BOOST_TEST(n == 2); // 2 and not 3 because one of the digit is on a pad that is not connected (hence invalid, hence not part of the iteration)
}

BOOST_AUTO_TEST_CASE(IterationOnEmptyDataShouldNotBeAnInfiniteLoop, *boost::unit_test::timeout(10))
{
  PedestalData d;
  auto c = std::distance(d.cbegin(), d.cend());
  BOOST_CHECK_EQUAL(c, 0);
}
BOOST_AUTO_TEST_SUITE_END()
