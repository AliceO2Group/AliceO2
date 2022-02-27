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

#include "MCHCalibration/PedestalDigit.h"
#include "MCHCalibration/PedestalData.h"
#include <vector>
#include <random>

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
    digits.emplace_back(721, 23, 13, 1234, 43, samples(13));
    digits.emplace_back(721, 26, 16, 2345, 46, samples(16));
    digits.emplace_back(328, 28, 18, 3456, 48, samples(18));
  }
  std::vector<PedestalDigit> digits;
};

BOOST_FIXTURE_TEST_SUITE(PedestalDataIterator, PedestalDigits)

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
  BOOST_TEST(n == PedestalData::MAXDS * PedestalData::MAXCHANNEL * 2);
}

BOOST_AUTO_TEST_CASE(TestIteratorInCountIfAlgorithm)
{
  PedestalData pd;
  pd.fill(digits);
  auto n = std::count_if(pd.begin(), pd.end(), [](o2::mch::calibration::PedestalChannel& c) {
    return c.mEntries > 0;
  });
  BOOST_TEST(n == 3);
}

BOOST_AUTO_TEST_SUITE_END()
