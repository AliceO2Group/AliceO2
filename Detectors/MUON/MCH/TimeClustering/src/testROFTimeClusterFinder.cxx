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

#define BOOST_TEST_MODULE Test MCHRaw ROFFinder
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <vector>
#include <boost/test/unit_test.hpp>
#include <fmt/format.h>
#include "MCHTimeClustering/ROFTimeClusterFinder.h"

BOOST_AUTO_TEST_SUITE(o2_mch_raw)

BOOST_AUTO_TEST_SUITE(timeclustering)

using ROFRecord = o2::mch::ROFRecord;
using ROFVector = std::vector<ROFRecord>;

//static constexpr uint32_t sWinSize = 1000 / 25;    // number of BC in 1 us
//static constexpr uint32_t sBinsInOneWindow = 5;     // number of bins in wich the 1 us window is divided for the peak search
//static constexpr uint32_t sBinWidth = sWinSize / sBinsInOneWindow; // 5 bins in one 1 us window

static ROFVector makeROFs(std::vector<int> binEntries, uint32_t winSize, uint32_t nBinsInOneWindow)
{
  uint32_t binWidth = winSize / nBinsInOneWindow;
  uint32_t orbit = 1;
  uint32_t tfTime = 0;
  size_t nBins = binEntries.size();
  ROFVector rofRecords;

  int firstDigitIdx = 0;
  for (int bin = 0; bin < nBins; bin++) {
    // one ROF in each peak search bin, for simplicity
    // skip empty bins
    if (binEntries[bin] == 0) {
      continue;
    }
    o2::InteractionRecord ir(tfTime + bin * binWidth, orbit);
    rofRecords.emplace_back(ir, firstDigitIdx, binEntries[bin], 4);
    firstDigitIdx += binEntries[bin];
  }
  return rofRecords;
}

//----------------------------------------------------------------------------
static ROFVector makeTimeROFs(std::vector<int> binEntries, uint32_t winSize, uint32_t nBinsInOneWindow)
{
  const auto& rofRecords = makeROFs(binEntries, winSize, nBinsInOneWindow);
  const std::vector<o2::mch::Digit> digits;

  o2::mch::ROFTimeClusterFinder rofProcessor(rofRecords, digits, winSize, nBinsInOneWindow, false, 1);
  rofProcessor.process();

  const auto& rofTimeRecords = rofProcessor.getROFRecords();

  return rofTimeRecords;
}

//----------------------------------------------------------------------------
static std::vector<int> getBinIntegral(std::vector<int> binEntries)
{
  std::vector<int> integral = binEntries;
  for (size_t i = 1; i < integral.size(); i++) {
    integral[i] += integral[i - 1];
    std::cout << fmt::format("bin[{}]={}  integral[{}]={}", i, binEntries[i], i, integral[i]) << std::endl;
  }
  return integral;
}

//----------------------------------------------------------------------------
static void checkROF(const ROFRecord rof, std::vector<int> binEntries, uint32_t winSize, uint32_t nBinsInOneWindow, int start, int width)
{
  uint32_t binWidth = winSize / nBinsInOneWindow;
  uint32_t orbit = 1;
  uint32_t tfTime = 0;
  std::vector<int> binIntegral = getBinIntegral(binEntries);

  // checks of indexes and sizes
  int firstIdx = binIntegral[start] - binEntries[start];
  int size = binIntegral[start + width - 1] - firstIdx;
  BOOST_CHECK_EQUAL(rof.getFirstIdx(), firstIdx);
  BOOST_CHECK_EQUAL(rof.getNEntries(), size);

  // checks of interaction records and BC widths
  o2::InteractionRecord irStart(tfTime + start * binWidth, orbit);
  BOOST_CHECK_EQUAL(rof.getBCData(), irStart);
  int bcWidth = (width - 1) * binWidth + 4;
  BOOST_CHECK_EQUAL(rof.getBCWidth(), bcWidth);
}

//----------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(OneTimeCluster)
{
  uint32_t orbit = 1;
  uint32_t tfTime = 0;
  uint32_t winSize = 1000 / 25;
  uint32_t nBins = 5;
  std::vector<int> binEntries = {1, 2, 5, 1, 2};
  std::vector<int> binIntegral = getBinIntegral(binEntries);

  const auto& rofTimeRecords = makeTimeROFs(binEntries, winSize, nBins);
  BOOST_CHECK_EQUAL(rofTimeRecords.size(), 1);

  checkROF(rofTimeRecords[0], binEntries, winSize, nBins, 0, 5);
}

//----------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(OneTimeClusterLargeWin)
{
  uint32_t orbit = 1;
  uint32_t tfTime = 0;
  uint32_t winSize = 1500 / 25;
  uint32_t nBins = 5;
  std::vector<int> binEntries = {1, 2, 5, 1, 2};
  std::vector<int> binIntegral = getBinIntegral(binEntries);

  const auto& rofTimeRecords = makeTimeROFs(binEntries, winSize, nBins);
  BOOST_CHECK_EQUAL(rofTimeRecords.size(), 1);

  checkROF(rofTimeRecords[0], binEntries, winSize, nBins, 0, 5);
}

//----------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(OneTimeCluster7bins)
{
  uint32_t orbit = 1;
  uint32_t tfTime = 0;
  uint32_t winSize = 1200 / 25;
  uint32_t nBins = 7;
  std::vector<int> binEntries = {1, 2, 1, 5, 1, 3, 2};
  std::vector<int> binIntegral = getBinIntegral(binEntries);

  const auto& rofTimeRecords = makeTimeROFs(binEntries, winSize, nBins);
  BOOST_CHECK_EQUAL(rofTimeRecords.size(), 1);

  checkROF(rofTimeRecords[0], binEntries, winSize, nBins, 0, 7);
}

//----------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(OneTimeClusterEmptyStart)
{
  uint32_t orbit = 1;
  uint32_t tfTime = 0;
  uint32_t winSize = 1000 / 25;
  uint32_t nBins = 5;
  std::vector<int> binEntries = {0, 1, 2, 5, 1};
  std::vector<int> binIntegral = getBinIntegral(binEntries);

  const auto& rofTimeRecords = makeTimeROFs(binEntries, winSize, nBins);
  BOOST_CHECK_EQUAL(rofTimeRecords.size(), 1);

  checkROF(rofTimeRecords[0], binEntries, winSize, nBins, 1, 4);
}

//----------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(OneTimeClusterEmptyEnd)
{
  uint32_t orbit = 1;
  uint32_t tfTime = 0;
  uint32_t winSize = 1000 / 25;
  uint32_t nBins = 5;
  std::vector<int> binEntries = {1, 2, 5, 1, 0};
  std::vector<int> binIntegral = getBinIntegral(binEntries);

  const auto& rofTimeRecords = makeTimeROFs(binEntries, winSize, nBins);
  BOOST_CHECK_EQUAL(rofTimeRecords.size(), 1);

  checkROF(rofTimeRecords[0], binEntries, winSize, nBins, 0, 4);
}

//----------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(TwoTimeClusters)
{
  uint32_t orbit = 1;
  uint32_t tfTime = 0;
  uint32_t winSize = 1000 / 25;
  uint32_t nBins = 5;
  std::vector<int> binEntries = {1, 2, 5, 1, 2, 2, 1, 6, 3, 1};
  std::vector<int> binIntegral = getBinIntegral(binEntries);

  const auto& rofTimeRecords = makeTimeROFs(binEntries, winSize, nBins);
  BOOST_CHECK_EQUAL(rofTimeRecords.size(), 2);

  checkROF(rofTimeRecords[0], binEntries, winSize, nBins, 0, 5);
  checkROF(rofTimeRecords[1], binEntries, winSize, nBins, 5, 5);
}

//----------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(TwoTimeClustersWithEmptyGap)
{
  uint32_t orbit = 1;
  uint32_t tfTime = 0;
  uint32_t winSize = 1000 / 25;
  uint32_t nBins = 5;
  std::vector<int> binEntries = {1, 2, 5, 1, 2, 0, 2, 1, 6, 3, 1};
  std::vector<int> binIntegral = getBinIntegral(binEntries);

  const auto& rofTimeRecords = makeTimeROFs(binEntries, winSize, nBins);
  BOOST_CHECK_EQUAL(rofTimeRecords.size(), 2);

  checkROF(rofTimeRecords[0], binEntries, winSize, nBins, 0, 5);
  checkROF(rofTimeRecords[1], binEntries, winSize, nBins, 6, 5);
}

//----------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(TwoTimeClustersWithNonEmptyGap)
{
  uint32_t orbit = 1;
  uint32_t winSize = 1000 / 25;
  uint32_t nBins = 5;
  std::vector<int> binEntries = {1, 2, 5, 1, 2, 4, 5, 2, 1, 6, 3, 1};
  std::vector<int> binIntegral = getBinIntegral(binEntries);

  const auto& rofTimeRecords = makeTimeROFs(binEntries, winSize, nBins);
  BOOST_CHECK_EQUAL(rofTimeRecords.size(), 3);

  checkROF(rofTimeRecords[0], binEntries, winSize, nBins, 0, 5);
  checkROF(rofTimeRecords[1], binEntries, winSize, nBins, 5, 2);
  checkROF(rofTimeRecords[2], binEntries, winSize, nBins, 7, 5);
}

//----------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(OneTimeClusterWithGapAtBeginning)
{
  uint32_t orbit = 1;
  uint32_t winSize = 1000 / 25;
  uint32_t nBins = 5;
  std::vector<int> binEntries = {1, 2, 5, 1, 6, 4, 5};
  std::vector<int> binIntegral = getBinIntegral(binEntries);

  const auto& rofTimeRecords = makeTimeROFs(binEntries, winSize, nBins);
  BOOST_CHECK_EQUAL(rofTimeRecords.size(), 2);

  checkROF(rofTimeRecords[0], binEntries, winSize, nBins, 0, 2);
  checkROF(rofTimeRecords[1], binEntries, winSize, nBins, 2, 5);
}

BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()
