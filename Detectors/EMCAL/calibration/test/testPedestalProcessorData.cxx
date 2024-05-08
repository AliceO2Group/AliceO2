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
#define BOOST_TEST_MODULE Test_EMCAL_Calibration
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <random>
#include <boost/test/unit_test.hpp>
#include "EMCALCalibration/PedestalProcessorData.h"
#include <TH1.h>
#include <TProfile.h>
#include <TRandom.h>

namespace o2
{

namespace emcal
{

const int NUMBERFECCHANNELS = 17664,
          NUMBERLEDMONCHANNELS = 480;

void compare(const PedestalProcessorData& testobject, const TProfile& refFECHG, const TProfile& refFECLG, const TProfile& refLEDMONHG, const TProfile& refLEDMONLG, bool testCount, bool verbose)
{
  const double PRECISION = FLT_EPSILON;
  const double PRECISIONMEAN = FLT_EPSILON;
  for (std::size_t ichan = 0; ichan < NUMBERFECCHANNELS; ichan++) {
    auto [meanHG, rmsHG] = testobject.getValue(ichan, false, false);
    auto [meanLG, rmsLG] = testobject.getValue(ichan, true, false);
    if (verbose) {
      std::cout << "Channel " << ichan << ", mean HG " << meanHG << ", RMS HG " << std::sqrt(rmsHG) << ", entries HG " << testobject.getEntriesForChannel(ichan, false, false) << " ref(mean " << refFECHG.GetBinContent(ichan + 1) << ", RMS " << refFECHG.GetBinError(ichan + 1) << ", entries " << refFECHG.GetBinEntries(ichan + 1) << ")" << std::endl;
      std::cout << "Channel " << ichan << ", mean LG " << meanLG << ", RMS LG " << std::sqrt(rmsLG) << ", entries LG " << testobject.getEntriesForChannel(ichan, true, false) << " ref(mean " << refFECLG.GetBinContent(ichan + 1) << ", RMS " << refFECLG.GetBinError(ichan + 1) << ", entries " << refFECLG.GetBinEntries(ichan + 1) << ")" << std::endl;
    }
    BOOST_CHECK_LE(std::abs(meanHG - refFECHG.GetBinContent(ichan + 1)), PRECISIONMEAN);
    BOOST_CHECK_LE(std::abs(meanLG - refFECLG.GetBinContent(ichan + 1)), PRECISIONMEAN);
    BOOST_CHECK_LE(std::abs(std::sqrt(rmsHG) - refFECHG.GetBinError(ichan + 1)), PRECISION);
    BOOST_CHECK_LE(std::abs(std::sqrt(rmsLG) - refFECLG.GetBinError(ichan + 1)), PRECISION);
    if (testCount) {
      BOOST_CHECK_EQUAL(testobject.getEntriesForChannel(ichan, false, false), refFECHG.GetBinEntries(ichan + 1));
      BOOST_CHECK_EQUAL(testobject.getEntriesForChannel(ichan, true, false), refFECLG.GetBinEntries(ichan + 1));
    }
  }
  for (std::size_t ichan = 0; ichan < NUMBERLEDMONCHANNELS; ichan++) {
    auto [meanHG, rmsHG] = testobject.getValue(ichan, false, true);
    auto [meanLG, rmsLG] = testobject.getValue(ichan, true, true);
    if (verbose) {
      std::cout << "LEDMON " << ichan << ", mean HG " << meanHG << ", RMS HG " << std::sqrt(rmsHG) << ", entries HG " << testobject.getEntriesForChannel(ichan, false, true) << " ref(mean " << refLEDMONHG.GetBinContent(ichan + 1) << ", RMS " << refLEDMONHG.GetBinError(ichan + 1) << ", entries " << refLEDMONHG.GetBinEntries(ichan + 1) << ")" << std::endl;
      std::cout << "LEDMON " << ichan << ", mean LG " << meanLG << ", RMS LG " << std::sqrt(rmsLG) << ", entries LG " << testobject.getEntriesForChannel(ichan, true, true) << " ref(mean " << refLEDMONLG.GetBinContent(ichan + 1) << ", RMS " << refLEDMONLG.GetBinError(ichan + 1) << ", entries " << refLEDMONLG.GetBinEntries(ichan + 1) << ")" << std::endl;
    }
    BOOST_CHECK_LE(std::abs(meanHG - refLEDMONHG.GetBinContent(ichan + 1)), PRECISIONMEAN);
    BOOST_CHECK_LE(std::abs(meanLG - refLEDMONLG.GetBinContent(ichan + 1)), PRECISIONMEAN);
    BOOST_CHECK_LE(std::abs(std::sqrt(rmsHG) - refLEDMONHG.GetBinError(ichan + 1)), PRECISION);
    BOOST_CHECK_LE(std::abs(std::sqrt(rmsLG) - refLEDMONLG.GetBinError(ichan + 1)), PRECISION);
    if (testCount) {
      BOOST_CHECK_EQUAL(testobject.getEntriesForChannel(ichan, false, true), refLEDMONHG.GetBinEntries(ichan + 1));
      BOOST_CHECK_EQUAL(testobject.getEntriesForChannel(ichan, true, true), refLEDMONLG.GetBinEntries(ichan + 1));
    }
  }
}

BOOST_AUTO_TEST_CASE(testPedestalProcessorData)
{
  // we compare with RMS, so the
  TProfile refFECHG("refFECHG", "Reference FEC HG", NUMBERFECCHANNELS, -0.5, NUMBERFECCHANNELS - 0.5, "s"),
    refFECLG("refFECLG", "Reference FEC LG", NUMBERFECCHANNELS, -0.5, NUMBERFECCHANNELS - 0.5, "s"),
    refLEDMONHG("refLEDMONHG", "Reference LEDMON HG", NUMBERLEDMONCHANNELS, -0.5, NUMBERLEDMONCHANNELS - 0.5, "s"),
    refLEDMONLG("refLEDMONLG", "Reference LEDMON LG", NUMBERLEDMONCHANNELS, -0.5, NUMBERLEDMONCHANNELS - 0.5, "s");
  refFECHG.Sumw2();
  refFECLG.Sumw2();
  refLEDMONHG.Sumw2();
  refLEDMONLG.Sumw2();

  PedestalProcessorData testobject;

  const int NUMBERALTROSAMPLES = 15;
  for (auto iev = 0; iev < 1000; iev++) {
    for (int ichan = 0; ichan < NUMBERFECCHANNELS; ichan++) {
      for (int isample = 0; isample < 15; isample++) {
        // short adc_hg = static_cast<short>(adcGeneratorHG(gen)),
        //       adc_lg = static_cast<short>(adcGeneratorLG(gen));
        auto raw_hg = gRandom->Gaus(40, 6),
             raw_lg = gRandom->Gaus(36, 10);
        unsigned short adc_hg = static_cast<unsigned short>(raw_hg > 0 ? raw_hg : 0),
                       adc_lg = static_cast<unsigned short>(raw_lg > 0 ? raw_lg : 0);
        refFECHG.Fill(ichan, adc_hg);
        refFECLG.Fill(ichan, adc_lg);
        testobject.fillADC(adc_hg, ichan, false, false);
        testobject.fillADC(adc_lg, ichan, true, false);
      }
    }
  }
  for (auto iev = 0; iev < 1000; iev++) {
    for (int ichan = 0; ichan < NUMBERLEDMONCHANNELS; ichan++) {
      for (int isample = 0; isample < 15; isample++) {
        // short adc_hg = static_cast<short>(adcGeneratorLEDMONHG(gen)),
        //      adc_lg = static_cast<short>(adcGeneratorLEDMONLG(gen));
        auto raw_hg = gRandom->Gaus(40, 6),
             raw_lg = gRandom->Gaus(36, 10);
        unsigned short adc_hg = static_cast<unsigned short>(raw_hg > 0 ? raw_hg : 0),
                       adc_lg = static_cast<unsigned short>(raw_lg > 0 ? raw_lg : 0);
        refLEDMONHG.Fill(ichan, adc_hg);
        refLEDMONLG.Fill(ichan, adc_lg);
        testobject.fillADC(adc_hg, ichan, false, true);
        testobject.fillADC(adc_lg, ichan, true, true);
      }
    }
  }

  // Compare channels
  compare(testobject, refFECHG, refFECLG, refLEDMONHG, refLEDMONLG, true, false);

  // Test operator+
  auto testdoubled = testobject + testobject;
  TProfile refFECHGdouble(refFECHG),
    refFECLGdouble(refFECLG),
    refLEDMONHGdouble(refLEDMONHG),
    refLEDMONLGdouble(refLEDMONLG);
  refFECHGdouble.Add(&refFECHG);
  refFECLGdouble.Add(&refFECLG);
  refLEDMONHGdouble.Add(&refLEDMONHG);
  refLEDMONLGdouble.Add(&refLEDMONLG);
  compare(testdoubled, refFECHGdouble, refFECLGdouble, refLEDMONHGdouble, refLEDMONLGdouble, false, false);

  // Test reset function
  auto testreset = testobject;
  testreset.reset();
  for (std::size_t ichan = 0; ichan < NUMBERFECCHANNELS; ichan++) {
    auto [meanHG, rmsHG] = testreset.getValue(ichan, false, false);
    auto [meanLG, rmsLG] = testreset.getValue(ichan, true, false);
    BOOST_CHECK_EQUAL(meanHG, 0.);
    BOOST_CHECK_EQUAL(meanLG, 0.);
    BOOST_CHECK_EQUAL(testreset.getEntriesForChannel(ichan, false, false), 0);
    BOOST_CHECK_EQUAL(testreset.getEntriesForChannel(ichan, true, false), 0);
  }
  for (std::size_t ichan = 0; ichan < NUMBERLEDMONCHANNELS; ichan++) {
    auto [meanHG, rmsHG] = testreset.getValue(ichan, false, true);
    auto [meanLG, rmsLG] = testreset.getValue(ichan, true, true);
    BOOST_CHECK_EQUAL(meanHG, 0.);
    BOOST_CHECK_EQUAL(meanLG, 0.);
    BOOST_CHECK_EQUAL(testreset.getEntriesForChannel(ichan, false, true), 0);
    BOOST_CHECK_EQUAL(testreset.getEntriesForChannel(ichan, true, true), 0);
  }
}

} // namespace emcal

} // namespace o2