// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file testSAMPAProcessing.cxx
/// \brief This task tests the SAMPAProcessing module of the TPC digitization
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de

#define BOOST_TEST_MODULE Test TPC SAMPAProcessing
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include "TPCSimulation/SAMPAProcessing.h"
#include "TPCBase/CDBInterface.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include "FairLogger.h"

namespace o2
{
namespace tpc
{

/// \brief Test of the conversion to ADC value
BOOST_AUTO_TEST_CASE(SAMPA_ADC_test)
{
  auto& cdb = CDBInterface::instance();
  cdb.setUseDefaults();
  auto& eleParam = ParameterElectronics::Instance();
  const SAMPAProcessing& sampa = SAMPAProcessing::instance();
  BOOST_CHECK_CLOSE(sampa.getADCvalue(1000.f),
                    1000.f * eleParam.ElectronCharge * 1.e15 * eleParam.ChipGain *
                      eleParam.ADCsaturation / eleParam.ADCdynamicRange,
                    1E-5);
}

/// \brief Test of the saturation effect
/// read in the file on which the spline for the SAMPA saturation are based and compare the final spline
/// to the contents of the file
BOOST_AUTO_TEST_CASE(SAMPA_saturation_test)
{
  auto& cdb = CDBInterface::instance();
  cdb.setUseDefaults();
  const SAMPAProcessing& sampa = SAMPAProcessing::instance();

  std::vector<float> ADCin = {{1.f, 50.f, 100.f, 1000.f, 1023.f, 1024.f, 2000.f, 10000.f}};
  std::vector<float> ADCout = {{1.f, 50.f, 100.f, 1000.f, 1023.f, 1023.f, 1023.f, 1023.f}};

  for (size_t i = 0; i < ADCin.size(); ++i) {
    BOOST_CHECK(sampa.getADCSaturation(ADCin[i]) == ADCout[i]);
  }
}

/// \brief Test of the Gamma4 function
BOOST_AUTO_TEST_CASE(SAMPA_Gamma4_test)
{
  auto& cdb = CDBInterface::instance();
  cdb.setUseDefaults();
  auto& eleParam = ParameterElectronics::Instance();
  const SAMPAProcessing& sampa = SAMPAProcessing::instance();
  float timeInit[4] = {0.1, 3.3, 1.f, 90.5};
  float startTimeInit[4] = {0.f, 3.f, 0.f, 90.f};
  float ADCinit[4] = {1.f, 50.f, 100.f, 100.f};
  Vc::float_v time;
  Vc::float_v startTime;
  Vc::float_v ADC;
  for (int i = 0; i < 4; ++i) {
    time[i] = timeInit[i];
    startTime[i] = startTimeInit[i];
    ADC[i] = ADCinit[i];
  }
  /// @todo here one should consider to load an exemplary wave form of a real SAMPA pulse (once available)
  /// and compare to the outcome of the Digitization
  Vc::float_v adcValue = 55.f * ADC * Vc::exp(-4.f * (time - startTime) / eleParam.PeakingTime) *
                         (time - startTime) / eleParam.PeakingTime * (time - startTime) /
                         eleParam.PeakingTime * (time - startTime) / eleParam.PeakingTime *
                         (time - startTime) / eleParam.PeakingTime;
  Vc::float_v signal = sampa.getGamma4(time, startTime, ADC);
  for (int i = 0; i < 4; ++i) {
    float currentSignal = signal[i];
    float currentADC = adcValue[i];
    BOOST_CHECK_CLOSE(currentSignal, currentADC, 1E-3);
  }
}

/// \brief Test of the conversion functions
BOOST_AUTO_TEST_CASE(SAMPA_Conversion_test)
{
  auto& cdb = CDBInterface::instance();
  cdb.setUseDefaults();
  auto& eleParam = ParameterElectronics::Instance();
  auto& detParam = ParameterDetector::Instance();
  static SAMPAProcessing& sampa = SAMPAProcessing::instance();
  BOOST_CHECK(sampa.getTimeBin(detParam.TPClength) == 0);
  BOOST_CHECK_CLOSE(sampa.getZfromTimeBin(0, Side::A), detParam.TPClength, 1E-6);
}
} // namespace tpc
} // namespace o2
