// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test_EMCAL_Calib
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include "EMCALCalib/TempCalibrationParams.h" // local header
#include "TFile.h"
#include "TH1.h"
#include "TCanvas.h"
#include <iostream>
#include <fstream>

/// \brief Standard tests for temperature Calibration

namespace o2
{
namespace emcal
{

BOOST_AUTO_TEST_CASE(testTempCalibration)
{

  // Single channel test
  //
  // For each channel test set and update, and compare read value
  // against set value
  for (unsigned short c = 0; c < 17664; c++) {
    TempCalibrationParams singletest;

    singletest.addTempCalibParam(c, 0., 1.);
    BOOST_CHECK_EQUAL(singletest.getTempCalibParamSlope(c), 0.); //Slope
    BOOST_CHECK_EQUAL(singletest.getTempCalibParamA0(c), 1.);    //A0 param

    singletest.addTempCalibParam(c, -1., 1.5);
    BOOST_CHECK_EQUAL(singletest.getTempCalibParamSlope(c), -1.);
    BOOST_CHECK_EQUAL(singletest.getTempCalibParamA0(c), 1.5);
  }

  // Pattern test
  //
  // For each channel check the wheather the temperature calibration coeffcient (slope and A0 param) are correctly written
  // Test data obtained from LHC18
  TempCalibrationParams parameter;

  const char* aliceO2env = std::getenv("O2_ROOT");
  std::string inputDir = " ";
  if (aliceO2env)
    inputDir = aliceO2env;
  inputDir += "/share/Detectors/EMC/files/";

  std::string file = inputDir + "TempCalibCoeff.txt";
  std::ifstream fileTemp(file, std::ifstream::in);
  if (!fileTemp.is_open())
    std::cout << "The file TempCalibCoeff was not opened\n";

  float Slope[17664], A0[17664];

  unsigned short icell = 0;
  std::string line;

  // Write to the container
  while (std::getline(fileTemp, line)) {
    std::stringstream streamLine(line);
    streamLine >> Slope[icell] >> A0[icell];
    parameter.addTempCalibParam(icell, Slope[icell], A0[icell]);
    icell++;
  }

  for (unsigned short icell = 0; icell < 17664; icell++) {
    BOOST_CHECK_EQUAL(parameter.getTempCalibParamSlope(icell), Slope[icell]);

    BOOST_CHECK_EQUAL(parameter.getTempCalibParamA0(icell), A0[icell]);
  }

  // Comparison test

  //comparison file
  TempCalibrationParams parameterLHC17;

  std::string fileLHC17 = inputDir + "TempCalibCoeff_LHC17.txt";
  std::ifstream fileTempLHC17(fileLHC17, std::ifstream::in);
  if (!fileTempLHC17.is_open())
    std::cout << "The file TempCalibCoeff_LHC17 was not opened\n";

  float Slope_LHC17, A0_LHC17;

  icell = 0;
  // Write to the container
  while (std::getline(fileTempLHC17, line)) {
    std::stringstream streamLine(line);
    streamLine >> Slope_LHC17 >> A0_LHC17;
    parameterLHC17.addTempCalibParam(icell, Slope_LHC17, A0_LHC17);
    icell++;
  }

  // Equal
  //
  // - Compare temperature calibration for LHC18 with itself. The result must be true.
  // - Compare temperature calibration for LHC18 with time calibration for LHC17. The result must be false
  BOOST_CHECK_EQUAL(parameter == parameter, true);
  BOOST_CHECK_EQUAL(parameter == parameterLHC17, false);

  fileTemp.close();
  fileTempLHC17.close();
}
} // namespace emcal
} // namespace o2
