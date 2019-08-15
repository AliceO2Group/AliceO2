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
#include "EMCALCalib/TempCalibParamSM.h" // local header
#include <iostream>
#include <fstream>

/// \brief Standard tests for temperature Calibration per SM

namespace o2
{
namespace emcal
{

BOOST_AUTO_TEST_CASE(testTempCalibrationSM)
{

  // Single channel test
  //
  // For each channel test set and update, and compare read value
  // against set value
  for (unsigned short iSM = 0; iSM < 20; iSM++) {
    TempCalibParamSM singletest;

    singletest.addTempCalibParamPerSM(iSM, 0.);
    BOOST_CHECK_EQUAL(singletest.getTempCalibParamPerSM(iSM), 0.);

    singletest.addTempCalibParamPerSM(iSM, 1.);
    BOOST_CHECK_EQUAL(singletest.getTempCalibParamPerSM(iSM), 1.);
  }

  // Pattern test
  //
  // For each channel check the wheather the temperature calibration coeffcient per SM are correctly written
  // Test data obtained from LHC18k run 289165
  TempCalibParamSM parameter;

  const char* aliceO2env = std::getenv("O2_ROOT");
  std::string inputDir = " ";
  if (aliceO2env)
    inputDir = aliceO2env;
  inputDir += "/share/Detectors/EMCAL/files/";

  std::string file = inputDir + "TempCalibSM_LHC18k_289166.txt";
  std::ifstream fileTempCalibSM(file, std::ifstream::in);
  if (!fileTempCalibSM.is_open())
    std::cout << "The file TempCalibSM_LHC18k_289166.txt was not opened\n";

  float TempCalibSM[20];

  unsigned short iSM = 0;
  std::string line;

  // Write to the container
  while (std::getline(fileTempCalibSM, line)) {
    std::stringstream streamLine(line);
    unsigned short sm;
    streamLine >> sm >> TempCalibSM[iSM];
    parameter.addTempCalibParamPerSM(sm, TempCalibSM[iSM]);
    iSM++;
  }

  for (unsigned short ism = 0; ism < 20; ism++)
    BOOST_CHECK_EQUAL(parameter.getTempCalibParamPerSM(ism), TempCalibSM[ism]);

  // Comparison test

  //comparison file
  TempCalibParamSM parameterLHC18k;

  std::string fileLHC18k = inputDir + "TempCalibSM_LHC18k_289201.txt";
  std::ifstream fileTempCalibSM_LHC18k(fileLHC18k, std::ifstream::in);
  if (!fileTempCalibSM_LHC18k.is_open())
    std::cout << "The file TempCalibSM_LHC18k_289201.txt was not opened\n";

  unsigned char TempCalibSM_LHC18k;

  iSM = 0;
  // Write to the container
  while (std::getline(fileTempCalibSM_LHC18k, line)) {
    std::stringstream streamLine(line);
    unsigned short sm;
    streamLine >> sm >> TempCalibSM_LHC18k;
    parameterLHC18k.addTempCalibParamPerSM(sm, TempCalibSM_LHC18k);
    iSM++;
  }

  // Equal
  //
  // - Compare L1 phase shifts for LHC18k run 289165 with itself. The result must be true.
  // - Compare L1 phase shifts for LHC18k run 289165 with L1 phase shifts for LHC18k run 289166. The result must be false
  BOOST_CHECK_EQUAL(parameter == parameter, true);
  BOOST_CHECK_EQUAL(parameter == parameterLHC18k, false);

  fileTempCalibSM.close();
  fileTempCalibSM_LHC18k.close();
}
} // namespace emcal
} // namespace o2
