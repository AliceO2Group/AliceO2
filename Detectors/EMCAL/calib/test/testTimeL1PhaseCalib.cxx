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
#include "EMCALCalib/TimeCalibParamL1Phase.h" // local header
#include <iostream>
#include <fstream>

/// \brief Standard tests for temperature Calibration

namespace o2
{
namespace emcal
{

BOOST_AUTO_TEST_CASE(testTimeL1PhaseCalib)
{

  // Single channel test
  //
  // For each channel test set and update, and compare read value
  // against set value
  for (unsigned short iSM = 0; iSM < 20; iSM++) {
    TimeCalibParamL1Phase singletest;

    singletest.addTimeCalibParamL1Phase(iSM, 0);
    BOOST_CHECK_EQUAL(singletest.getTimeCalibParamL1Phase(iSM), 0);

    singletest.addTimeCalibParamL1Phase(iSM, 2);
    BOOST_CHECK_EQUAL(singletest.getTimeCalibParamL1Phase(iSM), 2);
  }

  // Pattern test
  //
  // For each channel check the wheather the L1 phase shifts are correctly written
  // Test data obtained from LHC18q run 295585
  TimeCalibParamL1Phase parameter;

  const char* aliceO2env = std::getenv("O2_ROOT");
  std::string inputDir = " ";
  if (aliceO2env)
    inputDir = aliceO2env;
  inputDir += "/share/Detectors/EMCAL/files/";

  std::string file = inputDir + "TimeL1Phase_LHC18q_295585.txt";
  std::ifstream fileL1Phase(file, std::ifstream::in);
  if (!fileL1Phase.is_open())
    std::cout << "The file TimeL1Phase_LHC18q_295585 was not opened\n";

  unsigned char L1Phase[20];

  unsigned short iSM = 0;
  std::string line;

  // Write to the container
  while (std::getline(fileL1Phase, line)) {
    std::stringstream streamLine(line);
    unsigned short sm;
    streamLine >> sm >> L1Phase[iSM];
    parameter.addTimeCalibParamL1Phase(sm, L1Phase[iSM]);
    iSM++;
  }

  for (unsigned short iSM = 0; iSM < 20; iSM++)
    BOOST_CHECK_EQUAL(parameter.getTimeCalibParamL1Phase(iSM), L1Phase[iSM]);

  // Comparison test

  //comparison file
  TimeCalibParamL1Phase parameterLHC18q;

  std::string fileLHC18q = inputDir + "TimeL1Phase_LHC18q_296623.txt";
  std::ifstream fileL1PhaseLHC18q(fileLHC18q, std::ifstream::in);
  if (!fileL1PhaseLHC18q.is_open())
    std::cout << "The file TimeL1Phase_LHC18q_296623 was not opened\n";

  unsigned char L1Phase_LHC18q;

  iSM = 0;
  // Write to the container
  while (std::getline(fileL1PhaseLHC18q, line)) {
    std::stringstream streamLine(line);
    unsigned short sm;
    streamLine >> sm >> L1Phase_LHC18q;
    parameterLHC18q.addTimeCalibParamL1Phase(sm, L1Phase_LHC18q);
    iSM++;
  }

  // Equal
  //
  // - Compare L1 phase shifts for LHC18q run 295585 with itself. The result must be true.
  // - Compare L1 phase shifts for LHC18q run 295585 with L1 phase shifts for LHC18q run 296623. The result must be false
  BOOST_CHECK_EQUAL(parameter == parameter, true);
  BOOST_CHECK_EQUAL(parameter == parameterLHC18q, false);

  fileL1Phase.close();
  fileL1PhaseLHC18q.close();
}
} // namespace emcal
} // namespace o2
