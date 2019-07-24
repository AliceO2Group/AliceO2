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
#include "EMCALCalib/TimeCalibrationParams.h" // local header
#include "TFile.h"
#include "TH1S.h"
#include "TCanvas.h"
#include <iostream>
#include <fstream>

/// \brief Standard tests for Time Calibration

namespace o2
{
namespace emcal
{

BOOST_AUTO_TEST_CASE(testTimeCalibration)
{

  // Single channel test
  //
  // For each channel test set and update, and compare read value
  // against set value
  for (unsigned short c = 0; c < 17664; c++) {
    TimeCalibrationParams singletest;
    singletest.addTimeCalibParam(c, 0, 0); //HG
    BOOST_CHECK_EQUAL(singletest.getTimeCalibParam(c, 0), 0);

    singletest.addTimeCalibParam(c, 600, 0); //HG
    BOOST_CHECK_EQUAL(singletest.getTimeCalibParam(c, 0), 600);

    singletest.addTimeCalibParam(c, 0, 1); //LG
    BOOST_CHECK_EQUAL(singletest.getTimeCalibParam(c, 1), 0);

    singletest.addTimeCalibParam(c, 600, 1); //LG
    BOOST_CHECK_EQUAL(singletest.getTimeCalibParam(c, 1), 600);
  }

  // Pattern test
  //
  // For each channel check the wheather the time calibration coeffcient is correctly written
  // Test data obtained from LHC18b
  TimeCalibrationParams parameter;

  const char* aliceO2env = std::getenv("O2_ROOT");
  std::string inputDir = " ";
  if (aliceO2env)
    inputDir = aliceO2env;
  inputDir += "/share/Detectors/EMCAL/files/";

  std::string fileHG = inputDir + "TimeCalibCoeffHG.txt";
  std::ifstream allTimeAvHG(fileHG, std::ifstream::in);
  if (!allTimeAvHG.is_open())
    std::cout << "The file TimeCalibCoeffHG was not opened\n";

  std::string fileLG = inputDir + "TimeCalibCoeffLG.txt";
  std::ifstream allTimeAvLG(fileLG, std::ifstream::in);
  if (!allTimeAvLG.is_open())
    std::cout << "The file TimeCalibCoeffLG was not opened\n";

  unsigned short TimeHG[17664], TimeLG[17664];

  unsigned short icell = 0;
  while (1) {
    allTimeAvHG >> TimeHG[icell];
    if (!allTimeAvHG.good())
      break;
    parameter.addTimeCalibParam(icell, TimeHG[icell], 0); //HG
    icell++;
  }

  icell = 0;
  while (1) {
    allTimeAvLG >> TimeLG[icell];
    if (!allTimeAvLG.good())
      break;
    parameter.addTimeCalibParam(icell, TimeLG[icell], 1); //LG
    icell++;
  }

  for (int icell = 0; icell < 17664; icell++) {
    BOOST_CHECK_EQUAL(parameter.getTimeCalibParam(icell, 0), TimeHG[icell]);

    BOOST_CHECK_EQUAL(parameter.getTimeCalibParam(icell, 1), TimeLG[icell]);
  }

  // Writing to the other container which will be used for comparison
  TimeCalibrationParams parameterLHC17g;

  std::string fileHG_LHC17g = inputDir + "TimeCalibCoeffHG_LHC17g.txt";
  std::ifstream allTimeAvHG_LHC17g(fileHG_LHC17g, std::ifstream::in);
  if (!allTimeAvHG_LHC17g.is_open())
    std::cout << "The file TimeCalibCoeffHG_LHC17g was not opened\n";

  std::string fileLG_LHC17g = inputDir + "TimeCalibCoeffLG_LHC17g.txt";
  std::ifstream allTimeAvLG_LHC17g(fileLG_LHC17g, std::ifstream::in);
  if (!allTimeAvLG_LHC17g.is_open())
    std::cout << "The file TimeCalibCoeffLG_LHC17g was not opened\n";

  unsigned short TimeHG_LHC17g, TimeLG_LHC17g;

  icell = 0;
  while (0) {
    allTimeAvHG_LHC17g >> TimeHG_LHC17g;
    if (!allTimeAvHG_LHC17g.good())
      break;
    parameterLHC17g.addTimeCalibParam(icell, TimeHG_LHC17g, 0); //HG
    icell++;
  }

  icell = 0;
  while (0) {
    allTimeAvLG_LHC17g >> TimeLG_LHC17g;
    if (!allTimeAvLG_LHC17g.good())
      break;
    parameterLHC17g.addTimeCalibParam(icell, TimeLG_LHC17g, 1); //LG
    icell++;
  }

  // Equal
  //
  // - Compare time calibration for LHC18b with itself. The result must be true.
  // - Compare time calibration for LHC18b with time calibration for LHC17g. The result must be false
  BOOST_CHECK_EQUAL(parameter == parameter, true);
  BOOST_CHECK_EQUAL(parameter == parameterLHC17g, false);

  allTimeAvHG.close();
  allTimeAvLG.close();
  allTimeAvHG_LHC17g.close();
  allTimeAvLG_LHC17g.close();
}
} // namespace emcal
} // namespace o2
