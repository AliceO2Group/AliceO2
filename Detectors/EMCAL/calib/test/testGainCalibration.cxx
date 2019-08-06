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
#include "EMCALCalib/GainCalibrationFactors.h" // local header
#include "EMCALBase/Geometry.h"
#include <iostream>
#include <fstream>

/// \brief Standard tests for Gain Calibration

namespace o2
{
namespace emcal
{

BOOST_AUTO_TEST_CASE(testGainCalibration)
{

  auto geo = o2::emcal::Geometry::GetInstanceFromRunNumber(300000);

  // Single channel test
  //
  // For each channel test set and update, and compare read value
  // against set value
  Int_t sms = geo->GetNumberOfSuperModules();

  for (unsigned short icell = 0; icell < 17664; icell++) {
    GainCalibrationFactors singletest;
    singletest.addGainCalibFactor(icell, 0);
    BOOST_CHECK_EQUAL(singletest.getGainCalibFactors(icell), 0);

    singletest.addGainCalibFactor(icell, 1);
    BOOST_CHECK_EQUAL(singletest.getGainCalibFactors(icell), 1);
  }

  // Pattern test
  //
  // For each channel check the wheather the gain calibration factor is correctly written
  // Test data obtained from LHC18b
  GainCalibrationFactors parameter;

  const char* aliceO2env = std::getenv("O2_ROOT");
  std::string inputDir = " ";
  if (aliceO2env)
    inputDir = aliceO2env;
  inputDir += "/share/Detectors/EMCAL/files/";

  std::string fileNameGainCalib = inputDir + "GainCalibrationFactors_LHC18q.txt";
  std::ifstream fileGainCalib(fileNameGainCalib, std::ifstream::in);
  if (!fileGainCalib.is_open())
    std::cout << "The file GainCalibrationFactors_LHC18q was not opened\n";

  float GainCalibFactors[17664];

  unsigned short icell = 0;
  std::string line;

  // Write to the container
  while (std::getline(fileGainCalib, line)) {
    std::stringstream streamLine(line);
    unsigned short iSM, iCol, iRow;
    float Gain;
    streamLine >> iSM >> iCol >> iRow >> Gain;
    GainCalibFactors[icell] = Gain;
    parameter.addGainCalibFactor(icell, Gain);
    icell++;
  }

  for (unsigned short icell = 0; icell < 17664; icell++)
    BOOST_CHECK_EQUAL(parameter.getGainCalibFactors(icell), GainCalibFactors[icell]);

  // Writing to the other container which will be used for comparison
  GainCalibrationFactors parameterLHC15;

  std::string fileNameGainCalib_LHC15 = inputDir + "GainCalibrationFactors_LHC15.txt";
  std::ifstream fileGainCalib_LHC15(fileNameGainCalib_LHC15, std::ifstream::in);
  if (!fileGainCalib_LHC15.is_open())
    std::cout << "The file GainCalibrationFactors_LHC15 was not opened\n";

  // Write to the container
  icell = 0;
  while (std::getline(fileGainCalib_LHC15, line)) {
    std::stringstream streamLine(line);
    Int_t iSM, iCol, iRow;
    float GainCalib;
    streamLine >> iSM >> iCol >> iRow >> GainCalib;
    parameterLHC15.addGainCalibFactor(icell, GainCalib);
    icell++;
  }

  // Equal
  //
  // - Compare gain calibration factors for LHC18q with itself. The result must be true.
  // - Compare gain calibration factors for LHC18q with the gain calibration for LHC15. The result must be false
  BOOST_CHECK_EQUAL(parameter == parameter, true);
  BOOST_CHECK_EQUAL(parameter == parameterLHC15, false);

  fileGainCalib.close();
  fileGainCalib_LHC15.close();
}
} // namespace emcal
} // namespace o2
