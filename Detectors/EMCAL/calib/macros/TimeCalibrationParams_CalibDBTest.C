// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include "EMCALCalib/TimeCalibrationParams.h"
#include "EMCALCalib/CalibDB.h"
#include "RStringView.h"
#include "TH1S.h"
#include "TCanvas.h"
#include <ctime>
#include <fstream>
#endif

/// \brief Converting time into numerical time stamp representation
unsigned long create_timestamp(int year, int month, int day, int hour, int minutes, int seconds)
{
  struct tm timeinfo;
  timeinfo.tm_year = year;
  timeinfo.tm_mon = month;
  timeinfo.tm_mday = day;
  timeinfo.tm_hour = hour;
  timeinfo.tm_min = minutes;
  timeinfo.tm_sec = seconds;

  time_t timeformat = mktime(&timeinfo);
  return static_cast<unsigned long>(timeformat);
}

/// \brief Read-write test
///
/// Writing to EMCAL CCDB server
/// Attention: Might overwrite existing CCDB content - use with care!
void TimeCalibrationParams_CalibDBTest(const std::string_view ccdbserver = "emcccdb-test.cern.ch")
{
  std::cout << "Using CCDB server " << ccdbserver << std::endl;
  o2::emcal::CalibDB ccdbhandler(ccdbserver);

  // Prepare database object
  o2::emcal::TimeCalibrationParams* tcp = new o2::emcal::TimeCalibrationParams;

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

  unsigned short TimeHG, TimeLG;

  unsigned short icell = 0;
  while (1) {
    allTimeAvHG >> TimeHG;
    if (!allTimeAvHG.good())
      break;
    tcp->addTimeCalibParam(icell, TimeHG, 0); //HG
    icell++;
  }

  icell = 0;
  while (1) {
    allTimeAvLG >> TimeLG;
    if (!allTimeAvLG.good())
      break;
    tcp->addTimeCalibParam(icell, TimeLG, 1); //LG
    icell++;
  }

  allTimeAvHG.close();
  allTimeAvLG.close();

  // Set time limits: These are from the start of the run validity range (285009) to the end of the run validity range (285396) LHC18b
  auto rangestart = create_timestamp(2018, 4, 18, 23, 58, 48),
       rangeend = create_timestamp(2018, 4, 27, 1, 5, 52);
  // Set time limits: These are from the start of the run validity range (270882) to the end of the run validity range (271777) LHC17g
  //auto rangestart = create_timestamp(2017, 5, 28, 9, 6, 42),
  //     rangeend = create_timestamp(2017, 6, 12, 22, 39, 50);

  std::cout << "Using time stamps " << rangestart << " and " << rangeend << std::endl;
  std::map<std::string, std::string> metadata;
  ccdbhandler.storeTimeCalibParam(tcp, metadata, rangestart, rangeend);

  // Read time calibration coefficients from CCDB, check whether they are the same
  auto rangetest = create_timestamp(2018, 4, 18, 23, 58, 48); //LHC18b
  //auto rangetest = create_timestamp(2017, 5, 28, 9, 6, 42); //LHC17g

  std::cout << "Using read timestamp " << rangetest << "(omitted untill function is implemented server side)" << std::endl;
  o2::emcal::TimeCalibrationParams* read(nullptr);
  try {
    read = ccdbhandler.readTimeCalibParam(rangetest, metadata);
  } catch (o2::emcal::CalibDB::ObjectNotFoundException& oe) {
    std::cerr << "CCDB error: " << oe.what() << std::endl;
    return;
  } catch (o2::emcal::CalibDB::TypeMismatchException& te) {
    std::cout << "CCDB error: " << te.what() << std::endl;
    return;
  }
  if (!read) {
    std::cerr << "No object received from CCDB" << std::endl;
    return;
  }
  std::cout << "Obtained time calibration coefficients from CCDB - test for match" << std::endl;
  if (*tcp == *read) {
    std::cout << "time calibration coefficients matching - test successfull" << std::endl;
  } else {
    std::cerr << "time calibration coefficients don't match" << std::endl;
  }

  TH1S* HG = (TH1S*)read->getHistogramRepresentation(0);
  TH1S* LG = (TH1S*)read->getHistogramRepresentation(1);

  TCanvas* c1 = new TCanvas("HG");
  c1->cd();
  HG->Draw();

  TCanvas* c2 = new TCanvas("LG");
  c2->cd();
  LG->Draw();
}
