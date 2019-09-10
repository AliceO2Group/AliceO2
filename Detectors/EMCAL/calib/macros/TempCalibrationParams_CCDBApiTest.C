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
#include "CCDB/CcdbApi.h"
#include "CCDB/TObjectWrapper.h"
#include "EMCALCalib/TempCalibrationParams.h"
#include "RStringView.h"
#include "TH1F.h"
#include "TCanvas.h"
#include <ctime>
#include <string>
#include <sstream>
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
void TempCalibrationParams_CCDBApiTest(const std::string_view ccdbserver = "emcccdb-test.cern.ch")
{
  std::cout << "Using CCDB server " << ccdbserver << std::endl;
  o2::ccdb::CcdbApi ccdbhandler;
  ccdbhandler.init(ccdbserver.data());

  // Prepare database object
  o2::emcal::TempCalibrationParams* tcp = new o2::emcal::TempCalibrationParams;

  const char* aliceO2env = std::getenv("O2_ROOT");
  std::string inputDir = " ";
  if (aliceO2env)
    inputDir = aliceO2env;
  inputDir += "/share/Detectors/EMCAL/files/";

  std::string file = inputDir + "TempCalibCoeff.txt";
  std::ifstream fileTemp(file, std::ifstream::in);
  if (!fileTemp.is_open())
    std::cout << "The file TempCalibCoeff was not opened\n";

  float Slope, A0;

  unsigned short icell = 0;
  std::string line;

  // Write to the container
  while (std::getline(fileTemp, line)) {
    std::stringstream streamLine(line);
    streamLine >> Slope >> A0;
    tcp->addTempCalibParam(icell, Slope, A0);
    icell++;
  }

  fileTemp.close();

  // Set time limits: These are from the start of the run validity range (285009) to the end of the run validity range (297595) LHC18
  auto rangestart = create_timestamp(2018, 4, 18, 23, 58, 48),
       rangeend = create_timestamp(2018, 12, 2, 13, 39, 47);

  // Set time limits: These are from the start of the run validity range (270581) to the end of the run validity range (282704) LHC17
  //auto rangestart = create_timestamp(2017, 5, 23, 23, 7, 44),
  //     rangeend = create_timestamp(2017, 11, 26, 11, 32, 38);

  // Set time limits: These are from the start of the run validity range (252235) to the end of the run validity range (267166) LHC16
  //auto rangestart = create_timestamp(2016, 4, 23, 0, 58, 40),
  //    rangeend = create_timestamp(2016, 12, 5, 6, 3, 19);

  // Set time limits: These are from the start of the run validity range (235716) to the end of the run validity range (246994) LHC15
  //auto rangestart = create_timestamp(2015, 9, 12, 5, 7, 8),
  //    rangeend = create_timestamp(2015, 12, 13, 11, 46, 20);

  std::cout << "Using time stamps " << rangestart << " and " << rangeend << std::endl;
  std::map<std::string, std::string> metadata;
  ccdbhandler.storeAsTFile(new o2::TObjectWrapper<o2::emcal::TempCalibrationParams>(tcp), "EMC/TempCalibParams", metadata, rangestart, rangeend);

  // Read temperature calibration coefficients from CCDB, check whether they are the same
  auto rangetest = create_timestamp(2018, 4, 27, 1, 5, 52); //LHC18 run 285396
  //auto rangetest = create_timestamp(2017, 6, 12, 22, 39, 50); //LHC17 run 271777
  //auto rangetest = create_timestamp(2016, 7, 9, 2, 0, 8); //LHC16 run 257735
  //auto rangetest = create_timestamp(2015, 12, 9, 23, 10, 3); //LHC15 run 246583
  std::cout << "Using read timestamp " << rangetest << std::endl;
  o2::emcal::TempCalibrationParams* read(nullptr);
  auto res = ccdbhandler.retrieveFromTFile("EMC/TempCalibParams", metadata, rangetest);
  if (!res) {
    std::cerr << "Failed retrieving object from CCDB" << std::endl;
    return;
  }
  std::cout << "Object found, type " << res->IsA()->GetName() << std::endl;
  auto objw = dynamic_cast<o2::TObjectWrapper<o2::emcal::TempCalibrationParams>*>(res);
  if (!objw) {
    std::cerr << "failed casting to TObjectWrapper" << std::endl;
    return;
  }
  read = objw->getObj();
  if (!read) {
    std::cerr << "No object received from CCDB" << std::endl;
    return;
  }
  std::cout << "Obtained temperature calibration coefficients from CCDB - test for match" << std::endl;
  if (*tcp == *read) {
    std::cout << "temperature calibration coefficients matching - test successfull" << std::endl;
  } else {
    std::cerr << "temperature calibration coefficients don't match" << std::endl;
  }

  TH1F* hSlope = (TH1F*)read->getHistogramRepresentationSlope();
  TH1F* hA0 = (TH1F*)read->getHistogramRepresentationA0();

  TCanvas* c1 = new TCanvas("Slope");
  c1->cd();
  hSlope->Draw();

  TCanvas* c2 = new TCanvas("A0");
  c2->cd();
  hA0->Draw();
}
