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
#include "EMCALCalib/GainCalibrationFactors.h"
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
void GainCalibrationFactors_CCDBApiTest(const std::string_view ccdbserver = "emcccdb-test.cern.ch")
{
  std::cout << "Using CCDB server " << ccdbserver << std::endl;
  o2::ccdb::CcdbApi ccdbhandler;
  ccdbhandler.init(ccdbserver.data());

  // Prepare database object
  o2::emcal::GainCalibrationFactors* gcf = new o2::emcal::GainCalibrationFactors;

  const char* aliceO2env = std::getenv("O2_ROOT");
  std::string inputDir = " ";
  if (aliceO2env)
    inputDir = aliceO2env;
  inputDir += "/share/Detectors/EMCAL/files/";

  std::string fileNameGainCalib = inputDir + "GainCalibrationFactors_LHC18q.txt";
  std::ifstream fileGainCalib(fileNameGainCalib, std::ifstream::in);
  if (!fileGainCalib.is_open())
    std::cout << "The file GainCalibrationFactors_LHC18q was not opened\n";

  unsigned short icell = 0;
  std::string line;

  // Write to the container
  while (std::getline(fileGainCalib, line)) {
    std::stringstream streamLine(line);
    unsigned short iSM, iCol, iRow;
    float Gain;
    streamLine >> iSM >> iCol >> iRow >> Gain;
    gcf->addGainCalibFactor(icell, Gain);
    icell++;
  }

  fileGainCalib.close();

  // Set time limits: These are from the start of the run validity range (295275) to the end of the run validity range (297595) LHC18q
  auto rangestart = create_timestamp(2018, 11, 3, 13, 51, 41),
       rangeend = create_timestamp(2018, 12, 2, 13, 39, 47);

  // Set time limits: These are from the start of the run validity range (235716) to the end of the run validity range (295274) LHC15
  //auto rangestart = create_timestamp(2015, 11, 19, 15, 55, 58),
  //    rangeend = create_timestamp(2018, 11, 3, 13, 51, 16);

  std::cout << "Using time stamps " << rangestart << " and " << rangeend << std::endl;
  std::map<std::string, std::string> metadata;
  ccdbhandler.storeAsTFile(new o2::TObjectWrapper<o2::emcal::GainCalibrationFactors>(gcf), "EMC/GainCalibFactors", metadata, rangestart, rangeend);

  // Read gain calibration factors from CCDB, check whether they are the same
  auto rangetest = create_timestamp(2018, 11, 16, 20, 55, 3); //LHC18q run 296273
  //auto rangetest = create_timestamp(2015, 12, 9, 23, 10, 3); //LHC15 run 246583
  std::cout << "Using read timestamp " << rangetest << std::endl;
  o2::emcal::GainCalibrationFactors* read(nullptr);
  auto res = ccdbhandler.retrieveFromTFile("EMC/GainCalibFactors", metadata, rangetest);
  if (!res) {
    std::cerr << "Failed retrieving object from CCDB" << std::endl;
    return;
  }
  std::cout << "Object found, type " << res->IsA()->GetName() << std::endl;
  auto objw = dynamic_cast<o2::TObjectWrapper<o2::emcal::GainCalibrationFactors>*>(res);
  if (!objw) {
    std::cerr << "failed casting to TObjectWrapper" << std::endl;
    return;
  }
  read = objw->getObj();
  if (!read) {
    std::cerr << "No object received from CCDB" << std::endl;
    return;
  }
  std::cout << "Obtained gain calibration factors from CCDB - test for match" << std::endl;
  if (*gcf == *read) {
    std::cout << "Gain calibration factors matching - test successfull" << std::endl;
  } else {
    std::cerr << "Gain calibration factors don't match" << std::endl;
  }

  TH1F* hGainFactors = (TH1F*)read->getHistogramRepresentation();

  TCanvas* c1 = new TCanvas("Gain");
  c1->cd();
  hGainFactors->Draw();
}
