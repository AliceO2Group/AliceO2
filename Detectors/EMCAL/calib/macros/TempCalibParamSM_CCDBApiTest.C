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
#include "EMCALCalib/TempCalibParamSM.h"
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
void TempCalibParamSM_CCDBApiTest(const std::string_view ccdbserver = "emcccdb-test.cern.ch")
{
  std::cout << "Using CCDB server " << ccdbserver << std::endl;
  o2::ccdb::CcdbApi ccdbhandler;
  ccdbhandler.init(ccdbserver.data());

  // Prepare database object
  o2::emcal::TempCalibParamSM* tcp = new o2::emcal::TempCalibParamSM;

  const char* aliceO2env = std::getenv("O2_ROOT");
  std::string inputDir = " ";
  if (aliceO2env)
    inputDir = aliceO2env;
  inputDir += "/share/Detectors/EMCAL/files/";

  std::string file = inputDir + "TempCalibSM_LHC18k_289166.txt";
  std::ifstream fileTempCalibSM(file, std::ifstream::in);
  if (!fileTempCalibSM.is_open())
    std::cout << "The file TempCalibSM_LHC18k_289166.txt was not opened\n";

  unsigned short iSM = 0;
  std::string line;

  // Write to the container
  while (std::getline(fileTempCalibSM, line)) {
    std::stringstream streamLine(line);
    unsigned short sm;
    float TempCalibSM;
    streamLine >> sm >> TempCalibSM;
    tcp->addTempCalibParamPerSM(sm, TempCalibSM);
    iSM++;
  }

  fileTempCalibSM.close();

  // Set time limits: These are from the start of the run validity range to the end of the run validity range (289166) LHC18k
  auto rangestart = create_timestamp(2018, 7, 8, 7, 22, 0),
       rangeend = create_timestamp(2018, 7, 8, 8, 3, 15);

  // Set time limits: These are from the start of the run validity range to the end of the run validity range (289201) LHC18k
  //auto rangestart = create_timestamp(2018, 7, 9, 1, 1, 17),
  //     rangeend = create_timestamp(2018, 7, 9, 3, 22, 9);

  std::cout << "Using time stamps " << rangestart << " and " << rangeend << std::endl;
  std::map<std::string, std::string> metadata;
  ccdbhandler.storeAsTFile(new o2::TObjectWrapper<o2::emcal::TempCalibParamSM>(tcp), "EMC/TempCalibParamsSM", metadata, rangestart, rangeend);

  // Read temperature calibration coefficients from CCDB, check whether they are the same
  auto rangetest = create_timestamp(2018, 7, 8, 7, 22, 0); //LHC18k 289166
  //auto rangetest = create_timestamp(2018, 7, 9, 1, 1, 17); //LHC18k 289201

  std::cout << "Using read timestamp " << rangetest << "(omitted untill function is implemented server side)" << std::endl;
  o2::emcal::TempCalibParamSM* read(nullptr);
  auto res = ccdbhandler.retrieveFromTFile("EMC/TempCalibParamsSM", metadata, rangetest);
  if (!res) {
    std::cerr << "Failed retrieving object from CCDB" << std::endl;
    return;
  }
  std::cout << "Object found, type " << res->IsA()->GetName() << std::endl;
  auto objw = dynamic_cast<o2::TObjectWrapper<o2::emcal::TempCalibParamSM>*>(res);
  if (!objw) {
    std::cerr << "failed casting to TObjectWrapper" << std::endl;
    return;
  }
  read = objw->getObj();
  if (!read) {
    std::cerr << "No object received from CCDB" << std::endl;
    return;
  }
  std::cout << "Obtained temperature calibration coefficients per SM from CCDB - test for match" << std::endl;
  if (*tcp == *read) {
    std::cout << "temperature calibration coefficients per SM matching - test successfull" << std::endl;
  } else {
    std::cerr << "temperature calibration coefficients per SM don't match" << std::endl;
  }

  TH1F* TempSM = (TH1F*)read->getHistogramRepresentation();

  TCanvas* c1 = new TCanvas("TempSM");
  c1->cd();
  TempSM->Draw();
}
