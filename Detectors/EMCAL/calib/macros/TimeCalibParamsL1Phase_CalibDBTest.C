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
#include "EMCALCalib/TimeCalibParamL1Phase.h"
#include "EMCALCalib/CalibDB.h"
#include "RStringView.h"
#include "TH1C.h"
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
void TimeCalibParamsL1Phase_CalibDBTest(const std::string_view ccdbserver = "emcccdb-test.cern.ch")
{
  std::cout << "Using CCDB server " << ccdbserver << std::endl;
  o2::emcal::CalibDB ccdbhandler(ccdbserver);

  // Prepare database object
  o2::emcal::TimeCalibParamL1Phase* tcp = new o2::emcal::TimeCalibParamL1Phase();

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
    tcp->addTimeCalibParamL1Phase(sm, L1Phase[iSM]);
    iSM++;
  }

  fileL1Phase.close();

  // Set time limits: These are from the start of the run validity range to the end of the run validity range (295585) LHC18q
  auto rangestart = create_timestamp(2018, 11, 8, 21, 57, 7),
       rangeend = create_timestamp(2018, 11, 8, 22, 17, 4);
  // Set time limits: These are from the start of the run validity range to the end of the run validity range (296623) LHC18q
  //auto rangestart = create_timestamp(2018, 11, 21, 6, 27, 28),
  //     rangeend = create_timestamp(2018, 11, 21, 7, 34, 53);

  std::cout << "Using time stamps " << rangestart << " and " << rangeend << std::endl;
  std::map<std::string, std::string> metadata;
  ccdbhandler.storeTimeCalibParamL1Phase(tcp, metadata, rangestart, rangeend);

  // Read L1 Phase shifts from CCDB, check whether they are the same
  auto rangetest = create_timestamp(2018, 11, 8, 21, 57, 7); //LHC18q 295585
  //auto rangetest = create_timestamp(2018, 11, 21, 6, 27, 28); //LHC18q 296623

  std::cout << "Using read timestamp " << rangetest << "(omitted untill function is implemented server side)" << std::endl;
  o2::emcal::TimeCalibParamL1Phase* read(nullptr);
  try {
    read = ccdbhandler.readTimeCalibParamL1Phase(rangetest, metadata);
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
  std::cout << "Obtained L1 phase shifts from CCDB - test for match" << std::endl;
  if (*tcp == *read) {
    std::cout << "L1 phase shifts matching - test successfull" << std::endl;
  } else {
    std::cerr << "L1 phase shifts don't match" << std::endl;
  }

  TH1C* L1PhaseShifts = (TH1C*)read->getHistogramRepresentation();

  TCanvas* c1 = new TCanvas("L1PhaseShifts");
  c1->cd();
  L1PhaseShifts->Draw();
}
