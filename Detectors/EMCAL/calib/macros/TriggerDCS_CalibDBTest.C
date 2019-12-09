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
#include "EMCALCalib/TriggerDCS.h"
#include "EMCALCalib/CalibDB.h"
#include "RStringView.h"
#include <ctime>
#include <iostream>
#include <sstream>
#endif

// functions getting test data
void ConfigureTRU(o2::emcal::TriggerTRUDCS& testobject)
{
  testobject.setSELPF(7711);
  testobject.setL0SEL(1);
  testobject.setL0COSM(100);
  testobject.setGTHRL0(132);
  testobject.setMaskReg(1024, 0);
  testobject.setMaskReg(0, 1);
  testobject.setMaskReg(512, 2);
  testobject.setMaskReg(31985, 3);
  testobject.setMaskReg(0, 4);
  testobject.setMaskReg(0, 5);
  testobject.setRLBKSTU(0);
  testobject.setFw(0x21);
}

void ConfigureSTU(o2::emcal::TriggerSTUDCS& testobject)
{
  testobject.setGammaHigh(0, 0);
  testobject.setGammaHigh(1, 0);
  testobject.setGammaHigh(2, 115);
  testobject.setGammaLow(0, 0);
  testobject.setGammaLow(1, 0);
  testobject.setGammaLow(2, 51);
  testobject.setJetHigh(0, 0);
  testobject.setJetHigh(1, 0);
  testobject.setJetHigh(2, 255);
  testobject.setJetLow(0, 0);
  testobject.setJetLow(1, 0);
  testobject.setJetLow(2, 204);
  testobject.setPatchSize(2);
  testobject.setFw(0x2A012);
  testobject.setMedianMode(0);
  testobject.setRegion(0xffffffff);
  for (int i = 0; i < 4; i++)
    testobject.setPHOSScale(i, 0);
}

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
void TriggerDCS_CalibDBTest(const std::string_view ccdbserver = "emcccdb-test.cern.ch")
{
  std::cout << "Using CCDB server " << ccdbserver << std::endl;
  o2::emcal::CalibDB ccdbhandler(ccdbserver);

  // Prepare database object
  o2::emcal::TriggerDCS* dcs = new o2::emcal::TriggerDCS;

  o2::emcal::TriggerSTUDCS stuEMCal;
  ConfigureSTU(stuEMCal);

  o2::emcal::TriggerSTUDCS stuDCal;
  ConfigureSTU(stuDCal);
  stuDCal.setRegion(0xffffff7f);

  o2::emcal::TriggerTRUDCS tru;
  ConfigureTRU(tru);

  dcs->setSTUEMCal(stuEMCal);
  dcs->setSTUDCal(stuDCal);
  dcs->setTRU(tru);

  // Set time limits: These are from the start of the run validity range (252235) to the end of the run validity range (267166) LHC16
  auto rangestart = create_timestamp(2016, 4, 23, 0, 58, 40),
       rangeend = create_timestamp(2016, 12, 5, 6, 3, 19);
  std::cout << "Using time stamps " << rangestart << " and " << rangeend << std::endl;
  std::map<std::string, std::string> metadata;
  ccdbhandler.storeTriggerDCSData(dcs, metadata, rangestart, rangeend);

  // Read Trigger DCS from CCDB, check whether they are the same
  auto rangetest = create_timestamp(2016, 4, 23, 0, 58, 40); //LHC16
  std::cout << "Using read timestamp " << rangetest << std::endl;
  o2::emcal::TriggerDCS* read(nullptr);
  try {
    read = ccdbhandler.readTriggerDCSData(rangetest, metadata);
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
  std::cout << "Obtained Trigger DCS data from CCDB - test for match" << std::endl;
  if (*dcs == *read) {
    std::cout << "Trigger DCS data matching - test successfull" << std::endl;
  } else {
    std::cerr << "Trigger DCS data don't match" << std::endl;
  }
}
