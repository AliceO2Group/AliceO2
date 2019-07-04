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
#include "EMCALCalib/BadChannelMap.h"
#include "EMCALCalib/CalibDB.h"
#include "RStringView.h"
#include <ctime>
#include <iostream>
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

o2::emcal::BadChannelMap* ReadTestBadChannelMap_CCDBApi(const std::string_view ccdbserver = "emcccdb-test.cern.ch")
{
  std::cout << "Using CCDB server " << ccdbserver << std::endl;
  o2::emcal::CalibDB ccdbhandler(ccdbserver);

  o2::emcal::BadChannelMap* read(nullptr);
  try {
    std::map<std::string, std::string> metadata;
    read = ccdbhandler.readBadChannelMap(61493766524, metadata);
  } catch (o2::emcal::CalibDB::ObjectNotFoundException& oe) {
    std::cerr << "CCDB error: " << oe.what() << std::endl;
    return nullptr;
  } catch (o2::emcal::CalibDB::TypeMismatchException& te) {
    std::cout << "CCDB error: " << te.what() << std::endl;
    return nullptr;
  }
  return read;
}