// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

// macro to read the TOF DCS information from CCDB
// default ts is very big: Saturday, November 20, 2286 5:46:39 PM

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include "CCDB/CcdbApi.h"
#include "DetectorsDCS/DataPointIdentifier.h"
#include "TOFCalibration/TOFDCSProcessor.h"

#include <string>
#include <unordered_map>
#include <chrono>
#include <bitset>
#endif

void readTOFDCSentries(long ts = 9999999999000, const char* ccdb = "http://localhost:8080")
{

  o2::ccdb::CcdbApi api;
  api.init(ccdb); // or http://ccdb-test.cern.ch:8080
  std::map<std::string, std::string> metadata;
  if (ts == 9999999999000) {
    ts = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
  }

  std::unordered_map<o2::dcs::DataPointIdentifier, o2::tof::TOFDCSinfo>* m = api.retrieveFromTFileAny<std::unordered_map<o2::dcs::DataPointIdentifier, o2::tof::TOFDCSinfo>>("TOF/DCSDPs", metadata, ts);
  std::cout << "size of map = " << m->size() << std::endl;

  for (auto& i : *m) {
    std::cout << "id = " << i.first << std::endl;
    i.second.print();
  }

  std::bitset<o2::tof::Geo::NCHANNELS>* feac = api.retrieveFromTFileAny<std::bitset<o2::tof::Geo::NCHANNELS>>("TOF/LVStatus", metadata, ts);
  //std::cout << "LV info (FEAC): " << feac->to_string() << std::endl;
  std::cout << "LV info (FEAC): number of channels that are ON = " << feac->count() << std::endl;
  for (int ich = 0; ich < o2::tof::Geo::NCHANNELS; ++ich) {
    if (feac->test(ich) != 0) {
      std::cout << "LV for channel " << ich << " is ON" << std::endl;
    }
  }

  std::bitset<o2::tof::Geo::NCHANNELS>* hv = api.retrieveFromTFileAny<std::bitset<o2::tof::Geo::NCHANNELS>>("TOF/HVStatus", metadata, ts);
  //std::cout << "HV info       : " << hv->to_string() << std::endl;
  std::cout << "HV info       : number of channels that are ON = " << hv->count() << std::endl;
  for (int ich = 0; ich < o2::tof::Geo::NCHANNELS; ++ich) {
    if (hv->test(ich) != 0) {
      std::cout << "HV for channel " << ich << " is ON" << std::endl;
    }
  }

  return;
}
