// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#if !defined(__CLING__) || defined(__ROOTCLING__)

#include "Framework/Logger.h"
#include "CCDB/CcdbApi.h"
#include <string>
#include <map>

#endif

#include "ZDCBase/Constants.h"
#include "ZDCCalib/WaveformCalibConfig.h"

using namespace o2::zdc;
using namespace std;

void CreateWaveformCalibConfig(long tmin = 0, long tmax = -1, std::string ccdbHost = "")
{

  // This object configures the measurement of the average waveforms for the ZDC channels
  WaveformCalibConfig conf;

  // Threshold to include the interpolated waveform into the average
  // One should avoid to sum signals that are too small and have
  // low signal/background
  // By taking into account the baseline (~1800) and the waveform
  // range -2048 : 2047 one should not use signals too close to
  // maximum allowed amplitude (1800+2048)
  conf.setCuts(600, 3000);

  conf.setDescription("Simulated data");
  conf.setMinEntries(200);

  conf.print();

  o2::ccdb::CcdbApi api;
  map<string, string> metadata; // can be empty
  if (ccdbHost.size() == 0 || ccdbHost == "external") {
    ccdbHost = "http://alice-ccdb.cern.ch:8080";
  } else if (ccdbHost == "internal") {
    ccdbHost = "http://o2-ccdb.internal/";
  } else if (ccdbHost == "test") {
    ccdbHost = "http://ccdb-test.cern.ch:8080";
  } else if (ccdbHost == "local") {
    ccdbHost = "http://localhost:8080";
  }
  api.init(ccdbHost.c_str());
  LOG(info) << "CCDB server: " << api.getURL();
  // store abitrary user object in strongly typed manner
  api.storeAsTFileAny(&conf, CCDBPathInterCalibConfig, metadata, tmin, tmax);

  // return conf;
}
