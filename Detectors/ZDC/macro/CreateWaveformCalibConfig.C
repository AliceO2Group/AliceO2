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

#include <string>
#include <map>
#include <TFile.h>
#include "Framework/Logger.h"
#include "CCDB/CcdbApi.h"
#include "ZDCBase/Constants.h"
#include "ZDCCalib/WaveformCalibConfig.h"

#endif

#include "ZDCBase/Helpers.h"
using namespace o2::zdc;
using namespace std;

void CreateWaveformCalibConfig(long tmin = 0, long tmax = -1, std::string ccdbHost = "")
{
  // Shortcuts: internal, external, test, local, root

  // This object configures the measurement of the average waveforms for the ZDC channels
  WaveformCalibConfig conf;

  // Threshold to include the interpolated waveform into the average
  // One should avoid to sum signals that are too small and have
  // low signal/background
  // By taking into account the baseline (~1800) and the waveform
  // range -2048 : 2047 one should not use signals too close to
  // maximum allowed amplitude (1800+2048)
  conf.setCuts(100, 3000);
  conf.setCuts(o2::zdc::IdZNA1, 100, 2500);
  conf.setCuts(o2::zdc::IdZNA2, 100, 2500);
  conf.setCuts(o2::zdc::IdZNA3, 100, 2500);
  conf.setCuts(o2::zdc::IdZNA4, 100, 2500);
  conf.setCuts(o2::zdc::IdZPA1, 50, 2500);
  conf.setCuts(o2::zdc::IdZPA2, 100, 2500);
  conf.setCuts(o2::zdc::IdZPA3, 100, 2500);
  conf.setCuts(o2::zdc::IdZPA4, 100, 2500);
  conf.setCuts(o2::zdc::IdZNC1, 100, 2500);
  conf.setCuts(o2::zdc::IdZNC2, 100, 2500);
  conf.setCuts(o2::zdc::IdZNC3, 100, 2500);
  conf.setCuts(o2::zdc::IdZNC4, 100, 2500);
  conf.setCuts(o2::zdc::IdZPC1, 100, 2500);
  conf.setCuts(o2::zdc::IdZPC2, 100, 2500);
  conf.setCuts(o2::zdc::IdZPC3, 100, 2500);
  conf.setCuts(o2::zdc::IdZPC4, 50, 2500);

  conf.setDescription("Simulated data");
  conf.setMinEntries(200);

  // Restrict waveform range (default is -3, 6 as defined in WaveformCalib_NBB
  // WaveformCalib_NBA in file Detectors/ZDC/base/include/ZDCBase/Constants.h)
  // conf.restrictRange(-1, 0);

  conf.print();

  std::string ccdb_host = ccdbShortcuts(ccdbHost, conf.Class_Name(), CCDBPathWaveformCalibConfig);

  if (endsWith(ccdb_host, ".root")) {
    TFile f(ccdb_host.data(), "recreate");
    f.WriteObjectAny(&conf, conf.Class_Name(), "ccdb_object");
    f.Close();
    return;
  }

  o2::ccdb::CcdbApi api;
  map<string, string> metadata; // can be empty
  api.init(ccdb_host.c_str());
  LOG(info) << "CCDB server: " << api.getURL();
  // store abitrary user object in strongly typed manner
  api.storeAsTFileAny(&conf, CCDBPathWaveformCalibConfig, metadata, tmin, tmax);
}
