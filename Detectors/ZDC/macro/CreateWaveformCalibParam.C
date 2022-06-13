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

#include <TH1.h>
#include <TFile.h>
#include <string>
#include <map>
#include "CCDB/CcdbApi.h"

#endif

#include "Framework/Logger.h"
#include "ZDCBase/Constants.h"
#include "ZDCCalib/WaveformCalibParam.h"

using namespace o2::zdc;
using namespace std;

void CreateWaveformCalibParam(long tmin = 0, long tmax = -1, std::string ccdbHost = "", std::string sourceDataPath = "signal_shapes.root")
{

  TFile sourceData(sourceDataPath.c_str());
  if (!sourceData.IsOpen() || sourceData.IsZombie()) {
    LOG(fatal) << "Failed to open input file " << sourceDataPath;
  }

  // Source of line shapes, pedestal and noise for each channel
  // Missing histos for: towers 1-4 of all calorimeters, zem1, all towers of zpc
  std::string ShapeName[o2::zdc::NChannels] = {
    "znatc", "znatc", "znatc", "znatc", "znatc", "znatc", // ZNAC, ZNA1, ZNA2, ZNA3, ZNA4, ZNAS (shape not used)
    "zpatc", "zpatc", "zpatc", "zpatc", "zpatc", "zpatc", // ZPAC, ZPA1, ZPA2, ZPA3, ZPA4, ZPAS (shape not used)
    "zem2", "zem2",                                       // ZEM1, ZEM2
    "znctc", "znctc", "znctc", "znctc", "znctc", "znctc", // ZNCC, ZNC1, ZNC2, ZNC3, ZNC4, ZNCS (shape not used)
    "zpatc", "zpatc", "zpatc", "zpatc", "zpatc", "zpatc"  // ZPCC, ZPC1, ZPC2, ZPC3, ZPC4, ZPCS (shape not used)
  };

  o2::zdc::WaveformCalibParam conf;

  for (int ic = 0; ic < o2::zdc::NChannels; ic++) {
    auto& channel = conf.channels[ic];
    std::string histoShapeName = "hw_" + ShapeName[ic];
    TH1* histoShape = (TH1*)sourceData.GetObjectUnchecked(histoShapeName.c_str());
    if (!histoShape) {
      LOG(fatal) << "Failed to extract the shape histogram  " << histoShapeName;
    }
    int nb = histoShape->GetNbinsX();
    channel.shape.resize(nb);
    // we need min amplitude and its bin
    double ampMin = histoShape->GetBinContent(1);
    channel.ampMinID = 0;
    for (int i = 0; i < nb; i++) {
      channel.shape[i] = histoShape->GetBinContent(i + 1);
      if (channel.shape[i] < ampMin) {
        ampMin = channel.shape[i];
        channel.ampMinID = i;
      }
    }
    if (ampMin == 0.) {
      LOG(fatal) << "Amplitude minimum =0 for histo " << histoShapeName;
    }
  }

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
  api.storeAsTFileAny(&conf, CCDBPathWaveformCalib, metadata, tmin, tmax);

  // return conf;
}
