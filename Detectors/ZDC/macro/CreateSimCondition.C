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
#include <TH1.h>
#include <TFile.h>
#include <TRandom.h>
#endif

#include "ZDCSimulation/SimCondition.h"
#include "ZDCBase/Constants.h"

using namespace std;

void CreateSimCondition(long tmin = 0, long tmax = -1, std::string ccdbHost = "", std::string sourceDataPath = "signal_shapes.root")
{
  TFile sourceData(sourceDataPath.c_str());
  if (!sourceData.IsOpen() || sourceData.IsZombie()) {
    LOG(fatal) << "Failed to open input file " << sourceDataPath;
  }
  o2::zdc::SimCondition conf;

  const float Gains[5] = {15.e-3, 30.e-3, 100.e-3, 15.e-3, 30.e-3}; // gain (response per photoelectron)
  const float fudgeFactor = 5.0;                                    // ad hoc factor to tune the gain in the MC

  // Source of line shapes, pedestal and noise for each channel
  // Missing histos for: towers 1-4 of all calorimeters, zem1, all towers of zpc
  std::string ShapeName[o2::zdc::NChannels] = {
    "znatc", "znatc", "znatc", "znatc", "znatc", "znatc", // ZNAC, ZNA1, ZNA2, ZNA3, ZNA4, ZNAS (shape not used)
    "zpatc", "zpatc", "zpatc", "zpatc", "zpatc", "zpatc", // ZPAC, ZPA1, ZPA2, ZPA3, ZPA4, ZPAS (shape not used)
    "zem2", "zem2",                                       // ZEM1, ZEM2
    "znctc", "znctc", "znctc", "znctc", "znctc", "znctc", // ZNCC, ZNC1, ZNC2, ZNC3, ZNC4, ZNCS (shape not used)
    "zpatc", "zpatc", "zpatc", "zpatc", "zpatc", "zpatc"  // ZPCC, ZPC1, ZPC2, ZPC3, ZPC4, ZPCS (shape not used)
  };

  // clang-format off
  // Compensation of signal arrival time (ns)
  float pos[o2::zdc::NChannels+1]={
-1, // pos_ZNAC
-1, // pos_ZNA1
-1, // pos_ZNA2
-1, // pos_ZNA3
-1, // pos_ZNA4
-1, // pos_ZNAS
-1, // pos_ZPAC
-1, // pos_ZPA1
-1, // pos_ZPA2
-1, // pos_ZPA3
-1, // pos_ZPA4
-1, // pos_ZPAS
-1, // pos_ZEM1
-1, // pos_ZEM2
-1, // pos_ZNCC
-1, // pos_ZNC1
-1, // pos_ZNC2
-1, // pos_ZNC3
-1, // pos_ZNC4
-1, // pos_ZNCS
-1, // pos_ZPCC
-1, // pos_ZPC1
-1, // pos_ZPC2
-1, // pos_ZPC3
-1, // pos_ZPC4
-1, // pos_ZPCS
0}; // pos_END
  // clang-format on

  for (int ic = 0; ic < o2::zdc::NChannels; ic++) {

    auto& channel = conf.channels[ic];
    int tower = 0;
    int det = o2::zdc::toDet(ic, tower); // detector ID for this channel
    //
    channel.gain = (tower != o2::zdc::Sum) ? fudgeFactor * Gains[det - 1] : 1.0;
    if (ic == o2::zdc::IdZPA4 || ic == o2::zdc::IdZPC4) {
      channel.gainInSum = 0.5;
      channel.gain = channel.gain / channel.gainInSum;
    }
    //
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
    auto ampMinAInv = 1. / std::abs(ampMin);
    for (int i = 0; i < nb; i++) {
      channel.shape[i] *= ampMinAInv;
    }
    //
    channel.pedestal = gRandom->Gaus(1800., 30.);
    //
    std::string histoPedNoiseName = "hb_" + ShapeName[ic];
    TH1* histoPedNoise = (TH1*)sourceData.GetObjectUnchecked(histoPedNoiseName.c_str());
    if (!histoPedNoise) {
      LOG(fatal) << "Failed to extract the pedestal noise histogram  " << histoPedNoise;
    }
    channel.pedestalNoise = histoPedNoise->GetRMS();
    //
    std::string histoPedFluctName = "hp_" + ShapeName[ic];
    TH1* histoPedFluct = (TH1*)sourceData.GetObjectUnchecked(histoPedFluctName.c_str());
    if (!histoPedFluct) {
      LOG(fatal) << "Failed to extract the pedestal fluctuation histogram  " << histoPedFluct;
    }
    channel.pedestalFluct = histoPedFluct->GetRMS();
    //
    channel.timeJitter = 0.291668f;
    //
    channel.timePosition = 12.5f;
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
  api.storeAsTFileAny(&conf, o2::zdc::CCDBPathConfigSim, metadata, tmin, tmax);

  //  return conf;
}
