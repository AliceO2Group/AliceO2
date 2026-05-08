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
#include <iostream>
#include <array>
#endif

#include "CCDB/CcdbApi.h"
#include "CCDB/CCDBTimeStampUtils.h"
#include "TH1F.h"
#include "DataFormatsFT0/EventsPerBc.h"
#include "Framework/Logger.h"
#include "CommonConstants/LHCConstants.h"

std::unique_ptr<TH1F> hist;
std::unique_ptr<TCanvas> canvas;

void FT0readEventsPerBc(std::string ccdbUrl, long timestamp)
{
  o2::ccdb::CcdbApi ccdbApi;
  ccdbApi.init(ccdbUrl);
  const std::string ccdbPath = "FT0/Calib/EventsPerBc";
  std::map<std::string, std::string> metadata;

  if (timestamp < 0) {
    timestamp = o2::ccdb::getCurrentTimestamp();
  }

  std::unique_ptr<o2::ft0::EventsPerBc> events(ccdbApi.retrieveFromTFileAny<o2::ft0::EventsPerBc>(ccdbPath, metadata, timestamp));

  if (!events) {
    LOGP(fatal, "EventsPerBc object not found in {}/{} for timestamp {}.", ccdbUrl, ccdbPath, timestamp);
    return;
  }

  hist = std::make_unique<TH1F>("eventsPerBcHist", "Events per BC", o2::constants::lhc::LHCMaxBunches, 0, o2::constants::lhc::LHCMaxBunches - 1);
  for (int idx = 0; idx < o2::constants::lhc::LHCMaxBunches; idx++) {
    hist->Fill(idx, events->histogram[idx]);
  }
  canvas = std::make_unique<TCanvas>();
  hist->Draw();
  canvas->Draw();
}