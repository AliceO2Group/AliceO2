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

#include "CommonUtils/ValueMonitor.h"
#include "TFile.h"
#include <fairlogger/Logger.h>
#include <memory>

using namespace o2::utils;

ValueMonitor::ValueMonitor(std::string filename) : mFileName(filename) {}

ValueMonitor::~ValueMonitor()
{
  if (mHistos.size() > 0) {
    auto outfile = std::make_unique<TFile>(mFileName.c_str(), "RECREATE");
    // write all histos
    for (auto& h : mHistos) {
      LOG(INFO) << "ValueMonitor: WRITING HISTO " << h.second->GetName();
      h.second->Write();
    }
    outfile->Close();
  }
}
