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

/// \file CheckConfigEvent.C
/// \brief Check a config event

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <TFile.h>
#include <TTree.h>
#include <TH1F.h>
#include <TH2F.h>
#include <TProfile.h>
#include <TCanvas.h>
#include <TLegend.h>

#include <fairlogger/Logger.h>
#include "DataFormatsTRD/Digit.h"
#include "DataFormatsTRD/Constants.h"
#include "DataFormatsTRD/TrapConfigEvent.h"
#endif

using namespace o2::trd;

constexpr int kMINENTRIES = 100;

void CheckDigits(std::string configeventfile = "trddigits.root", uint32_t mcmid = 42)
{
  TFile* fin = TFile::Open(configeventfile.data());
  TTree* configeventTree = (TTree*)fin->Get("o2sim");
  TrapConfigEvent* trapconfigevent = nullptr;
  configeventTree->SetBranchAddress("ConfigEvent", &trapconfigevent);
  int nev = configeventTree->GetEntries();

  LOG(info) << nev << " entries found";
  configeventTree->GetEvent(0);
  for (int reg = 0; reg < 433; ++reg) {
    LOGP(info, "register :{}, value :{} ", trapconfigevent->getRegisterName(reg), trapconfigevent->getRegisterValue(reg, mcmid));
  }
}
