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

// macro showing how to read a CCDB object created from DCS data points

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include "CCDB/BasicCCDBManager.h"
#include "CCDB/CcdbApi.h"
#include "DetectorsDCS/DataPointIdentifier.h"
#include "DataFormatsTRD/Constants.h"
#include "DataFormatsTRD/HelperMethods.h"

#include "TH2.h"
#include "TCanvas.h"

#include <string>
#include <vector>
#include <unordered_map>
#include <chrono>
#include <bitset>
#include <sstream>
#include <iomanip>

#endif

void readTRDVoltages()
{
  // prepare some histograms for the voltages
  std::vector<std::unique_ptr<TH2F>> histsAnode;
  std::vector<std::unique_ptr<TH2F>> histsDrift;
  for (int iSec = 0; iSec < o2::trd::constants::NSECTOR; ++iSec) {
    histsAnode.emplace_back(std::make_unique<TH2F>(Form("anodeSec%i", iSec), Form("Anodes sec %i;stack;layer;U(V)", iSec), 5, -0.5, 4.5, 6, -0.5, 5.5));
    histsDrift.emplace_back(std::make_unique<TH2F>(Form("driftSec%i", iSec), Form("Drift sec %i;stack;layer;U(V)", iSec), 5, -0.5, 4.5, 6, -0.5, 5.5));
    histsAnode.back()->SetStats(0);
    histsDrift.back()->SetStats(0);
  }

  // now, access the actual calibration object from CCDB
  auto& ccdbmgr = o2::ccdb::BasicCCDBManager::instance();
  auto cal = ccdbmgr.get<unordered_map<o2::dcs::DataPointIdentifier, float>>("TRD/Calib/DCSDPsU");

  o2::dcs::DataPointIdentifier dpidAnode; // used as key to access the map
  o2::dcs::DataPointIdentifier dpidDrift; // used as key to access the map

  for (int iDet = 0; iDet < o2::trd::constants::MAXCHAMBER; ++iDet) {
    std::stringstream ss;
    ss << std::setw(3) << std::setfill('0') << iDet;
    std::string aliasAnode = "trd_hvAnodeUmon" + ss.str();
    std::string aliasDrift = "trd_hvDriftUmon" + ss.str();
    o2::dcs::DataPointIdentifier::FILL(dpidAnode, aliasAnode, o2::dcs::DeliveryType::DPVAL_DOUBLE);
    o2::dcs::DataPointIdentifier::FILL(dpidDrift, aliasDrift, o2::dcs::DeliveryType::DPVAL_DOUBLE);
    auto uAnode = cal->at(dpidAnode);
    auto uDrift = cal->at(dpidDrift);
    int sec = o2::trd::HelperMethods::getSector(iDet);
    int stack = o2::trd::HelperMethods::getStack(iDet);
    int ly = o2::trd::HelperMethods::getLayer(iDet);
    histsAnode[sec]->SetBinContent(histsAnode[sec]->GetXaxis()->FindBin(stack), histsAnode[sec]->GetYaxis()->FindBin(ly), uAnode);
    histsDrift[sec]->SetBinContent(histsDrift[sec]->GetXaxis()->FindBin(stack), histsDrift[sec]->GetYaxis()->FindBin(ly), uDrift);
  }

  // plot the obtained values
  auto cAnode = std::make_unique<TCanvas>("cAnode", "cAnode", 1800, 1000);
  cAnode->Divide(6, 3);
  for (int iSec = 0; iSec < o2::trd::constants::NSECTOR; ++iSec) {
    auto pad = cAnode->cd(iSec + 1);
    pad->SetRightMargin(0.15);
    histsAnode[iSec]->GetZaxis()->SetRangeUser(0, 1500);
    histsAnode[iSec]->Draw("colz text");
  }
  cAnode->SaveAs("anodeVoltages.pdf");

  auto cDrift = std::make_unique<TCanvas>("cDrift", "cDrift", 1800, 1000);
  cDrift->Divide(6, 3);
  for (int iSec = 0; iSec < o2::trd::constants::NSECTOR; ++iSec) {
    auto pad = cDrift->cd(iSec + 1);
    pad->SetRightMargin(0.15);
    histsDrift[iSec]->GetZaxis()->SetRangeUser(0, 2500);
    histsDrift[iSec]->Draw("colz text");
  }
  cDrift->SaveAs("driftVoltages.pdf");

  return;
}
