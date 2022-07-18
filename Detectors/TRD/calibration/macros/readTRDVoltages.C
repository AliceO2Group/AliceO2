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

#else

#error This macro must run in compiled mode

#endif

/*
Hint: to get a time stamp for a given time, use the date utility like so:

We are interested in the voltage values on 27th of June 2022 at 1:22 am, so we do:
date -d "01:22 2022-06-27" +%s
This returns 1656285720. Since this time is given in seconds it has to be multiplied by 1000.
Then we can do readTRDVoltages(1656285720000)
*/

void readTRDVoltages(long ts = -1, bool savePlots = false)
{
  // prepare some histograms for the voltages
  std::vector<TH2F*> histsAnode;
  std::vector<TH2F*> histsDrift;
  for (int iSec = 0; iSec < o2::trd::constants::NSECTOR; ++iSec) {
    histsAnode.emplace_back(new TH2F(Form("anodeSec%i", iSec), Form("Anodes sec %i;stack;layer;U(V)", iSec), 5, -0.5, 4.5, 6, -0.5, 5.5));
    histsDrift.emplace_back(new TH2F(Form("driftSec%i", iSec), Form("Drift sec %i;stack;layer;U(V)", iSec), 5, -0.5, 4.5, 6, -0.5, 5.5));
    histsAnode.back()->SetStats(0);
    histsDrift.back()->SetStats(0);
  }

  // now, access the actual calibration object from CCDB
  auto& ccdbmgr = o2::ccdb::BasicCCDBManager::instance();
  // if no ts is supplied we just take the current CCDB object
  auto cal = (ts < 0) ? ccdbmgr.get<unordered_map<o2::dcs::DataPointIdentifier, float>>("TRD/Calib/DCSDPsU") : ccdbmgr.getForTimeStamp<unordered_map<o2::dcs::DataPointIdentifier, float>>("TRD/Calib/DCSDPsU", ts);

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
  auto cAnode = new TCanvas("cAnode", "cAnode", 1800, 1000);
  cAnode->Divide(6, 3);
  for (int iSec = 0; iSec < o2::trd::constants::NSECTOR; ++iSec) {
    auto pad = cAnode->cd(iSec + 1);
    pad->SetRightMargin(0.15);
    histsAnode[iSec]->GetZaxis()->SetRangeUser(0, 1500);
    histsAnode[iSec]->Draw("colz text");
  }
  if (savePlots) {
    cAnode->SaveAs("anodeVoltages.pdf");
  }

  auto cDrift = new TCanvas("cDrift", "cDrift", 1800, 1000);
  cDrift->Divide(6, 3);
  for (int iSec = 0; iSec < o2::trd::constants::NSECTOR; ++iSec) {
    auto pad = cDrift->cd(iSec + 1);
    pad->SetRightMargin(0.15);
    histsDrift[iSec]->GetZaxis()->SetRangeUser(0, 2500);
    histsDrift[iSec]->Draw("colz text");
  }
  if (savePlots) {
    cDrift->SaveAs("driftVoltages.pdf");
  }

  return;
}
