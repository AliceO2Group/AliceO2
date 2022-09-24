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

/// \file plotVdriftExB.C
/// \brief Plot calibration values in ccdb for a particluar duration.
/// \author Felix Schlepper

#if !defined(__CLING__) || defined(__ROOTCLING__)
// ROOT header
#include <TFile.h>
#include <TAxis.h>
#include <TGraph.h>
#include <TMultiGraph.h>

// O2 header
#include "CCDB/BasicCCDBManager.h"
#include "CCDB/CcdbApi.h"
#include "DataFormatsTRD/CalVdriftExB.h"
#include "DataFormatsParameters/GRPECSObject.h"

#include <map>
#include <string>
#include <vector>
#endif

using timePoint = o2::parameters::GRPECSObject::timePoint;
static std::array<std::vector<std::tuple<double, double, timePoint>>, 540> vmap; // Map holding all values for all detectors.
static std::array<bool, 540> good;                                               // Marks wether or not a chamber was 'good' for the entire duration.

// Download the values and populate the map.
void ccdbDownload(unsigned int runNumber, std::string ccdb, timePoint queryInterval)
{
  auto& ccdbMgr = o2::ccdb::BasicCCDBManager::instance();
  ccdbMgr.setURL("https://alice-ccdb.cern.ch/");
  auto runDuration = ccdbMgr.getRunDuration(runNumber);
  std::map<std::string, std::string> md;
  md["runNumber"] = std::to_string(runNumber);
  const auto* grp = ccdbMgr.getSpecific<o2::parameters::GRPECSObject>("GLO/Config/GRPECS", (runDuration.first + runDuration.second) / 2, md);
  grp->print();
  const auto startTime = grp->getTimeStart();
  const auto endTime = grp->getTimeEnd();
  ccdbMgr.setURL(ccdb);

  for (timePoint time = startTime; time < endTime; time += queryInterval) {
    ccdbMgr.setTimestamp(time);
    std::cout << "Getting vDriftExB for " << time << std::endl;
    auto calVdriftExB =
      ccdbMgr.get<o2::trd::CalVdriftExB>("TRD/Calib/CalVdriftExB");
    for (int iDet = 0; iDet < 540; ++iDet) {
      vmap[iDet].push_back(std::make_tuple(
        calVdriftExB->getVdrift(iDet), calVdriftExB->getExB(iDet), static_cast<int>(time - startTime)));
    }
  }
}

// Find all good chambers
// 'Good' is a chamber when it does not contain the defautl values.
void find_good()
{
  for (int iDet = 0; iDet < 540; ++iDet) {
    good[iDet] = true;
    if (vmap[iDet].size() == 0) { // No data for detector
      good[iDet] = false;
      continue;
    }
    for (auto& e : vmap[iDet]) {
      if (std::get<0>(e) == 1.0 && std::get<1>(e) == 0.0) {
        good[iDet] = false;
      }
    }
  }
}

// Print out all 'good' chambers.
void print_good()
{
  printf("Good Chamber(s):\n");
  for (int iDet = 0; iDet < 540; ++iDet) {
    if (!good[iDet])
      continue;

    printf("%i;", iDet);
  }
  printf("\n");
}

std::unique_ptr<TMultiGraph> draw(int i)
{
  auto mg = std::make_unique<TMultiGraph>();
  auto g_vDrift = new TGraph();
  g_vDrift->SetLineColor(kBlue);
  auto g_ExB = new TGraph();
  g_ExB->SetLineColor(kRed);
  for (auto& e : vmap[i]) {
    g_vDrift->AddPoint(std::get<2>(e), std::get<0>(e));
    g_ExB->AddPoint(std::get<2>(e), std::get<1>(e));
  }

  g_vDrift->Fit("pol0", "q");
  g_ExB->Fit("pol0", "q");
  mg->Add(g_vDrift);
  mg->Add(g_ExB);
  mg->SetName(Form("vDrift_ExV_%d", i));
  mg->SetTitle(Form("VDrift (=blue) and ExB (=red) - Chamber %d (%s)", i, (good[i]) ? "GOOD" : "BAD"));
  mg->GetXaxis()->SetTitle("time since RunStart (ms)");

  return std::move(mg);
}

// Plot the calibration values for a particular duration.
// The times must be given in unix epoch ms format.
// The default Run is 523677 for the test ccdb.
// The offset for the next calibration is 15 Minutes.
void plotVdriftExB(unsigned int runNumber = 523677, std::string ccdb = "http://ccdb-test.cern.ch:8080", timePoint queryInterval = 900000)
{
  ccdbDownload(runNumber, ccdb, queryInterval);
  find_good();
  print_good();

  std::unique_ptr<TFile> outFile(TFile::Open("plotVdriftExB.root", "RECREATE"));
  for (int i = 0; i < 540; ++i) {
    auto g = draw(i);
    g->Write();
  }
}
