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

// root includes
#include "TStyle.h"
#include "TCanvas.h"
#include "TGraph.h"
#include "TAxis.h"
#include "TMultiGraph.h"
#include "TLegend.h"

// o2 includes
#include "TPCQC/DCSPTemperature.h"
#include "DataFormatsTPC/DCS.h"

#include <fmt/format.h>

ClassImp(o2::tpc::qc::DCSPTemperature);

using namespace o2::tpc::qc;

DCSPTemperature::~DCSPTemperature()
{
  while (mCanVec.size()) {
    delete mCanVec.back();
    mCanVec.pop_back();
  }
}

void DCSPTemperature::initializeCanvases()
{
  gStyle->SetPalette(kRainBow);
  gStyle->SetTitleSize(0.05, "XY");
  gStyle->SetTitleOffset(1.05, "XY");

  auto cStats = new TCanvas("c_temperature_stats", "c_temperature_stats", 1000, 1000);
  auto cSensorsMultiGraph = new TCanvas("c_temperature_sensors", "c_temperature_sensors", 1000, 1000);

  mCanVec.emplace_back(cStats);
  mCanVec.emplace_back(cSensorsMultiGraph);

  gStyle->SetTimeOffset(0);
}

void DCSPTemperature::processData(const std::vector<std::unique_ptr<o2::tpc::dcs::Temperature>>& data)
{
  int nPointA = 0;
  int nPointC = 0;
  int sensorCounter = 0;

  auto grM = new TMultiGraph;
  std::vector<TGraph*> grVec; // one graph per sensor
  auto grStatsAMean = new TGraph;
  auto grStatsAGradX = new TGraph;
  auto grStatsAGradY = new TGraph;
  auto grStatsCMean = new TGraph;
  auto grStatsCGradX = new TGraph;
  auto grStatsCGradY = new TGraph;

  std::vector<int> pointCountersSensors;

  for (int i = 0; i < 18; i++) {
    pointCountersSensors.emplace_back(0);
    grVec.emplace_back(new TGraph);
    grVec.back()->SetNameTitle(fmt::format("tempSensor{}", i + 1).data(), fmt::format("Temperature Sensor {};;T (K)", i + 1).data());
    grVec.back()->GetXaxis()->SetTimeDisplay(1);
    grVec.back()->GetXaxis()->SetTimeFormat("#splitline{%d.%m.%y}{%H:%M:%S}");
    grVec.back()->SetMarkerStyle(20);
  }

  for (const auto& t : data) {
    sensorCounter = 0;
    for (const auto sensor : t->raw) {
      for (const auto& value : sensor.data) {
        grVec.at(sensorCounter)->SetPoint(pointCountersSensors.at(sensorCounter), double(value.time) / 1000., value.value);
        pointCountersSensors.at(sensorCounter)++;
      }
      sensorCounter++;
    }
    for (const auto& value : t->statsA.data) {
      grStatsAMean->SetPoint(nPointA, double(value.time) / 1000., value.value.mean);
      grStatsAGradX->SetPoint(nPointA, double(value.time) / 1000., value.value.gradX);
      grStatsAGradY->SetPoint(nPointA, double(value.time) / 1000., value.value.gradY);
      nPointA++;
    }
    for (const auto& value : t->statsC.data) {
      grStatsCMean->SetPoint(nPointC, double(value.time) / 1000., value.value.mean);
      grStatsCGradX->SetPoint(nPointC, double(value.time) / 1000., value.value.gradX);
      grStatsCGradY->SetPoint(nPointC, double(value.time) / 1000., value.value.gradY);
      nPointC++;
    }
  }

  for (auto& gr : grVec) {
    gr->Sort();
    grM->Add(gr);
  }

  mCanVec.at(0)->Divide(2, 3);

  grStatsAMean->SetTitle("Temperature mean (A-side);;T (K)");
  grStatsAMean->SetMarkerStyle(20);
  grStatsAMean->GetXaxis()->SetTimeDisplay(1);
  grStatsAMean->GetXaxis()->SetTimeFormat("#splitline{%d.%m.%y}{%H:%M:%S}");
  mCanVec.at(0)->cd(1);
  grStatsAMean->Sort();
  grStatsAMean->Draw("apl");

  grStatsCMean->SetTitle("Temperature mean (C-side);;T (K)");
  grStatsCMean->SetMarkerStyle(20);
  grStatsCMean->GetXaxis()->SetTimeDisplay(1);
  grStatsCMean->GetXaxis()->SetTimeFormat("#splitline{%d.%m.%y}{%H:%M:%S}");
  mCanVec.at(0)->cd(2);
  grStatsCMean->Sort();
  grStatsCMean->Draw("apl");

  grStatsAGradX->SetTitle("Temperature gradient X (A-side);;gradient (K/cm)");
  grStatsAGradX->SetMarkerStyle(20);
  grStatsAGradX->GetXaxis()->SetTimeDisplay(1);
  grStatsAGradX->GetXaxis()->SetTimeFormat("#splitline{%d.%m.%y}{%H:%M:%S}");
  mCanVec.at(0)->cd(3);
  grStatsAGradX->Sort();
  grStatsAGradX->Draw("apl");

  grStatsCGradX->SetTitle("Temperature gradient X (C-side);;gradient (K/cm)");
  grStatsCGradX->SetMarkerStyle(20);
  grStatsCGradX->GetXaxis()->SetTimeDisplay(1);
  grStatsCGradX->GetXaxis()->SetTimeFormat("#splitline{%d.%m.%y}{%H:%M:%S}");
  mCanVec.at(0)->cd(4);
  grStatsCGradX->Sort();
  grStatsCGradX->Draw("apl");

  grStatsAGradY->SetTitle("Temperature gradient Y (A-side);;gradient (K/cm)");
  grStatsAGradY->SetMarkerStyle(20);
  grStatsAGradY->GetXaxis()->SetTimeDisplay(1);
  grStatsAGradY->GetXaxis()->SetTimeFormat("#splitline{%d.%m.%y}{%H:%M:%S}");
  mCanVec.at(0)->cd(5);
  grStatsAGradY->Sort();
  grStatsAGradY->Draw("apl");

  grStatsCGradY->SetTitle("Temperature gradient Y (C-side);;gradient (K/cm)");
  grStatsCGradY->SetMarkerStyle(20);
  grStatsCGradY->GetXaxis()->SetTimeDisplay(1);
  grStatsCGradY->GetXaxis()->SetTimeFormat("#splitline{%d.%m.%y}{%H:%M:%S}");
  mCanVec.at(0)->cd(6);
  grStatsCGradY->Sort();
  grStatsCGradY->Draw("apl");

  TLegend* legend = new TLegend(0.85, 0.6, 0.9, 0.9);
  for (int i = 0; i < grVec.size(); i++) {
    legend->AddEntry(grVec.at(i), grVec.at(i)->GetName(), "p");
  }

  mCanVec.at(1)->cd();
  grM->GetXaxis()->SetTimeDisplay(1);
  grM->GetXaxis()->SetTimeFormat("#splitline{%d.%m.%y}{%H:%M:%S}");
  grM->SetTitle("Raw temperature values;;T (K)");
  grM->Draw("A pmc plc");
  legend->Draw("same");
  mCanVec.at(1)->Update();

  /// associate the graphs with the canvases
  grM->SetBit(TObject::kCanDelete);
  grStatsAMean->SetBit(TObject::kCanDelete);
  grStatsAGradX->SetBit(TObject::kCanDelete);
  grStatsAGradY->SetBit(TObject::kCanDelete);
  grStatsCMean->SetBit(TObject::kCanDelete);
  grStatsCGradX->SetBit(TObject::kCanDelete);
  grStatsCGradY->SetBit(TObject::kCanDelete);
}
