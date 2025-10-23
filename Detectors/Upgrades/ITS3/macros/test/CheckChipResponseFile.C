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

/// \file CheckChipResponseFile.C
/// \brief Simple macro to check the chip response files

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <TFile.h>
#include <TGraph.h>
#include <TCanvas.h>
#include <TH1.h>
#include <TLegend.h>
#include <iostream>
#include <vector>
#include <string>

#define ENABLE_UPGRADES
#include "ITSMFTSimulation/AlpideSimResponse.h"
#include "ITS3Simulation/ChipSimResponse.h"

#include "ITS3Base/SegmentationMosaix.h"
#include "fairlogger/Logger.h"
#endif

using SegmentationMosaix = o2::its3::SegmentationMosaix;

double um2cm(double um) { return um * 1e-4; }
double cm2um(double cm) { return cm * 1e+4; }

std::unique_ptr<o2::its3::ChipSimResponse> mAlpSimResp0, mAlpSimResp1, mAptSimResp1;

std::unique_ptr<o2::its3::ChipSimResponse> loadResponse(const std::string& fileName, const std::string& respName)
{
  TFile* f = TFile::Open(fileName.data());
  if (!f) {
    std::cerr << fileName << " not found" << std::endl;
    return nullptr;
  }
  auto base = f->Get<o2::itsmft::AlpideSimResponse>(respName.c_str());
  if (!base) {
    std::cerr << respName << " not found in " << fileName << std::endl;
    return nullptr;
  }
  return std::make_unique<o2::its3::ChipSimResponse>(base);
}

void LoadRespFunc()
{
  std::string AptsFile = "$(O2_ROOT)/share/Detectors/Upgrades/ITS3/data/ITS3ChipResponseData/APTSResponseData.root";
  std::string AlpideFile = "$(O2_ROOT)/share/Detectors/ITSMFT/data/AlpideResponseData/AlpideResponseData.root";

  std::cout << "=====================\n";
  LOGP(info, "ALPIDE Vbb=0V response");
  mAlpSimResp0 = loadResponse(AlpideFile, "response0"); // Vbb=0V
  mAlpSimResp0->computeCentreFromData();
  mAlpSimResp0->print();
  LOGP(info, "Response Centre {}", mAlpSimResp0->getRespCentreDep());
  std::cout << "=====================\n";
  LOGP(info, "ALPIDE Vbb=-3V response");
  mAlpSimResp1 = loadResponse(AlpideFile, "response1"); // Vbb=-3V
  mAlpSimResp1->computeCentreFromData();
  mAlpSimResp1->print();
  LOGP(info, "Response Centre {}", mAlpSimResp1->getRespCentreDep());
  std::cout << "=====================\n";
  LOGP(info, "APTS response");
  mAptSimResp1 = loadResponse(AptsFile, "response1"); // APTS
  mAptSimResp1->computeCentreFromData();
  mAptSimResp1->print();
  LOGP(info, "Response Centre {}", mAptSimResp1->getRespCentreDep());
  std::cout << "=====================\n";
}

std::vector<float> getCollectionSeediciencies(o2::its3::ChipSimResponse* resp,
                                              const std::vector<float>& depths)
{
  std::vector<float> seed;
  bool flipRow = false, flipCol = false;
  for (auto depth : depths) {
    auto rspmat = resp->getResponse(0.0, 0.0,
                                    um2cm(depth) + 1.e-9,
                                    flipRow, flipCol);
    seed.push_back(rspmat ? rspmat->getValue(2, 2) : 0.f);
  }
  return seed;
}

std::vector<float> getShareValues(o2::its3::ChipSimResponse* resp,
                                  const std::vector<float>& depths)
{
  std::vector<float> share;
  bool flipRow = false, flipCol = false;
  for (auto depth : depths) {
    auto rspmat = resp->getResponse(0.0, 0.0,
                                    um2cm(depth) + 1.e-9,
                                    flipRow, flipCol);
    float s = 0;
    int npix = resp->getNPix();
    if (rspmat) {
      for (int i = 0; i < npix; ++i)
        for (int j = 0; j < npix; ++j)
          if (!(i == npix / 2 && j == npix / 2))
            s += rspmat->getValue(i, j);
    }
    share.push_back(s);
  }
  return share;
}

std::vector<float> getEffValues(o2::its3::ChipSimResponse* resp,
                                const std::vector<float>& depths)
{
  std::vector<float> all;
  bool flipRow = false, flipCol = false;
  for (auto depth : depths) {
    auto rspmat = resp->getResponse(0.0, 0.0,
                                    um2cm(depth) + 1.e-9,
                                    flipRow, flipCol);
    float s = 0;
    int npix = resp->getNPix();
    if (rspmat) {
      for (int i = 0; i < npix; ++i)
        for (int j = 0; j < npix; ++j)
          s += rspmat->getValue(i, j);
    }
    all.push_back(s);
  }
  return all;
}

void CheckChipResponseFile()
{
  LoadRespFunc();
  LOG(info) << "Response function loaded" << std::endl;

  std::vector<float> vecDepth;
  int numPoints = 100;
  for (int i = 0; i < numPoints; ++i) {
    float value = -50 + i * (100.0f / (numPoints - 1));
    vecDepth.push_back(value);
  }

  int colors[] = {kOrange + 7, kRed + 1, kAzure + 4};
  struct RespInfo {
    std::unique_ptr<o2::its3::ChipSimResponse>& resp;
    std::string title;
    int color;
  };
  std::vector<RespInfo> responses = {
    {mAptSimResp1, "APTS", colors[0]},
    {mAlpSimResp0, "ALPIDE Vbb=0V", colors[1]},
    {mAlpSimResp1, "ALPIDE Vbb=-3V", colors[2]}};

  TCanvas* c1 = new TCanvas("c1", "c1", 800, 600);
  TH1* frame = c1->DrawFrame(-50, -0.049, 50, 1.049);
  frame->SetTitle(";Depth(um);Charge Collection Seed / Share / Eff");
  TLegend* leg = new TLegend(0.15, 0.5, 0.4, 0.85);
  leg->SetFillStyle(0);
  leg->SetBorderSize(0);

  for (auto& r : responses) {
    if (!r.resp)
      continue;
    auto seed = getCollectionSeediciencies(r.resp.get(), vecDepth);
    auto shr = getShareValues(r.resp.get(), vecDepth);
    auto all = getEffValues(r.resp.get(), vecDepth);

    auto grSeed = new TGraph(vecDepth.size(), vecDepth.data(), seed.data());
    grSeed->SetTitle(Form("%s seed", r.title.c_str()));
    grSeed->SetLineColor(r.color);
    grSeed->SetLineWidth(2);
    grSeed->SetMarkerColor(r.color);
    grSeed->SetMarkerStyle(kFullCircle);
    grSeed->SetMarkerSize(0.8);
    grSeed->Draw("SAME LP");
    leg->AddEntry(grSeed, Form("%s seed", r.title.c_str()), "lp");

    auto grShare = new TGraph(vecDepth.size(), vecDepth.data(), shr.data());
    grShare->SetLineColor(r.color);
    grShare->SetLineWidth(2);
    grShare->SetMarkerColor(r.color);
    grShare->SetMarkerStyle(kOpenSquare);
    grShare->SetMarkerSize(1);
    grShare->Draw("SAME LP");
    leg->AddEntry(grShare, Form("%s share", r.title.c_str()), "p");

    auto grEff = new TGraph(vecDepth.size(), vecDepth.data(), all.data());
    grEff->SetLineColor(r.color);
    grEff->SetLineWidth(2);
    grEff->SetMarkerColor(r.color);
    grEff->SetMarkerStyle(kFullDiamond);
    grEff->SetMarkerSize(1);
    grEff->Draw("SAME LP");
    leg->AddEntry(grEff, Form("%s eff", r.title.c_str()), "p");
  }
  leg->Draw();

  c1->SaveAs("ChipResponse.pdf");
}
