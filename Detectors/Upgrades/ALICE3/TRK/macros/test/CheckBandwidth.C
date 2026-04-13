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

/// \file CheckDigits.C
/// \brief Simple macro to check TRK digits

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <algorithm>
#include <cmath>
#include <map>
#include <TCanvas.h>
#include <TFile.h>
#include <TH1D.h>
#include <TH2F.h>
#include <TLatex.h>
#include <TString.h>
#include <TTree.h>
#include <TStyle.h>

#include "TRKBase/GeometryTGeo.h"
#include "DataFormatsITSMFT/Digit.h"
#include "MathUtils/Utils.h"
#include "DetectorsBase/GeometryManager.h"

#include "DataFormatsITSMFT/ROFRecord.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "SimulationDataFormat/DigitizationContext.h"

#endif

namespace
{
constexpr double DigitBits = 16.;
constexpr double BunchCrossingNS = 25.;
constexpr int ReadoutCycleBC = 18;
constexpr int ReadoutCycleSimBC = 18;
constexpr double ReadoutCycleSeconds = ReadoutCycleBC * BunchCrossingNS * 1.e-9;
} // namespace

void CheckBandwidth(std::string digifile = "trkdigits.root", std::string inputGeom = "o2sim_geometry.root", std::string collContextFile = "collisioncontext.root")
{
  gStyle->SetPalette(55);
  gStyle->SetOptStat(0);

  auto drawSummary = [](double averageValue, double peakValue, const char* unit) {
    TLatex latex;
    latex.SetNDC();
    latex.SetTextSize(0.03);
    latex.SetTextAlign(13);
    latex.DrawLatex(0.04, 0.05, Form("avg: %.3f %s", averageValue, unit));
    latex.DrawLatex(0.34, 0.05, Form("peak: %.3f %s", peakValue, unit));
  };

  auto drawCollisionSummary = [](double averageValue, double nonEmptyAverageValue, double peakValue) {
    TLatex latex;
    latex.SetNDC();
    latex.SetTextSize(0.03);
    latex.SetTextAlign(13);
    latex.DrawLatex(0.04, 0.025, Form("avg: %.3f collisions/ROF", averageValue));
    latex.DrawLatex(0.42, 0.025, Form("peak: %.3f collisions/ROF", peakValue));
    latex.DrawLatex(0.04, 0.06, Form("avg non-empty: %.3f collisions/ROF", nonEmptyAverageValue));
  };

  using namespace o2::base;
  using namespace o2::trk;

  TFile* f = TFile::Open("CheckBandwidth.root", "recreate");

  // Geometry
  o2::base::GeometryManager::loadGeometry(inputGeom);
  auto* gman = o2::trk::GeometryTGeo::Instance();
  gman->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::L2G));

  // Collision Context
  TFile* ccFile = TFile::Open(collContextFile.data());
  auto* digiContext = (o2::steer::DigitizationContext*)ccFile->Get("DigitizationContext");
  const o2::InteractionRecord firstSampledIR{0, digiContext->getFirstOrbitForSampling()};
  std::vector<unsigned int> collisionsPerROF;

  for (const auto& record : digiContext->getEventRecords()) {
    auto nbc = record.differenceInBC(firstSampledIR);
    if (record.getTimeOffsetWrtBC() < 0. && nbc > 0) {
      --nbc;
    }
    if (nbc < 0) {
      continue;
    }

    const size_t rofID = nbc / ReadoutCycleSimBC;
    if (rofID >= collisionsPerROF.size()) {
      collisionsPerROF.resize(rofID + 1, 0u);
    }
    ++collisionsPerROF[rofID];
  }

  // Digits
  TFile* digFile = TFile::Open(digifile.data());
  TTree* digTree = (TTree*)digFile->Get("o2sim");
  const int nDigitTreeEntries = digTree->GetEntries();

  std::vector<o2::itsmft::Digit>* digArr = nullptr;
  digTree->SetBranchAddress("TRKDigit", &digArr);

  // Get Read Out Frame arrays
  std::vector<o2::itsmft::ROFRecord>* ROFRecordArrray = nullptr;
  digTree->SetBranchAddress("TRKDigitROF", &ROFRecordArrray);
  std::vector<o2::itsmft::ROFRecord>& ROFRecordArrrayRef = *ROFRecordArrray;

  digTree->GetEntry(0);

  if (nDigitTreeEntries > 1) {
    LOG(warning) << "Digit tree has " << nDigitTreeEntries << " entries, but this macro processes entry 0 only.";
  }

  std::vector<unsigned long long> digitsPerChip(gman->getNumberOfChips(), 0ull);
  std::vector<unsigned int> maxDigitsPerROFPerChip(gman->getNumberOfChips(), 0u);
  std::vector<unsigned int> digitsInCurrentROFPerChip(gman->getNumberOfChips(), 0u);

  const int nROFRec = (int)ROFRecordArrrayRef.size();
  const int nCollisionROFBins = std::max(nROFRec, static_cast<int>(collisionsPerROF.size()));

  if (nCollisionROFBins > 0) {
    auto* hCollisionsPerROF = new TH1D("h_collisions_per_rof", "Collisions per ROF;ROF id;N collisions", nCollisionROFBins, -0.5, nCollisionROFBins - 0.5);
    double totalCollisionsPerROF = 0.;
    double peakCollisionsPerROF = 0.;
    int nNonEmptyROFs = 0;

    for (int rofID = 0; rofID < nCollisionROFBins; ++rofID) {
      const double nCollisions = rofID < static_cast<int>(collisionsPerROF.size()) ? collisionsPerROF[rofID] : 0.;
      hCollisionsPerROF->SetBinContent(rofID + 1, nCollisions);
      totalCollisionsPerROF += nCollisions;
      peakCollisionsPerROF = std::max(peakCollisionsPerROF, nCollisions);
      if (nCollisions > 0.) {
        ++nNonEmptyROFs;
      }
    }

    auto* canvCollisionsPerROF = new TCanvas("canvCollisionsPerROF", "Collisions per ROF", 1050, 1050);
    canvCollisionsPerROF->SetTopMargin(0.08);
    hCollisionsPerROF->Draw("hist");
    drawCollisionSummary(totalCollisionsPerROF / nCollisionROFBins,
                         nNonEmptyROFs > 0 ? totalCollisionsPerROF / nNonEmptyROFs : 0.,
                         peakCollisionsPerROF);
    canvCollisionsPerROF->SaveAs("trk_collisions_per_rof.png");
  }

  unsigned int rofIndex = 0;
  unsigned int rofNEntries = 0;

  // LOOP on : ROFRecord array
  for (unsigned int iROF = 0; iROF < ROFRecordArrrayRef.size(); iROF++) {
    std::vector<int> touchedChips;

    rofIndex = ROFRecordArrrayRef[iROF].getFirstEntry();
    rofNEntries = ROFRecordArrrayRef[iROF].getNEntries();

    // LOOP on : digits array
    for (unsigned int iDigit = rofIndex; iDigit < rofIndex + rofNEntries; iDigit++) {
      if (iDigit % 1000 == 0)
        std::cout << "Reading digit " << iDigit << " / " << digArr->size() << std::endl;

      Int_t iDetID = (*digArr)[iDigit].getChipIndex();
      Int_t disk = gman->getDisk(iDetID);
      Int_t subDetID = gman->getSubDetID(iDetID);

      if (subDetID == 1 && disk == -1) {
        if (digitsInCurrentROFPerChip[iDetID] == 0) {
          touchedChips.push_back(iDetID);
        }
        digitsPerChip[iDetID]++;
        ++digitsInCurrentROFPerChip[iDetID];
      }

    } // end loop on digits array

    for (const auto chipID : touchedChips) {
      maxDigitsPerROFPerChip[chipID] = std::max(maxDigitsPerROFPerChip[chipID], digitsInCurrentROFPerChip[chipID]);
      digitsInCurrentROFPerChip[chipID] = 0;
    }

  } // end loop on ROFRecords array

  const double rofNorm = nROFRec > 0 ? 1. / nROFRec : 0.;
  const double bitsToMbps = ReadoutCycleSeconds > 0. ? DigitBits / ReadoutCycleSeconds / 1.e6 : 0.;
  const int nMLOTLayers = gman->getNumberOfLayersMLOT();

  for (int layer = 0; layer < nMLOTLayers; ++layer) {
    int nStaves = gman->extractNumberOfStavesMLOT(layer);
    std::map<int, std::vector<std::pair<double, int>>> chipsPerStave;
    std::vector<int> sensorIdPerChip(gman->getNumberOfChips(), -1);
    int maxSensorsPerStave = 0;

    for (int chipID = 0; chipID < gman->getNumberOfChips(); ++chipID) {
      if (gman->getSubDetID(chipID) != 1 || gman->getLayer(chipID) != layer) {
        continue;
      }
      const int staveID = gman->getStave(chipID);
      const auto sensorCenter = gman->getMatrixL2G(chipID)(o2::math_utils::Point3D<float>(0.f, 0.f, 0.f));
      chipsPerStave[staveID].push_back({sensorCenter.Z(), chipID});
    }

    for (auto& [staveID, chips] : chipsPerStave) {
      std::sort(chips.begin(), chips.end(), [](const auto& left, const auto& right) {
        if (std::abs(left.first - right.first) > 1.e-4) {
          return left.first < right.first;
        }
        return left.second < right.second;
      });

      for (size_t sensorIndex = 0; sensorIndex < chips.size(); ++sensorIndex) {
        sensorIdPerChip[chips[sensorIndex].second] = sensorIndex;
      }

      maxSensorsPerStave = std::max(maxSensorsPerStave, static_cast<int>(chips.size()));
    }

    if (maxSensorsPerStave == 0) {
      continue;
    }

    auto* hDigitsPerROF = new TH2F(Form("h_digits_per_rof_layer%d", layer),
                                   Form("Layer %d average digits per ROF;stave id;sensor id in stave;digits / ROF", layer),
                                   nStaves, -0.5, nStaves - 0.5, maxSensorsPerStave, -0.5, maxSensorsPerStave - 0.5);
    auto* hMaxDigitsPerROF = new TH2F(Form("h_max_digits_per_rof_layer%d", layer),
                                      Form("Layer %d max digits in one ROF;stave id;sensor id in stave;max digits / ROF", layer),
                                      nStaves, -0.5, nStaves - 0.5, maxSensorsPerStave, -0.5, maxSensorsPerStave - 0.5);
    auto* hBandwidth = new TH2F(Form("h_bandwidth_layer%d", layer),
                                Form("Layer %d bandwidth map;stave id;sensor id in stave;bandwidth (Mbit/s)", layer),
                                nStaves, -0.5, nStaves - 0.5, maxSensorsPerStave, -0.5, maxSensorsPerStave - 0.5);
    double totalAvgDigitsPerROF = 0.;
    double totalMaxDigitsPerROF = 0.;
    double totalBandwidthMbps = 0.;
    double peakAvgDigitsPerROF = 0.;
    double peakMaxDigitsPerROF = 0.;
    double peakBandwidthMbps = 0.;
    int nFilledSensors = 0;

    for (int chipID = 0; chipID < gman->getNumberOfChips(); ++chipID) {
      if (gman->getSubDetID(chipID) != 1 || gman->getLayer(chipID) != layer) {
        continue;
      }

      const int staveID = gman->getStave(chipID);
      const int sensorID = sensorIdPerChip[chipID];
      const double avgDigitsPerROF = digitsPerChip[chipID] * rofNorm;
      const double maxDigitsPerROF = maxDigitsPerROFPerChip[chipID];
      const double bandwidthMbps = avgDigitsPerROF * bitsToMbps;

      if (sensorID >= 0) {
        hDigitsPerROF->Fill(staveID, sensorID, avgDigitsPerROF);
        hMaxDigitsPerROF->Fill(staveID, sensorID, maxDigitsPerROF);
        hBandwidth->Fill(staveID, sensorID, bandwidthMbps);
        totalAvgDigitsPerROF += avgDigitsPerROF;
        totalMaxDigitsPerROF += maxDigitsPerROF;
        totalBandwidthMbps += bandwidthMbps;
        peakAvgDigitsPerROF = std::max(peakAvgDigitsPerROF, avgDigitsPerROF);
        peakMaxDigitsPerROF = std::max(peakMaxDigitsPerROF, maxDigitsPerROF);
        peakBandwidthMbps = std::max(peakBandwidthMbps, bandwidthMbps);
        ++nFilledSensors;
      }
    }

    auto* canvLayer = new TCanvas(Form("canvBandwidthLayer%d", layer), Form("Layer %d bandwidth", layer), 1050, 1050);
    canvLayer->SetTopMargin(0.08);
    canvLayer->SetRightMargin(0.18);
    const double avgDigitsPerROFLayer = nFilledSensors > 0 ? totalAvgDigitsPerROF / nFilledSensors : 0.;
    const double avgMaxDigitsPerROFLayer = nFilledSensors > 0 ? totalMaxDigitsPerROF / nFilledSensors : 0.;
    const double avgBandwidthMbps = nFilledSensors > 0 ? totalBandwidthMbps / nFilledSensors : 0.;
    hBandwidth->GetZaxis()->SetRangeUser(0., avgBandwidthMbps > 0. ? 3. * avgBandwidthMbps : 1.);
    hBandwidth->Draw("colz");
    drawSummary(avgBandwidthMbps, peakBandwidthMbps, "Mbit/s");
    canvLayer->SaveAs(Form("trk_layer%d_bandwidth_map.png", layer));

    auto* canvLayerDigits = new TCanvas(Form("canvDigitsLayer%d", layer), Form("Layer %d digits per ROF", layer), 1050, 1050);
    canvLayerDigits->SetTopMargin(0.08);
    canvLayerDigits->SetRightMargin(0.18);
    hDigitsPerROF->Draw("colz");
    drawSummary(avgDigitsPerROFLayer, peakAvgDigitsPerROF, "digits/ROF");
    canvLayerDigits->SaveAs(Form("trk_layer%d_digits_per_rof_map.png", layer));

    auto* canvLayerMaxDigits = new TCanvas(Form("canvMaxDigitsLayer%d", layer), Form("Layer %d max digits per ROF", layer), 1050, 1050);
    canvLayerMaxDigits->SetTopMargin(0.08);
    canvLayerMaxDigits->SetRightMargin(0.18);
    hMaxDigitsPerROF->Draw("colz");
    drawSummary(avgMaxDigitsPerROFLayer, peakMaxDigitsPerROF, "digits/ROF");
    canvLayerMaxDigits->SaveAs(Form("trk_layer%d_max_digits_per_rof_map.png", layer));
  }

  f->Write();
  f->Close();
}
