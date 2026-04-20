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

/// \file CheckBandwidth.C
/// \brief Simple macro to check TRK bandwidth

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <TCanvas.h>
#include <TFile.h>
#include <TH1D.h>
#include <TH2F.h>
#include <TPaveText.h>
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
constexpr double DigitBits = 24.;
constexpr double BunchCrossingNS = 25.;
constexpr int ReadoutCycleBC = 18;
constexpr int ReadoutCycleSimBC = 18;
constexpr double ReadoutCycleSeconds = ReadoutCycleBC * BunchCrossingNS * 1.e-9;
} // namespace

void CheckBandwidth(std::string digifile = "trkdigits.root", std::string inputGeom = "o2sim_geometry.root", std::string collContextFile = "collisioncontext.root")
{
  gStyle->SetPalette(55);
  gStyle->SetOptStat(0);

  // --- Drawing helpers ---

  auto drawSummary = [](double averageValue, double peakValue, const char* unit) {
    TLatex latex;
    latex.SetNDC();
    latex.SetTextSize(0.03);
    latex.SetTextAlign(13);
    latex.DrawLatex(0.04, 0.06, Form("avg: %.3f %s", averageValue, unit));
    latex.DrawLatex(0.04, 0.03, Form("peak: %.3f %s", peakValue, unit));
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

  auto drawCollisionInfoBox = [](double averageValue) {
    const double effectiveIRRateHz = ReadoutCycleSeconds > 0. ? averageValue / ReadoutCycleSeconds : 0.;
    TPaveText infoBox(0.55, 0.79, 0.88, 0.9, "NDC");
    infoBox.SetFillColor(0);
    infoBox.SetBorderSize(1);
    infoBox.SetTextAlign(12);
    infoBox.SetTextSize(0.028);
    infoBox.AddText(Form("effective IR: %.3f MHz", effectiveIRRateHz * 1.e-6));
    infoBox.AddText(Form("ROF length: %d BC", ReadoutCycleBC));
    infoBox.DrawClone();
  };

  const TString outputPdf = "trk_bandwidth_report.pdf";
  bool pdfOpened = false;
  TCanvas* lastPdfCanvas = nullptr;
  auto appendCanvasToPdf = [&](TCanvas* canvas) {
    if (!pdfOpened) {
      canvas->Print(Form("%s[", outputPdf.Data()));
      pdfOpened = true;
    }
    canvas->Print(outputPdf.Data());
    lastPdfCanvas = canvas;
  };

  using namespace o2::base;
  using namespace o2::trk;

  TFile* f = TFile::Open("CheckBandwidth.root", "recreate");

  // --- Geometry ---

  o2::base::GeometryManager::loadGeometry(inputGeom);
  auto* gman = o2::trk::GeometryTGeo::Instance();
  gman->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::L2G));

  const int nVDPetals = gman->extractNumberOfPetalsVD();
  const int nVDLayers = gman->extractNumberOfLayersVD();
  const int nMLOTLayers = gman->getNumberOfLayersMLOT();
  const int nTotalLayers = nVDLayers + nMLOTLayers;
  const int nChips = gman->getNumberOfChips();

  // Precompute per-chip geometry — centralises all gman queries.
  // globalLayer maps VD layers to [0, nVDLayers) and MLOT layers to [nVDLayers, nTotalLayers).
  // disk == -1 identifies barrel chips (the only ones that produce digits in this detector).
  struct ChipGeom {
    int subDetID = -1, localLayer = -1, globalLayer = -1, disk = -1;
    int stave = -1, halfStave = -1, petal = -1;
  };
  std::vector<ChipGeom> chipGeom(nChips);
  for (int chipID = 0; chipID < nChips; ++chipID) {
    auto& g = chipGeom[chipID];
    g.subDetID = gman->getSubDetID(chipID);
    g.localLayer = gman->getLayer(chipID);
    g.disk = gman->getDisk(chipID);
    g.globalLayer = g.localLayer + g.subDetID * nVDLayers;
    g.stave = gman->getStave(chipID);
    g.halfStave = std::max(0, gman->getHalfStave(chipID));
    g.petal = (g.subDetID == 0) ? gman->getPetalCase(chipID) : -1;
  }

  // Number of barrel chips per global layer (used for per-layer bandwidth normalisation).
  std::vector<unsigned int> chipsPerLayer(nTotalLayers, 0u);
  for (int chipID = 0; chipID < nChips; ++chipID) {
    const auto& g = chipGeom[chipID];
    if (g.disk != -1 || g.globalLayer < 0 || g.globalLayer >= nTotalLayers) {
      continue;
    }
    ++chipsPerLayer[g.globalLayer];
  }

  // MLOT sensor index within its half-stave, ordered by Z position.
  // Precomputed here so the plotting loop only reads results.
  std::vector<int> chipSensorIndex(nChips, -1);
  std::vector<int> maxSensorsPerHalfStaveMLOT(nMLOTLayers, 0);
  for (int layer = 0; layer < nMLOTLayers; ++layer) {
    std::map<std::pair<int, int>, std::vector<std::pair<double, int>>> chipsPerHalfStave;
    for (int chipID = 0; chipID < nChips; ++chipID) {
      const auto& g = chipGeom[chipID];
      if (g.subDetID != 1 || g.localLayer != layer || g.disk != -1) {
        continue;
      }
      const auto center = gman->getMatrixL2G(chipID)(o2::math_utils::Point3D<float>(0.f, 0.f, 0.f));
      chipsPerHalfStave[{g.stave, g.halfStave}].push_back({center.Z(), chipID});
    }
    for (auto& [key, chips] : chipsPerHalfStave) {
      std::sort(chips.begin(), chips.end(), [](const auto& a, const auto& b) {
        return std::abs(a.first - b.first) > 1.e-4 ? a.first < b.first : a.second < b.second;
      });
      for (size_t i = 0; i < chips.size(); ++i) {
        chipSensorIndex[chips[i].second] = (int)i;
      }
      maxSensorsPerHalfStaveMLOT[layer] = std::max(maxSensorsPerHalfStaveMLOT[layer], (int)chips.size());
    }
  }

  // --- Collision context ---

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

  // --- Digits ---

  TFile* digFile = TFile::Open(digifile.data());
  TTree* digTree = (TTree*)digFile->Get("o2sim");
  const int nDigitTreeEntries = digTree->GetEntries();

  std::vector<o2::itsmft::Digit>* digArr = nullptr;
  std::vector<o2::itsmft::ROFRecord>* rofRecords = nullptr;
  digTree->SetBranchAddress("TRKDigit", &digArr);
  digTree->SetBranchAddress("TRKDigitROF", &rofRecords);

  digTree->GetEntry(0);
  if (nDigitTreeEntries > 1) {
    LOG(warning) << "Digit tree has " << nDigitTreeEntries << " entries, but this macro processes entry 0 only.";
  }

  const int nROFRec = (int)rofRecords->size();
  if (nROFRec != (int)collisionsPerROF.size()) {
    LOG(fatal) << "Mismatch between number of ROF records in digit tree (" << nROFRec
               << ") and number of ROFs computed from collisioncontext.root (" << collisionsPerROF.size()
               << "). Check input files.";
  }

  // --- Accumulate per-chip digit counts across all ROFs ---

  const double rofNorm = nROFRec > 0 ? 1. / nROFRec : 0.;
  const double bitsToGbps = ReadoutCycleSeconds > 0. ? DigitBits / ReadoutCycleSeconds / 1.e9 : 0.;

  std::vector<unsigned long long> digitsPerChip(nChips, 0ull);
  std::vector<unsigned int> maxDigitsPerROFPerChip(nChips, 0u);
  std::vector<unsigned int> digitsInCurrentROFPerChip(nChips, 0u);

  for (unsigned int iROF = 0; iROF < rofRecords->size(); ++iROF) {
    std::vector<int> touchedChips;
    const unsigned int rofStart = (*rofRecords)[iROF].getFirstEntry();
    const unsigned int rofEnd = rofStart + (*rofRecords)[iROF].getNEntries();

    for (unsigned int iDigit = rofStart; iDigit < rofEnd; ++iDigit) {
      if (iDigit % 1000 == 0) {
        std::cout << "Reading digit " << iDigit << " / " << digArr->size() << "\r" << std::flush;
      }
      const int chipID = (*digArr)[iDigit].getChipIndex();
      if (chipGeom[chipID].disk != -1) {
        continue;
      }
      if (digitsInCurrentROFPerChip[chipID] == 0) {
        touchedChips.push_back(chipID);
      }
      ++digitsPerChip[chipID];
      ++digitsInCurrentROFPerChip[chipID];
    }

    for (const int chipID : touchedChips) {
      maxDigitsPerROFPerChip[chipID] = std::max(maxDigitsPerROFPerChip[chipID], digitsInCurrentROFPerChip[chipID]);
      digitsInCurrentROFPerChip[chipID] = 0;
    }
  }

  // --- Per-layer bandwidth distribution histograms (second scan over digits) ---

  // Per-layer peak digit count (from per-chip maxima) — drives histogram binning.
  std::vector<unsigned int> maxDigitsPerLayer(nTotalLayers, 0u);
  for (int chipID = 0; chipID < nChips; ++chipID) {
    const auto& g = chipGeom[chipID];
    if (g.disk != -1 || g.globalLayer < 0 || g.globalLayer >= nTotalLayers) {
      continue;
    }
    maxDigitsPerLayer[g.globalLayer] = std::max(maxDigitsPerLayer[g.globalLayer], maxDigitsPerROFPerChip[chipID]);
  }

  std::vector<TH1D*> hDigitsDistPerLayer(nTotalLayers, nullptr);
  for (int l = 0; l < nTotalLayers; ++l) {
    if (chipsPerLayer[l] == 0 || maxDigitsPerLayer[l] == 0) {
      continue;
    }
    const int nBins = std::min((int)maxDigitsPerLayer[l] + 1, 200);
    hDigitsDistPerLayer[l] = new TH1D(Form("h_digits_dist_layer%d", l),
                                      Form("Layer %d;Fired pixels / ROF / chip;Probability", l),
                                      nBins, -0.5, (double)maxDigitsPerLayer[l] + 0.5);
  }
  // digitsInCurrentROFPerChip is all zeros after the first scan — reuse it here.
  {
    std::vector<int> touchedChips;
    for (unsigned int iROF = 0; iROF < rofRecords->size(); ++iROF) {
      touchedChips.clear();
      const unsigned int rofStart = (*rofRecords)[iROF].getFirstEntry();
      const unsigned int rofEnd = rofStart + (*rofRecords)[iROF].getNEntries();
      for (unsigned int iDigit = rofStart; iDigit < rofEnd; ++iDigit) {
        const int chipID = (*digArr)[iDigit].getChipIndex();
        if (chipGeom[chipID].disk != -1) {
          continue;
        }
        if (digitsInCurrentROFPerChip[chipID] == 0) {
          touchedChips.push_back(chipID);
        }
        ++digitsInCurrentROFPerChip[chipID];
      }
      for (const int chipID : touchedChips) {
        const int l = chipGeom[chipID].globalLayer;
        if (hDigitsDistPerLayer[l]) {
          hDigitsDistPerLayer[l]->Fill(digitsInCurrentROFPerChip[chipID]);
        }
        digitsInCurrentROFPerChip[chipID] = 0;
      }
    }
  }

  // --- Per-layer bandwidth statistics, normalised by chips per layer ---
  //
  // avgDigitsPerROF     : mean over chips of (total chip digits / nROFs)
  // peakAvgDigitsPerROF : max  over chips of (total chip digits / nROFs)
  // avgMaxDigitsPerROF  : mean over chips of (peak single-ROF digit count)
  // peakMaxDigitsPerROF : max  over chips of (peak single-ROF digit count)
  // avg/peakBandwidthGbps derived from the avg/peak digit quantities above.

  struct LayerStats {
    double avgDigitsPerROF = 0.;
    double peakAvgDigitsPerROF = 0.;
    double avgMaxDigitsPerROF = 0.;
    double peakMaxDigitsPerROF = 0.;
    double avgBandwidthGbps = 0.;
    double peakBandwidthGbps = 0.;
  };
  std::vector<LayerStats> layerStats(nTotalLayers);

  for (int chipID = 0; chipID < nChips; ++chipID) {
    const auto& g = chipGeom[chipID];
    if (g.disk != -1 || g.globalLayer < 0 || g.globalLayer >= nTotalLayers) {
      continue;
    }
    const int l = g.globalLayer;
    const double avgDigits = digitsPerChip[chipID] * rofNorm;
    const double maxDigits = (double)maxDigitsPerROFPerChip[chipID];
    layerStats[l].avgDigitsPerROF += avgDigits;
    layerStats[l].avgMaxDigitsPerROF += maxDigits;
    layerStats[l].peakAvgDigitsPerROF = std::max(layerStats[l].peakAvgDigitsPerROF, avgDigits);
    layerStats[l].peakMaxDigitsPerROF = std::max(layerStats[l].peakMaxDigitsPerROF, maxDigits);
  }
  for (int l = 0; l < nTotalLayers; ++l) {
    if (chipsPerLayer[l] > 0) {
      const double norm = 1. / chipsPerLayer[l];
      layerStats[l].avgDigitsPerROF *= norm;
      layerStats[l].avgMaxDigitsPerROF *= norm;
    }
    layerStats[l].avgBandwidthGbps = layerStats[l].avgDigitsPerROF * bitsToGbps;
    layerStats[l].peakBandwidthGbps = layerStats[l].peakAvgDigitsPerROF * bitsToGbps;
  }

  // --- Collision plots ---

  if (nROFRec > 0) {
    auto* hCollisionsPerROF = new TH1D("h_collisions_per_rof", "Collisions per ROF;ROF id;N collisions",
                                       nROFRec, -0.5, nROFRec - 0.5);
    double totalCollisionsPerROF = 0.;
    double peakCollisionsPerROF = 0.;
    int nNonEmptyROFs = 0;

    for (int rofID = 0; rofID < nROFRec; ++rofID) {
      const double nColl = collisionsPerROF[rofID];
      hCollisionsPerROF->SetBinContent(rofID + 1, nColl);
      totalCollisionsPerROF += nColl;
      peakCollisionsPerROF = std::max(peakCollisionsPerROF, nColl);
      if (nColl > 0.) {
        ++nNonEmptyROFs;
      }
    }

    const double avgCollisionsPerROF = totalCollisionsPerROF / nROFRec;
    auto* canvCollisionsPerROF = new TCanvas("canvCollisionsPerROF", "Collisions per ROF", 1050, 1050);
    canvCollisionsPerROF->SetTopMargin(0.08);
    hCollisionsPerROF->Draw("hist");
    drawCollisionSummary(avgCollisionsPerROF,
                         nNonEmptyROFs > 0 ? totalCollisionsPerROF / nNonEmptyROFs : 0.,
                         peakCollisionsPerROF);
    drawCollisionInfoBox(avgCollisionsPerROF);
    appendCanvasToPdf(canvCollisionsPerROF);
  }

  // --- VD plots ---

  auto* hVDDigitsPerROF = new TH2F("h_digits_per_rof_vd",
                                   "VD average digits per ROF;petal id;layer id;digits / ROF",
                                   nVDPetals, -0.5, nVDPetals - 0.5, nVDLayers, -0.5, nVDLayers - 0.5);
  auto* hVDMaxDigitsPerROF = new TH2F("h_max_digits_per_rof_vd",
                                      "VD max digits in one ROF;petal id;layer id;max digits / ROF",
                                      nVDPetals, -0.5, nVDPetals - 0.5, nVDLayers, -0.5, nVDLayers - 0.5);
  auto* hVDBandwidth = new TH2F("h_bandwidth_vd",
                                "VD bandwidth map;petal id;layer id;bandwidth (Gbit/s)",
                                nVDPetals, -0.5, nVDPetals - 0.5, nVDLayers, -0.5, nVDLayers - 0.5);

  for (auto* hist : {hVDDigitsPerROF, hVDMaxDigitsPerROF, hVDBandwidth}) {
    for (int petalID = 0; petalID < nVDPetals; ++petalID) {
      hist->GetXaxis()->SetBinLabel(petalID + 1, Form("%d", petalID));
    }
    for (int layerID = 0; layerID < nVDLayers; ++layerID) {
      hist->GetYaxis()->SetBinLabel(layerID + 1, Form("%d", layerID));
    }
    hist->GetXaxis()->SetNdivisions(0, kFALSE);
    hist->GetYaxis()->SetNdivisions(0, kFALSE);
    hist->LabelsOption("h", "X");
    hist->LabelsOption("h", "Y");
  }

  double totalVDAvgDigits = 0., peakVDAvgDigits = 0.;
  double totalVDMaxDigits = 0., peakVDMaxDigits = 0.;
  double totalVDBandwidth = 0., peakVDBandwidth = 0.;

  for (int chipID = 0; chipID < nChips; ++chipID) {
    const auto& g = chipGeom[chipID];
    if (g.subDetID != 0 || g.disk != -1 || g.localLayer < 0 || g.localLayer >= nVDLayers) {
      continue;
    }
    if (g.petal < 0 || g.petal >= nVDPetals) {
      continue;
    }
    const double avgDigits = digitsPerChip[chipID] * rofNorm;
    const double maxDigits = (double)maxDigitsPerROFPerChip[chipID];
    const double bandwidth = avgDigits * bitsToGbps;

    hVDDigitsPerROF->SetBinContent(g.petal + 1, g.localLayer + 1, avgDigits);
    hVDMaxDigitsPerROF->SetBinContent(g.petal + 1, g.localLayer + 1, maxDigits);
    hVDBandwidth->SetBinContent(g.petal + 1, g.localLayer + 1, bandwidth);

    totalVDAvgDigits += avgDigits;
    totalVDMaxDigits += maxDigits;
    totalVDBandwidth += bandwidth;
    peakVDAvgDigits = std::max(peakVDAvgDigits, avgDigits);
    peakVDMaxDigits = std::max(peakVDMaxDigits, maxDigits);
    peakVDBandwidth = std::max(peakVDBandwidth, bandwidth);
  }

  const int nVDBarrelChips = std::accumulate(chipsPerLayer.begin(), chipsPerLayer.begin() + nVDLayers, 0);
  const double normVD = nVDBarrelChips > 0 ? 1. / nVDBarrelChips : 0.;
  const double avgVDAvgDigits = totalVDAvgDigits * normVD;
  const double avgVDMaxDigits = totalVDMaxDigits * normVD;
  const double avgVDBandwidth = totalVDBandwidth * normVD;

  auto* canvVDBandwidth = new TCanvas("canvBandwidthVD", "VD bandwidth", 1050, 1050);
  canvVDBandwidth->SetTopMargin(0.08);
  canvVDBandwidth->SetRightMargin(0.18);
  hVDBandwidth->GetZaxis()->SetRangeUser(0., avgVDBandwidth > 0. ? 3. * avgVDBandwidth : 1.);
  hVDBandwidth->SetMarkerSize(1.8);
  hVDBandwidth->Draw("colz text");
  drawSummary(avgVDBandwidth, peakVDBandwidth, "Gbit/s");
  appendCanvasToPdf(canvVDBandwidth);

  auto* canvVDDigits = new TCanvas("canvDigitsVD", "VD digits per ROF", 1050, 1050);
  canvVDDigits->SetTopMargin(0.08);
  canvVDDigits->SetRightMargin(0.18);
  hVDDigitsPerROF->SetMarkerSize(1.8);
  hVDDigitsPerROF->Draw("colz text");
  drawSummary(avgVDAvgDigits, peakVDAvgDigits, "digits/ROF");
  appendCanvasToPdf(canvVDDigits);

  auto* canvVDMaxDigits = new TCanvas("canvMaxDigitsVD", "VD max digits per ROF", 1050, 1050);
  canvVDMaxDigits->SetTopMargin(0.08);
  canvVDMaxDigits->SetRightMargin(0.18);
  hVDMaxDigitsPerROF->SetMarkerSize(1.8);
  hVDMaxDigitsPerROF->Draw("colz text");
  drawSummary(avgVDMaxDigits, peakVDMaxDigits, "digits/ROF");
  appendCanvasToPdf(canvVDMaxDigits);

  // --- MLOT per-layer plots ---

  for (int layer = 0; layer < nMLOTLayers; ++layer) {
    if (maxSensorsPerHalfStaveMLOT[layer] == 0) {
      continue;
    }
    const int outputLayer = nVDLayers + layer;
    const int nStaves = gman->extractNumberOfStavesMLOT(layer);
    const int nHalfStaves = std::max(1, gman->getNumberOfHalfStaves(layer));
    const int maxSensors = maxSensorsPerHalfStaveMLOT[layer];

    auto* hDigitsPerROF = new TH2F(Form("h_digits_per_rof_layer%d", outputLayer),
                                   Form("Layer %d average digits per ROF;stave id / half-stave;sensor id in half-stave;digits / ROF", outputLayer),
                                   nStaves * nHalfStaves, -0.5, nStaves - 0.5, maxSensors, -0.5, maxSensors - 0.5);
    auto* hMaxDigitsPerROF = new TH2F(Form("h_max_digits_per_rof_layer%d", outputLayer),
                                      Form("Layer %d max digits in one ROF;stave id / half-stave;sensor id in half-stave;max digits / ROF", outputLayer),
                                      nStaves * nHalfStaves, -0.5, nStaves - 0.5, maxSensors, -0.5, maxSensors - 0.5);
    auto* hBandwidth = new TH2F(Form("h_bandwidth_layer%d", outputLayer),
                                Form("Layer %d bandwidth map;stave id / half-stave;sensor id in half-stave;bandwidth (Gbit/s)", outputLayer),
                                nStaves * nHalfStaves, -0.5, nStaves - 0.5, maxSensors, -0.5, maxSensors - 0.5);

    for (int chipID = 0; chipID < nChips; ++chipID) {
      const auto& g = chipGeom[chipID];
      if (g.subDetID != 1 || g.localLayer != layer || g.disk != -1) {
        continue;
      }
      const int sensorID = chipSensorIndex[chipID];
      if (sensorID < 0) {
        continue;
      }
      const double staveBinX = g.stave + (g.halfStave + 0.5) / nHalfStaves - 0.5;
      const double avgDigits = digitsPerChip[chipID] * rofNorm;
      const double maxDigits = (double)maxDigitsPerROFPerChip[chipID];

      hDigitsPerROF->Fill(staveBinX, sensorID, avgDigits);
      hMaxDigitsPerROF->Fill(staveBinX, sensorID, maxDigits);
      hBandwidth->Fill(staveBinX, sensorID, avgDigits * bitsToGbps);
    }

    const auto& ls = layerStats[outputLayer];

    auto* canvLayer = new TCanvas(Form("canvBandwidthLayer%d", outputLayer), Form("Layer %d bandwidth", outputLayer), 1050, 1050);
    canvLayer->SetTopMargin(0.08);
    canvLayer->SetRightMargin(0.18);
    hBandwidth->GetZaxis()->SetRangeUser(0., ls.avgBandwidthGbps > 0. ? 3. * ls.avgBandwidthGbps : 1.);
    hBandwidth->Draw("colz");
    drawSummary(ls.avgBandwidthGbps, ls.peakBandwidthGbps, "Gbit/s");
    appendCanvasToPdf(canvLayer);

    auto* canvLayerDigits = new TCanvas(Form("canvDigitsLayer%d", outputLayer), Form("Layer %d digits per ROF", outputLayer), 1050, 1050);
    canvLayerDigits->SetTopMargin(0.08);
    canvLayerDigits->SetRightMargin(0.18);
    hDigitsPerROF->Draw("colz");
    drawSummary(ls.avgDigitsPerROF, ls.peakAvgDigitsPerROF, "digits/ROF");
    appendCanvasToPdf(canvLayerDigits);

    auto* canvLayerMaxDigits = new TCanvas(Form("canvMaxDigitsLayer%d", outputLayer), Form("Layer %d max digits per ROF", outputLayer), 1050, 1050);
    canvLayerMaxDigits->SetTopMargin(0.08);
    canvLayerMaxDigits->SetRightMargin(0.18);
    hMaxDigitsPerROF->Draw("colz");
    drawSummary(ls.avgMaxDigitsPerROF, ls.peakMaxDigitsPerROF, "digits/ROF");
    appendCanvasToPdf(canvLayerMaxDigits);
  }

  // --- Digits distribution per layer ---
  // Each histogram shows the distribution of total-layer bandwidth across ROFs.

  {
    const int nCols = std::max(1, (int)std::ceil(std::sqrt((double)nTotalLayers)));
    const int nRows = (nTotalLayers + nCols - 1) / nCols;
    auto* canvBwDist = new TCanvas("canvDigitsDistPerLayer", "Digits distribution per layer", 350 * nCols, 300 * nRows);
    canvBwDist->Divide(nCols, nRows);
    for (int layer = 0; layer < nTotalLayers; ++layer) {
      if (!hDigitsDistPerLayer[layer]) {
        continue;
      }
      canvBwDist->cd(layer + 1);
      gPad->SetLogy();
      gPad->SetTopMargin(0.10);
      gPad->SetBottomMargin(0.14);
      gPad->SetLeftMargin(0.14);
      hDigitsDistPerLayer[layer]->Scale(1. / hDigitsDistPerLayer[layer]->GetEntries());
      hDigitsDistPerLayer[layer]->Draw("hist");
    }
    appendCanvasToPdf(canvBwDist);
  }

  // --- Summary: bandwidth vs layer ---

  auto* hAvgBandwidthVsLayer = new TH1D("h_avg_bandwidth_vs_layer",
                                        "Average bandwidth by layer;layer id;average bandwidth (Gbit/s)",
                                        nTotalLayers, -0.5, nTotalLayers - 0.5);
  auto* hPeakBandwidthVsLayer = new TH1D("h_peak_bandwidth_vs_layer",
                                         "Peak bandwidth by layer;layer id;peak bandwidth (Gbit/s)",
                                         nTotalLayers, -0.5, nTotalLayers - 0.5);
  for (int layer = 0; layer < nTotalLayers; ++layer) {
    hAvgBandwidthVsLayer->SetBinContent(layer + 1, layerStats[layer].avgBandwidthGbps);
    hPeakBandwidthVsLayer->SetBinContent(layer + 1, layerStats[layer].peakBandwidthGbps);
  }

  auto* canvBandwidthSummary = new TCanvas("canvBandwidthSummary", "Bandwidth summary by layer", 1050, 1050);
  gStyle->SetOptTitle(0);
  canvBandwidthSummary->cd();
  canvBandwidthSummary->SetTopMargin(0.08);
  canvBandwidthSummary->SetBottomMargin(0.14);
  canvBandwidthSummary->SetLogy();
  hAvgBandwidthVsLayer->SetTitle("Average bandwidth by layer;layer id;Bandwidth (Gbit/s)");
  hAvgBandwidthVsLayer->Draw("hist");
  hPeakBandwidthVsLayer->SetLineColor(kRed);
  hPeakBandwidthVsLayer->Draw("hist same");
  canvBandwidthSummary->BuildLegend(0.6, 0.75, 0.9, 0.9);
  appendCanvasToPdf(canvBandwidthSummary);

  if (lastPdfCanvas != nullptr) {
    lastPdfCanvas->Print(Form("%s]", outputPdf.Data()));
  }

  f->Write();
  f->Close();
}
