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
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <TCanvas.h>
#include <TF1.h>
#include <TH1.h>
#include <TH1D.h>
#include <TFile.h>
#include <TLegend.h>
#include <TPad.h>
#include <TPaveText.h>
#include <TStyle.h>
#include <TTree.h>

#include "DataFormatsITS/Vertex.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTrack.h"
#include "SimulationDataFormat/O2DatabasePDG.h"
#include "Steer/MCKinematicsReader.h"
#endif

constexpr const char* tracFile = "o2trac_its.root";
constexpr const char* collContextFile = "collisioncontext.root";

namespace
{
namespace fs = std::filesystem;

constexpr float MinPt = 0.05f;
constexpr float MaxEta = 1.1f;
constexpr int NMultiplicityBins = 11;
constexpr std::array<const char*, NMultiplicityBins> MultiplicityLabels{{"2", "3", "4-5", "6-8", "9-13", "14-21", "22-33", "34-52", "53-83", "84-128", "128+"}};

struct TruthInfo {
  int multiplicity = 0;
  float x = 0.f;
  float y = 0.f;
  float z = 0.f;
};

struct BestRecoInfo {
  o2::its::Vertex vertex;
  float purity = -1.f;
};

struct GaussianSummary {
  bool valid = false;
  double mean = 0.;
  double sigma = 0.;
};

bool isTrueVertexLabel(const o2::MCCompLabel& label)
{
  return label.isValid() && !label.isFake() && label.getSourceID() == 0;
}

bool isChargedPrimary(const o2::MCTrack& track)
{
  if (!track.isPrimary() || track.GetPt() < MinPt || std::abs(track.GetEta()) > MaxEta) {
    return false;
  }
  auto* pdg = o2::O2DatabasePDG::Instance()->GetParticle(track.GetPdgCode());
  return pdg != nullptr && pdg->Charge() != 0.;
}

bool isBetterReco(const o2::its::Vertex& candidate, float candidatePurity, const o2::its::Vertex& current, float currentPurity)
{
  if (candidatePurity != currentPurity) {
    return candidatePurity > currentPurity;
  }
  if (candidate.getNContributors() != current.getNContributors()) {
    return candidate.getNContributors() > current.getNContributors();
  }
  return candidate.getChi2() < current.getChi2();
}

int getMultiplicityCategory(int multiplicity)
{
  if (multiplicity <= 2) {
    return 1;
  }
  if (multiplicity <= 3) {
    return 2;
  }
  if (multiplicity <= 5) {
    return 3;
  }
  if (multiplicity <= 8) {
    return 4;
  }
  if (multiplicity <= 13) {
    return 5;
  }
  if (multiplicity <= 21) {
    return 6;
  }
  if (multiplicity <= 33) {
    return 7;
  }
  if (multiplicity <= 52) {
    return 8;
  }
  if (multiplicity <= 83) {
    return 9;
  }
  if (multiplicity <= 128) {
    return 10;
  }
  return 11;
}

void fillMultiplicityHistogram(TH1* hist, int multiplicity)
{
  hist->Fill(getMultiplicityCategory(multiplicity));
}

GaussianSummary fitGaussianCore(TH1* hist, const char* funcName)
{
  if (hist == nullptr || hist->GetEntries() < 20) {
    return {};
  }
  const auto rms = hist->GetRMS();
  if (!(rms > 0.)) {
    return {};
  }

  TF1 fit(funcName, "gaus", hist->GetMean() - 2. * rms, hist->GetMean() + 2. * rms);
  fit.SetParameters(hist->GetMaximum(), hist->GetMean(), rms);
  hist->Fit(&fit, "Q0R");

  const auto mean = fit.GetParameter(1);
  const auto sigma = std::abs(fit.GetParameter(2));
  if (!(sigma > 0.)) {
    return {};
  }

  fit.SetRange(mean - 2. * sigma, mean + 2. * sigma);
  hist->Fit(&fit, "Q0R");
  return {true, fit.GetParameter(1), std::abs(fit.GetParameter(2))};
}

TH1D* makeNormalizedCopy(const TH1D* source, const char* name, const char* title)
{
  auto* copy = static_cast<TH1D*>(source->Clone(name));
  copy->SetTitle(title);
  const auto integral = copy->Integral("width");
  if (integral > 0.) {
    copy->Scale(1. / integral);
  }
  return copy;
}

void setMultiplicityBinLabels(TH1* hist)
{
  for (int i = 0; i < NMultiplicityBins; ++i) {
    hist->GetXaxis()->SetBinLabel(i + 1, MultiplicityLabels[i]);
  }
}

void setHistogramStyle(TH1* hist, int color, int marker)
{
  hist->SetLineColor(color);
  hist->SetMarkerColor(color);
  hist->SetMarkerStyle(marker);
  hist->SetLineWidth(2);
}

void printGaussianByMultiplicity(const std::array<GaussianSummary, NMultiplicityBins>& summaries, const char* title)
{
  std::printf("%s:\n", title);
  for (int i = 0; i < NMultiplicityBins; ++i) {
    if (summaries[i].valid) {
      std::printf("  %-4s : mean=%.6g sigma=%.6g\n", MultiplicityLabels[i], summaries[i].mean, summaries[i].sigma);
    } else {
      std::printf("  %-4s : n/a\n", MultiplicityLabels[i]);
    }
  }
}

fs::path resolveContextFile(const fs::path& dir)
{
  const std::array<fs::path, 3> candidates{
    dir / collContextFile,
    dir.parent_path() / collContextFile,
    fs::current_path() / collContextFile};
  for (const auto& candidate : candidates) {
    if (!candidate.empty() && fs::exists(candidate) && fs::is_regular_file(candidate)) {
      return candidate;
    }
  }
  return {};
}

void printBinnedFractions(const TH1* numerator, const TH1* denominator, const char* title)
{
  if (numerator == nullptr || denominator == nullptr) {
    return;
  }
  std::printf("%s:\n", title);
  for (int iBin = 1; iBin <= denominator->GetNbinsX(); ++iBin) {
    const auto den = denominator->GetBinContent(iBin);
    const auto num = numerator->GetBinContent(iBin);
    const auto value = den > 0. ? num / den : 0.;
    std::printf("  %-4s : %.4f (%g / %g)\n", denominator->GetXaxis()->GetBinLabel(iBin), value, num, den);
  }
}

std::vector<fs::path> findDirs(const std::string& roots)
{
  fs::path root = roots.empty() ? fs::current_path() : fs::path{roots};
  std::vector<fs::path> result;
  const auto hasFiles = [](const fs::path& dir) {
    const auto tracPath = dir / tracFile;
    return fs::exists(tracPath) && fs::is_regular_file(tracPath);
  };

  if (fs::is_directory(root) && hasFiles(root)) {
    result.push_back(root);
    return result;
  }

  for (const auto& entry : fs::recursive_directory_iterator(root)) {
    if (entry.is_directory() && hasFiles(entry.path())) {
      result.push_back(entry.path());
    }
  }
  std::sort(result.begin(), result.end());
  return result;
}

} // namespace

void CheckSeeding(const std::string& dir = "")
{
  using Vertex = o2::its::Vertex;
  const auto cwd = fs::current_path();
  gStyle->SetOptStat(0);
  TH1::AddDirectory(kFALSE);

  auto dirs = findDirs(dir);
  std::printf("Will iterate over %zu input dirs\n", dirs.size());
  if (dirs.empty()) {
    std::printf("No input directories containing %s were found.\n", tracFile);
    return;
  }

  auto* hTruthMultiplicityFindable = new TH1D("hTruthMultiplicityFindable",
                                              "Findable truth vertices;truth multiplicity bin;vertices",
                                              NMultiplicityBins, 0.5, NMultiplicityBins + 0.5);
  auto* hTruthMultiplicityFound = new TH1D("hTruthMultiplicityFound",
                                           "Found truth vertices;truth multiplicity bin;vertices",
                                           NMultiplicityBins, 0.5, NMultiplicityBins + 0.5);
  auto* hRecoMultiplicityTrue = new TH1D("hRecoMultiplicityTrue",
                                         "True reconstructed vertices;reco multiplicity bin;vertices",
                                         NMultiplicityBins, 0.5, NMultiplicityBins + 0.5);
  auto* hRecoMultiplicityFake = new TH1D("hRecoMultiplicityFake",
                                         "Fake reconstructed vertices;reco multiplicity bin;vertices",
                                         NMultiplicityBins, 0.5, NMultiplicityBins + 0.5);
  auto* hDx = new TH1D("hDx", "Matched vertex residuals;x_{reco}-x_{MC} (cm);vertices", 400, -0.02, 0.02);
  auto* hDy = new TH1D("hDy", "Matched vertex residuals;y_{reco}-y_{MC} (cm);vertices", 400, -0.02, 0.02);
  auto* hDz = new TH1D("hDz", "Matched vertex residuals;z_{reco}-z_{MC} (cm);vertices", 400, -0.02, 0.02);
  auto* hPullX = new TH1D("hPullX", "Matched vertex pulls;x pull;vertices", 600, -30., 30.);
  auto* hPullY = new TH1D("hPullY", "Matched vertex pulls;y pull;vertices", 600, -30., 30.);
  auto* hPullZ = new TH1D("hPullZ", "Matched vertex pulls;z pull;vertices", 600, -30., 30.);
  std::array<TH1D*, NMultiplicityBins> hPullXByMult{};
  std::array<TH1D*, NMultiplicityBins> hPullYByMult{};
  std::array<TH1D*, NMultiplicityBins> hPullZByMult{};
  for (int i = 0; i < NMultiplicityBins; ++i) {
    const auto nameX = std::string("hPullX_") + std::to_string(i + 1);
    const auto nameY = std::string("hPullY_") + std::to_string(i + 1);
    const auto nameZ = std::string("hPullZ_") + std::to_string(i + 1);
    const auto titleX = std::string("x pull ") + MultiplicityLabels[i] + ";x pull;vertices";
    const auto titleY = std::string("y pull ") + MultiplicityLabels[i] + ";y pull;vertices";
    const auto titleZ = std::string("z pull ") + MultiplicityLabels[i] + ";z pull;vertices";
    hPullXByMult[i] = new TH1D(nameX.c_str(), titleX.c_str(), 600, -30., 30.);
    hPullYByMult[i] = new TH1D(nameY.c_str(), titleY.c_str(), 600, -30., 30.);
    hPullZByMult[i] = new TH1D(nameZ.c_str(), titleZ.c_str(), 600, -30., 30.);
  }

  setMultiplicityBinLabels(hTruthMultiplicityFindable);
  setMultiplicityBinLabels(hTruthMultiplicityFound);
  setMultiplicityBinLabels(hRecoMultiplicityTrue);
  setMultiplicityBinLabels(hRecoMultiplicityFake);
  setHistogramStyle(hTruthMultiplicityFindable, kGray + 2, 20);
  setHistogramStyle(hTruthMultiplicityFound, kAzure + 2, 20);
  setHistogramStyle(hRecoMultiplicityTrue, kAzure + 2, 20);
  setHistogramStyle(hRecoMultiplicityFake, kOrange + 7, 24);
  setHistogramStyle(hDx, kAzure + 2, 20);
  setHistogramStyle(hDy, kGreen + 2, 21);
  setHistogramStyle(hDz, kRed + 1, 24);
  setHistogramStyle(hPullX, kAzure + 2, 20);
  setHistogramStyle(hPullY, kGreen + 2, 21);
  setHistogramStyle(hPullZ, kRed + 1, 24);

  size_t findable = 0;
  size_t totalFound = 0;
  size_t trueFound = 0;
  size_t fakeFound = 0;
  size_t uniqueTrueReco = 0;
  size_t uniqueFindableFound = 0;
  size_t sigmaXCount = 0;
  size_t sigmaYCount = 0;
  size_t sigmaZCount = 0;
  double sumSigmaX = 0.;
  double sumSigmaY = 0.;
  double sumSigmaZ = 0.;

  for (const auto& inputDir : dirs) {
    fs::current_path(inputDir);
    std::printf("Working on %s\n", inputDir.c_str());
    const auto contextPath = resolveContextFile(inputDir);
    if (contextPath.empty()) {
      std::printf("Skipping %s: could not locate %s\n", inputDir.c_str(), collContextFile);
      continue;
    }

    o2::steer::MCKinematicsReader mcReader(contextPath.string());
    if (!mcReader.isInitialized()) {
      std::printf("Skipping %s: failed to initialize MCKinematicsReader from %s\n", inputDir.c_str(), contextPath.c_str());
      continue;
    }

    std::unordered_map<int, TruthInfo> findableTruths;
    std::unordered_set<int> uniqueTrueLabelsReco;
    std::unordered_set<int> uniqueFindableTruthFound;
    std::unordered_map<int, BestRecoInfo> bestRecoByTruth;

    const int iSrc = 0;
    const auto nEvents = static_cast<int>(mcReader.getNEvents(iSrc));
    for (int iEve = 0; iEve < nEvents; ++iEve) {
      const auto& tracks = mcReader.getTracks(iSrc, iEve);
      const auto contributors = static_cast<int>(std::count_if(tracks.begin(), tracks.end(), isChargedPrimary));
      if (contributors >= 2) {
        const auto& header = mcReader.getMCEventHeader(iSrc, iEve);
        findableTruths.emplace(iEve, TruthInfo{contributors, (float)header.GetX(), (float)header.GetY(), (float)header.GetZ()});
        fillMultiplicityHistogram(hTruthMultiplicityFindable, contributors);
      }
      mcReader.releaseTracksForSourceAndEvent(iSrc, iEve);
    }

    auto* tracFileHandle = TFile::Open((inputDir / tracFile).c_str());
    if (tracFileHandle == nullptr || tracFileHandle->IsZombie()) {
      std::printf("Skipping %s: failed to open %s\n", inputDir.c_str(), tracFile);
      delete tracFileHandle;
      continue;
    }

    auto* tracTree = tracFileHandle->Get<TTree>("o2sim");
    if (tracTree == nullptr) {
      std::printf("Skipping %s: missing o2sim tree in %s\n", inputDir.c_str(), tracFile);
      tracFileHandle->Close();
      delete tracFileHandle;
      continue;
    }

    if (tracTree->GetBranch("Vertices") == nullptr || tracTree->GetBranch("ITSVertexMCTruth") == nullptr) {
      std::printf("Skipping %s: missing vertex branches in %s\n", inputDir.c_str(), tracFile);
      tracFileHandle->Close();
      delete tracFileHandle;
      continue;
    }

    std::vector<Vertex>* vertices = nullptr;
    std::vector<o2::MCCompLabel>* labels = nullptr;
    std::vector<float>* purities = nullptr;
    const bool hasPurityBranch = tracTree->GetBranch("ITSVertexMCPurity") != nullptr;

    tracTree->SetBranchAddress("Vertices", &vertices);
    tracTree->SetBranchAddress("ITSVertexMCTruth", &labels);
    if (hasPurityBranch) {
      tracTree->SetBranchAddress("ITSVertexMCPurity", &purities);
    }

    const auto nEntries = tracTree->GetEntriesFast();
    for (Long64_t iEntry = 0; iEntry < nEntries; ++iEntry) {
      tracTree->GetEntry(iEntry);
      if (vertices == nullptr || labels == nullptr) {
        continue;
      }
      auto nVertices = std::min(vertices->size(), labels->size());
      if (hasPurityBranch && purities != nullptr) {
        nVertices = std::min(nVertices, purities->size());
      }

      for (size_t iVtx = 0; iVtx < nVertices; ++iVtx) {
        const auto& vertex = (*vertices)[iVtx];
        const auto& label = (*labels)[iVtx];
        const auto multiplicity = static_cast<int>(vertex.getNContributors());
        ++totalFound;

        if (!isTrueVertexLabel(label)) {
          ++fakeFound;
          fillMultiplicityHistogram(hRecoMultiplicityFake, multiplicity);
          continue;
        }

        ++trueFound;
        const auto eventID = label.getEventID();
        uniqueTrueLabelsReco.insert(eventID);
        fillMultiplicityHistogram(hRecoMultiplicityTrue, multiplicity);

        const auto truthIt = findableTruths.find(eventID);
        if (truthIt == findableTruths.end()) {
          continue;
        }

        uniqueFindableTruthFound.insert(eventID);
        const auto purity = (hasPurityBranch && purities != nullptr) ? (*purities)[iVtx] : -1.f;
        const auto bestIt = bestRecoByTruth.find(eventID);
        if (bestIt == bestRecoByTruth.end() || isBetterReco(vertex, purity, bestIt->second.vertex, bestIt->second.purity)) {
          bestRecoByTruth[eventID] = BestRecoInfo{vertex, purity};
        }
      }
    }

    tracFileHandle->Close();
    delete tracFileHandle;

    findable += findableTruths.size();
    uniqueTrueReco += uniqueTrueLabelsReco.size();
    uniqueFindableFound += uniqueFindableTruthFound.size();

    for (const auto eventID : uniqueFindableTruthFound) {
      const auto truthIt = findableTruths.find(eventID);
      if (truthIt != findableTruths.end()) {
        fillMultiplicityHistogram(hTruthMultiplicityFound, truthIt->second.multiplicity);
      }
    }

    for (const auto& [eventID, reco] : bestRecoByTruth) {
      const auto truthIt = findableTruths.find(eventID);
      if (truthIt == findableTruths.end()) {
        continue;
      }
      const auto dx = reco.vertex.getX() - truthIt->second.x;
      const auto dy = reco.vertex.getY() - truthIt->second.y;
      const auto dz = reco.vertex.getZ() - truthIt->second.z;
      hDx->Fill(dx);
      hDy->Fill(dy);
      hDz->Fill(dz);
      if (reco.vertex.getSigmaX() > 0.f) {
        const auto pullX = dx / reco.vertex.getSigmaX();
        hPullX->Fill(pullX);
        hPullXByMult[getMultiplicityCategory(reco.vertex.getNContributors()) - 1]->Fill(pullX);
        sumSigmaX += reco.vertex.getSigmaX();
        ++sigmaXCount;
      }
      if (reco.vertex.getSigmaY() > 0.f) {
        const auto pullY = dy / reco.vertex.getSigmaY();
        hPullY->Fill(pullY);
        hPullYByMult[getMultiplicityCategory(reco.vertex.getNContributors()) - 1]->Fill(pullY);
        sumSigmaY += reco.vertex.getSigmaY();
        ++sigmaYCount;
      }
      if (reco.vertex.getSigmaZ() > 0.f) {
        const auto pullZ = dz / reco.vertex.getSigmaZ();
        hPullZ->Fill(pullZ);
        hPullZByMult[getMultiplicityCategory(reco.vertex.getNContributors()) - 1]->Fill(pullZ);
        sumSigmaZ += reco.vertex.getSigmaZ();
        ++sigmaZCount;
      }
    }
    fs::current_path(cwd);
  }

  auto* hTruthMultiplicityEfficiency = static_cast<TH1D*>(hTruthMultiplicityFound->Clone("hTruthMultiplicityEfficiency"));
  hTruthMultiplicityEfficiency->SetTitle("Unique efficiency vs truth multiplicity;truth multiplicity bin;efficiency");
  hTruthMultiplicityEfficiency->Divide(hTruthMultiplicityFound, hTruthMultiplicityFindable, 1., 1., "B");
  setMultiplicityBinLabels(hTruthMultiplicityEfficiency);
  hTruthMultiplicityEfficiency->SetMinimum(0.);
  hTruthMultiplicityEfficiency->SetMaximum(1.05);

  auto* hRecoMultiplicityTotal = static_cast<TH1D*>(hRecoMultiplicityTrue->Clone("hRecoMultiplicityTotal"));
  hRecoMultiplicityTotal->SetTitle("All reconstructed vertices;reco multiplicity bin;vertices");
  hRecoMultiplicityTotal->Add(hRecoMultiplicityFake);
  setMultiplicityBinLabels(hRecoMultiplicityTotal);

  auto* hRecoMultiplicityPurity = static_cast<TH1D*>(hRecoMultiplicityTrue->Clone("hRecoMultiplicityPurity"));
  hRecoMultiplicityPurity->SetTitle("Purity vs reconstructed multiplicity;reco multiplicity bin;purity");
  hRecoMultiplicityPurity->Divide(hRecoMultiplicityTrue, hRecoMultiplicityTotal, 1., 1., "B");
  setMultiplicityBinLabels(hRecoMultiplicityPurity);
  hRecoMultiplicityPurity->SetMinimum(0.);
  hRecoMultiplicityPurity->SetMaximum(1.05);

  const auto duplicates = trueFound >= uniqueTrueReco ? (trueFound - uniqueTrueReco) : 0UL;

  const double uniqueEfficiency = findable > 0 ? static_cast<double>(uniqueFindableFound) / findable : 0.;
  const double purity = totalFound > 0 ? static_cast<double>(trueFound) / totalFound : 0.;
  const double fakeRate = totalFound > 0 ? static_cast<double>(fakeFound) / totalFound : 0.;
  const double duplicateRate = trueFound > 0 ? static_cast<double>(duplicates) / trueFound : 0.;
  const double f1 = (uniqueEfficiency + purity) > 0. ? 2. * uniqueEfficiency * purity / (uniqueEfficiency + purity) : 0.;

  const auto dxFit = fitGaussianCore(hDx, "fitDx");
  const auto dyFit = fitGaussianCore(hDy, "fitDy");
  const auto dzFit = fitGaussianCore(hDz, "fitDz");
  const auto pullXFit = fitGaussianCore(hPullX, "fitPullX");
  const auto pullYFit = fitGaussianCore(hPullY, "fitPullY");
  const auto pullZFit = fitGaussianCore(hPullZ, "fitPullZ");
  std::array<GaussianSummary, NMultiplicityBins> pullXByMultFit{};
  std::array<GaussianSummary, NMultiplicityBins> pullYByMultFit{};
  std::array<GaussianSummary, NMultiplicityBins> pullZByMultFit{};
  for (int i = 0; i < NMultiplicityBins; ++i) {
    const auto fitX = std::string("fitPullX_") + std::to_string(i + 1);
    const auto fitY = std::string("fitPullY_") + std::to_string(i + 1);
    const auto fitZ = std::string("fitPullZ_") + std::to_string(i + 1);
    pullXByMultFit[i] = fitGaussianCore(hPullXByMult[i], fitX.c_str());
    pullYByMultFit[i] = fitGaussianCore(hPullYByMult[i], fitY.c_str());
    pullZByMultFit[i] = fitGaussianCore(hPullZByMult[i], fitZ.c_str());
  }

  std::printf("\nVertex validation summary\n");
  std::printf("  findable truth vertices      : %zu\n", findable);
  std::printf("  total reconstructed vertices : %zu\n", totalFound);
  std::printf("  true reconstructed vertices  : %zu\n", trueFound);
  std::printf("  fake reconstructed vertices  : %zu\n", fakeFound);
  std::printf("  unique true labels (all)     : %zu\n", uniqueTrueReco);
  std::printf("  unique findable truth found  : %zu\n", uniqueFindableFound);
  std::printf("  unique efficiency            : %.5f\n", uniqueEfficiency);
  std::printf("  purity                       : %.5f\n", purity);
  std::printf("  fake rate                    : %.5f\n", fakeRate);
  std::printf("  duplicate rate               : %.5f\n", duplicateRate);
  std::printf("  F1(purity,efficiency)        : %.5f\n", f1);
  std::printf("  mean reported sigma x/y/z    : %.6g / %.6g / %.6g cm\n",
              sigmaXCount > 0 ? sumSigmaX / sigmaXCount : 0.,
              sigmaYCount > 0 ? sumSigmaY / sigmaYCount : 0.,
              sigmaZCount > 0 ? sumSigmaZ / sigmaZCount : 0.);

  if (dxFit.valid) {
    std::printf("  x residual Gaussian: mean=%.6g cm sigma=%.6g cm\n", dxFit.mean, dxFit.sigma);
  }
  if (dyFit.valid) {
    std::printf("  y residual Gaussian: mean=%.6g cm sigma=%.6g cm\n", dyFit.mean, dyFit.sigma);
  }
  if (dzFit.valid) {
    std::printf("  z residual Gaussian: mean=%.6g cm sigma=%.6g cm\n", dzFit.mean, dzFit.sigma);
  }
  if (pullXFit.valid) {
    std::printf("  x pull Gaussian    : mean=%.6g sigma=%.6g\n", pullXFit.mean, pullXFit.sigma);
  }
  if (pullYFit.valid) {
    std::printf("  y pull Gaussian    : mean=%.6g sigma=%.6g\n", pullYFit.mean, pullYFit.sigma);
  }
  if (pullZFit.valid) {
    std::printf("  z pull Gaussian    : mean=%.6g sigma=%.6g\n", pullZFit.mean, pullZFit.sigma);
  }
  printGaussianByMultiplicity(pullXByMultFit, "x pull Gaussian by reconstructed multiplicity");
  printGaussianByMultiplicity(pullYByMultFit, "y pull Gaussian by reconstructed multiplicity");
  printGaussianByMultiplicity(pullZByMultFit, "z pull Gaussian by reconstructed multiplicity");

  printBinnedFractions(hTruthMultiplicityFound, hTruthMultiplicityFindable, "Efficiency vs truth multiplicity");
  printBinnedFractions(hRecoMultiplicityTrue, hRecoMultiplicityTotal, "Purity vs reconstructed multiplicity");

  auto* cValidation = new TCanvas("cVertexValidation", "Vertex validation summary", 1800, 1000);
  cValidation->Divide(3, 2);

  cValidation->cd(1);
  gPad->SetMargin(0.05, 0.05, 0.05, 0.05);
  auto* summary = new TPaveText(0.02, 0.02, 0.98, 0.98, "NDC");
  summary->SetBorderSize(0);
  summary->SetFillColor(0);
  summary->SetTextAlign(12);
  summary->SetTextFont(42);
  char line[256];
  std::snprintf(line, sizeof(line), "Findable truth vertices : %zu", findable);
  summary->AddText(line);
  std::snprintf(line, sizeof(line), "Total reconstructed     : %zu", totalFound);
  summary->AddText(line);
  std::snprintf(line, sizeof(line), "True reconstructed      : %zu", trueFound);
  summary->AddText(line);
  std::snprintf(line, sizeof(line), "Fake reconstructed      : %zu", fakeFound);
  summary->AddText(line);
  summary->AddText("");
  std::snprintf(line, sizeof(line), "Unique truth found      : %zu", uniqueFindableFound);
  summary->AddText(line);
  std::snprintf(line, sizeof(line), "Unique efficiency       : %.5f", uniqueEfficiency);
  summary->AddText(line);
  std::snprintf(line, sizeof(line), "Purity                  : %.5f", purity);
  summary->AddText(line);
  std::snprintf(line, sizeof(line), "Fake rate               : %.5f", fakeRate);
  summary->AddText(line);
  std::snprintf(line, sizeof(line), "Duplicate rate          : %.5f", duplicateRate);
  summary->AddText(line);
  std::snprintf(line, sizeof(line), "F1                      : %.5f", f1);
  summary->AddText(line);
  std::snprintf(line, sizeof(line), "mean sigma x/y/z cm     : %.3g / %.3g / %.3g",
                sigmaXCount > 0 ? sumSigmaX / sigmaXCount : 0.,
                sigmaYCount > 0 ? sumSigmaY / sigmaYCount : 0.,
                sigmaZCount > 0 ? sumSigmaZ / sigmaZCount : 0.);
  summary->AddText(line);
  summary->AddText("");
  if (dxFit.valid) {
    std::snprintf(line, sizeof(line), "dx fit mean/sigma cm : %.3g / %.3g", dxFit.mean, dxFit.sigma);
    summary->AddText(line);
  }
  if (dyFit.valid) {
    std::snprintf(line, sizeof(line), "dy fit mean/sigma cm : %.3g / %.3g", dyFit.mean, dyFit.sigma);
    summary->AddText(line);
  }
  if (dzFit.valid) {
    std::snprintf(line, sizeof(line), "dz fit mean/sigma cm : %.3g / %.3g", dzFit.mean, dzFit.sigma);
    summary->AddText(line);
  }
  if (pullXFit.valid) {
    std::snprintf(line, sizeof(line), "pull x sigma         : %.3g", pullXFit.sigma);
    summary->AddText(line);
  }
  if (pullYFit.valid) {
    std::snprintf(line, sizeof(line), "pull y sigma         : %.3g", pullYFit.sigma);
    summary->AddText(line);
  }
  if (pullZFit.valid) {
    std::snprintf(line, sizeof(line), "pull z sigma         : %.3g", pullZFit.sigma);
    summary->AddText(line);
  }
  summary->Draw();

  cValidation->cd(2);
  gPad->SetGridy();
  hTruthMultiplicityEfficiency->Draw("hist e1");

  cValidation->cd(3);
  gPad->SetGridy();
  const auto maxReco = std::max(hRecoMultiplicityTrue->GetMaximum(), hRecoMultiplicityFake->GetMaximum());
  hRecoMultiplicityTrue->SetMaximum(1.2 * std::max(1., maxReco));
  hRecoMultiplicityTrue->Draw("hist e1");
  hRecoMultiplicityFake->Draw("hist e1 same");
  auto hRecoMultiplicitySum = (TH1D*)hRecoMultiplicityTrue->Clone("hRecoMultiplicitySum");
  hRecoMultiplicitySum->Add(hRecoMultiplicityFake);
  setHistogramStyle(hRecoMultiplicitySum, kBlack, 23);
  hRecoMultiplicitySum->Draw("hist e1 same");
  {
    auto* legend = new TLegend(0.58, 0.75, 0.88, 0.88);
    legend->SetBorderSize(0);
    legend->AddEntry(hRecoMultiplicityTrue, "true", "lep");
    legend->AddEntry(hRecoMultiplicityFake, "fake", "lep");
    legend->AddEntry(hRecoMultiplicitySum, "sum", "lep");
    legend->Draw();
  }

  cValidation->cd(4);
  gPad->SetGridy();
  hRecoMultiplicityPurity->Draw("hist e1");

  cValidation->cd(5);
  gPad->SetGridy();
  auto* hDxNorm = makeNormalizedCopy(hDx, "hDxNorm", "Matched vertex residuals;residual (cm);normalized entries");
  auto* hDyNorm = makeNormalizedCopy(hDy, "hDyNorm", "Matched vertex residuals;residual (cm);normalized entries");
  auto* hDzNorm = makeNormalizedCopy(hDz, "hDzNorm", "Matched vertex residuals;residual (cm);normalized entries");
  const auto maxResidual = std::max({hDxNorm->GetMaximum(), hDyNorm->GetMaximum(), hDzNorm->GetMaximum()});
  hDzNorm->SetMaximum(1.2 * std::max(1., maxResidual));
  hDzNorm->Draw("hist");
  hDxNorm->Draw("hist same");
  hDyNorm->Draw("hist same");
  {
    auto* legend = new TLegend(0.62, 0.72, 0.88, 0.88);
    legend->SetBorderSize(0);
    legend->AddEntry(hDxNorm, "dx", "l");
    legend->AddEntry(hDyNorm, "dy", "l");
    legend->AddEntry(hDzNorm, "dz", "l");
    legend->Draw();
  }

  cValidation->cd(6);
  gPad->SetGridy();
  auto* hPullXNorm = makeNormalizedCopy(hPullX, "hPullXNorm", "Matched vertex pulls;pull;normalized entries");
  auto* hPullYNorm = makeNormalizedCopy(hPullY, "hPullYNorm", "Matched vertex pulls;pull;normalized entries");
  auto* hPullZNorm = makeNormalizedCopy(hPullZ, "hPullZNorm", "Matched vertex pulls;pull;normalized entries");
  const auto maxPull = std::max({hPullXNorm->GetMaximum(), hPullYNorm->GetMaximum(), hPullZNorm->GetMaximum()});
  hPullZNorm->SetMaximum(1.2 * std::max(1., maxPull));
  hPullZNorm->Draw("hist");
  hPullXNorm->Draw("hist same");
  hPullYNorm->Draw("hist same");
  {
    auto* legend = new TLegend(0.62, 0.72, 0.88, 0.88);
    legend->SetBorderSize(0);
    legend->AddEntry(hPullXNorm, "pull x", "l");
    legend->AddEntry(hPullYNorm, "pull y", "l");
    legend->AddEntry(hPullZNorm, "pull z", "l");
    legend->Draw();
  }

  cValidation->cd();
  cValidation->Update();
  cValidation->SaveAs("checkSeeding.pdf");
}
