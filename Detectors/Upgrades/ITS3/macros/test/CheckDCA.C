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

/// \file CheckDCA.C
/// \brief Simple macro to check ITS3 impact parameter resolution

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <TROOT.h>
#include <TStyle.h>
#include <TCanvas.h>
#include <TFile.h>
#include <TF1.h>
#include <TH2F.h>
#include <TLegend.h>
#include <TPad.h>
#include <TTree.h>
#include <TSystem.h>

#include "DataFormatsITS/TrackITS.h"
#include "DataFormatsTPC/TrackTPC.h"
#include "DetectorsBase/Propagator.h"
#include "Field/MagneticField.h"
#include "ITSBase/GeometryTGeo.h"
#include "DetectorsBase/Propagator.h"
#include "ReconstructionDataFormats/TrackTPCITS.h"
#include "ReconstructionDataFormats/Vertex.h"
#include "ReconstructionDataFormats/DCA.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCEventHeader.h"
#include "SimulationDataFormat/MCTrack.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/MCUtils.h"
#include "SimulationDataFormat/TrackReference.h"
#include "Steer/MCKinematicsReader.h"

#include <array>
#include <map>
#include <iostream>
#include <vector>
#include <filesystem>
#include <optional>
#include <regex>
#endif

namespace fs = std::filesystem;

constexpr auto mMatCorr{o2::base::Propagator::MatCorrType::USEMatCorrNONE};
constexpr float mMaxStep{2};

constexpr float rapMax{0.9};

std::vector<fs::path> find_dirs(fs::path const& dir, std::function<bool(fs::path const&)> filter, std::optional<std::function<bool(fs::path const&, fs::path const&)>> sort = std::nullopt)
{
  std::vector<fs::path> result;
  if (fs::exists(dir)) { // Find Dirs matching filter
    for (auto const& entry : fs::recursive_directory_iterator(dir, fs::directory_options::follow_directory_symlink)) {
      if (fs::is_directory(entry) && filter(entry)) {
        result.emplace_back(entry);
      }
    }
  }
  if (sort) { // Optionally sort paths
    std::sort(result.begin(), result.end(), *sort);
  }
  return result;
}

void CheckDCA(const std::string& collisioncontextFileName = "collisioncontext.root",
              const std::string& tpcTracksFileName = "tpctracks.root",
              const std::string& itsTracksFileName = "o2trac_its.root",
              const std::string& itstpcTracksFileName = "o2match_itstpc.root",
              const std::string& magFileName = "o2sim_grp.root")
{
  gROOT->SetBatch();
  gStyle->SetOptStat(0);
  gStyle->SetPalette(kRainBow);
  gStyle->SetPadLeftMargin(0.16);
  gStyle->SetPadTickX(1);
  gStyle->SetPadTickY(1);
  gErrorIgnoreLevel = 2001; // suppress warnings
  ProcInfo_t procInfo;

  const int nPtBins = 35;
  const int nPtBinsEff = 39;
  double ptLimits[nPtBins] = {0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1., 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2., 2.2, 2.5, 3., 4., 5., 6., 8., 10., 15., 20.};
  double ptLimitsEff[nPtBinsEff] = {0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1., 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2., 2.2, 2.5, 3., 4., 5., 6., 8., 10., 15., 20.};

  const std::regex tf_pattern(R"(tf\d+)");
  auto tf_matcher = [&tf_pattern](fs::path const& p) -> bool {
    return std::regex_search(p.string(), tf_pattern);
  };
  auto tf_sorter = [&tf_pattern](fs::path const& a, fs::path const& b) -> bool {
    const auto &as = a.string(), &bs = b.string();
    std::smatch am, bm;
    if (std::regex_search(as, am, tf_pattern) && std::regex_search(bs, bm, tf_pattern)) {
      return std::stoi(am.str().substr(2)) < std::stoi(bm.str().substr(2));
    } else {
      LOGP(fatal, "TF Regex matching failed");
      return false;
    }
  };

  const int nSpecies = 4;
  std::array<int, nSpecies> pdgCodes{11, 211, 321, 2212};
  auto fGaus = new TF1("fGaus", "gaus", -200., 200.);
  std::map<int, std::string> partNames = {
    {11, "Electrons"},
    {211, "Pions"},
    {321, "Kaons"},
    {2212, "Protons"}};
  std::map<int, int> colors{{11, kOrange + 7}, {211, kRed + 1}, {321, kAzure + 4}, {2212, kGreen + 2}};
  /// ITS
  std::map<int, TH1F*> hDcaxyResAllLayersITS = {
    {11, new TH1F("hDcaxyResElectronsAllLayersITS", "ITS;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{xy}}) (#mum)", nPtBins - 1, ptLimits)},
    {211, new TH1F("hDcaxyResPionsAllLayersITS", "ITS;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{xy}}) (#mum)", nPtBins - 1, ptLimits)},
    {321, new TH1F("hDcaxyResKaonsAllLayersITS", "ITS;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{xy}}) (#mum)", nPtBins - 1, ptLimits)},
    {2212, new TH1F("hDcaxyResProtonsAllLayersITS", "ITS;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{xy}}) (#mum)", nPtBins - 1, ptLimits)}};
  std::map<int, TH1F*> hDcazResAllLayersITS = {
    {11, new TH1F("hDcazResElectronsAllLayersITS", "ITS;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{z}}) (#mum)", nPtBins - 1, ptLimits)},
    {211, new TH1F("hDcazResPionsAllLayersITS", "ITS;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{z}}) (#mum)", nPtBins - 1, ptLimits)},
    {321, new TH1F("hDcazResKaonsAllLayersITS", "ITS;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{z}}) (#mum)", nPtBins - 1, ptLimits)},
    {2212, new TH1F("hDcazResProtonsAllLayersITS", "ITS;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{z}}) (#mum)", nPtBins - 1, ptLimits)}};
  std::map<int, TH1F*> hPtResAllLayersITS = {
    {11, new TH1F("hPtResElectronsAllLayersITS", "ITS;#it{p}_{T} (GeV/#it{c});#sigma(#Delta#it{p}_{T}/#it{p}_{T})", nPtBins - 1, ptLimits)},
    {211, new TH1F("hPtResPionsAllLayersITS", "ITS;#it{p}_{T} (GeV/#it{c});#sigma(#Delta#it{p}_{T}/#it{p}_{T})", nPtBins - 1, ptLimits)},
    {321, new TH1F("hPtResKaonsAllLayersITS", "ITS;#it{p}_{T} (GeV/#it{c});#sigma(#Delta#it{p}_{T}/#it{p}_{T})", nPtBins - 1, ptLimits)},
    {2212, new TH1F("hPtResProtonsAllLayersITS", "ITS;#it{p}_{T} (GeV/#it{c});#sigma(#Delta#it{p}_{T}/#it{p}_{T})", nPtBins - 1, ptLimits)}};
  std::map<int, TH1F*> hDcaxyResNoFirstLayerITS = {
    {11, new TH1F("hDcaxyResElectronsNoFirstLayerITS", "ITS;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{xy}}) (#mum)", nPtBins - 1, ptLimits)},
    {211, new TH1F("hDcaxyResPionsNoFirstLayerITS", "ITS;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{xy}}) (#mum)", nPtBins - 1, ptLimits)},
    {321, new TH1F("hDcaxyResKaonsNoFirstLayerITS", "ITS;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{xy}}) (#mum)", nPtBins - 1, ptLimits)},
    {2212, new TH1F("hDcaxyResProtonsNoFirstLayerITS", "ITS;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{xy}}) (#mum)", nPtBins - 1, ptLimits)}};
  std::map<int, TH1F*> hDcazResNoFirstLayerITS = {
    {11, new TH1F("hDcazResElectronsNoFirstLayerITS", "ITS;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{z}}) (#mum)", nPtBins - 1, ptLimits)},
    {211, new TH1F("hDcazResPionsNoFirstLayerITS", "ITS;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{z}}) (#mum)", nPtBins - 1, ptLimits)},
    {321, new TH1F("hDcazResKaonsNoFirstLayerITS", "ITS;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{z}}) (#mum)", nPtBins - 1, ptLimits)},
    {2212, new TH1F("hDcazResProtonsNoFirstLayerITS", "ITS;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{z}}) (#mum)", nPtBins - 1, ptLimits)}};
  std::map<int, TH1F*> hDcaxyReskAnyITS = {
    {11, new TH1F("hDcaxyResElectronskAnyITS", "ITS;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{xy}}) (#mum)", nPtBins - 1, ptLimits)},
    {211, new TH1F("hDcaxyResPionskAnyITS", "ITS;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{xy}}) (#mum)", nPtBins - 1, ptLimits)},
    {321, new TH1F("hDcaxyResKaonskAnyITS", "ITS;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{xy}}) (#mum)", nPtBins - 1, ptLimits)},
    {2212, new TH1F("hDcaxyResProtonskAnyITS", "ITS;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{xy}}) (#mum)", nPtBins - 1, ptLimits)}};
  std::map<int, TH1F*> hDcazReskAnyITS = {
    {11, new TH1F("hDcazResElectronskAnyITS", "ITS;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{z}}) (#mum)", nPtBins - 1, ptLimits)},
    {211, new TH1F("hDcazResPionskAnyITS", "ITS;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{z}}) (#mum)", nPtBins - 1, ptLimits)},
    {321, new TH1F("hDcazResKaonskAnyITS", "ITS;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{z}}) (#mum)", nPtBins - 1, ptLimits)},
    {2212, new TH1F("hDcazResProtonskAnyITS", "ITS;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{z}}) (#mum)", nPtBins - 1, ptLimits)}};

  std::map<int, TH2F*> hDcaxyVsPtAllLayersITS = {
    {11, new TH2F("hDcaxyVsPtElectronsAllLayersITS", "ITS Electrons;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{xy}}) (#mum)", nPtBins - 1, ptLimits, 1000, -500, 500)},
    {211, new TH2F("hDcaxyVsPtPionsAllLayersITS", "ITS Pions;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{xy}}) (#mum)", nPtBins - 1, ptLimits, 1000, -500, 500)},
    {321, new TH2F("hDcaxyVsPtKaonsAllLayersITS", "ITS Kaons;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{xy}}) (#mum)", nPtBins - 1, ptLimits, 1000, -500, 500)},
    {2212, new TH2F("hDcaxyVsPtProtonsAllLayersITS", "ITS Protons;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{xy}}) (#mum)", nPtBins - 1, ptLimits, 1000, -500, 500)}};
  std::map<int, TH2F*> hDcazVsPtAllLayersITS = {
    {11, new TH2F("hDcazVsPtElectronsAllLayersITS", "ITS Electrons;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{z}}) (#mum)", nPtBins - 1, ptLimits, 1000, -500, 500)},
    {211, new TH2F("hDcazVsPtPionsAllLayersITS", "ITS Pions;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{z}}) (#mum)", nPtBins - 1, ptLimits, 1000, -500, 500)},
    {321, new TH2F("hDcazVsPtKaonsAllLayersITS", "ITS Kaons;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{z}}) (#mum)", nPtBins - 1, ptLimits, 1000, -500, 500)},
    {2212, new TH2F("hDcazVsPtProtonsAllLayersITS", "ITS Protons;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{z}}) (#mum)", nPtBins - 1, ptLimits, 1000, -500, 500)}};
  std::map<int, TH2F*> hDcaxyVsPhiAllLayersITS = {
    {11, new TH2F("hDcaxyVsPhiElectronsAllLayersITS", "ITS Electrons (>2 Gev);#varphi (rad);#sigma(DCA_{#it{xy}}) (#mum)", 100, 0.f, 2 * TMath::Pi(), 1000, -500, 500)},
    {211, new TH2F("hDcaxyVsPhiPionsAllLayersITS", "ITS Pions (>2 Gev);#varphi (rad);#sigma(DCA_{#it{xy}}) (#mum)", 100, 0.f, 2 * TMath::Pi(), 1000, -500, 500)},
    {321, new TH2F("hDcaxyVsPhiKaonsAllLayersITS", "ITS Kaons (>2 Gev);#varphi (rad);#sigma(DCA_{#it{xy}}) (#mum)", 100, 0.f, 2 * TMath::Pi(), 1000, -500, 500)},
    {2212, new TH2F("hDcaxyVsPhiProtonsAllLayersITS", "ITS Protons (>2 Gev);#varphi (rad);#sigma(DCA_{#it{xy}}) (#mum)", 100, 0.f, 2 * TMath::Pi(), 1000, -500, 500)}};
  std::map<int, TH2F*> hDcazVsPhiAllLayersITS = {
    {11, new TH2F("hDcazVsPhiElectronsAllLayersITS", "ITS Electrons (>2 Gev);#varphi (rad);#sigma(DCA_{#it{z}}) (#mum)", 100, 0.f, 2 * TMath::Pi(), 1000, -500, 500)},
    {211, new TH2F("hDcazVsPhiPionsAllLayersITS", "ITS Pions (>2 Gev);#varphi (rad);#sigma(DCA_{#it{z}}) (#mum)", 100, 0.f, 2 * TMath::Pi(), 1000, -500, 500)},
    {321, new TH2F("hDcazVsPhiKaonsAllLayersITS", "ITS Kaons (>2 Gev);#varphi (rad);#sigma(DCA_{#it{z}}) (#mum)", 100, 0.f, 2 * TMath::Pi(), 1000, -500, 500)},
    {2212, new TH2F("hDcazVsPhiProtonsAllLayersITS", "ITS Protons (>2 Gev);#varphi (rad);#sigma(DCA_{#it{z}}) (#mum)", 100, 0.f, 2 * TMath::Pi(), 1000, -500, 500)}};
  std::map<int, TH2F*> hDcaxyVsPtNoFirstLayerITS = {
    {11, new TH2F("hDcaxyVsPtElectronsNoFirstLayerITS", "ITS Electrons;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{xy}}) (#mum)", nPtBins - 1, ptLimits, 1000, -500, 500)},
    {211, new TH2F("hDcaxyVsPtPionsNoFirstLayerITS", "ITS Pions;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{xy}}) (#mum)", nPtBins - 1, ptLimits, 1000, -500, 500)},
    {321, new TH2F("hDcaxyVsPtKaonsNoFirstLayerITS", "ITS Kaons;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{xy}}) (#mum)", nPtBins - 1, ptLimits, 1000, -500, 500)},
    {2212, new TH2F("hDcaxyVsPtProtonsNoFirstLayerITS", "ITS Protons;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{xy}}) (#mum)", nPtBins - 1, ptLimits, 1000, -500, 500)}};
  std::map<int, TH2F*> hDcazVsPtNoFirstLayerITS = {
    {11, new TH2F("hDcazVsPtElectronsNoFirstLayerITS", "ITS Electrons;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{z}}) (#mum)", nPtBins - 1, ptLimits, 1000, -500, 500)},
    {211, new TH2F("hDcazVsPtPionsNoFirstLayerITS", "ITS Pions;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{z}}) (#mum)", nPtBins - 1, ptLimits, 1000, -500, 500)},
    {321, new TH2F("hDcazVsPtKaonsNoFirstLayerITS", "ITS Kaons;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{z}}) (#mum)", nPtBins - 1, ptLimits, 1000, -500, 500)},
    {2212, new TH2F("hDcazVsPtProtonsNoFirstLayerITS", "ITS Protons;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{z}}) (#mum)", nPtBins - 1, ptLimits, 1000, -500, 500)}};
  std::map<int, TH2F*> hDcazVsPtkAnyITS = {
    {11, new TH2F("hDcazVsPtElectronskAnyITS ", "ITS Electrons;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{z}}) (#mum)", nPtBins - 1, ptLimits, 1000, -500, 500)},
    {211, new TH2F("hDcazVsPtPionskAnyITS", "ITS Pions;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{z}}) (#mum)", nPtBins - 1, ptLimits, 1000, -500, 500)},
    {321, new TH2F("hDcazVsPtKaonskAnyITS", "ITS Kaons;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{z}}) (#mum)", nPtBins - 1, ptLimits, 1000, -500, 500)},
    {2212, new TH2F("hDcazVsPtProtonskAnyITS", "ITS Protons;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{z}}) (#mum)", nPtBins - 1, ptLimits, 1000, -500, 500)}};
  std::map<int, TH2F*> hDcaxyVsPtkAnyITS = {
    {11, new TH2F("hDcaxyVsPtElectronskAnyITS", "ITS Electrons;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{xy}}) (#mum)", nPtBins - 1, ptLimits, 1000, -500, 500)},
    {211, new TH2F("hDcaxyVsPtPionskAnyITS", "ITS Pions;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{xy}}) (#mum)", nPtBins - 1, ptLimits, 1000, -500, 500)},
    {321, new TH2F("hDcaxyVsPtKaonskAnyITS", "ITS Kaons;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{xy}}) (#mum)", nPtBins - 1, ptLimits, 1000, -500, 500)},
    {2212, new TH2F("hDcaxyVsPtProtonskAnyITS", "ITS Protons;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{xy}}) (#mum)", nPtBins - 1, ptLimits, 1000, -500, 500)}};
  std::map<int, TH2F*> hDeltaPtVsPtAllLayersITS = {
    {11, new TH2F("hDeltaPtVsPtElectronsAllLayersITS", "ITS Electrons;#it{p}_{T} (GeV/#it{c});#sigma(#Delta#it{p}_{T}/#it{p}_{T})", nPtBins - 1, ptLimits, 200, -0.2, 0.2)},
    {211, new TH2F("hDeltaPtVsPtPionsAllLayersITS", "ITS Pions;#it{p}_{T} (GeV/#it{c});#sigma(#Delta#it{p}_{T}/#it{p}_{T})", nPtBins - 1, ptLimits, 200, -0.2, 0.2)},
    {321, new TH2F("hDeltaPtVsPtKaonsAllLayersITS", "ITS Kaons;#it{p}_{T} (GeV/#it{c});#sigma(#Delta#it{p}_{T}/#it{p}_{T})", nPtBins - 1, ptLimits, 200, -0.2, 0.2)},
    {2212, new TH2F("hDeltaPtVsPtProtonsAllLayersITS", "ITS Protons;#it{p}_{T} (GeV/#it{c});#sigma(#Delta#it{p}_{T}/#it{p}_{T})", nPtBins - 1, ptLimits, 200, -0.2, 0.2)}};
  // ITS-TPC
  std::map<int, TH1F*> hDcaxyResAllLayersITSTPC = {
    {11, new TH1F("hDcaxyResElectronsAllLayersITSTPC", "ITS-TPC;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{xy}}) (#mum)", nPtBins - 1, ptLimits)},
    {211, new TH1F("hDcaxyResPionsAllLayersITSTPC", "ITS-TPC;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{xy}}) (#mum)", nPtBins - 1, ptLimits)},
    {321, new TH1F("hDcaxyResKaonsAllLayersITSTPC", "ITS-TPC;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{xy}}) (#mum)", nPtBins - 1, ptLimits)},
    {2212, new TH1F("hDcaxyResProtonsAllLayersITSTPC", "ITS-TPC;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{xy}}) (#mum)", nPtBins - 1, ptLimits)}};
  std::map<int, TH1F*> hDcazResAllLayersITSTPC = {
    {11, new TH1F("hDcazResElectronsAllLayersITSTPC", "ITS-TPC;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{z}}) (#mum)", nPtBins - 1, ptLimits)},
    {211, new TH1F("hDcazResPionsAllLayersITSTPC", "ITS-TPC;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{z}}) (#mum)", nPtBins - 1, ptLimits)},
    {321, new TH1F("hDcazResKaonsAllLayersITSTPC", "ITS-TPC;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{z}}) (#mum)", nPtBins - 1, ptLimits)},
    {2212, new TH1F("hDcazResProtonsAllLayersITSTPC", "ITS-TPC;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{z}}) (#mum)", nPtBins - 1, ptLimits)}};
  std::map<int, TH1F*> hPtResAllLayersITSTPC = {
    {11, new TH1F("hPtResElectronsAllLayersITSTPC", "ITS-TPC;#it{p}_{T} (GeV/#it{c});#sigma(#Delta#it{p}_{T}/#it{p}_{T})", nPtBins - 1, ptLimits)},
    {211, new TH1F("hPtResPionsAllLayersITSTPC", "ITS-TPC;#it{p}_{T} (GeV/#it{c});#sigma(#Delta#it{p}_{T}/#it{p}_{T})", nPtBins - 1, ptLimits)},
    {321, new TH1F("hPtResKaonsAllLayersITSTPC", "ITS-TPC;#it{p}_{T} (GeV/#it{c});#sigma(#Delta#it{p}_{T}/#it{p}_{T})", nPtBins - 1, ptLimits)},
    {2212, new TH1F("hPtResProtonsAllLayersITSTPC", "ITS-TPC;#it{p}_{T} (GeV/#it{c});#sigma(#Delta#it{p}_{T}/#it{p}_{T})", nPtBins - 1, ptLimits)}};
  std::map<int, TH1F*> hDcaxyResNoFirstLayerITSTPC = {
    {11, new TH1F("hDcaxyResElectronsNoFirstLayerITSTPC", "ITS-TPC;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{xy}}) (#mum)", nPtBins - 1, ptLimits)},
    {211, new TH1F("hDcaxyResPionsNoFirstLayerITSTPC", "ITS-TPC;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{xy}}) (#mum)", nPtBins - 1, ptLimits)},
    {321, new TH1F("hDcaxyResKaonsNoFirstLayerITSTPC", "ITS-TPC;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{xy}}) (#mum)", nPtBins - 1, ptLimits)},
    {2212, new TH1F("hDcaxyResProtonsNoFirstLayerITSTPC", "ITS-TPC;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{xy}}) (#mum)", nPtBins - 1, ptLimits)}};
  std::map<int, TH1F*> hDcazResNoFirstLayerITSTPC = {
    {11, new TH1F("hDcazResElectronsNoFirstLayerITSTPC", "ITS-TPC;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{z}}) (#mum)", nPtBins - 1, ptLimits)},
    {211, new TH1F("hDcazResPionsNoFirstLayerITSTPC", "ITS-TPC;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{z}}) (#mum)", nPtBins - 1, ptLimits)},
    {321, new TH1F("hDcazResKaonsNoFirstLayerITSTPC", "ITS-TPC;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{z}}) (#mum)", nPtBins - 1, ptLimits)},
    {2212, new TH1F("hDcazResProtonsNoFirstLayerITSTPC", "ITS-TPC;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{z}}) (#mum)", nPtBins - 1, ptLimits)}};
  std::map<int, TH1F*> hDcaxyReskAnyITSTPC = {
    {11, new TH1F("hDcaxyResElectronskAnyITSTPC", "ITS-TPC;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{xy}}) (#mum)", nPtBins - 1, ptLimits)},
    {211, new TH1F("hDcaxyResPionskAnyITSTPC", "ITS-TPC;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{xy}}) (#mum)", nPtBins - 1, ptLimits)},
    {321, new TH1F("hDcaxyResKaonskAnyITSTPC", "ITS-TPC;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{xy}}) (#mum)", nPtBins - 1, ptLimits)},
    {2212, new TH1F("hDcaxyResProtonskAnyITSTPC", "ITS-TPC;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{xy}}) (#mum)", nPtBins - 1, ptLimits)}};
  std::map<int, TH1F*> hDcazReskAnyITSTPC = {
    {11, new TH1F("hDcazResElectronskAnyITSTPC", "ITS-TPC;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{z}}) (#mum)", nPtBins - 1, ptLimits)},
    {211, new TH1F("hDcazResPionskAnyITSTPC", "ITS-TPC;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{z}}) (#mum)", nPtBins - 1, ptLimits)},
    {321, new TH1F("hDcazResKaonskAnyITSTPC", "ITS-TPC;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{z}}) (#mum)", nPtBins - 1, ptLimits)},
    {2212, new TH1F("hDcazResProtonskAnyITSTPC", "ITS-TPC;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{z}}) (#mum)", nPtBins - 1, ptLimits)}};

  std::map<int, TH2F*> hDcaxyVsPtAllLayersITSTPC = {
    {11, new TH2F("hDcaxyVsPtElectronsAllLayersITSTPC", "ITS-TPC Electrons;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{xy}}) (#mum)", nPtBins - 1, ptLimits, 1000, -500, 500)},
    {211, new TH2F("hDcaxyVsPtPionsAllLayersITSTPC", "ITS-TPC Pions;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{xy}}) (#mum)", nPtBins - 1, ptLimits, 1000, -500, 500)},
    {321, new TH2F("hDcaxyVsPtKaonsAllLayersITSTPC", "ITS-TPC Kaons;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{xy}}) (#mum)", nPtBins - 1, ptLimits, 1000, -500, 500)},
    {2212, new TH2F("hDcaxyVsPtProtonsAllLayersITSTPC", "ITS-TPC Protons;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{xy}}) (#mum)", nPtBins - 1, ptLimits, 1000, -500, 500)}};
  std::map<int, TH2F*> hDcazVsPtAllLayersITSTPC = {
    {11, new TH2F("hDcazVsPtElectronsAllLayersITSTPC", "ITS-TPC Electrons;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{z}}) (#mum)", nPtBins - 1, ptLimits, 1000, -500, 500)},
    {211, new TH2F("hDcazVsPtPionsAllLayersITSTPC", "ITS-TPC Pions;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{z}}) (#mum)", nPtBins - 1, ptLimits, 1000, -500, 500)},
    {321, new TH2F("hDcazVsPtKaonsAllLayersITSTPC", "ITS-TPC Kaons;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{z}}) (#mum)", nPtBins - 1, ptLimits, 1000, -500, 500)},
    {2212, new TH2F("hDcazVsPtProtonsAllLayersITSTPC", "ITS-TPC Protons;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{z}}) (#mum)", nPtBins - 1, ptLimits, 1000, -500, 500)}};
  std::map<int, TH2F*> hDcaxyVsPhiAllLayersITSTPC = {
    {11, new TH2F("hDcaxyVsPhiElectronsAllLayersITSTPC", "ITS-TPC Electrons (>2 Gev);#varphi (rad);#sigma(DCA_{#it{xy}}) (#mum)", 100, 0.f, 2 * TMath::Pi(), 1000, -500, 500)},
    {211, new TH2F("hDcaxyVsPhiPionsAllLayersITSTPC", "ITS-TPC Pions (>2 Gev);#varphi (rad);#sigma(DCA_{#it{xy}}) (#mum)", 100, 0.f, 2 * TMath::Pi(), 1000, -500, 500)},
    {321, new TH2F("hDcaxyVsPhiKaonsAllLayersITSTPC", "ITS-TPC Kaons (>2 Gev);#varphi (rad);#sigma(DCA_{#it{xy}}) (#mum)", 100, 0.f, 2 * TMath::Pi(), 1000, -500, 500)},
    {2212, new TH2F("hDcaxyVsPhiProtonsAllLayersITSTPC", "ITS-TPC Protons (>2 Gev);#varphi (rad);#sigma(DCA_{#it{xy}}) (#mum)", 100, 0.f, 2 * TMath::Pi(), 1000, -500, 500)}};
  std::map<int, TH2F*> hDcazVsPhiAllLayersITSTPC = {
    {11, new TH2F("hDcazVsPhiElectronsAllLayersITSTPC", "ITS-TPC Electrons (>2 Gev);#varphi (rad);#sigma(DCA_{#it{z}}) (#mum)", 100, 0.f, 2 * TMath::Pi(), 1000, -500, 500)},
    {211, new TH2F("hDcazVsPhiPionsAllLayersITSTPC", "ITS-TPC Pions (>2 Gev);#varphi (rad);#sigma(DCA_{#it{z}}) (#mum)", 100, 0.f, 2 * TMath::Pi(), 1000, -500, 500)},
    {321, new TH2F("hDcazVsPhiKaonsAllLayersITSTPC", "ITS-TPC Kaons (>2 Gev);#varphi (rad);#sigma(DCA_{#it{z}}) (#mum)", 100, 0.f, 2 * TMath::Pi(), 1000, -500, 500)},
    {2212, new TH2F("hDcazVsPhiProtonsAllLayersITSTPC", "ITS-TPC Protons (>2 Gev);#varphi (rad);#sigma(DCA_{#it{z}}) (#mum)", 100, 0.f, 2 * TMath::Pi(), 1000, -500, 500)}};
  std::map<int, TH2F*> hDcaxyVsPtNoFirstLayerITSTPC = {
    {11, new TH2F("hDcaxyVsPtElectronsNoFirstLayerITSTPC", "ITS-TPC Electrons;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{xy}}) (#mum)", nPtBins - 1, ptLimits, 1000, -500, 500)},
    {211, new TH2F("hDcaxyVsPtPionsNoFirstLayerITSTPC", "ITS-TPC Pions;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{xy}}) (#mum)", nPtBins - 1, ptLimits, 1000, -500, 500)},
    {321, new TH2F("hDcaxyVsPtKaonsNoFirstLayerITSTPC", "ITS-TPC Kaons;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{xy}}) (#mum)", nPtBins - 1, ptLimits, 1000, -500, 500)},
    {2212, new TH2F("hDcaxyVsPtProtonsNoFirstLayerITSTPC", "ITS-TPC Protons;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{xy}}) (#mum)", nPtBins - 1, ptLimits, 1000, -500, 500)}};
  std::map<int, TH2F*> hDcazVsPtNoFirstLayerITSTPC = {
    {11, new TH2F("hDcazVsPtElectronsNoFirstLayerITSTPC", "ITS-TPC Electrons;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{z}}) (#mum)", nPtBins - 1, ptLimits, 1000, -500, 500)},
    {211, new TH2F("hDcazVsPtPionsNoFirstLayerITSTPC", "ITS-TPC Pions;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{z}}) (#mum)", nPtBins - 1, ptLimits, 1000, -500, 500)},
    {321, new TH2F("hDcazVsPtKaonsNoFirstLayerITSTPC", "ITS-TPC Kaons;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{z}}) (#mum)", nPtBins - 1, ptLimits, 1000, -500, 500)},
    {2212, new TH2F("hDcazVsPtProtonsNoFirstLayerITSTPC", "ITS-TPC Protons;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{z}}) (#mum)", nPtBins - 1, ptLimits, 1000, -500, 500)}};
  std::map<int, TH2F*> hDcazVsPtkAnyITSTPC = {
    {11, new TH2F("hDcazVsPtElectronskAnyITSTPC", "ITS-TPC Electrons;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{z}}) (#mum)", nPtBins - 1, ptLimits, 1000, -500, 500)},
    {211, new TH2F("hDcazVsPtPionskAnyITSTPC", "ITS-TPC Pions;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{z}}) (#mum)", nPtBins - 1, ptLimits, 1000, -500, 500)},
    {321, new TH2F("hDcazVsPtKaonskAnyITSTPC", "ITS-TPC Kaons;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{z}}) (#mum)", nPtBins - 1, ptLimits, 1000, -500, 500)},
    {2212, new TH2F("hDcazVsPtProtonskAnyITSTPC", "ITS-TPC Protons;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{z}}) (#mum)", nPtBins - 1, ptLimits, 1000, -500, 500)}};
  std::map<int, TH2F*> hDcaxyVsPtkAnyITSTPC = {
    {11, new TH2F("hDcaxyVsPtElectronskAnyITSTPC", "ITS-TPC Electrons;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{xy}}) (#mum)", nPtBins - 1, ptLimits, 1000, -500, 500)},
    {211, new TH2F("hDcaxyVsPtPionskAnyITSTPC", "ITS-TPC Pions;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{xy}}) (#mum)", nPtBins - 1, ptLimits, 1000, -500, 500)},
    {321, new TH2F("hDcaxyVsPtKaonskAnyITSTPC", "ITS-TPC Kaons;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{xy}}) (#mum)", nPtBins - 1, ptLimits, 1000, -500, 500)},
    {2212, new TH2F("hDcaxyVsPtProtonskAnyITSTPC", "ITS-TPC Protons;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{xy}}) (#mum)", nPtBins - 1, ptLimits, 1000, -500, 500)}};
  std::map<int, TH2F*> hDeltaPtVsPtAllLayersITSTPC = {
    {11, new TH2F("hDeltaPtVsPtElectronsAllLayersITSTPC", "ITS-TPC Electrons;#it{p}_{T} (GeV/#it{c});#sigma(#Delta#it{p}_{T}/#it{p}_{T})", nPtBins - 1, ptLimits, 200, -0.2, 0.2)},
    {211, new TH2F("hDeltaPtVsPtPionsAllLayersITSTPC", "ITS-TPC Pions;#it{p}_{T} (GeV/#it{c});#sigma(#Delta#it{p}_{T}/#it{p}_{T})", nPtBins - 1, ptLimits, 200, -0.2, 0.2)},
    {321, new TH2F("hDeltaPtVsPtKaonsAllLayersITSTPC", "ITS-TPC Kaons;#it{p}_{T} (GeV/#it{c});#sigma(#Delta#it{p}_{T}/#it{p}_{T})", nPtBins - 1, ptLimits, 200, -0.2, 0.2)},
    {2212, new TH2F("hDeltaPtVsPtProtonsAllLayersITSTPC", "ITS-TPC Protons;#it{p}_{T} (GeV/#it{c});#sigma(#Delta#it{p}_{T}/#it{p}_{T})", nPtBins - 1, ptLimits, 200, -0.2, 0.2)}};

  o2::dataformats::VertexBase collision;
  o2::dataformats::DCA impactParameter;

  const auto origWD{fs::current_path()};
  const auto tfDirs = find_dirs(fs::current_path(), tf_matcher, tf_sorter);
  for (const auto& tfDir : tfDirs) {
    LOGP(info, "Analysing {:?}", tfDir.c_str());
    fs::current_path(tfDir);

    // MC Information
    o2::steer::MCKinematicsReader mcReader;
    if (!mcReader.initFromDigitContext(collisioncontextFileName)) {
      LOGP(error, "Cannot init MC reader in {:?}", tfDir.c_str());
      continue;
    }

    // Magnetic field and Propagator
    float bz{-999};
    static bool initOnce{false};
    if (!initOnce) {
      initOnce = true;
      o2::base::Propagator::initFieldFromGRP(magFileName);
      bz = o2::base::Propagator::Instance()->getNominalBz();
    }

    LOGP(info, "Loading ITS Tracks");
    auto fITSTracks = TFile::Open(itsTracksFileName.c_str(), "READ");
    auto tITSTracks = fITSTracks->Get<TTree>("o2sim");
    std::vector<o2::its::TrackITS>* itsTracks{nullptr};
    tITSTracks->SetBranchAddress("ITSTrack", &itsTracks);
    std::vector<o2::MCCompLabel>* itsTrkLab{nullptr};
    tITSTracks->SetBranchAddress("ITSTrackMCTruth", &itsTrkLab);

    for (Long64_t iEntry{0}; tITSTracks->LoadTree(iEntry) >= 0; ++iEntry) {
      tITSTracks->GetEntry(iEntry);
      for (size_t iTrk{0}; iTrk < itsTracks->size(); ++iTrk) {
        auto trk = itsTracks->at(iTrk);
        const auto& lbl = itsTrkLab->at(iTrk);
        if (!lbl.isValid()) {
          continue;
        }

        const auto& mcEvent = mcReader.getMCEventHeader(lbl.getSourceID(), lbl.getEventID());
        const auto& mcTrack = mcReader.getTrack(lbl);
        if (!mcTrack->isPrimary() || !(std::abs(mcTrack->GetEta()) < rapMax)) {
          continue;
        }
        auto pdg = std::abs(mcTrack->GetPdgCode());
        if (pdg != 11 && pdg != 211 && pdg != 321 && pdg != 2212) {
          continue;
        }

        collision.setXYZ(mcEvent.GetX(), mcEvent.GetY(), mcEvent.GetZ());
        if (!o2::base::Propagator::Instance()->propagateToDCA(collision, trk, bz, mMaxStep, mMatCorr, &impactParameter)) {
          continue;
        }

        auto ptReco = trk.getPt();
        auto ptGen = mcTrack->GetPt();
        auto deltaPt = (1. / ptReco - 1. / ptGen) / (1. / ptGen);
        auto dcaXY = impactParameter.getY() * 10000;
        auto dcaZ = impactParameter.getZ() * 10000;
        auto phiReco = trk.getPhi();

        if (trk.getNumberOfClusters() == 7) {
          hDcaxyVsPtAllLayersITS[pdg]->Fill(ptGen, dcaXY);
          hDcazVsPtAllLayersITS[pdg]->Fill(ptGen, dcaZ);
          hDeltaPtVsPtAllLayersITS[pdg]->Fill(ptGen, deltaPt);
          if (ptGen > 2.) {
            hDcaxyVsPhiAllLayersITS[pdg]->Fill(phiReco, dcaXY);
            hDcazVsPhiAllLayersITS[pdg]->Fill(phiReco, dcaZ);
          }
        } else if (!trk.hasHitOnLayer(0)) {
          hDcaxyVsPtNoFirstLayerITS[pdg]->Fill(ptGen, dcaXY);
          hDcazVsPtNoFirstLayerITS[pdg]->Fill(ptGen, dcaZ);
        } else {
          hDcaxyVsPtkAnyITS[pdg]->Fill(ptGen, dcaXY);
          hDcazVsPtkAnyITS[pdg]->Fill(ptGen, dcaZ);
        }
      }
    }

    LOGP(info, "Loading ITS-TPC Tracks");
    auto fITSTPCTracks = TFile::Open(itstpcTracksFileName.c_str(), "READ");
    auto tITSTPCTracks = fITSTPCTracks->Get<TTree>("matchTPCITS");
    std::vector<o2::dataformats::TrackTPCITS>* itstpcTracks{nullptr};
    tITSTPCTracks->SetBranchAddress("TPCITS", &itstpcTracks);
    std::vector<o2::MCCompLabel>* itstpcTrkLab{nullptr};
    tITSTPCTracks->SetBranchAddress("MatchMCTruth", &itstpcTrkLab);
    // TPC Tracks
    auto fTPCTracks = TFile::Open(tpcTracksFileName.c_str(), "READ");
    auto tTPCTracks = fTPCTracks->Get<TTree>("tpcrec");
    std::vector<o2::tpc::TrackTPC>* tpcTracks{nullptr};
    tTPCTracks->SetBranchAddress("TPCTracks", &tpcTracks);
    std::vector<o2::MCCompLabel>* tpcTrkLab{nullptr};
    tTPCTracks->SetBranchAddress("TPCTracksMCTruth", &tpcTrkLab);
    for (Long64_t iEntry{0}; tITSTPCTracks->LoadTree(iEntry) >= 0; ++iEntry) {
      tITSTPCTracks->GetEntry(iEntry);
      tITSTracks->GetEntry(iEntry);
      tTPCTracks->GetEntry(iEntry);
      for (size_t iTrk{0}; iTrk < itstpcTracks->size(); ++iTrk) {
        auto trk = itstpcTracks->at(iTrk);
        const auto& lbl = itstpcTrkLab->at(iTrk);

        const auto& trkITS = itsTracks->at(trk.getRefITS().getIndex());
        const auto& trkITSLbl = itsTrkLab->at(trk.getRefITS().getIndex());
        const auto& trkTPC = tpcTracks->at(trk.getRefTPC().getIndex());
        const auto& trkTPCLbl = tpcTrkLab->at(trk.getRefTPC().getIndex());
        if (!lbl.isValid() || trkITSLbl != trkTPCLbl) {
          continue;
        }

        const auto& mcEvent = mcReader.getMCEventHeader(lbl.getSourceID(), lbl.getEventID());
        const auto& mcTrack = mcReader.getTrack(lbl);
        if (!mcTrack->isPrimary() || !(std::abs(mcTrack->GetEta()) < rapMax)) {
          continue;
        }

        auto pdg = std::abs(mcTrack->GetPdgCode());
        if (pdg != 11 && pdg != 211 && pdg != 321 && pdg != 2212) {
          continue;
        }

        collision.setXYZ(mcEvent.GetX(), mcEvent.GetY(), mcEvent.GetZ());
        if (!o2::base::Propagator::Instance()->propagateToDCA(collision, trk, bz, mMaxStep, mMatCorr, &impactParameter)) {
          continue;
        }

        auto ptReco = trk.getPt();
        auto ptGen = mcTrack->GetPt();
        auto deltaPt = (1. / ptReco - 1. / ptGen) / (1. / ptGen);
        auto dcaXY = impactParameter.getY() * 10000;
        auto dcaZ = impactParameter.getZ() * 10000;
        auto phiReco = trk.getPhi();

        if (trkITS.getNumberOfClusters() == 7) {
          hDcaxyVsPtAllLayersITSTPC[pdg]->Fill(ptGen, dcaXY);
          hDcazVsPtAllLayersITSTPC[pdg]->Fill(ptGen, dcaZ);
          hDeltaPtVsPtAllLayersITSTPC[pdg]->Fill(ptGen, deltaPt);
          if (ptGen > 2.) {
            hDcaxyVsPhiAllLayersITSTPC[pdg]->Fill(phiReco, dcaXY);
            hDcazVsPhiAllLayersITSTPC[pdg]->Fill(phiReco, dcaZ);
          }
        } else if (!trkITS.hasHitOnLayer(0) && !trkITS.hasHitOnLayer(1) && !trkITS.hasHitOnLayer(2)) {
          hDcaxyVsPtNoFirstLayerITSTPC[pdg]->Fill(ptGen, dcaXY);
          hDcazVsPtNoFirstLayerITSTPC[pdg]->Fill(ptGen, dcaZ);
        } else {
          hDcaxyVsPtkAnyITSTPC[pdg]->Fill(ptGen, dcaXY);
          hDcazVsPtkAnyITSTPC[pdg]->Fill(ptGen, dcaZ);
        }
      }
    }

    delete itsTracks;
    delete itsTrkLab;
    delete tpcTracks;
    delete tpcTrkLab;
    delete itstpcTracks;
    delete itstpcTrkLab;
    delete tITSTracks;
    delete tTPCTracks;
    delete tITSTPCTracks;
    delete fITSTracks;
    delete fTPCTracks;
    delete fITSTPCTracks;

    gSystem->GetProcInfo(&procInfo);
    LOGF(info, "MemVirtual (%ld), MemResident (%ld)", procInfo.fMemVirtual, procInfo.fMemResident);
    LOGP(info, "Done with {:?}", tfDir.c_str());
    if (procInfo.fMemResident > 200'000'000) {
      LOGP(error, "Exceeding 200GBs stopping!");
      break;
    }
  }
  LOGP(info, "Restoring original CWD to {:?}", origWD.c_str());
  fs::current_path(origWD); // restore original wd

  LOGP(info, "Projecting Plots");
  TH1* hProj;
  for (const auto& pdgCode : pdgCodes) {
    for (auto iPt{0}; iPt < nPtBins; ++iPt) {
      // ITS
      auto ptMin = hDcaxyVsPtAllLayersITS[pdgCode]->GetXaxis()->GetBinLowEdge(iPt + 1);
      float minFit = (ptMin < 1.) ? -200. : -50.;
      float maxFit = (ptMin < 1.) ? 200. : 50.;

      hProj = hDeltaPtVsPtAllLayersITS[pdgCode]->ProjectionY(Form("hProjDeltaPt%d%dITS", pdgCode, iPt), iPt + 1, iPt + 1);
      hProj->Fit("fGaus", "Q0");
      hPtResAllLayersITS[pdgCode]->SetBinContent(iPt + 1, fGaus->GetParameter(2));
      hPtResAllLayersITS[pdgCode]->SetBinError(iPt + 1, fGaus->GetParError(2));

      hProj = hDcaxyVsPtAllLayersITS[pdgCode]->ProjectionY(Form("hProjDcaxy%d%dITS", pdgCode, iPt), iPt + 1, iPt + 1);
      hProj->Fit("fGaus", "Q0", "", minFit, maxFit);
      hDcaxyResAllLayersITS[pdgCode]->SetBinContent(iPt + 1, fGaus->GetParameter(2));
      hDcaxyResAllLayersITS[pdgCode]->SetBinError(iPt + 1, fGaus->GetParError(2));

      hProj = hDcazVsPtAllLayersITS[pdgCode]->ProjectionY(Form("hProjDcaz%d%dITS", pdgCode, iPt), iPt + 1, iPt + 1);
      hProj->Fit("fGaus", "Q0", "", minFit, maxFit);
      hDcazResAllLayersITS[pdgCode]->SetBinContent(iPt + 1, fGaus->GetParameter(2));
      hDcazResAllLayersITS[pdgCode]->SetBinError(iPt + 1, fGaus->GetParError(2));

      hProj = hDcaxyVsPtNoFirstLayerITS[pdgCode]->ProjectionY(Form("hProjDcaxy%d%dITS", pdgCode, iPt), iPt + 1, iPt + 1);
      hProj->Fit("fGaus", "Q0", "", minFit, maxFit);
      hDcaxyResNoFirstLayerITS[pdgCode]->SetBinContent(iPt + 1, fGaus->GetParameter(2));
      hDcaxyResNoFirstLayerITS[pdgCode]->SetBinError(iPt + 1, fGaus->GetParError(2));

      hProj = hDcazVsPtNoFirstLayerITS[pdgCode]->ProjectionY(Form("hProjDcaz%d%dITS", pdgCode, iPt), iPt + 1, iPt + 1);
      hProj->Fit("fGaus", "Q0", "", minFit, maxFit);
      hDcazResNoFirstLayerITS[pdgCode]->SetBinContent(iPt + 1, fGaus->GetParameter(2));
      hDcazResNoFirstLayerITS[pdgCode]->SetBinError(iPt + 1, fGaus->GetParError(2));

      hProj = hDcaxyVsPtkAnyITS[pdgCode]->ProjectionY(Form("hProjDcaxy%d%dITS", pdgCode, iPt), iPt + 1, iPt + 1);
      hProj->Fit("fGaus", "Q0", "", minFit, maxFit);
      hDcaxyReskAnyITS[pdgCode]->SetBinContent(iPt + 1, fGaus->GetParameter(2));
      hDcaxyReskAnyITS[pdgCode]->SetBinError(iPt + 1, fGaus->GetParError(2));

      hProj = hDcazVsPtkAnyITS[pdgCode]->ProjectionY(Form("hProjDcaz%d%dITS", pdgCode, iPt), iPt + 1, iPt + 1);
      hProj->Fit("fGaus", "Q0", "", minFit, maxFit);
      hDcazReskAnyITS[pdgCode]->SetBinContent(iPt + 1, fGaus->GetParameter(2));
      hDcazReskAnyITS[pdgCode]->SetBinError(iPt + 1, fGaus->GetParError(2));

      // ITS-TPC
      hProj = hDeltaPtVsPtAllLayersITSTPC[pdgCode]->ProjectionY(Form("hProjDeltaPt%d%dITSTPC", pdgCode, iPt), iPt + 1, iPt + 1);
      hProj->Fit("fGaus", "Q0");
      hPtResAllLayersITSTPC[pdgCode]->SetBinContent(iPt + 1, fGaus->GetParameter(2));
      hPtResAllLayersITSTPC[pdgCode]->SetBinError(iPt + 1, fGaus->GetParError(2));

      ptMin = hDcaxyVsPtAllLayersITSTPC[pdgCode]->GetXaxis()->GetBinLowEdge(iPt + 1);
      minFit = (ptMin < 1.) ? -200. : -50.;
      maxFit = (ptMin < 1.) ? 200. : 50.;

      hProj = hDcaxyVsPtAllLayersITSTPC[pdgCode]->ProjectionY(Form("hProjDcaxy%d%dITSTPC", pdgCode, iPt), iPt + 1, iPt + 1);
      hProj->Fit("fGaus", "Q0", "", minFit, maxFit);
      hDcaxyResAllLayersITSTPC[pdgCode]->SetBinContent(iPt + 1, fGaus->GetParameter(2));
      hDcaxyResAllLayersITSTPC[pdgCode]->SetBinError(iPt + 1, fGaus->GetParError(2));

      hProj = hDcazVsPtAllLayersITSTPC[pdgCode]->ProjectionY(Form("hProjDcaz%d%dITSTPC", pdgCode, iPt), iPt + 1, iPt + 1);
      hProj->Fit("fGaus", "Q0", "", minFit, maxFit);
      hDcazResAllLayersITSTPC[pdgCode]->SetBinContent(iPt + 1, fGaus->GetParameter(2));
      hDcazResAllLayersITSTPC[pdgCode]->SetBinError(iPt + 1, fGaus->GetParError(2));

      hProj = hDcaxyVsPtNoFirstLayerITSTPC[pdgCode]->ProjectionY(Form("hProjDcaxy%d%dITSTPC", pdgCode, iPt), iPt + 1, iPt + 1);
      hProj->Fit("fGaus", "Q0", "", minFit, maxFit);
      hDcaxyResNoFirstLayerITSTPC[pdgCode]->SetBinContent(iPt + 1, fGaus->GetParameter(2));
      hDcaxyResNoFirstLayerITSTPC[pdgCode]->SetBinError(iPt + 1, fGaus->GetParError(2));

      hProj = hDcazVsPtNoFirstLayerITSTPC[pdgCode]->ProjectionY(Form("hProjDcaz%d%dITSTPC", pdgCode, iPt), iPt + 1, iPt + 1);
      hProj->Fit("fGaus", "Q0", "", minFit, maxFit);
      hDcazResNoFirstLayerITSTPC[pdgCode]->SetBinContent(iPt + 1, fGaus->GetParameter(2));
      hDcazResNoFirstLayerITSTPC[pdgCode]->SetBinError(iPt + 1, fGaus->GetParError(2));

      hProj = hDcaxyVsPtkAnyITSTPC[pdgCode]->ProjectionY(Form("hProjDcaxy%d%dITSTPC", pdgCode, iPt), iPt + 1, iPt + 1);
      hProj->Fit("fGaus", "Q0", "", minFit, maxFit);
      hDcaxyReskAnyITSTPC[pdgCode]->SetBinContent(iPt + 1, fGaus->GetParameter(2));
      hDcaxyReskAnyITSTPC[pdgCode]->SetBinError(iPt + 1, fGaus->GetParError(2));

      hProj = hDcazVsPtkAnyITSTPC[pdgCode]->ProjectionY(Form("hProjDcaz%d%dITSTPC", pdgCode, iPt), iPt + 1, iPt + 1);
      hProj->Fit("fGaus", "Q0", "", minFit, maxFit);
      hDcazReskAnyITSTPC[pdgCode]->SetBinContent(iPt + 1, fGaus->GetParameter(2));
      hDcazReskAnyITSTPC[pdgCode]->SetBinError(iPt + 1, fGaus->GetParError(2));
    }
  }

  // Style
  LOGP(info, "Styling Plots");
  for (const auto& pdgCode : pdgCodes) {
    // ITS
    hPtResAllLayersITS[pdgCode]->SetLineWidth(2);
    hPtResAllLayersITS[pdgCode]->SetMarkerColor(colors[pdgCode]);
    hPtResAllLayersITS[pdgCode]->SetLineColor(colors[pdgCode]);
    hPtResAllLayersITS[pdgCode]->SetMarkerStyle(kFullCircle);

    hDcaxyResAllLayersITS[pdgCode]->SetLineWidth(2);
    hDcaxyResAllLayersITS[pdgCode]->SetMarkerColor(colors[pdgCode]);
    hDcaxyResAllLayersITS[pdgCode]->SetLineColor(colors[pdgCode]);
    hDcaxyResAllLayersITS[pdgCode]->SetMarkerStyle(kFullCircle);

    hDcazResAllLayersITS[pdgCode]->SetLineWidth(2);
    hDcazResAllLayersITS[pdgCode]->SetMarkerColor(colors[pdgCode]);
    hDcazResAllLayersITS[pdgCode]->SetLineColor(colors[pdgCode]);
    hDcazResAllLayersITS[pdgCode]->SetMarkerStyle(kFullCircle);

    hDcaxyResNoFirstLayerITS[pdgCode]->SetLineWidth(2);
    hDcaxyResNoFirstLayerITS[pdgCode]->SetMarkerColor(colors[pdgCode]);
    hDcaxyResNoFirstLayerITS[pdgCode]->SetLineColor(colors[pdgCode]);
    hDcaxyResNoFirstLayerITS[pdgCode]->SetMarkerStyle(kFullCircle);

    hDcazResNoFirstLayerITS[pdgCode]->SetLineWidth(2);
    hDcazResNoFirstLayerITS[pdgCode]->SetMarkerColor(colors[pdgCode]);
    hDcazResNoFirstLayerITS[pdgCode]->SetLineColor(colors[pdgCode]);
    hDcazResNoFirstLayerITS[pdgCode]->SetMarkerStyle(kFullCircle);

    hDcaxyReskAnyITS[pdgCode]->SetLineWidth(2);
    hDcaxyReskAnyITS[pdgCode]->SetMarkerColor(colors[pdgCode]);
    hDcaxyReskAnyITS[pdgCode]->SetLineColor(colors[pdgCode]);
    hDcaxyReskAnyITS[pdgCode]->SetMarkerStyle(kFullCircle);

    hDcazReskAnyITS[pdgCode]->SetLineWidth(2);
    hDcazReskAnyITS[pdgCode]->SetMarkerColor(colors[pdgCode]);
    hDcazReskAnyITS[pdgCode]->SetLineColor(colors[pdgCode]);
    hDcazReskAnyITS[pdgCode]->SetMarkerStyle(kFullCircle);

    // ITS-TPC
    hPtResAllLayersITSTPC[pdgCode]->SetLineWidth(2);
    hPtResAllLayersITSTPC[pdgCode]->SetMarkerColor(colors[pdgCode]);
    hPtResAllLayersITSTPC[pdgCode]->SetLineColor(colors[pdgCode]);
    hPtResAllLayersITSTPC[pdgCode]->SetMarkerStyle(kOpenCircle);

    hDcaxyResAllLayersITSTPC[pdgCode]->SetLineWidth(2);
    hDcaxyResAllLayersITSTPC[pdgCode]->SetMarkerColor(colors[pdgCode]);
    hDcaxyResAllLayersITSTPC[pdgCode]->SetLineColor(colors[pdgCode]);
    hDcaxyResAllLayersITSTPC[pdgCode]->SetMarkerStyle(kOpenCircle);

    hDcazResAllLayersITSTPC[pdgCode]->SetLineWidth(2);
    hDcazResAllLayersITSTPC[pdgCode]->SetMarkerColor(colors[pdgCode]);
    hDcazResAllLayersITSTPC[pdgCode]->SetLineColor(colors[pdgCode]);
    hDcazResAllLayersITSTPC[pdgCode]->SetMarkerStyle(kOpenCircle);

    hDcaxyResNoFirstLayerITSTPC[pdgCode]->SetLineWidth(2);
    hDcaxyResNoFirstLayerITSTPC[pdgCode]->SetMarkerColor(colors[pdgCode]);
    hDcaxyResNoFirstLayerITSTPC[pdgCode]->SetLineColor(colors[pdgCode]);
    hDcaxyResNoFirstLayerITSTPC[pdgCode]->SetMarkerStyle(kOpenCircle);

    hDcazResNoFirstLayerITSTPC[pdgCode]->SetLineWidth(2);
    hDcazResNoFirstLayerITSTPC[pdgCode]->SetMarkerColor(colors[pdgCode]);
    hDcazResNoFirstLayerITSTPC[pdgCode]->SetLineColor(colors[pdgCode]);
    hDcazResNoFirstLayerITSTPC[pdgCode]->SetMarkerStyle(kOpenCircle);

    hDcaxyReskAnyITSTPC[pdgCode]->SetLineWidth(2);
    hDcaxyReskAnyITSTPC[pdgCode]->SetMarkerColor(colors[pdgCode]);
    hDcaxyReskAnyITSTPC[pdgCode]->SetLineColor(colors[pdgCode]);
    hDcaxyReskAnyITSTPC[pdgCode]->SetMarkerStyle(kOpenCircle);

    hDcazReskAnyITSTPC[pdgCode]->SetLineWidth(2);
    hDcazReskAnyITSTPC[pdgCode]->SetMarkerColor(colors[pdgCode]);
    hDcazReskAnyITSTPC[pdgCode]->SetLineColor(colors[pdgCode]);
    hDcazReskAnyITSTPC[pdgCode]->SetMarkerStyle(kOpenCircle);
  }

  /// Output
  LOGP(info, "Writing final output");
  // ITS
  auto canvPtDeltaITS = new TCanvas("canvPtDeltaITS", "", 1500, 500);
  canvPtDeltaITS->Divide(nSpecies, 1);
  canvPtDeltaITS->cd(1)->SetLogz();
  hDeltaPtVsPtAllLayersITS[11]->Draw("colz");
  canvPtDeltaITS->cd(2)->SetLogz();
  hDeltaPtVsPtAllLayersITS[211]->Draw("colz");
  canvPtDeltaITS->cd(3)->SetLogz();
  hDeltaPtVsPtAllLayersITS[321]->Draw("colz");
  canvPtDeltaITS->cd(4)->SetLogz();
  hDeltaPtVsPtAllLayersITS[2212]->Draw("colz");

  auto canvDcaVsPtITS = new TCanvas("canvDcaVsPtITS", "", 1500, 1000);
  canvDcaVsPtITS->Divide(nSpecies, 2);
  canvDcaVsPtITS->cd(1)->SetLogz();
  hDcaxyVsPtAllLayersITS[11]->Draw("colz");
  canvDcaVsPtITS->cd(2)->SetLogz();
  hDcaxyVsPtAllLayersITS[211]->Draw("colz");
  canvDcaVsPtITS->cd(3)->SetLogz();
  hDcaxyVsPtAllLayersITS[321]->Draw("colz");
  canvDcaVsPtITS->cd(4)->SetLogz();
  hDcaxyVsPtAllLayersITS[2212]->Draw("colz");
  canvDcaVsPtITS->cd(5)->SetLogz();
  hDcazVsPtAllLayersITS[11]->Draw("colz");
  canvDcaVsPtITS->cd(6)->SetLogz();
  hDcazVsPtAllLayersITS[211]->Draw("colz");
  canvDcaVsPtITS->cd(7)->SetLogz();
  hDcazVsPtAllLayersITS[321]->Draw("colz");
  canvDcaVsPtITS->cd(8)->SetLogz();
  hDcazVsPtAllLayersITS[2212]->Draw("colz");

  auto canvPtResITS = new TCanvas("canvPtResITS", "", 500, 500);
  canvPtResITS->DrawFrame(ptLimits[0], 0., ptLimits[nPtBins - 1], 0.2, "ITS;#it{p}_{T} (GeV/#it{c});#sigma(#Delta#it{p}_{T}/#it{p}_{T})");
  for (const auto& pdgCode : pdgCodes) {
    hPtResAllLayersITS[pdgCode]->Draw("same");
  }

  auto canvDcaxyResITS = new TCanvas("canvDcaxyResITS", "", 500, 500);
  canvDcaxyResITS->DrawFrame(ptLimits[0], 1., ptLimits[nPtBins - 1], 1.e3, "ITS;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{xy}}) (#mum)");
  canvDcaxyResITS->SetLogx();
  canvDcaxyResITS->SetLogy();
  canvDcaxyResITS->SetGrid();
  for (const auto& pdgCode : pdgCodes) {
    hDcaxyResAllLayersITS[pdgCode]->Draw("same");
  }

  auto canvDcazResITS = new TCanvas("canvDcazResITS", "", 500, 500);
  canvDcazResITS->DrawFrame(ptLimits[0], 1., ptLimits[nPtBins - 1], 1.e3, "ITS;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{z}}) (#mum)");
  canvDcazResITS->SetLogx();
  canvDcazResITS->SetLogy();
  canvDcazResITS->SetGrid();
  for (const auto& pdgCode : pdgCodes) {
    hDcazResAllLayersITS[pdgCode]->Draw("same");
  }

  auto canvDcaxyResNoFirstLayerITS = new TCanvas("canvDcaxyResNoFirstLayerITS", "", 500, 500);
  canvDcaxyResNoFirstLayerITS->DrawFrame(ptLimits[0], 1., ptLimits[nPtBins - 1], 1.e3, "ITS;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{xy}}) (#mum)");
  canvDcaxyResNoFirstLayerITS->SetLogx();
  canvDcaxyResNoFirstLayerITS->SetLogy();
  canvDcaxyResNoFirstLayerITS->SetGrid();
  for (const auto& pdgCode : pdgCodes) {
    hDcaxyResNoFirstLayerITS[pdgCode]->Draw("same");
  }

  auto canvDcazResNoFirstLayerITS = new TCanvas("canvDcazResNoFirstLayerITS", "", 500, 500);
  canvDcazResNoFirstLayerITS->DrawFrame(ptLimits[0], 1., ptLimits[nPtBins - 1], 1.e3, "ITS;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{z}}) (#mum)");
  canvDcazResNoFirstLayerITS->SetLogx();
  canvDcazResNoFirstLayerITS->SetLogy();
  canvDcazResNoFirstLayerITS->SetGrid();
  for (const auto& pdgCode : pdgCodes) {
    hDcazResNoFirstLayerITS[pdgCode]->Draw("same");
  }

  auto canvDcaxyReskAnyITS = new TCanvas("canvDcaxyReskAnyITS", "", 500, 500);
  canvDcaxyReskAnyITS->DrawFrame(ptLimits[0], 1., ptLimits[nPtBins - 1], 1.e3, "ITS;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{xy}}) (#mum)");
  canvDcaxyReskAnyITS->SetLogx();
  canvDcaxyReskAnyITS->SetLogy();
  canvDcaxyReskAnyITS->SetGrid();
  for (const auto& pdgCode : pdgCodes) {
    hDcaxyReskAnyITS[pdgCode]->Draw("same");
  }

  auto canvDcazReskAnyITS = new TCanvas("canvDcazReskAnyITS", "", 500, 500);
  canvDcazReskAnyITS->DrawFrame(ptLimits[0], 1., ptLimits[nPtBins - 1], 1.e3, "ITS;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{z}}) (#mum)");
  canvDcazReskAnyITS->SetLogx();
  canvDcazReskAnyITS->SetLogy();
  canvDcazReskAnyITS->SetGrid();
  for (const auto& pdgCode : pdgCodes) {
    hDcazReskAnyITS[pdgCode]->Draw("same");
  }

  // ITS-TPC
  auto canvPtDeltaITSTPC = new TCanvas("canvPtDeltaITSTPC", "", 1500, 500);
  canvPtDeltaITSTPC->Divide(nSpecies, 1);
  canvPtDeltaITSTPC->cd(1)->SetLogz();
  hDeltaPtVsPtAllLayersITSTPC[11]->Draw("colz");
  canvPtDeltaITSTPC->cd(2)->SetLogz();
  hDeltaPtVsPtAllLayersITSTPC[211]->Draw("colz");
  canvPtDeltaITSTPC->cd(3)->SetLogz();
  hDeltaPtVsPtAllLayersITSTPC[321]->Draw("colz");
  canvPtDeltaITSTPC->cd(4)->SetLogz();
  hDeltaPtVsPtAllLayersITSTPC[2212]->Draw("colz");

  auto canvDcaVsPtITSTPC = new TCanvas("canvDcaVsPtITSTPC", "", 1500, 1000);
  canvDcaVsPtITSTPC->Divide(nSpecies, 2);
  canvDcaVsPtITSTPC->cd(1)->SetLogz();
  hDcaxyVsPtAllLayersITSTPC[11]->Draw("colz");
  canvDcaVsPtITSTPC->cd(2)->SetLogz();
  hDcaxyVsPtAllLayersITSTPC[211]->Draw("colz");
  canvDcaVsPtITSTPC->cd(3)->SetLogz();
  hDcaxyVsPtAllLayersITSTPC[321]->Draw("colz");
  canvDcaVsPtITSTPC->cd(4)->SetLogz();
  hDcaxyVsPtAllLayersITSTPC[2212]->Draw("colz");
  canvDcaVsPtITSTPC->cd(5)->SetLogz();
  hDcazVsPtAllLayersITSTPC[11]->Draw("colz");
  canvDcaVsPtITSTPC->cd(6)->SetLogz();
  hDcazVsPtAllLayersITSTPC[211]->Draw("colz");
  canvDcaVsPtITSTPC->cd(7)->SetLogz();
  hDcazVsPtAllLayersITSTPC[321]->Draw("colz");
  canvDcaVsPtITSTPC->cd(8)->SetLogz();
  hDcazVsPtAllLayersITSTPC[2212]->Draw("colz");

  auto canvPtResITSTPC = new TCanvas("canvPtResITSTPC", "", 500, 500);
  canvPtResITSTPC->DrawFrame(ptLimits[0], 0., ptLimits[nPtBins - 1], 0.2, "ITS-TPC;#it{p}_{T} (GeV/#it{c});#sigma(#Delta#it{p}_{T}/#it{p}_{T})");
  for (const auto& pdgCode : pdgCodes) {
    hPtResAllLayersITSTPC[pdgCode]->Draw("same");
  }

  auto canvDcaxyResITSTPC = new TCanvas("canvDcaxyResITSTPC", "", 500, 500);
  canvDcaxyResITSTPC->DrawFrame(ptLimits[0], 1., ptLimits[nPtBins - 1], 1.e3, "ITS-TPC;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{xy}}) (#mum)");
  canvDcaxyResITSTPC->SetLogx();
  canvDcaxyResITSTPC->SetLogy();
  canvDcaxyResITSTPC->SetGrid();
  for (const auto& pdgCode : pdgCodes) {
    hDcaxyResAllLayersITSTPC[pdgCode]->Draw("same");
  }

  auto canvDcazResITSTPC = new TCanvas("canvDcazResITSTPC", "", 500, 500);
  canvDcazResITSTPC->DrawFrame(ptLimits[0], 1., ptLimits[nPtBins - 1], 1.e3, "ITS-TPC;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{z}}) (#mum)");
  canvDcazResITSTPC->SetLogx();
  canvDcazResITSTPC->SetLogy();
  canvDcazResITSTPC->SetGrid();
  for (const auto& pdgCode : pdgCodes) {
    hDcazResAllLayersITSTPC[pdgCode]->Draw("same");
  }

  auto canvDcaxyResNoFirstLayerITSTPC = new TCanvas("canvDcaxyResNoFirstLayerITSTPC", "", 500, 500);
  canvDcaxyResNoFirstLayerITSTPC->DrawFrame(ptLimits[0], 1., ptLimits[nPtBins - 1], 1.e3, "ITS-TPC;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{xy}}) (#mum)");
  canvDcaxyResNoFirstLayerITSTPC->SetLogx();
  canvDcaxyResNoFirstLayerITSTPC->SetLogy();
  canvDcaxyResNoFirstLayerITSTPC->SetGrid();
  for (const auto& pdgCode : pdgCodes) {
    hDcaxyResNoFirstLayerITSTPC[pdgCode]->Draw("same");
  }

  auto canvDcazResNoFirstLayerITSTPC = new TCanvas("canvDcazResNoFirstLayerITSTPC", "", 500, 500);
  canvDcazResNoFirstLayerITSTPC->DrawFrame(ptLimits[0], 1., ptLimits[nPtBins - 1], 1.e3, "ITS-TPC;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{z}}) (#mum)");
  canvDcazResNoFirstLayerITSTPC->SetLogx();
  canvDcazResNoFirstLayerITSTPC->SetLogy();
  canvDcazResNoFirstLayerITSTPC->SetGrid();
  for (const auto& pdgCode : pdgCodes) {
    hDcazResNoFirstLayerITSTPC[pdgCode]->Draw("same");
  }

  auto canvDcaxyReskAnyITSTPC = new TCanvas("canvDcaxyReskAnyITSTPC", "", 500, 500);
  canvDcaxyReskAnyITSTPC->DrawFrame(ptLimits[0], 1., ptLimits[nPtBins - 1], 1.e3, "ITS-TPC;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{xy}}) (#mum)");
  canvDcaxyReskAnyITSTPC->SetLogx();
  canvDcaxyReskAnyITSTPC->SetLogy();
  canvDcaxyReskAnyITSTPC->SetGrid();
  for (const auto& pdgCode : pdgCodes) {
    hDcaxyReskAnyITSTPC[pdgCode]->Draw("same");
  }

  auto canvDcazReskAnyITSTPC = new TCanvas("canvDcazReskAnyITSTPC", "", 500, 500);
  canvDcazReskAnyITSTPC->DrawFrame(ptLimits[0], 1., ptLimits[nPtBins - 1], 1.e3, "ITS-TPC;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{z}}) (#mum)");
  canvDcazReskAnyITSTPC->SetLogx();
  canvDcazReskAnyITSTPC->SetLogy();
  canvDcazReskAnyITSTPC->SetGrid();
  for (const auto& pdgCode : pdgCodes) {
    hDcazReskAnyITSTPC[pdgCode]->Draw("same");
  }

  // Compare ITS-TPC resolution;
  auto canvDcaxyResComp = new TCanvas("canvDcaxyResAllLayersComp", "", 500, 500);
  canvDcaxyResComp->DrawFrame(ptLimits[0], 1., ptLimits[nPtBins - 1], 1.e3, "ITS vs. ITS-TPC (all layers);#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{xy}}) (#mum)");
  canvDcaxyResComp->SetLogx();
  canvDcaxyResComp->SetLogy();
  canvDcaxyResComp->SetGrid();
  hDcaxyResAllLayersITS[211]->Draw("same");
  hDcaxyResAllLayersITSTPC[211]->Draw("same");
  gPad->BuildLegend(0.8, 0.8, 0.94, 0.94);

  auto canvDcazResComp = new TCanvas("canvDcazResAllLayersComp", "", 500, 500);
  canvDcazResComp->DrawFrame(ptLimits[0], 1., ptLimits[nPtBins - 1], 1.e3, "ITS vs. ITS-TPC (all layers);#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{z}}) (#mum)");
  canvDcazResComp->SetLogx();
  canvDcazResComp->SetLogy();
  canvDcazResComp->SetGrid();
  hDcazResAllLayersITS[211]->Draw("same");
  hDcazResAllLayersITSTPC[211]->Draw("same");
  gPad->BuildLegend(0.8, 0.8, 0.94, 0.94);

  auto canvPtResComp = new TCanvas("canvPtResAllLayersComp", "", 500, 500);
  canvPtResComp->DrawFrame(ptLimits[0], 0., ptLimits[nPtBins - 1], 0.2, "ITS vs. ITS-TPC (all layers);#it{p}_{T} (GeV/#it{c});#sigma(#Delta#it{p}_{T}/#it{p}_{T})");
  canvPtResComp->SetLogx();
  canvPtResComp->SetGrid();
  hPtResAllLayersITS[211]->Draw("same");
  hPtResAllLayersITSTPC[211]->Draw("same");
  gPad->BuildLegend(0.8, 0.8, 0.94, 0.94);

  auto canvDcaPtComp = new TCanvas("canvDcaPtAllLayersComp", "", 500, 500);
  canvDcaPtComp->Divide(2, 2);
  canvDcaPtComp->cd(1);
  hDcaxyVsPtAllLayersITS[211]->Draw();
  canvDcaPtComp->cd(2);
  hDcazVsPtAllLayersITS[211]->Draw();
  canvDcaPtComp->cd(3);
  hDcaxyVsPtAllLayersITSTPC[211]->Draw();
  canvDcaPtComp->cd(4);
  hDcazVsPtAllLayersITSTPC[211]->Draw();

  auto canvDcaPhiComp = new TCanvas("canvDcaPhiAllLayersComp", "", 500, 500);
  canvDcaPhiComp->Divide(2, 2);
  canvDcaPhiComp->cd(1);
  hDcaxyVsPhiAllLayersITS[211]->Draw();
  canvDcaPhiComp->cd(2);
  hDcazVsPhiAllLayersITS[211]->Draw();
  canvDcaPhiComp->cd(3);
  hDcaxyVsPhiAllLayersITSTPC[211]->Draw();
  canvDcaPhiComp->cd(4);
  hDcazVsPhiAllLayersITSTPC[211]->Draw();

  // Write
  TFile outFile("checkDCA.root", "RECREATE");
  outFile.mkdir("ITS");
  outFile.cd("ITS");
  gDirectory->WriteTObject(canvPtResITS);
  gDirectory->WriteTObject(canvDcaxyResITS);
  gDirectory->WriteTObject(canvDcazResITS);
  gDirectory->WriteTObject(canvDcazResNoFirstLayerITS);
  gDirectory->WriteTObject(canvDcaxyResNoFirstLayerITS);
  gDirectory->WriteTObject(canvDcaxyReskAnyITS);
  gDirectory->WriteTObject(canvDcazReskAnyITS);
  gDirectory->WriteTObject(canvPtDeltaITS);
  gDirectory->WriteTObject(canvDcaVsPtITS);

  outFile.mkdir("ITS-TPC");
  outFile.cd("ITS-TPC");
  gDirectory->WriteTObject(canvPtResITSTPC);
  gDirectory->WriteTObject(canvDcaxyResITSTPC);
  gDirectory->WriteTObject(canvDcazResITSTPC);
  gDirectory->WriteTObject(canvDcazResNoFirstLayerITSTPC);
  gDirectory->WriteTObject(canvDcaxyResNoFirstLayerITSTPC);
  gDirectory->WriteTObject(canvDcaxyReskAnyITSTPC);
  gDirectory->WriteTObject(canvDcazReskAnyITSTPC);
  gDirectory->WriteTObject(canvPtDeltaITSTPC);
  gDirectory->WriteTObject(canvDcaVsPtITSTPC);

  outFile.mkdir("Compare");
  outFile.cd("Compare");
  gDirectory->WriteTObject(canvDcaxyResComp);
  gDirectory->WriteTObject(canvDcazResComp);
  gDirectory->WriteTObject(canvPtResComp);
  gDirectory->WriteTObject(canvDcaPtComp);
  gDirectory->WriteTObject(canvDcaPhiComp);

  for (const auto& pdgCode : pdgCodes) {
    const char* dirName = partNames[pdgCode].c_str();
    auto dir = outFile.mkdir(dirName);
    outFile.cd(dirName);

    gDirectory->mkdir("ITS");
    gDirectory->cd("ITS");
    gDirectory->WriteTObject(hDeltaPtVsPtAllLayersITS[pdgCode]);
    gDirectory->WriteTObject(hDcaxyVsPtAllLayersITS[pdgCode]);
    gDirectory->WriteTObject(hDcazVsPtAllLayersITS[pdgCode]);
    gDirectory->WriteTObject(hDcazResAllLayersITS[pdgCode]);
    gDirectory->WriteTObject(hDcaxyResAllLayersITS[pdgCode]);
    gDirectory->WriteTObject(hDcaxyVsPtNoFirstLayerITS[pdgCode]);
    gDirectory->WriteTObject(hDcazVsPtNoFirstLayerITS[pdgCode]);
    gDirectory->WriteTObject(hDcazResNoFirstLayerITS[pdgCode]);
    gDirectory->WriteTObject(hDcaxyResNoFirstLayerITS[pdgCode]);
    gDirectory->WriteTObject(hDcaxyVsPhiAllLayersITS[pdgCode]);
    gDirectory->WriteTObject(hDcazVsPhiAllLayersITS[pdgCode]);

    dir->cd();
    gDirectory->mkdir("ITS-TPC");
    gDirectory->cd("ITS-TPC");
    gDirectory->WriteTObject(hDeltaPtVsPtAllLayersITSTPC[pdgCode]);
    gDirectory->WriteTObject(hDcaxyVsPtAllLayersITSTPC[pdgCode]);
    gDirectory->WriteTObject(hDcazVsPtAllLayersITSTPC[pdgCode]);
    gDirectory->WriteTObject(hDcazResAllLayersITSTPC[pdgCode]);
    gDirectory->WriteTObject(hDcaxyResAllLayersITSTPC[pdgCode]);
    gDirectory->WriteTObject(hDcaxyVsPtNoFirstLayerITSTPC[pdgCode]);
    gDirectory->WriteTObject(hDcazVsPtNoFirstLayerITSTPC[pdgCode]);
    gDirectory->WriteTObject(hDcazResNoFirstLayerITSTPC[pdgCode]);
    gDirectory->WriteTObject(hDcaxyResNoFirstLayerITSTPC[pdgCode]);
    gDirectory->WriteTObject(hDcaxyVsPhiAllLayersITSTPC[pdgCode]);
    gDirectory->WriteTObject(hDcazVsPhiAllLayersITSTPC[pdgCode]);
  }
}
