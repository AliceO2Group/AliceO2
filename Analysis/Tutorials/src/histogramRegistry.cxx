// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "Framework/AnalysisDataModel.h"
#include "Framework/HistogramRegistry.h"
#include <TH1F.h>

#include <cmath>

using namespace o2;
using namespace o2::framework;

// This is a very simple example showing how to create an histogram
// FIXME: this should really inherit from AnalysisTask but
//        we need GCC 7.4+ for that
struct ATask {
  /// Construct a registry object with direct declaration
  HistogramRegistry registry{
    "registry",
    true,
    {
      {"eta", "#eta", {HistType::kTH1F, {{102, -2.01, 2.01}}}},     //
      {"phi", "#varphi", {HistType::kTH1F, {{100, 0., 2. * M_PI}}}} //
    }                                                               //
  };

  void process(aod::Tracks const& tracks)
  {
    for (auto& track : tracks) {
      registry.get<TH1>("eta")->Fill(track.eta());
      registry.get<TH1>("phi")->Fill(track.phi());
    }
  }
};

struct BTask {
  /// Construct a registry object with direct declaration
  HistogramRegistry registry{
    "registry",
    true,
    {
      {"eta", "#eta", {HistType::kTH1F, {{102, -2.01, 2.01}}}},                            //
      {"ptToPt", "#ptToPt", {HistType::kTH2F, {{100, -0.01, 10.01}, {100, -0.01, 10.01}}}} //
    }                                                                                      //
  };

  void process(aod::Tracks const& tracks)
  {
    registry.fill<aod::track::Eta>("eta", tracks, aod::track::eta > 0.0f);
    registry.fill<aod::track::Pt, aod::track::Pt>("ptToPt", tracks, aod::track::pt < 5.0f);
  }
};

struct CTask {

  HistogramRegistry registry{
    "registry",
    true,
    {
      {"1d", "test 1d", {HistType::kTH1I, {{100, -10.0f, 10.0f}}}},                                                                                               //
      {"2d", "test 2d", {HistType::kTH2F, {{100, -10.0f, 10.01f}, {100, -10.0f, 10.01f}}}},                                                                       //
      {"3d", "test 3d", {HistType::kTH3D, {{100, -10.0f, 10.01f}, {100, -10.0f, 10.01f}, {100, -10.0f, 10.01f}}}},                                                //
      {"4d", "test 4d", {HistType::kTHnC, {{100, -10.0f, 10.01f}, {100, -10.0f, 10.01f}, {100, -10.0f, 10.01f}, {100, -10.0f, 10.01f}}}},                         //
      {"5d", "test 5d", {HistType::kTHnSparseL, {{10, -10.0f, 10.01f}, {10, -10.0f, 10.01f}, {10, -10.0f, 10.01f}, {10, -10.0f, 10.01f}, {10, -10.0f, 10.01f}}}}, //
    }                                                                                                                                                             //
  };

  void init(o2::framework::InitContext&)
  {
    registry.add({"7d", "test 7d", {HistType::kTHnC, {{3, -10.0f, 10.01f}, {3, -10.0f, 10.01f}, {3, -10.0f, 10.01f}, {3, -10.0f, 10.01f}, {3, -10.0f, 10.01f}, {3, -10.0f, 10.01f}, {3, -10.0f, 10.01f}}}});

    registry.add({"6d", "test 6d", {HistType::kTHnC, {{3, -10.0f, 10.01f}, {3, -10.0f, 10.01f}, {3, -10.0f, 10.01f}, {3, -10.0f, 10.01f}, {3, -10.0f, 10.01f}, {3, -10.0f, 10.01f}}}});

    registry.add({"1d-profile", "test 1d profile", {HistType::kTProfile, {{20, 0.0f, 10.01f}}}});
    registry.add({"2d-profile", "test 2d profile", {HistType::kTProfile2D, {{20, 0.0f, 10.01f}, {20, 0.0f, 10.01f}}}});
    registry.add({"3d-profile", "test 3d profile", {HistType::kTProfile3D, {{20, 0.0f, 10.01f}, {20, 0.0f, 10.01f}, {20, 0.0f, 10.01f}}}});

    registry.add({"2d-weight", "test 2d weight", {HistType::kTH2C, {{2, -10.0f, 10.01f}, {2, -10.0f, 10.01f}}}, true});

    registry.add({"3d-weight", "test 3d weight", {HistType::kTH3C, {{2, -10.0f, 10.01f}, {2, -10.0f, 10.01f}, {2, -10.0f, 10.01f}}}, true});

    registry.add({"4d-weight", "test 4d weight", {HistType::kTHnC, {{2, -10.0f, 10.01f}, {2, -10.0f, 10.01f}, {2, -10.0f, 10.01f}, {100, -10.0f, 10.01f}}}, true});

    registry.add({"1d-profile-weight", "test 1d profile weight", {HistType::kTProfile, {{2, -10.0f, 10.01f}}}, true});
    registry.add({"2d-profile-weight", "test 2d profile weight", {HistType::kTProfile2D, {{2, -10.0f, 10.01f}, {2, -10.0f, 10.01f}}}, true});
  }

  void process(aod::Tracks const& tracks)
  {
    using namespace aod::track;
    // does not work with dynamic columns (e.g. Charge, NormalizedPhi)
    registry.fill<Eta>("1d", tracks, eta > -0.7f);
    registry.fill<Pt, Eta, RawPhi>("3d", tracks, eta > 0.f);
    registry.fill<Pt, Eta, RawPhi, P, X>("5d", tracks, pt > 0.15f);
    registry.fill<Pt, Eta, RawPhi, P, X, Y, Z>("7d", tracks, pt > 0.15f);
    registry.fill<Pt, Eta, RawPhi>("2d-profile", tracks, eta > -0.5f);

    // fill 4d histogram with weight (column X)
    registry.fillWeight<Pt, Eta, RawPhi, Z, X>("4d-weight", tracks, eta > 0.f);

    registry.fillWeight<Pt, Eta, RawPhi>("2d-weight", tracks, eta > 0.f);

    registry.fillWeight<Pt, Eta, RawPhi>("1d-profile-weight", tracks, eta > 0.f);

    for (auto& track : tracks) {
      registry.fill("2d", track.eta(), track.pt());
      registry.fill("4d", track.pt(), track.eta(), track.phi(), track.signed1Pt());
      registry.fill("6d", track.pt(), track.eta(), track.phi(), track.snp(), track.tgl(), track.alpha());
      registry.fill("1d-profile", track.pt(), track.eta());
      registry.fill("3d-profile", track.pt(), track.eta(), track.phi(), track.snp());

      // fill 3d histogram with weight (2.)
      registry.fillWeight("3d-weight", track.pt(), track.eta(), track.phi(), 2.);

      registry.fillWeight("2d-profile-weight", track.pt(), track.eta(), track.phi(), 5.);
    }
  }
};

struct DTask {
  HistogramRegistry spectra{"spectra", true, {}};
  HistogramRegistry etaStudy{"etaStudy", true, {}};

  void init(o2::framework::InitContext&)
  {
    std::vector<double> ptBinning = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                                     1.1, 1.2, 1.3, 1.4, 1.5, 2.0, 5.0, 10.0, 20.0, 50.0};
    std::vector<double> centBinning = {0., 30., 60., 90.};

    AxisSpec ptAxis = {ptBinning, "#it{p}_{T} (GeV/c)"};
    AxisSpec centAxis = {centBinning, "centrality"};
    AxisSpec etaAxis = {5, -0.8, 0.8, "#eta"};
    AxisSpec phiAxis = {4, 0., 2. * M_PI, "#phi"};
    const int nCuts = 5;
    AxisSpec cutAxis = {nCuts, -0.5, nCuts - 0.5, "cut setting"};

    HistogramConfigSpec defaultParticleHist({HistType::kTHnF, {ptAxis, etaAxis, centAxis, cutAxis}});

    spectra.add("myControlHist", "a", kTH2F, {ptAxis, etaAxis});
    spectra.get<TH2>("myControlHist")->GetYaxis()->SetTitle("my-y-axis");
    spectra.get<TH2>("myControlHist")->SetTitle("something meaningful");

    spectra.add("charged/pions", "Pions", defaultParticleHist);
    spectra.add("neutral/pions", "Pions", defaultParticleHist);
    spectra.add("one/two/three/four/kaons", "Kaons", defaultParticleHist);
    spectra.add("sigmas", "Sigmas", defaultParticleHist);
    spectra.add("lambdas", "Lambd", defaultParticleHist);

    spectra.get<THn>("lambdas")->SetTitle("Lambdas");

    etaStudy.add("positive", "A side spectra", kTH1I, {ptAxis});
    etaStudy.add("negative", "C side spectra", kTH1I, {ptAxis});
  }

  void process(aod::Tracks const& tracks)
  {
    using namespace aod::track;

    etaStudy.fill<Pt>("positive", tracks, eta > 0.f);
    etaStudy.fill<Pt>("negative", tracks, eta < 0.f);

    for (auto& track : tracks) {
      spectra.fill("myControlHist", track.pt(), track.eta());
      spectra.fill("charged/pions", track.pt(), track.eta(), 50., 0.);
      spectra.fill("charged/pions", track.pt(), track.eta(), 50., 0.);
      spectra.fill("neutral/pions", track.pt(), track.eta(), 50., 0.);
      spectra.fill("one/two/three/four/kaons", track.pt(), track.eta(), 50., 0.);
      spectra.fill("sigmas", track.pt(), track.eta(), 50., 0.);
      spectra.fill("lambdas", track.pt(), track.eta(), 50., 0.);
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<ATask>("eta-and-phi-histograms"),
    adaptAnalysisTask<BTask>("filtered-histograms"),
    adaptAnalysisTask<CTask>("dimension-test"),
    adaptAnalysisTask<DTask>("realistic-example")};
}
