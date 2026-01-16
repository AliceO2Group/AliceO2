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
///
/// \brief FullTracks is a join of Tracks, TracksCov, and TracksExtra.
/// \author
/// \since

#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include <TH2F.h>

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;

namespace o2::aod
{
O2ORIGIN("EMB");
namespace skimmedExampleTrack
{
DECLARE_SOA_COLUMN(Pt, pt, float);   //!
DECLARE_SOA_COLUMN(Eta, eta, float); //!
} // namespace skimmedExampleTrack

DECLARE_SOA_TABLE(SkimmedExampleTrack, "AOD", "SKIMEXTRK", //!
                  skimmedExampleTrack::Pt,
                  skimmedExampleTrack::Eta);
} // namespace o2::aod

struct EtaAndClsHistogramsSimple {
  OutputObj<TH2F> etaClsH{TH2F("eta_vs_pt", "#eta vs pT", 102, -2.01, 2.01, 100, 0, 10)};
  Produces<o2::aod::SkimmedExampleTrack> skimEx;
  Configurable<std::string> trackFilterString{"track-filter", "o2::aod::track::pt < 10.f", "Track filter string"};
  Filter trackFilter = o2::aod::track::pt < 10.f;

  HistogramRegistry registry{
    "registry",
    {
      {"a/b/eta", "#Eta", {HistType::kTH1F, {{100, -2.0, 2.0}}}},                          //
      {"a/phi", "#Phi", {HistType::kTH1D, {{102, 0, 2 * M_PI}}}},                          //
      {"c/pt", "p_{T}", {HistType::kTH1D, {{1002, -0.01, 50.1}}}},                         //
      {"ptToPt", "#ptToPt", {HistType::kTH2F, {{100, -0.01, 10.01}, {100, -0.01, 10.01}}}} //
    } //
  };

  HistogramRegistry registry2{
    "registry2",
    {
      {"a/foo/b/eta", "#Eta", {HistType::kTH1F, {{100, -2.0, 2.0}}}},                           //
      {"fii/c/hpt", "p_{T}", {HistType::kTH1D, {{1002, -0.01, 50.1}}}},                         //
      {"a/foobar/phi", "#Phi", {HistType::kTH1D, {{102, 0, 2 * M_PI}}}},                        //
      {"fifi/ptToPt", "#ptToPt", {HistType::kTH2F, {{100, -0.01, 10.01}, {100, -0.01, 10.01}}}} //
    },
    OutputObjHandlingPolicy::AnalysisObject,
    false,
    true};

  void init(InitContext&)
  {
    if (!trackFilterString->empty()) {
      trackFilter = Parser::parse((std::string)trackFilterString);
    }
  }

  void process(soa::Filtered<aod::Tracks> const& tracks, aod::FT0s const&)
  {
    LOGP(info, "Invoking the simple one");
    for (auto& track : tracks) {
      etaClsH->Fill(track.eta(), track.pt());
      skimEx(track.pt(), track.eta());

      registry.fill(HIST("a/b/eta"), track.eta());
      registry.fill(HIST("a/phi"), track.phi());
      registry.fill(HIST("c/pt"), track.pt());
      registry.fill(HIST("ptToPt"), track.pt(), track.signed1Pt());
    }
  }
};

struct EtaAndClsHistogramsIUSimple {
  OutputObj<TH2F> etaClsH{TH2F("eta_vs_pt", "#eta vs pT", 102, -2.01, 2.01, 100, 0, 10)};
  Produces<o2::aod::SkimmedExampleTrack> skimEx;
  Configurable<std::string> trackFilterString{"track-filter", "o2::aod::track::pt < 10.f", "Track filter string"};
  Filter trackFilter = o2::aod::track::pt < 10.f;

  HistogramRegistry registry{
    "registry",
    {
      {"a/b/eta", "#Eta", {HistType::kTH1F, {{100, -2.0, 2.0}}}},                          //
      {"a/phi", "#Phi", {HistType::kTH1D, {{102, 0, 2 * M_PI}}}},                          //
      {"c/pt", "p_{T}", {HistType::kTH1D, {{1002, -0.01, 50.1}}}},                         //
      {"ptToPt", "#ptToPt", {HistType::kTH2F, {{100, -0.01, 10.01}, {100, -0.01, 10.01}}}} //
    } //
  };

  void init(InitContext&)
  {
    if (!trackFilterString->empty()) {
      trackFilter = Parser::parse((std::string)trackFilterString);
    }
  }

  void process(soa::Filtered<aod::TracksIU> const& tracks, aod::FT0s const&)
  {
    LOGP(info, "Invoking the simple one IU");
    for (auto& track : tracks) {
      etaClsH->Fill(track.eta(), track.pt());
      skimEx(track.pt(), track.eta());

      registry.fill(HIST("a/b/eta"), track.eta());
      registry.fill(HIST("a/phi"), track.phi());
      registry.fill(HIST("c/pt"), track.pt());
      registry.fill(HIST("ptToPt"), track.pt(), track.signed1Pt());
    }
  }
};

struct EtaAndClsHistogramsFull {
  OutputObj<TH3F> etaClsH{TH3F("eta_vs_cls_vs_sigmapT", "#eta vs N_{cls} vs sigma_{1/pT}", 102, -2.01, 2.01, 160, -0.5, 159.5, 100, 0, 10)};

  HistogramRegistry registry{
    "registry",
    {
      {"a/b/eta", "#Eta", {HistType::kTH1F, {{100, -2.0, 2.0}}}},                          //
      {"a/phi", "#Phi", {HistType::kTH1D, {{102, 0, 2 * M_PI}}}},                          //
      {"c/pt", "p_{T}", {HistType::kTH1D, {{1002, -0.01, 50.1}}}},                         //
      {"ptToPt", "#ptToPt", {HistType::kTH2F, {{100, -0.01, 10.01}, {100, -0.01, 10.01}}}} //
    } //
  };

  Configurable<std::string> trackFilterString{"track-filter", "o2::aod::track::pt < 10.f", "Track filter string"};
  Filter trackFilter = o2::aod::track::pt < 10.f;

  void init(InitContext&)
  {
    if (!trackFilterString->empty()) {
      trackFilter = Parser::parse((std::string)trackFilterString);
    }
  }

  void process(soa::Filtered<soa::Join<aod::FullTracks, aod::TracksCov>> const& tracks)
  {
    LOGP(info, "Invoking the run 3 one");
    for (auto& track : tracks) {
      etaClsH->Fill(track.eta(), track.tpcNClsFindable(), track.sigma1Pt());

      registry.fill(HIST("a/b/eta"), track.eta());
      registry.fill(HIST("a/phi"), track.phi());
      registry.fill(HIST("c/pt"), track.pt());
      registry.fill(HIST("ptToPt"), track.pt(), track.signed1Pt());
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  std::string runType = "3";
  std::vector<std::string> tables;
  if (cfgc.options().hasOption("aod-metadata-Run")) {
    runType = cfgc.options().get<std::string>("aod-metadata-Run");
  }
  if (cfgc.options().hasOption("aod-metadata-tables")) {
    tables = cfgc.options().get<std::vector<std::string>>("aod-metadata-tables");
  }
  LOGP(info, "Runtype is {}", runType);
  bool hasTrackCov = false;
  bool hasTrackIU = false;
  for (auto& table : tables) {
    if (table == "O2trackcov") {
      hasTrackCov = true;
    }
    if (table.starts_with("O2track_iu")) {
      hasTrackIU = true;
    }
    LOGP(info, "- {} present.", table);
  }
  // Notice it's important for the tasks to use the same name, otherwise topology generation will be confused.
  if (runType == "2" || !hasTrackCov) {
    LOGP(info, "Using only tracks {}", runType);
    if (hasTrackIU) {
      return WorkflowSpec{
        adaptAnalysisTask<EtaAndClsHistogramsIUSimple>(cfgc, TaskName{"simple-histos"}),
      };
    }
    return WorkflowSpec{
      adaptAnalysisTask<EtaAndClsHistogramsSimple>(cfgc, TaskName{"simple-histos"}),
    };
  } else {
    LOGP(info, "Using tracks extra {}", runType);
    if (hasTrackIU) {
      return WorkflowSpec{
        adaptAnalysisTask<EtaAndClsHistogramsIUSimple>(cfgc, TaskName{"simple-histos"}),
      };
    }
    return WorkflowSpec{
      adaptAnalysisTask<EtaAndClsHistogramsFull>(cfgc, TaskName{"simple-histos"}),
    };
  }
}
