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
#include <cmath>

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;

struct EtaAndClsHistogramsSimple {
  OutputObj<TH2F> etaClsH{TH2F("eta_vs_pt", "#eta vs pT", 102, -2.01, 2.01, 100, 0, 10)};

  void process(aod::Tracks const& tracks)
  {
    LOGP(info, "Invoking the simple one");
    for (auto& track : tracks) {
      etaClsH->Fill(track.eta(), track.pt(), 0);
    }
  }
};

struct EtaAndClsHistogramsFull {
  OutputObj<TH3F> etaClsH{TH3F("eta_vs_cls_vs_sigmapT", "#eta vs N_{cls} vs sigma_{1/pT}", 102, -2.01, 2.01, 160, -0.5, 159.5, 100, 0, 10)};

  void process(soa::Join<aod::FullTracks, aod::TracksCov> const& tracks)
  {
    LOGP(info, "Invoking the run 3 one");
    for (auto& track : tracks) {
      etaClsH->Fill(track.eta(), track.tpcNClsFindable(), track.sigma1Pt());
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
  for (auto& table : tables) {
    if (table.starts_with("O2trackcov")) {
      hasTrackCov = true;
    }
    LOGP(info, "- {} present.", table);
  }
  // Notice it's important for the tasks to use the same name, otherwise topology generation will be confused.
  if (runType == "2" || !hasTrackCov) {
    LOGP(info, "Using only tracks {}", runType);
    return WorkflowSpec{
      adaptAnalysisTask<EtaAndClsHistogramsSimple>(cfgc, TaskName{"simple-histos"}),
    };
  } else {
    LOGP(info, "Using tracks extra {}", runType);
    return WorkflowSpec{
      adaptAnalysisTask<EtaAndClsHistogramsFull>(cfgc, TaskName{"simple-histos"}),
    };
  }
}
