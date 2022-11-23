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

struct EtaAndClsHistograms {
  OutputObj<TH3F> etaClsH{TH3F("eta_vs_cls_vs_sigmapT", "#eta vs N_{cls} vs sigma_{1/pT}", 102, -2.01, 2.01, 160, -0.5, 159.5, 100, 0, 10)};

  void process(soa::Join<aod::FullTracks, aod::TracksCov> const& tracks)
  {
    for (auto& track : tracks) {
      etaClsH->Fill(track.eta(), track.tpcNClsFindable(), track.sigma1Pt());
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{
    adaptAnalysisTask<EtaAndClsHistograms>(cfgc),
  };
}
