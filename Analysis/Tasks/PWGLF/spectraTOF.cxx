// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

// O2 includes
#include "ReconstructionDataFormats/Track.h"
#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "Framework/AnalysisDataModel.h"
#include "Framework/ASoAHelpers.h"
#include "PID/PIDResponse.h"
#include "Analysis/TrackSelectionTables.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;

const int Np = 9;
const TString pN[Np] = {"El", "Mu", "Pi", "Ka", "Pr", "De", "Tr", "He", "Al"};
const TString pT[Np] = {"#mu", "#pi", "K", "p", "d", "t", "^{3}He", "#alpha"};
struct TOFSpectraTask {
  HistogramRegistry histos{"Histos", {}, OutputObjHandlingPolicy::AnalysisObject};

  void init(o2::framework::InitContext&)
  {
    histos.add("p/Unselected", "Unselected;#it{p} (GeV/#it{c})", kTH1F, {{100, 0, 20}});
    histos.add("pt/Unselected", "Unselected;#it{p}_{T} (GeV/#it{c})", kTH1F, {{100, 0, 20}});
    for (int i = 0; i < Np; i++) {
      histos.add(Form("p/%s", pN[i].Data()), Form("%s;#it{p} (GeV/#it{c})", pT[i].Data()), kTH1F, {{100, 0, 20}});
      histos.add(Form("pt/%s", pN[i].Data()), Form("%s;#it{p}_{T} (GeV/#it{c})", pT[i].Data()), kTH1F, {{100, 0, 20}});
    }
    histos.add("electronbeta/hp_El", ";#it{p} (GeV/#it{c})", kTH1F, {{100, 0, 20}});
    histos.add("electronbeta/hpt_El", ";#it{p}_{T} (GeV/#it{c})", kTH1F, {{100, 0, 20}});
    histos.add("electronbeta/hlength_El", ";Track Length (cm);Tracks", kTH1D, {{100, 0, 1000}});
    histos.add("electronbeta/htime_El", ";TOF Time (ns);Tracks", kTH1D, {{1000, 0, 600}});
    histos.add("electronbeta/hp_beta_El", ";#it{p} (GeV/#it{c});#beta - #beta_{e};Tracks", kTH2D, {{100, 0, 20}, {100, -0.01, 0.01}});
    histos.add("electronbeta/hp_betasigma_El", ";#it{p} (GeV/#it{c});(#beta - #beta_{e})/#sigma;Tracks", kTH2D, {{100, 0, 20}, {100, -5, 5}});
  }

  Configurable<float> nsigmacut{"nsigmacut", 3, "Value of the Nsigma cut"};

  Filter trackFilter = aod::track::isGlobalTrack == true;

  using TrackCandidates = soa::Filtered<soa::Join<aod::Tracks, aod::TracksExtra, aod::pidRespTOF, aod::pidRespTOFbeta, aod::TrackSelection>>;
  void process(TrackCandidates::iterator const& track)
  {
    const float nsigma[Np] = {track.tofNSigmaEl(), track.tofNSigmaMu(), track.tofNSigmaPi(),
                              track.tofNSigmaKa(), track.tofNSigmaPr(), track.tofNSigmaDe(),
                              track.tofNSigmaTr(), track.tofNSigmaHe(), track.tofNSigmaAl()};
    histos.fill("p/Unselected", track.p());
    histos.fill("pt/Unselected", track.pt());
    for (int i = 0; i < Np; i++) {
      if (abs(nsigma[i]) > nsigmacut.value) {
        continue;
      }
      histos.fill(Form("p/%s", pN[i].Data()), track.p());
      histos.fill(Form("pt/%s", pN[i].Data()), track.pt());
    }
    //
    if (TMath::Abs(track.separationbetael() < 1.f)) {
      histos.fill("electronbeta/hp_El", track.p());
      histos.fill("electronbeta/hpt_El", track.pt());
      histos.fill("electronbeta/hlength_El", track.length());
      histos.fill("electronbeta/htime_El", track.tofSignal() / 1000);
      histos.fill("electronbeta/hp_beta_El", track.p(), track.diffbetael());
      histos.fill("electronbeta/hp_betasigma_El", track.p(), track.separationbetael());
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  WorkflowSpec workflow{adaptAnalysisTask<TOFSpectraTask>("tofspectra-task")};
  return workflow;
}
