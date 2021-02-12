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
#include "AnalysisDataModel/PID/PIDResponse.h"
#include "AnalysisDataModel/TrackSelectionTables.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;
using namespace o2::track::pid_constants;
using namespace o2::track;

struct TPCSpectraTaskAuto {
  static constexpr ID id = PID::Pion;
  static constexpr int Np = 9;
  static constexpr const char* pT[Np] = {"e", "#mu", "#pi", "K", "p", "d", "t", "^{3}He", "#alpha"};
  HistogramRegistry histos{"Histos", {}, OutputObjHandlingPolicy::AnalysisObject};

  void init(o2::framework::InitContext&)
  {
    histos.add("p/Unselected", "Unselected;#it{p} (GeV/#it{c})", kTH1F, {{100, 0, 20}});
    histos.add("pt/Unselected", "Unselected;#it{p}_{T} (GeV/#it{c})", kTH1F, {{100, 0, 20}});
    histos.add("p/Selected", Form("Selected %s;#it{p} (GeV/#it{c})", pT[id]), kTH1F, {{100, 0, 20}});
    histos.add("pt/Selected", Form("Selected %s;#it{p}_{T} (GeV/#it{c})", pT[id]), kTH1F, {{100, 0, 20}});
  }

  Configurable<float> cfgCutVertex{"cfgCutVertex", 10.0f, "Accepted z-vertex range"};
  Configurable<float> cfgCutEta{"cfgCutEta", 0.8f, "Eta range for tracks"};
  Configurable<float> nsigmacut{"nsigmacut", 3, "Value of the Nsigma cut"};

  Filter collisionFilter = nabs(aod::collision::posZ) < cfgCutVertex;
  Filter trackFilter = (nabs(aod::track::eta) < cfgCutEta) && (aod::track::isGlobalTrack == (uint8_t) true);
  using TrackCandidates = soa::Filtered<soa::Join<aod::Tracks, aod::TracksExtra,
                                                  aod::AutoPIDTPCPi,
                                                  aod::TrackSelection>>;

  void process(TrackCandidates::iterator const& track)
  {
    histos.fill(HIST("p/Unselected"), track.p());
    histos.fill(HIST("pt/Unselected"), track.pt());

    const float nsigma = -999.f;
    // const float nsigma = track.AutoTPCNSigmaPi();
    // if constexpr (id == PID::kElectron) {
    //   sigma = track.tpcNSigmaEl();
    // } else if constexpr (id == PID::kMuon) {
    //   sigma = track.tpcNSigmaMu();
    // } else if constexpr (id == PID::Pion) {
    //   sigma = track.tpcNSigmaMu();
    // } else if constexpr (id == PID::Kaon) {
    //   sigma = track.tpcNSigmaMu();
    // } else if constexpr (id == PID::Proton) {
    //   sigma = track.tpcNSigmaMu();
    // } else if constexpr (id == PID::Deuteron) {
    //   sigma = track.tpcNSigmaMu();
    // } else if constexpr (id == PID::Triton) {
    //   sigma = track.tpcNSigmaMu();
    // } else if constexpr (id == PID::Helium3) {
    //   sigma = track.tpcNSigmaMu();
    // } else if constexpr (id == PID::Alpha) {
    //   sigma = track.tpcNSigmaMu();
    // }

    if (abs(nsigma) > nsigmacut.value) {
      return;
    }
    histos.fill(HIST("p/Selected"), track.p());
    histos.fill(HIST("p/Selected"), track.pt());
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  WorkflowSpec workflow{adaptAnalysisTask<TPCSpectraTaskAuto>("tpcspectra-auto-task-pi")};
  return workflow;
}
