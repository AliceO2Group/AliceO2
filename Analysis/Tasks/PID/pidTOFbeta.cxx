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
#include "Framework/AnalysisTask.h"
#include "Framework/HistogramRegistry.h"
#include "AnalysisDataModel/TrackSelectionTables.h"
#include "AnalysisDataModel/PID/PIDResponse.h"
#include "AnalysisDataModel/PID/PIDTOF.h"

using namespace o2;
using namespace o2::pid;
using namespace o2::track;
using namespace o2::framework;
using namespace o2::framework::expressions;

void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  std::vector<ConfigParamSpec> options{{"add-qa", VariantType::Int, 0, {"Produce TOF PID QA histograms"}}};
  std::swap(workflowOptions, options);
}

#include "Framework/runDataProcessing.h"

struct pidTOFTaskBeta {
  using Trks = soa::Join<aod::Tracks, aod::TracksExtra>;
  using Coll = aod::Collision;
  Produces<aod::pidTOFbeta> tablePIDBeta;
  tof::Beta<Coll, Trks::iterator, PID::Electron> responseElectron;
  Configurable<float> expreso{"tof-expreso", 80, "Expected resolution for the computation of the expected beta"};

  void init(o2::framework::InitContext&)
  {
    responseElectron.mExpectedResolution = expreso.value;
  }

  void process(Coll const& collision, Trks const& tracks)
  {
    tablePIDBeta.reserve(tracks.size());
    for (auto const& trk : tracks) {
      tablePIDBeta(responseElectron.GetBeta(collision, trk),
                   responseElectron.GetExpectedSigma(collision, trk),
                   responseElectron.GetExpectedSignal(collision, trk),
                   responseElectron.GetExpectedSigma(collision, trk),
                   responseElectron.GetSeparation(collision, trk));
    }
  }
};

struct pidTOFTaskQABeta {

  static constexpr int Np = 9;
  static constexpr const char* pT[Np] = {"e", "#mu", "#pi", "K", "p", "d", "t", "^{3}He", "#alpha"};
  static constexpr std::string_view hexpected[Np] = {"expected/El", "expected/Mu", "expected/Pi",
                                                     "expected/Ka", "expected/Pr", "expected/De",
                                                     "expected/Tr", "expected/He", "expected/Al"};
  static constexpr std::string_view hexpected_diff[Np] = {"expected_diff/El", "expected_diff/Mu", "expected_diff/Pi",
                                                          "expected_diff/Ka", "expected_diff/Pr", "expected_diff/De",
                                                          "expected_diff/Tr", "expected_diff/He", "expected_diff/Al"};
  static constexpr std::string_view hnsigma[Np] = {"nsigma/El", "nsigma/Mu", "nsigma/Pi",
                                                   "nsigma/Ka", "nsigma/Pr", "nsigma/De",
                                                   "nsigma/Tr", "nsigma/He", "nsigma/Al"};
  HistogramRegistry histos{"Histos", {}, OutputObjHandlingPolicy::QAObject};

  Configurable<int> logAxis{"logAxis", 1, "Flag to use a log momentum axis"};
  Configurable<int> nBinsP{"nBinsP", 400, "Number of bins for the momentum"};
  Configurable<float> MinP{"MinP", 0.1f, "Minimum momentum in range"};
  Configurable<float> MaxP{"MaxP", 5.f, "Maximum momentum in range"};
  Configurable<int> nBinsDelta{"nBinsDelta", 200, "Number of bins for the Delta"};
  Configurable<float> MinDelta{"MinDelta", -1000.f, "Minimum Delta in range"};
  Configurable<float> MaxDelta{"MaxDelta", 1000.f, "Maximum Delta in range"};
  Configurable<int> nBinsNSigma{"nBinsNSigma", 200, "Number of bins for the NSigma"};
  Configurable<float> MinNSigma{"MinNSigma", -10.f, "Minimum NSigma in range"};
  Configurable<float> MaxNSigma{"MaxNSigma", 10.f, "Maximum NSigma in range"};

  template <typename T>
  void makelogaxis(T h)
  {
    if (logAxis == 0) {
      return;
    }
    const int nbins = h->GetNbinsX();
    double binp[nbins + 1];
    double max = h->GetXaxis()->GetBinUpEdge(nbins);
    double min = h->GetXaxis()->GetBinLowEdge(1);
    if (min <= 0) {
      min = 0.00001;
    }
    double lmin = TMath::Log10(min);
    double ldelta = (TMath::Log10(max) - lmin) / ((double)nbins);
    for (int i = 0; i < nbins; i++) {
      binp[i] = TMath::Exp(TMath::Log(10) * (lmin + i * ldelta));
    }
    binp[nbins] = max + 1;
    h->GetXaxis()->Set(nbins, binp);
  }

  void init(o2::framework::InitContext&)
  {
    // Event properties
    histos.add("event/vertexz", ";Vtx_{z} (cm);Entries", HistType::kTH1F, {{100, -20, 20}});
    histos.add("event/colltime", ";Collision time (ps);Entries", HistType::kTH1F, {{100, -2000, 2000}});
    histos.add("event/tofsignal", ";#it{p} (GeV/#it{c});TOF Signal", HistType::kTH2F, {{nBinsP, MinP, MaxP}, {10000, 0, 2e6}});
    makelogaxis(histos.get<TH2>(HIST("event/tofsignal")));
    histos.add("event/tofbeta", ";#it{p} (GeV/#it{c});TOF #beta", HistType::kTH2F, {{nBinsP, MinP, MaxP}, {1000, 0, 2}});
    makelogaxis(histos.get<TH2>(HIST("event/tofbeta")));
    histos.add("event/eta", ";#it{#eta};Entries", HistType::kTH1F, {{100, -2, 2}});
    histos.add("event/length", ";Track length (cm);Entries", HistType::kTH1F, {{100, 0, 500}});
    histos.add("event/pt", ";#it{p}_{T} (GeV/#it{c});Entries", HistType::kTH1F, {{nBinsP, MinP, MaxP}});
    histos.add("event/p", ";#it{p} (GeV/#it{c});Entries", HistType::kTH1F, {{nBinsP, MinP, MaxP}});
    histos.add("event/ptreso", ";#it{p} (GeV/#it{c});Entries", HistType::kTH2F, {{nBinsP, MinP, MaxP}, {100, 0, 0.1}});
  }

  template <uint8_t i, typename T>
  void fillParticleHistos(const T& t, const float tof, const float exp_diff, const float nsigma)
  {
    histos.fill(HIST(hexpected[i]), t.p(), tof - exp_diff);
    histos.fill(HIST(hexpected_diff[i]), t.p(), exp_diff);
    histos.fill(HIST(hnsigma[i]), t.p(), nsigma);
  }

  void process(soa::Join<aod::Tracks, aod::TracksExtra, aod::TracksCov, aod::pidTOFbeta, aod::TrackSelection>::iterator const& track)
  {
    //
    if (!track.hasTOF()) { // Skipping tracks without TOF
      return;
    }
    if (!track.isGlobalTrack()) {
      return;
    }
    histos.fill(HIST("event/tofbeta"), track.p(), track.beta());
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  auto workflow = WorkflowSpec{adaptAnalysisTask<pidTOFTaskBeta>(cfgc)};
  if (cfgc.options().get<int>("add-qa")) {
    workflow.push_back(adaptAnalysisTask<pidTOFTaskQABeta>(cfgc, TaskName{"pidTOFQA-task"}));
  }
  return workflow;
}
