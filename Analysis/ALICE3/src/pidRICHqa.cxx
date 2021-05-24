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
#include "ALICE3Analysis/RICH.h"
#include "AnalysisDataModel/TrackSelectionTables.h"
#include "AnalysisCore/MC.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;

namespace o2::aod
{

namespace indices
{
DECLARE_SOA_INDEX_COLUMN(Track, track);
DECLARE_SOA_INDEX_COLUMN(RICH, rich);
} // namespace indices

DECLARE_SOA_INDEX_TABLE_USER(RICHTracksIndex, Tracks, "RICHTRK", indices::TrackId, indices::RICHId);
} // namespace o2::aod

struct RICHindexbuilder {
  Builds<o2::aod::RICHTracksIndex> ind;
  void init(o2::framework::InitContext&)
  {
  }
};

struct pidRICHQAMC {
  HistogramRegistry histos{"Histos", {}, OutputObjHandlingPolicy::QAObject};
  Configurable<int> pdgCode{"pdgCode", 0, "pdg code of the particles to accept"};
  Configurable<int> useOnlyPhysicsPrimary{"useOnlyPhysicsPrimary", 1,
                                          "Whether to use only physical primary particles."};
  Configurable<float> minLength{"minLength", 0, "Minimum length of accepted tracks (cm)"};
  Configurable<float> maxLength{"maxLength", 1000, "Maximum length of accepted tracks (cm)"};
  Configurable<float> minEta{"minEta", -1.4, "Minimum eta of accepted tracks"};
  Configurable<float> maxEta{"maxEta", 1.4, "Maximum eta of accepted tracks"};
  Configurable<int> nBinsP{"nBinsP", 500, "Number of momentum bins"};
  Configurable<float> minP{"minP", 0.01, "Minimum momentum plotted (GeV/c)"};
  Configurable<float> maxP{"maxP", 100, "Maximum momentum plotted (GeV/c)"};
  Configurable<int> nBinsNsigma{"nBinsNsigma", 600, "Number of Nsigma bins"};
  Configurable<float> minNsigma{"minNsigma", -100.f, "Minimum Nsigma plotted"};
  Configurable<float> maxNsigma{"maxNsigma", 100.f, "Maximum Nsigma plotted"};
  Configurable<int> nBinsDelta{"nBinsDelta", 100, "Number of delta bins"};
  Configurable<float> minDelta{"minDelta", -1.f, "Minimum delta plotted (rad)"};
  Configurable<float> maxDelta{"maxDelta", 1.f, "Maximum delta plotted (rad)"};
  Configurable<int> logAxis{"logAxis", 1, "Flag to use a log momentum axis"};

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
    AxisSpec momAxis{nBinsP, minP, maxP};
    AxisSpec nsigmaAxis{nBinsNsigma, minNsigma, maxNsigma};
    AxisSpec deltaAxis{nBinsDelta, minDelta, maxDelta};

    histos.add("event/vertexz", ";Vtx_{z} (cm);Entries", kTH1F, {{100, -20, 20}});
    histos.add("p/Unselected", "Unselected;#it{p} (GeV/#it{c})", kTH1F, {momAxis});
    histos.add("p/Prim", "Primaries;#it{p} (GeV/#it{c})", kTH1F, {momAxis});
    histos.add("p/Sec", "Secondaries;#it{p} (GeV/#it{c})", kTH1F, {momAxis});
    histos.add("pt/Unselected", "Unselected;#it{p} (GeV/#it{c})", kTH1F, {momAxis});
    histos.add("qa/signal", ";Cherenkov angle (rad)", kTH1F, {{100, 0, 1}});
    histos.add("qa/signalerror", ";Cherenkov angle (rad)", kTH1F, {{100, 0, 1}});
    histos.add("qa/signalvsP", ";#it{p} (GeV/#it{c});Cherenkov angle (rad)", kTH2F, {momAxis, {1000, 0, 0.3}});
    histos.add("qa/deltaEl", ";#it{p} (GeV/#it{c});#Delta(e) (rad)", kTH2F, {momAxis, deltaAxis});
    histos.add("qa/deltaMu", ";#it{p} (GeV/#it{c});#Delta(#mu) (rad)", kTH2F, {momAxis, deltaAxis});
    histos.add("qa/deltaPi", ";#it{p} (GeV/#it{c});#Delta(#pi) (rad)", kTH2F, {momAxis, deltaAxis});
    histos.add("qa/deltaKa", ";#it{p} (GeV/#it{c});#Delta(K) (rad)", kTH2F, {momAxis, deltaAxis});
    histos.add("qa/deltaPr", ";#it{p} (GeV/#it{c});#Delta(p) (rad)", kTH2F, {momAxis, deltaAxis});
    histos.add("qa/nsigmaEl", ";#it{p} (GeV/#it{c});N_{#sigma}^{RICH}(e)", kTH2F, {momAxis, nsigmaAxis});
    histos.add("qa/nsigmaMu", ";#it{p} (GeV/#it{c});N_{#sigma}^{RICH}(#mu)", kTH2F, {momAxis, nsigmaAxis});
    histos.add("qa/nsigmaPi", ";#it{p} (GeV/#it{c});N_{#sigma}^{RICH}(#pi)", kTH2F, {momAxis, nsigmaAxis});
    histos.add("qa/nsigmaKa", ";#it{p} (GeV/#it{c});N_{#sigma}^{RICH}(K)", kTH2F, {momAxis, nsigmaAxis});
    histos.add("qa/nsigmaPr", ";#it{p} (GeV/#it{c});N_{#sigma}^{RICH}(p)", kTH2F, {momAxis, nsigmaAxis});
    makelogaxis(histos.get<TH2>(HIST("qa/signalvsP")));
    makelogaxis(histos.get<TH2>(HIST("qa/deltaEl")));
    makelogaxis(histos.get<TH2>(HIST("qa/deltaMu")));
    makelogaxis(histos.get<TH2>(HIST("qa/deltaPi")));
    makelogaxis(histos.get<TH2>(HIST("qa/deltaKa")));
    makelogaxis(histos.get<TH2>(HIST("qa/deltaPr")));
    makelogaxis(histos.get<TH2>(HIST("qa/nsigmaEl")));
    makelogaxis(histos.get<TH2>(HIST("qa/nsigmaMu")));
    makelogaxis(histos.get<TH2>(HIST("qa/nsigmaPi")));
    makelogaxis(histos.get<TH2>(HIST("qa/nsigmaKa")));
    makelogaxis(histos.get<TH2>(HIST("qa/nsigmaPr")));
  }

  using Trks = soa::Join<aod::Tracks, aod::RICHTracksIndex, aod::TracksExtra>;
  void process(const Trks& tracks,
               const aod::McTrackLabels& labels,
               const aod::RICHs&,
               const aod::McParticles& mcParticles,
               const aod::Collisions& colls)
  {
    for (const auto& col : colls) {
      histos.fill(HIST("event/vertexz"), col.posZ());
    }
    for (const auto& track : tracks) {
      if (!track.has_rich()) {
        continue;
      }
      if (track.length() < minLength) {
        continue;
      }
      if (track.length() > maxLength) {
        continue;
      }
      if (track.eta() > maxEta || track.eta() < minEta) {
        continue;
      }
      const auto mcParticle = labels.iteratorAt(track.globalIndex()).mcParticle();
      if (pdgCode != 0 && abs(mcParticle.pdgCode()) != pdgCode) {
        continue;
      }
      if (useOnlyPhysicsPrimary == 1 && !MC::isPhysicalPrimary(mcParticles, mcParticle)) { // Selecting primaries
        histos.fill(HIST("p/Sec"), track.p());
        continue;
      }
      histos.fill(HIST("p/Prim"), track.p());
      histos.fill(HIST("p/Unselected"), track.p());
      histos.fill(HIST("pt/Unselected"), track.pt());
      histos.fill(HIST("qa/signal"), track.rich().richSignal());
      histos.fill(HIST("qa/signalerror"), track.rich().richSignalError());
      histos.fill(HIST("qa/signalvsP"), track.p(), track.rich().richSignal());
      histos.fill(HIST("qa/deltaEl"), track.p(), track.rich().richDeltaEl());
      histos.fill(HIST("qa/deltaMu"), track.p(), track.rich().richDeltaMu());
      histos.fill(HIST("qa/deltaPi"), track.p(), track.rich().richDeltaPi());
      histos.fill(HIST("qa/deltaKa"), track.p(), track.rich().richDeltaKa());
      histos.fill(HIST("qa/deltaPr"), track.p(), track.rich().richDeltaPr());
      histos.fill(HIST("qa/nsigmaEl"), track.p(), track.rich().richNsigmaEl());
      histos.fill(HIST("qa/nsigmaMu"), track.p(), track.rich().richNsigmaMu());
      histos.fill(HIST("qa/nsigmaPi"), track.p(), track.rich().richNsigmaPi());
      histos.fill(HIST("qa/nsigmaKa"), track.p(), track.rich().richNsigmaKa());
      histos.fill(HIST("qa/nsigmaPr"), track.p(), track.rich().richNsigmaPr());
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfg)
{
  return WorkflowSpec{adaptAnalysisTask<RICHindexbuilder>(cfg), adaptAnalysisTask<pidRICHQAMC>(cfg, TaskName{"pidRICH-qa"})};
}
