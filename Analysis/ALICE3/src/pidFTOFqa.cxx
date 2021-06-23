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
#include "ALICE3Analysis/FTOF.h"
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
DECLARE_SOA_INDEX_COLUMN(FTOF, ftof);
} // namespace indices

DECLARE_SOA_INDEX_TABLE_USER(FTOFTracksIndex, Tracks, "FTOFTRK", indices::TrackId, indices::FTOFId);
} // namespace o2::aod

struct ftofIndexBuilder {
  Builds<o2::aod::FTOFTracksIndex> ind;
  void init(o2::framework::InitContext&)
  {
  }
};

struct ftofPidQaMC {
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
  Configurable<float> minDelta{"minDelta", -1.f, "Minimum delta plotted"};
  Configurable<float> maxDelta{"maxDelta", 1.f, "Maximum delta plotted"};
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
    histos.add("event/length", ";FTOF length (cm)", kTH1F, {{100, 0, 100}});
    histos.add("event/eta", ";#it{#eta}", kTH1F, {{100, -10, 10}});
    histos.add("p/Unselected", "Unselected;#it{p} (GeV/#it{c})", kTH1F, {momAxis});
    histos.add("p/Prim", "Primaries;#it{p} (GeV/#it{c})", kTH1F, {momAxis});
    histos.add("p/Sec", "Secondaries;#it{p} (GeV/#it{c})", kTH1F, {momAxis});
    histos.add("pt/Unselected", "Unselected;#it{p} (GeV/#it{c})", kTH1F, {momAxis});
    histos.add("qa/length", ";FTOF length (cm)", kTH1F, {{100, 0, 100}});
    histos.add("qa/signal", ";FTOF signal (ps)", kTH1F, {{100, 0, 1000}});
    histos.add("qa/signalvsP", ";#it{p} (GeV/#it{c});FTOF signal (ps)", kTH2F, {momAxis, {1000, 0, 0.3}});
    histos.add("qa/deltaEl", ";#it{p} (GeV/#it{c});#Delta(e) (ps)", kTH2F, {momAxis, deltaAxis});
    histos.add("qa/deltaMu", ";#it{p} (GeV/#it{c});#Delta(#mu) (ps)", kTH2F, {momAxis, deltaAxis});
    histos.add("qa/deltaPi", ";#it{p} (GeV/#it{c});#Delta(#pi) (ps)", kTH2F, {momAxis, deltaAxis});
    histos.add("qa/deltaKa", ";#it{p} (GeV/#it{c});#Delta(K) (ps)", kTH2F, {momAxis, deltaAxis});
    histos.add("qa/deltaPr", ";#it{p} (GeV/#it{c});#Delta(p) (ps)", kTH2F, {momAxis, deltaAxis});
    histos.add("qa/nsigmaEl", ";#it{p} (GeV/#it{c});N_{#sigma}^{FTOF}(e)", kTH2F, {momAxis, nsigmaAxis});
    histos.add("qa/nsigmaMu", ";#it{p} (GeV/#it{c});N_{#sigma}^{FTOF}(#mu)", kTH2F, {momAxis, nsigmaAxis});
    histos.add("qa/nsigmaPi", ";#it{p} (GeV/#it{c});N_{#sigma}^{FTOF}(#pi)", kTH2F, {momAxis, nsigmaAxis});
    histos.add("qa/nsigmaKa", ";#it{p} (GeV/#it{c});N_{#sigma}^{FTOF}(K)", kTH2F, {momAxis, nsigmaAxis});
    histos.add("qa/nsigmaPr", ";#it{p} (GeV/#it{c});N_{#sigma}^{FTOF}(p)", kTH2F, {momAxis, nsigmaAxis});
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

  using Trks = soa::Join<aod::Tracks, aod::FTOFTracksIndex, aod::TracksExtra>;
  void process(const Trks& tracks,
               const aod::McTrackLabels& labels,
               const aod::FTOFs&,
               const aod::McParticles& mcParticles,
               const aod::Collisions& colls)
  {
    for (const auto& col : colls) {
      histos.fill(HIST("event/vertexz"), col.posZ());
    }
    for (const auto& track : tracks) {
      histos.fill(HIST("event/length"), track.ftof().ftofLength());
      histos.fill(HIST("event/eta"), track.eta());
      if (!track.has_ftof()) {
        continue;
      }
      if (track.ftof().ftofLength() < minLength) {
        continue;
      }
      if (track.ftof().ftofLength() > maxLength) {
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
      histos.fill(HIST("qa/length"), track.ftof().ftofLength());
      histos.fill(HIST("qa/signal"), track.ftof().ftofSignal());
      histos.fill(HIST("qa/signalvsP"), track.p(), track.ftof().ftofSignal());
      histos.fill(HIST("qa/deltaEl"), track.p(), track.ftof().ftofDeltaEl());
      histos.fill(HIST("qa/deltaMu"), track.p(), track.ftof().ftofDeltaMu());
      histos.fill(HIST("qa/deltaPi"), track.p(), track.ftof().ftofDeltaPi());
      histos.fill(HIST("qa/deltaKa"), track.p(), track.ftof().ftofDeltaKa());
      histos.fill(HIST("qa/deltaPr"), track.p(), track.ftof().ftofDeltaPr());
      histos.fill(HIST("qa/nsigmaEl"), track.p(), track.ftof().ftofNsigmaEl());
      histos.fill(HIST("qa/nsigmaMu"), track.p(), track.ftof().ftofNsigmaMu());
      histos.fill(HIST("qa/nsigmaPi"), track.p(), track.ftof().ftofNsigmaPi());
      histos.fill(HIST("qa/nsigmaKa"), track.p(), track.ftof().ftofNsigmaKa());
      histos.fill(HIST("qa/nsigmaPr"), track.p(), track.ftof().ftofNsigmaPr());
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfg)
{
  return WorkflowSpec{adaptAnalysisTask<ftofIndexBuilder>(cfg), adaptAnalysisTask<ftofPidQaMC>(cfg)};
}
