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
#include "AnalysisDataModel/TrackSelectionTables.h"
#include "AnalysisCore/MC.h"
#include "AnalysisDataModel/TrackSelectionTables.h"
#include "AnalysisDataModel/TrackSelectionTables.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;

struct pidHMPIDQA {
  HistogramRegistry histos{"Histos", {}, OutputObjHandlingPolicy::QAObject};
  Configurable<int> nBinsP{"nBinsP", 500, "Number of momentum bins"};
  Configurable<float> minP{"minP", 0.01, "Minimum momentum plotted (GeV/c)"};
  Configurable<float> maxP{"maxP", 10, "Maximum momentum plotted (GeV/c)"};
  Configurable<float> maxDCA{"maxDCA", 3, "Maximum DCA xy use for the plot (cm)"};

  template <typename T>
  void makelogaxis(T h)
  {
    // return;
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
    histos.add("qa/signalvsP", ";#it{p} (GeV/#it{c});Cherenkov angle (rad)", kTH2F, {momAxis, {1000, 0, 1}});
    histos.add("distance/selected", ";HMPID distance", kTH1F, {{100, 0, 20}});
    histos.add("distance/nonselected", ";HMPID distance", kTH1F, {{100, 0, 20}});
    histos.add("qmip/selected", ";HMPID mip charge (ADC)", kTH1F, {{100, 0, 4000}});
    histos.add("qmip/nonselected", ";HMPID mip charge (ADC)", kTH1F, {{100, 0, 4000}});
    histos.add("nphotons/selected", ";HMPID number of detected photons", kTH1F, {{100, 0, 1000}});
    histos.add("nphotons/nonselected", ";HMPID number of detected photons", kTH1F, {{100, 0, 1000}});
  }

  using TrackCandidates = soa::Join<aod::Tracks, aod::TracksExtra, aod::TracksExtended, aod::TrackSelection>;
  void process(const TrackCandidates& tracks,
               const aod::HMPIDs& hmpids,
               const aod::Collisions& colls)
  {
    for (const auto& t : hmpids) {

      if (t.track_as<TrackCandidates>().isGlobalTrack() != (uint8_t) true) {
        continue;
      }
      if (abs(t.track_as<TrackCandidates>().dcaXY()) > maxDCA) {
        continue;
      }
      histos.fill(HIST("distance/nonselected"), t.hmpidDistance());
      histos.fill(HIST("qmip/nonselected"), t.hmpidQMip());
      histos.fill(HIST("nphotons/nonselected"), t.hmpidNPhotons());
      if (t.hmpidDistance() > 5.f) {
        continue;
      }
      if (t.hmpidQMip() < 120.f) {
        continue;
      }
      histos.fill(HIST("distance/selected"), t.hmpidDistance());
      histos.fill(HIST("qmip/selected"), t.hmpidQMip());
      histos.fill(HIST("nphotons/selected"), t.hmpidNPhotons());
      histos.fill(HIST("qa/signalvsP"), t.track_as<TrackCandidates>().p(), t.hmpidSignal());
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfg)
{
  WorkflowSpec workflow{adaptAnalysisTask<pidHMPIDQA>(cfg, TaskName{"pidHMPID-qa"})};
  return workflow;
}
