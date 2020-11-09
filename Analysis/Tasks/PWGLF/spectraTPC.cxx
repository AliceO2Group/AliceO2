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
#include "Framework/AnalysisTask.h"
#include "Framework/AnalysisDataModel.h"
#include "Framework/ASoAHelpers.h"
#include "PID/PIDResponse.h"
#include "Analysis/TrackSelectionTables.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;

void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  std::vector<ConfigParamSpec> options{
    {"add-tof-histos", VariantType::Int, 0, {"Generate TPC with TOF histograms"}}};
  std::swap(workflowOptions, options);
}

#include "Framework/runDataProcessing.h"

#define CANDIDATE_SELECTION                                                           \
  Configurable<float> cfgCutVertex{"cfgCutVertex", 10.0f, "Accepted z-vertex range"}; \
  Configurable<float> cfgCutEta{"cfgCutEta", 0.8f, "Eta range for tracks"};           \
  Filter collisionFilter = nabs(aod::collision::posZ) < cfgCutVertex;                 \
  Filter trackFilter = (nabs(aod::track::eta) < cfgCutEta) && (aod::track::isGlobalTrack == true);

#define makelogaxis(h)                                            \
  {                                                               \
    const Int_t nbins = h->GetNbinsX();                           \
    double binp[nbins + 1];                                       \
    double max = h->GetXaxis()->GetBinUpEdge(nbins);              \
    double min = h->GetXaxis()->GetBinLowEdge(1);                 \
    double lmin = TMath::Log10(min);                              \
    double ldelta = (TMath::Log10(max) - lmin) / ((double)nbins); \
    for (int i = 0; i < nbins; i++) {                             \
      binp[i] = TMath::Exp(TMath::Log(10) * (lmin + i * ldelta)); \
    }                                                             \
    binp[nbins] = max + 1;                                        \
    h->GetXaxis()->Set(nbins, binp);                              \
  }

constexpr int Np = 9;
struct TPCSpectraTask {
  static constexpr const char* pT[Np] = {"e", "#mu", "#pi", "K", "p", "d", "t", "^{3}He", "#alpha"};
  static constexpr const char* hp[Np] = {"p/El", "p/Mu", "p/Pi", "p/Ka", "p/Pr", "p/De", "p/Tr", "p/He", "p/Al"};
  static constexpr const char* hpt[Np] = {"pt/El", "pt/Mu", "pt/Pi", "pt/Ka", "pt/Pr", "pt/De", "pt/Tr", "pt/He", "pt/Al"};
  HistogramRegistry histos{"Histos", {}, OutputObjHandlingPolicy::AnalysisObject};

  void init(o2::framework::InitContext&)
  {
    histos.add("p/Unselected", "Unselected;#it{p} (GeV/#it{c})", kTH1F, {{100, 0, 20}});
    histos.add("pt/Unselected", "Unselected;#it{p}_{T} (GeV/#it{c})", kTH1F, {{100, 0, 20}});
    for (int i = 0; i < Np; i++) {
      histos.add(hp[i], Form("%s;#it{p} (GeV/#it{c})", pT[i]), kTH1F, {{100, 0, 20}});
      histos.add(hpt[i], Form("%s;#it{p}_{T} (GeV/#it{c})", pT[i]), kTH1F, {{100, 0, 20}});
    }
  }

  //Defining filters and input
  CANDIDATE_SELECTION

  Configurable<float> nsigmacut{"nsigmacut", 3, "Value of the Nsigma cut"};

  using TrackCandidates = soa::Filtered<soa::Join<aod::Tracks, aod::TracksExtra, aod::pidRespTPC, aod::TrackSelection>>;
  void process(TrackCandidates::iterator const& track)
  {
    const float nsigma[Np] = {track.tpcNSigmaEl(), track.tpcNSigmaMu(), track.tpcNSigmaPi(),
                              track.tpcNSigmaKa(), track.tpcNSigmaPr(), track.tpcNSigmaDe(),
                              track.tpcNSigmaTr(), track.tpcNSigmaHe(), track.tpcNSigmaAl()};
    histos.fill("p/Unselected", track.p());
    histos.fill("pt/Unselected", track.pt());
    for (int i = 0; i < Np; i++) {
      if (abs(nsigma[i]) > nsigmacut.value) {
        continue;
      }
      histos.fill(hp[i], track.p());
      histos.fill(hpt[i], track.pt());
    }
  }
};

struct TPCPIDQASignalwTOFTask {
  static constexpr const char* pT[Np] = {"e", "#mu", "#pi", "K", "p", "d", "t", "^{3}He", "#alpha"};
  static constexpr const char* htpcsignal[Np] = {"tpcsignal/El", "tpcsignal/Mu", "tpcsignal/Pi",
                                                 "tpcsignal/Ka", "tpcsignal/Pr", "tpcsignal/De",
                                                 "tpcsignal/Tr", "tpcsignal/He", "tpcsignal/Al"};
  HistogramRegistry histos{"Histos", {}, OutputObjHandlingPolicy::AnalysisObject};

  void init(o2::framework::InitContext&)
  {
    for (int i = 0; i < Np; i++) {
      histos.add(htpcsignal[i], Form(";#it{p} (GeV/#it{c});TPC Signal;N_{#sigma}^{TPC}(%s)", pT[i]), kTH3D, {{1000, 0.001, 20}, {1000, 0, 1000}, {20, -10, 10}});
      makelogaxis(histos.get<TH3>(htpcsignal[i]));
    }
  }

  // Filters
  CANDIDATE_SELECTION

  Filter trackFilterTOF = (aod::track::tofSignal > 0.f); // Skip tracks without TOF
  using TrackCandidates = soa::Filtered<soa::Join<aod::Tracks, aod::TracksExtra, aod::pidRespTPC, aod::pidRespTOF, aod::TrackSelection>>;
  void process(TrackCandidates::iterator const& track)
  {
    // const float mom = track.p();
    const float mom = track.tpcInnerParam();
    histos.fill(htpcsignal[0], mom, track.tpcSignal(), track.tofNSigmaEl());
    histos.fill(htpcsignal[1], mom, track.tpcSignal(), track.tofNSigmaMu());
    histos.fill(htpcsignal[2], mom, track.tpcSignal(), track.tofNSigmaPi());
    histos.fill(htpcsignal[3], mom, track.tpcSignal(), track.tofNSigmaKa());
    histos.fill(htpcsignal[4], mom, track.tpcSignal(), track.tofNSigmaPr());
    histos.fill(htpcsignal[5], mom, track.tpcSignal(), track.tofNSigmaDe());
    histos.fill(htpcsignal[6], mom, track.tpcSignal(), track.tofNSigmaTr());
    histos.fill(htpcsignal[7], mom, track.tpcSignal(), track.tofNSigmaHe());
    histos.fill(htpcsignal[8], mom, track.tpcSignal(), track.tofNSigmaAl());
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  int TPCwTOF = cfgc.options().get<int>("add-tof-histos");
  WorkflowSpec workflow{adaptAnalysisTask<TPCSpectraTask>("tpcspectra-task")};
  if (TPCwTOF) {
    workflow.push_back(adaptAnalysisTask<TPCPIDQASignalwTOFTask>("TPCpidqa-signalwTOF-task"));
  }
  return workflow;
}
