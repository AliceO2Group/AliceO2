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

// ROOT includes
#include <TH1F.h>

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

#define DOTH1F(OBJ, ...) \
  OutputObj<TH1F> OBJ{TH1F(#OBJ, __VA_ARGS__)};
#define DOTH2F(OBJ, ...) \
  OutputObj<TH2F> OBJ{TH2F(#OBJ, __VA_ARGS__)};
#define DOTH3F(OBJ, ...) \
  OutputObj<TH3F> OBJ{TH3F(#OBJ, __VA_ARGS__)};

#define CANDIDATE_SELECTION                                                                                                                    \
  Configurable<float> cfgCutVertex{"cfgCutVertex", 10.0f, "Accepted z-vertex range"};                                                          \
  Configurable<float> cfgCutEta{"cfgCutEta", 0.8f, "Eta range for tracks"};                                                                    \
  Filter collisionFilter = nabs(aod::collision::posZ) < cfgCutVertex;                                                                          \
  Filter trackFilter = (nabs(aod::track::eta) < cfgCutEta) && ((aod::track::isGlobalTrack == true) || (aod::track::isGlobalTrackSDD == true)); \
  using TrackCandidates = soa::Filtered<soa::Join<aod::Tracks, aod::TracksExtra, aod::pidRespTPC, aod::TrackSelection>>;                       \
  using TrackCandidateswTOF = soa::Filtered<soa::Join<aod::Tracks, aod::TracksExtra, aod::pidRespTPC, aod::pidRespTOF, aod::TrackSelection>>;  \
  using CollisionCandidates = soa::Filtered<aod::Collisions>;

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

struct TPCPIDQAExpSignalTask {
  // Options
#define BIN_AXIS 1000, 0.001, 20, 1000, 0, 1000

  DOTH1F(hvertexz, ";Vtx_{z} (cm);Entries", 100, -20, 20);
  DOTH2F(htpcsignal, ";#it{p} (GeV/#it{c});TPC Signal;Tracks", BIN_AXIS);
  DOTH2F(hexpEl, ";#it{p} (GeV/#it{c});TPC expected signal e;Tracks", BIN_AXIS);
  DOTH2F(hexpMu, ";#it{p} (GeV/#it{c});TPC expected signal #mu;Tracks", BIN_AXIS);
  DOTH2F(hexpPi, ";#it{p} (GeV/#it{c});TPC expected signal #pi;Tracks", BIN_AXIS);
  DOTH2F(hexpKa, ";#it{p} (GeV/#it{c});TPC expected signal K;Tracks", BIN_AXIS);
  DOTH2F(hexpPr, ";#it{p} (GeV/#it{c});TPC expected signal p;Tracks", BIN_AXIS);
  DOTH2F(hexpDe, ";#it{p} (GeV/#it{c});TPC expected signal d;Tracks", BIN_AXIS);
  DOTH2F(hexpTr, ";#it{p} (GeV/#it{c});TPC expected signal t;Tracks", BIN_AXIS);
  DOTH2F(hexpHe, ";#it{p} (GeV/#it{c});TPC expected signal ^{3}He;Tracks", BIN_AXIS);
  DOTH2F(hexpAl, ";#it{p} (GeV/#it{c});TPC expected signal #alpha;Tracks", BIN_AXIS);

#undef BIN_AXIS

  void init(o2::framework::InitContext&)
  {
    // Log binning for p
    makelogaxis(htpcsignal);
    makelogaxis(hexpEl);
    makelogaxis(hexpMu);
    makelogaxis(hexpPi);
    makelogaxis(hexpKa);
    makelogaxis(hexpPr);
    makelogaxis(hexpDe);
    makelogaxis(hexpTr);
    makelogaxis(hexpHe);
    makelogaxis(hexpAl);
  }

  // Filters
  CANDIDATE_SELECTION

  void process(CollisionCandidates::iterator const& collision, TrackCandidates const& tracks)
  {
    hvertexz->Fill(collision.posZ());
    for (auto const& i : tracks) {
      // const float mom = i.p();
      const float mom = i.tpcInnerParam();
      htpcsignal->Fill(mom, i.tpcSignal());
      hexpEl->Fill(mom, i.tpcExpSignalEl());
      hexpMu->Fill(mom, i.tpcExpSignalMu());
      hexpPi->Fill(mom, i.tpcExpSignalPi());
      hexpKa->Fill(mom, i.tpcExpSignalKa());
      hexpPr->Fill(mom, i.tpcExpSignalPr());
      hexpDe->Fill(mom, i.tpcExpSignalDe());
      hexpTr->Fill(mom, i.tpcExpSignalTr());
      hexpHe->Fill(mom, i.tpcExpSignalHe());
      hexpAl->Fill(mom, i.tpcExpSignalAl());
    }
  }
};

struct TPCPIDQANSigmaTask {
  // Options
#define BIN_AXIS 1000, 0.001, 20, 1000, -10, 10

  // TPC NSigma
  DOTH2F(hnsigmaEl, ";#it{p} (GeV/#it{c});TPC N_{sigma e};Tracks", BIN_AXIS);
  DOTH2F(hnsigmaMu, ";#it{p} (GeV/#it{c});TPC N_{sigma #mu};Tracks", BIN_AXIS);
  DOTH2F(hnsigmaPi, ";#it{p} (GeV/#it{c});TPC N_{sigma #pi};Tracks", BIN_AXIS);
  DOTH2F(hnsigmaKa, ";#it{p} (GeV/#it{c});TPC N_{sigma K};Tracks", BIN_AXIS);
  DOTH2F(hnsigmaPr, ";#it{p} (GeV/#it{c});TPC N_{sigma p};Tracks", BIN_AXIS);
  DOTH2F(hnsigmaDe, ";#it{p} (GeV/#it{c});TPC N_{sigma d};Tracks", BIN_AXIS);
  DOTH2F(hnsigmaTr, ";#it{p} (GeV/#it{c});TPC N_{sigma t};Tracks", BIN_AXIS);
  DOTH2F(hnsigmaHe, ";#it{p} (GeV/#it{c});TPC N_{sigma ^{3}He};Tracks", BIN_AXIS);
  DOTH2F(hnsigmaAl, ";#it{p} (GeV/#it{c});TPC N_{sigma #alpha};Tracks", BIN_AXIS);

#undef BIN_AXIS

  void init(o2::framework::InitContext&)
  {
    // Log binning for p
    makelogaxis(hnsigmaEl);
    makelogaxis(hnsigmaMu);
    makelogaxis(hnsigmaPi);
    makelogaxis(hnsigmaKa);
    makelogaxis(hnsigmaPr);
    makelogaxis(hnsigmaDe);
    makelogaxis(hnsigmaTr);
    makelogaxis(hnsigmaHe);
    makelogaxis(hnsigmaAl);
  }

  // Filters
  CANDIDATE_SELECTION

  void process(CollisionCandidates::iterator const& collision, TrackCandidates const& tracks)
  {
    for (auto const& i : tracks) {
      hnsigmaEl->Fill(i.p(), i.tpcNSigmaEl());
      hnsigmaMu->Fill(i.p(), i.tpcNSigmaMu());
      hnsigmaPi->Fill(i.p(), i.tpcNSigmaPi());
      hnsigmaKa->Fill(i.p(), i.tpcNSigmaKa());
      hnsigmaPr->Fill(i.p(), i.tpcNSigmaPr());
      hnsigmaDe->Fill(i.p(), i.tpcNSigmaDe());
      hnsigmaTr->Fill(i.p(), i.tpcNSigmaTr());
      hnsigmaHe->Fill(i.p(), i.tpcNSigmaHe());
      hnsigmaAl->Fill(i.p(), i.tpcNSigmaAl());
    }
  }
};

struct TPCPIDQASignalwTOFTask {

#define BIN_AXIS 1000, 0.001, 20, 1000, 0, 1000, 20, -10, 10

  DOTH3F(htpcsignalEl, ";#it{p} (GeV/#it{c});TPC Signal;TOF N#sigma e", BIN_AXIS);
  DOTH3F(htpcsignalMu, ";#it{p} (GeV/#it{c});TPC Signal;TOF N#sigma #mu", BIN_AXIS);
  DOTH3F(htpcsignalPi, ";#it{p} (GeV/#it{c});TPC Signal;TOF N#sigma #pi", BIN_AXIS);
  DOTH3F(htpcsignalKa, ";#it{p} (GeV/#it{c});TPC Signal;TOF N#sigma K", BIN_AXIS);
  DOTH3F(htpcsignalPr, ";#it{p} (GeV/#it{c});TPC Signal;TOF N#sigma p", BIN_AXIS);
  DOTH3F(htpcsignalDe, ";#it{p} (GeV/#it{c});TPC Signal;TOF N#sigma d", BIN_AXIS);
  DOTH3F(htpcsignalTr, ";#it{p} (GeV/#it{c});TPC Signal;TOF N#sigma t", BIN_AXIS);
  DOTH3F(htpcsignalHe, ";#it{p} (GeV/#it{c});TPC Signal;TOF N#sigma ^{3}He", BIN_AXIS);
  DOTH3F(htpcsignalAl, ";#it{p} (GeV/#it{c});TPC Signal;TOF N#sigma #alpha", BIN_AXIS);

#undef BIN_AXIS

  void init(o2::framework::InitContext&)
  {
    // Log binning for p
    makelogaxis(htpcsignalEl);
    makelogaxis(htpcsignalMu);
    makelogaxis(htpcsignalPi);
    makelogaxis(htpcsignalKa);
    makelogaxis(htpcsignalPr);
    makelogaxis(htpcsignalDe);
    makelogaxis(htpcsignalTr);
    makelogaxis(htpcsignalHe);
    makelogaxis(htpcsignalAl);
  }

  // Filters
  CANDIDATE_SELECTION

  void process(CollisionCandidates::iterator const& collision, TrackCandidateswTOF const& tracks)
  {
    for (auto const& i : tracks) {
      // const float mom = i.p();
      const float mom = i.tpcInnerParam();
      if (i.tofSignal() < 0) { // Skip tracks without TOF
        continue;
      }
      htpcsignalEl->Fill(mom, i.tpcSignal(), i.tofNSigmaEl());
      htpcsignalMu->Fill(mom, i.tpcSignal(), i.tofNSigmaMu());
      htpcsignalPi->Fill(mom, i.tpcSignal(), i.tofNSigmaPi());
      htpcsignalKa->Fill(mom, i.tpcSignal(), i.tofNSigmaKa());
      htpcsignalPr->Fill(mom, i.tpcSignal(), i.tofNSigmaPr());
      htpcsignalDe->Fill(mom, i.tpcSignal(), i.tofNSigmaDe());
      htpcsignalTr->Fill(mom, i.tpcSignal(), i.tofNSigmaTr());
      htpcsignalHe->Fill(mom, i.tpcSignal(), i.tofNSigmaHe());
      htpcsignalAl->Fill(mom, i.tpcSignal(), i.tofNSigmaAl());
    }
  }
};

struct TPCSpectraTask {

  // Pt
#define TIT ";#it{p}_{T} (GeV/#it{c});Tracks"
  DOTH1F(hpt, TIT, 100, 0, 20);
  DOTH1F(hpt_El, TIT, 100, 0, 20);
  DOTH1F(hpt_Pi, TIT, 100, 0, 20);
  DOTH1F(hpt_Ka, TIT, 100, 0, 20);
  DOTH1F(hpt_Pr, TIT, 100, 0, 20);
#undef TIT
  // P
#define TIT ";#it{p} (GeV/#it{c});Tracks"
  DOTH1F(hp, TIT, 100, 0, 20);
  DOTH1F(hp_El, TIT, 100, 0, 20);
  DOTH1F(hp_Pi, TIT, 100, 0, 20);
  DOTH1F(hp_Ka, TIT, 100, 0, 20);
  DOTH1F(hp_Pr, TIT, 100, 0, 20);
#undef TIT

  //Defining filters and input
  CANDIDATE_SELECTION

  void process(CollisionCandidates::iterator const& collision, TrackCandidates const& tracks)
  {
    for (auto i : tracks) {
      hp->Fill(i.p());
      hpt->Fill(i.pt());
      if (TMath::Abs(i.tpcNSigmaEl()) < 3) {
        hp_El->Fill(i.p());
        hpt_El->Fill(i.pt());
      }
      if (TMath::Abs(i.tpcNSigmaPi()) < 3) {
        hp_Pi->Fill(i.p());
        hpt_Pi->Fill(i.pt());
      }
      if (TMath::Abs(i.tpcNSigmaKa()) < 3) {
        hp_Ka->Fill(i.p());
        hpt_Ka->Fill(i.pt());
      }
      if (TMath::Abs(i.tpcNSigmaPr()) < 3) {
        hp_Pr->Fill(i.p());
        hpt_Pr->Fill(i.pt());
      }
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  int TPCwTOF = cfgc.options().get<int>("add-tof-histos");
  WorkflowSpec workflow{adaptAnalysisTask<TPCPIDQAExpSignalTask>("TPCpidqa-expsignal-task"),
                        adaptAnalysisTask<TPCPIDQANSigmaTask>("TPCpidqa-nsigma-task"),
                        adaptAnalysisTask<TPCSpectraTask>("tpcspectra-task")};
  if (TPCwTOF)
    workflow.push_back(adaptAnalysisTask<TPCPIDQASignalwTOFTask>("TPCpidqa-signalwTOF-task"));
  return workflow;
}
