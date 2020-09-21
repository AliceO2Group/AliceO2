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
#include "Framework/AnalysisDataModel.h"
#include "ReconstructionDataFormats/Track.h"
#include "PID/PIDResponse.h"
#include "Framework/ASoAHelpers.h"
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

#define O2_DEFINE_CONFIGURABLE(NAME, TYPE, DEFAULT, HELP) Configurable<TYPE> NAME{#NAME, DEFAULT, HELP};

#define DOTH1F(OBJ, ...) \
  OutputObj<TH1F> OBJ{TH1F(#OBJ, __VA_ARGS__)};
#define DOTH2F(OBJ, ...) \
  OutputObj<TH2F> OBJ{TH2F(#OBJ, __VA_ARGS__)};

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
  O2_DEFINE_CONFIGURABLE(cfgCutVertex, float, 10.0f, "Accepted z-vertex range")
  O2_DEFINE_CONFIGURABLE(cfgCutEta, float, 0.8f, "Eta range for tracks")

#define BIN_AXIS 1000, 0.001, 20, 1000, 0, 1000

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
  Filter collisionFilter = nabs(aod::collision::posZ) < cfgCutVertex;
  Filter trackFilter = (nabs(aod::track::eta) < cfgCutEta) && ((aod::track::isGlobalTrack == (uint8_t)1) || (aod::track::isGlobalTrackSDD == (uint8_t)1));
  void process(aod::Collision const& collision, soa::Filtered<soa::Join<aod::Tracks, aod::TracksExtra, aod::pidRespTPC, aod::TrackSelection>> const& tracks)
  {
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
  O2_DEFINE_CONFIGURABLE(cfgCutVertex, float, 10.0f, "Accepted z-vertex range")
  O2_DEFINE_CONFIGURABLE(cfgCutEta, float, 0.8f, "Eta range for tracks")

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
  Filter collisionFilter = nabs(aod::collision::posZ) < cfgCutVertex;
  Filter trackFilter = (nabs(aod::track::eta) < cfgCutEta) && ((aod::track::isGlobalTrack == (uint8_t)1) || (aod::track::isGlobalTrackSDD == (uint8_t)1));

  void process(aod::Collision const& collision, soa::Filtered<soa::Join<aod::Tracks, aod::TracksExtra, aod::pidRespTPC, aod::TrackSelection>> const& tracks)
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
  // Options
  O2_DEFINE_CONFIGURABLE(cfgCutVertex, float, 10.0f, "Accepted z-vertex range")
  O2_DEFINE_CONFIGURABLE(cfgCutEta, float, 0.8f, "Eta range for tracks")

#define BIN_AXIS 1000, 0.001, 20, 1000, 0, 1000

  DOTH2F(htpcsignalEl, ";#it{p} (GeV/#it{c});TPC Signal;Tracks", BIN_AXIS);
  DOTH2F(htpcsignalMu, ";#it{p} (GeV/#it{c});TPC Signal;Tracks", BIN_AXIS);
  DOTH2F(htpcsignalPi, ";#it{p} (GeV/#it{c});TPC Signal;Tracks", BIN_AXIS);
  DOTH2F(htpcsignalKa, ";#it{p} (GeV/#it{c});TPC Signal;Tracks", BIN_AXIS);
  DOTH2F(htpcsignalPr, ";#it{p} (GeV/#it{c});TPC Signal;Tracks", BIN_AXIS);
  DOTH2F(htpcsignalDe, ";#it{p} (GeV/#it{c});TPC Signal;Tracks", BIN_AXIS);
  DOTH2F(htpcsignalTr, ";#it{p} (GeV/#it{c});TPC Signal;Tracks", BIN_AXIS);
  DOTH2F(htpcsignalHe, ";#it{p} (GeV/#it{c});TPC Signal;Tracks", BIN_AXIS);
  DOTH2F(htpcsignalAl, ";#it{p} (GeV/#it{c});TPC Signal;Tracks", BIN_AXIS);

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
  Filter collisionFilter = nabs(aod::collision::posZ) < cfgCutVertex;
  Filter trackFilter = (nabs(aod::track::eta) < cfgCutEta) && ((aod::track::isGlobalTrack == (uint8_t)1) || (aod::track::isGlobalTrackSDD == (uint8_t)1));

  void process(aod::Collision const& collision, soa::Filtered<soa::Join<aod::Tracks, aod::TracksExtra, aod::pidRespTPC, aod::pidRespTOF, aod::TrackSelection>> const& tracks)
  {
    for (auto const& i : tracks) {
      // Require kTIME and kTOFout
      if (!(i.flags() & 0x2000))
        continue;
      if (!(i.flags() & 0x80000000))
        continue;
      //

      // const float mom = i.p();
      const float mom = i.tpcInnerParam();
      if (abs(i.tofNSigmaEl()) < 2) {
        htpcsignalEl->Fill(mom, i.tpcSignal());
      }
      if (abs(i.tofNSigmaMu()) < 2) {
        htpcsignalMu->Fill(mom, i.tpcSignal());
      }
      if (abs(i.tofNSigmaPi()) < 2) {
        htpcsignalPi->Fill(mom, i.tpcSignal());
      }
      if (abs(i.tofNSigmaKa()) < 2) {
        htpcsignalKa->Fill(mom, i.tpcSignal());
      }
      if (abs(i.tofNSigmaPr()) < 2) {
        htpcsignalPr->Fill(mom, i.tpcSignal());
      }
      if (abs(i.tofNSigmaDe()) < 2) {
        htpcsignalDe->Fill(mom, i.tpcSignal());
      }
      if (abs(i.tofNSigmaTr()) < 2) {
        htpcsignalTr->Fill(mom, i.tpcSignal());
      }
      if (abs(i.tofNSigmaHe()) < 2) {
        htpcsignalHe->Fill(mom, i.tpcSignal());
      }
      if (abs(i.tofNSigmaAl()) < 2) {
        htpcsignalAl->Fill(mom, i.tpcSignal());
      }
    }
  }
};

struct TPCSpectraTask {
  // Pt
#define TIT ";#it{p}_{T} (GeV/#it{c});Tracks"
  DOTH1F(hpt_El, TIT, 100, 0, 20);
  DOTH1F(hpt_Pi, TIT, 100, 0, 20);
  DOTH1F(hpt_Ka, TIT, 100, 0, 20);
  DOTH1F(hpt_Pr, TIT, 100, 0, 20);
#undef TIT
  // P
#define TIT ";#it{p} (GeV/#it{c});Tracks"
  DOTH1F(hp_El, TIT, 100, 0, 20);
  DOTH1F(hp_Pi, TIT, 100, 0, 20);
  DOTH1F(hp_Ka, TIT, 100, 0, 20);
  DOTH1F(hp_Pr, TIT, 100, 0, 20);
#undef TIT

  void process(soa::Join<aod::Tracks, aod::TracksExtra, aod::pidRespTPC> const& tracks)
  {
    for (auto i : tracks) {
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
