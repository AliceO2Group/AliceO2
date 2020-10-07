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
#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "Framework/AnalysisDataModel.h"
#include "ReconstructionDataFormats/Track.h"
#include "PID/PIDResponse.h"
#include "Framework/ASoAHelpers.h"

// ROOT includes
#include <TH1F.h>

#define DOTH1F(OBJ, ...) \
  OutputObj<TH1F> OBJ{TH1F(#OBJ, __VA_ARGS__)};
#define DOTH2F(OBJ, ...) \
  OutputObj<TH2F> OBJ{TH2F(#OBJ, __VA_ARGS__)};

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;

#define TRACKSELECTION                                                                                                \
  UChar_t clustermap = i.itsClusterMap();                                                                             \
  bool issel = (i.tpcNClsFindable() > 70) && (i.flags() & 0x4) && (TESTBIT(clustermap, 0) || TESTBIT(clustermap, 1)); \
  issel = issel && (i.flags() & 0x2000);                                                                              \
  issel = issel && (i.flags() & 0x80000000);                                                                          \
  if (!issel)                                                                                                         \
    continue;

// #define TRACKSELECTION 1;

struct TOFQATask {
  // Event quantities
  DOTH1F(hvtxz, ";Vertex Z position;Events", 300, -15, 15);
  DOTH1F(hevtime, ";Event time (ns);Tracks", 100, -2, 2);
  // Track quantities
  DOTH1F(heta, ";#eta;Tracks", 100, -1, 1);
  DOTH1F(hp, ";#it{p} (GeV/#it{c});Tracks", 100, 0, 20);
  DOTH1F(hpt, ";#it{p}_{T} (GeV/#it{c});Tracks", 100, 0, 20);
  DOTH1F(hlength, ";Track Length (cm);Tracks", 100, 0, 1000);
  DOTH1F(htime, ";TOF Time (ns);Tracks", 1000, 0, 600);
  // Beta
  DOTH2F(hp_beta, ";#it{p} (GeV/#it{c});TOF #beta;Tracks", 100, 0, 20, 100, 0, 2);

  void process(aod::Collision const& collision, soa::Join<aod::Tracks, aod::TracksExtra, aod::pidRespTOF, aod::pidRespTOFbeta> const& tracks)
  {
    hvtxz->Fill(collision.posZ());
    for (auto i : tracks) {
      // Track selection
      TRACKSELECTION;
      //
      hevtime->Fill(collision.collisionTime() / 1000);
      // hevtime->Fill(collision.collisionTime0() / 1000);
      const float psq = sqrt(i.px() * i.px() + i.py() * i.py() + i.pz() * i.pz());
      heta->Fill(i.eta());
      hp->Fill(i.p());
      hpt->Fill(i.pt());
      //
      hlength->Fill(i.length());
      htime->Fill(i.tofSignal() / 1000);
      // Beta
      hp_beta->Fill(i.p(), i.beta());
    }
  }
};

struct TOFExpTimeQATask {
  // T-Texp
#define TIT(part) Form(";#it{p} (GeV/#it{c});(t-t_{evt}-t_{exp %s});Tracks", part)
  DOTH2F(htimediffEl, TIT("e"), 100, 0, 5, 100, -1000, 1000);
  DOTH2F(htimediffMu, TIT("#mu"), 100, 0, 5, 100, -1000, 1000);
  DOTH2F(htimediffPi, TIT("#pi"), 100, 0, 5, 100, -1000, 1000);
  DOTH2F(htimediffKa, TIT("K"), 100, 0, 5, 100, -1000, 1000);
  DOTH2F(htimediffPr, TIT("p"), 100, 0, 5, 100, -1000, 1000);
  DOTH2F(htimediffDe, TIT("d"), 100, 0, 5, 100, -1000, 1000);
  DOTH2F(htimediffTr, TIT("t"), 100, 0, 5, 100, -1000, 1000);
  DOTH2F(htimediffHe, TIT("^{3}He"), 100, 0, 5, 100, -1000, 1000);
  DOTH2F(htimediffAl, TIT("#alpha"), 100, 0, 5, 100, -1000, 1000);
#undef TIT

  void process(aod::Collision const& collision, soa::Join<aod::Tracks, aod::TracksExtra, aod::pidRespTOF, aod::pidRespTOFbeta> const& tracks)
  {
    for (auto i : tracks) {
      // Track selection
      TRACKSELECTION;
      //
      const float tof = i.tofSignal() - collision.collisionTime();
      htimediffEl->Fill(i.p(), tof - i.tofExpSignalEl());
      htimediffMu->Fill(i.p(), tof - i.tofExpSignalMu());
      htimediffPi->Fill(i.p(), tof - i.tofExpSignalPi());
      htimediffKa->Fill(i.p(), tof - i.tofExpSignalKa());
      htimediffPr->Fill(i.p(), tof - i.tofExpSignalPr());
      htimediffDe->Fill(i.p(), tof - i.tofExpSignalDe());
      htimediffTr->Fill(i.p(), tof - i.tofExpSignalTr());
      htimediffHe->Fill(i.p(), tof - i.tofExpSignalHe());
      htimediffAl->Fill(i.p(), tof - i.tofExpSignalAl());
    }
  }
};

struct TOFNSigmaQATask {
  // NSigma
#define TIT(part) Form(";#it{p} (GeV/#it{c});(t-t_{evt}-t_{exp %s})/N_{sigma %s};Tracks", part, part)
  DOTH2F(hnsigmaEl, TIT("e"), 100, 0, 5, 100, -10, 10);
  DOTH2F(hnsigmaMu, TIT("#mu"), 100, 0, 5, 100, -10, 10);
  DOTH2F(hnsigmaPi, TIT("#pi"), 100, 0, 5, 100, -10, 10);
  DOTH2F(hnsigmaKa, TIT("K"), 100, 0, 5, 100, -10, 10);
  DOTH2F(hnsigmaPr, TIT("p"), 100, 0, 5, 100, -10, 10);
  DOTH2F(hnsigmaDe, TIT("d"), 100, 0, 5, 100, -10, 10);
  DOTH2F(hnsigmaTr, TIT("t"), 100, 0, 5, 100, -10, 10);
  DOTH2F(hnsigmaHe, TIT("^{3}He"), 100, 0, 5, 100, -10, 10);
  DOTH2F(hnsigmaAl, TIT("#alpha"), 100, 0, 5, 100, -10, 10);
#undef TIT

  void process(aod::Collision const& collision, soa::Join<aod::Tracks, aod::TracksExtra, aod::pidRespTOF, aod::pidRespTOFbeta> const& tracks)
  {
    for (auto i : tracks) {
      // Track selection
      TRACKSELECTION;
      //
      hnsigmaEl->Fill(i.p(), i.tofNSigmaEl());
      hnsigmaMu->Fill(i.p(), i.tofNSigmaMu());
      hnsigmaPi->Fill(i.p(), i.tofNSigmaPi());
      hnsigmaKa->Fill(i.p(), i.tofNSigmaKa());
      hnsigmaPr->Fill(i.p(), i.tofNSigmaPr());
      hnsigmaDe->Fill(i.p(), i.tofNSigmaDe());
      hnsigmaTr->Fill(i.p(), i.tofNSigmaTr());
      hnsigmaHe->Fill(i.p(), i.tofNSigmaHe());
      hnsigmaAl->Fill(i.p(), i.tofNSigmaAl());
    }
  }
};

struct TOFSpectraTask {
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
  //
  DOTH1F(hlength_El, ";Track Length (cm);Tracks", 100, 0, 1000);
  DOTH1F(htime_El, ";TOF Time (ns);Tracks", 1000, 0, 600);
  // DOTH1F(hevtime_El, ";Event time (ns);Tracks", 100, -2, 2);
  //
  DOTH2F(hp_beta_El, ";#it{p} (GeV/#it{c});#beta - #beta_{el};Tracks", 100, 0, 20, 100, -0.01, 0.01);
  DOTH2F(hp_betasigma_El, ";#it{p} (GeV/#it{c});(#beta - #beta_{el})/#sigma;Tracks", 100, 0, 20, 100, -5, 5);

  // Filter trk_filter = (aod::track::tpcNClsFindable > 70);

  void process(soa::Join<aod::Tracks, aod::TracksExtra, aod::pidRespTOF, aod::pidRespTOFbeta> const& tracks)
  {
    for (auto i : tracks) {
      // Track selection
      TRACKSELECTION;
      //
      if (TMath::Abs(i.tofNSigmaPi()) < 3) {
        hp_Pi->Fill(i.p());
        hpt_Pi->Fill(i.pt());
      } else if (TMath::Abs(i.tofNSigmaKa()) < 3) {
        hp_Ka->Fill(i.p());
        hpt_Ka->Fill(i.pt());
      } else if (TMath::Abs(i.tofNSigmaPr()) < 3) {
        hp_Pr->Fill(i.p());
        hpt_Pr->Fill(i.pt());
      }
      if (TMath::Abs(i.separationbetael() < 1.f)) {
        hp_El->Fill(i.p());
        hpt_El->Fill(i.pt());
        //
        hlength_El->Fill(i.length());
        htime_El->Fill(i.tofSignal() / 1000);
        //
        hp_beta_El->Fill(i.p(), i.diffbetael());
        hp_betasigma_El->Fill(i.p(), i.separationbetael());
      }
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<TOFQATask>("tofqa-task"),
    adaptAnalysisTask<TOFExpTimeQATask>("tofexptime-task"),
    adaptAnalysisTask<TOFNSigmaQATask>("tofnsigma-task"),
    adaptAnalysisTask<TOFSpectraTask>("tofspectra-task")};
}

#undef TRACKSELECTION
