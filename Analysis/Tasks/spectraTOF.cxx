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

struct TOFPIDQATask {
  // Track quantities
  DOTH1F(hp_NoCut, ";#it{p} (GeV/#it{c});Tracks", 100, 0, 20);
  DOTH1F(hp_TrkCut, ";#it{p} (GeV/#it{c});Tracks", 100, 0, 20);
  DOTH1F(hp_TOFCut, ";#it{p} (GeV/#it{c});Tracks", 100, 0, 20);
  // TOF Quantities
  DOTH1F(hlength_NoCut, ";Track Length (cm);Tracks", 100, 0, 1000);
  DOTH1F(htime_NoCut, ";TOF Time (ns);Tracks", 1000, 0, 600);
  DOTH1F(hevtime_NoCut, ";Event time (ns);Tracks", 100, -2, 2);
  DOTH2F(hp_pTOFexp_NoCut, ";#it{p} (GeV/#it{c});#it{p}_{Exp TOF} (GeV/#it{c});Tracks", 100, 0, 20, 100, 0, 20);
  // T-Texp
  DOTH2F(htimediffEl_NoCut, ";#it{p} (GeV/#it{c});(t-t_{evt}-t_{exp e});Tracks", 100, 0, 5, 100, -1000, 1000);
  DOTH2F(htimediffMu_NoCut, ";#it{p} (GeV/#it{c});(t-t_{evt}-t_{exp #mu});Tracks", 100, 0, 5, 100, -1000, 1000);
  DOTH2F(htimediffPi_NoCut, ";#it{p} (GeV/#it{c});(t-t_{evt}-t_{exp #pi});Tracks", 100, 0, 5, 100, -1000, 1000);
  DOTH2F(htimediffKa_NoCut, ";#it{p} (GeV/#it{c});(t-t_{evt}-t_{exp K});Tracks", 100, 0, 5, 100, -1000, 1000);
  DOTH2F(htimediffPr_NoCut, ";#it{p} (GeV/#it{c});(t-t_{evt}-t_{exp p});Tracks", 100, 0, 5, 100, -1000, 1000);
  // NSigma
  DOTH2F(hnsigmaEl_NoCut, ";#it{p} (GeV/#it{c});(t-t_{evt}-t_{exp e})/N_{sigma e};Tracks", 100, 0, 5, 100, -10, 10);
  DOTH2F(hnsigmaMu_NoCut, ";#it{p} (GeV/#it{c});(t-t_{evt}-t_{exp #mu})/N_{sigma #mu};Tracks", 100, 0, 5, 100, -10, 10);
  DOTH2F(hnsigmaPi_NoCut, ";#it{p} (GeV/#it{c});(t-t_{evt}-t_{exp #pi})/N_{sigma #pi};Tracks", 100, 0, 5, 100, -10, 10);
  DOTH2F(hnsigmaKa_NoCut, ";#it{p} (GeV/#it{c});(t-t_{evt}-t_{exp K})/N_{sigma K};Tracks", 100, 0, 5, 100, -10, 10);
  DOTH2F(hnsigmaPr_NoCut, ";#it{p} (GeV/#it{c});(t-t_{evt}-t_{exp p})/N_{sigma p};Tracks", 100, 0, 5, 100, -10, 10);
  // Beta
  DOTH2F(hp_beta, ";#it{p} (GeV/#it{c});TOF #beta;Tracks", 100, 0, 20, 100, 0, 2);

  void process(aod::Collision const& collision, soa::Join<aod::Tracks, aod::TracksExtra, aod::pidRespTOF, aod::pidRespTOFbeta> const& tracks)
  {
    for (auto i : tracks) {
      hp_NoCut->Fill(i.p());
      // Track selection
      UChar_t clustermap = i.itsClusterMap();
      bool issel = (i.tpcNClsFindable() > 70) && (i.flags() & 0x4) && (TESTBIT(clustermap, 0) || TESTBIT(clustermap, 1));
      if (issel)
        hp_TrkCut->Fill(i.p());
      issel = issel && (i.flags() & 0x2000);     //kTOFout
      issel = issel && (i.flags() & 0x80000000); //kTIME
      if (issel)
        hp_TOFCut->Fill(i.p());
      hp_pTOFexp_NoCut->Fill(i.p(), i.tofExpMom() / (TMath::C() * 1.0e2f * 1.0e-12f));
      //
      hlength_NoCut->Fill(i.length());
      htime_NoCut->Fill(i.tofSignal() / 1000);
      //
      hevtime_NoCut->Fill(collision.collisionTime() / 1000);
      // hevtime_NoCut->Fill(collision.collisionTime0() / 1000);
      //
      htimediffEl_NoCut->Fill(i.p(), i.tofSignal() - collision.collisionTime() - i.expTimeEl());
      htimediffMu_NoCut->Fill(i.p(), i.tofSignal() - collision.collisionTime() - i.expTimeMu());
      htimediffPi_NoCut->Fill(i.p(), i.tofSignal() - collision.collisionTime() - i.expTimePi());
      htimediffKa_NoCut->Fill(i.p(), i.tofSignal() - collision.collisionTime() - i.expTimeKa());
      htimediffPr_NoCut->Fill(i.p(), i.tofSignal() - collision.collisionTime() - i.expTimePr());
      //
      hnsigmaEl_NoCut->Fill(i.p(), i.nSigmaEl());
      hnsigmaMu_NoCut->Fill(i.p(), i.nSigmaMu());
      hnsigmaPi_NoCut->Fill(i.p(), i.nSigmaPi());
      hnsigmaKa_NoCut->Fill(i.p(), i.nSigmaKa());
      hnsigmaPr_NoCut->Fill(i.p(), i.nSigmaPr());
      // Beta
      hp_beta->Fill(i.p(), i.beta());
    }
  }
};

struct SpectraTask {
  // Pt
  DOTH1F(hpt_El, ";#it{p}_{T} (GeV/#it{c});Tracks", 100, 0, 20);
  DOTH1F(hpt_Pi, ";#it{p}_{T} (GeV/#it{c});Tracks", 100, 0, 20);
  DOTH1F(hpt_Ka, ";#it{p}_{T} (GeV/#it{c});Tracks", 100, 0, 20);
  DOTH1F(hpt_Pr, ";#it{p}_{T} (GeV/#it{c});Tracks", 100, 0, 20);
  // P
  DOTH1F(hp_El, ";#it{p} (GeV/#it{c});Tracks", 100, 0, 20);
  DOTH1F(hp_Pi, ";#it{p} (GeV/#it{c});Tracks", 100, 0, 20);
  DOTH1F(hp_Ka, ";#it{p} (GeV/#it{c});Tracks", 100, 0, 20);
  DOTH1F(hp_Pr, ";#it{p} (GeV/#it{c});Tracks", 100, 0, 20);
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
      UChar_t clustermap = i.itsClusterMap();
      bool issel = (i.tpcNClsFindable() > 70) && (i.flags() & 0x4) && (TESTBIT(clustermap, 0) || TESTBIT(clustermap, 1));
      issel = issel && (i.flags() & 0x2000);     //kTOFout
      issel = issel && (i.flags() & 0x80000000); //kTIME
      if (!issel)
        continue;
      if (TMath::Abs(i.nSigmaPi()) < 3) {
        hp_El->Fill(i.p());
        hpt_El->Fill(i.pt());
      } else if (TMath::Abs(i.nSigmaKa()) < 3) {
        hp_Ka->Fill(i.p());
        hpt_Ka->Fill(i.pt());
      } else if (TMath::Abs(i.nSigmaPr()) < 3) {
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
    adaptAnalysisTask<pidTOFTask>("pidTOF-task"),
    adaptAnalysisTask<TOFPIDQATask>("tofpidqa-task"),
    adaptAnalysisTask<SpectraTask>("filterEl-task")};
}
