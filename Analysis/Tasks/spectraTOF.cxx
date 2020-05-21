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
  DOTH1F(hp_NoCut, ";#it{p} (GeV/#it{c});Tracks", 100, 0, 20);
  DOTH1F(hp_TrkCut, ";#it{p} (GeV/#it{c});Tracks", 100, 0, 20);
  DOTH1F(hp_TOFCut, ";#it{p} (GeV/#it{c});Tracks", 100, 0, 20);
  //
  DOTH1F(hlength_NoCut, ";Track Length (cm);Tracks", 100, 0, 1000);
  DOTH1F(htime_NoCut, ";TOF Time (ns);Tracks", 1000, 0, 600);
  DOTH1F(hevtime_NoCut, ";Event time (ns);Tracks", 100, -2, 2);
  // NSigma
  DOTH2F(hnsigmaPi_NoCut, ";#it{p} (GeV/#it{c});Tracks", 100, 0, 5, 100, -10, 10);
  DOTH2F(hnsigmaKa_NoCut, ";#it{p} (GeV/#it{c});Tracks", 100, 0, 5, 100, -10, 10);
  DOTH2F(hnsigmaPr_NoCut, ";#it{p} (GeV/#it{c});Tracks", 100, 0, 5, 100, -10, 10);
  // Beta
  DOTH2F(hp_beta, ";#it{p} (GeV/#it{c});TOF #beta;Tracks", 100, 0, 20, 100, 0, 2);

  void process(aod::Collision const& collision, soa::Join<aod::Tracks, aod::TracksExtra, aod::pidRespTOF> const& tracks)
  {
    for (auto i : tracks) {
      float Mom = p(i.eta(), i.signed1Pt());
      hp_NoCut->Fill(Mom);
      // Track selection
      UChar_t clustermap = i.itsClusterMap();
      bool issel = (i.tpcNClsFindable() > 70) && (i.flags() & 0x4) && (TESTBIT(clustermap, 0) || TESTBIT(clustermap, 1));
      if (issel)
        hp_TrkCut->Fill(Mom);
      issel = issel && (i.flags() & 0x2000);     //kTOFout
      issel = issel && (i.flags() & 0x80000000); //kTIME
      if (issel)
        hp_TOFCut->Fill(Mom);
      //
      hlength_NoCut->Fill(i.length());
      htime_NoCut->Fill(i.tofSignal() / 1000);
      //
      hevtime_NoCut->Fill(collision.collisionTime() / 1000);
      // hevtime_NoCut->Fill(collision.collisionTime0() / 1000);
      //
      hnsigmaPi_NoCut->Fill(i.p(), i.nsigmaPi());
      hnsigmaKa_NoCut->Fill(i.p(), i.nsigmaKa());
      hnsigmaPr_NoCut->Fill(i.p(), i.nsigmaPr());
      // Beta
      hp_beta->Fill(i.p(), i.beta());
    }
  }
};

struct ExpectedTOFQATask {
// Diff between exp and computed exp time
#define TITBINNING(pname) Form(";#it{p} (GeV/#it{c});t_{Exp. Comp. Pr}(%s) - t_{Exp. %s} (ps);Tracks", pname, pname), 100, 0, 20, 2000, -10, 10
  DOTH2F(h_p_expdiff_El, TITBINNING("El"));
  DOTH2F(h_p_expdiff_Mu, TITBINNING("Mu"));
  DOTH2F(h_p_expdiff_Pi, TITBINNING("Pi"));
  DOTH2F(h_p_expdiff_Ka, TITBINNING("Ka"));
  DOTH2F(h_p_expdiff_Pr, TITBINNING("Pr"));
  DOTH2F(h_p_expdiff_De, TITBINNING("De"));
  DOTH2F(h_p_expdiff_Tr, TITBINNING("Tr"));
  DOTH2F(h_p_expdiff_He, TITBINNING("He"));
  DOTH2F(h_p_expdiff_Al, TITBINNING("Al"));
#undef TITBINNING
#define TITBINNING(pname) Form(";#it{p} (GeV/#it{c});#frac{t_{Exp. Comp. Pr}(%s) - t_{Exp. %s}}{t_{Exp. %s}};Tracks", pname, pname, pname), 100, 0, 20, 2000, -0.5, 0.5
  // Diff between exp time and computed exp time
  DOTH2F(h_p_expdiff_El_Rel, TITBINNING("El"));
  DOTH2F(h_p_expdiff_Mu_Rel, TITBINNING("Mu"));
  DOTH2F(h_p_expdiff_Pi_Rel, TITBINNING("Pi"));
  DOTH2F(h_p_expdiff_Ka_Rel, TITBINNING("Ka"));
  DOTH2F(h_p_expdiff_Pr_Rel, TITBINNING("Pr"));
  DOTH2F(h_p_expdiff_De_Rel, TITBINNING("De"));
  DOTH2F(h_p_expdiff_Tr_Rel, TITBINNING("Tr"));
  DOTH2F(h_p_expdiff_He_Rel, TITBINNING("He"));
  DOTH2F(h_p_expdiff_Al_Rel, TITBINNING("Al"));
#undef TITBINNING

  void process(aod::Collision const& collision, soa::Join<aod::Tracks, aod::TracksExtra, aod::pidRespTOF> const& tracks)
  {
    for (auto i : tracks) {
      // Track selection
      UChar_t clustermap = i.itsClusterMap();
      bool issel = (i.tpcNClsFindable() > 70) && (i.flags() & 0x4) && (TESTBIT(clustermap, 0) || TESTBIT(clustermap, 1));
      issel = issel && (i.flags() & 0x2000);     //kTOFout
      issel = issel && (i.flags() & 0x80000000); //kTIME
      if (!issel)
        continue;
      // Diff
      // const Float_t expp = ComputeExpectedMomentum(i.tofExpEl(), i.length(), kElectronMass);
      // const Float_t expp = ComputeExpectedMomentum(i.tofExpMu(), i.length(), kMuonMass);
      // const Float_t expp = ComputeExpectedMomentum(i.tofExpPi(), i.length(), kPionMass);
      // const Float_t expp = ComputeExpectedMomentum(i.tofExpKa(), i.length(), kKaonMass);
      const Float_t expp = ComputeExpectedMomentum(i.tofExpPr(), i.length(), kProtonMass);
      // const Float_t expp = ComputeExpectedMomentum(i.tofExpDe(), i.length(), kDeuteronMass);
      // const Float_t expp = ComputeExpectedMomentum(i.tofExpTr(), i.length(), kH3Mass);
      // const Float_t expp = ComputeExpectedMomentum(i.tofExpHe(), i.length(), kHe3Mass);
      // const Float_t expp = ComputeExpectedMomentum(i.tofExpAl(), i.length(), kHe4Mass);
      h_p_expdiff_El->Fill(i.p(), ComputeTOFExpTime(expp, i.length(), kElectronMass) - i.tofExpEl());
      h_p_expdiff_Mu->Fill(i.p(), ComputeTOFExpTime(expp, i.length(), kMuonMass) - i.tofExpMu());
      h_p_expdiff_Pi->Fill(i.p(), ComputeTOFExpTime(expp, i.length(), kPionMass) - i.tofExpPi());
      h_p_expdiff_Ka->Fill(i.p(), ComputeTOFExpTime(expp, i.length(), kKaonMass) - i.tofExpKa());
      h_p_expdiff_Pr->Fill(i.p(), ComputeTOFExpTime(expp, i.length(), kProtonMass) - i.tofExpPr());
      h_p_expdiff_De->Fill(i.p(), ComputeTOFExpTime(expp, i.length(), kDeuteronMass) - i.tofExpDe());
      h_p_expdiff_Tr->Fill(i.p(), ComputeTOFExpTime(expp, i.length(), kH3Mass) - i.tofExpTr());
      // h_p_expdiff_He->Fill(i.p(), ComputeTOFExpTime(expp, i.length(), kHe3Mass) - i.tofExpHe());
      // h_p_expdiff_Al->Fill(i.p(), ComputeTOFExpTime(expp, i.length(), kHe4Mass) - i.tofExpAl());
      // Diff Rel
      h_p_expdiff_El_Rel->Fill(i.p(), (ComputeTOFExpTime(expp, i.length(), kElectronMass) - i.tofExpEl()) / i.tofExpEl());
      h_p_expdiff_Mu_Rel->Fill(i.p(), (ComputeTOFExpTime(expp, i.length(), kMuonMass) - i.tofExpMu()) / i.tofExpMu());
      h_p_expdiff_Pi_Rel->Fill(i.p(), (ComputeTOFExpTime(expp, i.length(), kPionMass) - i.tofExpPi()) / i.tofExpPi());
      h_p_expdiff_Ka_Rel->Fill(i.p(), (ComputeTOFExpTime(expp, i.length(), kKaonMass) - i.tofExpKa()) / i.tofExpKa());
      h_p_expdiff_Pr_Rel->Fill(i.p(), (ComputeTOFExpTime(expp, i.length(), kProtonMass) - i.tofExpPr()) / i.tofExpPr());
      h_p_expdiff_De_Rel->Fill(i.p(), (ComputeTOFExpTime(expp, i.length(), kDeuteronMass) - i.tofExpDe()) / i.tofExpDe());
      h_p_expdiff_Tr_Rel->Fill(i.p(), (ComputeTOFExpTime(expp, i.length(), kH3Mass) - i.tofExpTr()) / i.tofExpTr());
      // h_p_expdiff_He_Rel->Fill(i.p(), (ComputeTOFExpTime(expp, i.length(), kHe3Mass) - i.tofExpHe()) / i.tofExpHe());
      // h_p_expdiff_Al_Rel->Fill(i.p(), (ComputeTOFExpTime(expp, i.length(), kHe4Mass) - i.tofExpAl()) / i.tofExpAl());
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

  // void process(soa::Filtered<soa::Join<aod::Tracks, aod::TracksExtra, aod::pidRespTOF>> const& tracks)
  void process(soa::Join<aod::Tracks, aod::TracksExtra, aod::pidRespTOF> const& tracks)
  {
    for (auto i : tracks) {
      UChar_t clustermap = i.itsClusterMap();
      bool issel = (i.tpcNClsFindable() > 70) && (i.flags() & 0x4) && (TESTBIT(clustermap, 0) || TESTBIT(clustermap, 1));
      issel = issel && (i.flags() & 0x2000);     //kTOFout
      issel = issel && (i.flags() & 0x80000000); //kTIME
      if (!issel)
        continue;
      if (TMath::Abs(i.nsigmaPi()) < 3) {
        hp_El->Fill(i.p());
        hpt_El->Fill(i.pt());
      } else if (TMath::Abs(i.nsigmaKa()) < 3) {
        hp_Ka->Fill(i.p());
        hpt_Ka->Fill(i.pt());
      } else if (TMath::Abs(i.nsigmaPr()) < 3) {
        hp_Pr->Fill(i.p());
        hpt_Pr->Fill(i.pt());
      }
      if (TMath::Abs(i.separationbetael() < 1.f)) {
        hp_El->Fill(i.p());
        hpt_El->Fill(i.pt());
        //
        hlength_El->Fill(i.length());
        htime_El->Fill(i.tofSignal() / 1000);
        // hevtime_El->Fill(collision.eventTime() / 1000);
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
    adaptAnalysisTask<ExpectedTOFQATask>("expectedtofqa-task"),
    adaptAnalysisTask<SpectraTask>("filterEl-task")};
}
