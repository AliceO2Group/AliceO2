// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "Framework/AnalysisDataModel.h"
#include "Analysis/EventSelection.h"
#include "Analysis/Centrality.h"
#include "Analysis/TrackSelection.h"
#include "Analysis/TrackSelectionTables.h"
#include "Analysis/MC.h"
#include <TH1F.h>
#include <TParticlePDG.h>
#include <TDatabasePDG.h>
#include <cmath>

using namespace o2;
using namespace o2::framework;

struct raacharged {
  // data members
  OutputObj<THnF> fHistTrack{"fHistTrack"};
  OutputObj<THnF> fHistEvent{"fHistEvent"};
  OutputObj<THnF> fHistMC{"fHistMC"};
  Configurable<int> selectedTracks{"select", 1, "Choice of track selection. 0 = no selection, 1 = globalTracks, 2 = globalTracksSDD"};
  // member functions
  void init(InitContext const&)
  {
    constexpr Int_t nbinsMultCent = 11;
    constexpr Int_t nnAcc = 54;
    constexpr Int_t nbinspT = 82;
    constexpr Int_t nChargeBins = 7;

    Double_t MultCent[] = {0., 5., 10., 20., 30., 40., 50., 60., 70., 80., 90., 100};
    Double_t nAccbins[55]; //mult6kcoarse option in AlidNdPtTools
    nAccbins[0] = -0.5;
    int i = 0;
    for (; i <= 10; i++) {
      nAccbins[i + 1] = nAccbins[i] + 1;
    }
    for (; i <= 10 + 9; i++) {
      nAccbins[i + 1] = nAccbins[i] + 10;
    }
    for (; i <= 10 + 9 + 9; i++) {
      nAccbins[i + 1] = nAccbins[i] + 100;
    }
    for (; i <= 10 + 9 + 9 + 25; i++) {
      nAccbins[i + 1] = nAccbins[i] + 200;
    }
    Double_t pTBins[] = {0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0, 32.0, 34.0, 36.0, 40.0, 45.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 180.0, 200.0};
    Double_t ChargeBins[] = {-2.5, -1.5, -0.5, 0., 0.5, 1.5, 2.5};

    Int_t nBinsTrack[4] = {nbinsMultCent - 1, nnAcc - 1, nbinspT - 1, nChargeBins - 1};
    Double_t minTrack[4] = {MultCent[0], nAccbins[0], pTBins[0], ChargeBins[0]};
    Double_t maxTrack[4] = {MultCent[nbinsMultCent - 1], nAccbins[nnAcc - 1], pTBins[nbinspT - 1], ChargeBins[nChargeBins - 1]};

    fHistTrack.setObject(new THnF("fHistTrack", "Hist. for tracks", 4, nBinsTrack, minTrack, maxTrack));
    fHistTrack->SetBinEdges(0, MultCent);
    fHistTrack->SetBinEdges(1, nAccbins);
    fHistTrack->SetBinEdges(2, pTBins);
    fHistTrack->SetBinEdges(3, ChargeBins);
    fHistTrack->GetAxis(0)->SetTitle("cent");
    fHistTrack->GetAxis(1)->SetTitle("nAcc");
    fHistTrack->GetAxis(2)->SetTitle("p_{T} (GeV/c)");
    fHistTrack->GetAxis(3)->SetTitle("Q");

    constexpr Int_t nbinszV = 13;
    Double_t ZvBins[] = {-30., -25., -20., -15., -10., -5., 0., 5., 10., 15., 20., 25., 30.};

    Int_t nBinsEvent[3] = {nbinsMultCent - 1, nnAcc - 1, nbinszV - 1};
    Double_t minEvent[3] = {MultCent[0], nAccbins[0], ZvBins[0]};
    Double_t maxEvent[3] = {MultCent[nbinsMultCent - 1], nAccbins[nnAcc - 1], ZvBins[nbinszV - 1]};

    fHistEvent.setObject(new THnF("fHistEvent", "Histogram for Events", 3, nBinsEvent, minEvent, maxEvent));
    fHistEvent->SetBinEdges(0, MultCent);
    fHistEvent->SetBinEdges(1, nAccbins);
    fHistEvent->SetBinEdges(2, ZvBins);
    fHistEvent->GetAxis(0)->SetTitle("cent");
    fHistEvent->GetAxis(1)->SetTitle("nAcc");
    fHistEvent->GetAxis(2)->SetTitle("Zv (cm)");

    constexpr Int_t nParTypeBins = 11;
    constexpr Int_t nMCinfo = 4;

    Double_t ParTypeBins[11] = {-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5}; // 0=e, 1=mu, 2=pi, 3=K, 4=p, 6=sigmaP, 7=sigmaM, 8=xi, 9=omega, 5=other
    Double_t MCinfoBins[4] = {-0.5, 0.5, 1.5, 2.5};                                      // 0=prim, 1=sec.(decays) 2=genprim , sec. from material not considered

    Int_t nBinsMC[5] = {nbinspT - 1, nParTypeBins - 1, nMCinfo - 1, nChargeBins - 1};
    Double_t minMC[5] = {pTBins[0], ParTypeBins[0], MCinfoBins[0], ChargeBins[0]};
    Double_t maxMC[5] = {pTBins[nbinspT - 1], ParTypeBins[nParTypeBins - 1], MCinfoBins[nMCinfo - 1], ChargeBins[nChargeBins - 1]};

    fHistMC.setObject(new THnF("fHistMC", "Hist. for MC Info", 4, nBinsMC, minMC, maxMC));
    fHistMC->SetBinEdges(0, pTBins);
    fHistMC->SetBinEdges(1, ParTypeBins);
    fHistMC->SetBinEdges(2, MCinfoBins);
    fHistMC->SetBinEdges(3, ChargeBins);
    fHistMC->GetAxis(0)->SetTitle("MCp_{T} (GeV/c)");
    fHistMC->GetAxis(1)->SetTitle("Particle Type");
    fHistMC->GetAxis(2)->SetTitle("MCinfo");
    fHistMC->GetAxis(3)->SetTitle("Q");
  }

  Int_t WhichParticle(Int_t pdgCode)
  { //see in AlidNdPtTools
    if (pdgCode == 0) {
      return -1;
    }
    if (pdgCode == std::abs(211)) {
      return 0; //pi+, pi-
    }
    if (pdgCode == std::abs(321)) {
      return 1; //K+, K-
    }
    if (pdgCode == std::abs(2212)) {
      return 2; //p, pbar
    }
    if (pdgCode == 3222 || pdgCode == -3112) {
      return 3; //sigmaPlus, SigmaBarMinus
    }
    if (pdgCode == 3112 || pdgCode == -3222) {
      return 4; //sigmaMinus, SigmaBarPlus
    }
    if (pdgCode == std::abs(11)) {
      return 5; //e-, e+
    }
    if (pdgCode == std::abs(3312)) {
      return 6; //XiP, XiM
    }
    if (pdgCode == std::abs(13)) {
      return 7; //mu,antimu
    }
    if (pdgCode == std::abs(3334)) {
      return 8; //OmegaP, OmegaM
    }

    return 9; //other
  }
  Double_t MCCharge(Int_t pdgCode)
  {
    TParticlePDG* par = TDatabasePDG::Instance()->GetParticle(pdgCode);
    Double_t charge = par->Charge() / 3.0;
    return charge;
  }

  Configurable<bool> isMC{"isMC", 1, "0 - data, 1 - MC"};

  void process(soa::Join<aod::Collisions, aod::McCollisionLabels, aod::EvSels, aod::Cents>::iterator const& collision, soa::Join<aod::Tracks, aod::McTrackLabels, aod::TrackSelection> const& tracks, aod::McParticles& mcParticles)
  {
    if (!collision.alias()[kINT7]) {
      return;
    }
    if (!collision.sel7()) {
      return;
    }

    Double_t eventValues[3] = {0.0, 0.0, collision.posZ()};
    fHistEvent->Fill(eventValues);

    for (auto& track : tracks) {
      if (selectedTracks == 1 && !track.isGlobalTrack()) {
        continue;
      } else if (selectedTracks == 2 && !track.isGlobalTrackSDD()) {
        continue;
      }

      Double_t trackValues[4] = {0.0, 0.0, track.pt(), (Double_t)track.charge()};
      fHistTrack->Fill(trackValues);

      Double_t mcInfoVal;
      if (!isMC) {
        continue;
      }
      if (MC::isPhysicalPrimary(mcParticles, track.label())) {
        mcInfoVal = 0.0;
      } else {
        mcInfoVal = 1.0;
      }

      Double_t MCpt = track.label().pt();
      Double_t parType = (Double_t)WhichParticle(track.label().pdgCode());
      Double_t MCcharge = (Double_t)track.charge();
      Double_t MCvalues[4] = {MCpt, parType, mcInfoVal, MCcharge};

      fHistMC->Fill(MCvalues);
    }
    if (isMC) {
      for (auto& mcParticle : mcParticles) {

        if (abs(mcParticle.eta()) > 0.8) {
          continue;
        }
        if (!MC::isPhysicalPrimary(mcParticles, mcParticle)) {
          continue;
        }

        Double_t MCpt = mcParticle.pt();
        Double_t parType = (Double_t)WhichParticle(mcParticle.pdgCode());
        Int_t pdg = (Int_t)mcParticle.pdgCode();
        Double_t MCcharge = MCCharge(pdg);

        if (MCcharge == 0.0) {
          continue;
        }
        Double_t MCvalues[4] = {MCpt, parType, 2.0, MCcharge};
        fHistMC->Fill(MCvalues);
      }
    }
  }
};

//--------------------------------------------------------------------
// Workflow definition
//--------------------------------------------------------------------
WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<raacharged>("raa-charged")};
}
