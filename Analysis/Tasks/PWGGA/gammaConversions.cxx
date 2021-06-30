// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Framework/AnalysisTask.h"
#include "Framework/AnalysisDataModel.h"

#include "AnalysisDataModel/EventSelection.h"

#include "AnalysisDataModel/StrangenessTables.h"
#include "AnalysisDataModel/HFSecondaryVertex.h" // for BigTracks

#include "AnalysisDataModel/PID/PIDResponse.h"
#include "AnalysisDataModel/PID/PIDTPC.h"

#include <TH1F.h>
#include <TVector3.h>

#include <cmath>
#include <iostream>

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;

using collisionIt = aod::Collisions::iterator;
using collisionEvSelIt = soa::Join<aod::Collisions, aod::EvSels>::iterator;

using tracksAndTPCInfo = soa::Join<aod::Tracks, aod::TracksExtra, aod::pidTPCEl, aod::pidTPCPi>;
using tracksAndTPCInfoMC = soa::Join<aod::Tracks, aod::TracksExtra, aod::pidTPCEl, aod::pidTPCPi, aod::McTrackLabels>;

void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  ConfigParamSpec optionDoMC{"doMC", VariantType::Bool, false, {"Use MC info"}};
  workflowOptions.push_back(optionDoMC);
}
#include "Framework/runDataProcessing.h"

struct GammaConversions {

  Configurable<float> fSinglePtCut{"fSinglePtCut", 0.04, "minimum daughter track pt"};

  Configurable<float> fEtaCut{"fEtaCut", 0.8, "accepted eta range"};

  Configurable<float> fMinR{"fMinR", 5., "minimum conversion radius of the V0s"};
  Configurable<float> fMaxR{"fMaxR", 180., "maximum conversion radius of the V0s"};

  Configurable<float> fPIDnSigmaBelowElectronLine{"fPIDnSigmaBelowElectronLine", -3., "minimum sigma electron PID for V0 daughter tracks"};

  Configurable<float> fPIDnSigmaAboveElectronLine{"fPIDnSigmaAboveElectronLine", 3., "maximum sigma electron PID for V0 daughter tracks"};

  Configurable<float> fPIDnSigmaAbovePionLine{"fPIDnSigmaAbovePionLine", 3., "minimum sigma to be over the pion line for low momentum tracks"}; //case 4: 3.0sigma, 1.0 sigma at high momentum

  Configurable<float> fPIDnSigmaAbovePionLineHighP{"fPIDnSigmaAbovePionLineHighP", 1., "minimum sigma to be over the pion line for high momentum tracks"};

  Configurable<float> fPIDMinPnSigmaAbovePionLine{"fPIDMinPnSigmaAbovePionLine", 0.4, "minimum track momentum to apply any pion rejection"}; //case 7:  // 0.4 GeV

  Configurable<float> fPIDMaxPnSigmaAbovePionLine{"fPIDMaxPnSigmaAbovePionLine", 8., "border between low and high momentum pion rejection"}; //case 7:  // 8. GeV

  Configurable<float> fMinTPCFoundOverFindableCls{"fMinTPCNClsFoundOverFindable", 0.6, "minimum ratio found tpc clusters over findable"}; //case 9:  // 0.6

  Configurable<float> fMinTPCCrossedRowsOverFindableCls{"fMinTPCCrossedRowsOverFindableCls", 0.0, "minimum ratio TPC crossed rows over findable clusters"};

  Configurable<float> fQtPtMax{"fQtPtMax", 0.11, "up to fQtMax, multiply the pt of the V0s by this value to get the maximum qt "};

  Configurable<float> fQtMax{"fQtMax", 0.040, "maximum qt"};
  Configurable<float> fMaxPhotonAsymmetry{"fMaxPhotonAsymmetry", 0.95, "maximum photon asymetry"};
  Configurable<float> fPsiPairCut{"fPsiPairCut", 0.1, "maximum psi angle of the track pair"};

  Configurable<float> fCosPAngleCut{"fCosPAngleCut", 0.85, "mimimum cosinus of the pointing angle"}; // case 4

  HistogramRegistry registry{
    "registry",
    {
      {"IsPhotonSelected", "IsPhotonSelected", {HistType::kTH1F, {{13, -0.5f, 11.5f}}}},

      {"beforeCuts/hPtRec", "hPtRec_before", {HistType::kTH1F, {{100, 0.0f, 25.0f}}}},
      {"beforeCuts/hEtaRec", "hEtaRec_before", {HistType::kTH1F, {{1000, -2.f, 2.f}}}},
      {"beforeCuts/hPhiRec", "hEtaRec_before", {HistType::kTH1F, {{1000, 0.f, 2.f * M_PI}}}},
      {"beforeCuts/hConvPointR", "hConvPointR_before", {HistType::kTH1F, {{800, 0.f, 200.f}}}},

      {"hPtRec", "hPtRec", {HistType::kTH1F, {{100, 0.0f, 25.0f}}}},
      {"hEtaRec", "hEtaRec", {HistType::kTH1F, {{1000, -2.f, 2.f}}}},
      {"hPhiRec", "hEtaRec", {HistType::kTH1F, {{1000, 0.f, 2.f * M_PI}}}},
      {"hConvPointR", "hConvPointR", {HistType::kTH1F, {{800, 0.f, 200.f}}}},

      {"hTPCdEdxSigEl", "hTPCdEdxSigEl", {HistType::kTH2F, {{150, 0.03f, 20.f}, {400, -10.f, 10.f}}}},
      {"hTPCdEdxSigPi", "hTPCdEdxSigPi", {HistType::kTH2F, {{150, 0.03f, 20.f}, {400, -10.f, 10.f}}}},
      {"hTPCdEdx", "hTPCdEdx", {HistType::kTH2F, {{150, 0.03f, 20.f}, {800, 0.f, 200.f}}}},

      {"hTPCFoundOverFindableCls", "hTPCFoundOverFindableCls", {HistType::kTH1F, {{100, 0.f, 1.f}}}},
      {"hTPCCrossedRowsOverFindableCls", "hTPCCrossedRowsOverFindableCls", {HistType::kTH1F, {{100, 0.f, 1.5f}}}},

      {"hArmenteros", "hArmenteros", {HistType::kTH2F, {{200, -1.f, 1.f}, {250, 0.f, 25.f}}}},
      {"hPsiPtRec", "hPsiPtRec", {HistType::kTH2F, {{500, -2.f, 2.f}, {100, 0.f, 25.f}}}},

      {"hCosPAngle", "hCosPAngle", {HistType::kTH1F, {{1000, -1.f, 1.2f}}}},

      // resolution histos
      {"resolutions/hPtRes", "hPtRes_Rec-MC", {HistType::kTH1F, {{1000, -0.5f, 0.5f}}}},
      {"resolutions/hEtaRes", "hEtaRes_Rec-MC", {HistType::kTH1F, {{1000, -2.f, 2.f}}}},
      {"resolutions/hPhiRes", "hPhiRes_Rec-MC", {HistType::kTH1F, {{1000, -M_PI, M_PI}}}},

      {"resolutions/hConvPointRRes", "hConvPointRRes_Rec-MC", {HistType::kTH1F, {{1000, -200.f, 200.f}}}},
      {"resolutions/hConvPointAbsoluteDistanceRes", "hConvPointAbsoluteDistanceRes", {HistType::kTH1F, {{1000, -0.0f, 200.f}}}},
    },
  };

  enum photonCuts {
    kPhotonIn = 0,
    kTrackEta,
    kTrackPt,
    kElectronPID,
    kPionRejLowMom,
    kPionRejHighMom,
    kTPCFoundOverFindableCls,
    kTPCCrossedRowsOverFindableCls,
    kV0Radius,
    kArmenteros,
    kPsiPair,
    kCosinePA,
    kPhotonOut
  };

  std::vector<TString> fPhotCutsLabels{
    "kPhotonIn",
    "kTrackEta",
    "kTrackPt",
    "kElectronPID",
    "kPionRejLowMom",
    "kPionRejHighMom",
    "kTPCFoundOverFindableCls",
    "kTPCCrossedRowsOverFindableCls",
    "kV0Radius",
    "kArmenteros",
    "kPsiPair",
    "kCosinePA",
    "kPhotonOut"};

  void init(InitContext const&)
  {
    TAxis* lXaxis = registry.get<TH1>(HIST("IsPhotonSelected"))->GetXaxis();
    for (size_t i = 0; i < fPhotCutsLabels.size(); ++i) {
      lXaxis->SetBinLabel(i + 1, fPhotCutsLabels[i]);
    }
  }

  template <typename TTRACKS, typename TCOLL, typename TV0>
  bool processPhoton(TCOLL const& theCollision, const TV0& theV0)
  {
    fillHistogramsBeforeCuts(theV0);

    auto lTrackPos = theV0.template posTrack_as<TTRACKS>(); //positive daughter
    auto lTrackNeg = theV0.template negTrack_as<TTRACKS>(); //negative daughter

    // apply track cuts
    if (!(trackPassesCuts(lTrackPos) && trackPassesCuts(lTrackNeg))) {
      return kFALSE;
    }

    float lV0CosinePA = theV0.v0cosPA(theCollision.posX(), theCollision.posY(), theCollision.posZ());

    // apply photon cuts
    if (!passesPhotonCuts(theV0, lV0CosinePA)) {
      return kFALSE;
    }

    fillHistogramsAfterCuts(theV0, lTrackPos, lTrackNeg, lV0CosinePA);

    return kTRUE;
  }

  template <typename TV0, typename TMC>
  void processTruePhoton(const TV0& theV0, const TMC& theMcParticles)
  {
    auto lTrackPos = theV0.template posTrack_as<tracksAndTPCInfoMC>(); //positive daughter
    auto lTrackNeg = theV0.template negTrack_as<tracksAndTPCInfoMC>(); //negative daughter

    // todo: verify it is enough to check only mother0 being equal
    if (lTrackPos.mcParticle().mother0() > -1 &&
        lTrackPos.mcParticle().mother0() == lTrackNeg.mcParticle().mother0()) {
      auto lMother = theMcParticles.iteratorAt(lTrackPos.mcParticle().mother0());

      if (lMother.pdgCode() == 22) {

        registry.fill(HIST("resolutions/hPtRes"), theV0.pt() - lMother.pt());
        registry.fill(HIST("resolutions/hEtaRes"), theV0.eta() - lMother.eta());
        registry.fill(HIST("resolutions/hPhiRes"), theV0.phi() - lMother.phi());

        TVector3 lConvPointRec(theV0.x(), theV0.y(), theV0.z());
        TVector3 lPosTrackVtxMC(lTrackPos.mcParticle().vx(), lTrackPos.mcParticle().vy(), lTrackPos.mcParticle().vz());
        // take the origin of the positive mc track as conversion point (should be identical with negative, verified this on a few photons)
        TVector3 lConvPointMC(lPosTrackVtxMC);

        registry.fill(HIST("resolutions/hConvPointRRes"), lConvPointRec.Perp() - lConvPointMC.Perp());
        registry.fill(HIST("resolutions/hConvPointAbsoluteDistanceRes"), TVector3(lConvPointRec - lConvPointMC).Mag());
      }
    }
  }

  void processData(collisionEvSelIt const& theCollision,
                   aod::V0Datas const& theV0s,
                   tracksAndTPCInfo const& theTracks)
  {
    if (!theCollision.alias()[kINT7] || !theCollision.sel7()) {
      return;
    }

    for (auto& lV0 : theV0s) {
      if (!processPhoton<tracksAndTPCInfo>(theCollision, lV0)) {
        continue;
      }
    }
  }

  void processMC(collisionIt const& theCollision,
                 aod::V0Datas const& theV0s,
                 tracksAndTPCInfoMC const& theTracks,
                 aod::McParticles const& theMcParticles)
  {
    //~ if (!theCollision.sel7()) {
    //~ return;
    //~ }

    for (auto& lV0 : theV0s) {
      if (!processPhoton<tracksAndTPCInfoMC>(theCollision, lV0)) {
        continue;
      }
      processTruePhoton(lV0, theMcParticles);
    }
  }

  template <typename T>
  void fillHistogramsBeforeCuts(const T& theV0)
  {
    // fill some QA histograms before any cuts
    registry.fill(HIST("beforeCuts/hPtRec"), theV0.pt());
    registry.fill(HIST("beforeCuts/hEtaRec"), theV0.eta());
    registry.fill(HIST("beforeCuts/hPhiRec"), theV0.phi());
    registry.fill(HIST("beforeCuts/hConvPointR"), theV0.v0radius());
    registry.fill(HIST("IsPhotonSelected"), kPhotonIn);
  }

  template <typename T>
  bool trackPassesCuts(const T& theTrack)
  {

    // single track eta cut
    if (TMath::Abs(theTrack.eta()) > fEtaCut) {
      registry.fill(HIST("IsPhotonSelected"), kTrackEta);
      return kFALSE;
    }

    // single track pt cut
    if (theTrack.pt() < fSinglePtCut) {
      registry.fill(HIST("IsPhotonSelected"), kTrackPt);
      return kFALSE;
    }

    if (!(selectionPIDTPC_track(theTrack))) {
      return kFALSE;
    }

    if (theTrack.tpcFoundOverFindableCls() < fMinTPCFoundOverFindableCls) {
      registry.fill(HIST("IsPhotonSelected"), kTPCFoundOverFindableCls);
      return kFALSE;
    }

    if (theTrack.tpcCrossedRowsOverFindableCls() < fMinTPCCrossedRowsOverFindableCls) {
      registry.fill(HIST("IsPhotonSelected"), kTPCCrossedRowsOverFindableCls);
      return kFALSE;
    }

    return kTRUE;
  }

  template <typename T>
  bool passesPhotonCuts(const T& theV0, float theV0CosinePA)
  {
    if (theV0.v0radius() < fMinR || theV0.v0radius() > fMaxR) {
      registry.fill(HIST("IsPhotonSelected"), kV0Radius);
      return kFALSE;
    }

    if (!ArmenterosQtCut(theV0.alpha(), theV0.qtarm(), theV0.pt())) {
      registry.fill(HIST("IsPhotonSelected"), kArmenteros);
      return kFALSE;
    }

    if (TMath::Abs(theV0.psipair()) > fPsiPairCut) {
      registry.fill(HIST("IsPhotonSelected"), kPsiPair);
      return kFALSE;
    }

    if (theV0CosinePA < fCosPAngleCut) {
      registry.fill(HIST("IsPhotonSelected"), kCosinePA);
      return kFALSE;
    }

    return kTRUE;
  }

  template <typename TV0, typename TTRACK>
  void fillHistogramsAfterCuts(const TV0& theV0, const TTRACK& theTrackPos, const TTRACK& theTrackNeg, float theV0CosinePA)
  {
    registry.fill(HIST("IsPhotonSelected"), kPhotonOut);

    registry.fill(HIST("hPtRec"), theV0.pt());
    registry.fill(HIST("hEtaRec"), theV0.eta());
    registry.fill(HIST("hPhiRec"), theV0.phi());
    registry.fill(HIST("hConvPointR"), theV0.v0radius());

    registry.fill(HIST("hTPCdEdxSigEl"), theTrackNeg.p(), theTrackNeg.tpcNSigmaEl());
    registry.fill(HIST("hTPCdEdxSigEl"), theTrackPos.p(), theTrackPos.tpcNSigmaEl());
    registry.fill(HIST("hTPCdEdxSigPi"), theTrackNeg.p(), theTrackNeg.tpcNSigmaPi());
    registry.fill(HIST("hTPCdEdxSigPi"), theTrackPos.p(), theTrackPos.tpcNSigmaPi());

    registry.fill(HIST("hTPCdEdx"), theTrackNeg.p(), theTrackNeg.tpcSignal());
    registry.fill(HIST("hTPCdEdx"), theTrackPos.p(), theTrackPos.tpcSignal());

    registry.fill(HIST("hTPCFoundOverFindableCls"), theTrackNeg.tpcFoundOverFindableCls());
    registry.fill(HIST("hTPCFoundOverFindableCls"), theTrackPos.tpcFoundOverFindableCls());

    registry.fill(HIST("hTPCCrossedRowsOverFindableCls"), theTrackNeg.tpcCrossedRowsOverFindableCls());
    registry.fill(HIST("hTPCCrossedRowsOverFindableCls"), theTrackPos.tpcCrossedRowsOverFindableCls());

    registry.fill(HIST("hArmenteros"), theV0.alpha(), theV0.qtarm());
    registry.fill(HIST("hPsiPtRec"), theV0.psipair(), theV0.pt());

    registry.fill(HIST("hCosPAngle"), theV0CosinePA);
  }

  Bool_t ArmenterosQtCut(Double_t theAlpha, Double_t theQt, Double_t thePt)
  {
    // in AliPhysics this is the cut for if fDo2DQt && fDoQtGammaSelection == 2
    Float_t lQtMaxPtDep = fQtPtMax * thePt;
    if (lQtMaxPtDep > fQtMax) {
      lQtMaxPtDep = fQtMax;
    }
    if (!(TMath::Power(theAlpha / fMaxPhotonAsymmetry, 2) + TMath::Power(theQt / lQtMaxPtDep, 2) < 1)) {
      return kFALSE;
    }
    return kTRUE;
  }

  template <typename T>
  bool selectionPIDTPC_track(const T& theTrack)
  {
    // TPC Electron Line
    if (theTrack.tpcNSigmaEl() < fPIDnSigmaBelowElectronLine || theTrack.tpcNSigmaEl() > fPIDnSigmaAboveElectronLine) {
      registry.fill(HIST("IsPhotonSelected"), kElectronPID);
      return kFALSE;
    }

    // TPC Pion Line
    if (theTrack.p() > fPIDMinPnSigmaAbovePionLine) {
      // low pt Pion rej
      if (theTrack.p() < fPIDMaxPnSigmaAbovePionLine) {
        if (theTrack.tpcNSigmaEl() > fPIDnSigmaBelowElectronLine && theTrack.tpcNSigmaEl() < fPIDnSigmaAboveElectronLine && theTrack.tpcNSigmaPi() < fPIDnSigmaAbovePionLine) {
          registry.fill(HIST("IsPhotonSelected"), kPionRejLowMom);
          return kFALSE;
        }
      }
      // High Pt Pion rej
      else {
        if (theTrack.tpcNSigmaEl() > fPIDnSigmaBelowElectronLine && theTrack.tpcNSigmaEl() < fPIDnSigmaAboveElectronLine && theTrack.tpcNSigmaPi() < fPIDnSigmaAbovePionLineHighP) {
          registry.fill(HIST("IsPhotonSelected"), kPionRejHighMom);
          return kFALSE;
        }
      }
    }
    return kTRUE;
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  if (!cfgc.options().get<bool>("doMC")) {
    return WorkflowSpec{
      adaptAnalysisTask<GammaConversions>(cfgc, Processes{&GammaConversions::processData}),
    };
  }
  return WorkflowSpec{
    adaptAnalysisTask<GammaConversions>(cfgc, Processes{&GammaConversions::processMC}),
  };
}
