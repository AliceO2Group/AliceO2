// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
//
// Contact: iarsene@cern.ch, i.c.arsene@fys.uio.no
//
#include <TString.h>
#include "PWGDQCore/HistogramManager.h"
#include "PWGDQCore/VarManager.h"

namespace o2::aod
{
namespace dqhistograms
{
void DefineHistograms(HistogramManager* hm, const char* histClass, const char* groupName, const char* subGroupName = "");
}
} // namespace o2::aod

void o2::aod::dqhistograms::DefineHistograms(HistogramManager* hm, const char* histClass, const char* groupName, const char* subGroupName)
{
  //
  // Add a predefined group of histograms to the HistogramManager hm and histogram class histClass
  // NOTE: The groupName and subGroupName arguments may contain several keywords, but the user should take care of
  //       ambiguities. TODO: fix it!
  // NOTE: All of the histograms which match any of the group or subgroup names will be added to the same histogram class !!
  //            So one has to make sure not to mix e.g. event-wise with track-wise histograms
  // NOTE: The subgroup name can be empty. In this case just a minimal set of histograms corresponding to the group name will be defined
  //
  TString groupStr = groupName;
  groupStr.ToLower();
  TString subGroupStr = subGroupName;
  subGroupStr.ToLower();
  if (groupStr.Contains("event")) {
    hm->AddHistogram(histClass, "VtxZ", "Vtx Z", false, 60, -15.0, 15.0, VarManager::kVtxZ);

    if (subGroupStr.Contains("trigger")) {
      hm->AddHistogram(histClass, "IsINT7", "Is INT7", false, 2, -0.5, 1.5, VarManager::kIsINT7);
      if (subGroupStr.Contains("muon") || subGroupStr.Contains("all")) {
        hm->AddHistogram(histClass, "IsINT7inMUON", "INT7inMUON", false, 2, -0.5, 1.5, VarManager::kIsINT7inMUON);
        hm->AddHistogram(histClass, "IsMuonSingleLowPt7", "Is MuonSingleLowPt7", false, 2, -0.5, 1.5, VarManager::kIsMuonSingleLowPt7);
        hm->AddHistogram(histClass, "IsMuonSingleHighPt7", "Is MuonSingleHighPt7", false, 2, -0.5, 1.5, VarManager::kIsMuonSingleHighPt7);
        hm->AddHistogram(histClass, "IsMuonUnlikeLowPt7", "Is MuonUnlikeLowPt7", false, 2, -0.5, 1.5, VarManager::kIsMuonUnlikeLowPt7);
        hm->AddHistogram(histClass, "IsMuonLikeLowPt7", "Is MuonLikeLowPt7", false, 2, -0.5, 1.5, VarManager::kIsMuonLikeLowPt7);
      }
      if (subGroupStr.Contains("up") || subGroupStr.Contains("all")) {
        hm->AddHistogram(histClass, "IsCUP8", "CUP8", false, 2, -0.5, 1.5, VarManager::kIsCUP8);
        hm->AddHistogram(histClass, "IsCUP9", "CUP9", false, 2, -0.5, 1.5, VarManager::kIsCUP9);
        hm->AddHistogram(histClass, "IsMUP10", "MUP10", false, 2, -0.5, 1.5, VarManager::kIsMUP10);
        hm->AddHistogram(histClass, "IsMUP11", "MUP11", false, 2, -0.5, 1.5, VarManager::kIsMUP11);
      }
      if (subGroupStr.Contains("emc") || subGroupStr.Contains("all")) {
        hm->AddHistogram(histClass, "IsEMC7", "EMC7", false, 2, -0.5, 1.5, VarManager::kIsEMC7);
      }
    }
    if (subGroupStr.Contains("vtx")) {
      hm->AddHistogram(histClass, "VtxX", "Vtx X", false, 100, -0.5, 0.5, VarManager::kVtxX);
      hm->AddHistogram(histClass, "VtxY", "Vtx Y", false, 100, -0.5, 0.5, VarManager::kVtxY);
      hm->AddHistogram(histClass, "VtxYVtxX", "Vtx Y vs Vtx X", false, 50, -0.5, 0.5, VarManager::kVtxX, 50, -0.5, 0.5, VarManager::kVtxY);
    }
    if (subGroupStr.Contains("vtxpp")) {
      hm->AddHistogram(histClass, "VtxNContrib", "Vtx n contributors", false, 100, 0.0, 100.0, VarManager::kVtxNcontrib);
    }
    if (subGroupStr.Contains("vtxPbPb")) {
      hm->AddHistogram(histClass, "VtxNContrib", "Vtx n contributors", false, 100, 0.0, 20000.0, VarManager::kVtxNcontrib);
    }
    if (subGroupStr.Contains("cent")) {
      hm->AddHistogram(histClass, "CentV0M", "CentV0M", false, 100, 0., 100., VarManager::kCentVZERO);
      hm->AddHistogram(histClass, "CentV0M_vtxZ", "CentV0M vs Vtx Z", false, 60, -15.0, 15.0, VarManager::kVtxZ, 20, 0., 100., VarManager::kCentVZERO);
    }
  }

  if (groupStr.Contains("track")) {
    hm->AddHistogram(histClass, "Pt", "p_{T} distribution", false, 200, 0.0, 20.0, VarManager::kPt);
    hm->AddHistogram(histClass, "Eta", "#eta distribution", false, 500, -5.0, 5.0, VarManager::kEta);
    hm->AddHistogram(histClass, "Phi", "#varphi distribution", false, 500, -6.3, 6.3, VarManager::kPhi);

    if (subGroupStr.Contains("kine")) {
      hm->AddHistogram(histClass, "Phi_Eta", "#phi vs #eta distribution", false, 200, -5.0, 5.0, VarManager::kEta, 200, -6.3, 6.3, VarManager::kPhi);
      hm->AddHistogram(histClass, "Eta_Pt", "", false, 20, -1.0, 1.0, VarManager::kEta, 100, 0.0, 20.0, VarManager::kPt);
      hm->AddHistogram(histClass, "Px", "p_{x} distribution", false, 200, 0.0, 20.0, VarManager::kPx);
      hm->AddHistogram(histClass, "Py", "p_{y} distribution", false, 200, 0.0, 20.0, VarManager::kPy);
      hm->AddHistogram(histClass, "Pz", "p_{z} distribution", false, 200, 0.0, 20.0, VarManager::kPz);
    }
    if (subGroupStr.Contains("its")) {
      hm->AddHistogram(histClass, "ITSncls", "Number of cluster in ITS", false, 8, -0.5, 7.5, VarManager::kITSncls);
      hm->AddHistogram(histClass, "ITSchi2", "ITS chi2", false, 100, 0.0, 50.0, VarManager::kITSchi2);
      hm->AddHistogram(histClass, "IsITSrefit", "", false, 2, -0.5, 1.5, VarManager::kIsITSrefit);
      hm->AddHistogram(histClass, "IsSPDany", "", false, 2, -0.5, 1.5, VarManager::kIsSPDany);
      hm->AddHistogram(histClass, "IsSPDfirst", "", false, 2, -0.5, 1.5, VarManager::kIsSPDfirst);
    }
    if (subGroupStr.Contains("tpc")) {
      hm->AddHistogram(histClass, "TPCncls", "Number of cluster in TPC", false, 160, -0.5, 159.5, VarManager::kTPCncls);
      hm->AddHistogram(histClass, "TPCncls_Run", "Number of cluster in TPC", true, (VarManager::GetNRuns() > 0 ? VarManager::GetNRuns() : 1), 0.5, 0.5 + VarManager::GetNRuns(), VarManager::kRunId,
                       10, -0.5, 159.5, VarManager::kTPCncls, 10, 0., 1., VarManager::kNothing, VarManager::GetRunStr().Data());
      hm->AddHistogram(histClass, "TPCnclsCR", "Number of crossed rows in TPC", false, 160, -0.5, 159.5, VarManager::kTPCnclsCR);
      hm->AddHistogram(histClass, "IsTPCrefit", "", false, 2, -0.5, 1.5, VarManager::kIsTPCrefit);
      hm->AddHistogram(histClass, "IsGoldenChi2", "", false, 2, -0.5, 1.5, VarManager::kIsGoldenChi2);
      hm->AddHistogram(histClass, "TPCchi2", "TPC chi2", false, 100, 0.0, 10.0, VarManager::kTPCchi2);
    }
    if (subGroupStr.Contains("tpcpid")) {
      hm->AddHistogram(histClass, "TPCdedx_pIN", "TPC dE/dx vs pIN", false, 200, 0.0, 10.0, VarManager::kPin, 200, 0.0, 200., VarManager::kTPCsignal);
      hm->AddHistogram(histClass, "TPCdedxRandomized_pIN", "TPC dE/dx (randomized) vs pIN", false, 200, 0.0, 10.0, VarManager::kPin, 200, 0.0, 200., VarManager::kTPCsignalRandomized);
      hm->AddHistogram(histClass, "TPCdedxRandomizedDelta_pIN", "TPC dE/dx (randomized - delta) vs pIN", false, 200, 0.0, 10.0, VarManager::kPin, 100, 0.0, 10., VarManager::kTPCsignalRandomizedDelta);
      hm->AddHistogram(histClass, "TPCnSigEle_pIN", "TPC n-#sigma(e) vs pIN", false, 200, 0.0, 10.0, VarManager::kPin, 100, -5.0, 5.0, VarManager::kTPCnSigmaEl);
      hm->AddHistogram(histClass, "TPCnSigEleRandomized_pIN", "TPC n-#sigma(e) - randomized - vs pIN", false, 200, 0.0, 10.0, VarManager::kPin, 100, -5.0, 5.0, VarManager::kTPCnSigmaElRandomized);
      hm->AddHistogram(histClass, "TPCnSigEleRandomizedDelta_pIN", "TPC n-#sigma(e) - randomized delta - vs pIN", false, 20, 0.0, 10.0, VarManager::kPin, 200, -0.5, 0.5, VarManager::kTPCnSigmaElRandomizedDelta);
      hm->AddHistogram(histClass, "TPCnSigEleRandomized_TPCnSigEle", "TPC n-#sigma(e) - randomized - vs TPC n-#sigma(e)", false, 100, -5.0, 5.0, VarManager::kTPCnSigmaEl, 100, -5.0, 5.0, VarManager::kTPCnSigmaElRandomized);
      hm->AddHistogram(histClass, "TPCnSigPiRandomized_TPCnSigPi", "TPC n-#sigma(#pi) - randomized - vs TPC n-#sigma(#pi)", false, 100, -5.0, 5.0, VarManager::kTPCnSigmaPi, 100, -5.0, 5.0, VarManager::kTPCnSigmaPiRandomized);
      hm->AddHistogram(histClass, "TPCnSigPrRandomized_TPCnSigPr", "TPC n-#sigma(p) - randomized - vs TPC n-#sigma(p)", false, 100, -5.0, 5.0, VarManager::kTPCnSigmaPr, 100, -5.0, 5.0, VarManager::kTPCnSigmaPrRandomized);
      hm->AddHistogram(histClass, "TPCnSigPiRandomized_pIN", "TPC n-#sigma(#pi) - randomized - vs pIN", false, 200, 0.0, 10.0, VarManager::kPin, 100, -5.0, 5.0, VarManager::kTPCnSigmaPiRandomized);
      hm->AddHistogram(histClass, "TPCnSigPrRandomized_pIN", "TPC n-#sigma(p) - randomized - vs pIN", false, 200, 0.0, 10.0, VarManager::kPin, 100, -5.0, 5.0, VarManager::kTPCnSigmaPrRandomized);
    }
    if (subGroupStr.Contains("tofpid")) {
      hm->AddHistogram(histClass, "TOFbeta_pIN", "TOF #beta vs pIN", false, 200, 0.0, 20.0, VarManager::kPin, 120, 0.0, 1.2, VarManager::kTOFbeta);
      hm->AddHistogram(histClass, "TOFnSigEle_pIN", "TOF n-#sigma(e) vs pIN", false, 200, 0.0, 10.0, VarManager::kPin, 100, -5.0, 5.0, VarManager::kTOFnSigmaEl);
    }
    if (subGroupStr.Contains("dca")) {
      hm->AddHistogram(histClass, "DCAxy", "DCAxy", false, 100, -3.0, 3.0, VarManager::kTrackDCAxy);
      hm->AddHistogram(histClass, "DCAz", "DCAz", false, 100, -5.0, 5.0, VarManager::kTrackDCAz);
    }
    if (subGroupStr.Contains("muon")) {
      hm->AddHistogram(histClass, "MuonNClusters", "", false, 100, 0.0, 10.0, VarManager::kMuonNClusters);
      hm->AddHistogram(histClass, "pdca", "", false, 100, 0.0, 500., VarManager::kMuonPDca);
      hm->AddHistogram(histClass, "RAtAbsorberEnd", "", false, 100, 0.0, 200., VarManager::kMuonRAtAbsorberEnd);
      hm->AddHistogram(histClass, "Chi2", "", false, 100, 0.0, 200.0, VarManager::kMuonChi2);
      hm->AddHistogram(histClass, "Chi2MCHMID", "", false, 100, 0.0, 200.0, VarManager::kMuonChi2MatchMCHMID);
      hm->AddHistogram(histClass, "Chi2MCHMFT", "", false, 100, 0.0, 200.0, VarManager::kMuonChi2MatchMCHMFT);
      hm->AddHistogram(histClass, "Chi2MatchScoreMCHMFT", "", false, 100, 0.0, 200.0, VarManager::kMuonMatchScoreMCHMFT);
      hm->AddHistogram(histClass, "MuonCXX", "", false, 100, -1.0, 1.0, VarManager::kMuonCXX);
      hm->AddHistogram(histClass, "MuonCYY", "", false, 100, -1.0, 1.0, VarManager::kMuonCYY);
      hm->AddHistogram(histClass, "MuonCPhiPhi", "", false, 100, -1.0, 1.0, VarManager::kMuonCPhiPhi);
      hm->AddHistogram(histClass, "MuonCTglTgl", "", false, 100, -1.0, 1.0, VarManager::kMuonCTglTgl);
      hm->AddHistogram(histClass, "MuonC1Pt21Pt2", "", false, 100, -1.0, 1.0, VarManager::kMuonC1Pt21Pt2);
    }
  }

  if (groupStr.Contains("pair_lmee")) {
    hm->AddHistogram(histClass, "Mass", "", false, 500, 0.0, 5.0, VarManager::kMass);
    hm->AddHistogram(histClass, "Mass_Pt", "", false, 500, 0.0, 5.0, VarManager::kMass, 200, 0.0, 20.0, VarManager::kPt);
    hm->AddHistogram(histClass, "Eta_Pt", "", false, 40, -2.0, 2.0, VarManager::kEta, 200, 0.0, 20.0, VarManager::kPt);
    hm->AddHistogram(histClass, "Mass_VtxZ", "", true, 30, -15.0, 15.0, VarManager::kVtxZ, 500, 0.0, 5.0, VarManager::kMass);
  } else if (groupStr.Contains("pair_barrel")) {
    hm->AddHistogram(histClass, "Mass", "", false, 125, 0.0, 5.0, VarManager::kMass);
    hm->AddHistogram(histClass, "Mass_Pt", "", false, 125, 0.0, 5.0, VarManager::kMass, 100, 0.0, 20.0, VarManager::kPt);
    hm->AddHistogram(histClass, "Eta_Pt", "", false, 125, -2.0, 2.0, VarManager::kEta, 100, 0.0, 20.0, VarManager::kPt);
    hm->AddHistogram(histClass, "Mass_VtxZ", "", true, 30, -15.0, 15.0, VarManager::kVtxZ, 100, 0.0, 20.0, VarManager::kMass);
    hm->AddHistogram(histClass, "cosThetaHE", "", false, 100, -1.5, 1., VarManager::kCosThetaHE);
    if (subGroupStr.Contains("vertexing-barrel")) {
      hm->AddHistogram(histClass, "Lxy", "", false, 100, 0.0, 10.0, VarManager::kVertexingLxy);
      hm->AddHistogram(histClass, "Lxyz", "", false, 100, 0.0, 10.0, VarManager::kVertexingLxyz);
      hm->AddHistogram(histClass, "LxyErr", "", false, 100, 0.0, 10.0, VarManager::kVertexingLxyErr);
      hm->AddHistogram(histClass, "LxyzErr", "", false, 100, 0.0, 10.0, VarManager::kVertexingLxyzErr);
      hm->AddHistogram(histClass, "VtxingProcCode", "", false, 10, 0.0, 10.0, VarManager::kVertexingProcCode);
      hm->AddHistogram(histClass, "VtxingChi2PCA", "", false, 100, 0.0, 10.0, VarManager::kVertexingChi2PCA);
    }
  } else if (groupStr.Contains("pair_dimuon")) {
    hm->AddHistogram(histClass, "Mass", "", false, 750, 0.0, 15.0, VarManager::kMass);
    hm->AddHistogram(histClass, "Pt", "", false, 120, 0.0, 30.0, VarManager::kPt);
    hm->AddHistogram(histClass, "Rapidity", "", false, 200, 2.5, 4.0, VarManager::kRap);
    hm->AddHistogram(histClass, "Mass_Pt", "", false, 750, 0.0, 15.0, VarManager::kMass, 120, 0.0, 30.0, VarManager::kPt);
    hm->AddHistogram(histClass, "Mass_Rapidity", "", false, 750, 0.0, 15.0, VarManager::kMass, 200, 2.5, 4.0, VarManager::kRap);
    hm->AddHistogram(histClass, "Mass_VtxZ", "", true, 30, -15.0, 15.0, VarManager::kVtxZ, 750, 0.0, 15.0, VarManager::kMass);
  } else if (groupStr.Contains("pair_electronmuon")) {
    hm->AddHistogram(histClass, "Mass", "", false, 750, 0.0, 30.0, VarManager::kMass);
    hm->AddHistogram(histClass, "Pt", "", false, 120, 0.0, 30.0, VarManager::kPt);
    hm->AddHistogram(histClass, "Rapidity", "", false, 500, -1.0, 4.0, VarManager::kRap);
    hm->AddHistogram(histClass, "Mass_Pt", "", false, 750, 0.0, 30.0, VarManager::kMass, 120, 0.0, 30.0, VarManager::kPt);
    hm->AddHistogram(histClass, "Mass_Rapidity", "", false, 750, 0.0, 30.0, VarManager::kMass, 500, -1.0, 4.0, VarManager::kRap);
    hm->AddHistogram(histClass, "Mass_VtxZ", "", true, 30, -15.0, 15.0, VarManager::kVtxZ, 750, 0.0, 30.0, VarManager::kMass);
  }

  if (groupStr.Contains("dilepton-hadron-mass")) {
    hm->AddHistogram(histClass, "Mass_Pt", "", false, 40, 0.0, 20.0, VarManager::kPairMass, 40, 0.0, 20.0, VarManager::kPairPt);
  }

  if (groupStr.Contains("dilepton-hadron-correlation")) {
    hm->AddHistogram(histClass, "DeltaEta_DeltaPhi", "", false, 20, -2.0, 2.0, VarManager::kDeltaEta, 50, -8.0, 8.0, VarManager::kDeltaPhi);
    hm->AddHistogram(histClass, "DeltaEta_DeltaPhiSym", "", false, 20, -2.0, 2.0, VarManager::kDeltaEta, 50, -8.0, 8.0, VarManager::kDeltaPhiSym);
  }
}
