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
      hm->AddHistogram(histClass, "IsINT7inMUON", "INT7inMUON", false, 2, -0.5, 1.5, VarManager::kIsINT7inMUON);
      hm->AddHistogram(histClass, "IsMuonSingleLowPt7", "Is MuonSingleLowPt7", false, 2, -0.5, 1.5, VarManager::kIsMuonSingleLowPt7);
      hm->AddHistogram(histClass, "IsMuonUnlikeLowPt7", "Is MuonUnlikeLowPt7", false, 2, -0.5, 1.5, VarManager::kIsMuonUnlikeLowPt7);
      hm->AddHistogram(histClass, "IsMuonLikeLowPt7", "Is MuonLikeLowPt7", false, 2, -0.5, 1.5, VarManager::kIsMuonLikeLowPt7);
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
    }
    if (subGroupStr.Contains("tpc")) {
      hm->AddHistogram(histClass, "TPCncls", "Number of cluster in TPC", false, 160, -0.5, 159.5, VarManager::kTPCncls);
      hm->AddHistogram(histClass, "TPCncls_Run", "Number of cluster in TPC", true, (VarManager::GetNRuns() > 0 ? VarManager::GetNRuns() : 1), 0.5, 0.5 + VarManager::GetNRuns(), VarManager::kRunId,
                       10, -0.5, 159.5, VarManager::kTPCncls, 10, 0., 1., VarManager::kNothing, VarManager::GetRunStr().Data());
      hm->AddHistogram(histClass, "IsTPCrefit", "", false, 2, -0.5, 1.5, VarManager::kIsTPCrefit);
      hm->AddHistogram(histClass, "TPCchi2", "TPC chi2", false, 100, 0.0, 10.0, VarManager::kTPCchi2);
    }
    if (subGroupStr.Contains("tpcpid")) {
      hm->AddHistogram(histClass, "TPCdedx_pIN", "TPC dE/dx vs pIN", false, 200, 0.0, 20.0, VarManager::kPin, 200, 0.0, 200., VarManager::kTPCsignal);
    }
    if (subGroupStr.Contains("dca")) {
      hm->AddHistogram(histClass, "DCAxy", "DCAxy", false, 100, -3.0, 3.0, VarManager::kTrackDCAxy);
      hm->AddHistogram(histClass, "DCAz", "DCAz", false, 100, -5.0, 5.0, VarManager::kTrackDCAz);
    }
    if (subGroupStr.Contains("muon")) {
      hm->AddHistogram(histClass, "InvBendingMom", "", false, 100, 0.0, 1.0, VarManager::kMuonInvBendingMomentum);
      hm->AddHistogram(histClass, "ThetaX", "", false, 100, -1.0, 1.0, VarManager::kMuonThetaX);
      hm->AddHistogram(histClass, "ThetaY", "", false, 100, -2.0, 2.0, VarManager::kMuonThetaY);
      hm->AddHistogram(histClass, "ZMu", "", false, 100, -30.0, 30.0, VarManager::kMuonZMu);
      hm->AddHistogram(histClass, "BendingCoor", "", false, 100, 0.32, 0.35, VarManager::kMuonBendingCoor);
      hm->AddHistogram(histClass, "NonBendingCoor", "", false, 100, 0.065, 0.07, VarManager::kMuonNonBendingCoor);
      hm->AddHistogram(histClass, "Chi2", "", false, 100, 0.0, 200.0, VarManager::kMuonChi2);
      hm->AddHistogram(histClass, "Chi2MatchTrigger", "", false, 100, 0.0, 20.0, VarManager::kMuonChi2MatchTrigger);
      hm->AddHistogram(histClass, "RAtAbsorberEnd", "", false, 140, 10, 150, VarManager::kMuonRAtAbsorberEnd);
      hm->AddHistogram(histClass, "p x dca", "", false, 700, 0.0, 700, VarManager::kMuonRAtAbsorberEnd);
    }
  }

  if (groupStr.Contains("pair")) {
    hm->AddHistogram(histClass, "Mass", "", false, 125, 0.0, 5.0, VarManager::kMass);
    hm->AddHistogram(histClass, "Mass_Pt", "", false, 125, 0.0, 5.0, VarManager::kMass, 100, 0.0, 20.0, VarManager::kPt);
    hm->AddHistogram(histClass, "Eta_Pt", "", false, 125, -2.0, 2.0, VarManager::kEta, 100, 0.0, 20.0, VarManager::kPt);
    hm->AddHistogram(histClass, "Mass_VtxZ", "", true, 30, -15.0, 15.0, VarManager::kVtxZ, 100, 0.0, 20.0, VarManager::kMass);
  }

  if (groupStr.Contains("pair-hadron-mass")) {
    hm->AddHistogram(histClass, "Mass_Pt", "", false, 40, 0.0, 20.0, VarManager::kPairMass, 40, 0.0, 20.0, VarManager::kPairPt);
  }

  if (groupStr.Contains("pair-hadron-correlation")) {
    hm->AddHistogram(histClass, "DeltaEta_DeltaPhi", "", false, 20, -2.0, 2.0, VarManager::kDeltaEta, 50, -8.0, 8.0, VarManager::kDeltaPhi);
    hm->AddHistogram(histClass, "DeltaEta_DeltaPhiSym", "", false, 20, -2.0, 2.0, VarManager::kDeltaEta, 50, -8.0, 8.0, VarManager::kDeltaPhiSym);
  }
}
