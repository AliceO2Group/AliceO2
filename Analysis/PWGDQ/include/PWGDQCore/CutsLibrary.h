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
#include "PWGDQCore/AnalysisCut.h"
#include "PWGDQCore/AnalysisCompositeCut.h"
#include "PWGDQCore/VarManager.h"

namespace o2::aod
{
namespace dqcuts
{
AnalysisCompositeCut* GetCompositeCut(const char* cutName);
AnalysisCut* GetAnalysisCut(const char* cutName);
} // namespace dqcuts
} // namespace o2::aod

AnalysisCompositeCut* o2::aod::dqcuts::GetCompositeCut(const char* cutName)
{
  //
  // define composie cuts, typically combinations of all the ingredients needed for a full cut
  //
  // TODO: Agree on some conventions for the naming
  //       Think of possible customization of the predefined cuts via names

  AnalysisCompositeCut* cut = new AnalysisCompositeCut(cutName, cutName);
  std::string nameStr = cutName;

  if (!nameStr.compare("jpsiKineAndQuality")) {
    cut->AddCut(GetAnalysisCut("jpsiStandardKine"));
    cut->AddCut(GetAnalysisCut("electronStandardQuality"));
    cut->AddCut(GetAnalysisCut("standardPrimaryTrack"));
    return cut;
  }

  if (!nameStr.compare("jpsiPID1")) {
    cut->AddCut(GetAnalysisCut("jpsiStandardKine")); // standard kine cuts usually are applied via Filter in the task
    cut->AddCut(GetAnalysisCut("electronStandardQuality"));
    cut->AddCut(GetAnalysisCut("standardPrimaryTrack"));
    cut->AddCut(GetAnalysisCut("electronPID1"));
    return cut;
  }

  if (!nameStr.compare("jpsiPID2")) {
    cut->AddCut(GetAnalysisCut("jpsiStandardKine"));
    cut->AddCut(GetAnalysisCut("electronStandardQuality"));
    cut->AddCut(GetAnalysisCut("standardPrimaryTrack"));
    cut->AddCut(GetAnalysisCut("electronPID2"));
    return cut;
  }

  if (!nameStr.compare("jpsiPIDnsigma")) {
    cut->AddCut(GetAnalysisCut("jpsiStandardKine"));
    cut->AddCut(GetAnalysisCut("electronStandardQuality"));
    cut->AddCut(GetAnalysisCut("standardPrimaryTrack"));
    cut->AddCut(GetAnalysisCut("electronPIDnsigma"));
    return cut;
  }

  //---------------------------------------------------------------------------------------
  // NOTE: Below there are several TPC pid cuts used for studies of the dE/dx degradation
  //    and its impact on the high lumi pp quarkonia triggers
  //  To be removed when not needed anymore
  if (!nameStr.compare("jpsiPID1Randomized")) {
    cut->AddCut(GetAnalysisCut("jpsiStandardKine")); // standard kine cuts usually are applied via Filter in the task
    cut->AddCut(GetAnalysisCut("electronStandardQuality"));
    cut->AddCut(GetAnalysisCut("standardPrimaryTrack"));
    cut->AddCut(GetAnalysisCut("electronPID1randomized"));
    return cut;
  }

  if (!nameStr.compare("jpsiPID2Randomized")) {
    cut->AddCut(GetAnalysisCut("jpsiStandardKine"));
    cut->AddCut(GetAnalysisCut("electronStandardQuality"));
    cut->AddCut(GetAnalysisCut("standardPrimaryTrack"));
    cut->AddCut(GetAnalysisCut("electronPID2randomized"));
    return cut;
  }

  if (!nameStr.compare("jpsiPIDnsigmaRandomized")) {
    cut->AddCut(GetAnalysisCut("jpsiStandardKine"));
    cut->AddCut(GetAnalysisCut("electronStandardQuality"));
    cut->AddCut(GetAnalysisCut("standardPrimaryTrack"));
    cut->AddCut(GetAnalysisCut("electronPIDnsigmaRandomized"));
    return cut;
  }

  if (!nameStr.compare("jpsiPIDworseRes")) {
    cut->AddCut(GetAnalysisCut("jpsiStandardKine"));
    cut->AddCut(GetAnalysisCut("electronStandardQuality"));
    cut->AddCut(GetAnalysisCut("standardPrimaryTrack"));
    cut->AddCut(GetAnalysisCut("electronPIDworseRes"));
    return cut;
  }

  if (!nameStr.compare("jpsiPIDshift")) {
    cut->AddCut(GetAnalysisCut("jpsiStandardKine"));
    cut->AddCut(GetAnalysisCut("electronStandardQuality"));
    cut->AddCut(GetAnalysisCut("standardPrimaryTrack"));
    cut->AddCut(GetAnalysisCut("electronPIDshift"));
    return cut;
  }

  if (!nameStr.compare("jpsiPID1shiftUp")) {
    cut->AddCut(GetAnalysisCut("jpsiStandardKine"));
    cut->AddCut(GetAnalysisCut("electronStandardQuality"));
    cut->AddCut(GetAnalysisCut("standardPrimaryTrack"));
    cut->AddCut(GetAnalysisCut("electronPID1shiftUp"));
    return cut;
  }

  if (!nameStr.compare("jpsiPID1shiftDown")) {
    cut->AddCut(GetAnalysisCut("jpsiStandardKine"));
    cut->AddCut(GetAnalysisCut("electronStandardQuality"));
    cut->AddCut(GetAnalysisCut("standardPrimaryTrack"));
    cut->AddCut(GetAnalysisCut("electronPID1shiftDown"));
    return cut;
  }
  // -------------------------------------------------------------------------------------------------

  if (!nameStr.compare("lmeePID_TPChadrejTOFrec")) {
    cut->AddCut(GetAnalysisCut("lmeeStandardKine"));
    cut->AddCut(GetAnalysisCut("TightGlobalTrack"));
    cut->AddCut(GetAnalysisCut("standardPrimaryTrack"));

    AnalysisCompositeCut* cut_tpc_hadrej = new AnalysisCompositeCut("pid_TPChadrej", "pid_TPChadrej", kTRUE);
    cut_tpc_hadrej->AddCut(GetAnalysisCut("tpc_electron"));
    cut_tpc_hadrej->AddCut(GetAnalysisCut("tpc_pion_rejection"));
    cut_tpc_hadrej->AddCut(GetAnalysisCut("tpc_kaon_rejection"));
    cut_tpc_hadrej->AddCut(GetAnalysisCut("tpc_proton_rejection"));

    AnalysisCompositeCut* cut_tof_rec = new AnalysisCompositeCut("pid_tof_rec", "pid_tof_rec", kTRUE);
    cut_tof_rec->AddCut(GetAnalysisCut("tpc_electron"));
    cut_tof_rec->AddCut(GetAnalysisCut("tof_electron"));

    AnalysisCompositeCut* cut_pid_OR = new AnalysisCompositeCut("pid_TPChadrejTOFrec", "pid_TPChadrejTOFrec", kFALSE);
    cut_pid_OR->AddCut(cut_tpc_hadrej);
    cut_pid_OR->AddCut(cut_tof_rec);
    cut->AddCut(cut_pid_OR);
    return cut;
  }

  if (!nameStr.compare("lmeePID_TPChadrej")) {
    cut->AddCut(GetAnalysisCut("lmeeStandardKine"));
    cut->AddCut(GetAnalysisCut("TightGlobalTrack"));
    cut->AddCut(GetAnalysisCut("standardPrimaryTrack"));

    AnalysisCompositeCut* cut_tpc_hadrej = new AnalysisCompositeCut("pid_TPChadrej", "pid_TPChadrej", kTRUE);
    cut_tpc_hadrej->AddCut(GetAnalysisCut("tpc_electron"));
    cut_tpc_hadrej->AddCut(GetAnalysisCut("tpc_pion_rejection"));
    cut_tpc_hadrej->AddCut(GetAnalysisCut("tpc_kaon_rejection"));
    cut_tpc_hadrej->AddCut(GetAnalysisCut("tpc_proton_rejection"));
    cut->AddCut(cut_tpc_hadrej);
    return cut;
  }

  if (!nameStr.compare("lmeePID_TOFrec")) {
    cut->AddCut(GetAnalysisCut("lmeeStandardKine"));
    cut->AddCut(GetAnalysisCut("TightGlobalTrack"));
    cut->AddCut(GetAnalysisCut("standardPrimaryTrack"));

    AnalysisCompositeCut* cut_tof_rec = new AnalysisCompositeCut("pid_tof_rec", "pid_tof_rec", kTRUE);
    cut_tof_rec->AddCut(GetAnalysisCut("tpc_electron"));
    cut_tof_rec->AddCut(GetAnalysisCut("tof_electron"));

    cut->AddCut(cut_tof_rec);
    return cut;
  }

  if (!nameStr.compare("lmee_GlobalTrack")) {
    cut->AddCut(GetAnalysisCut("lmeeStandardKine"));
    cut->AddCut(GetAnalysisCut("TightGlobalTrack"));
    cut->AddCut(GetAnalysisCut("standardPrimaryTrack"));
    return cut;
  }

  if (!nameStr.compare("muonQualityCuts")) {
    cut->AddCut(GetAnalysisCut("muonQualityCuts"));
    return cut;
  }

  if (!nameStr.compare("pairNoCut")) {
    cut->AddCut(GetAnalysisCut("pairNoCut"));
    return cut;
  }

  if (!nameStr.compare("pairMassLow")) {
    cut->AddCut(GetAnalysisCut("pairMassLow"));
    return cut;
  }

  if (!nameStr.compare("pairJpsi")) {
    cut->AddCut(GetAnalysisCut("pairJpsi"));
    return cut;
  }

  if (!nameStr.compare("pairPsi2S")) {
    cut->AddCut(GetAnalysisCut("pairPsi2S"));
    return cut;
  }

  if (!nameStr.compare("pairUpsilon")) {
    cut->AddCut(GetAnalysisCut("pairUpsilon"));
    return cut;
  }

  if (!nameStr.compare("pairJpsiLowPt1")) {
    cut->AddCut(GetAnalysisCut("pairJpsi"));
    cut->AddCut(GetAnalysisCut("pairPtLow1"));
    return cut;
  }

  if (!nameStr.compare("pairJpsiLowPt2")) {
    cut->AddCut(GetAnalysisCut("pairJpsi"));
    cut->AddCut(GetAnalysisCut("pairPtLow2"));
    return cut;
  }

  delete cut;
  return nullptr;
}

AnalysisCut* o2::aod::dqcuts::GetAnalysisCut(const char* cutName)
{
  //
  // define here cuts which are likely to be used often
  //
  AnalysisCut* cut = new AnalysisCut(cutName, cutName);
  std::string nameStr = cutName;

  if (!nameStr.compare("eventStandard")) {
    cut->AddCut(VarManager::kVtxZ, -10.0, 10.0);
    cut->AddCut(VarManager::kIsINT7, 0.5, 1.5);
    return cut;
  }

  if (!nameStr.compare("eventStandardNoINT7")) {
    cut->AddCut(VarManager::kVtxZ, -10.0, 10.0);
    return cut;
  }

  if (!nameStr.compare("eventDimuonStandard")) {
    cut->AddCut(VarManager::kIsMuonUnlikeLowPt7, 0.5, 1.5);
    return cut;
  }

  if (!nameStr.compare("eventMuonStandard")) {
    cut->AddCut(VarManager::kIsMuonSingleLowPt7, 0.5, 1.5);
    return cut;
  }

  if (!nameStr.compare("int7vtxZ5")) {
    cut->AddCut(VarManager::kVtxZ, -5.0, 5.0);
    cut->AddCut(VarManager::kIsINT7, 0.5, 1.5);
    return cut;
  }

  if (!nameStr.compare("jpsiStandardKine")) {
    cut->AddCut(VarManager::kPt, 1.0, 1000.0);
    cut->AddCut(VarManager::kEta, -0.9, 0.9);
    return cut;
  }

  if (!nameStr.compare("lmeeStandardKine")) {
    cut->AddCut(VarManager::kPt, 0.2, 10.0);
    cut->AddCut(VarManager::kEta, -0.8, 0.8);
    return cut;
  }

  if (!nameStr.compare("lmeeLowBKine")) {
    cut->AddCut(VarManager::kPt, 0.075, 10.0);
    cut->AddCut(VarManager::kEta, -0.8, 0.8);
    return cut;
  }

  if (!nameStr.compare("TightGlobalTrack")) {
    cut->AddCut(VarManager::kIsSPDfirst, 0.5, 1.5);
    cut->AddCut(VarManager::kIsITSrefit, 0.5, 1.5);
    cut->AddCut(VarManager::kIsTPCrefit, 0.5, 1.5);
    cut->AddCut(VarManager::kTPCchi2, 0.0, 4.0);
    cut->AddCut(VarManager::kITSchi2, 0.0, 5.0);
    cut->AddCut(VarManager::kTPCnclsCR, 80.0, 161.);
    cut->AddCut(VarManager::kITSncls, 3.5, 7.5);
    return cut;
  }

  if (!nameStr.compare("electronStandardQuality")) {
    cut->AddCut(VarManager::kIsSPDany, 0.5, 1.5);
    cut->AddCut(VarManager::kIsITSrefit, 0.5, 1.5);
    cut->AddCut(VarManager::kIsTPCrefit, 0.5, 1.5);
    cut->AddCut(VarManager::kTPCchi2, 0.0, 4.0);
    cut->AddCut(VarManager::kITSchi2, 0.1, 36.0);
    cut->AddCut(VarManager::kTPCncls, 100.0, 161.);
    return cut;
  }

  if (!nameStr.compare("standardPrimaryTrack")) {
    cut->AddCut(VarManager::kTrackDCAxy, -1.0, 1.0);
    cut->AddCut(VarManager::kTrackDCAz, -3.0, 3.0);
    return cut;
  }

  TF1* cutLow1 = new TF1("cutLow1", "pol1", 0., 10.);
  if (!nameStr.compare("electronPID1")) {
    cutLow1->SetParameters(130., -40.0);
    cut->AddCut(VarManager::kTPCsignal, 70., 100.);
    cut->AddCut(VarManager::kTPCsignal, cutLow1, 100.0, false, VarManager::kPin, 0.5, 3.0);
    return cut;
  }

  if (!nameStr.compare("electronPID1shiftUp")) {
    cut->AddCut(VarManager::kTPCsignal, 70. - 0.85, 100. - 0.85);
    cutLow1->SetParameters(130. - 0.85, -40.0);
    cut->AddCut(VarManager::kTPCsignal, cutLow1, 100.0 - 0.85, false, VarManager::kPin, 0.5, 3.0);
    return cut;
  }

  if (!nameStr.compare("electronPID1shiftDown")) {
    cut->AddCut(VarManager::kTPCsignal, 70.0 + 0.85, 100.0 + 0.85);
    cutLow1->SetParameters(130. + 0.85, -40.0);
    cut->AddCut(VarManager::kTPCsignal, cutLow1, 100.0 + 0.85, false, VarManager::kPin, 0.5, 3.0);
    return cut;
  }

  if (!nameStr.compare("electronPID1randomized")) {
    cutLow1->SetParameters(130., -40.0);
    cut->AddCut(VarManager::kTPCsignalRandomized, 70., 100.);
    cut->AddCut(VarManager::kTPCsignalRandomized, cutLow1, 100.0, false, VarManager::kPin, 0.5, 3.0);
    return cut;
  }

  if (!nameStr.compare("electronPID2")) {
    cutLow1->SetParameters(130., -40.0);
    cut->AddCut(VarManager::kTPCsignal, 73., 100.);
    cut->AddCut(VarManager::kTPCsignal, cutLow1, 100.0, false, VarManager::kPin, 0.5, 3.0);
    return cut;
  }

  if (!nameStr.compare("electronPID2randomized")) {
    cutLow1->SetParameters(130., -40.0);
    cut->AddCut(VarManager::kTPCsignalRandomized, 73., 100.);
    cut->AddCut(VarManager::kTPCsignalRandomized, cutLow1, 100.0, false, VarManager::kPin, 0.5, 3.0);
    return cut;
  }

  if (!nameStr.compare("electronPIDnsigma")) {
    cut->AddCut(VarManager::kTPCnSigmaEl, -3.0, 3.0);
    cut->AddCut(VarManager::kTPCnSigmaPr, 3.0, 3000.0);
    cut->AddCut(VarManager::kTPCnSigmaPi, 3.0, 3000.0);
    return cut;
  }

  if (!nameStr.compare("electronPIDnsigmaRandomized")) {
    cut->AddCut(VarManager::kTPCnSigmaElRandomized, -3.0, 3.0);
    cut->AddCut(VarManager::kTPCnSigmaPrRandomized, 3.0, 3000.0);
    cut->AddCut(VarManager::kTPCnSigmaPiRandomized, 3.0, 3000.0);
    return cut;
  }

  if (!nameStr.compare("electronPIDworseRes")) {
    cut->AddCut(VarManager::kTPCnSigmaEl, -3.0, 3.0);
    cut->AddCut(VarManager::kTPCnSigmaPr, 3.0 * 0.8, 3000.0); // emulates a 20% degradation in PID resolution
    cut->AddCut(VarManager::kTPCnSigmaPi, 3.0 * 0.8, 3000.0); //    proton and pion rejections are effectively relaxed by 20%
    return cut;
  }

  if (!nameStr.compare("electronPIDshift")) {
    cut->AddCut(VarManager::kTPCnSigmaEl, -3.0, 3.0);
    cut->AddCut(VarManager::kTPCnSigmaPr, 3.0 - 0.2, 3000.0);
    cut->AddCut(VarManager::kTPCnSigmaPi, 3.0 - 0.2, 3000.0);
    return cut;
  }

  if (!nameStr.compare("tpc_pion_rejection")) {
    TF1* f1maxPi = new TF1("f1maxPi", "[0]+[1]*x", 0, 10);
    f1maxPi->SetParameters(85, -50);
    cut->AddCut(VarManager::kTPCsignal, 70, f1maxPi, true, VarManager::kPin, 0.0, 0.4, false);
    return cut;
  }

  if (!nameStr.compare("tpc_kaon_rejection")) {
    TF1* f1minKa = new TF1("f1minKa", "[0]+[1]*x", 0, 10);
    f1minKa->SetParameters(220, -300);
    TF1* f1maxKa = new TF1("f1maxKa", "[0]+[1]*x", 0, 10);
    f1maxKa->SetParameters(182.5, -150);
    cut->AddCut(VarManager::kTPCsignal, f1minKa, f1maxKa, true, VarManager::kPin, 0.4, 0.8, false);
    return cut;
  }

  if (!nameStr.compare("tpc_proton_rejection")) {
    TF1* f1minPr = new TF1("f1minPr", "[0]+[1]*x", 0, 10);
    f1minPr->SetParameters(170, -100);
    TF1* f1maxPr = new TF1("f1maxPr", "[0]+[1]*x", 0, 10);
    f1maxPr->SetParameters(175, -75);
    cut->AddCut(VarManager::kTPCsignal, f1minPr, f1maxPr, true, VarManager::kPin, 0.8, 1.4, false);
    return cut;
  }

  if (!nameStr.compare("tpc_electron")) {
    cut->AddCut(VarManager::kTPCsignal, 70, 90, false, VarManager::kPin, 0.0, 1e+10, false);
    return cut;
  }

  if (!nameStr.compare("tof_electron")) {
    cut->AddCut(VarManager::kTOFbeta, 0.99, 1.01, false, VarManager::kPin, 0.0, 1e+10, false);
    return cut;
  }

  if (!nameStr.compare("muonQualityCuts")) {
    cut->AddCut(VarManager::kEta, -4.0, -2.5);
    cut->AddCut(VarManager::kMuonRAtAbsorberEnd, 17.6, 89.5);
    cut->AddCut(VarManager::kMuonPDca, 0.0, 594.0, false, VarManager::kMuonRAtAbsorberEnd, 17.6, 26.5);
    cut->AddCut(VarManager::kMuonPDca, 0.0, 324.0, false, VarManager::kMuonRAtAbsorberEnd, 26.5, 89.5);
    cut->AddCut(VarManager::kMuonChi2, 0.0, 1e6);
    return cut;
  }

  if (!nameStr.compare("pairNoCut")) {
    cut->AddCut(VarManager::kMass, 0.0, 1000.0);
    return cut;
  }

  if (!nameStr.compare("pairMassLow")) {
    cut->AddCut(VarManager::kMass, 2.5, 1000.0);
    return cut;
  }

  if (!nameStr.compare("pairJpsi")) {
    cut->AddCut(VarManager::kMass, 2.8, 3.3);
    return cut;
  }

  if (!nameStr.compare("pairPsi2S")) {
    cut->AddCut(VarManager::kMass, 3.4, 3.9);
    return cut;
  }

  if (!nameStr.compare("pairUpsilon")) {
    cut->AddCut(VarManager::kMass, 8.0, 11.0);
    return cut;
  }

  if (!nameStr.compare("pairPtLow1")) {
    cut->AddCut(VarManager::kPt, 2.0, 1000.0);
    return cut;
  }

  if (!nameStr.compare("pairPtLow2")) {
    cut->AddCut(VarManager::kPt, 5.0, 1000.0);
    return cut;
  }

  delete cut;
  return nullptr;
}
