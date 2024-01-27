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

#include "GlobalTracking/MatchITSTPCQC.h"
#include "ReconstructionDataFormats/TrackTPCITS.h"
#include "DataFormatsTPC/TrackTPC.h"
#include "Framework/InputSpec.h"
#include "ReconstructionDataFormats/TrackParametrization.h"
#include "DetectorsBase/Propagator.h"
#include "SimulationDataFormat/MCUtils.h"
#include <algorithm>
#include "TGraphAsymmErrors.h"
#include "GlobalTracking/TrackCuts.h"
#include <DetectorsBase/GRPGeomHelper.h>

using namespace o2::globaltracking;
using namespace o2::mcutils;
using MCTrack = o2::MCTrackT<float>;

MatchITSTPCQC::~MatchITSTPCQC()
{

  deleteHistograms();
}

//_______________________________________________________

void MatchITSTPCQC::deleteHistograms()
{

  for (int i = 0; i < matchType::SIZE; ++i) {
    // Pt
    delete mPtNum[i];
    delete mPtDen[i];
    delete mFractionITSTPCmatch[i];
    delete mPtNum_noEta0[i];
    delete mPtDen_noEta0[i];
    delete mFractionITSTPCmatch_noEta0[i];
    delete mPtPhysPrimNum[i];
    delete mPtPhysPrimDen[i];
    delete mFractionITSTPCmatchPhysPrim[i];

    // Phi
    delete mPhiNum[i];
    delete mPhiDen[i];
    delete mFractionITSTPCmatchPhi[i];
    delete mPhiPhysPrimNum[i];
    delete mPhiPhysPrimDen[i];
    delete mFractionITSTPCmatchPhiPhysPrim[i];
    delete mPhiVsPtNum[i];
    delete mPhiVsPtDen[i];
    delete mFractionITSTPCmatchPhiVsPt[i];

    // Eta
    delete mEtaNum[i];
    delete mEtaDen[i];
    delete mFractionITSTPCmatchEta[i];
    delete mEtaPhysPrimNum[i];
    delete mEtaPhysPrimDen[i];
    delete mFractionITSTPCmatchEtaPhysPrim[i];
    delete mEtaVsPtNum[i];
    delete mEtaVsPtDen[i];
    delete mFractionITSTPCmatchEtaVsPt[i];

    // Clusters
    delete mClsVsPtNum[i];
    delete mClsVsPtDen[i];
    delete mFractionITSTPCmatchClsVsPt[i];

    // Chi2
    delete mChi2VsPtNum[i];
    delete mChi2VsPtDen[i];
    delete mFractionITSTPCmatchChi2VsPt[i];

    if (mUseTrkPID) { // Vs Tracking PID hypothesis
      for (int j = 0; j < o2::track::PID::NIDs; ++j) {
        // Pt
        delete mPtNumVsTrkPID[i][j];
        delete mPtDenVsTrkPID[i][j];
        delete mFractionITSTPCmatchPtVsTrkPID[i][j];
        // Phi
        delete mPhiNumVsTrkPID[i][j];
        delete mPhiDenVsTrkPID[i][j];
        delete mFractionITSTPCmatchPhiVsTrkPID[i][j];
        // Eta
        delete mEtaNumVsTrkPID[i][j];
        delete mEtaDenVsTrkPID[i][j];
        delete mFractionITSTPCmatchEtaVsTrkPID[i][j];
      }
    }

    // 1/Pt
    delete m1OverPtNum[i];
    delete m1OverPtDen[i];
    delete mFractionITSTPCmatch1OverPt[i];
    delete m1OverPtPhysPrimNum[i];
    delete m1OverPtPhysPrimDen[i];
    delete mFractionITSTPCmatchPhysPrim1OverPt[i];
  }

  // Residuals
  delete mResidualPt;
  delete mResidualPhi;
  delete mResidualEta;
  // Others
  delete mChi2Matching;
  delete mChi2Refit;
  delete mTimeResVsPt;
  delete mDCAr;
  delete mDCArVsPtNum;
  delete mDCArVsPtDen;
  delete mFractionITSTPCmatchDCArVsPt;
}

//__________________________________________________________

void MatchITSTPCQC::reset()
{
  for (int i = 0; i < matchType::SIZE; ++i) {
    // Pt
    mPtNum[i]->Reset();
    mPtDen[i]->Reset();
    mPtNum_noEta0[i]->Reset();
    mPtDen_noEta0[i]->Reset();
    mPtPhysPrimNum[i]->Reset();
    mPtPhysPrimDen[i]->Reset();

    // Phi
    mPhiNum[i]->Reset();
    mPhiDen[i]->Reset();
    mPhiPhysPrimNum[i]->Reset();
    mPhiPhysPrimDen[i]->Reset();
    mPhiVsPtNum[i]->Reset();
    mPhiVsPtDen[i]->Reset();

    // Eta
    mEtaNum[i]->Reset();
    mEtaDen[i]->Reset();
    mEtaPhysPrimNum[i]->Reset();
    mEtaPhysPrimDen[i]->Reset();
    mEtaVsPtNum[i]->Reset();
    mEtaVsPtDen[i]->Reset();

    // Clusters
    mClsVsPtNum[i]->Reset();
    mClsVsPtDen[i]->Reset();

    // Chi2
    mChi2VsPtNum[i]->Reset();
    mChi2VsPtDen[i]->Reset();

    if (mUseTrkPID) { // Vs Tracking PID hypothesis
      for (int j = 0; j < o2::track::PID::NIDs; ++j) {
        // Pt
        mPtNumVsTrkPID[i][j]->Reset();
        mPtDenVsTrkPID[i][j]->Reset();
        // Phi
        mPhiNumVsTrkPID[i][j]->Reset();
        mPhiDenVsTrkPID[i][j]->Reset();
        // Eta
        mEtaNumVsTrkPID[i][j]->Reset();
        mEtaDenVsTrkPID[i][j]->Reset();
      }
    }
    // 1/Pt
    m1OverPtNum[i]->Reset();
    m1OverPtDen[i]->Reset();
    m1OverPtPhysPrimNum[i]->Reset();
    m1OverPtPhysPrimDen[i]->Reset();
  }
  // Residuals
  mResidualPt->Reset();
  mResidualPhi->Reset();
  mResidualEta->Reset();
  // Others
  mChi2Matching->Reset();
  mChi2Refit->Reset();
  mTimeResVsPt->Reset();
  mDCAr->Reset();
  mDCArVsPtNum->Reset();
  mDCArVsPtDen->Reset();
}

//__________________________________________________________
bool MatchITSTPCQC::init()
{

  std::array<std::string, 2> title{"TPC", "ITS"};
  std::array<std::string, 2> etaSel{"", ", |eta| < 0.9"};
  std::array<int, 2> maxNCls{156, 7};
  for (int i = 0; i < matchType::SIZE; ++i) {
    // Pt
    mPtNum[i] = new TH1D(Form("mPtNum_%s", title[i].c_str()), Form("Pt distribution of ITSTPC matched tracks, wrt %s tracks %s; Pt [GeV/c]; dNdPt", title[i].c_str(), etaSel[i].c_str()), 100, 0.f, 20.f);
    mPtDen[i] = new TH1D(Form("mPtDen_%s", title[i].c_str()), Form("Pt distribution of %s tracks %s; Pt [GeV/c]; dNdPt", title[i].c_str(), etaSel[i].c_str()), 100, 0.f, 20.f);
    mFractionITSTPCmatch[i] = new TEfficiency(Form("mFractionITSTPCmatch_%s", title[i].c_str()), Form("Fraction of ITSTPC matched tracks wrt %s tracks vs Pt %s; Pt [GeV/c]; Eff", title[i].c_str(), etaSel[i].c_str()), 100, 0.f, 20.f);
    mPtNum_noEta0[i] = new TH1D(Form("mPtNum_noEta0_%s", title[i].c_str()), Form("Pt distribution of ITSTPC matched tracks without |eta| < 0.05, wrt %s tracks %s; Pt [GeV/c]; dNdPt", title[i].c_str(), etaSel[i].c_str()), 100, 0.f, 20.f);
    mPtDen_noEta0[i] = new TH1D(Form("mPtDen_noEta0_%s", title[i].c_str()), Form("Pt distribution of %s tracks without |eta| < 0.05 %s; Pt [GeV/c]; dNdPt", title[i].c_str(), etaSel[i].c_str()), 100, 0.f, 20.f);
    mFractionITSTPCmatch_noEta0[i] = new TEfficiency(Form("mFractionITSTPCmatch_noEta0_%s", title[i].c_str()), Form("Fraction of ITSTPC matched tracks wrt %s tracks vs Pt without |eta| < 0.05 %s; Pt [GeV/c]; Eff", title[i].c_str(), etaSel[i].c_str()), 100, 0.f, 20.f);
    // Phi
    mPhiNum[i] = new TH1F(Form("mPhiNum_%s", title[i].c_str()), Form("Phi distribution of ITSTPC matched tracks, wrt %s tracks %s; Phi [rad]; dNdPhi", title[i].c_str(), etaSel[i].c_str()), 100, 0.f, 2 * TMath::Pi());
    mPhiDen[i] = new TH1F(Form("mPhiDen_%s", title[i].c_str()), Form("Phi distribution of %s tracks %s; Phi [rad]; dNdPhi", title[i].c_str(), etaSel[i].c_str()), 100, 0.f, 2 * TMath::Pi());
    mFractionITSTPCmatchPhi[i] = new TEfficiency(Form("mFractionITSTPCmatchPhi_%s", title[i].c_str()), Form("Fraction of ITSTPC matched tracks vs Phi wrt %s tracks %s; Phi [rad]; Eff", title[i].c_str(), etaSel[i].c_str()), 100, 0.f, 2 * TMath::Pi());
    mPhiVsPtNum[i] = new TH2F(Form("mPhiVsPtNum_%s", title[i].c_str()), Form("Phi vs Pt distribution of ITSTPC matched tracks wrt %s %s; #it{p}_{T} [GeV#it{c}]; Phi [rad]; dNdPhi", title[i].c_str(), etaSel[i].c_str()), 100, 0.f, 20.f, 100, 0.f, 2 * TMath::Pi());
    mPhiVsPtDen[i] = new TH2F(Form("mPhiVsPtDen_%s", title[i].c_str()), Form("Phi vs Pt distribution of %s tracks %s; #it{p}_{T} [GeV#it{c}]; Phi [rad]; dNdPhi", title[i].c_str(), etaSel[i].c_str()), 100, 0.f, 20.f, 100, 0.f, 2 * TMath::Pi());
    mFractionITSTPCmatchPhiVsPt[i] = new TEfficiency(Form("mFractionITSTPCmatchPhiVsPt_%s", title[i].c_str()), Form("Fraction of ITSTPC matched tracks wrt %s tracks %s, Phi vs Pt; #it{p}_{T} [GeV#it{c}]; Phi [rad]; Eff", title[i].c_str(), etaSel[i].c_str()), 100, 0.f, 20.f, 100, 0.f, 2 * TMath::Pi());
    // Eta
    mEtaNum[i] = new TH1F(Form("mEtaNum_%s", title[i].c_str()), Form("Eta distribution of ITSTPC matched tracks, wrt %s tracks; Eta; dNdEta", title[i].c_str()), 100, -2.f, 2.f);
    mEtaDen[i] = new TH1F(Form("mEtaDen_%s", title[i].c_str()), Form("Eta distribution of %s tracks; Eta; dNdEta", title[i].c_str()), 100, -2.f, 2.f);
    mFractionITSTPCmatchEta[i] = new TEfficiency(Form("mFractionITSTPCmatchEta_%s", title[i].c_str()), Form("Fraction of ITSTPC matched tracks , wrt %s tracks, vs Eta; Eta; Eff", title[i].c_str()), 100, -2.f, 2.f);
    mEtaVsPtNum[i] = new TH2F(Form("mEtaVsPtNum_%s", title[i].c_str()), Form("Eta vs Pt distribution of ITSTPC matched tracks, wrt %s tracks; #it{p}_{T} [GeV#it{c}]; Eta", title[i].c_str()), 100, 0.f, 20.f, 100, -2.f, 2.f);
    mEtaVsPtDen[i] = new TH2F(Form("mEtaVsPtDen_%s", title[i].c_str()), Form("Eta vs Pt distribution of %s tracks; #it{p}_{T} [GeV#it{c}]; Eta", title[i].c_str()), 100, 0.f, 20.f, 100, -2.f, 2.f);
    mFractionITSTPCmatchEtaVsPt[i] = new TEfficiency(Form("mFractionITSTPCmatchEtaVsPt_%s", title[i].c_str()), Form("Fraction of ITSTPC matched tracks, wrt %s tracks, Eta vs Pt; #it{p}_{T} [GeV#it{c}]; Eta; Eff", title[i].c_str()), 100, 0.f, 20.f, 100, -2.f, 2.f);
    // Clusters
    mClsVsPtNum[i] = new TH2F(Form("mClsVsPtNum_%s", title[i].c_str()), Form("#Clusters vs Pt distribution of ITSTPC matched tracks, wrt %s tracks; #it{p}_{T} [GeV#it{c}]; #Clusters", title[i].c_str()), 100, 0.f, 20.f, maxNCls[i], 0, maxNCls[i]);
    mClsVsPtDen[i] = new TH2F(Form("mClsVsPtDen_%s", title[i].c_str()), Form("#Clusters vs Pt distribution of %s tracks; #it{p}_{T} [GeV#it{c}]; #Clusters", title[i].c_str()), 100, 0.f, 20.f, maxNCls[i], 0, maxNCls[i]);
    mFractionITSTPCmatchClsVsPt[i] = new TEfficiency(Form("mFractionITSTPCmatchClsVsPt_%s", title[i].c_str()), Form("Fraction of ITSTPC matched tracks, wrt %s tracks, #Clusters vs Pt; #it{p}_{T} [GeV#it{c}]; #Clusters; Eff", title[i].c_str()), 100, 0.f, 20.f, maxNCls[i], 0, maxNCls[i]);
    // Chi2
    mChi2VsPtNum[i] = new TH2F(Form("mChi2VsPtNum_%s", title[i].c_str()), Form("Chi2 vs Pt distribution of ITSTPC matched tracks, wrt %s tracks; #it{p}_{T} [GeV#it{c}]; Chi2", title[i].c_str()), 100, 0.f, 20.f, 200, 0, 300);
    mChi2VsPtDen[i] = new TH2F(Form("mChi2VsPtDen_%s", title[i].c_str()), Form("Chi2 vs Pt distribution of %s tracks; #it{p}_{T} [GeV#it{c}]; Chi2", title[i].c_str()), 100, 0.f, 20.f, 200, 0, 300);
    mFractionITSTPCmatchChi2VsPt[i] = new TEfficiency(Form("mFractionITSTPCmatchChi2VsPt_%s", title[i].c_str()), Form("Fraction of ITSTPC matched tracks, wrt %s tracks, Chi2 vs Pt; #it{p}_{T} [GeV#it{c}]; Chi2; Eff", title[i].c_str()), 100, 0.f, 20.f, 200, 0, 300);
    if (mUseTrkPID) { // Vs Tracking PID hypothesis
      for (int j = 0; j < o2::track::PID::NIDs; ++j) {
        // Pt
        mPtNumVsTrkPID[i][j] = new TH1D(Form("mPtNumVsTrkPID_%s_PID%i", title[i].c_str(), j), Form("Pt distribution of ITSTPC matched tracks, wrt %s tracks %s, TrkPID %i; Pt [GeV/c]; dNdPt", title[i].c_str(), etaSel[i].c_str(), j), 100, 0.f, 20.f);
        mPtDenVsTrkPID[i][j] = new TH1D(Form("mPtDenVsTrkPID_%s_PID%i", title[i].c_str(), j), Form("Pt distribution of %s tracks %s, TrkPID %i; Pt [GeV/c]; dNdPt", title[i].c_str(), etaSel[i].c_str(), j), 100, 0.f, 20.f);
        mFractionITSTPCmatchPtVsTrkPID[i][j] = new TEfficiency(Form("mFractionITSTPCmatchPtVsTrkPID_%s_PID%i", title[i].c_str(), j), Form("Fraction of ITSTPC matched tracks wrt %s tracks vs Pt %s, TrkPID %i; Pt [GeV/c]; Eff", title[i].c_str(), etaSel[i].c_str(), j), 100, 0.f, 20.f);
        // Phi
        mPhiNumVsTrkPID[i][j] = new TH1D(Form("mPhiNumVsTrkPID_%s_PID%i", title[i].c_str(), j), Form("Phi distribution of ITSTPC matched tracks, wrt %s tracks %s, TrkPID %i; Phi [rad]; dNdPhi", title[i].c_str(), etaSel[i].c_str(), j), 100, 0.f, 2 * TMath::Pi());
        mPhiDenVsTrkPID[i][j] = new TH1D(Form("mPhiDenVsTrkPID_%s_PID%i", title[i].c_str(), j), Form("Phi distribution of %s tracks %s, TrkPID %i; Phi [rad]; dNdPhi", title[i].c_str(), etaSel[i].c_str(), j), 100, 0.f, 2 * TMath::Pi());
        mFractionITSTPCmatchPhiVsTrkPID[i][j] = new TEfficiency(Form("mFractionITSTPCmatchPhiVsTrkPID_%s_PID%i", title[i].c_str(), j), Form("Fraction of ITSTPC matched tracks wrt %s tracks vs Phi %s, TrkPID %i; Phi [rad]; Eff", title[i].c_str(), etaSel[i].c_str(), j), 100, 0.f, 2 * TMath::Pi());
        // Eta
        mEtaNumVsTrkPID[i][j] = new TH1D(Form("mEtaNumVsTrkPID_%s_PID%i", title[i].c_str(), j), Form("Eta distribution of ITSTPC matched tracks, wrt %s tracks %s, TrkPID %i; Eta; dNdEta", title[i].c_str(), etaSel[i].c_str(), j), 100, -2.f, 2.f);
        mEtaDenVsTrkPID[i][j] = new TH1D(Form("mEtaDenVsTrkPID_%s_PID%i", title[i].c_str(), j), Form("Eta distribution of %s tracks %s, TrkPID %i; Eta; dNdEta", title[i].c_str(), etaSel[i].c_str(), j), 100, -2.f, 2.f);
        mFractionITSTPCmatchEtaVsTrkPID[i][j] = new TEfficiency(Form("mFractionITSTPCmatchEtaVsTrkPID_%s_PID%i", title[i].c_str(), j), Form("Fraction of ITSTPC matched tracks wrt %s tracks vs Eta %s, TrkPID %i; Eta; Eff", title[i].c_str(), etaSel[i].c_str(), j), 100, -2.f, 2.f);
      }
    }

    // 1/pt
    m1OverPtNum[i] = new TH1D(Form("m1OverPtNum_%s", title[i].c_str()), Form("1/Pt distribution of matched tracks, wrt %s tracks %s; 1/Pt [c/GeV]; dNdPt", title[i].c_str(), etaSel[i].c_str()), 100, -20.f, 20.f);
    m1OverPtDen[i] = new TH1D(Form("m1OverPtDen_%s", title[i].c_str()), Form("1/Pt distribution of %s tracks %s; 1/Pt [c/GeV]; dNdPt", title[i].c_str(), etaSel[i].c_str()), 100, -20.f, 20.f);
    mFractionITSTPCmatch1OverPt[i] = new TEfficiency(Form("mFractionITSTPCmatch1OverPt_%s", title[i].c_str()), Form("Fraction of ITSTPC matched tracks vs 1/Pt, wrt %s tracks %s; 1/Pt [c/GeV]; Eff", title[i].c_str(), etaSel[i].c_str()), 100, -20.f, 20.f);

    // These will be empty in case of no MC info...
    mPhiPhysPrimNum[i] = new TH1F(Form("mPhiPhysPrimNum_%s", title[i].c_str()), Form("Phi distribution of matched tracks (physical primary), wrt %s tracks %s; Phi [rad]; dNdPhi", title[i].c_str(), etaSel[i].c_str()), 100, 0.f, 2 * TMath::Pi());
    mPhiPhysPrimDen[i] = new TH1F(Form("mPhiPhysPrimDen_%s", title[i].c_str()), Form("Phi distribution of %s tracks (physical primary) %s; Phi [rad]; dNdPhi", title[i].c_str(), etaSel[i].c_str()), 100, 0.f, 2 * TMath::Pi());
    mFractionITSTPCmatchPhiPhysPrim[i] = new TEfficiency(Form("mFractionITSTPCmatchPhiPhysPrim_%s", title[i].c_str()), Form("Fraction of ITSTPC matched tracks vs Phi (physical primary), wrt %s tracks %s; Phi [rad]; Eff", title[i].c_str(), etaSel[i].c_str()), 100, 0.f, 2 * TMath::Pi());
    mEtaPhysPrimNum[i] = new TH1F(Form("mEtaPhysPrimNum_%s", title[i].c_str()), Form("Eta distribution of matched tracks (physical primary), wrt %s tracks; Eta; dNdEta", title[i].c_str()), 100, -2.f, 2.f);
    mEtaPhysPrimDen[i] = new TH1F(Form("mEtaPhysPrimDen_%s", title[i].c_str()), Form("Eta distribution of %s tracks (physical primary); Eta; dNdEta", title[i].c_str()), 100, -2.f, 2.f);
    mFractionITSTPCmatchEtaPhysPrim[i] = new TEfficiency(Form("mFractionITSTPCmatchEtaPhysPrim_%s", title[i].c_str()), Form("Fraction of ITSTPC matched tracks vs Eta (physical primary), wrt %s tracks; Eta; Eff", title[i].c_str()), 100, -2.f, 2.f);
  }

  mResidualPt = new TH2F("mResidualPt", "Residuals of ITS-TPC matching in #it{p}_{T}; #it{p}_{T}^{ITS-TPC} [GeV/c]; #it{p}_{T}^{ITS-TPC} - #it{p}_{T}^{TPC} [GeV/c]", 100, 0.f, 20.f, 100, -1.f, 1.f);
  mResidualPhi = new TH2F("mResidualPhi", "Residuals of ITS-TPC matching in #it{#phi}; #it{#phi}^{ITS-TPC} [rad]; #it{#phi}^{ITS-TPC} - #it{#phi}^{TPC} [rad]", 100, 0.f, 2 * TMath::Pi(), 100, -1.f, 1.f);
  mResidualEta = new TH2F("mResidualEta", "Residuals of ITS-TPC matching in #it{#eta}; #it{#eta}^{ITS-TPC}; #it{#eta}^{ITS-TPC} - #it{#eta}^{TPC}", 100, -2.f, 2.f, 100, -1.f, 1.f);
  mChi2Matching = new TH1F("mChi2Matching", "Chi2 of matching; chi2", 200, 0, 300);
  mChi2Refit = new TH1F("mChi2Refit", "Chi2 of refit; chi2", 200, 0, 300);
  mDCAr = new TH1F("mDCAr", "DCA of TPC tracks; DCAr", 200, -100, 100);
  mDCArVsPtNum = new TH2F("mDCArVsPtNum", "DCA of TPC tracks Vs Pt Num; #it{p}_{T} [GeV/c]; DCAr", 100, 0, 20., 200, -30, 30);
  mDCArVsPtDen = new TH2F("mDCArVsPtDen", "DCA of TPC tracks Vs Pt Den; #it{p}_{T} [GeV/c]; DCAr", 100, 0, 20., 200, -30, 30);
  mFractionITSTPCmatchDCArVsPt = new TEfficiency("mFractionITSTPCmatchDCArVsPt", "Fraction of ITSTPC matched tracks wrt TPC vs DCAr; #it{p}_{T} [GeV#it{c}]; DCAr; Eff", 100, 0, 20., 200, -30, 30);

  // log binning for pT
  const Int_t nbinsPt = 100;
  const Double_t xminPt = 0.01;
  const Double_t xmaxPt = 20;
  Double_t* xbinsPt = new Double_t[nbinsPt + 1];
  Double_t xlogminPt = TMath::Log10(xminPt);
  Double_t xlogmaxPt = TMath::Log10(xmaxPt);
  Double_t dlogxPt = (xlogmaxPt - xlogminPt) / nbinsPt;
  for (int i = 0; i <= nbinsPt; i++) {
    Double_t xlogPt = xlogminPt + i * dlogxPt;
    xbinsPt[i] = TMath::Exp(TMath::Log(10) * xlogPt);
  }
  mTimeResVsPt = new TH2F("mTimeResVsPt", "Time resolution vs Pt; Pt [GeV/c]; time res [us]", nbinsPt, xbinsPt, 100, 0.f, 2.f);

  for (int i = 0; i < matchType::SIZE; ++i) {
    mPtPhysPrimNum[i] = new TH1F(Form("mPtPhysPrimNum_%s", title[i].c_str()), Form("Pt distribution of matched tracks (physical primary), wrt %s tracks %s; Pt [GeV/c]; dNdPt", title[i].c_str(), etaSel[i].c_str()), nbinsPt, xbinsPt);
    mPtPhysPrimDen[i] = new TH1F(Form("mPtPhysPrimDen_%s", title[i].c_str()), Form("Pt distribution of %s tracks (physical primary) %s; Pt [GeV/c]; dNdPt", title[i].c_str(), etaSel[i].c_str()), nbinsPt, xbinsPt);
    mFractionITSTPCmatchPhysPrim[i] = new TEfficiency(Form("mFractionITSTPCmatchPhysPrim_%s", title[i].c_str()), Form("Fraction of ITSTPC matched tracks vs Pt (physical primary), wrt %s tracks %s; Pt [GeV/c]; Eff", title[i].c_str(), etaSel[i].c_str()), nbinsPt, xbinsPt);
    m1OverPtPhysPrimNum[i] = new TH1D(Form("m1OverPtPhysPrimNum_%s", title[i].c_str()), Form("1/Pt distribution of matched tracks (physical primary), wrt %s tracks %s; 1/Pt [c/GeV]; dNd1/Pt", title[i].c_str(), etaSel[i].c_str()), 100, -20.f, 20.f);
    m1OverPtPhysPrimDen[i] = new TH1D(Form("m1OverPtPhysPrimDen_%s", title[i].c_str()), Form("1/PtPt distribution of %s tracks (physical primary) %s; 1/Pt [c/GeV]; dNd1/Pt", title[i].c_str(), etaSel[i].c_str()), 100, -20.f, 20.f);
    mFractionITSTPCmatchPhysPrim1OverPt[i] = new TEfficiency(Form("mFractionITSTPCmatchPhysPrim1OverPt_%s", title[i].c_str()), Form("Fraction of ITSTPC matched tracks vs 1/Pt (physical primary), wrt %s tracks %s; 1/Pt [c/GeV]; Eff", title[i].c_str(), etaSel[i].c_str()), 100, -20.f, 20.f);

    // some extra settings
    // Pt
    mPtNum[i]->Sumw2();
    mPtDen[i]->Sumw2();
    mPtNum_noEta0[i]->Sumw2();
    mPtDen_noEta0[i]->Sumw2();
    mPtPhysPrimNum[i]->Sumw2();
    mPtPhysPrimDen[i]->Sumw2();

    // Phi
    mPhiNum[i]->Sumw2();
    mPhiDen[i]->Sumw2();
    mPhiVsPtNum[i]->Sumw2();
    mPhiVsPtDen[i]->Sumw2();
    mPhiPhysPrimNum[i]->Sumw2();
    mPhiPhysPrimDen[i]->Sumw2();

    // Eta
    mEtaNum[i]->Sumw2();
    mEtaDen[i]->Sumw2();
    mEtaPhysPrimNum[i]->Sumw2();
    mEtaPhysPrimDen[i]->Sumw2();
    mEtaVsPtNum[i]->Sumw2();
    mEtaVsPtDen[i]->Sumw2();

    // Clusters
    mClsVsPtNum[i]->Sumw2();
    mClsVsPtDen[i]->Sumw2();

    // Chi2
    mChi2VsPtNum[i]->Sumw2();
    mChi2VsPtDen[i]->Sumw2();

    if (mUseTrkPID) { // Vs Tracking PID hypothesis
      for (int j = 0; j < o2::track::PID::NIDs; ++j) {
        // Pt
        mPtNumVsTrkPID[i][j]->Sumw2();
        mPtDenVsTrkPID[i][j]->Sumw2();
        // Phi
        mPhiNumVsTrkPID[i][j]->Sumw2();
        mPhiDenVsTrkPID[i][j]->Sumw2();
        // Eta
        mEtaNumVsTrkPID[i][j]->Sumw2();
        mEtaDenVsTrkPID[i][j]->Sumw2();
      }
    }

    m1OverPtNum[i]->Sumw2();
    m1OverPtDen[i]->Sumw2();
    m1OverPtPhysPrimNum[i]->Sumw2();
    m1OverPtPhysPrimDen[i]->Sumw2();

    mPtNum_noEta0[i]->SetOption("logy");
    mPtDen_noEta0[i]->SetOption("logy");
    mPtNum[i]->SetOption("logy");
    mPtDen[i]->SetOption("logy");

    mPtNum[i]->GetYaxis()->SetTitleOffset(1.4);
    mPtDen[i]->GetYaxis()->SetTitleOffset(1.4);
    mPtNum_noEta0[i]->GetYaxis()->SetTitleOffset(1.4);
    mPtDen_noEta0[i]->GetYaxis()->SetTitleOffset(1.4);
    mEtaNum[i]->GetYaxis()->SetTitleOffset(1.4);
    mEtaDen[i]->GetYaxis()->SetTitleOffset(1.4);
  }

  mChi2Matching->SetOption("logy");
  mChi2Refit->SetOption("logy");
  mTimeResVsPt->SetOption("colz logz logy logx");

  mChi2Matching->GetYaxis()->SetTitleOffset(1.4);
  mChi2Refit->GetYaxis()->SetTitleOffset(1.4);
  mTimeResVsPt->GetYaxis()->SetTitleOffset(1.4);

  mDCArVsPtNum->Sumw2();
  mDCArVsPtDen->Sumw2();

  if (mUseMC) {
    mcReader.initFromDigitContext("collisioncontext.root");
  }

  return true;
}

//__________________________________________________________

void MatchITSTPCQC::initDataRequest()
{

  // initialize data request, if it was not already done

  mSrc &= mAllowedSources;

  if (mSrc[GID::Source::ITSTPC] == 0 || mSrc[GID::Source::TPC] == 0 || mSrc[GID::Source::ITS] == 0) {
    LOG(fatal) << "We cannot do ITSTPC QC, some sources are missing, check sources in " << mSrc;
  }

  mDataRequest = std::make_shared<o2::globaltracking::DataRequest>();
  mDataRequest->requestTracks(mSrc, mUseMC);
}

//__________________________________________________________

void MatchITSTPCQC::run(o2::framework::ProcessingContext& ctx)
{

  // Getting the B field
  mBz = o2::base::Propagator::Instance()->getNominalBz();

  static int evCount = 0;
  mRecoCont.collectData(ctx, *mDataRequest.get());
  mTPCTracks = mRecoCont.getTPCTracks();
  mITSTracks = mRecoCont.getITSTracks();
  mITSTPCTracks = mRecoCont.getTPCITSTracks();

  LOG(debug) << "****** Number of found ITSTPC tracks = " << mITSTPCTracks.size();
  LOG(debug) << "****** Number of found TPC    tracks = " << mTPCTracks.size();
  LOG(debug) << "****** Number of found ITS    tracks = " << mITSTracks.size();

  // cache selection for TPC and ITS tracks
  std::vector<bool> isTPCTrackSelectedEntry(mTPCTracks.size(), false);
  std::vector<bool> isITSTrackSelectedEntry(mITSTracks.size(), false);
  TrackCuts cuts;
  // ITS track
  cuts.setMinPtITSCut(mPtITSCut);
  cuts.setEtaITSCut(mEtaITSCut);
  cuts.setMinNClustersITS(mMinNClustersITS);
  cuts.setMaxChi2PerClusterITS(mMaxChi2PerClusterITS);
  for (auto it = mRequiredITSHits.begin(); it != mRequiredITSHits.end(); it++) {
    cuts.setRequireHitsInITSLayers((*it).first, (*it).second);
  }
  // TPC track
  cuts.setMinPtTPCCut(mPtTPCCut);
  cuts.setEtaTPCCut(mEtaTPCCut);
  cuts.setMinNTPCClustersCut(mNTPCClustersCut);
  cuts.setMaxDCATPCCut(mDCATPCCut);
  cuts.setMaxDCATPCCutY(mDCATPCCutY);
  // ITS-TPC track kinematics
  cuts.setMinPtCut(mPtCut);
  cuts.setMaxPtCut(mPtMaxCut);
  cuts.setEtaCut(-mEtaCut, mEtaCut);

  for (size_t itrk = 0; itrk < mTPCTracks.size(); ++itrk) {
    auto const& trkTpc = mTPCTracks[itrk];
    // if (selectTrack(trkTpc)) {
    //   isTPCTrackSelectedEntry[itrk] = true;
    // }
    o2::dataformats::GlobalTrackID id(itrk, GID::TPC);
    if (cuts.isSelected(id, mRecoCont)) {
      // NB: same cuts for numerator and denominator tracks of ITS-TPC matching
      // To change cuts only for numerator, something like o2::dataformats::GlobalTrackID id(itrk, GID::ITSTPC) is necessary
      isTPCTrackSelectedEntry[itrk] = true;
    }
  }

  for (size_t itrk = 0; itrk < mITSTracks.size(); ++itrk) {
    auto const& trkIts = mITSTracks[itrk];
    o2::dataformats::GlobalTrackID id(itrk, GID::ITS);
    if (cuts.isSelected(id, mRecoCont)) {
      // NB: same cuts for numerator and denominator tracks of ITS-TPC matching
      // To change cuts only for numerator, something like o2::dataformats::GlobalTrackID id(itrk, GID::ITSTPC) is necessary
      isITSTrackSelectedEntry[itrk] = true;
    }
  }

  // numerator + eta, chi2...
  if (mUseMC) {
    for (int i = 0; i < matchType::SIZE; ++i) {
      mMapLabels[i].clear();
    }
    for (int itrk = 0; itrk < static_cast<int>(mITSTPCTracks.size()); ++itrk) {
      auto const& trk = mITSTPCTracks[itrk];
      auto idxTrkTpc = trk.getRefTPC().getIndex();
      if (isTPCTrackSelectedEntry[idxTrkTpc] == true) {
        auto lbl = mRecoCont.getTrackMCLabel({(unsigned int)(itrk), GID::Source::ITSTPC});
        if (!lbl.isValid()) {
          continue;
        }
        if (mMapLabels[matchType::TPC].find(lbl) == mMapLabels[matchType::TPC].end()) {
          int source = lbl.getSourceID();
          int event = lbl.getEventID();
          const std::vector<o2::MCTrack>& pcontainer = mcReader.getTracks(source, event);
          const o2::MCTrack& p = pcontainer[lbl.getTrackID()];
          if (MCTrackNavigator::isPhysicalPrimary(p, pcontainer)) {
            mMapLabels[matchType::TPC].insert({lbl, {itrk, true}});
          } else {
            mMapLabels[matchType::TPC].insert({lbl, {itrk, false}});
          }
        } else {
          // winner (if more tracks have the same label) has the highest pt
          if (mITSTPCTracks[mMapLabels[matchType::TPC].at(lbl).mIdx].getPt() < trk.getPt()) {
            mMapLabels[matchType::TPC].at(lbl).mIdx = itrk;
          }
        }
      }
      auto idxTrkIts = trk.getRefITS().getIndex();
      if (isITSTrackSelectedEntry[idxTrkIts] == true) {
        auto lbl = mRecoCont.getTrackMCLabel({(unsigned int)(itrk), GID::Source::ITSTPC});
        if (!lbl.isValid()) {
          continue;
        }
        if (mMapLabels[matchType::ITS].find(lbl) == mMapLabels[matchType::ITS].end()) {
          int source = lbl.getSourceID();
          int event = lbl.getEventID();
          const std::vector<o2::MCTrack>& pcontainer = mcReader.getTracks(source, event);
          const o2::MCTrack& p = pcontainer[lbl.getTrackID()];
          if (MCTrackNavigator::isPhysicalPrimary(p, pcontainer)) {
            mMapLabels[matchType::ITS].insert({lbl, {itrk, true}});
          } else {
            mMapLabels[matchType::ITS].insert({lbl, {itrk, false}});
          }
        } else {
          // winner (if more tracks have the same label) has the highest pt
          if (mITSTPCTracks[mMapLabels[matchType::ITS].at(lbl).mIdx].getPt() < trk.getPt()) {
            mMapLabels[matchType::ITS].at(lbl).mIdx = itrk;
          }
        }
      }
    }
    LOG(info) << "number of entries in map for nominator (without duplicates) = " << mMapLabels.size();
    // now we use only the tracks in the map to fill the histograms (--> tracks have passed the
    // track selection and there are no duplicated tracks wrt the same MC label)
    for (int i = 0; i < matchType::SIZE; ++i) {
      for (auto const& el : mMapLabels[i]) {
        auto const& trk = mITSTPCTracks[el.second.mIdx];
        o2::track::TrackParCov trkDen;
        bool isEtaITSOk = true;
        if (i == matchType::TPC) {
          trkDen = mTPCTracks[trk.getRefTPC()];
        } else {
          trkDen = mITSTracks[trk.getRefITS()];
          if (std::abs(trkDen.getEta()) > 0.9) {
            // ITS track outside |eta | < 0.9, we don't fill pt, nor phi , nor phi vs pt histos
            isEtaITSOk = false;
          }
        }
        if (isEtaITSOk) {
          mPtNum[i]->Fill(trkDen.getPt());
          if (std::abs(trkDen.getEta()) > 0.05) {
            mPtNum_noEta0[i]->Fill(trkDen.getPt());
          }
          mPhiNum[i]->Fill(trkDen.getPhi());
          mPhiVsPtNum[i]->Fill(trkDen.getPt(), trkDen.getPhi());
          m1OverPtNum[i]->Fill(trkDen.getSign() * trkDen.getPtInv());
          // we fill also the denominator
          mPtDen[i]->Fill(trkDen.getPt());
          if (std::abs(trkDen.getEta()) > 0.05) {
            mPtDen_noEta0[i]->Fill(trkDen.getPt());
          }
          mPhiDen[i]->Fill(trkDen.getPhi());
          mPhiVsPtDen[i]->Fill(trkDen.getPt(), trkDen.getPhi());
          m1OverPtDen[i]->Fill(trkDen.getSign() * trkDen.getPtInv());
          if (mUseTrkPID) { // Vs Tracking PID hypothesis
            mPtNumVsTrkPID[i][trkDen.getPID()]->Fill(trkDen.getPt());
            mPhiNumVsTrkPID[i][trkDen.getPID()]->Fill(trkDen.getPhi());
            // we fill also the denominator
            mPtDenVsTrkPID[i][trkDen.getPID()]->Fill(trkDen.getPt());
            mPhiDenVsTrkPID[i][trkDen.getPID()]->Fill(trkDen.getPhi());
          }
        }
        mEtaNum[i]->Fill(trkDen.getEta());
        mEtaVsPtNum[i]->Fill(trkDen.getPt(), trkDen.getEta());
        // we fill also the denominator
        mEtaDen[i]->Fill(trkDen.getEta());
        mEtaVsPtDen[i]->Fill(trkDen.getPt(), trkDen.getEta());
        if (i == matchType::TPC) {
          auto tpcTrk = mTPCTracks[trk.getRefTPC()];
          mClsVsPtNum[i]->Fill(tpcTrk.getPt(), tpcTrk.getNClusters());
          mChi2VsPtNum[i]->Fill(tpcTrk.getPt(), tpcTrk.getChi2());
          mClsVsPtDen[i]->Fill(tpcTrk.getPt(), tpcTrk.getNClusters());
          mChi2VsPtDen[i]->Fill(tpcTrk.getPt(), tpcTrk.getChi2());
          math_utils::Point3D<float> v{};
          std::array<float, 2> dca{};
          if (tpcTrk.propagateParamToDCA(v, mBz, &dca)) {
            mDCArVsPtNum->Fill(tpcTrk.getPt(), dca[0]);
            mDCArVsPtDen->Fill(tpcTrk.getPt(), dca[0]);
          }
        } else {
          const auto& itsTrk = mITSTracks[trk.getRefITS()];
          mClsVsPtNum[i]->Fill(itsTrk.getPt(), itsTrk.getNClusters());
          mChi2VsPtNum[i]->Fill(itsTrk.getPt(), itsTrk.getChi2());
          mClsVsPtDen[i]->Fill(itsTrk.getPt(), itsTrk.getNClusters());
          mChi2VsPtDen[i]->Fill(itsTrk.getPt(), itsTrk.getChi2());
        }
        if (mUseTrkPID) { // Vs Tracking PID hypothesis
          mEtaNumVsTrkPID[i][trkDen.getPID()]->Fill(trkDen.getEta());
          // we fill also the denominator
          mEtaDenVsTrkPID[i][trkDen.getPID()]->Fill(trkDen.getEta());
        }
        if (el.second.mIsPhysicalPrimary) {
          if (isEtaITSOk) {
            mPtPhysPrimNum[i]->Fill(trkDen.getPt());
            mPhiPhysPrimNum[i]->Fill(trkDen.getPhi());
            m1OverPtPhysPrimNum[i]->Fill(trkDen.getSign() * trkDen.getPtInv());
            // we fill also the denominator
            mPtPhysPrimDen[i]->Fill(trkDen.getPt());
            mPhiPhysPrimDen[i]->Fill(trkDen.getPhi());
            m1OverPtPhysPrimDen[i]->Fill(trkDen.getSign() * trkDen.getPtInv());
          }
          mEtaPhysPrimNum[i]->Fill(trkDen.getEta());
          // we fill also the denominator
          mEtaPhysPrimDen[i]->Fill(trkDen.getEta());
        }
        ++mNITSTPCSelectedTracks[i];
      }
    }
  }
  int iITSTPC = 0;
  for (auto const& trk : mITSTPCTracks) {
    if (trk.getRefTPC().getIndex() >= mTPCTracks.size()) {
      LOG(fatal) << "******************** ATTENTION! for TPC track associated to matched track: idx = " << trk.getRefTPC().getIndex() << ", size of container = " << mTPCTracks.size() << " in TF " << evCount;
    }
    std::array<std::string, 2> title{"TPC", "ITS"};
    for (int i = 0; i < matchType::SIZE; ++i) {
      o2::track::TrackParCov trkRef;
      int idxTrkRef;
      bool fillHisto = false;
      bool isEtaITSOk = true;
      if (i == matchType::TPC) {
        trkRef = mTPCTracks[trk.getRefTPC()];
        idxTrkRef = trk.getRefTPC().getIndex();
        if (isTPCTrackSelectedEntry[idxTrkRef] == true) {
          fillHisto = true;
          ++mNITSTPCSelectedTracks[i];
        }
      } else {
        idxTrkRef = trk.getRefITS().getIndex();
        if (trk.getRefITS().getSource() == GID::ITSAB) {
          // do not use afterburner tracks
          LOG(debug) << "Track (ITS) with id " << idxTrkRef << " for ITSTPC track " << iITSTPC << " is from afterburner";
          continue;
        }
        if (idxTrkRef >= mITSTracks.size()) {
          LOG(fatal) << "******************** ATTENTION! for ITS track associated to matched track (NOT from AB): idx = " << trk.getRefITS().getIndex() << ", size of container = " << mITSTracks.size() << " in TF " << evCount;
        }
        trkRef = mITSTracks[trk.getRefITS()];
        LOG(debug) << "Checking track (ITS) with id " << idxTrkRef << " for ITSTPC track " << iITSTPC << " and pt = " << trkRef.getPt();
        if (isITSTrackSelectedEntry[idxTrkRef] == true) {
          LOG(debug) << "Track was selected (ITS), with id " << idxTrkRef << " for ITSTPC track " << iITSTPC << " , we keep it in the numerator, pt = " << trkRef.getPt();
          fillHisto = true;
          ++mNITSTPCSelectedTracks[i];
        } else {
          LOG(debug) << "Track was not selected (ITS), with id " << idxTrkRef << " for ITSTPC track " << iITSTPC << " , we don't keep it in the numerator, pt = " << trkRef.getPt();
        }
        if (std::abs(trkRef.getEta()) > 0.9) {
          // ITS track outside |eta | < 0.9, we don't fill pt, nor phi , nor phi vs pt histos
          isEtaITSOk = false;
          LOG(debug) << "Track (ITS), with id " << idxTrkRef << " for ITSTPC track " << iITSTPC << " will be discarded when filling pt of phi related histograms, since eta = " << trkRef.getEta() << " , we don't keep it in the numerator, pt = " << trkRef.getPt();
        }
      }
      if (fillHisto == true) {
        if (!mUseMC) {
          LOG(debug) << "Filling num (" << title[i] << ") with track with id " << idxTrkRef << " for ITSTPC track " << iITSTPC << " with pt = " << trkRef.getPt();
          if (isEtaITSOk) {
            mPtNum[i]->Fill(trkRef.getPt());
            if (std::abs(trkRef.getEta()) > 0.05) {
              mPtNum_noEta0[i]->Fill(trkRef.getPt());
            }
            mPhiNum[i]->Fill(trkRef.getPhi());
            if (mUseTrkPID) { // Vs Tracking PID hypothesis
              mPtNumVsTrkPID[i][trkRef.getPID()]->Fill(trkRef.getPt());
              mPhiNumVsTrkPID[i][trkRef.getPID()]->Fill(trkRef.getPhi());
            }
            mPhiVsPtNum[i]->Fill(trkRef.getPt(), trkRef.getPhi());
            m1OverPtNum[i]->Fill(trkRef.getSign() * trkRef.getPtInv());
          }
          mEtaNum[i]->Fill(trkRef.getEta());
          if (mUseTrkPID) { // Vs Tracking PID hypothesis
            mEtaNumVsTrkPID[i][trkRef.getPID()]->Fill(trkRef.getEta());
          }
          mEtaVsPtNum[i]->Fill(trkRef.getPt(), trkRef.getEta());
          if (i == matchType::TPC) {
            const auto& tpcTrk = mTPCTracks[trk.getRefTPC()];
            mClsVsPtNum[i]->Fill(tpcTrk.getPt(), tpcTrk.getNClusters());
            mChi2VsPtNum[i]->Fill(tpcTrk.getPt(), tpcTrk.getChi2());
          } else {
            const auto& itsTrk = mITSTracks[trk.getRefITS()];
            mClsVsPtNum[i]->Fill(itsTrk.getPt(), itsTrk.getNClusters());
            mChi2VsPtNum[i]->Fill(itsTrk.getPt(), itsTrk.getChi2());
          }
        }
        if (i == matchType::TPC) {
          mResidualPt->Fill(trk.getPt(), trk.getPt() - trkRef.getPt());
          mResidualPhi->Fill(trk.getPhi(), trk.getPhi() - trkRef.getPhi());
          mResidualEta->Fill(trk.getEta(), trk.getEta() - trkRef.getEta());
          mChi2Matching->Fill(trk.getChi2Match());
          mChi2Refit->Fill(trk.getChi2Refit());
          mTimeResVsPt->Fill(trkRef.getPt(), trk.getTimeMUS().getTimeStampError());
          math_utils::Point3D<float> v{};
          std::array<float, 2> dca;
          if (trkRef.propagateParamToDCA(v, mBz, &dca)) {
            mDCAr->Fill(dca[0]);
            if (!mUseMC) {
              mDCArVsPtNum->Fill(trkRef.getPt(), dca[0]);
            }
          }
          LOG(debug) << "*** chi2Matching = " << trk.getChi2Match() << ", chi2refit = " << trk.getChi2Refit() << ", timeResolution = " << trk.getTimeMUS().getTimeStampError();
        }
      } else {
        LOG(debug) << "Not filling num (" << title[i] << ") for ITSTPC track " << iITSTPC << " for track with pt " << trkRef.getPt();
      }
    }
    ++iITSTPC;
  }

  // now filling the denominator for the efficiency calculation
  if (mUseMC) {
    for (int i = 0; i < matchType::SIZE; ++i) {
      mMapRefLabels[i].clear();
    }
    // filling the map where we store for each MC label, the track id of the reconstructed
    // track with the highest number of TPC clusters
    for (int itrk = 0; itrk < static_cast<int>(mTPCTracks.size()); ++itrk) {
      auto const& trk = mTPCTracks[itrk];
      if (isTPCTrackSelectedEntry[itrk] == true) {
        auto lbl = mRecoCont.getTrackMCLabel({(unsigned int)(itrk), GID::Source::TPC});
        if (!lbl.isValid()) {
          continue;
        }
        if (mMapLabels[matchType::TPC].find(lbl) != mMapLabels[matchType::TPC].end()) {
          // the track was already added to the denominator
          continue;
        }
        if (mMapRefLabels[matchType::TPC].find(lbl) == mMapRefLabels[matchType::TPC].end()) {
          int source = lbl.getSourceID();
          int event = lbl.getEventID();
          const std::vector<o2::MCTrack>& pcontainer = mcReader.getTracks(source, event);
          const o2::MCTrack& p = pcontainer[lbl.getTrackID()];
          if (MCTrackNavigator::isPhysicalPrimary(p, pcontainer)) {
            mMapRefLabels[matchType::TPC].insert({lbl, {itrk, true}});
          } else {
            mMapRefLabels[matchType::TPC].insert({lbl, {itrk, false}});
          }
        } else {
          // winner (if more tracks have the same label) has the highest number of TPC clusters
          if (mTPCTracks[mMapRefLabels[matchType::TPC].at(lbl).mIdx].getNClusters() < trk.getNClusters()) {
            mMapRefLabels[matchType::TPC].at(lbl).mIdx = itrk;
          }
        }
      }
    }
    // same for ITS
    // filling the map where we store for each MC label, the track id of the reconstructed
    // track with the highest number of ITS clusters
    for (int itrk = 0; itrk < static_cast<int>(mITSTracks.size()); ++itrk) {
      auto const& trk = mITSTracks[itrk];
      if (isITSTrackSelectedEntry[itrk] == true) {
        auto lbl = mRecoCont.getTrackMCLabel({(unsigned int)(itrk), GID::Source::ITS});
        if (!lbl.isValid()) {
          continue;
        }
        if (mMapLabels[matchType::ITS].find(lbl) != mMapLabels[matchType::ITS].end()) {
          // the track was already added to the denominator
          continue;
        }
        if (mMapRefLabels[matchType::ITS].find(lbl) == mMapRefLabels[matchType::ITS].end()) {
          int source = lbl.getSourceID();
          int event = lbl.getEventID();
          const std::vector<o2::MCTrack>& pcontainer = mcReader.getTracks(source, event);
          const o2::MCTrack& p = pcontainer[lbl.getTrackID()];
          if (MCTrackNavigator::isPhysicalPrimary(p, pcontainer)) {
            mMapRefLabels[matchType::ITS].insert({lbl, {itrk, true}});
          } else {
            mMapRefLabels[matchType::ITS].insert({lbl, {itrk, false}});
          }
        } else {
          // winner (if more tracks have the same label) has the highest number of ITS clusters
          if (mITSTracks[mMapRefLabels[matchType::ITS].at(lbl).mIdx].getNClusters() < trk.getNClusters()) {
            mMapRefLabels[matchType::ITS].at(lbl).mIdx = itrk;
          }
        }
      }
    }
    LOG(info) << "number of entries in map for denominator of TPC tracks (without duplicates) = " << mMapRefLabels[matchType::TPC].size() + mMapLabels[matchType::TPC].size();
    LOG(info) << "number of entries in map for denominator of ITS tracks (without duplicates) = " << mMapRefLabels[matchType::ITS].size() + mMapLabels[matchType::ITS].size();
    // now we use only the tracks in the map to fill the histograms (--> tracks have passed the
    // track selection and there are no duplicated tracks wrt the same MC label)
    for (auto const& el : mMapRefLabels[matchType::TPC]) {
      auto const& trk = mTPCTracks[el.second.mIdx];
      mPtDen[matchType::TPC]->Fill(trk.getPt());
      if (std::abs(trk.getEta()) > 0.05) {
        mPtDen_noEta0[matchType::TPC]->Fill(trk.getPt());
      }
      mPhiDen[matchType::TPC]->Fill(trk.getPhi());
      mPhiVsPtDen[matchType::TPC]->Fill(trk.getPt(), trk.getPhi());
      mEtaDen[matchType::TPC]->Fill(trk.getEta());
      mEtaVsPtDen[matchType::TPC]->Fill(trk.getPt(), trk.getEta());
      m1OverPtDen[matchType::TPC]->Fill(trk.getSign() * trk.getPtInv());
      mClsVsPtDen[matchType::TPC]->Fill(trk.getPt(), trk.getNClusters());
      mChi2VsPtDen[matchType::TPC]->Fill(trk.getPt(), trk.getChi2());
      math_utils::Point3D<float> v{};
      std::array<float, 2> dca{};
      if (auto trc = trk; trc.propagateParamToDCA(v, mBz, &dca)) {
        mDCArVsPtDen->Fill(trc.getPt(), dca[0]);
      }
      if (el.second.mIsPhysicalPrimary) {
        mPtPhysPrimDen[matchType::TPC]->Fill(trk.getPt());
        mPhiPhysPrimDen[matchType::TPC]->Fill(trk.getPhi());
        mEtaPhysPrimDen[matchType::TPC]->Fill(trk.getEta());
        m1OverPtPhysPrimDen[matchType::TPC]->Fill(trk.getSign() * trk.getPtInv());
      }
      ++mNTPCSelectedTracks;
    }
    for (auto const& el : mMapRefLabels[matchType::ITS]) {
      auto const& trk = mITSTracks[el.second.mIdx];
      if (std::abs(trk.getEta()) < 0.9) {
        mPtDen[matchType::ITS]->Fill(trk.getPt());
        if (std::abs(trk.getEta()) > 0.05) {
          mPtDen_noEta0[matchType::ITS]->Fill(trk.getPt());
        }
        mPhiDen[matchType::ITS]->Fill(trk.getPhi());
        mPhiVsPtDen[matchType::ITS]->Fill(trk.getPt(), trk.getPhi());
        m1OverPtDen[matchType::ITS]->Fill(trk.getSign() * trk.getPtInv());
      }
      mEtaDen[matchType::ITS]->Fill(trk.getEta());
      mEtaVsPtDen[matchType::ITS]->Fill(trk.getPt(), trk.getEta());
      mClsVsPtDen[matchType::ITS]->Fill(trk.getPt(), trk.getNClusters());
      mChi2VsPtDen[matchType::ITS]->Fill(trk.getPt(), trk.getChi2());
      if (el.second.mIsPhysicalPrimary) {
        if (std::abs(trk.getEta()) < 0.9) {
          mPtPhysPrimDen[matchType::ITS]->Fill(trk.getPt());
          mPhiPhysPrimDen[matchType::ITS]->Fill(trk.getPhi());
          m1OverPtPhysPrimDen[matchType::ITS]->Fill(trk.getSign() * trk.getPtInv());
        }
        mEtaPhysPrimDen[matchType::ITS]->Fill(trk.getEta());
      }
      ++mNITSSelectedTracks;
    }
  } else {
    // if we are in data, we loop over all tracks (no check on the label)
    for (size_t itrk = 0; itrk < mTPCTracks.size(); ++itrk) {
      auto const& trk = mTPCTracks[itrk];
      if (isTPCTrackSelectedEntry[itrk] == true) {
        LOG(debug) << "Filling den (TPC) with track with pt = " << trk.getPt();
        mPtDen[matchType::TPC]->Fill(trk.getPt());
        if (std::abs(trk.getEta()) > 0.05) {
          mPtDen_noEta0[matchType::TPC]->Fill(trk.getPt());
        } else {
          LOG(debug) << "Track (ITS) " << itrk << " with pt = " << trk.getPt() << " and eta = " << trk.getEta() << " not used for den pt, phi, phi vs pt, 1.pt histos";
        }
        mPhiDen[matchType::TPC]->Fill(trk.getPhi());
        mPhiVsPtDen[matchType::TPC]->Fill(trk.getPt(), trk.getPhi());
        mEtaDen[matchType::TPC]->Fill(trk.getEta());
        mEtaVsPtDen[matchType::TPC]->Fill(trk.getPt(), trk.getEta());
        m1OverPtDen[matchType::TPC]->Fill(trk.getSign() * trk.getPtInv());
        mClsVsPtDen[matchType::TPC]->Fill(trk.getPt(), trk.getNClusters());
        mChi2VsPtDen[matchType::TPC]->Fill(trk.getPt(), trk.getChi2());
        math_utils::Point3D<float> v{};
        std::array<float, 2> dca{};
        if (auto trc = trk; trc.propagateParamToDCA(v, mBz, &dca)) {
          mDCArVsPtDen->Fill(trc.getPt(), dca[0]);
        }
        ++mNTPCSelectedTracks;
      }
    }
    for (size_t itrk = 0; itrk < mITSTracks.size(); ++itrk) {
      auto const& trk = mITSTracks[itrk];
      LOG(debug) << "Checking den for track (ITS) " << itrk << " with pt " << trk.getPt() << " and eta = " << trk.getEta();
      if (isITSTrackSelectedEntry[itrk] == true) {
        if (std::abs(trk.getEta()) < 0.9) {
          LOG(debug) << "Filling den for track (ITS) " << itrk << " with pt = " << trk.getPt() << " and eta = " << trk.getEta();
          mPtDen[matchType::ITS]->Fill(trk.getPt());
          if (std::abs(trk.getEta()) > 0.05) {
            mPtDen_noEta0[matchType::ITS]->Fill(trk.getPt());
          }
          mPhiDen[matchType::ITS]->Fill(trk.getPhi());
          mPhiVsPtDen[matchType::ITS]->Fill(trk.getPt(), trk.getPhi());
          m1OverPtDen[matchType::ITS]->Fill(trk.getSign() * trk.getPtInv());
        } else {
          LOG(debug) << "Track (ITS) " << itrk << " with pt = " << trk.getPt() << " and eta = " << trk.getEta() << " not used for num pt, phi, phi vs pt, 1.pt histos";
        }
        mEtaDen[matchType::ITS]->Fill(trk.getEta());
        mEtaVsPtDen[matchType::ITS]->Fill(trk.getPt(), trk.getEta());
        mClsVsPtDen[matchType::ITS]->Fill(trk.getPt(), trk.getNClusters());
        mChi2VsPtDen[matchType::ITS]->Fill(trk.getPt(), trk.getChi2());
        ++mNITSSelectedTracks;
      } else {
        LOG(debug) << "Not filling for this track (ITS) " << itrk << " with pt = " << trk.getPt();
      }
    }
  }
  evCount++;
}

//__________________________________________________________

bool MatchITSTPCQC::selectTrack(o2::tpc::TrackTPC const& track)
{

  if (track.getPt() < mPtCut) {
    return false;
  }
  if (std::abs(track.getEta()) > mEtaCut) {
    return false;
  }
  if (track.getNClusters() < mNTPCClustersCut) {
    return false;
  }

  math_utils::Point3D<float> v{};
  std::array<float, 2> dca{};
  if (!(const_cast<o2::tpc::TrackTPC&>(track).propagateParamToDCA(v, mBz, &dca, mDCATPCCut)) || std::abs(dca[0]) > mDCATPCCutY) {
    return false;
  }

  return true;
}

//__________________________________________________________

void MatchITSTPCQC::finalize()
{

  std::array<std::string, 2> title{"TPC", "ITS"};

  // first we use denominators and nominators to set the TEfficiency; later they are scaled

  // some checks
  for (int ti = 0; ti < matchType::SIZE; ++ti) {
    for (int i = 0; i < mPtDen[ti]->GetNbinsX(); ++i) {
      if (mPtDen[ti]->GetBinContent(i + 1) < mPtNum[ti]->GetBinContent(i + 1)) {
        LOG(error) << title[ti] << ": bin " << i + 1 << " in [" << mPtNum[ti]->GetBinLowEdge(i + 1) << " , " << mPtNum[ti]->GetBinLowEdge(i + 1) + mPtNum[ti]->GetBinWidth(i + 1) << "]: mPtDen[i] = " << mPtDen[ti]->GetBinContent(i + 1) << ", mPtNum[i] = " << mPtNum[ti]->GetBinContent(i + 1);
      }
    }
    for (int i = 0; i < mPtDen_noEta0[ti]->GetNbinsX(); ++i) {
      if (mPtDen_noEta0[ti]->GetBinContent(i + 1) < mPtNum_noEta0[ti]->GetBinContent(i + 1)) {
        LOG(error) << title[ti] << ": bin " << i + 1 << " in [" << mPtNum_noEta0[ti]->GetBinLowEdge(i + 1) << " , " << mPtNum_noEta0[ti]->GetBinLowEdge(i + 1) + mPtNum_noEta0[ti]->GetBinWidth(i + 1) << "]: mPtDen_noEta0[i] = " << mPtDen_noEta0[ti]->GetBinContent(i + 1) << ", mPtNum_noEta0[i] = " << mPtNum_noEta0[ti]->GetBinContent(i + 1);
      }
    }
    for (int i = 0; i < mPhiDen[ti]->GetNbinsX(); ++i) {
      if (mPhiDen[ti]->GetBinContent(i + 1) < mPhiNum[ti]->GetBinContent(i + 1)) {
        LOG(error) << title[ti] << ": bin " << i + 1 << " in [" << mPhiNum[ti]->GetBinLowEdge(i + 1) << " , " << mPhiNum[ti]->GetBinLowEdge(i + 1) + mPhiNum[ti]->GetBinWidth(i + 1) << "]: mPhiDen[i] = " << mPhiDen[ti]->GetBinContent(i + 1) << ", mPhiNum[i] = " << mPhiNum[ti]->GetBinContent(i + 1);
      }
    }
    for (int i = 0; i < mEtaDen[ti]->GetNbinsX(); ++i) {
      if (mEtaDen[ti]->GetBinContent(i + 1) < mEtaNum[ti]->GetBinContent(i + 1)) {
        LOG(error) << title[ti] << ": bin " << i + 1 << " in [" << mEtaNum[ti]->GetBinLowEdge(i + 1) << " , " << mEtaNum[ti]->GetBinLowEdge(i + 1) + mEtaNum[ti]->GetBinWidth(i + 1) << "]: mEtaDen[i] = " << mEtaDen[ti]->GetBinContent(i + 1) << ", mEtaNum[i] = " << mEtaNum[ti]->GetBinContent(i + 1);
      }
    }

    // filling the efficiency
    setEfficiency(mFractionITSTPCmatch[ti], mPtNum[ti], mPtDen[ti]);
    setEfficiency(mFractionITSTPCmatch_noEta0[ti], mPtNum_noEta0[ti], mPtDen_noEta0[ti]);
    setEfficiency(mFractionITSTPCmatchPhi[ti], mPhiNum[ti], mPhiDen[ti]);
    setEfficiency(mFractionITSTPCmatchEta[ti], mEtaNum[ti], mEtaDen[ti]);
    setEfficiency(mFractionITSTPCmatchPhiVsPt[ti], mPhiVsPtNum[ti], mPhiVsPtDen[ti], true);
    setEfficiency(mFractionITSTPCmatchEtaVsPt[ti], mEtaVsPtNum[ti], mEtaVsPtDen[ti], true);
    setEfficiency(mFractionITSTPCmatch1OverPt[ti], m1OverPtNum[ti], m1OverPtDen[ti]);
    setEfficiency(mFractionITSTPCmatchClsVsPt[ti], mClsVsPtNum[ti], mClsVsPtDen[ti], true);
    setEfficiency(mFractionITSTPCmatchChi2VsPt[ti], mChi2VsPtNum[ti], mChi2VsPtDen[ti], true);
    if (mUseTrkPID) { // Vs Tracking PID hypothesis
      for (int j = 0; j < o2::track::PID::NIDs; ++j) {
        setEfficiency(mFractionITSTPCmatchPtVsTrkPID[ti][j], mPtNumVsTrkPID[ti][j], mPtDenVsTrkPID[ti][j]);
        setEfficiency(mFractionITSTPCmatchPhiVsTrkPID[ti][j], mPhiNumVsTrkPID[ti][j], mPhiDenVsTrkPID[ti][j]);
        setEfficiency(mFractionITSTPCmatchEtaVsTrkPID[ti][j], mEtaNumVsTrkPID[ti][j], mEtaDenVsTrkPID[ti][j]);
      }
    }
    if (mUseMC) {
      setEfficiency(mFractionITSTPCmatchPhysPrim[ti], mPtPhysPrimNum[ti], mPtPhysPrimDen[ti]);
      setEfficiency(mFractionITSTPCmatchPhiPhysPrim[ti], mPhiPhysPrimNum[ti], mPhiPhysPrimDen[ti]);
      setEfficiency(mFractionITSTPCmatchEtaPhysPrim[ti], mEtaPhysPrimNum[ti], mEtaPhysPrimDen[ti]);
      setEfficiency(mFractionITSTPCmatchPhysPrim1OverPt[ti], m1OverPtPhysPrimNum[ti], m1OverPtPhysPrimDen[ti]);
    }
  }
  setEfficiency(mFractionITSTPCmatchDCArVsPt, mDCArVsPtNum, mDCArVsPtDen, true);
  /*
  mPtTPC->Scale(scaleFactTPC);
  mPt->Scale(scaleFactITSTPC);
  mPhiTPC->Scale(scaleFactTPC);
  mPhi->Scale(scaleFactITSTPC);
  if (mUseMC) {
    mPtTPCPhysPrim->Scale(scaleFactTPC);
    mPtPhysPrim->Scale(scaleFactITSTPC);
    mPhiTPCPhysPrim->Scale(scaleFactTPC);
    mPhiPhysPrim->Scale(scaleFactITSTPC);
  }
  mEta->Scale(scaleFactITSTPC);
  mChi2Matching->Scale(scaleFactITSTPC);
  mChi2Refit->Scale(scaleFactITSTPC);
  //mTimeResVsPt->Scale(scaleFactITSTPC); // if to few entries, one sees nothing after normalization --> let's not normalize
  */
}

//__________________________________________________________

void MatchITSTPCQC::setEfficiency(TEfficiency* eff, TH1* hnum, TH1* hden, bool is2D)
{
  if (eff == nullptr) {
    LOG(fatal) << "Cannot get TEfficiency object ";
  }
  if (hnum == nullptr) {
    LOG(fatal) << "Cannot get numerator histogram for TEfficiency object " << eff->GetName();
  }
  if (hden == nullptr) {
    LOG(fatal) << "Cannot get denominator histogram for TEfficiency object " << eff->GetName();
  }

  // we need to force to replace the total histogram, otherwise it will compare it to the previous passed one, and it might get an error of inconsistency in the bin contents
  if constexpr (false) { // checking
    bool bad{false};
    LOG(info) << "Setting efficiency " << eff->GetName() << " from " << hnum->GetName() << " and " << hden->GetName();
    LOG(info) << "Num " << hnum->GetName() << " " << hnum->GetNbinsX() << " " << hnum->GetNbinsY() << " with " << hnum->GetEntries() << " entries";
    LOG(info) << "Den " << hden->GetName() << " " << hden->GetNbinsX() << " " << hden->GetNbinsY() << " with " << hden->GetEntries() << " entries";
    if (hnum->GetDimension() != hden->GetDimension()) {
      LOGP(warning, "Histograms have different dimensions (num={} to den={})", hnum->GetDimension(), hden->GetDimension());
      bad = true;
    }
    if (!TEfficiency::CheckBinning(*hnum, *hden)) {
      LOGP(warning, "Histograms do not have a compatible binning");
      bad = true;
    }
    if (!is2D) {
      for (int i = 1; i <= hden->GetNbinsX(); i++) {
        if (hden->GetBinContent(i) < hnum->GetBinContent(i)) {
          LOG(warning) << "bin " << i << " den: " << hden->GetBinContent(i) << " < num: " << hnum->GetBinContent(i) << " should be the opposite";
          bad = true;
        }
      }
    } else {
      for (int i = 1; i <= hden->GetNbinsX(); i++) {
        for (int j = 1; j <= hden->GetNbinsY(); j++) {
          if (hden->GetBinContent(i, j) < hnum->GetBinContent(i, j)) {
            LOGP(warning, "bin {}/{} -> den: {} < num: {}", i, j, hden->GetBinContent(i, j), hnum->GetBinContent(i, j));
            bad = true;
          }
        }
      }
    }
    if (bad) {
      return;
    }
  }
  // we need to force to replace the total histogram, otherwise it will compare it to the previous passed one, and it might get an error of inconsistency in the bin contents
  if (!eff->SetTotalHistogram(*hden, "f")) {
    LOG(fatal) << "Something went wrong when defining the efficiency denominator " << eff->GetName() << " from " << hnum->GetName();
  }
  if (!eff->SetPassedHistogram(*hnum, "")) {
    LOG(fatal) << "Something went wrong when defining the efficiency numerator " << eff->GetName() << " from " << hnum->GetName();
  }
  if (is2D) {
    eff->SetTitle(Form("%s;%s;%s;%s", eff->GetTitle(), hnum->GetXaxis()->GetTitle(), hnum->GetYaxis()->GetTitle(), "Efficiency"));
  } else {
    eff->SetTitle(Form("%s;%s;%s", eff->GetTitle(), hnum->GetXaxis()->GetTitle(), "Efficiency"));
  }
}

//__________________________________________________________

void MatchITSTPCQC::getHistos(TObjArray& objar)
{

  for (int i = 0; i < matchType::SIZE; ++i) {
    objar.Add(mPtNum[i]);
    objar.Add(mPtDen[i]);
    objar.Add(mFractionITSTPCmatch[i]);

    objar.Add(mPtNum_noEta0[i]);
    objar.Add(mPtDen_noEta0[i]);
    objar.Add(mFractionITSTPCmatch_noEta0[i]);

    objar.Add(mPtPhysPrimNum[i]);
    objar.Add(mPtPhysPrimDen[i]);
    objar.Add(mFractionITSTPCmatchPhysPrim[i]);

    objar.Add(mPhiNum[i]);
    objar.Add(mPhiDen[i]);
    objar.Add(mFractionITSTPCmatchPhi[i]);

    if (mUseTrkPID) { // Vs Tracking PID hypothesis
      for (int j = 0; j < o2::track::PID::NIDs; ++j) {
        // Pt
        objar.Add(mPtNumVsTrkPID[i][j]);
        objar.Add(mPtDenVsTrkPID[i][j]);
        objar.Add(mFractionITSTPCmatchPtVsTrkPID[i][j]);
        // Phi
        objar.Add(mPhiNumVsTrkPID[i][j]);
        objar.Add(mPhiDenVsTrkPID[i][j]);
        objar.Add(mFractionITSTPCmatchPhiVsTrkPID[i][j]);
        // Eta
        objar.Add(mEtaNumVsTrkPID[i][j]);
        objar.Add(mEtaDenVsTrkPID[i][j]);
        objar.Add(mFractionITSTPCmatchEtaVsTrkPID[i][j]);
      }
    }

    objar.Add(mPhiPhysPrimNum[i]);
    objar.Add(mPhiPhysPrimDen[i]);
    objar.Add(mFractionITSTPCmatchPhiPhysPrim[i]);

    objar.Add(mPhiVsPtNum[i]);
    objar.Add(mPhiVsPtDen[i]);
    objar.Add(mFractionITSTPCmatchPhiVsPt[i]);

    objar.Add(mEtaNum[i]);
    objar.Add(mEtaDen[i]);
    objar.Add(mFractionITSTPCmatchEta[i]);

    objar.Add(mEtaPhysPrimNum[i]);
    objar.Add(mEtaPhysPrimDen[i]);
    objar.Add(mFractionITSTPCmatchEtaPhysPrim[i]);

    objar.Add(mEtaVsPtNum[i]);
    objar.Add(mEtaVsPtDen[i]);
    objar.Add(mFractionITSTPCmatchEtaVsPt[i]);

    objar.Add(mClsVsPtNum[i]);
    objar.Add(mClsVsPtDen[i]);
    objar.Add(mFractionITSTPCmatchClsVsPt[i]);

    objar.Add(mChi2VsPtNum[i]);
    objar.Add(mChi2VsPtDen[i]);
    objar.Add(mFractionITSTPCmatchChi2VsPt[i]);

    objar.Add(m1OverPtNum[i]);
    objar.Add(m1OverPtDen[i]);
    objar.Add(mFractionITSTPCmatch1OverPt[i]);

    objar.Add(m1OverPtPhysPrimNum[i]);
    objar.Add(m1OverPtPhysPrimDen[i]);
    objar.Add(mFractionITSTPCmatchPhysPrim1OverPt[i]);
  }
  objar.Add(mChi2Matching);
  objar.Add(mChi2Refit);
  objar.Add(mTimeResVsPt);
  objar.Add(mResidualPt);
  objar.Add(mResidualPhi);
  objar.Add(mResidualEta);
  objar.Add(mDCAr);
  objar.Add(mDCArVsPtNum);
  objar.Add(mDCArVsPtDen);
  objar.Add(mFractionITSTPCmatchDCArVsPt);
}
