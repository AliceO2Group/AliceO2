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

/// \file MatchTPCITS.h
/// \brief Class to perform TPC ITS matching
/// \author ruben.shahoyan@cern.ch

#ifndef ALICEO2_GLOBTRACKING_MATCHTPCITS_QC_
#define ALICEO2_GLOBTRACKING_MATCHTPCITS_QC_

#include <TH1D.h>
#include <TH1F.h>
#include <TH2F.h>
#include <TEfficiency.h>
#include <TObjArray.h>
#include "DataFormatsGlobalTracking/RecoContainer.h"
#include "Framework/ProcessingContext.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTrack.h"
#include "Steer/MCKinematicsReader.h"
#include <unordered_map>
#include <vector>
#include <array>

namespace o2
{

namespace globaltracking
{

using GID = o2::dataformats::GlobalTrackID;

struct LblInfo {

  int mIdx = -1;
  bool mIsPhysicalPrimary = false;
};

class MatchITSTPCQC
{
 public:
  enum matchType : int { TPC = 0,
                         ITS,
                         SIZE };

  MatchITSTPCQC() = default;
  ~MatchITSTPCQC();

  bool init();
  void initDataRequest();
  void run(o2::framework::ProcessingContext& ctx);
  void setDataRequest(std::shared_ptr<o2::globaltracking::DataRequest> dr) { mDataRequest = dr; }
  void finalize();
  void reset();

  TH1D* getHistoPtNum(matchType m) const { return mPtNum[m]; }
  TH1D* getHistoPtDen(matchType m) const { return mPtDen[m]; }
  TEfficiency* getFractionITSTPCmatch(matchType m) const { return mFractionITSTPCmatch[m]; }

  TH1D* getHistoPtNumNoEta0(matchType m) const { return mPtNum_noEta0[m]; }
  TH1D* getHistoPtDenNoEta0(matchType m) const { return mPtDen_noEta0[m]; }
  TEfficiency* getFractionITSTPCmatchNoEta0(matchType m) const { return mFractionITSTPCmatch_noEta0[m]; }

  TH1F* getHistoPhiNum(matchType m) const { return mPhiNum[m]; }
  TH1F* getHistoPhiDen(matchType m) const { return mPhiDen[m]; }
  TEfficiency* getFractionITSTPCmatchPhi(matchType m) const { return mFractionITSTPCmatchPhi[m]; }

  TH2F* getHistoPhiVsPtNum(matchType m) const { return mPhiVsPtNum[m]; }
  TH2F* getHistoPhiVsPtDen(matchType m) const { return mPhiVsPtDen[m]; }
  TEfficiency* getFractionITSTPCmatchPhiVsPt(matchType m) const { return mFractionITSTPCmatchPhiVsPt[m]; }

  TH1F* getHistoEtaNum(matchType m) const { return mEtaNum[m]; }
  TH1F* getHistoEtaDen(matchType m) const { return mEtaDen[m]; }
  TEfficiency* getFractionITSTPCmatchEta(matchType m) const { return mFractionITSTPCmatchEta[m]; }

  TH2F* getHistoEtaVsPtNum(matchType m) const { return mEtaVsPtNum[m]; }
  TH2F* getHistoEtaVsPtDen(matchType m) const { return mEtaVsPtDen[m]; }
  TEfficiency* getFractionITSTPCmatchEtaVsPt(matchType m) const { return mFractionITSTPCmatchEtaVsPt[m]; }

  TH1F* getHistoPtPhysPrimNum(matchType m) const { return mPtPhysPrimNum[m]; }
  TH1F* getHistoPtPhysPrimDen(matchType m) const { return mPtPhysPrimDen[m]; }
  TEfficiency* getFractionITSTPCmatchPhysPrim(matchType m) const { return mFractionITSTPCmatchPhysPrim[m]; }

  TH1F* getHistoPhiPhysPrimNum(matchType m) const { return mPhiPhysPrimNum[m]; }
  TH1F* getHistoPhiPhysPrimDen(matchType m) const { return mPhiPhysPrimDen[m]; }
  TEfficiency* getFractionITSTPCmatchPhiPhysPrim(matchType m) const { return mFractionITSTPCmatchPhiPhysPrim[m]; }

  TH1F* getHistoEtaPhysPrimNum(matchType m) const { return mEtaPhysPrimNum[m]; }
  TH1F* getHistoEtaPhysPrimDen(matchType m) const { return mEtaPhysPrimDen[m]; }
  TEfficiency* getFractionITSTPCmatchEtaPhysPrim(matchType m) const { return mFractionITSTPCmatchEtaPhysPrim[m]; }

  TH2F* getHistoResidualPt() const { return mResidualPt; }
  TH2F* getHistoResidualPhi() const { return mResidualPhi; }
  TH2F* getHistoResidualEta() const { return mResidualEta; }

  TH1F* getHistoChi2Matching() const { return mChi2Matching; }
  TH1F* getHistoChi2Refit() const { return mChi2Refit; }
  TH2F* getHistoTimeResVsPt() const { return mTimeResVsPt; }
  TH1F* getHistoDCAr() const { return mDCAr; }

  TH1D* getHisto1OverPtNum(matchType m) const { return m1OverPtNum[m]; }
  TH1D* getHisto1OverPtDen(matchType m) const { return m1OverPtDen[m]; }
  TEfficiency* getFractionITSTPCmatch1OverPt(matchType m) const { return mFractionITSTPCmatch1OverPt[m]; }

  TH1D* getHisto1OverPtPhysPrimNum(matchType m) const { return m1OverPtPhysPrimNum[m]; }
  TH1D* getHisto1OverPtPhysPrimDen(matchType m) const { return m1OverPtPhysPrimDen[m]; }
  TEfficiency* getFractionITSTPCmatchPhysPrim1OverPt(matchType m) const { return mFractionITSTPCmatchPhysPrim1OverPt[m]; }

  void getHistos(TObjArray& objar);

  void setSources(GID::mask_t src) { mSrc = src; }
  void setUseMC(bool b) { mUseMC = b; }
  bool getUseMC() const { return mUseMC; }
  void deleteHistograms();
  void setBz(float bz) { mBz = bz; }

  // track selection
  bool selectTrack(o2::tpc::TrackTPC const& track);
  void setPtCut(float v) { mPtCut = v; }
  void setEtaCut(float v) { mEtaCut = v; }
  void setMinNTPCClustersCut(float v) { mNTPCClustersCut = v; }
  void setMinDCAtoBeamPipeCut(std::array<float, 2> v)
  {
    setMinDCAtoBeamPipeDistanceCut(v[0]);
    setMinDCAtoBeamPipeYCut(v[1]);
  }
  void setMinDCAtoBeamPipeDistanceCut(float v) { mDCACut = v; }
  void setMinDCAtoBeamPipeYCut(float v) { mDCACutY = v; }

  // to remove after merging QC PR
  TH1D* getHistoPt() const { return nullptr; }                               // old
  TH1D* getHistoPtTPC() const { return nullptr; }                            // old
  TEfficiency* getFractionITSTPCmatch() const { return nullptr; }            // old
  TH1F* getHistoPhi() const { return nullptr; }                              // old
  TH1F* getHistoPhiTPC() const { return nullptr; }                           // old
  TEfficiency* getFractionITSTPCmatchPhi() const { return nullptr; }         // old
  TH2F* getHistoPhiVsPt() const { return nullptr; }                          // old
  TEfficiency* getFractionITSTPCmatchPhiVsPt() const { return nullptr; }     // old
  TH1F* getHistoEta() const { return nullptr; }                              // old
  TH1F* getHistoEtaTPC() const { return nullptr; }                           // old
  TEfficiency* getFractionITSTPCmatchEta() const { return nullptr; }         // old
  TH2F* getHistoEtaVsPt() const { return nullptr; }                          // old
  TH2F* getHistoEtaVsPtTPC() const { return nullptr; }                       // old
  TEfficiency* getFractionITSTPCmatchEtaVsPt() const { return nullptr; }     // old
  TH1F* getHistoPtPhysPrim() const { return nullptr; }                       // old
  TH1F* getHistoPtTPCPhysPrim() const { return nullptr; }                    // old
  TEfficiency* getFractionITSTPCmatchPhysPrim() const { return nullptr; }    // old
  TH1F* getHistoPhiPhysPrim() const { return nullptr; }                      // old
  TH1F* getHistoPhiTPCPhysPrim() const { return nullptr; }                   // old
  TEfficiency* getFractionITSTPCmatchPhiPhysPrim() const { return nullptr; } // old
  TH1F* getHistoEtaPhysPrim() const { return nullptr; }                      // old
  TH1F* getHistoEtaTPCPhysPrim() const { return nullptr; }                   // old
  TEfficiency* getFractionITSTPCmatchEtaPhysPrim() const { return nullptr; } // old

 private:
  std::shared_ptr<o2::globaltracking::DataRequest> mDataRequest;
  o2::globaltracking::RecoContainer mRecoCont;
  GID::mask_t mSrc = GID::getSourcesMask("ITS,TPC,ITS-TPC");
  GID::mask_t mAllowedSources = GID::getSourcesMask("ITS,TPC,ITS-TPC");
  // TPC
  gsl::span<const o2::tpc::TrackTPC> mTPCTracks;
  // ITS
  gsl::span<const o2::its::TrackITS> mITSTracks;
  // ITS-TPC
  gsl::span<const o2::dataformats::TrackTPCITS> mITSTPCTracks;
  bool mUseMC = false;
  float mBz = 0;                                              ///< nominal Bz
  std::array<std::unordered_map<o2::MCCompLabel, LblInfo>, matchType::SIZE> mMapLabels;    // map with labels that have been found for the matched ITSTPC tracks; key is the label,
                                                                                           // value is the LbLinfo with the id of the track with the highest pT found with that label so far,
                                                                                           // and the flag to say if it is a physical primary or not
  std::array<std::unordered_map<o2::MCCompLabel, LblInfo>, matchType::SIZE> mMapRefLabels; // map with labels that have been found for the unmatched TPC tracks; key is the label,
                                                                                           // value is the LblInfo with the id of the track with the highest number of TPC clusters found
                                                                                           // with that label so far, and the flag to say if it is a physical primary or not
  o2::steer::MCKinematicsReader mcReader;                     // reader of MC information

  // Pt
  TH1D* mPtNum[matchType::SIZE] = {};
  TH1D* mPtDen[matchType::SIZE] = {};
  TEfficiency* mFractionITSTPCmatch[matchType::SIZE] = {};
  TH1D* mPtNum_noEta0[matchType::SIZE] = {};
  TH1D* mPtDen_noEta0[matchType::SIZE] = {};
  TEfficiency* mFractionITSTPCmatch_noEta0[matchType::SIZE] = {};
  TH1F* mPtPhysPrimNum[matchType::SIZE] = {};
  TH1F* mPtPhysPrimDen[matchType::SIZE] = {};
  TEfficiency* mFractionITSTPCmatchPhysPrim[matchType::SIZE] = {};
  // Phi
  TH1F* mPhiNum[matchType::SIZE] = {};
  TH1F* mPhiDen[matchType::SIZE] = {};
  TEfficiency* mFractionITSTPCmatchPhi[matchType::SIZE] = {};
  TH1F* mPhiPhysPrimNum[matchType::SIZE] = {};
  TH1F* mPhiPhysPrimDen[matchType::SIZE] = {};
  TEfficiency* mFractionITSTPCmatchPhiPhysPrim[matchType::SIZE] = {};
  TH2F* mPhiVsPtNum[matchType::SIZE] = {};
  TH2F* mPhiVsPtDen[matchType::SIZE] = {};
  TEfficiency* mFractionITSTPCmatchPhiVsPt[matchType::SIZE] = {};
  // Eta
  TH1F* mEtaNum[matchType::SIZE] = {};
  TH1F* mEtaDen[matchType::SIZE] = {};
  TEfficiency* mFractionITSTPCmatchEta[matchType::SIZE] = {};
  TH1F* mEtaPhysPrimNum[matchType::SIZE] = {};
  TH1F* mEtaPhysPrimDen[matchType::SIZE] = {};
  TEfficiency* mFractionITSTPCmatchEtaPhysPrim[matchType::SIZE] = {};
  TH2F* mEtaVsPtNum[matchType::SIZE] = {};
  TH2F* mEtaVsPtDen[matchType::SIZE] = {};
  TEfficiency* mFractionITSTPCmatchEtaVsPt[matchType::SIZE] = {};
  // Residuals
  TH2F* mResidualPt = nullptr;
  TH2F* mResidualPhi = nullptr;
  TH2F* mResidualEta = nullptr;
  // Others
  TH1F* mChi2Matching = nullptr;
  TH1F* mChi2Refit = nullptr;
  TH2F* mTimeResVsPt = nullptr;
  TH1F* mDCAr = nullptr;
  // 1/Pt
  TH1D* m1OverPtNum[matchType::SIZE] = {};
  TH1D* m1OverPtDen[matchType::SIZE] = {};
  TEfficiency* mFractionITSTPCmatch1OverPt[matchType::SIZE] = {};
  TH1D* m1OverPtPhysPrimNum[matchType::SIZE] = {};
  TH1D* m1OverPtPhysPrimDen[matchType::SIZE] = {};
  TEfficiency* mFractionITSTPCmatchPhysPrim1OverPt[matchType::SIZE] = {};

  void setEfficiency(TEfficiency* eff, TH1* hnum, TH1* hden, bool is2D = false);

  int mNTPCSelectedTracks = 0;
  int mNITSSelectedTracks = 0;
  int mNITSTPCSelectedTracks[matchType::SIZE] = {0, 0};

  // cut values
  float mPtCut = 0.1f;
  float mEtaCut = 1.4f;
  int32_t mNTPCClustersCut = 60;
  float mDCACut = 100.f;
  float mDCACutY = 10.f;

  ClassDefNV(MatchITSTPCQC, 1);
};
} // namespace globaltracking
} // namespace o2

#endif
