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
  MatchITSTPCQC() = default;
  ~MatchITSTPCQC();

  bool init();
  void initDataRequest();
  void run(o2::framework::ProcessingContext& ctx);
  void setDataRequest(std::shared_ptr<o2::globaltracking::DataRequest> dr) { mDataRequest = dr; }
  void finalize();
  void reset();

  TH1F* getHistoPt() const { return mPt; }
  TH1F* getHistoPtTPC() const { return mPtTPC; }
  TEfficiency* getFractionITSTPCmatch() const { return mFractionITSTPCmatch; }

  TH1F* getHistoPhi() const { return mPhi; }
  TH1F* getHistoPhiTPC() const { return mPhiTPC; }
  TEfficiency* getFractionITSTPCmatchPhi() const { return mFractionITSTPCmatchPhi; }

  TH1F* getHistoEta() const { return mEta; }
  TH1F* getHistoEtaTPC() const { return mEtaTPC; }
  TEfficiency* getFractionITSTPCmatchEta() const { return mFractionITSTPCmatchEta; }

  TH1F* getHistoPtPhysPrim() const { return mPtPhysPrim; }
  TH1F* getHistoPtTPCPhysPrim() const { return mPtTPCPhysPrim; }
  TEfficiency* getFractionITSTPCmatchPhysPrim() const { return mFractionITSTPCmatchPhysPrim; }

  TH1F* getHistoPhiPhysPrim() const { return mPhiPhysPrim; }
  TH1F* getHistoPhiTPCPhysPrim() const { return mPhiTPCPhysPrim; }
  TEfficiency* getFractionITSTPCmatchPhiPhysPrim() const { return mFractionITSTPCmatchPhiPhysPrim; }

  TH1F* getHistoEtaPhysPrim() const { return mEtaPhysPrim; }
  TH1F* getHistoEtaTPCPhysPrim() const { return mEtaTPCPhysPrim; }
  TEfficiency* getFractionITSTPCmatchEtaPhysPrim() const { return mFractionITSTPCmatchEtaPhysPrim; }

  TH2F* getHistoResidualPt() const { return mResidualPt; }
  TH2F* getHistoResidualPhi() const { return mResidualPhi; }
  TH2F* getHistoResidualEta() const { return mResidualEta; }

  TH1F* getHistoChi2Matching() const { return mChi2Matching; }
  TH1F* getHistoChi2Refit() const { return mChi2Refit; }
  TH2F* getHistoTimeResVsPt() const { return mTimeResVsPt; }
  void getHistos(TObjArray& objar);
  void setSources(GID::mask_t src) { mSrc = src; }
  void setUseMC(bool b) { mUseMC = b; }
  bool getUseMC() const { return mUseMC; }
  void deleteHistograms();
  void setGRPFileName(std::string fn) { mGRPFileName = fn; }
  void setGeomFileName(std::string fn) { mGeomFileName = fn; }
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

 private:
  std::shared_ptr<o2::globaltracking::DataRequest> mDataRequest;
  o2::globaltracking::RecoContainer mRecoCont;
  GID::mask_t mSrc = GID::getSourcesMask("TPC,ITS-TPC");
  GID::mask_t mAllowedSources = GID::getSourcesMask("TPC,ITS-TPC");
  // TPC
  gsl::span<const o2::tpc::TrackTPC> mTPCTracks;
  // ITS-TPC
  gsl::span<const o2::dataformats::TrackTPCITS> mITSTPCTracks;
  bool mUseMC = false;
  std::string mGRPFileName = "o2sim_grp.root";
  std::string mGeomFileName = "o2sim_geometry-aligned.root";
  float mBz = 0;                                              ///< nominal Bz
  std::unordered_map<o2::MCCompLabel, LblInfo> mMapLabels;    // map with labels that have been found for the matched ITSTPC tracks; key is the label,
                                                              // value is the LbLinfo with the id of the track with the highest pT found with that label so far,
                                                              // and the flag to say if it is a physical primary or not
  std::unordered_map<o2::MCCompLabel, LblInfo> mMapTPCLabels; // map with labels that have been found for the unmatched TPC tracks; key is the label,
                                                              // value is the LblInfo with the id of the track with the highest number of TPC clusters found
                                                              // with that label so far, and the flag to say if it is a physical primary or not
  o2::steer::MCKinematicsReader mcReader;                     // reader of MC information

  // Pt
  TH1F* mPt = nullptr;
  TH1F* mPtTPC = nullptr;
  TEfficiency* mFractionITSTPCmatch = nullptr;
  TH1F* mPtPhysPrim = nullptr;
  TH1F* mPtTPCPhysPrim = nullptr;
  TEfficiency* mFractionITSTPCmatchPhysPrim = nullptr;
  // Phi
  TH1F* mPhi = nullptr;
  TH1F* mPhiTPC = nullptr;
  TEfficiency* mFractionITSTPCmatchPhi = nullptr;
  TH1F* mPhiPhysPrim = nullptr;
  TH1F* mPhiTPCPhysPrim = nullptr;
  TEfficiency* mFractionITSTPCmatchPhiPhysPrim = nullptr;
  // Eta
  TH1F* mEta = nullptr;
  TH1F* mEtaTPC = nullptr;
  TEfficiency* mFractionITSTPCmatchEta = nullptr;
  TH1F* mEtaPhysPrim = nullptr;
  TH1F* mEtaTPCPhysPrim = nullptr;
  TEfficiency* mFractionITSTPCmatchEtaPhysPrim = nullptr;
  // Residuals
  TH2F* mResidualPt = nullptr;
  TH2F* mResidualPhi = nullptr;
  TH2F* mResidualEta = nullptr;
  // Others
  TH1F* mChi2Matching = nullptr;
  TH1F* mChi2Refit = nullptr;
  TH2F* mTimeResVsPt = nullptr;

  int mNTPCSelectedTracks = 0;
  int mNITSTPCSelectedTracks = 0;

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
