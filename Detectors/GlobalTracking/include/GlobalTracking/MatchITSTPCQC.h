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

namespace o2
{

namespace globaltracking
{

using GID = o2::dataformats::GlobalTrackID;

class MatchITSTPCQC
{
 public:
  MatchITSTPCQC() = default;
  ~MatchITSTPCQC();

  bool init();
  void run(o2::framework::ProcessingContext& ctx);
  void finalize();
  void reset();
  TH1F* getHistoPtTPC() const { return mPtTPC; }
  TEfficiency* getFractionITSTPCmatch() const { return mFractionITSTPCmatch; }
  TEfficiency* getHistoFractionITSTPCmatch() const { return mFractionITSTPCmatch; }
  TH1F* getHistoPt() const { return mPt; }
  TH1F* getHistoEta() const { return mEta; }
  TH1F* getHistoChi2Matching() const { return mChi2Matching; }
  TH1F* getHistoChi2Refit() const { return mChi2Refit; }
  TH2F* getHistoTimeResVsPt() const { return mTimeResVsPt; }
  void getHistos(TObjArray& objar);
  void setSources(GID::mask_t src) { mSrc = src; }
  void setUseMC(bool b) { mUseMC = b; }
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
  std::string mGeomFileName = "o2sim_geometry.root";
  float mBz = 0; ///< nominal Bz

  TH1F* mPtTPC = nullptr;
  TEfficiency* mFractionITSTPCmatch = nullptr;
  TH1F* mPt = nullptr;
  TH1F* mEta = nullptr;
  TH1F* mChi2Matching = nullptr;
  TH1F* mChi2Refit = nullptr;
  TH2F* mTimeResVsPt = nullptr;

  int mNTPCSelectedTracks = 0;
  int mNITSTPCSelectedTracks = 0;

  // cut values
  float mPtCut = 0.1f;
  float mEtaCut = 1.4f;
  int32_t mNTPCClustersCut = 40;
  float mDCACut = 100.f;
  float mDCACutY = 10.f;

  ClassDefNV(MatchITSTPCQC, 1);
};

} // namespace globaltracking
} // namespace o2

#endif
