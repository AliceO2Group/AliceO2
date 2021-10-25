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
#include "DetectorsBase/GeometryManager.h"
#include <algorithm>

using namespace o2::globaltracking;

MatchITSTPCQC::~MatchITSTPCQC()
{

  deleteHistograms();
}

//_______________________________________________________

void MatchITSTPCQC::deleteHistograms()
{

  delete mPtTPC;
  delete mFractionITSTPCmatch;
  delete mPt;
  delete mPhiTPC;
  delete mFractionITSTPCmatchPhi;
  delete mPhi;
  delete mEta;
  delete mChi2Matching;
  delete mChi2Refit;
  delete mTimeResVsPt;
}

//__________________________________________________________

void MatchITSTPCQC::reset()
{
  mPtTPC->Reset();
  mPt->Reset();
  mEta->Reset();
  mChi2Matching->Reset();
  mChi2Refit->Reset();
  mTimeResVsPt->Reset();
}

//__________________________________________________________

bool MatchITSTPCQC::init()
{

  mPtTPC = new TH1F("mPtTPC", "Pt distribution of TPC tracks; Pt [GeV/c]; dNdPt", 100, 0.f, 20.f);
  mFractionITSTPCmatch = new TEfficiency("mFractionITSTPCmatch", "Fraction of ITSTPC matched tracks vs Pt; Pt [GeV/c]; Eff", 100, 0.f, 20.f);
  mPt = new TH1F("mPt", "Pt distribution of matched tracks; Pt [GeV/c]; dNdPt", 100, 0.f, 20.f);
  mPhiTPC = new TH1F("mPhiTPC", "Phi distribution of TPC tracks; Phi [rad]; dNdPhi", 100, 0.f, 2 * TMath::Pi());
  mFractionITSTPCmatchPhi = new TEfficiency("mFractionITSTPCmatchPhi", "Fraction of ITSTPC matched tracks vs Phi; Phi [rad]; Eff", 100, 0.f, 2 * TMath::Pi());
  mPhi = new TH1F("mPhi", "Phi distribution of matched tracks; Phi [rad]; dNdPhi", 100, 0.f, 2 * TMath::Pi());
  mEta = new TH1F("mEta", "Eta distribution of matched tracks; Eta; dNdEta", 100, -2.f, 2.f);
  mChi2Matching = new TH1F("mChi2Matching", "Chi2 of matching; chi2", 200, 0, 20);
  mChi2Refit = new TH1F("mChi2Refit", "Chi2 of refit; chi2", 200, 0, 20);
  mTimeResVsPt = new TH2F("mTimeResVsPt", "Time resolution vs Pt; Pt [GeV/c]; time res [us]", 100, 0.f, 20.f, 100, 0.f, 2.f);

  mPtTPC->Sumw2();
  mPt->Sumw2();
  mPhiTPC->Sumw2();
  mPhi->Sumw2();

  mSrc &= mAllowedSources;

  if ((mSrc[GID::Source::ITSTPC] == 0 || mSrc[GID::Source::TPC] == 0)) {
    LOG(fatal) << "We cannot do ITSTPC QC, some sources are missing, check sources in " << mSrc;
  }

  mDataRequest = std::make_shared<o2::globaltracking::DataRequest>();
  mDataRequest->requestTracks(mSrc, mUseMC);

  o2::base::GeometryManager::loadGeometry(mGeomFileName);
  o2::base::Propagator::initFieldFromGRP(mGRPFileName);
  mBz = o2::base::Propagator::Instance()->getNominalBz();

  return true;
}

//__________________________________________________________

void MatchITSTPCQC::run(o2::framework::ProcessingContext& ctx)
{

  mRecoCont.collectData(ctx, *mDataRequest.get());
  mTPCTracks = mRecoCont.getTPCTracks();
  mITSTPCTracks = mRecoCont.getTPCITSTracks();

  LOG(INFO) << "****** Number of found ITSTPC tracks = " << mITSTPCTracks.size();
  LOG(INFO) << "****** Number of found TPC    tracks = " << mTPCTracks.size();

  // cache selection for TPC tracks
  for (auto itrk = 0; itrk < mTPCTracks.size(); ++itrk) {
    auto const& trkTpc = mTPCTracks[itrk];
    if (selectTrack(trkTpc)) {
      mSelectedTPCtracks.push_back(itrk);
    }
  }

  // numerator + eta, chi2...
  if (mUseMC) {
    mMapLabels.clear();
    for (auto itrk = 0; itrk < mITSTPCTracks.size(); ++itrk) {
      auto const& trk = mITSTPCTracks[itrk];
      auto idxTrkTpc = trk.getRefTPC().getIndex();
      if (std::any_of(mSelectedTPCtracks.begin(), mSelectedTPCtracks.end(), [idxTrkTpc](int el) { return el == idxTrkTpc; })) {
        auto lbl = mRecoCont.getTrackMCLabel({uint64_t(itrk), GID::Source::ITSTPC});
        if (mMapLabels.find(lbl) == mMapLabels.end()) {
          mMapLabels.insert({lbl, itrk});
        } else {
          // winner (if more tracks have the same label) has the highest pt
          if (mITSTPCTracks[mMapLabels.at(lbl)].getPt() < trk.getPt()) {
            mMapLabels.at(lbl) = itrk;
          }
        }
      }
    }
    LOG(INFO) << "number of entries in map for nominator (without duplicates) = " << mMapLabels.size();
    // now we use only the tracks in the map to fill the histograms (--> tracks have passed the
    // track selection and there are no duplicated tracks wrt the same MC label)
    for (auto const& el : mMapLabels) {
      auto const& trk = mITSTPCTracks[el.second];
      auto const& trkTpc = mTPCTracks[trk.getRefTPC()];
      mPt->Fill(trkTpc.getPt());
      mPhi->Fill(trkTpc.getPhi());
      // we fill also the denominator
      mPtTPC->Fill(trkTpc.getPt());
      mPhiTPC->Fill(trkTpc.getPhi());
      ++mNITSTPCSelectedTracks;
    }
  }
  for (auto const& trk : mITSTPCTracks) {
    auto const& trkTpc = mTPCTracks[trk.getRefTPC()];
    auto idxTrkTpc = trk.getRefTPC().getIndex();
    if (std::any_of(mSelectedTPCtracks.begin(), mSelectedTPCtracks.end(), [idxTrkTpc](int el) { return el == idxTrkTpc; })) {
      if (!mUseMC) {
        mPt->Fill(trkTpc.getPt());
        mPhi->Fill(trkTpc.getPhi());
      }
      mEta->Fill(trkTpc.getEta());
      mChi2Matching->Fill(trk.getChi2Match());
      mChi2Refit->Fill(trk.getChi2Refit());
      mTimeResVsPt->Fill(trkTpc.getPt(), trk.getTimeMUS().getTimeStampError());
      LOG(DEBUG) << "*** chi2Matching = " << trk.getChi2Match() << ", chi2refit = " << trk.getChi2Refit() << ", timeResolution = " << trk.getTimeMUS().getTimeStampError();
      ++mNITSTPCSelectedTracks;
    }
  }

  // now filling the denominator for the efficiency calculation
  if (mUseMC) {
    mMapTPCLabels.clear();
    // filling the map where we store for each MC label, the track id of the reconstructed
    // track with the highest number of TPC clusters
    for (auto itrk = 0; itrk < mTPCTracks.size(); ++itrk) {
      auto const& trk = mTPCTracks[itrk];
      if (std::any_of(mSelectedTPCtracks.begin(), mSelectedTPCtracks.end(), [itrk](int el) { return el == itrk; })) {
        auto lbl = mRecoCont.getTrackMCLabel({uint64_t(itrk), GID::Source::TPC});
        if (mMapLabels.find(lbl) != mMapLabels.end()) {
          // the track was already added to the denominator
          continue;
        }
        if (mMapTPCLabels.find(lbl) == mMapTPCLabels.end()) {
          mMapTPCLabels.insert({lbl, itrk});
        } else {
          // winner (if more tracks have the same label) has the highest number of TPC clusters
          if (mTPCTracks[mMapTPCLabels.at(lbl)].getNClusters() < trk.getNClusters()) {
            mMapTPCLabels.at(lbl) = itrk;
          }
        }
      }
    }
    LOG(INFO) << "number of entries in map for denominator (without duplicates) = " << mMapTPCLabels.size() + mMapLabels.size();
    // now we use only the tracks in the map to fill the histograms (--> tracks have passed the
    // track selection and there are no duplicated tracks wrt the same MC label)
    for (auto const& el : mMapTPCLabels) {
      auto const& trk = mTPCTracks[el.second];
      mPtTPC->Fill(trk.getPt());
      mPhiTPC->Fill(trk.getPhi());
      ++mNTPCSelectedTracks;
    }
  } else {
    // if we are in data, we loop over all tracks (no check on the label)
    for (int itrk = 0; itrk < mTPCTracks.size(); ++itrk) {
      auto const& trk = mTPCTracks[itrk];
      if (std::any_of(mSelectedTPCtracks.begin(), mSelectedTPCtracks.end(), [itrk](int el) { return el == itrk; })) {
        mPtTPC->Fill(trk.getPt());
        mPhiTPC->Fill(trk.getPhi());
        ++mNTPCSelectedTracks;
      }
    }
  }
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
  std::array<float, 2> dca;
  if (!(const_cast<o2::tpc::TrackTPC&>(track).propagateParamToDCA(v, mBz, &dca, mDCACut)) || std::abs(dca[0]) > mDCACutY) {
    return false;
  }

  return true;
}

//__________________________________________________________

void MatchITSTPCQC::finalize()
{

  // first we use denominators and nominators to set the TEfficiency; later they are scaled

  for (int i = 0; i < mPtTPC->GetNbinsX(); ++i) {
    if (mPtTPC->GetBinContent(i + 1) < mPt->GetBinContent(i + 1)) {
      LOG(ERROR) << "bin " << i + 1 << ": mPtTPC[i] = " << mPtTPC->GetBinContent(i + 1) << ", mPt[i] = " << mPt->GetBinContent(i + 1);
    }
  }
  for (int i = 0; i < mPhiTPC->GetNbinsX(); ++i) {
    if (mPhiTPC->GetBinContent(i + 1) < mPhi->GetBinContent(i + 1)) {
      LOG(ERROR) << "bin " << i + 1 << ": mPhiTPC[i] = " << mPhiTPC->GetBinContent(i + 1) << ", mPhi[i] = " << mPhi->GetBinContent(i + 1);
    }
  }

  if (!mFractionITSTPCmatch->SetTotalHistogram(*mPtTPC, "") ||
      !mFractionITSTPCmatch->SetPassedHistogram(*mPt, "")) {
    LOG(FATAL) << "Something wrong when defining the efficiency histograms vs Pt!";
  }
  if (!mFractionITSTPCmatchPhi->SetTotalHistogram(*mPhiTPC, "") ||
      !mFractionITSTPCmatchPhi->SetPassedHistogram(*mPhi, "")) {
    LOG(FATAL) << "Something wrong when defining the efficiency histograms vs Phi!";
  }

  float scaleFactTPC = 1. / mNTPCSelectedTracks;
  float scaleFactITSTPC = 1. / mNITSTPCSelectedTracks;
  mPtTPC->Scale(scaleFactTPC);
  mPt->Scale(scaleFactITSTPC);
  mPhiTPC->Scale(scaleFactTPC);
  mPhi->Scale(scaleFactITSTPC);
  mEta->Scale(scaleFactITSTPC);
  mChi2Matching->Scale(scaleFactITSTPC);
  mChi2Refit->Scale(scaleFactITSTPC);
  //mTimeResVsPt->Scale(scaleFactITSTPC); // if to few entries, one sees nothing after normalization --> let's not normalize
}

//__________________________________________________________

void MatchITSTPCQC::getHistos(TObjArray& objar)
{

  objar.Add(mPtTPC);
  objar.Add(mFractionITSTPCmatch);
  objar.Add(mPt);
  objar.Add(mPhiTPC);
  objar.Add(mFractionITSTPCmatchPhi);
  objar.Add(mPhi);
  objar.Add(mEta);
  objar.Add(mChi2Matching);
  objar.Add(mChi2Refit);
  objar.Add(mTimeResVsPt);
}

