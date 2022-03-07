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
#include "TGraphAsymmErrors.h"

using namespace o2::globaltracking;
using MCTrack = o2::MCTrackT<float>;

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
  delete mPtTPCPhysPrim;
  delete mFractionITSTPCmatchPhysPrim;
  delete mPtPhysPrim;
  delete mPhiTPCPhysPrim;
  delete mFractionITSTPCmatchPhiPhysPrim;
  delete mPhiPhysPrim;
  delete mEta;
  delete mChi2Matching;
  delete mChi2Refit;
  delete mTimeResVsPt;
}

//__________________________________________________________

void MatchITSTPCQC::reset()
{
  mPt->Reset();
  mPtTPC->Reset();
  mPhi->Reset();
  mPhiTPC->Reset();
  mPtPhysPrim->Reset();
  mPtTPCPhysPrim->Reset();
  mPhiPhysPrim->Reset();
  mPhiTPCPhysPrim->Reset();
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
  // These will be empty in case of no MC info...
  mPhiTPCPhysPrim = new TH1F("mPhiTPCPhysPrim", "Phi distribution of TPC tracks (physical primary); Phi [rad]; dNdPhi", 100, 0.f, 2 * TMath::Pi());
  mFractionITSTPCmatchPhiPhysPrim = new TEfficiency("mFractionITSTPCmatchPhiPhysPrim", "Fraction of ITSTPC matched tracks vs Phi (physical primary); Phi [rad]; Eff", 100, 0.f, 2 * TMath::Pi());
  mPhiPhysPrim = new TH1F("mPhiPhysPrim", "Phi distribution of matched tracks (physical primary); Phi [rad]; dNdPhi", 100, 0.f, 2 * TMath::Pi());
  // ...till here
  mEta = new TH1F("mEta", "Eta distribution of matched tracks; Eta; dNdEta", 100, -2.f, 2.f);
  mChi2Matching = new TH1F("mChi2Matching", "Chi2 of matching; chi2", 100, 0, 30);
  mChi2Refit = new TH1F("mChi2Refit", "Chi2 of refit; chi2", 200, 0, 100);

  // log binning for pT
  const Int_t nbinsPt = 100;
  const Double_t xminPt = 0.01;
  const Double_t xmaxPt = 20;
  Double_t* xbinsPt = new Double_t[nbinsPt + 1];
  Double_t xlogminPt = TMath::Log10(xminPt);
  Double_t xlogmaxPt = TMath::Log10(xmaxPt);
  Double_t dlogxPt = (xlogmaxPt - xlogminPt) / nbinsPt;
  Double_t yPt = (xlogmaxPt - xlogminPt) / nbinsPt;
  for (int i = 0; i <= nbinsPt; i++) {
    Double_t xlogPt = xlogminPt + i * dlogxPt;
    xbinsPt[i] = TMath::Exp(TMath::Log(10) * xlogPt);
  }
  mTimeResVsPt = new TH2F("mTimeResVsPt", "Time resolution vs Pt; Pt [GeV/c]; time res [us]", nbinsPt, xbinsPt, 100, 0.f, 2.f);
  mPtTPCPhysPrim = new TH1F("mPtTPPhysPrimC", "Pt distribution of TPC tracks (physical primary); Pt [GeV/c]; dNdPt", nbinsPt, xbinsPt);
  mFractionITSTPCmatchPhysPrim = new TEfficiency("mFractionITSTPCmatchPhysPrim", "Fraction of ITSTPC matched tracks vs Pt (physical primary); Pt [GeV/c]; Eff", nbinsPt, xbinsPt);
  mPtPhysPrim = new TH1F("mPtPhysPrim", "Pt distribution of matched tracks (physical primary); Pt [GeV/c]; dNdPt", nbinsPt, xbinsPt);

  mPtTPC->Sumw2();
  mPt->Sumw2();
  mPhiTPC->Sumw2();
  mPhi->Sumw2();
  mPtTPCPhysPrim->Sumw2();
  mPtPhysPrim->Sumw2();
  mPhiTPCPhysPrim->Sumw2();
  mPhiPhysPrim->Sumw2();

  mPtTPC->SetOption("logy");
  mPt->SetOption("logy");
  mEta->SetOption("logy");
  mChi2Matching->SetOption("logy");
  mChi2Refit->SetOption("logy");
  mTimeResVsPt->SetOption("colz logz logy logx");

  mPtTPC->GetYaxis()->SetTitleOffset(1.4);
  mPt->GetYaxis()->SetTitleOffset(1.4);
  mEta->GetYaxis()->SetTitleOffset(1.4);
  mChi2Matching->GetYaxis()->SetTitleOffset(1.4);
  mChi2Refit->GetYaxis()->SetTitleOffset(1.4);
  mTimeResVsPt->GetYaxis()->SetTitleOffset(1.4);

  o2::base::GeometryManager::loadGeometry(mGeomFileName);
  o2::base::Propagator::initFieldFromGRP(mGRPFileName);
  mBz = o2::base::Propagator::Instance()->getNominalBz();

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

  if ((mSrc[GID::Source::ITSTPC] == 0 || mSrc[GID::Source::TPC] == 0)) {
    LOG(fatal) << "We cannot do ITSTPC QC, some sources are missing, check sources in " << mSrc;
  }

  mDataRequest = std::make_shared<o2::globaltracking::DataRequest>();
  mDataRequest->requestTracks(mSrc, mUseMC);
}

//__________________________________________________________

void MatchITSTPCQC::run(o2::framework::ProcessingContext& ctx)
{
  static int evCount = 0;
  mSelectedTPCtracks.clear();
  mRecoCont.collectData(ctx, *mDataRequest.get());
  mTPCTracks = mRecoCont.getTPCTracks();
  mITSTPCTracks = mRecoCont.getTPCITSTracks();

  LOG(debug) << "****** Number of found ITSTPC tracks = " << mITSTPCTracks.size();
  LOG(debug) << "****** Number of found TPC    tracks = " << mTPCTracks.size();

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
        auto lbl = mRecoCont.getTrackMCLabel({(unsigned int)(itrk), GID::Source::ITSTPC});
        if (mMapLabels.find(lbl) == mMapLabels.end()) {
          auto const* mcParticle = mcReader.getTrack(lbl);
          int source = lbl.getSourceID();
          int event = lbl.getEventID();
          const std::vector<o2::MCTrack>& pcontainer = mcReader.getTracks(source, event);
          if (isPhysicalPrimary(mcParticle, pcontainer)) {
            mMapLabels.insert({lbl, {itrk, true}});
          } else {
            mMapLabels.insert({lbl, {itrk, false}});
          }
        } else {
          // winner (if more tracks have the same label) has the highest pt
          if (mITSTPCTracks[mMapLabels.at(lbl).mIdx].getPt() < trk.getPt()) {
            mMapLabels.at(lbl).mIdx = itrk;
          }
        }
      }
    }
    LOG(info) << "number of entries in map for nominator (without duplicates) = " << mMapLabels.size();
    // now we use only the tracks in the map to fill the histograms (--> tracks have passed the
    // track selection and there are no duplicated tracks wrt the same MC label)
    for (auto const& el : mMapLabels) {
      auto const& trk = mITSTPCTracks[el.second.mIdx];
      auto const& trkTpc = mTPCTracks[trk.getRefTPC()];
      mPt->Fill(trkTpc.getPt());
      mPhi->Fill(trkTpc.getPhi());
      // we fill also the denominator
      mPtTPC->Fill(trkTpc.getPt());
      mPhiTPC->Fill(trkTpc.getPhi());
      if (el.second.mIsPhysicalPrimary) {
        mPtPhysPrim->Fill(trkTpc.getPt());
        mPhiPhysPrim->Fill(trkTpc.getPhi());
        // we fill also the denominator
        mPtTPCPhysPrim->Fill(trkTpc.getPt());
        mPhiTPCPhysPrim->Fill(trkTpc.getPhi());
      }
      ++mNITSTPCSelectedTracks;
    }
  }

  for (auto const& trk : mITSTPCTracks) {
    if (trk.getRefTPC().getIndex() >= mTPCTracks.size()) {
      LOG(fatal) << "******************** ATTENTION! idx = " << trk.getRefTPC().getIndex() << ", size of container = " << mTPCTracks.size() << " in TF " << evCount;
      continue;
    }
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
      LOG(debug) << "*** chi2Matching = " << trk.getChi2Match() << ", chi2refit = " << trk.getChi2Refit() << ", timeResolution = " << trk.getTimeMUS().getTimeStampError();
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
        auto lbl = mRecoCont.getTrackMCLabel({(unsigned int)(itrk), GID::Source::TPC});
        if (mMapLabels.find(lbl) != mMapLabels.end()) {
          // the track was already added to the denominator
          continue;
        }
        if (mMapTPCLabels.find(lbl) == mMapTPCLabels.end()) {
          o2::MCTrack const* mcParticle = mcReader.getTrack(lbl);
          int source = lbl.getSourceID();
          int event = lbl.getEventID();
          const std::vector<o2::MCTrack>& pcontainer = mcReader.getTracks(source, event);
          if (isPhysicalPrimary(mcParticle, pcontainer)) {
            mMapTPCLabels.insert({lbl, {itrk, true}});
          } else {
            mMapTPCLabels.insert({lbl, {itrk, false}});
          }
        } else {
          // winner (if more tracks have the same label) has the highest number of TPC clusters
          if (mTPCTracks[mMapTPCLabels.at(lbl).mIdx].getNClusters() < trk.getNClusters()) {
            mMapTPCLabels.at(lbl).mIdx = itrk;
          }
        }
      }
    }
    LOG(info) << "number of entries in map for denominator (without duplicates) = " << mMapTPCLabels.size() + mMapLabels.size();
    // now we use only the tracks in the map to fill the histograms (--> tracks have passed the
    // track selection and there are no duplicated tracks wrt the same MC label)
    for (auto const& el : mMapTPCLabels) {
      auto const& trk = mTPCTracks[el.second.mIdx];
      mPtTPC->Fill(trk.getPt());
      mPhiTPC->Fill(trk.getPhi());
      if (el.second.mIsPhysicalPrimary) {
        mPtTPCPhysPrim->Fill(trk.getPt());
        mPhiTPCPhysPrim->Fill(trk.getPhi());
      }
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
      LOG(error) << "bin " << i + 1 << ": mPtTPC[i] = " << mPtTPC->GetBinContent(i + 1) << ", mPt[i] = " << mPt->GetBinContent(i + 1);
    }
  }
  for (int i = 0; i < mPhiTPC->GetNbinsX(); ++i) {
    if (mPhiTPC->GetBinContent(i + 1) < mPhi->GetBinContent(i + 1)) {
      LOG(error) << "bin " << i + 1 << ": mPhiTPC[i] = " << mPhiTPC->GetBinContent(i + 1) << ", mPhi[i] = " << mPhi->GetBinContent(i + 1);
    }
  }

  // we need to force to replace the total histogram, otherwise it will compare it to the previous passed one, and it might get an error of inconsistency in the bin contents
  if (!mFractionITSTPCmatch->SetTotalHistogram(*mPtTPC, "f") ||
      !mFractionITSTPCmatch->SetPassedHistogram(*mPt, "")) {
    LOG(fatal) << "Something went wrong when defining the efficiency histograms vs Pt!";
  }
  if (!mFractionITSTPCmatchPhi->SetTotalHistogram(*mPhiTPC, "f") ||
      !mFractionITSTPCmatchPhi->SetPassedHistogram(*mPhi, "")) {
    LOG(fatal) << "Something went wrong when defining the efficiency histograms vs Phi!";
  }

  if (mUseMC) {
    if (!mFractionITSTPCmatchPhysPrim->SetTotalHistogram(*mPtTPCPhysPrim, "f") ||
        !mFractionITSTPCmatchPhysPrim->SetPassedHistogram(*mPtPhysPrim, "")) {
      LOG(fatal) << "Something went wrong when defining the efficiency histograms vs Pt (PhysPrim)!";
    }
    if (!mFractionITSTPCmatchPhiPhysPrim->SetTotalHistogram(*mPhiTPCPhysPrim, "f") ||
        !mFractionITSTPCmatchPhiPhysPrim->SetPassedHistogram(*mPhiPhysPrim, "")) {
      LOG(fatal) << "Something went wrong when defining the efficiency histograms vs Phi (PhysPrim)!";
    }
  }

  float scaleFactTPC = 1. / mNTPCSelectedTracks;
  float scaleFactITSTPC = 1. / mNITSTPCSelectedTracks;
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

void MatchITSTPCQC::getHistos(TObjArray& objar)
{

  objar.Add(mPtTPC);
  objar.Add(mFractionITSTPCmatch);
  objar.Add(mPt);
  objar.Add(mPhiTPC);
  objar.Add(mFractionITSTPCmatchPhi);
  objar.Add(mPhi);
  objar.Add(mPtTPCPhysPrim);
  objar.Add(mFractionITSTPCmatchPhysPrim);
  objar.Add(mPtPhysPrim);
  objar.Add(mPhiTPCPhysPrim);
  objar.Add(mFractionITSTPCmatchPhiPhysPrim);
  objar.Add(mPhiPhysPrim);
  objar.Add(mEta);
  objar.Add(mChi2Matching);
  objar.Add(mChi2Refit);
  objar.Add(mTimeResVsPt);
}

//__________________________________________________________
// adapted from AliceO2/DataFormats/simulation/src/MCUtils.cxx class

o2::MCTrack const* MatchITSTPCQC::getMother(o2::MCTrack const* mcTrk, const std::vector<o2::MCTrack>& pcontainer)
{
  const auto mid = mcTrk->getMotherTrackId();
  if (mid < 0 or mid > pcontainer.size()) {
    return nullptr;
  }
  return &(pcontainer[mid]);
}

o2::MCTrack const* MatchITSTPCQC::getDaughter(o2::MCTrack const* mcTrk, const std::vector<o2::MCTrack>& pcontainer)
{
  const auto did = mcTrk->getFirstDaughterTrackId();
  if (did < 0 or did > pcontainer.size()) {
    return nullptr;
  }
  return &(pcontainer[did]);
}

bool MatchITSTPCQC::isPhysicalPrimary(o2::MCTrack const* mcTrk, const std::vector<o2::MCTrack>& pcontainer)
{
  // Test if a particle is a physical primary according to the following definition:
  // Particles produced in the collision including products of strong and
  // electromagnetic decay and excluding feed-down from weak decays of strange
  // particles.
  //

  const int ist = mcTrk->getStatusCode(); // the generator status code
  const int pdg = std::abs(mcTrk->GetPdgCode());
  //
  // Initial state particle
  // Solution for K0L decayed by Pythia6
  // ->
  // ist > 1 --> essentially means unstable
  if ((ist > 1) && (pdg != 130) && mcTrk->isPrimary()) {
    return false;
  }
  if ((ist > 1) && mcTrk->isSecondary()) {
    return false;
  }
  if (!isStable(pdg)) {
    return false;
  }

  if (mcTrk->isPrimary()) {
    //
    // Particle produced by generator
    // Solution for K0L decayed by Pythia6
    // ->
    const auto ipm = getMother(mcTrk, pcontainer);
    if (ipm != nullptr) {
      if (std::abs(ipm->GetPdgCode()) == 130) {
        return false;
      }
    }
    // <-
    // check for direct photon in parton shower
    // ->
    if (pdg == 22) {
      const auto ipd = getDaughter(mcTrk, pcontainer);
      if (ipd && ipd->GetPdgCode() == 22) {
        return false;
      }
    }
    // <-
    return true;
  } else {
    //
    // Particle produced during transport
    //

    auto* pm = getMother(mcTrk, pcontainer);
    int mpdg = std::abs(pm->GetPdgCode());
    // Check for Sigma0
    if ((mpdg == 3212) && pm->isPrimary()) {
      return true;
    }

    //
    // Check if it comes from a pi0 decay
    //
    if ((mpdg == kPi0) && pm->isPrimary()) {
      return true;
    }

    // Check if this is a heavy flavor decay product
    int mfl = int(mpdg / std::pow(10, int(std::log10(mpdg))));
    //
    // Light hadron
    if (mfl < 4) {
      return false;
    }

    //
    // Heavy flavor hadron produced by generator
    if (pm->isPrimary()) {
      return true;
    }

    // To be sure that heavy flavor has not been produced in a secondary interaction
    // Loop back to the generated mother
    while (!pm->isPrimary()) {
      pm = getMother(pm, pcontainer);
    }
    mpdg = std::abs(pm->GetPdgCode());
    mfl = int(mpdg / std::pow(10, int(std::log10(mpdg))));

    if (mfl < 4) {
      return false;
    } else {
      return true;
    }
  }
}
