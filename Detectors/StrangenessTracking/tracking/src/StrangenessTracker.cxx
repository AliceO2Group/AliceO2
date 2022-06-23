// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
// StrangenessTracker
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
/// \file StrangenessTracker.cxx
/// \brief

#include <numeric>
#include "StrangenessTracking/StrangenessTracker.h"

namespace o2
{
namespace strangeness_tracking
{

bool StrangenessTracker::loadData(gsl::span<const o2::its::TrackITS> InputITStracks, std::vector<ITSCluster>& InputITSclusters, gsl::span<const int> InputITSidxs, gsl::span<const V0> InputV0tracks, gsl::span<const Cascade> InputCascadeTracks, o2::its::GeometryTGeo* geomITS)
{
  mInputV0tracks = InputV0tracks;
  mInputCascadeTracks = InputCascadeTracks;
  mInputITStracks = InputITStracks;
  mInputITSclusters = InputITSclusters;
  mInputITSidxs = InputITSidxs;
  LOG(info) << "all tracks loaded";
  LOG(info) << "V0 tracks size: " << mInputV0tracks.size();
  LOG(info) << "V0 tracks size: " << mInputCascadeTracks.size();
  LOG(info) << "ITS tracks size: " << mInputITStracks.size();
  LOG(info) << "ITS clusters size: " << mInputITSclusters.size();
  LOG(info) << "ITS idxs size: " << mInputITSidxs.size();
  mGeomITS = geomITS;
  setupFitters();
  return true;
}

void StrangenessTracker::initialise()
{
  mTracksIdxTable.clear();

  for (int iTrack{0}; iTrack < mInputITStracks.size(); iTrack++) {
    mSortedITStracks.push_back(mInputITStracks[iTrack]);
    mSortedITSindexes.push_back(iTrack);
  }

  mTracksIdxTable.resize(mUtils.mPhiBins * mUtils.mEtaBins + 1);
  std::sort(mSortedITStracks.begin(), mSortedITStracks.end(), [&](o2::its::TrackITS& a, o2::its::TrackITS& b) { return mUtils.getBinIndex(a.getEta(), a.getPhi()) < mUtils.getBinIndex(b.getEta(), b.getPhi()); });
  std::sort(mSortedITSindexes.begin(), mSortedITSindexes.end(), [&](int i, int j) { return mUtils.getBinIndex(mInputITStracks[i].getEta(), mInputITStracks[i].getPhi()) < mUtils.getBinIndex(mInputITStracks[j].getEta(), mInputITStracks[j].getPhi()); });

  for (auto& track : mSortedITStracks) {
    mTracksIdxTable[mUtils.getBinIndex(track.getEta(), track.getPhi())]++;
  }
  std::exclusive_scan(mTracksIdxTable.begin(), mTracksIdxTable.begin() + mUtils.mPhiBins * mUtils.mEtaBins, mTracksIdxTable.begin(), 0);
  mTracksIdxTable[mUtils.mPhiBins * mUtils.mEtaBins] = mSortedITStracks.size();
}

void StrangenessTracker::process()
{

  int counter = 0;
  for (auto& v0 : mInputV0tracks) {
    counter++;
    LOG(debug) << "Analysing V0: " << counter << "/" << mInputV0tracks.size();

    auto posTrack = v0.getProng(0);
    auto negTrack = v0.getProng(1);
    auto alphaV0 = calcV0alpha(v0);
    alphaV0 > 0 ? posTrack.setAbsCharge(2) : negTrack.setAbsCharge(2);
    V0 correctedV0; // recompute V0 for Hypertriton

    if (!recreateV0(posTrack, negTrack, v0.getProngID(0), v0.getProngID(1), correctedV0))
      continue;

    auto v0R2 = v0.calcR2();
    auto iBinsV0 = mUtils.getBinRect(correctedV0.getEta(), correctedV0.getPhi(), 0.1, 0.1);
    for (int& iBinV0 : iBinsV0) {
      LOG(debug) << "iBinV0: " << iBinV0;
      for (int iTrack{mTracksIdxTable[iBinV0]}; iTrack < TMath::Min(mTracksIdxTable[iBinV0 + 1], int(mSortedITStracks.size())); iTrack++) {
        auto trackClusters = getTrackClusters();
        auto& lastClus = trackClusters[0];

        mStrangeTrack.mMother = (o2::track::TrackParCovF)correctedV0;
        mStrangeTrack.mDaughterFirst = alphaV0 > 0 ? correctedV0.getProng(0) : correctedV0.getProng(1);
        mStrangeTrack.mDaughterSecond = alphaV0 < 0 ? correctedV0.getProng(0) : correctedV0.getProng(1);
        mStrangeTrack.mMatchChi2 = getMatchingChi2(correctedV0, mITStrack, lastClus);

        mITStrack = mSortedITStracks[iTrack];
        auto& ITSindexRef = mSortedITSindexes[iTrack];

        LOG(debug) << "V0 pos: " << correctedV0.getProngID(0) << " V0 neg: " << correctedV0.getProngID(1) << " V0pt: " << correctedV0.getPt() << " ITSpt: " << mITStrack.getPt();
        LOG(debug) << "V0 eta: " << correctedV0.getEta() << " V0 phi: " << correctedV0.getPhi() << " ITS eta: " << mITStrack.getEta() << " ITS phi: " << mITStrack.getPhi();

        std::vector<ITSCluster> motherClusters;
        std::array<unsigned int, 7> nAttachments;

        updateTopology(nAttachments, motherClusters, trackClusters, v0R2, false);

        o2::track::TrackParCov motherTrackClone = mStrangeTrack.mMother; // clone and reset covariance
        motherTrackClone.resetCovariance();

        if (motherClusters.size() >= mMinMotherClus) { // fill only if at least mMinMotherClus clusters of the mother V0 have been attached
          std::reverse(motherClusters.begin(), motherClusters.end());
          for (auto& clus : motherClusters) {
            if (!updateTrack(clus, motherTrackClone))
              break;
          }

          // final 3body refit
          if (refitTopology(motherTrackClone)) {
            LOG(debug) << "------------------------------------------------------";
            LOG(debug) << "Pushing back v0: " << v0.getProngID(0) << ", " << v0.getProngID(1);
            LOG(debug) << "number of clusters attached: " << motherClusters.size();
            LOG(debug) << "Number of ITS track clusters: " << mITStrack.getNumberOfClusters();
            LOG(debug) << "number of clusters attached to V0: " << nAttachments[0] << ", " << nAttachments[1] << ", " << nAttachments[2] << ", " << nAttachments[3] << ", " << nAttachments[4] << ", " << nAttachments[5] << ", " << nAttachments[6];
            auto& lastClus = trackClusters[0];
            LOG(debug) << "Matching chi2: " << getMatchingChi2(correctedV0, mITStrack, lastClus);

            mStrangeTrackVec.push_back(mStrangeTrack);
            mITStrackRefVec.push_back(ITSindexRef);
            ClusAttachments structClus;
            structClus.arr = nAttachments;
            mClusAttachments.push_back(structClus);
          }
        }
      }
    }
  }
}

bool StrangenessTracker::updateTopology(std::array<unsigned int, 7>& nAttachments, std::vector<ITSCluster>& motherClusters, const std::vector<ITSCluster>& trackClusters, float decayRadius, bool isCascade)
{
  int nUpdates = 0;
  bool isMotherUpdated = false;

  for (auto& clus : trackClusters) {
    auto diffR2 = decayRadius - clus.getX() * clus.getX() - clus.getY() * clus.getY(); // difference between decay radius and Layer R2
    if (diffR2 > -mRadiusTol) {
      LOG(debug) << "Try to attach cluster to Mother, layer: " << mGeomITS->getLayer(clus.getSensorID());
      if (updateTrack(clus, mStrangeTrack.mMother)) {
        motherClusters.push_back(clus);
        nAttachments[mGeomITS->getLayer(clus.getSensorID())] = kMother;
        isMotherUpdated = true;
        nUpdates++;
        continue;
      } else {
        if (isMotherUpdated == true) {
          break; // daughter clusters cannot be attached now
        }
      }
    }

    // if Mother is not found, check for V0 daughters compatibility
    if (diffR2 < mRadiusTol && !isMotherUpdated) {
      LOG(debug) << "Try to attach cluster to Daughters, layer: " << mGeomITS->getLayer(clus.getSensorID());

      if (!isCascade) {
        if (updateTrack(clus, mStrangeTrack.mDaughterFirst))
          nAttachments[mGeomITS->getLayer(clus.getSensorID())] = kFirstDaughter;
        else if (updateTrack(clus, mStrangeTrack.mDaughterSecond))
          nAttachments[mGeomITS->getLayer(clus.getSensorID())] = kSecondDaughter;
        else
          break;
      }
      // if Mother is not found, check for cascade bachelor compatibility
      else {
        if (updateTrack(clus, mStrangeTrack.mBachelor))
          nAttachments[mGeomITS->getLayer(clus.getSensorID())] = kBachelor;
        else
          break;
      }
      nUpdates++;
    }
  }
  if (nUpdates < trackClusters.size() || motherClusters.size() == 0)
    return false;

  return true;
}

bool StrangenessTracker::updateTrack(const ITSCluster& clus, o2::track::TrackParCov& track)
{
  auto propInstance = o2::base::Propagator::Instance();
  float alpha = mGeomITS->getSensorRefAlpha(clus.getSensorID()), x = clus.getX();
  int layer{mGeomITS->getLayer(clus.getSensorID())};

  if (!track.rotate(alpha))
    return false;

  if (!propInstance->propagateToX(track, x, getBz(), o2::base::PropagatorImpl<float>::MAX_SIN_PHI, o2::base::PropagatorImpl<float>::MAX_STEP, mCorrType))
    return false;
  if (mCorrType == o2::base::PropagatorF::MatCorrType::USEMatCorrNONE) {
    float thick = layer < 3 ? 0.005 : 0.01;
    constexpr float radl = 9.36f; // Radiation length of Si [cm]
    constexpr float rho = 2.33f;  // Density of Si [g/cm^3]
    if (!track.correctForMaterial(thick, thick * rho * radl))
      return false;
  }
  auto chi2 = std::abs(track.getPredictedChi2(clus));
  LOG(debug) << "chi2: " << track.getPredictedChi2(clus);
  if (chi2 > mMaxChi2 || chi2 < 0)
    return false;

  if (!track.update(clus))
    return false;

  return true;
}

bool StrangenessTracker::recreateV0(const o2::track::TrackParCov& posTrack, const o2::track::TrackParCov& negTrack, const GIndex posID, const GIndex negID, V0& newV0)
{

  int nCand;
  try {
    nCand = mFitterV0.process(posTrack, negTrack);
  } catch (std::runtime_error& e) {
    return false;
  }
  if (!nCand)
    return false;

  mFitterV0.propagateTracksToVertex();
  const auto& v0XYZ = mFitterV0.getPCACandidatePos();

  auto& propPos = mFitterV0.getTrack(0, 0);
  auto& propNeg = mFitterV0.getTrack(1, 0);

  std::array<float, 3> pP, pN;
  propPos.getPxPyPzGlo(pP);
  propNeg.getPxPyPzGlo(pN);
  std::array<float, 3> pV0 = {pP[0] + pN[0], pP[1] + pN[1], pP[2] + pN[2]};
  newV0 = V0(v0XYZ, pV0, mFitterV0.calcPCACovMatrixFlat(0), propPos, propNeg, posID, negID, o2::track::PID::HyperTriton);
  return true;
};

std::vector<o2::strangeness_tracking::StrangenessTracker::ITSCluster> StrangenessTracker::getTrackClusters()
{
  std::vector<ITSCluster> outVec;
  auto firstClus = mITStrack.getFirstClusterEntry();
  auto ncl = mITStrack.getNumberOfClusters();
  for (int icl = 0; icl < ncl; icl++) {
    outVec.push_back(mInputITSclusters[mInputITSidxs[firstClus + icl]]);
  }
  return outVec;
};

bool StrangenessTracker::refitTopology(o2::track::TrackParCovF& ITSmotherTrack)
{
  int cand = 0; // best V0 candidate
  int nCand;

  try {
    nCand = mFitter3Body.process(ITSmotherTrack, mStrangeTrack.mDaughterFirst, mStrangeTrack.mDaughterSecond);
  } catch (std::runtime_error& e) {
    return false;
  }
  if (!nCand)
    return false;

  mFitter3Body.propagateTracksToVertex();

  mStrangeTrack.mDaughterFirst = mFitter3Body.getTrack(1, 0);
  mStrangeTrack.mDaughterSecond = mFitter3Body.getTrack(2, 0);
  mStrangeTrack.decayVtx = mFitter3Body.getPCACandidatePos();
  mStrangeTrack.mTopoChi2 = mFitter3Body.getChi2AtPCACandidate();

  return true;
};

float StrangenessTracker::getMatchingChi2(V0 v0, const TrackITS ITStrack, ITSCluster matchingClus)
{
  float alpha = mGeomITS->getSensorRefAlpha(matchingClus.getSensorID()), x = matchingClus.getX();
  if (v0.rotate(alpha)) {
    if (v0.propagateTo(x, mBz)) {
      return v0.getPredictedChi2(ITStrack.getParamOut());
    }
  }
  return -100;
};

double StrangenessTracker::calcV0alpha(const V0& v0)
{
  std::array<float, 3> fV0mom, fPmom, fNmom = {0, 0, 0};
  v0.getProng(0).getPxPyPzGlo(fPmom);
  v0.getProng(1).getPxPyPzGlo(fNmom);
  v0.getPxPyPzGlo(fV0mom);

  TVector3 momNeg(fNmom[0], fNmom[1], fNmom[2]);
  TVector3 momPos(fPmom[0], fPmom[1], fPmom[2]);
  TVector3 momTot(fV0mom[0], fV0mom[1], fV0mom[2]);

  Double_t lQlNeg = momNeg.Dot(momTot) / momTot.Mag();
  Double_t lQlPos = momPos.Dot(momTot) / momTot.Mag();

  return (lQlPos - lQlNeg) / (lQlPos + lQlNeg);
};

} // namespace strangeness_tracking
} // namespace o2