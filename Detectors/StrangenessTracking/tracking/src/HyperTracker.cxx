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
/// \file HyperTracker.cxx
/// \brief

#include "StrangenessTracking/HyperTracker.h"
#include <numeric>

namespace o2
{
namespace strangeness_tracking
{
using ITSCluster = o2::BaseCluster<float>;

int indexTableUtils::getBinIndex(float eta, float phi)
{
  float deltaPhi = 2 * TMath::Pi() / (mPhiBins);
  float deltaEta = (maxEta - minEta) / (mEtaBins);
  int bEta = (eta + (maxEta - minEta) / 2) / deltaEta;
  int bPhi = phi / deltaPhi;
  return abs(eta) > 1.5 ? mEtaBins * mPhiBins : bEta + mEtaBins * bPhi;
}

bool HyperTracker::loadData(gsl::span<const o2::its::TrackITS> InputITStracks, std::vector<ITSCluster>& InputITSclusters, gsl::span<const int> InputITSidxs, gsl::span<const V0> InputV0tracks, o2::its::GeometryTGeo* geomITS)
{
  mInputV0tracks = InputV0tracks;
  mInputITStracks = InputITStracks;
  mInputITSclusters = InputITSclusters;
  mInputITSidxs = InputITSidxs;
  LOG(info) << "all tracks loaded";
  LOG(info) << "V0 tracks size: " << mInputV0tracks.size();
  LOG(info) << "ITS tracks size: " << mInputITStracks.size();
  LOG(info) << "ITS clusters size: " << mInputITSclusters.size();
  LOG(info) << "ITS idxs size: " << mInputITSidxs.size();
  mGeomITS = geomITS;
  setupFitters();
  return true;
}

void HyperTracker::initialise()
{
  mTracksIdxTable.clear();

  for (auto& track : mInputITStracks) {
    mSortedITStracks.push_back(track);
  }

  mTracksIdxTable.resize(mUtils.mPhiBins * mUtils.mEtaBins + 1);
  std::sort(mSortedITStracks.begin(), mSortedITStracks.end(), [&](o2::its::TrackITS& a, o2::its::TrackITS& b) { return mUtils.getBinIndex(a.getEta(), a.getPhi()) < mUtils.getBinIndex(b.getEta(), b.getPhi()); });
  for (auto& track : mSortedITStracks) {
    mTracksIdxTable[mUtils.getBinIndex(track.getEta(), track.getPhi())]++;
  }
  std::exclusive_scan(mTracksIdxTable.begin(), mTracksIdxTable.begin() + mUtils.mPhiBins * mUtils.mEtaBins, mTracksIdxTable.begin(), 0);
  mTracksIdxTable[mUtils.mPhiBins * mUtils.mEtaBins] = mSortedITStracks.size();
  for (int iPhi{0}; iPhi < mUtils.mPhiBins; ++iPhi) {
    for (int iEta{0}; iEta < mUtils.mEtaBins; ++iEta) {
      std::cout << mTracksIdxTable[iEta + iPhi * mUtils.mEtaBins] << "\t";
    }
    std::cout << std::endl;
  }
}

void HyperTracker::process()
{

  int counter = 0;
  for (auto& v0 : mInputV0tracks) {
    counter++;
    LOG(info) << "Analysing V0: " << counter << "/" << mInputV0tracks.size();

    auto posTrack = v0.getProng(0);
    auto negTrack = v0.getProng(1);
    auto alphaV0 = calcV0alpha(v0);

    alphaV0 > 0 ? posTrack.setAbsCharge(2) : negTrack.setAbsCharge(2);
    if (!recreateV0(posTrack, negTrack, v0.getProngID(0), v0.getProngID(1)))
      continue;

    auto tmpV0 = mV0;
    auto v0R2 = v0.calcR2();

    int ibinV0 = mUtils.getBinIndex(tmpV0.getEta(), tmpV0.getPhi());
    // LOG(info) << "V0eta: " << v0.getEta() << " V0phi: " << v0.getPhi() << " V0r2: " << v0R2 << " ibinV0: " << ibinV0;
    LOG(info) << mTracksIdxTable[ibinV0] << " " << mTracksIdxTable[ibinV0 + 1];

    for (int iTrack{mTracksIdxTable[ibinV0]}; iTrack < TMath::Min(mTracksIdxTable[ibinV0 + 1], int(mSortedITStracks.size())); iTrack++) {

      mV0 = tmpV0;
      auto& he3Track = alphaV0 > 0 ? mV0.getProng(0) : mV0.getProng(1);
      auto& piTrack = alphaV0 < 0 ? mV0.getProng(0) : mV0.getProng(1);
      auto& ITStrack = mSortedITStracks[iTrack];

      // LOG(info) << "itrack: " << iTrack <<" V0 eta: " << tmpV0.getEta() << " phi: " << tmpV0.getPhi() << ", ITS eta: " << ITStrack.getEta() << " phi: " << ITStrack.getPhi();

      auto trackClusters = getTrackClusters(ITStrack);
      std::vector<ITSCluster> motherClusters;
      std::array<unsigned int, 7> nAttachments;

      int nUpdates = 0;
      bool isMotherUpdated = false;

      for (auto& clus : trackClusters) {
        auto diffR2 = v0R2 - clus.getX() * clus.getX() - clus.getY() * clus.getY(); // difference between V0 and Layer R2
        if (diffR2 > -mRadiusTol) {
          // LOG(info) << "Try to attach cluster to V0, layer: " << mGeomITS->getLayer(clus.getSensorID());
          if (updateTrack(clus, mV0)) {
            motherClusters.push_back(clus);
            nAttachments[mGeomITS->getLayer(clus.getSensorID())] = 1;
            isMotherUpdated = true;
            nUpdates++;
            continue;
          } else {
            if (isMotherUpdated == true) {
              break;
            } // no daughter clusters can be attached
          }
        }

        // if V0 is not found, check for daughters compatibility
        if (diffR2 < mRadiusTol && !isMotherUpdated) {
          // LOG(info) << "Try to attach cluster to daughters, layer: " << mGeomITS->getLayer(clus.getSensorID());
          if (updateTrack(clus, he3Track))
            nAttachments[mGeomITS->getLayer(clus.getSensorID())] = kFirstDaughter;
          else if (updateTrack(clus, piTrack))
            nAttachments[mGeomITS->getLayer(clus.getSensorID())] = kSecondDaughter;
          else
            break;
          nUpdates++;
        }
      }

      if (nUpdates < trackClusters.size() || motherClusters.size() == 0)
        continue;

      o2::track::TrackParCov hyperTrack = mV0;
      mV0.resetCovariance();
      std::reverse(motherClusters.begin(), motherClusters.end());
      for (auto& clus : motherClusters) {
        if (!updateTrack(clus, mV0))
          break;

        // final 3body refit
        if (refitTopology()) {
          LOG(debug) << "------------------------------------------------------";
          LOG(debug) << "Pushing back v0: " << v0.getProngID(0) << ", " << v0.getProngID(1);
          LOG(debug) << "number of clusters attached: " << motherClusters.size();
          LOG(debug) << "Number of ITS track clusters: " << ITStrack.getNumberOfClusters();
          LOG(debug) << "number of clusters attached to V0: " << nAttachments[0] << ", " << nAttachments[1] << ", " << nAttachments[2] << ", " << nAttachments[3] << ", " << nAttachments[4] << ", " << nAttachments[5] << ", " << nAttachments[6];
          auto& lastClus = trackClusters[0];
          LOG(debug) << "Matching chi2: " << getMatchingChi2(tmpV0, ITStrack, lastClus);

          mV0s.push_back(mV0);
          mHyperTracks.push_back(hyperTrack);
          mChi2.push_back(getMatchingChi2(tmpV0, ITStrack, lastClus));
          mR2.push_back(mV0.calcR2());
          mITStrackRef.push_back(iTrack);
          ClusAttachments structClus;
          structClus.arr = nAttachments;
          mClusAttachments.push_back(structClus);
        }
      }
    }
  }
}

bool HyperTracker::updateTrack(const ITSCluster& clus, o2::track::TrackParCov& track)
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
  // LOG(info) << "chi2" << track.getPredictedChi2(clus);
  if (chi2 > mMaxChi2 || chi2 < 0)
    return false;

  if (!track.update(clus))
    return false;

  return true;
}

bool HyperTracker::recreateV0(const o2::track::TrackParCov& posTrack, const o2::track::TrackParCov& negTrack, const GIndex posID, const GIndex negID)
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

  mV0 = V0(v0XYZ, pV0, mFitterV0.calcPCACovMatrixFlat(0), propPos, propNeg, posID, negID, o2::track::PID::HyperTriton);
  mV0.setAbsCharge(1);
  return true;
}

double HyperTracker::calcV0alpha(const V0& v0)
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
}

std::vector<o2::strangeness_tracking::HyperTracker::ITSCluster> HyperTracker::getTrackClusters(o2::its::TrackITS const& ITStrack)
{
  std::vector<ITSCluster> outVec;
  auto firstClus = ITStrack.getFirstClusterEntry();
  auto ncl = ITStrack.getNumberOfClusters();
  for (int icl = 0; icl < ncl; icl++) {
    outVec.push_back(mInputITSclusters[mInputITSidxs[firstClus + icl]]);
  }
  return outVec;
}

bool HyperTracker::refitTopology()
{
  int cand = 0; // best V0 candidate
  int nCand;

  try {
    nCand = mFitter3Body.process(mV0, mV0.getProng(0), mV0.getProng(1));
  } catch (std::runtime_error& e) {
    return false;
  }
  if (!nCand)
    return false;

  mFitter3Body.propagateTracksToVertex();
  auto& propPos = mFitter3Body.getTrack(1, 0);
  auto& propNeg = mFitter3Body.getTrack(2, 0);

  const auto& v0XYZ = mFitter3Body.getPCACandidatePos();
  std::array<float, 3> pP, pN;
  propPos.getPxPyPzGlo(pP);
  propNeg.getPxPyPzGlo(pN);
  std::array<float, 3> pV0 = {pP[0] + pN[0], pP[1] + pN[1], pP[2] + pN[2]};

  mV0 = V0(v0XYZ, pV0, mFitter3Body.calcPCACovMatrixFlat(cand), propPos, propNeg, mV0.getProngID(0), mV0.getProngID(1), o2::track::PID::HyperTriton);
  mV0.setAbsCharge(1);
  return true;
}

float HyperTracker::getMatchingChi2(V0 v0, const TrackITS ITStrack, ITSCluster matchingClus)
{
  float alpha = mGeomITS->getSensorRefAlpha(matchingClus.getSensorID()), x = matchingClus.getX();
  if (v0.rotate(alpha)) {
    if (v0.propagateTo(x, mBz)) {
      return v0.getPredictedChi2(ITStrack.getParamOut());
    }
  }
  return -100;
}

} // namespace strangeness_tracking
} // namespace o2