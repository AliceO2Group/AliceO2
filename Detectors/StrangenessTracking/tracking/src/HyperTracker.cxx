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

namespace o2
{
namespace strangeness_tracking
{
using ITSCluster = o2::BaseCluster<float>;

bool HyperTracker::loadData(gsl::span<const o2::its::TrackITS> InputITStracks, std::vector<ITSCluster>& InputITSclusters, gsl::span<const int> InputITSidxs, gsl::span<const V0> InputV0tracks, o2::its::GeometryTGeo* geomITS)
{
  mInputV0tracks = InputV0tracks;
  mInputITStracks = InputITStracks;
  mInputITSclusters = InputITSclusters;
  mInputITSidxs = InputITSidxs;
  LOG(INFO) << "all tracks loaded";
  LOG(INFO) << "V0 tracks size: " << mInputV0tracks.size();
  LOG(INFO) << "ITS tracks size: " << mInputITStracks.size();
  LOG(INFO) << "ITS clusters size: " << mInputITSclusters.size();
  LOG(INFO) << "ITS idxs size: " << mInputITSidxs.size();
  mGeomITS = geomITS;
  setupFitters();
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

void HyperTracker::process()
{

  int counter = 0;
  for (auto& v0 : mInputV0tracks) {
    counter++;
    LOG(INFO) << "Analysing V0: " << counter << "/" << mInputV0tracks.size();

    auto posTrack = v0.getProng(0);
    auto negTrack = v0.getProng(1);
    auto alphaV0 = calcV0alpha(v0);
    alphaV0 > 0 ? posTrack.setAbsCharge(2) : negTrack.setAbsCharge(2);
    if (!recreateV0(posTrack, negTrack, v0.getProngID(0), v0.getProngID(1)))
      continue;
    auto tmpV0 = mV0;
    auto v0R2 = v0.calcR2();

    for (int iTrack{0}; iTrack < mInputITStracks.size(); iTrack++) {

      mV0 = tmpV0;
      auto& ITStrack = mInputITStracks[iTrack];

      auto trackClusters = getTrackClusters(ITStrack);
      std::vector<ITSCluster> v0Clusters;
      int nUpdates = 0;
      for (auto& clus : trackClusters) {
        auto isV0Upd = false;
        auto diffR2 = v0R2 - clus.getX() * clus.getX() - clus.getY() * clus.getY(); // difference between V0 and Layer R2
        // check V0 compatibility
        if (diffR2 > -4) {
          if (updateTrack(clus, mV0)) {
            v0Clusters.push_back(clus);
            // LOG(INFO) << "Attach cluster to V0, layer: " << mGeomITS->getLayer(clus.getSensorID());
            isV0Upd = true;
            nUpdates++;
          }
        }
        // if V0 is not found, check He3 compatibility
        if (diffR2 < 4 && !isV0Upd) {
          auto& he3track = calcV0alpha(mV0) > 0 ? mV0.getProng(0) : mV0.getProng(1);
          if (!updateTrack(clus, he3track))
            break;
          if (!recreateV0(mV0.getProng(0), mV0.getProng(1), mV0.getProngID(0), mV0.getProngID(1)))
            break;

          isV0Upd = true;
          nUpdates++;
        }

        if (!isV0Upd)
          break;
      }

      if (nUpdates < trackClusters.size())
        continue;

      o2::track::TrackParCov hyperTrack = mV0;
      // outward V0 propagation
      if (v0Clusters.size() > 0) {
        mV0.resetCovariance();
        std::reverse(v0Clusters.begin(), v0Clusters.end());
        for (auto& clus : v0Clusters) {
          if (!updateTrack(clus, mV0))
            break;
        }

        // final 3body refit
        if (refitAllTracks()) {
          LOG(INFO) << "Pushing back v0: " << v0.getProngID(0) << ", " << v0.getProngID(1);
          auto& lastClus = trackClusters[0];

          mV0s.push_back(mV0);
          mHyperTracks.push_back(hyperTrack);
          mChi2.push_back(getMatchingChi2(tmpV0, ITStrack, lastClus));
          mITStrackRef.push_back(iTrack);
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
  mV0.setPID(o2::track::PID::HyperTriton);

  return true;
}

bool HyperTracker::refitAllTracks()
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