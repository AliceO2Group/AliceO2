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
/// \file StrangenessTracker.cxx
/// \brief

#include <numeric>
#include "StrangenessTracking/StrangenessTracker.h"
#include "ITStracking/IOUtils.h"

namespace o2
{
namespace strangeness_tracking
{

bool StrangenessTracker::loadData(const o2::globaltracking::RecoContainer& recoData)
{
  clear();

  mInputV0tracks = recoData.getV0s();
  mInputV0Indices = recoData.getV0sIdx();
  mInputCascadeTracks = recoData.getCascades();
  mInputCascadeIndices = recoData.getCascadesIdx();
  if (mInputV0Indices.size() != mInputV0tracks.size() || mInputCascadeIndices.size() != mInputCascadeTracks.size()) {
    LOGP(fatal, "Mismatch between input SVertices indices and kinematics (not requested?): V0: {}/{} Cascades: {}/{}",
         mInputV0Indices.size(), mInputV0tracks.size(), mInputCascadeIndices.size(), mInputCascadeTracks.size());
  }
  mInputITStracks = recoData.getITSTracks();
  mInputITSidxs = recoData.getITSTracksClusterRefs();

  auto compClus = recoData.getITSClusters();
  auto clusPatt = recoData.getITSClustersPatterns();
  auto pattIt = clusPatt.begin();
  mInputITSclusters.reserve(compClus.size());
  mInputClusterSizes.resize(compClus.size());
  o2::its::ioutils::convertCompactClusters(compClus, pattIt, mInputITSclusters, mDict);
  auto pattIt2 = clusPatt.begin();
  getClusterSizes(mInputClusterSizes, compClus, pattIt2, mDict);

  mITSvtxBrackets.resize(mInputITStracks.size());
  for (int i = 0; i < mInputITStracks.size(); i++) {
    mITSvtxBrackets[i] = {-1, -1};
  }

  // build time bracket for each ITS track
  auto trackIndex = recoData.getPrimaryVertexMatchedTracks(); // Global ID's for associated tracks
  auto vtxRefs = recoData.getPrimaryVertexMatchedTrackRefs(); // references from vertex to these track IDs

  if (mStrParams->mVertexMatching) {
    int nv = vtxRefs.size();
    for (int iv = 0; iv < nv; iv++) {
      const auto& vtref = vtxRefs[iv];
      int it = vtref.getFirstEntry(), itLim = it + vtref.getEntries();
      for (; it < itLim; it++) {
        auto tvid = trackIndex[it];
        if (!recoData.isTrackSourceLoaded(tvid.getSource()) || tvid.getSource() != GIndex::ITS) {
          continue;
        }
        if (mITSvtxBrackets[tvid.getIndex()].getMin() == -1) {
          mITSvtxBrackets[tvid.getIndex()].setMin(iv);
          mITSvtxBrackets[tvid.getIndex()].setMax(iv);
        } else {
          mITSvtxBrackets[tvid.getIndex()].setMax(iv);
        }
      }
    }
  }

  if (mMCTruthON) {
    mITSClsLabels = recoData.mcITSClusters.get();
    mITSTrkLabels = recoData.getITSTracksMCLabels();
  }

  LOG(debug) << "V0 tracks size: " << mInputV0tracks.size();
  LOG(debug) << "Cascade tracks size: " << mInputCascadeTracks.size();
  LOG(debug) << "ITS tracks size: " << mInputITStracks.size();
  LOG(debug) << "ITS idxs size: " << mInputITSidxs.size();
  LOG(debug) << "ITS clusters size: " << mInputITSclusters.size();
  LOG(debug) << "VtxRefs size: " << vtxRefs.size();

  return true;
}

void StrangenessTracker::prepareITStracks() // sort tracks by eta and phi and select only tracks with vertex matching
{

  for (int iTrack{0}; iTrack < mInputITStracks.size(); iTrack++) {
    if (mStrParams->mVertexMatching && mITSvtxBrackets[iTrack].getMin() == -1) {
      continue;
    }
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

void StrangenessTracker::processV0(int iv0, const V0& v0, const V0Index& v0Idx, int iThread)
{
  ClusAttachments structClus;
  auto& daughterTracks = mDaughterTracks[iThread];
  daughterTracks.resize(2); // resize to 2 prongs: first positive second negative
  auto posTrack = v0.getProng(kV0DauPos);
  auto negTrack = v0.getProng(kV0DauNeg);
  auto alphaV0 = calcV0alpha(v0);
  alphaV0 > 0 ? posTrack.setAbsCharge(2) : negTrack.setAbsCharge(2);
  V0 correctedV0; // recompute V0 for Hypertriton
  if (!recreateV0(posTrack, negTrack, correctedV0, iThread)) {
    return;
  }
  StrangeTrack strangeTrack;
  strangeTrack.mPartType = dataformats::kStrkV0;
  auto v0R = std::sqrt(v0.calcR2());
  auto iBinsV0 = mUtils.getBinRect(correctedV0.getEta(), correctedV0.getPhi(), mStrParams->mEtaBinSize, mStrParams->mPhiBinSize);
  for (int& iBinV0 : iBinsV0) {
    for (int iTrack{mTracksIdxTable[iBinV0]}; iTrack < TMath::Min(mTracksIdxTable[iBinV0 + 1], int(mSortedITStracks.size())); iTrack++) {
      strangeTrack.mMother = (o2::track::TrackParCovF)correctedV0;
      daughterTracks[kV0DauPos] = correctedV0.getProng(kV0DauPos);
      daughterTracks[kV0DauNeg] = correctedV0.getProng(kV0DauNeg);
      const auto& itsTrack = mSortedITStracks[iTrack];
      const auto& ITSindexRef = mSortedITSindexes[iTrack];
      if (mStrParams->mVertexMatching && (mITSvtxBrackets[ITSindexRef].getMin() > v0Idx.getVertexID() || mITSvtxBrackets[ITSindexRef].getMax() < v0Idx.getVertexID())) {
        continue;
      }
      if (matchDecayToITStrack(v0R, strangeTrack, structClus, itsTrack, daughterTracks, iThread)) {
        auto propInstance = o2::base::Propagator::Instance();
        o2::track::TrackParCov decayVtxTrackClone = strangeTrack.mMother; // clone track and propagate to decay vertex
        if (!propInstance->propagateToX(decayVtxTrackClone, strangeTrack.mDecayVtx[0], getBz(), o2::base::PropagatorImpl<float>::MAX_SIN_PHI, o2::base::PropagatorImpl<float>::MAX_STEP, mCorrType)) {
          LOG(debug) << "Mother propagation to decay vertex failed";
          continue;
        }
        decayVtxTrackClone.getPxPyPzGlo(strangeTrack.mDecayMom);
        std::array<float, 3> momPos, momNeg;
        mFitter3Body[iThread].getTrack(kV0DauPos).getPxPyPzGlo(momPos);
        mFitter3Body[iThread].getTrack(kV0DauNeg).getPxPyPzGlo(momNeg);
        if (alphaV0 > 0) {
          strangeTrack.mMasses[0] = calcMotherMass(momPos, momNeg, PID::Helium3, PID::Pion); // Hypertriton invariant mass at decay vertex
          strangeTrack.mMasses[1] = calcMotherMass(momPos, momNeg, PID::Alpha, PID::Pion);   // Hyperhydrogen4Lam invariant mass at decay vertex
        } else {
          strangeTrack.mMasses[0] = calcMotherMass(momPos, momNeg, PID::Pion, PID::Helium3); // Anti-Hypertriton invariant mass at decay vertex
          strangeTrack.mMasses[1] = calcMotherMass(momPos, momNeg, PID::Pion, PID::Alpha);   // Anti-Hyperhydrogen4Lam invariant mass at decay vertex
        }

        LOG(debug) << "ITS Track matched with a V0 decay topology ....";
        LOG(debug) << "Number of ITS track clusters attached: " << itsTrack.getNumberOfClusters();
        strangeTrack.mDecayRef = iv0;
        strangeTrack.mITSRef = mSortedITSindexes[iTrack];
        mStrangeTrackVec[iThread].push_back(strangeTrack);
        mClusAttachments[iThread].push_back(structClus);
        if (mMCTruthON) {
          auto lab = getStrangeTrackLabel(itsTrack, strangeTrack, structClus);
          mStrangeTrackLabels[iThread].push_back(lab);
        }
      }
    }
  }
}

void StrangenessTracker::processCascade(int iCasc, const Cascade& casc, const CascadeIndex& cascIdx, const V0& cascV0, int iThread)
{
  ClusAttachments structClus;
  auto& daughterTracks = mDaughterTracks[iThread];
  daughterTracks.resize(3); // resize to 3 prongs: first bachelor, second V0 pos, third V0 neg

  StrangeTrack strangeTrack;
  strangeTrack.mPartType = dataformats::kStrkCascade;
  // first: bachelor, second: V0 pos, third: V0 neg
  auto cascR = std::sqrt(casc.calcR2());
  auto iBinsCasc = mUtils.getBinRect(casc.getEta(), casc.getPhi(), mStrParams->mEtaBinSize, mStrParams->mPhiBinSize);
  for (int& iBinCasc : iBinsCasc) {
    for (int iTrack{mTracksIdxTable[iBinCasc]}; iTrack < TMath::Min(mTracksIdxTable[iBinCasc + 1], int(mSortedITStracks.size())); iTrack++) {
      strangeTrack.mMother = (o2::track::TrackParCovF)casc;
      daughterTracks[kV0DauPos] = cascV0.getProng(kV0DauPos);
      daughterTracks[kV0DauNeg] = cascV0.getProng(kV0DauNeg);
      daughterTracks[kBach] = casc.getBachelorTrack();
      const auto& itsTrack = mSortedITStracks[iTrack];
      const auto& ITSindexRef = mSortedITSindexes[iTrack];
      LOG(debug) << "----------------------";
      LOG(debug) << "CascV0: " << cascIdx.getV0ID() << ", Bach ID: " << cascIdx.getBachelorID() << ", ITS track ref: " << mSortedITSindexes[iTrack];
      if (mStrParams->mVertexMatching && (mITSvtxBrackets[ITSindexRef].getMin() > cascIdx.getVertexID() || mITSvtxBrackets[ITSindexRef].getMax() < cascIdx.getVertexID())) {
        LOG(debug) << "Vertex ID mismatch: " << mITSvtxBrackets[ITSindexRef].getMin() << " < " << cascIdx.getVertexID() << " < " << mITSvtxBrackets[ITSindexRef].getMax();
        continue;
      }
      if (matchDecayToITStrack(cascR, strangeTrack, structClus, itsTrack, daughterTracks, iThread)) {
        auto propInstance = o2::base::Propagator::Instance();
        o2::track::TrackParCov decayVtxTrackClone = strangeTrack.mMother; // clone track and propagate to decay vertex
        if (!propInstance->propagateToX(decayVtxTrackClone, strangeTrack.mDecayVtx[0], getBz(), o2::base::PropagatorImpl<float>::MAX_SIN_PHI, o2::base::PropagatorImpl<float>::MAX_STEP, mCorrType)) {
          LOG(debug) << "Mother propagation to decay vertex failed";
          continue;
        }
        decayVtxTrackClone.getPxPyPzGlo(strangeTrack.mDecayMom);
        std::array<float, 3> momV0, mombach;
        mFitter3Body[iThread].getTrack(0).getPxPyPzGlo(momV0);                             // V0 momentum at decay vertex
        mFitter3Body[iThread].getTrack(1).getPxPyPzGlo(mombach);                           // bachelor momentum at decay vertex
        strangeTrack.mMasses[0] = calcMotherMass(momV0, mombach, PID::Lambda, PID::Pion);  // Xi invariant mass at decay vertex
        strangeTrack.mMasses[1] = calcMotherMass(momV0, mombach, PID::Lambda, PID::Kaon);  // Omega invariant mass at decay vertex

        LOG(debug) << "ITS Track matched with a Cascade decay topology ....";
        LOG(debug) << "Number of ITS track clusters attached: " << itsTrack.getNumberOfClusters();

        strangeTrack.mDecayRef = iCasc;
        strangeTrack.mITSRef = mSortedITSindexes[iTrack];
        mStrangeTrackVec[iThread].push_back(strangeTrack);
        mClusAttachments[iThread].push_back(mStructClus);
        if (mMCTruthON) {
          auto lab = getStrangeTrackLabel(itsTrack, strangeTrack, structClus);
          mStrangeTrackLabels[iThread].push_back(lab);
        }
      }
    }
  }
}

void StrangenessTracker::process()
{
  // Loop over V0s
  for (int iV0{0}; iV0 < mInputV0tracks.size(); iV0++) {
    LOG(debug) << "Analysing V0: " << iV0 + 1 << "/" << mInputV0tracks.size();
    processV0(iV0, mInputV0tracks[iV0], mInputV0Indices[iV0]);
  }

  // Loop over Cascades
  for (int iCasc{0}; iCasc < mInputCascadeTracks.size(); iCasc++) {
    LOG(debug) << "Analysing Cascade: " << iCasc + 1 << "/" << mInputCascadeTracks.size();
    processCascade(iCasc, mInputCascadeTracks[iCasc], mInputCascadeIndices[iCasc], mInputV0tracks[mInputCascadeIndices[iCasc].getV0ID()]);
  }
}

bool StrangenessTracker::matchDecayToITStrack(float decayR, StrangeTrack& strangeTrack, ClusAttachments& structClus, const TrackITS& itsTrack, std::vector<o2::track::TrackParCovF>& daughterTracks, int iThread)
{
  auto geom = o2::its::GeometryTGeo::Instance();
  auto trackClusters = getTrackClusters(itsTrack);
  auto trackClusSizes = getTrackClusterSizes(itsTrack);
  auto& lastClus = trackClusters[0];
  strangeTrack.mMatchChi2 = getMatchingChi2(strangeTrack.mMother, itsTrack);

  auto radTol = decayR < 4 ? mStrParams->mRadiusTolIB : mStrParams->mRadiusTolOB;
  auto nMinClusMother = trackClusters.size() < 4 ? 2 : mStrParams->mMinMotherClus;

  std::vector<ITSCluster> motherClusters;
  std::vector<int> motherClusSizes;
  std::array<unsigned int, 7> nAttachments;
  nAttachments.fill(-1); // fill arr with -1

  int nUpdates = 0;
  bool isMotherUpdated = false;

  for (int iClus{0}; iClus < trackClusters.size(); iClus++) {
    auto& clus = trackClusters[iClus];
    auto& compClus = trackClusSizes[iClus];
    int nUpdOld = nUpdates;
    double clusRad = sqrt(clus.getX() * clus.getX() - clus.getY() * clus.getY());
    auto diffR = decayR - clusRad;
    auto relDiffR = diffR / decayR;
    // Look for the Mother if the Decay radius allows for it, within a tolerance
    LOG(debug) << "decayR: " << decayR << ", diffR: " << diffR << ", clus rad: " << clusRad << ", radTol: " << radTol;
    if (relDiffR > -radTol) {
      LOG(debug) << "Try to attach cluster to Mother, layer: " << geom->getLayer(clus.getSensorID());
      if (updateTrack(clus, strangeTrack.mMother)) {
        motherClusters.push_back(clus);
        motherClusSizes.push_back(compClus);
        nAttachments[geom->getLayer(clus.getSensorID())] = 0;
        isMotherUpdated = true;
        nUpdates++;
        LOG(debug) << "Cluster attached to Mother";
        continue; // if the cluster is attached to the mother, skip the rest of the loop
      }
    }

    // if Mother is not found, check for V0 daughters compatibility
    if (relDiffR < radTol && !isMotherUpdated) {
      bool isDauUpdated = false;
      LOG(debug) << "Try to attach cluster to Daughters, layer: " << geom->getLayer(clus.getSensorID());
      for (int iDau{0}; iDau < daughterTracks.size(); iDau++) {
        auto& dauTrack = daughterTracks[iDau];
        if (updateTrack(clus, dauTrack)) {
          nAttachments[geom->getLayer(clus.getSensorID())] = iDau + 1;
          isDauUpdated = true;
          break;
        }
      }
      if (!isDauUpdated) {
        break; // no daughter track updated, stop the loop
      }
      nUpdates++;
    }
    if (nUpdates == nUpdOld) {
      break; // no track updated, stop the loop
    }
  }

  if (nUpdates < trackClusters.size() || motherClusters.size() < nMinClusMother) {
    return false;
  }

  o2::track::TrackParCov motherTrackClone = strangeTrack.mMother; // clone and reset covariance for final topology refit
  motherTrackClone.resetCovariance();

  LOG(debug) << "Clusters attached, starting inward-outward refit";

  std::reverse(motherClusters.begin(), motherClusters.end());

  for (auto& clus : motherClusters) {
    if (!updateTrack(clus, motherTrackClone)) {
      break;
    }
  }

  // compute mother average cluster size
  strangeTrack.mITSClusSize = float(std::accumulate(motherClusSizes.begin(), motherClusSizes.end(), 0)) / motherClusSizes.size();

  LOG(debug) << "Inward-outward refit finished, starting final topology refit";
  // final Topology refit

  int cand = 0; // best V0 candidate
  int nCand;

  // refit cascade
  if (strangeTrack.mPartType == dataformats::kStrkCascade) {
    V0 cascV0Upd;
    if (!recreateV0(daughterTracks[kV0DauPos], daughterTracks[kV0DauNeg], cascV0Upd, iThread)) {
      LOG(debug) << "Cascade V0 refit failed";
      return false;
    }
    try {
      nCand = mFitter3Body[iThread].process(cascV0Upd, daughterTracks[kBach], motherTrackClone);
    } catch (std::runtime_error& e) {
      LOG(debug) << "Fitter3Body failed: " << e.what();
      return false;
    }
    if (!nCand || !mFitter3Body[iThread].propagateTracksToVertex()) {
      LOG(debug) << "Fitter3Body failed: propagation to vertex failed";
      return false;
    }
  }

  // refit V0
  else if (strangeTrack.mPartType == dataformats::kStrkV0) {
    try {
      nCand = mFitter3Body[iThread].process(daughterTracks[kV0DauPos], daughterTracks[kV0DauNeg], motherTrackClone);
    } catch (std::runtime_error& e) {
      LOG(debug) << "Fitter3Body failed: " << e.what();
      return false;
    }
    if (!nCand || !mFitter3Body[iThread].propagateTracksToVertex()) {
      LOG(debug) << "Fitter3Body failed: propagation to vertex failed";
      return false;
    }
  }

  strangeTrack.mDecayVtx = mFitter3Body[iThread].getPCACandidatePos();
  strangeTrack.mTopoChi2 = mFitter3Body[iThread].getChi2AtPCACandidate();
  structClus.arr = nAttachments;

  return true;
}

bool StrangenessTracker::updateTrack(const ITSCluster& clus, o2::track::TrackParCov& track)
{
  auto geom = o2::its::GeometryTGeo::Instance();
  auto propInstance = o2::base::Propagator::Instance();
  float alpha = geom->getSensorRefAlpha(clus.getSensorID()), x = clus.getX();
  int layer{geom->getLayer(clus.getSensorID())};

  if (!track.rotate(alpha)) {
    return false;
  }

  if (!propInstance->propagateToX(track, x, getBz(), o2::base::PropagatorImpl<float>::MAX_SIN_PHI, o2::base::PropagatorImpl<float>::MAX_STEP, mCorrType)) {
    return false;
  }

  if (mCorrType == o2::base::PropagatorF::MatCorrType::USEMatCorrNONE) {
    float thick = layer < 3 ? 0.005 : 0.01;
    constexpr float radl = 9.36f; // Radiation length of Si [cm]
    constexpr float rho = 2.33f;  // Density of Si [g/cm^3]
    if (!track.correctForMaterial(thick, thick * rho * radl)) {
      return false;
    }
  }
  auto chi2 = std::abs(track.getPredictedChi2(clus)); // abs to be understood
  LOG(debug) << "Chi2: " << chi2;
  if (chi2 > mStrParams->mMaxChi2 || chi2 < 0) {
    return false;
  }

  if (!track.update(clus)) {
    return false;
  }

  return true;
}

} // namespace strangeness_tracking
} // namespace o2
