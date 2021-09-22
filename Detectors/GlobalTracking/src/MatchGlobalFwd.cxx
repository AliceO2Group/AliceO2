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

#include "GlobalTracking/MatchGlobalFwd.h"

using namespace o2::globaltracking;

//_________________________________________________________
void MatchGlobalFwd::init(std::string matchFcn, std::string cutFcn)
{

  configMatching(matchFcn, cutFcn);
  o2::base::GeometryManager::loadGeometry();
}

//_________________________________________________________
void MatchGlobalFwd::run(const o2::globaltracking::RecoContainer& inp)
{
  mRecoCont = &inp;
  mStartIR = inp.startIR;

  clear();

  if (!prepareMFTData() || !prepareMCHData()) {
    return;
  }

  doMatching();
  fitTracks();
  finalize();
}

//_________________________________________________________
void MatchGlobalFwd::configMatching(const std::string& matchingFcn, const std::string& cutFcn)
{

  if (matchingFcn.find("matchALL") < matchingFcn.length()) {
    LOG(INFO) << " Setting MatchingFunction matchALL: " << matchingFcn;
    setMatchingFunction(&MatchGlobalFwd::matchMFT_MCH_TracksAllParam);
  } else if (matchingFcn.find("matchPhiTanlXY") < matchingFcn.length()) {
    LOG(INFO) << " Setting MatchingFunction matchPhiTanlXY: " << matchingFcn;
    setMatchingFunction(&MatchGlobalFwd::matchMFT_MCH_TracksXYPhiTanl);
  } else if (matchingFcn.find("matchXY") < matchingFcn.length()) {
    LOG(INFO) << " Setting MatchingFunction matchXY: " << matchingFcn;
    setMatchingFunction(&MatchGlobalFwd::matchMFT_MCH_TracksXY);
  } else if (matchingFcn.find("matchHiroshima") < matchingFcn.length()) {
    LOG(INFO) << " Setting MatchingFunction Hiroshima: " << matchingFcn;
    setMatchingFunction(&MatchGlobalFwd::matchHiroshima);
  } else {
    throw std::invalid_argument("Invalid matching function! Aborting...");
  }

  if (cutFcn.find("cutDisabled") < cutFcn.length()) {
    LOG(INFO) << " Setting CutFunction: " << cutFcn;
    setCutFunction(&MatchGlobalFwd::cutDisabled);
  } else {
    throw std::invalid_argument("Invalid cut function! Aborting...");
  }
}

//_________________________________________________________
void MatchGlobalFwd::finalize()
{
  LOG(INFO) << " Finalizing GlobalForwardMatch. Pushing " << mMatchedTracks.size() << " matched tracks";
}

//_________________________________________________________
void MatchGlobalFwd::clear()
{
  mMCHROFTimes.clear();
  mMCHWork.clear();
  mMFTROFTimes.clear();
  mMFTWork.clear();
  mMFTClusters.clear();
  mMatchedTracks.clear();
  mMatchLabels.clear();
}

//_________________________________________________________
bool MatchGlobalFwd::prepareMCHData()
{
  const auto& inp = *mRecoCont;

  // Load MCH tracks
  mMCHTracks = inp.getMCHTracks();
  mMCHTrackROFRec = inp.getMCHTracksROFRecords();
  if (mMCTruthON) {
    mMCHTrkLabels = inp.getMCHTracksMCLabels();
  }
  int nROFs = mMCHTrackROFRec.size();
  LOG(INFO) << "Loaded " << mMCHTracks.size() << " MCH Tracks "
            << " in " << nROFs << " ROFs";

  mMCHWork.reserve(mMCHTracks.size());

  for (int irof = 0; irof < nROFs; irof++) {
    const auto& rofRec = mMCHTrackROFRec[irof];

    int nBC = rofRec.getBCData().differenceInBC(mStartIR);
    float tMin = nBC * o2::constants::lhc::LHCBunchSpacingMUS;
    float tMax = (nBC + rofRec.getBCWidth()) * o2::constants::lhc::LHCBunchSpacingMUS;

    mMCHROFTimes.emplace_back(tMin, tMax); // MCH ROF min/max time
    LOG(DEBUG) << "MCH ROF # " << irof << " [tMin;tMax] = [" << tMin << ";" << tMax << "]";
    int trlim = rofRec.getFirstIdx() + rofRec.getNEntries();
    for (int it = rofRec.getFirstIdx(); it < trlim; it++) {
      auto& trcOrig = mMCHTracks[it];
      int nWorkTracks = mMCHWork.size();
      // working copy MCH track propagated to matching plane and converted to the forward track format
      o2::mch::TrackParam tempParam(trcOrig.getZ(), trcOrig.getParameters(), trcOrig.getCovariances());
      if (!o2::mch::TrackExtrap::extrapToVertexWithoutBranson(tempParam, mMatchingPlaneZ)) {
        LOG(WARNING) << "MCH track propagation to matching plane failed!";
        continue; // Does this break indices?
      }
      auto convertedTrack = MCHtoFwd(tempParam);
      mMCHWork.emplace_back(TrackLocMCH{convertedTrack, {tMin, tMax}});
    }
  }

  return true;
}

//_________________________________________________________
bool MatchGlobalFwd::prepareMFTData()
{
  const auto& inp = *mRecoCont;

  // MFT clusters
  mMFTClusterROFRec = inp.getMFTClustersROFRecords();
  mMFTTrackClusIdx = inp.getMFTTracksClusterRefs();
  const auto clusMFT = inp.getMFTClusters();
  if (mMFTClusterROFRec.empty() || clusMFT.empty()) {
    LOG(INFO) << "No MFT clusters";
    return false;
  }
  const auto patterns = inp.getMFTClustersPatterns();
  auto pattIt = patterns.begin();
  mMFTClusters.reserve(clusMFT.size());
  o2::mft::ioutils::convertCompactClusters(clusMFT, pattIt, mMFTClusters, *mMFTDict);

  // Load MFT tracks
  mMFTTracks = inp.getMFTTracks();

  mMFTTrackROFRec = inp.getMFTTracksROFRecords();
  if (mMCTruthON) {
    mMFTTrkLabels = inp.getMFTTracksMCLabels();
  }
  int nROFs = mMFTTrackROFRec.size();
  LOG(INFO) << "Loaded " << mMFTTracks.size() << " MFT Tracks in " << nROFs << " ROFs";

  mMFTWork.reserve(mMFTTracks.size());

  for (int irof = 0; irof < nROFs; irof++) {
    const auto& rofRec = mMFTTrackROFRec[irof];

    int nBC = rofRec.getBCData().differenceInBC(mStartIR);
    float tMin = nBC * o2::constants::lhc::LHCBunchSpacingMUS;
    float tMax = (nBC + mMFTROFrameLengthInBC) * o2::constants::lhc::LHCBunchSpacingMUS;
    if (!mMFTTriggered) {
      auto irofCont = nBC / mMFTROFrameLengthInBC;
      if (mMFTTrackROFContMapping.size() <= irofCont) { // there might be gaps in the non-empty rofs, this will map continuous ROFs index to non empty ones
        mMFTTrackROFContMapping.resize((1 + irofCont / 128) * 128, 0);
      }
      mMFTTrackROFContMapping[irofCont] = irof;
    }
    mMFTROFTimes.emplace_back(tMin, tMax); // MFT ROF min/max time
    LOG(DEBUG) << "MFT ROF # " << irof << " [tMin;tMax] = [" << tMin << ";" << tMax << "]";

    int trlim = rofRec.getFirstEntry() + rofRec.getNEntries();
    for (int it = rofRec.getFirstEntry(); it < trlim; it++) {
      const auto& trcOrig = mMFTTracks[it];

      int nWorkTracks = mMFTWork.size();
      // working copy of outer track param
      auto& trc = mMFTWork.emplace_back(TrackLocMFT{trcOrig.getOutParam(), {tMin, tMax}, irof});
      trc.propagateToZ(mMatchingPlaneZ, mBz);
    }
  }

  return true;
}

//_________________________________________________________
void MatchGlobalFwd::doMatching()
{
  // Range of compatible MCH ROFS for the first MFT track
  int nMCHROFs = mMCHROFTimes.size();

  LOG(INFO) << "Running MCH-MFT Track Matching.";
  // ROFrame of first MFT track
  auto firstMFTTrackIdInROF = 0;
  auto MFTROFId = mMFTWork.front().roFrame;
  while ((firstMFTTrackIdInROF < mMFTTracks.size()) && (MFTROFId < mMFTTrackROFRec.size())) {
    auto MFTROFId = mMFTWork[firstMFTTrackIdInROF].roFrame;
    const auto& thisMFTBracket = mMFTROFTimes[MFTROFId];
    auto nMFTTracksInROF = mMFTTrackROFRec[MFTROFId].getNEntries();
    firstMFTTrackIdInROF = mMFTTrackROFRec[MFTROFId].getFirstEntry();
    LOG(DEBUG) << "MFT ROF = " << MFTROFId << "; interval: [" << thisMFTBracket.getMin() << "," << thisMFTBracket.getMax() << "]";
    LOG(DEBUG) << "ROF " << MFTROFId << " : firstMFTTrackIdInROF " << firstMFTTrackIdInROF << " ; nMFTTracksInROF = " << nMFTTracksInROF;
    firstMFTTrackIdInROF += nMFTTracksInROF;
    int mchROF = 0;
    while (mchROF < nMCHROFs && (thisMFTBracket.isOutside(mMCHROFTimes[mchROF]))) {
      // LOG(INFO) << "mchROF = " << mchROF << " ===> thisMFTBracket.isOutside(mMCHROFTimes[mchROF]) = " << thisMFTBracket.isOutside(mMCHROFTimes[mchROF]);
      mchROF++;
    }
    int mchROFMatchFirst = -1;
    int mchROFMatchLast = -1;

    //LOG(INFO) << "mchROF = " << mchROF << " ===> thisMFTBracket.isOutside(mMCHROFTimes[mchROF]) = " << thisMFTBracket.isOutside(mMCHROFTimes[mchROF]);
    if (thisMFTBracket.isOutside(mMCHROFTimes[mchROF]) == 0) {
      mchROFMatchFirst = mchROF;

      while (mchROF < nMCHROFs && !(thisMFTBracket < mMCHROFTimes[mchROF])) {
        mchROF++;
      }
      mchROFMatchLast = mchROF - 1;
    } else {
      LOG(DEBUG) << "No compatible MCH ROF with MFT ROF " << MFTROFId << std::endl;
    }
    //std::cout << "First compatible MCH ROF = " << mchROFMatchFirst << " ; ";
    //std::cout << "Last compatible MCH ROF = " << mchROFMatchLast << std::endl;
    if (mchROFMatchFirst >= 0) {
      ROFMatch(MFTROFId, mchROFMatchFirst, mchROFMatchLast);
    }
  }
}

//_________________________________________________________
void MatchGlobalFwd::ROFMatch(int MFTROFId, int firstMCHROFId, int lastMCHROFId)
{
  /// Matches MFT tracks on a given ROF with MCH tracks in a range of ROFs
  const auto& thisMFTROF = mMFTTrackROFRec[MFTROFId];
  const auto& firstMCHROF = mMCHTrackROFRec[firstMCHROFId];
  const auto& lastMCHROF = mMCHTrackROFRec[lastMCHROFId];
  int nFakes = 0, nTrue = 0;

  auto firstMFTTrackID = thisMFTROF.getFirstEntry();
  auto lastMFTTrackID = firstMFTTrackID + thisMFTROF.getNEntries() - 1;

  auto firstMCHTrackID = firstMCHROF.getFirstIdx();
  auto lastMCHTrackID = lastMCHROF.getLastIdx();
  auto nMFTTracks = thisMFTROF.getNEntries();
  auto nMCHTracks = lastMCHTrackID - firstMCHTrackID + 1;

  LOG(DEBUG) << "Matching MFT ROF " << MFTROFId << " with MCH ROFs [" << firstMCHROFId << "->" << lastMCHROFId << "]";
  LOG(DEBUG) << "   firstMFTTrackID = " << firstMFTTrackID << " ; lastMFTTrackID = " << lastMFTTrackID;
  LOG(DEBUG) << "   firstMCHTrackID = " << firstMCHTrackID << " ; lastMCHTrackID = " << lastMCHTrackID << std::endl;

  // loop over all MCH tracks
  for (auto MCHid = firstMCHTrackID; MCHid <= lastMCHTrackID; MCHid++) {
    auto& thisMCHTrack = mMCHWork[MCHid];
    o2::MCCompLabel matchLabel;
    const o2::MCCompLabel* thisMCHLabel;
    const o2::MCCompLabel* thisMFTLabel;
    if (mMCTruthON) {
      thisMCHLabel = &mMCHTrkLabels->getElement(MCHid);
      //thisMCHLabel = &mMatchLabels[MCHid];
      matchLabel = *thisMCHLabel;
    }
    for (auto MFTid = firstMFTTrackID; MFTid <= lastMFTTrackID; MFTid++) {
      auto& thisMFTTrack = mMFTWork[MFTid];
      if (mMCTruthON) {
        thisMFTLabel = &mMFTTrkLabels[MFTid];
      }
      if (matchingCut(thisMCHTrack, thisMFTTrack)) {
        thisMCHTrack.countCandidate();
        if (mMCTruthON && ((*thisMFTLabel) == (*thisMCHLabel))) {
          thisMCHTrack.setCloseMatch();
        }
        auto chi2 = matchingEval(thisMCHTrack, thisMFTTrack);
        if (chi2 < thisMCHTrack.getMatchingChi2()) {
          thisMCHTrack.setMFTTrackID(MFTid);
          ;
          thisMCHTrack.setMatchingChi2(chi2);
        }
      }
    }
    auto bestMatchID = thisMCHTrack.getMFTTrackID();
    LOG(DEBUG) << "       Matching MCHid = " << MCHid << " ==> bestMatchID = " << thisMCHTrack.getMFTTrackID() << " ; thisMCHTrack.getMatchingChi2() =  " << thisMCHTrack.getMatchingChi2();
    LOG(DEBUG) << "         MCH COV<X,X> = " << thisMCHTrack.getSigma2X() << " ; COV<Y,Y> = " << thisMCHTrack.getSigma2Y() << " ; pt = " << thisMCHTrack.getPt();

    if (bestMatchID >= 0) { // If there is a match, add to output container

      if (mMCTruthON) {
        thisMFTLabel = &mMFTTrkLabels[bestMatchID];
        bool trueMatch = ((*thisMFTLabel) == (*thisMCHLabel));
        matchLabel.setFakeFlag(!trueMatch);
        if (thisMFTLabel->isFake() || thisMCHLabel->isFake()) {
          matchLabel.setFakeFlag(false);
        }
        LOG(DEBUG) << "          MCHTruth = " << *thisMCHLabel << "; MFTTruth = " << *thisMFTLabel << " MatchTruth = " << matchLabel;

        matchLabel.isFake() ? nFakes++ : nTrue++;
      }

      thisMCHTrack.setMFTTrackID(bestMatchID);
      std::cout << "    thisMCHTrack.getMFTTrackID() = " << thisMCHTrack.getMFTTrackID()
                << "; thisMCHTrack.getMatchingChi2() = " << thisMCHTrack.getMatchingChi2()
                << "; Label: " << matchLabel << std::endl;

      mMatchedTracks.emplace_back((thisMCHTrack));

      if (mMCTruthON) {
        mMatchLabels.push_back(matchLabel);
      }
    }

  } // /loop over MCH tracks seeds
  LOG(DEBUG) << " Done matching MFT ROF " << MFTROFId << " with " << nMFTTracks << " MFT tracks with " << nMCHTracks << "  MCH Tracks. nFakes = " << nFakes << " nTrue = " << nTrue;
}

//_________________________________________________________________________________________________
void MatchGlobalFwd::fitTracks()
{
  std::cout << "Fitting global muon tracks..." << std::endl;

  auto GTrackID = 0;

  for (auto& track : mMatchedTracks) {
    LOG(DEBUG) << "  ==> Fitting Global Track # " << GTrackID << " with MFT track # " << track.getMFTTrackID() << ":";
    fitGlobalMuonTrack(track);
    GTrackID++;
  }

  std::cout << "Finished fitting global muon tracks." << std::endl;
}

//_________________________________________________________________________________________________
void MatchGlobalFwd::fitGlobalMuonTrack(o2::dataformats::TrackGlobalFwd& gTrack)
{
  const auto& MFTMatchId = gTrack.getMFTTrackID();
  const auto& mftTrack = mMFTTracks[MFTMatchId];
  const auto& mftTrackOut = mMFTWork[MFTMatchId];
  auto ncls = mftTrack.getNumberOfPoints();
  auto offset = mftTrack.getExternalClusterIndexOffset();
  auto invQPt0 = gTrack.getInvQPt();
  auto sigmainvQPtsq = gTrack.getCovariances()(4, 4);

  // initialize the starting track parameters and cluster
  auto nPoints = mftTrack.getNumberOfPoints();
  auto k = TMath::Abs(o2::constants::math::B2C * mBz);
  auto Hz = std::copysign(1, mBz);

  LOG(DEBUG) << "\n ***************************** Start Fitting new track *****************************";
  LOG(DEBUG) << "  N Clusters = " << ncls;
  LOG(DEBUG) << "  Best MFT Track Match ID " << gTrack.getMFTTrackID();
  LOG(DEBUG) << "  MCHTrack: X = " << gTrack.getX() << " Y = " << gTrack.getY()
             << " Z = " << gTrack.getZ() << " Tgl = " << gTrack.getTanl()
             << "  Phi = " << gTrack.getPhi() << " pz = " << gTrack.getPz()
             << " qpt = " << 1.0 / gTrack.getInvQPt();

  gTrack.setX(mftTrackOut.getX());
  gTrack.setY(mftTrackOut.getY());
  gTrack.setZ(mftTrackOut.getZ());
  gTrack.setPhi(mftTrackOut.getPhi());
  gTrack.setTanl(mftTrackOut.getTanl());
  gTrack.setInvQPt(gTrack.getInvQPt());

  LOG(DEBUG) << "  MFTTrack: X = " << mftTrackOut.getX()
             << " Y = " << mftTrackOut.getY() << " Z = " << mftTrackOut.getZ()
             << " Tgl = " << mftTrackOut.getTanl()
             << "  Phi = " << mftTrackOut.getPhi() << " pz = " << mftTrackOut.getPz()
             << " qpt = " << 1.0 / mftTrackOut.getInvQPt();
  LOG(DEBUG) << "  initTrack GlobalTrack: q/pt = " << gTrack.getInvQPt() << std::endl;

  SMatrix55Sym lastParamCov;
  Double_t tanlsigma = TMath::Max(std::abs(mftTrackOut.getTanl()), .5);
  Double_t qptsigma = TMath::Max(std::abs(mftTrackOut.getInvQPt()), .5);

  lastParamCov(0, 0) = 10000. * mftTrackOut.getCovariances()(0, 0); // <X,X>
  lastParamCov(1, 1) = 10000. * mftTrackOut.getCovariances()(1, 1); // <Y,X>
  lastParamCov(2, 2) = 10000. * mftTrackOut.getCovariances()(2, 2); // TMath::Pi() * TMath::Pi() / 16 // <PHI,X>
  lastParamCov(3, 3) = 10000. * mftTrackOut.getCovariances()(3, 3); // 100. * tanlsigma * tanlsigma;  // mftTrack.getCovariances()(3, 3);     // <TANL,X>
  lastParamCov(4, 4) = gTrack.getCovariances()(4, 4);               //100. * qptsigma * qptsigma;  // <INVQPT,X>

  gTrack.setCovariances(lastParamCov);

  auto lastLayer = mMFTMapping.ChipID2Layer[mMFTClusters[offset + ncls - 1].getSensorID()];
  LOG(DEBUG) << "  Starting by MFTCluster offset " << offset + ncls - 1 << " at lastLayer " << lastLayer;

  for (int icls = ncls - 1; icls > -1; --icls) {
    auto clsEntry = mMFTTrackClusIdx[offset + icls];
    auto& thiscluster = mMFTClusters[clsEntry];
    LOG(DEBUG) << "   Computing MFTCluster clsEntry " << clsEntry << " at Z= " << thiscluster.getZ();

    computeCluster(gTrack, thiscluster, lastLayer);
  }
}

//_________________________________________________________________________________________________
bool MatchGlobalFwd::computeCluster(o2::dataformats::TrackGlobalFwd& track, const MFTCluster& cluster, int& startingLayerID)
{
  /// Propagate track to the z position of the new cluster
  /// accounting for MCS dispersion in the current layer and the other(s) crossed
  /// Recompute the parameters adding the cluster constraint with the Kalman filter
  /// Returns false in case of failure

  const auto& clx = cluster.getX();
  const auto& cly = cluster.getY();
  const auto& clz = cluster.getZ();
  const auto& sigmaX2 = cluster.getSigmaY2(); // ALPIDE local Y coordinate => MFT global X coordinate (ALPIDE rows)
  const auto& sigmaY2 = cluster.getSigmaZ2(); // ALPIDE local Z coordinate => MFT global Y coordinate (ALPIDE columns)

  const auto& newLayerID = mMFTMapping.ChipID2Layer[cluster.getSensorID()];
  LOG(DEBUG) << "computeCluster:     X = " << clx << " Y = " << cly << " Z = " << clz << " nCluster = " << newLayerID;

  if (!propagateToNextClusterWithMCS(track, clz, startingLayerID, newLayerID)) {
    return false;
  }

  LOG(DEBUG) << "   AfterExtrap: X = " << track.getX() << " Y = " << track.getY() << " Z = " << track.getZ() << " Tgl = " << track.getTanl() << "  Phi = " << track.getPhi() << " pz = " << track.getPz() << " q/pt = " << track.getInvQPt();
  LOG(DEBUG) << "Track covariances after extrap:" << std::endl
             << track.getCovariances() << std::endl;

  // recompute parameters
  const std::array<float, 2>& pos = {clx, cly};
  const std::array<float, 2>& cov = {sigmaX2, sigmaY2};

  if (track.update(pos, cov)) {
    LOG(DEBUG) << "   New Cluster: X = " << clx << " Y = " << cly << " Z = " << clz;
    LOG(DEBUG) << "   AfterKalman: X = " << track.getX() << " Y = " << track.getY() << " Z = " << track.getZ() << " Tgl = " << track.getTanl() << "  Phi = " << track.getPhi() << " pz = " << track.getPz() << " q/pt = " << track.getInvQPt();

    LOG(DEBUG) << "Track covariances after Kalman update: \n"
               << track.getCovariances() << std::endl;

    return true;
  }
  return false;
}

//_________________________________________________________________________________________________
double MatchGlobalFwd::matchingEval(const TrackLocMCH& mchTrack, const TrackLocMFT& mftTrack)
{
  return (this->*mMatchFunc)(mchTrack, mftTrack);
}

//_________________________________________________________________________________________________
bool MatchGlobalFwd::matchingCut(const TrackLocMCH& mchTrack, const TrackLocMFT& mftTrack)
{
  return (this->*mCutFunc)(mchTrack, mftTrack);
}

//_________________________________________________________
void MatchGlobalFwd::setMFTROFrameLengthMUS(float fums)
{
  mMFTROFrameLengthMUS = fums;
  mMFTROFrameLengthMUSInv = 1. / mMFTROFrameLengthMUS;
  mMFTROFrameLengthInBC = std::max(1, int(mMFTROFrameLengthMUS / (o2::constants::lhc::LHCBunchSpacingNS * 1e-3)));
}

//_________________________________________________________
void MatchGlobalFwd::setMFTROFrameLengthInBC(int nbc)
{
  mMFTROFrameLengthInBC = nbc;
  mMFTROFrameLengthMUS = nbc * o2::constants::lhc::LHCBunchSpacingNS * 1e-3;
  mMFTROFrameLengthMUSInv = 1. / mMFTROFrameLengthMUS;
}

//_________________________________________________________
void MatchGlobalFwd::setBunchFilling(const o2::BunchFilling& bf)
{
  mBunchFilling = bf;
  // find closest (from above) filled bunch
  int minBC = bf.getFirstFilledBC(), maxBC = bf.getLastFilledBC();
  if (minBC < 0) {
    throw std::runtime_error("Bunch filling is not set in MatchGlobalFwd");
  }
  int bcAbove = minBC;
  for (int i = o2::constants::lhc::LHCMaxBunches; i--;) {
    if (bf.testBC(i)) {
      bcAbove = i;
    }
    mClosestBunchAbove[i] = bcAbove;
  }
  int bcBelow = maxBC;
  for (int i = 0; i < o2::constants::lhc::LHCMaxBunches; i++) {
    if (bf.testBC(i)) {
      bcBelow = i;
    }
    mClosestBunchBelow[i] = bcBelow;
  }
}

//_________________________________________________________________________________________________
double MatchGlobalFwd::matchMFT_MCH_TracksAllParam(const TrackLocMCH& mchTrack, const TrackLocMFT& mftTrack)
{
  // Match two tracks evaluating all parameters: X,Y, phi, tanl & q/pt

  SMatrix55Sym I = ROOT::Math::SMatrixIdentity(), H_k, V_k;
  SVector5 m_k(mftTrack.getX(), mftTrack.getY(), mftTrack.getPhi(),
               mftTrack.getTanl(), mftTrack.getInvQPt()),
    r_k_kminus1;
  SVector5 GlobalMuonTrackParameters = mchTrack.getParameters();
  SMatrix55Sym GlobalMuonTrackCovariances = mchTrack.getCovariances();
  V_k(0, 0) = mftTrack.getCovariances()(0, 0);
  V_k(1, 1) = mftTrack.getCovariances()(1, 1);
  V_k(2, 2) = mftTrack.getCovariances()(2, 2);
  V_k(3, 3) = mftTrack.getCovariances()(3, 3);
  V_k(4, 4) = mftTrack.getCovariances()(4, 4);
  H_k(0, 0) = 1.0;
  H_k(1, 1) = 1.0;
  H_k(2, 2) = 1.0;
  H_k(3, 3) = 1.0;
  H_k(4, 4) = 1.0;

  // Covariance of residuals
  SMatrix55Std invResCov =
    (V_k + ROOT::Math::Similarity(H_k, GlobalMuonTrackCovariances));
  invResCov.Invert();

  // Kalman Gain Matrix
  SMatrix55Std K_k = GlobalMuonTrackCovariances * ROOT::Math::Transpose(H_k) * invResCov;

  // Update Parameters
  r_k_kminus1 = m_k - H_k * GlobalMuonTrackParameters; // Residuals of prediction

  auto matchChi2Track = ROOT::Math::Similarity(r_k_kminus1, invResCov);

  return matchChi2Track;
}

//_________________________________________________________________________________________________
o2::dataformats::TrackGlobalFwd MatchGlobalFwd::MCHtoFwd(const o2::mch::TrackParam& mchParam)
{
  // Convert a MCH Track parameters and covariances matrix to the
  // Forward track format. Must be called after propagation though the absorber

  o2::dataformats::TrackGlobalFwd convertedTrack;

  // Parameter conversion
  double alpha1, alpha3, alpha4, x2, x3, x4;

  alpha1 = mchParam.getNonBendingSlope();
  alpha3 = mchParam.getBendingSlope();
  alpha4 = mchParam.getInverseBendingMomentum();

  x2 = TMath::ATan2(-alpha3, -alpha1);
  x3 = -1. / TMath::Sqrt(alpha3 * alpha3 + alpha1 * alpha1);
  x4 = alpha4 * -x3 * TMath::Sqrt(1 + alpha3 * alpha3);

  auto K = alpha1 * alpha1 + alpha3 * alpha3;
  auto K32 = K * TMath::Sqrt(K);
  auto L = TMath::Sqrt(alpha3 * alpha3 + 1);

  // Covariances matrix conversion
  SMatrix55Std jacobian;
  SMatrix55Sym covariances;

  if (0) {

    std::cout << " MCHtoGlobal - MCH Covariances:\n";
    std::cout << " mchParam.getCovariances()(0, 0) =  "
              << mchParam.getCovariances()(0, 0)
              << " ; mchParam.getCovariances()(2, 2) = "
              << mchParam.getCovariances()(2, 2) << std::endl;
  }
  covariances(0, 0) = mchParam.getCovariances()(0, 0);
  covariances(0, 1) = mchParam.getCovariances()(0, 1);
  covariances(0, 2) = mchParam.getCovariances()(0, 2);
  covariances(0, 3) = mchParam.getCovariances()(0, 3);
  covariances(0, 4) = mchParam.getCovariances()(0, 4);

  covariances(1, 1) = mchParam.getCovariances()(1, 1);
  covariances(1, 2) = mchParam.getCovariances()(1, 2);
  covariances(1, 3) = mchParam.getCovariances()(1, 3);
  covariances(1, 4) = mchParam.getCovariances()(1, 4);

  covariances(2, 2) = mchParam.getCovariances()(2, 2);
  covariances(2, 3) = mchParam.getCovariances()(2, 3);
  covariances(2, 4) = mchParam.getCovariances()(2, 4);

  covariances(3, 3) = mchParam.getCovariances()(3, 3);
  covariances(3, 4) = mchParam.getCovariances()(3, 4);

  covariances(4, 4) = mchParam.getCovariances()(4, 4);

  jacobian(0, 0) = 1;

  jacobian(1, 2) = 1;

  jacobian(2, 1) = -alpha3 / K;
  jacobian(2, 3) = alpha1 / K;

  jacobian(3, 1) = alpha1 / K32;
  jacobian(3, 3) = alpha3 / K32;

  jacobian(4, 1) = -alpha1 * alpha4 * L / K32;
  jacobian(4, 3) = alpha3 * alpha4 * (1 / (TMath::Sqrt(K) * L) - L / K32);
  jacobian(4, 4) = L / TMath::Sqrt(K);

  // jacobian*covariances*jacobian^T
  covariances = ROOT::Math::Similarity(jacobian, covariances);

  // Set output
  convertedTrack.setX(mchParam.getNonBendingCoor());
  convertedTrack.setY(mchParam.getBendingCoor());
  convertedTrack.setZ(mchParam.getZ());
  convertedTrack.setPhi(x2);
  convertedTrack.setTanl(x3);
  convertedTrack.setInvQPt(x4);
  convertedTrack.setCharge(mchParam.getCharge());
  convertedTrack.setCovariances(covariances);

  return convertedTrack;
}

//_________________________________________________________________________________________________
double MatchGlobalFwd::matchMFT_MCH_TracksXY(const TrackLocMCH& mchTrack, const TrackLocMFT& mftTrack)
{
  // Calculate Matching Chi2 - X and Y positions

  SMatrix55Sym I = ROOT::Math::SMatrixIdentity();
  SMatrix25 H_k;
  SMatrix22 V_k;
  SVector2 m_k(mftTrack.getX(), mftTrack.getY()), r_k_kminus1;
  SVector5 GlobalMuonTrackParameters = mchTrack.getParameters();
  SMatrix55Sym GlobalMuonTrackCovariances = mchTrack.getCovariances();
  V_k(0, 0) = mftTrack.getCovariances()(0, 0);
  V_k(1, 1) = mftTrack.getCovariances()(1, 1);
  H_k(0, 0) = 1.0;
  H_k(1, 1) = 1.0;

  // Covariance of residuals
  SMatrix22 invResCov =
    (V_k + ROOT::Math::Similarity(H_k, GlobalMuonTrackCovariances));
  invResCov.Invert();

  // Kalman Gain Matrix
  SMatrix52 K_k =
    GlobalMuonTrackCovariances * ROOT::Math::Transpose(H_k) * invResCov;

  // Update Parameters
  r_k_kminus1 =
    m_k - H_k * GlobalMuonTrackParameters; // Residuals of prediction
  // GlobalMuonTrackParameters = GlobalMuonTrackParameters + K_k * r_k_kminus1;

  // Update covariances Matrix
  // SMatrix55Std updatedCov = (I - K_k * H_k) * GlobalMuonTrackCovariances;

  auto matchChi2Track = ROOT::Math::Similarity(r_k_kminus1, invResCov);

  // GlobalMuonTrack matchTrack(mchTrack);
  // matchTrack.setZ(mchTrack.getZ());
  // matchTrack.setParameters(GlobalMuonTrackParameters);
  // matchTrack.setCovariances(GlobalMuonTrackCovariances);
  // matchTrack.setMatchingChi2(matchChi2Track);
  return matchChi2Track;
}

//_________________________________________________________________________________________________
double MatchGlobalFwd::matchMFT_MCH_TracksXYPhiTanl(const TrackLocMCH& mchTrack, const TrackLocMFT& mftTrack)
{
  // Match two tracks evaluating positions & angles

  SMatrix55Sym I = ROOT::Math::SMatrixIdentity();
  SMatrix45 H_k;
  SMatrix44 V_k;
  SVector4 m_k(mftTrack.getX(), mftTrack.getY(), mftTrack.getPhi(),
               mftTrack.getTanl()),
    r_k_kminus1;
  SVector5 GlobalMuonTrackParameters = mchTrack.getParameters();
  SMatrix55Sym GlobalMuonTrackCovariances = mchTrack.getCovariances();
  V_k(0, 0) = mftTrack.getCovariances()(0, 0);
  V_k(1, 1) = mftTrack.getCovariances()(1, 1);
  V_k(2, 2) = mftTrack.getCovariances()(2, 2);
  V_k(3, 3) = mftTrack.getCovariances()(3, 3);
  H_k(0, 0) = 1.0;
  H_k(1, 1) = 1.0;
  H_k(2, 2) = 1.0;
  H_k(3, 3) = 1.0;

  // Covariance of residuals
  SMatrix44 invResCov =
    (V_k + ROOT::Math::Similarity(H_k, GlobalMuonTrackCovariances));
  invResCov.Invert();

  // Kalman Gain Matrix
  SMatrix54 K_k =
    GlobalMuonTrackCovariances * ROOT::Math::Transpose(H_k) * invResCov;

  // Update Parameters
  r_k_kminus1 =
    m_k - H_k * GlobalMuonTrackParameters; // Residuals of prediction
  // GlobalMuonTrackParameters = GlobalMuonTrackParameters + K_k * r_k_kminus1;

  // Update covariances Matrix
  // SMatrix55Std updatedCov = (I - K_k * H_k) * GlobalMuonTrackCovariances;

  auto matchChi2Track = ROOT::Math::Similarity(r_k_kminus1, invResCov);

  // GlobalMuonTrack matchTrack(mchTrack);
  // matchTrack.setZ(mchTrack.getZ());
  // matchTrack.setParameters(GlobalMuonTrackParameters);
  // matchTrack.setCovariances(GlobalMuonTrackCovariances);
  // matchTrack.setMatchingChi2(matchChi2Track);
  return matchChi2Track;
}

//_________________________________________________________________________________________________
double MatchGlobalFwd::matchHiroshima(const TrackLocMCH& mchTrack, const TrackLocMFT& mftTrack)
{

  //Hiroshima's Matching function

  //Matching constants
  Double_t LAbs = 415.;    //Absorber Length[cm]
  Double_t mumass = 0.106; //mass of muon [GeV/c^2]
  Double_t l;              //the length that extrapolated MCHtrack passes through absorber

  if (mMatchingPlaneZ >= -90.0) {
    l = LAbs;
  } else {
    l = 505.0 + mMatchingPlaneZ;
  }

  //defference between MFTtrack and MCHtrack

  auto dx = mftTrack.getX() - mchTrack.getX();
  auto dy = mftTrack.getY() - mchTrack.getY();
  auto dthetax = TMath::ATan(mftTrack.getPx() / TMath::Abs(mftTrack.getPz())) - TMath::ATan(mchTrack.getPx() / TMath::Abs(mchTrack.getPz()));
  auto dthetay = TMath::ATan(mftTrack.getPy() / TMath::Abs(mftTrack.getPz())) - TMath::ATan(mchTrack.getPy() / TMath::Abs(mchTrack.getPz()));

  //Multiple Scattering(=MS)

  auto pMCH = mchTrack.getP();
  auto lorentzbeta = pMCH / TMath::Sqrt(mumass * mumass + pMCH * pMCH);
  auto zMS = copysign(1.0, mchTrack.getCharge());
  auto thetaMS = 13.6 / (1000.0 * pMCH * lorentzbeta * 1.0) * zMS * TMath::Sqrt(60.0 * l / LAbs) * (1.0 + 0.038 * TMath::Log(60.0 * l / LAbs));
  auto xMS = thetaMS * l / TMath::Sqrt(3.0);

  //normalize by theoritical Multiple Coulomb Scattering width to be momentum-independent
  //make the dx and dtheta dimensionless

  auto dxnorm = dx / xMS;
  auto dynorm = dy / xMS;
  auto dthetaxnorm = dthetax / thetaMS;
  auto dthetaynorm = dthetay / thetaMS;

  //rotate distribution

  auto dxrot = dxnorm * TMath::Cos(TMath::Pi() / 4.0) - dthetaxnorm * TMath::Sin(TMath::Pi() / 4.0);
  auto dthetaxrot = dxnorm * TMath::Sin(TMath::Pi() / 4.0) + dthetaxnorm * TMath::Cos(TMath::Pi() / 4.0);
  auto dyrot = dynorm * TMath::Cos(TMath::Pi() / 4.0) - dthetaynorm * TMath::Sin(TMath::Pi() / 4.0);
  auto dthetayrot = dynorm * TMath::Sin(TMath::Pi() / 4.0) + dthetaynorm * TMath::Cos(TMath::Pi() / 4.0);

  //convert ellipse to circle

  auto k = 0.7; //need to optimize!!
  auto dxcircle = dxrot;
  auto dycircle = dyrot;
  auto dthetaxcircle = dthetaxrot / k;
  auto dthetaycircle = dthetayrot / k;

  //score

  auto scoreX = TMath::Sqrt(dxcircle * dxcircle + dthetaxcircle * dthetaxcircle);
  auto scoreY = TMath::Sqrt(dycircle * dycircle + dthetaycircle * dthetaycircle);
  auto score = TMath::Sqrt(scoreX * scoreX + scoreY * scoreY);

  return score;
};
