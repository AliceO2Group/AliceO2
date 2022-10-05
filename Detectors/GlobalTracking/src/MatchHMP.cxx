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
#include <TTree.h>
#include <cassert>

#include "FairLogger.h"
#include "Field/MagneticField.h"
#include "Field/MagFieldFast.h"
#include "TOFBase/Geo.h"

#include "SimulationDataFormat/MCTruthContainer.h"

#include "DetectorsBase/Propagator.h"

#include "MathUtils/Cartesian.h"
#include "MathUtils/Utils.h"
#include "CommonConstants/MathConstants.h"
#include "CommonConstants/PhysicsConstants.h"
#include "CommonConstants/GeomConstants.h"
#include "DetectorsBase/GeometryManager.h"

#include <Math/SMatrix.h>
#include <Math/SVector.h>
#include <TFile.h>
#include <TGeoGlobalMagField.h>
#include "DataFormatsParameters/GRPObject.h"
#include "ReconstructionDataFormats/PID.h"
#include "ReconstructionDataFormats/TrackLTIntegral.h"

#include "GlobalTracking/MatchHMP.h"

#include "TPCBase/ParameterGas.h"
#include "TPCBase/ParameterElectronics.h"
#include "TPCReconstruction/TPCFastTransformHelperO2.h"

#include "DataFormatsGlobalTracking/RecoContainer.h"
#include "DataFormatsGlobalTracking/RecoContainerCreateTracksVariadic.h"
#include "HMPIDBase/Param.h"

using namespace o2::globaltracking;
using evGIdx = o2::dataformats::EvIndex<int, o2::dataformats::GlobalTrackID>;
using Cluster = o2::hmpid::Cluster;
using GTrackID = o2::dataformats::GlobalTrackID;
using timeEst = o2::dataformats::TimeStampWithError<float, float>;

ClassImp(MatchHMP);

//______________________________________________
void MatchHMP::run(const o2::globaltracking::RecoContainer& inp)
{
  ///< running the matching
  mRecoCont = &inp;
  mRecoCont = &inp;
  mStartIR = inp.startIR;

  for (int i = 0; i < trkType::SIZEALL; i++) {
    mMatchedTracks[i].clear();
    mOutHMPLabels[i].clear();
  }

  for (int i = 0; i < trkType::SIZE; i++) {
    mTracksWork[i].clear();
    mTrackGid[i].clear();
  }
  for (int it = 0; it < trkType::SIZE; it++) {
    mMatchedTracksIndex[it].clear();
    mLTinfos[it].clear();
    if (mMCTruthON) {
      mTracksLblWork[it].clear();
    }
    for (int sec = o2::constants::math::NSectors; sec--;) {
      mTracksSectIndexCache[it][sec].clear();
    }
  }

  mSideTPC.clear();
  mExtraTPCFwdTime.clear();

  // mTimerTot.Start();
  bool isPrepareHMPClusters = prepareHMPClusters();
  mTimerTot.Stop();
  LOGF(info, "Timing prepareTOFCluster: Cpu: %.3e s Real: %.3e s in %d slots", mTimerTot.CpuTime(), mTimerTot.RealTime(), mTimerTot.Counter() - 1);

  if (!isPrepareHMPClusters) { // check cluster before of tracks to see also if MC is required
    return;
  }

  mTimerTot.Start();
  if (!prepareTracks()) {
    return;
  }
  mTimerTot.Stop();
  LOGF(info, "Timing prepare TPC tracks: Cpu: %.3e s Real: %.3e s in %d slots", mTimerTot.CpuTime(), mTimerTot.RealTime(), mTimerTot.Counter() - 1);

  mTimerTot.Start();

  // if (!prepareFITData()) { ef : removed in HMP.h
  //   return;
  // }

  mTimerTot.Stop();
  LOGF(info, "Timing prepare FIT data: Cpu: %.3e s Real: %.3e s in %d slots", mTimerTot.CpuTime(), mTimerTot.RealTime(), mTimerTot.Counter() - 1);

  mTimerTot.Start();
  for (int sec = o2::constants::math::NSectors; sec--;) {
    mMatchedTracksPairs.clear(); //  sector
    LOG(debug) << "Doing matching for sector " << sec << "...";
    if (mIsITSTPCused || mIsTPCTRDused || mIsITSTPCTRDused) {
      mTimerMatchITSTPC.Start(sec == o2::constants::math::NSectors - 1);
      doMatching(sec);
      mTimerMatchITSTPC.Stop();
    }

    /*if (mIsTPCused) {
      mTimerMatchTPC.Start(sec == o2::constants::math::NSectors - 1);
      // doMatchingForTPC(sec); ef : removed in HMP.h
      mTimerMatchTPC.Stop();
    }*/

    LOG(debug) << "...done. Now check the best matches";
    // selectBestMatches(); ef : removed in HMP
  }

  // re-arrange outputs from constrained/unconstrained to the 4 cases (TPC, ITS-TPC, TPC-TRD, ITS-TPC-TRD) to be implemented as soon as TPC-TRD and ITS-TPC-TRD tracks available

  mIsTPCused = false;
  mIsITSTPCused = false;
  mIsTPCTRDused = false;
  mIsITSTPCTRDused = false;

  mTimerTot.Stop();
  LOGF(info, "Timing Do Matching:             Cpu: %.3e s Real: %.3e s in %d slots", mTimerTot.CpuTime(), mTimerTot.RealTime(), mTimerTot.Counter() - 1);
  LOGF(info, "Timing Do Matching Constrained: Cpu: %.3e s Real: %.3e s in %d slots", mTimerMatchITSTPC.CpuTime(), mTimerMatchITSTPC.RealTime(), mTimerMatchITSTPC.Counter() - 1);
  LOGF(info, "Timing Do Matching TPC        : Cpu: %.3e s Real: %.3e s in %d slots", mTimerMatchTPC.CpuTime(), mTimerMatchTPC.RealTime(), mTimerMatchTPC.Counter() - 1);
}

//______________________________________________
bool MatchHMP::prepareTracks()
{
  mNotPropagatedToHMP[trkType::UNCONS] = 0;
  mNotPropagatedToHMP[trkType::CONSTR] = 0;

  auto creator = [this](auto& trk, GTrackID gid, float time0, float terr) {
    const int nclustersMin = 0;
    if constexpr (isTPCTrack<decltype(trk)>()) {
      if (trk.getNClusters() < nclustersMin) {
        return true;
      }

      if (std::abs(trk.getQ2Pt()) > mMaxInvPt) {
        return true;
      }
    }
    if constexpr (isTPCITSTrack<decltype(trk)>()) {
      if (trk.getParamOut().getX() < o2::constants::geom::XTPCOuterRef - 1.) {
        return true;
      }
    }

    return true;
  };

  mRecoCont->createTracksVariadic(creator);

  for (int it = 0; it < trkType::SIZE; it++) {
    mMatchedTracksIndex[it].resize(mTracksWork[it].size());
    std::fill(mMatchedTracksIndex[it].begin(), mMatchedTracksIndex[it].end(), -1); // initializing all to -1
  }

  if (mIsTPCused) {
    LOG(debug) << "Number of UNCONSTRAINED tracks that failed to be propagated to TOF = " << mNotPropagatedToHMP[trkType::UNCONS];

    // sort tracks in each sector according to their time (increasing in time)
    for (int sec = o2::constants::math::NSectors; sec--;) {
      auto& indexCache = mTracksSectIndexCache[trkType::UNCONS][sec];
      LOG(debug) << "Sorting sector" << sec << " | " << indexCache.size() << " tracks";
      if (!indexCache.size()) {
        continue;
      }
      std::sort(indexCache.begin(), indexCache.end(), [this](int a, int b) {
        auto& trcA = mTracksWork[trkType::UNCONS][a].second;
        auto& trcB = mTracksWork[trkType::UNCONS][b].second;
        return ((trcA.getTimeStamp() - trcA.getTimeStampError()) - (trcB.getTimeStamp() - trcB.getTimeStampError()) < 0.);
      });
    } // loop over tracks of single sector
  }
  if (mIsITSTPCused || mIsTPCTRDused || mIsITSTPCTRDused) {
    LOG(debug) << "Number of CONSTRAINED tracks that failed to be propagated to TOF = " << mNotPropagatedToHMP[trkType::CONSTR]; // was mNotPropagatedToTOF

    // sort tracks in each sector according to their time (increasing in time)
    for (int sec = o2::constants::math::NSectors; sec--;) {
      auto& indexCache = mTracksSectIndexCache[trkType::CONSTR][sec];
      LOG(debug) << "Sorting sector" << sec << " | " << indexCache.size() << " tracks";
      if (!indexCache.size()) {
        continue;
      }
      std::sort(indexCache.begin(), indexCache.end(), [this](int a, int b) {
        auto& trcA = mTracksWork[trkType::CONSTR][a].second;
        auto& trcB = mTracksWork[trkType::CONSTR][b].second;
        return ((trcA.getTimeStamp() - mSigmaTimeCut * trcA.getTimeStampError()) - (trcB.getTimeStamp() - mSigmaTimeCut * trcB.getTimeStampError()) < 0.);
      });
    } // loop over tracks of single sector
  }

  return true;
}
//______________________________________________
bool MatchHMP::prepareHMPClusters()
{
  mHMPClustersArrayInp = mRecoCont->getHMPClusters();
  mHMPClusLabels = mRecoCont->getHMPClustersMCLabels();
  mMCTruthON = mHMPClusLabels && mHMPClusLabels->getNElements();

  ///< prepare the tracks that we want to match to HMPID

  // copy the track params, propagate to reference X and build sector tables
  mHMPClusWork.clear();
  //  mHMPClusWork.reserve(mNumOfClusters); // we cannot do this, we don't have mNumOfClusters yet
  //  if (mMCTruthON) {
  //    mTOFClusLblWork.clear();
  //    mTOFClusLblWork.reserve(mNumOfClusters);
  //  }

  for (int sec = o2::constants::math::NSectors; sec--;) {
    mHMPClusSectIndexCache[sec].clear();
    // mHMPClusSectIndexCache[sec].reserve(100 + 1.2 * mNumOfClusters / o2::constants::math::NSectors); // we cannot do this, we don't have mNumOfClusters yet
  }

  int mNumOfClusters = 0;

  int nClusterInCurrentChunk = mHMPClustersArrayInp.size();
  LOG(debug) << "nClusterInCurrentChunk = " << nClusterInCurrentChunk;
  mNumOfClusters += nClusterInCurrentChunk;
  mHMPClusWork.reserve(mHMPClusWork.size() + mNumOfClusters);
  for (int it = 0; it < nClusterInCurrentChunk; it++) {
    const Cluster& clOrig = mHMPClustersArrayInp[it];
    // create working copy of track param
    mHMPClusWork.emplace_back(clOrig);
    auto& cl = mHMPClusWork.back();
    // cache work track index
    // mHMPClusSectIndexCache[cl.getSector()].push_back(mHMPClusWork.size() - 1); fix
  } //  ef: o2::hmpid::Cluster has no type getSector(); I changed cluster to hmpid::cluster instead of tof
    //
  // sort clusters in each sector according to their time (increasing in time)
  for (int sec = o2::constants::math::NSectors; sec--;) {
    auto& indexCache = mHMPClusSectIndexCache[sec];
    LOG(debug) << "Sorting sector" << sec << " | " << indexCache.size() << " TOF clusters";
    if (!indexCache.size()) {
      continue;
    }
    std::sort(indexCache.begin(), indexCache.end(), [this](int a, int b) {
      auto& clA = mHMPClusWork[a];
      auto& clB = mHMPClusWork[b];
      // return (clA.getTime() - clB.getTime()) < 0.; fix ef: original statement
      return 0; //(clA.getTime() - clB.getTime()) < 0.; //fix ef: o2::hmpid::Cluster
    });         //    has no type getTime()
  }             // loop over TOF clusters of single sector

  std::unique_ptr<int[]> mMatchedClustersIndex;
  mMatchedClustersIndex.reset(new int[mNumOfClusters]);

  /* ef : changed to smart/pointer
  if (mMatchedClustersIndex) {
    mMatchedClustersIndex;
  } */
  //= std::unique_ptr<int[]>(new int[mNumOfClusters]);  // ef : change to smart-pointer
  // std::fill_n(mMatchedClustersIndex, mNumOfClusters, -1); // initializing all to -1

  // ef : want to avoid raw-pointers,
  // are any of the following valid? :

  /*
  for(auto& f : mMatchedClustersIndex){
    f = -1;
  } */

  for (int n = 0; n < mNumOfClusters; n++) {
    mMatchedClustersIndex[n] = -1;
  }

  return true;
}
//______________________________________________
void MatchHMP::doMatching(int cham)
{
  trkType type = trkType::CONSTR;

  ///< do the real matching per chmaber
  auto& cacheHMP = mHMPClusSectIndexCache[cham];      // array of cached HMP cluster indices for this chmaber; reminder: they are ordered in time!
  auto& cacheTrk = mTracksSectIndexCache[type][cham]; // array of cached tracks indices for this chamber; reminder: they are ordered in time!
  int nTracks = cacheTrk.size(), nHMPCls = cacheHMP.size();

  // ef: changed here from sec 2 cham
  LOG(debug) << "Matching sector " << cham << ": number of tracks: " << nTracks << ", number of HMP clusters: " << nHMPCls;
  if (!nTracks || !nHMPCls) {
    return;
  }
  int ihmp0 = 0;                          // starting index in HMP clusters for matching of the track
  int detId[2][5];                        // at maximum one track can fall in 2 strips during the propagation; the second dimention of the array is the TOF det index
  float deltaPos[2][3];                   // at maximum one track can fall in 2 strips during the propagation; the second dimention of the array is the residuals
  o2::track::TrackLTIntegral trkLTInt[2]; // Here we store the integrated track length and time for the (max 2) matched strips
  int nStepsInsideSameStrip[2] = {0, 0};  // number of propagation steps in the same strip (since we have maximum 2 strips, it has dimention = 2)
  float deltaPosTemp[3];
  std::array<float, 3> pos;
  std::array<float, 3> posBeforeProp;
  float posFloat[3];

  // prematching for TPC only tracks (identify BC candidate to correct z for TPC track accordingly to v_drift)

  LOG(debug) << "Trying to match %d tracks" << cacheTrk.size();
  for (int itrk = 0; itrk < cacheTrk.size(); itrk++) {
    for (int ii = 0; ii < 2; ii++) {
      detId[ii][2] = -1; // before trying to match, we need to inizialize the detId corresponding to the strip number to -1; this is the array that we will use to save the det id of the maximum 2 strips matched
      nStepsInsideSameStrip[ii] = 0;
    }
    int nStripsCrossedInPropagation = 0; // how many strips were hit during the propagation
    auto& trackWork = mTracksWork[type][cacheTrk[itrk]];
    auto& trefTrk = trackWork.first;
    auto& intLT = mLTinfos[type][cacheTrk[itrk]];

    //    Printf("intLT (before doing anything): length = %f, time (Pion) = %f", intLT.getL(), intLT.getTOF(o2::track::PID::Pion));
    float minTrkTime = (trackWork.second.getTimeStamp() - mSigmaTimeCut * trackWork.second.getTimeStampError()) * 1.E6; // minimum time in ps
    float maxTrkTime = (trackWork.second.getTimeStamp() + mSigmaTimeCut * trackWork.second.getTimeStampError()) * 1.E6; // maximum time in ps
    int istep = 1;                                                                                                      // number of steps
    float step = 1.0;                                                                                                   // step size in cm

    // uncomment for local debug
    /*
    //trefTrk.getXYZGlo(posBeforeProp);
    //float posBeforeProp[3] = {trefTrk.getX(), trefTrk.getY(), trefTrk.getZ()}; // in local ref system
    //printf("Global coordinates: posBeforeProp[0] = %f, posBeforeProp[1] = %f, posBeforeProp[2] = %f\n", posBeforeProp[0], posBeforeProp[1], posBeforeProp[2]);
    //Printf("Radius xy = %f", TMath::Sqrt(posBeforeProp[0]*posBeforeProp[0] + posBeforeProp[1]*posBeforeProp[1]));
    //Printf("Radius xyz = %f", TMath::Sqrt(posBeforeProp[0]*posBeforeProp[0] + posBeforeProp[1]*posBeforeProp[1] + posBeforeProp[2]*posBeforeProp[2]));
    */

    // initializing
    for (int ii = 0; ii < 2; ii++) {
      for (int iii = 0; iii < 5; iii++) {
        detId[ii][iii] = -1;
      }
    }

    int detIdTemp[5] = {-1, -1, -1, -1, -1}; // HMP detector id at the current propagation point

    double reachedPoint = mXRef + istep * step;

    // modify this lines for the tracks propagation to the HMPID chambers

    /* while (propagateToRefX(trefTrk, reachedPoint, step, intLT) && nStripsCrossedInPropagation <= 2 && reachedPoint < Geo::RMAX) {
       // while (o2::base::Propagator::Instance()->PropagateToXBxByBz(trefTrk,  mXRef + istep * step, MAXSNP, step, 1, &intLT) && nStripsCrossedInPropagation <= 2 && mXRef + istep * step < Geo::RMAX) {

       trefTrk.getXYZGlo(pos);
       for (int ii = 0; ii < 3; ii++) { // we need to change the type...
         posFloat[ii] = pos[ii];
       }*/

    // uncomment below only for local debug; this will produce A LOT of output - one print per propagation step
    /*
    Printf("posFloat[0] = %f, posFloat[1] = %f, posFloat[2] = %f", posFloat[0], posFloat[1], posFloat[2]);
    Printf("radius xy = %f", TMath::Sqrt(posFloat[0]*posFloat[0] + posFloat[1]*posFloat[1]));
    Printf("radius xyz = %f", TMath::Sqrt(posFloat[0]*posFloat[0] + posFloat[1]*posFloat[1] + posFloat[2]*posFloat[2]));
    */

    for (int idet = 0; idet < 5; idet++) {
      detIdTemp[idet] = -1;
    }
    // cham is passed as input, but for tof it is sec? ef
    //  Geo::getPadDxDyDz(posFloat, detIdTemp, deltaPosTemp, cham);
    // Geo::getPadDxDyDz(posFloat, detIdTemp, deltaPosTemp, sec);
    reachedPoint += step;

    if (detIdTemp[2] == -1) {
      continue;
    }

    if (nStripsCrossedInPropagation == 0 ||                                                                                                                                                                                            // we are crossing a strip for the first time...
        (nStripsCrossedInPropagation >= 1 && (detId[nStripsCrossedInPropagation - 1][0] != detIdTemp[0] || detId[nStripsCrossedInPropagation - 1][1] != detIdTemp[1] || detId[nStripsCrossedInPropagation - 1][2] != detIdTemp[2]))) { // ...or we are crossing a new strip
      if (nStripsCrossedInPropagation == 0) {
        LOG(debug) << "We cross a strip for the first time";
      }
      if (nStripsCrossedInPropagation == 2) {
        break; // we have already matched 2 strips, we cannot match more
      }
      nStripsCrossedInPropagation++;
    }
  }

  /*    for (Int_t imatch = 0; imatch < nStripsCrossedInPropagation; imatch++) {
        // we take as residual the average of the residuals along the propagation in the same strip
        deltaPos[imatch][0] /= nStepsInsideSameStrip[imatch];
        deltaPos[imatch][1] /= nStepsInsideSameStrip[imatch];
        deltaPos[imatch][2] /= nStepsInsideSameStrip[imatch];
      }

      if (nStripsCrossedInPropagation == 0) {
        continue; // the track never hit a TOF strip during the propagation
      }*/

  bool foundCluster = false;

  for (auto ihmp = ihmp0; ihmp < nHMPCls; ihmp++) {
    //      printf("ihmp = %d\n", ihmp);
    auto& trefTOF = mHMPClusWork[cacheHMP[ihmp]];
    // compare the times of the track and the TOF clusters - remember that they both are ordered in time!
    // Printf("trefTOF.getTime() = %f, maxTrkTime = %f, minTrkTime = %f", trefTOF.getTime(), maxTrkTime, minTrkTime);

    /* //</> ef start comment if :no function getTime in hmp-clusters
    if (trefTOF.getTime() < minTrkTime) { // this cluster has a time that is too small for the current track, we will get to the next one
      //Printf("In trefTOF.getTime() < minTrkTime");
      ihmp0 = ihmp + 1; // but for the next track that we will check, we will ignore this cluster (the time is anyway too small)
      continue;
    } */
    //</> ef end comment if

    // ef no method getTime in hmpid:
    // if (trefTOF.getTime() > maxTrkTime) { // no more TOF clusters can be matched to this track
    // break;
    //}

    // int mainChannel = trefTOF.getMainContributingChannel(); // ef: o2::hmpid::Cluster
    //  has no type getMainContributingChannel

    int indices[5];
    // Geo::getVolumeIndices(mainChannel, indices);

    // compute fine correction using cluster position instead of pad center
    // this because in case of multiple-hit cluster position is averaged on all pads contributing to the cluster (then error position matrix can be used for Chi2 if nedeed)
    int ndigits = 1;
    float posCorr[3] = {0, 0, 0};

    // ef: o2::hmpid::Cluster
    // has no type isBitSet
    /* ef: uncomment later
    if (trefTOF.isBitSet(/*o2::tof::Cluster::*kLeft)) { // ef : temporary workaround bc
      posCorr[0] += Geo::XPAD, ndigits++;	       // hmpid does not have enum-type
    }						       // for bit
    if (trefTOF.isBitSet(kUpLeft)) {
      posCorr[0] += Geo::XPAD, posCorr[2] -= Geo::ZPAD, ndigits++;
    }
    if (trefTOF.isBitSet(kDownLeft)) {
      posCorr[0] += Geo::XPAD, posCorr[2] += Geo::ZPAD, ndigits++;
    }
    if (trefTOF.isBitSet(kUp)) {
      posCorr[2] -= Geo::ZPAD, ndigits++;
    }
    if (trefTOF.isBitSet(kDown)) {
      posCorr[2] += Geo::ZPAD, ndigits++;
    }
    if (trefTOF.isBitSet(kRight)) {
      posCorr[0] -= Geo::XPAD, ndigits++;
    }
    if (trefTOF.isBitSet(kUpRight)) {
      posCorr[0] -= Geo::XPAD, posCorr[2] -= Geo::ZPAD, ndigits++;
    }
    if (trefTOF.isBitSet(kDownRight)) {
      posCorr[0] -= Geo::XPAD, posCorr[2] += Geo::ZPAD, ndigits++;
    } */
    // ef uncomment later
    //</> ef: o2::hmpid::Cluster has no type isBitSet //</>

    float ndifInv = 1. / ndigits;
    if (ndigits > 1) {
      posCorr[0] *= ndifInv;
      posCorr[1] *= ndifInv;
      posCorr[2] *= ndifInv;
    }

    int trackIdTOF;
    int eventIdTOF;
    int sourceIdTOF;

    /*for (auto iPropagation = 0; iPropagation < nStripsCrossedInPropagation; iPropagation++) {
      LOG(debug) << "TOF Cluster [" << ihmp << ", " << cacheHMP[ihmp] << "]:      indices   = " << indices[0] << ", " << indices[1] << ", " << indices[2] << ", " << indices[3] << ", " << indices[4];
      LOG(debug) << "Propagated Track [" << itrk << "]: detId[" << iPropagation << "]  = " << detId[iPropagation][0] << ", " << detId[iPropagation][1] << ", " << detId[iPropagation][2] << ", " << detId[iPropagation][3] << ", " << detId[iPropagation][4];
      float resX = deltaPos[iPropagation][0] - (indices[4] - detId[iPropagation][4]) * Geo::XPAD + posCorr[0]; // readjusting the residuals due to the fact that the propagation fell in a pad that was not exactly the one of the cluster
      float resZ = deltaPos[iPropagation][2] - (indices[3] - detId[iPropagation][3]) * Geo::ZPAD + posCorr[2]; // readjusting the residuals due to the fact that the propagation fell in a pad that was not exactly the one of the cluster
      float res = TMath::Sqrt(resX * resX + resZ * resZ);

      LOG(debug) << "resX = " << resX << ", resZ = " << resZ << ", res = " << res;
      if (indices[0] != detId[iPropagation][0]) {
        continue;
      }
      if (indices[1] != detId[iPropagation][1]) {
        continue;
      }
      if (indices[2] != detId[iPropagation][2]) {
        continue;
      }
      float chi2 = res; // TODO: take into account also the time!

      if (res < mSpaceTolerance) { // matching ok!
        LOG(debug) << "MATCHING FOUND: We have a match! between track " << mTracksSectIndexCache[type][cham][itrk] << " and TOF cluster " << mHMPClusSectIndexCache[indices[0]][ihmp]; // sec2cham
        foundCluster = true;
        // set event indexes (to be checked)
        int eventIndexTOFCluster = mHMPClusSectIndexCache[indices[0]][ihmp];

  // ef commented out next line, no method named getTime()
        //mMatchedTracksPairs.emplace_back(cacheTrk[itrk], eventIndexTOFCluster, mHMPClusWork[cacheHMP[ihmp]].getTime(), chi2, trkLTInt[iPropagation], mTrackGid[type][cacheTrk[itrk]], type, (trefTOF.getTime() - (minTrkTime + maxTrkTime) * 0.5) * 1E-6, 0., resX, resZ); // TODO: check if this is correct!
      }
    }
  }  */
    // </>end for-loop
  }
  return;
}

//______________________________________________
bool MatchHMP::propagateToRefX(o2::track::TrackParCov& trc, float xRef, float stepInCm, o2::track::TrackLTIntegral& intLT)
{
  // propagate track to matching reference X
  o2::base::Propagator::MatCorrType matCorr = o2::base::Propagator::MatCorrType::USEMatCorrLUT; // material correction method
  static const float tanHalfSector = tan(o2::constants::math::SectorSpanRad / 2);
  bool refReached = false;
  float xStart = trc.getX();
  // the first propagation will be from 2m, if the track is not at least at 2m
  if (xStart < 50.) {
    xStart = 50.;
  }
  int istep = 1;
  bool hasPropagated = o2::base::Propagator::Instance()->PropagateToXBxByBz(trc, xStart + istep * stepInCm, MAXSNP, stepInCm, matCorr, &intLT);
  while (hasPropagated) {
    if (trc.getX() > xRef) {
      refReached = true; // we reached the 371cm reference
    }
    istep++;
    if (fabs(trc.getY()) > trc.getX() * tanHalfSector) { // we are still in the same sector
      // we need to rotate the track to go to the new sector
      // Printf("propagateToRefX: changing sector");
      auto alphaNew = o2::math_utils::angle2Alpha(trc.getPhiPos());
      if (!trc.rotate(alphaNew) != 0) {
        //  Printf("propagateToRefX: failed to rotate");
        break; // failed (this line is taken from MatchTPCITS and the following comment too: RS: check effect on matching tracks to neighbouring sector)
      }
    }
    if (refReached) {
      break;
    }
    hasPropagated = o2::base::Propagator::Instance()->PropagateToXBxByBz(trc, xStart + istep * stepInCm, MAXSNP, stepInCm, matCorr, &intLT);
  }

  //  if (std::abs(trc.getSnp()) > MAXSNP) Printf("propagateToRefX: condition on snp not ok, returning false");
  // Printf("propagateToRefX: snp of teh track is %f (--> %f grad)", trc.getSnp(), TMath::ASin(trc.getSnp())*TMath::RadToDeg());
  return refReached && std::abs(trc.getSnp()) < 0.95; // Here we need to put MAXSNP
}

//______________________________________________
bool MatchHMP::propagateToRefXWithoutCov(o2::track::TrackParCov& trc, float xRef, float stepInCm, float bzField)
{
  // propagate track to matching reference X without using the covariance matrix
  // we create the copy of the track in a TrackPar object (no cov matrix)
  o2::track::TrackPar trcNoCov(trc);
  const float tanHalfSector = tan(o2::constants::math::SectorSpanRad / 2);
  bool refReached = false;
  float xStart = trcNoCov.getX();
  // the first propagation will be from 2m, if the track is not at least at 2m
  if (xStart < 50.) {
    xStart = 50.;
  }
  int istep = 1;
  bool hasPropagated = trcNoCov.propagateParamTo(xStart + istep * stepInCm, bzField);
  while (hasPropagated) {
    if (trcNoCov.getX() > xRef) {
      refReached = true; // we reached the 371cm reference
    }
    istep++;
    if (fabs(trcNoCov.getY()) > trcNoCov.getX() * tanHalfSector) { // we are still in the same sector
      // we need to rotate the track to go to the new sector
      // Printf("propagateToRefX: changing sector");
      auto alphaNew = o2::math_utils::angle2Alpha(trcNoCov.getPhiPos());
      if (!trcNoCov.rotateParam(alphaNew) != 0) {
        //  Printf("propagateToRefX: failed to rotate");
        break; // failed (this line is taken from MatchTPCITS and the following comment too: RS: check effect on matching tracks to neighbouring sector)
      }
    }
    if (refReached) {
      break;
    }
    hasPropagated = trcNoCov.propagateParamTo(xStart + istep * stepInCm, bzField);
  }
  //  if (std::abs(trc.getSnp()) > MAXSNP) Printf("propagateToRefX: condition on snp not ok, returning false");
  // Printf("propagateToRefX: snp of teh track is %f (--> %f grad)", trcNoCov.getSnp(), TMath::ASin(trcNoCov.getSnp())*TMath::RadToDeg());

  // return refReached && std::abs(trcNoCov.getSnp()) < 0.95 && TMath::Abs(trcNoCov.getZ()) < Geo::MAXHZTOF; // Here we need to put MAXSNP

  return 999.;
}

//______________________________________________
