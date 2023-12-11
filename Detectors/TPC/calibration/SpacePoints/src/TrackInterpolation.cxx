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

/// \file TrackInterpolation.cxx
/// \brief Implementation of the TrackInterpolation class
///
/// \author Ole Schmidt, ole.schmidt@cern.ch
///

#include "SpacePoints/TrackInterpolation.h"
#include "SpacePoints/TrackResiduals.h"
#include "ITStracking/IOUtils.h"
#include "TPCBase/ParameterElectronics.h"
#include "DataFormatsTPC/TrackTPC.h"
#include "DataFormatsTPC/Defs.h"
#include "DataFormatsTRD/Constants.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "ReconstructionDataFormats/PrimaryVertex.h"
#include "ReconstructionDataFormats/VtxTrackRef.h"
#include "MathUtils/Tsallis.h"
#include "TRDBase/PadPlane.h"
#include "TMath.h"
#include "DataFormatsTPC/VDriftCorrFact.h"
#include "Framework/Logger.h"
#include <set>
#include <algorithm>
#include <random>

using namespace o2::tpc;
using GTrackID = o2::dataformats::GlobalTrackID;
using DetID = o2::detectors::DetID;

void TrackInterpolation::init(o2::dataformats::GlobalTrackID::mask_t src)
{
  // perform initialization
  LOG(info) << "Start initializing TrackInterpolation";
  if (mInitDone) {
    LOG(error) << "Initialization already performed.";
    return;
  }

  const auto& elParam = ParameterElectronics::Instance();
  mTPCTimeBinMUS = elParam.ZbinWidth;

  mFastTransform = std::move(TPCFastTransformHelperO2::instance()->create(0));

  mBz = o2::base::Propagator::Instance()->getNominalBz();
  mRecoParam.setBfield(mBz);
  mGeoTRD = o2::trd::Geometry::instance();
  mParams = &SpacePointsCalibConfParam::Instance();

  mSourcesConfigured = src;
  mTrackTypes.insert({GTrackID::ITSTPC, 0});
  mTrackTypes.insert({GTrackID::ITSTPCTRD, 1});
  mTrackTypes.insert({GTrackID::ITSTPCTOF, 2});
  mTrackTypes.insert({GTrackID::ITSTPCTRDTOF, 3});

  mInitDone = true;
  LOGP(info, "Done initializing TrackInterpolation. Configured track input: {}", GTrackID::getSourcesNames(mSourcesConfigured));
}

bool TrackInterpolation::isInputTrackAccepted(const GTrackID& gid, const o2::globaltracking::RecoContainer::GlobalIDSet& gidTable, const o2::dataformats::PrimaryVertex& pv) const
{
  LOGP(debug, "Check if input track {} is accepted", gid.asString());
  bool hasOuterPoint = gidTable[GTrackID::TRD].isIndexSet() || gidTable[GTrackID::TOF].isIndexSet();
  if (!hasOuterPoint && !mProcessITSTPConly) {
    return false; // don't do ITS-only extrapolation through TPC
  }
  const auto itsTrk = mRecoCont->getITSTrack(gidTable[GTrackID::ITS]);
  const auto tpcTrk = mRecoCont->getTPCTrack(gidTable[GTrackID::TPC]);

  if (gidTable[GTrackID::TRD].isIndexSet()) {
    // TRD specific cuts
    const auto& trdTrk = mRecoCont->getITSTPCTRDTrack<o2::trd::TrackTRD>(gidTable[GTrackID::ITSTPCTRD]);
    if (trdTrk.getNtracklets() < mParams->minTRDNTrklts) {
      return false;
    }
  }
  // reduced chi2 cut is the same for all track types
  if (itsTrk.getChi2() / itsTrk.getNumberOfClusters() > mParams->maxITSChi2 || tpcTrk.getChi2() / tpcTrk.getNClusterReferences() > mParams->maxTPCChi2) {
    return false;
  }
  if (!hasOuterPoint) {
    // ITS-TPC track (does not have outer points in TRD or TOF)
    if (itsTrk.getNumberOfClusters() < mParams->minITSNClsNoOuterPoint || tpcTrk.getNClusterReferences() < mParams->minTPCNClsNoOuterPoint) {
      return false;
    }
    if (itsTrk.getPt() < mParams->minPtNoOuterPoint) {
      return false;
    }
  } else {
    if (itsTrk.getNumberOfClusters() < mParams->minITSNCls || tpcTrk.getNClusterReferences() < mParams->minTPCNCls) {
      return false;
    }
  }

  auto trc = mRecoCont->getTrackParam(gid);
  o2::dataformats::DCA dca;
  auto prop = o2::base::Propagator::Instance();
  if (!prop->propagateToDCA(pv, trc, prop->getNominalBz(), 2., o2::base::PropagatorF::MatCorrType::USEMatCorrLUT, &dca)) {
    return false;
  }
  if (dca.getR2() > mParams->maxDCA * mParams->maxDCA) {
    return false;
  }
  return true;
}

GTrackID::Source TrackInterpolation::findValidSource(GTrackID::Source src) const
{
  LOGP(debug, "Trying to find valid source for {}", GTrackID::getSourcesNames(src));
  if (src == GTrackID::ITSTPCTRDTOF) {
    if (mSourcesConfigured[GTrackID::ITSTPCTRD]) {
      return GTrackID::ITSTPCTRD;
    } else if (mSourcesConfigured[GTrackID::ITSTPC]) {
      return GTrackID::ITSTPC;
    } else {
      return GTrackID::NSources;
    }
  } else if (src == GTrackID::ITSTPCTRD || src == GTrackID::ITSTPCTOF) {
    if (mSourcesConfigured[GTrackID::ITSTPC]) {
      return GTrackID::ITSTPC;
    } else {
      return GTrackID::NSources;
    }
  } else {
    return GTrackID::NSources;
  }
}

void TrackInterpolation::prepareInputTrackSample(const o2::globaltracking::RecoContainer& inp)
{
  mRecoCont = &inp;
  uint32_t nTrackSeeds = 0;
  uint32_t countSeedCandidates[4] = {0};
  auto pvvec = mRecoCont->getPrimaryVertices();
  auto trackIndex = mRecoCont->getPrimaryVertexMatchedTracks(); // Global ID's for associated tracks
  auto vtxRefs = mRecoCont->getPrimaryVertexMatchedTrackRefs(); // references from vertex to these track IDs
  int nv = vtxRefs.size() - 1;
  GTrackID::mask_t allowedSources = GTrackID::getSourcesMask("ITS-TPC,ITS-TPC-TRD,ITS-TPC-TOF,ITS-TPC-TRD-TOF");

  for (int iv = 0; iv < nv; iv++) {
    LOGP(debug, "processing PV {} of {}", iv, nv);

    const auto& vtref = vtxRefs[iv];
    auto pv = pvvec[iv];
    for (int is = GTrackID::NSources; is >= 0; is--) {
      if (!allowedSources[is]) {
        continue;
      }
      LOGP(debug, "Checking source {}", is);
      int idMin = vtref.getFirstEntryOfSource(is), idMax = idMin + vtref.getEntriesOfSource(is);
      for (int i = idMin; i < idMax; i++) {
        auto vid = trackIndex[i];
        if (mParams->ignoreNonPVContrib && !vid.isPVContributor()) {
          continue;
        }
        if (vid.isAmbiguous()) {
          continue;
        }
        auto gidTable = mRecoCont->getSingleDetectorRefs(vid);
        if (!mSourcesConfigured[is]) {
          auto src = findValidSource(static_cast<GTrackID::Source>(vid.getSource()));
          if (src == GTrackID::ITSTPCTRD || src == GTrackID::ITSTPC) {
            LOGP(debug, "Found valid source {}", GTrackID::getSourcesNames(src));
            vid = gidTable[src];
            gidTable = mRecoCont->getSingleDetectorRefs(vid);
          } else {
            break; // no valid source for this vertex track source exists
          }
        }
        ++countSeedCandidates[mTrackTypes[vid.getSource()]];
        LOGP(debug, "Checking vid {}", vid.asString());
        if (!isInputTrackAccepted(vid, gidTable, pv)) {
          continue;
        }
        mSeeds.push_back(mRecoCont->getITSTrack(gidTable[GTrackID::ITS]).getParamOut());
        mGIDs.push_back(vid);
        mGIDtables.push_back(gidTable);
        mTrackTimes.push_back(pv.getTimeStamp().getTimeStamp());
        mTrackIndices[mTrackTypes[vid.getSource()]].push_back(nTrackSeeds++);
      }
    }
  }

  LOGP(info, "Created {} seeds. {} out of {} ITS-TPC-TRD-TOF, {} out of {} ITS-TPC-TRD, {} out of {} ITS-TPC-TOF, {} out of {} ITS-TPC",
       nTrackSeeds,
       mTrackIndices[mTrackTypes[GTrackID::ITSTPCTRDTOF]].size(), countSeedCandidates[mTrackTypes[GTrackID::ITSTPCTRDTOF]],
       mTrackIndices[mTrackTypes[GTrackID::ITSTPCTRD]].size(), countSeedCandidates[mTrackTypes[GTrackID::ITSTPCTRD]],
       mTrackIndices[mTrackTypes[GTrackID::ITSTPCTOF]].size(), countSeedCandidates[mTrackTypes[GTrackID::ITSTPCTOF]],
       mTrackIndices[mTrackTypes[GTrackID::ITSTPC]].size(), countSeedCandidates[mTrackTypes[GTrackID::ITSTPC]]);
}

bool TrackInterpolation::isTrackSelected(const o2::track::TrackParCov& trk) const
{
  std::random_device rd;
  std::mt19937 g(rd());
  std::uniform_real_distribution<> distr(0., 1.);
  float weight = 0;
  return o2::math_utils::Tsallis::downsampleTsallisCharged(trk.getPt(), mParams->tsalisThreshold, mSqrtS, weight, distr(g), o2::constants::physics::MassPionCharged);
}

void TrackInterpolation::process()
{
  // main processing function
  if (!mInitDone) {
    LOG(error) << "Initialization not yet done. Aborting...";
    return;
  }
  // set the input containers
  mTPCTracksClusIdx = mRecoCont->getTPCTracksClusterRefs();
  mTPCClusterIdxStruct = &mRecoCont->getTPCClusters();
  if (mDumpTrackPoints) {
    if (!mITSDict) {
      LOG(error) << "No ITS dictionary available";
      return;
    }
    mITSTrackClusIdx = mRecoCont->getITSTracksClusterRefs();
    const auto clusITS = mRecoCont->getITSClusters();
    const auto patterns = mRecoCont->getITSClustersPatterns();
    auto pattIt = patterns.begin();
    mITSClustersArray.clear();
    mITSClustersArray.reserve(clusITS.size());
    LOGP(info, "We have {} ITS clusters and the number of patterns is {}", clusITS.size(), patterns.size());
    o2::its::ioutils::convertCompactClusters(clusITS, pattIt, mITSClustersArray, mITSDict);
  }

  // In case we have more input tracks available than are required per TF
  // we want to sample them. But we still prefer global ITS-TPC-TRD-TOF tracks
  // over ITS-TPC-TRD tracks and so on. So we have to shuffle the indices
  // in blocks.
  std::random_device rd;
  std::mt19937 g(rd());
  std::vector<GTrackID> trackIndices; // here we keep the GIDs for all track types in a single vector to use in loop
  std::shuffle(mTrackIndices[mTrackTypes[GTrackID::ITSTPCTRDTOF]].begin(), mTrackIndices[mTrackTypes[GTrackID::ITSTPCTRDTOF]].end(), g);
  std::shuffle(mTrackIndices[mTrackTypes[GTrackID::ITSTPCTRD]].begin(), mTrackIndices[mTrackTypes[GTrackID::ITSTPCTRD]].end(), g);
  std::shuffle(mTrackIndices[mTrackTypes[GTrackID::ITSTPCTOF]].begin(), mTrackIndices[mTrackTypes[GTrackID::ITSTPCTOF]].end(), g);
  std::shuffle(mTrackIndices[mTrackTypes[GTrackID::ITSTPC]].begin(), mTrackIndices[mTrackTypes[GTrackID::ITSTPC]].end(), g);
  trackIndices.insert(trackIndices.end(), mTrackIndices[mTrackTypes[GTrackID::ITSTPCTRDTOF]].begin(), mTrackIndices[mTrackTypes[GTrackID::ITSTPCTRDTOF]].end());
  trackIndices.insert(trackIndices.end(), mTrackIndices[mTrackTypes[GTrackID::ITSTPCTRD]].begin(), mTrackIndices[mTrackTypes[GTrackID::ITSTPCTRD]].end());
  trackIndices.insert(trackIndices.end(), mTrackIndices[mTrackTypes[GTrackID::ITSTPCTOF]].begin(), mTrackIndices[mTrackTypes[GTrackID::ITSTPCTOF]].end());
  trackIndices.insert(trackIndices.end(), mTrackIndices[mTrackTypes[GTrackID::ITSTPC]].begin(), mTrackIndices[mTrackTypes[GTrackID::ITSTPC]].end());

  int nSeeds = mSeeds.size();
  int maxOutputTracks = (mMaxTracksPerTF >= 0) ? mMaxTracksPerTF + mAddTracksITSTPC : nSeeds;
  mTrackData.reserve(maxOutputTracks);
  mClRes.reserve(maxOutputTracks * param::NPadRows);
  bool maxTracksReached = false;
  for (int iSeed = 0; iSeed < nSeeds; ++iSeed) {
    if (mMaxTracksPerTF >= 0 && mTrackDataCompact.size() >= mMaxTracksPerTF + mAddTracksITSTPC) {
      LOG(info) << "Maximum number of tracks per TF reached. Skipping the remaining " << nSeeds - iSeed << " tracks.";
      break;
    }
    int seedIndex = trackIndices[iSeed];
    if (mParams->enableTrackDownsampling && !isTrackSelected(mSeeds[seedIndex])) {
      continue;
    }
    if (mGIDs[seedIndex].includesDet(DetID::TRD) || mGIDs[seedIndex].includesDet(DetID::TOF)) {
      interpolateTrack(seedIndex);
      if (mProcessSeeds) {
        if (mGIDs[seedIndex].includesDet(DetID::TRD) && mGIDs[seedIndex].includesDet(DetID::TOF)) {
          mGIDs.push_back(mGIDtables[seedIndex][GTrackID::ITSTPCTRD]);
          mGIDtables.push_back(mRecoCont->getSingleDetectorRefs(mGIDs.back()));
          mTrackTimes.push_back(mTrackTimes[seedIndex]);
          mSeeds.push_back(mSeeds[seedIndex]);
        }
        mGIDs.push_back(mGIDtables[seedIndex][GTrackID::ITSTPC]);
        mGIDtables.push_back(mRecoCont->getSingleDetectorRefs(mGIDs.back()));
        mTrackTimes.push_back(mTrackTimes[seedIndex]);
        mSeeds.push_back(mSeeds[seedIndex]);
      }
    } else {
      extrapolateTrack(seedIndex);
    }
  }
  for (int iSeed = nSeeds; iSeed < (int)mSeeds.size(); ++iSeed) {
    // this loop will only be entered in case mProcessSeeds is set
    if (mGIDs[iSeed].includesDet(DetID::TRD) || mGIDs[iSeed].includesDet(DetID::TOF)) {
      interpolateTrack(iSeed);
    } else {
      extrapolateTrack(iSeed);
    }
  }
  LOG(info) << "Could process " << mTrackData.size() << " tracks successfully";
}

void TrackInterpolation::interpolateTrack(int iSeed)
{
  LOGP(debug, "Starting track interpolation for GID {}", mGIDs[iSeed].asString());
  TrackData trackData;
  std::unique_ptr<TrackDataExtended> trackDataExtended;
  std::vector<TPCClusterResiduals> clusterResiduals;
  auto propagator = o2::base::Propagator::Instance();
  const auto& gidTable = mGIDtables[iSeed];
  const auto& trkTPC = mRecoCont->getTPCTrack(gidTable[GTrackID::TPC]);
  const auto& trkITS = mRecoCont->getITSTrack(gidTable[GTrackID::ITS]);
  if (mDumpTrackPoints) {
    trackDataExtended = std::make_unique<TrackDataExtended>();
    (*trackDataExtended).gid = mGIDs[iSeed];
    (*trackDataExtended).clIdx.setFirstEntry(mClRes.size());
    (*trackDataExtended).trkITS = trkITS;
    (*trackDataExtended).trkTPC = trkTPC;
    auto nCl = trkITS.getNumberOfClusters();
    auto clEntry = trkITS.getFirstClusterEntry();
    for (int iCl = nCl - 1; iCl >= 0; iCl--) { // clusters are stored from outer to inner layers
      const auto& clsITS = mITSClustersArray[mITSTrackClusIdx[clEntry + iCl]];
      (*trackDataExtended).clsITS.push_back(clsITS);
    }
  }
  trackData.gid = mGIDs[iSeed];
  trackData.par = mSeeds[iSeed];
  auto& trkWork = mSeeds[iSeed];
  // reset the cache array (sufficient to set cluster available to zero)
  for (auto& elem : mCache) {
    elem.clAvailable = 0;
  }
  trackData.clIdx.setFirstEntry(mClRes.size()); // reference the first cluster residual belonging to this track
  float clusterTimeBinOffset = mTrackTimes[iSeed] / mTPCTimeBinMUS;

  // store the TPC cluster positions in the cache
  for (int iCl = trkTPC.getNClusterReferences(); iCl--;) {
    uint8_t sector, row;
    uint32_t clusterIndexInRow;
    const auto& clTPC = trkTPC.getCluster(mTPCTracksClusIdx, iCl, *mTPCClusterIdxStruct, sector, row);
    float clTPCX;
    std::array<float, 2> clTPCYZ;
    mFastTransform->TransformIdeal(sector, row, clTPC.getPad(), clTPC.getTime(), clTPCX, clTPCYZ[0], clTPCYZ[1], clusterTimeBinOffset);
    mCache[row].clSec = sector;
    mCache[row].clAvailable = 1;
    mCache[row].clY = clTPCYZ[0];
    mCache[row].clZ = clTPCYZ[1];
    mCache[row].clAngle = o2::math_utils::sector2Angle(sector);
  }

  // extrapolate seed through TPC and store track position at each pad row
  for (int iRow = 0; iRow < param::NPadRows; ++iRow) {
    if (!mCache[iRow].clAvailable) {
      continue;
    }
    if (!trkWork.rotate(mCache[iRow].clAngle)) {
      LOG(debug) << "Failed to rotate track during first extrapolation";
      return;
    }
    if (!propagator->PropagateToXBxByBz(trkWork, param::RowX[iRow], mParams->maxSnp, mParams->maxStep, mMatCorr)) {
      LOG(debug) << "Failed on first extrapolation";
      return;
    }
    mCache[iRow].y[ExtOut] = trkWork.getY();
    mCache[iRow].z[ExtOut] = trkWork.getZ();
    mCache[iRow].sy2[ExtOut] = trkWork.getSigmaY2();
    mCache[iRow].szy[ExtOut] = trkWork.getSigmaZY();
    mCache[iRow].sz2[ExtOut] = trkWork.getSigmaZ2();
    mCache[iRow].snp[ExtOut] = trkWork.getSnp();
    //printf("Track alpha at row %i: %.2f, Y(%.2f), Z(%.2f)\n", iRow, trkWork.getAlpha(), trkWork.getY(), trkWork.getZ());
  }

  // start from outermost cluster with outer refit and back propagation
  if (gidTable[GTrackID::TOF].isIndexSet()) {
    LOG(debug) << "TOF point available";
    const auto& clTOF = mRecoCont->getTOFClusters()[gidTable[GTrackID::TOF]];
    if (mDumpTrackPoints) {
      (*trackDataExtended).clsTOF = clTOF;
      (*trackDataExtended).matchTOF = mRecoCont->getTOFMatch(mGIDs[iSeed]);
    }
    const int clTOFSec = clTOF.getCount();
    const float clTOFAlpha = o2::math_utils::sector2Angle(clTOFSec);
    if (!trkWork.rotate(clTOFAlpha)) {
      LOG(debug) << "Failed to rotate into TOF cluster sector frame";
      return;
    }
    float clTOFX = clTOF.getX();
    std::array<float, 2> clTOFYZ{clTOF.getY(), clTOF.getZ()};
    std::array<float, 3> clTOFCov{mParams->sigYZ2TOF, 0.f, mParams->sigYZ2TOF}; // assume no correlation between y and z and equal cluster error sigma^2 = (3cm)^2 / 12
    if (!propagator->PropagateToXBxByBz(trkWork, clTOFX, mParams->maxSnp, mParams->maxStep, mMatCorr)) {
      LOG(debug) << "Failed final propagation to TOF radius";
      return;
    }
    // TODO: check if reset of covariance matrix is needed here (or, in case TOF point is not available at outermost TRD layer)
    if (!trkWork.update(clTOFYZ, clTOFCov)) {
      LOG(debug) << "Failed to update extrapolated ITS track with TOF cluster";
      //LOGF(info, "trkWork.y=%f, cl.y=%f, trkWork.z=%f, cl.z=%f", trkWork.getY(), clTOFYZ[0], trkWork.getZ(), clTOFYZ[1]);
      return;
    }
  }
  if (gidTable[GTrackID::TRD].isIndexSet()) {
    LOG(debug) << "TRD available";
    const auto& trkTRD = mRecoCont->getITSTPCTRDTrack<o2::trd::TrackTRD>(gidTable[GTrackID::ITSTPCTRD]);
    if (mDumpTrackPoints) {
      (*trackDataExtended).trkTRD = trkTRD;
    }
    for (int iLayer = o2::trd::constants::NLAYER - 1; iLayer >= 0; --iLayer) {
      int trkltIdx = trkTRD.getTrackletIndex(iLayer);
      if (trkltIdx < 0) {
        // no TRD tracklet in this layer
        continue;
      }
      const auto& trdSP = mRecoCont->getTRDCalibratedTracklets()[trkltIdx];
      const auto& trdTrklt = mRecoCont->getTRDTracklets()[trkltIdx];
      if (mDumpTrackPoints) {
        (*trackDataExtended).trkltTRD.push_back(trdTrklt);
        (*trackDataExtended).clsTRD.push_back(trdSP);
      }
      auto trkltDet = trdTrklt.getDetector();
      auto trkltSec = trkltDet / (o2::trd::constants::NLAYER * o2::trd::constants::NSTACK);
      if (trkltSec != o2::math_utils::angle2Sector(trkWork.getAlpha())) {
        if (!trkWork.rotate(o2::math_utils::sector2Angle(trkltSec))) {
          LOG(debug) << "Track could not be rotated in TRD tracklet coordinate system in layer " << iLayer;
          return;
        }
      }
      if (!propagator->PropagateToXBxByBz(trkWork, trdSP.getX(), mParams->maxSnp, mParams->maxStep, mMatCorr)) {
        LOG(debug) << "Failed propagation to TRD layer " << iLayer;
        return;
      }

      const auto* pad = mGeoTRD->getPadPlane(trkltDet);
      float tilt = tan(TMath::DegToRad() * pad->getTiltingAngle()); // tilt is signed! and returned in degrees
      float tiltCorrUp = tilt * (trdSP.getZ() - trkWork.getZ());
      float zPosCorrUp = trdSP.getZ() + mRecoParam.getZCorrCoeffNRC() * trkWork.getTgl(); // maybe Z can be corrected on avarage already by the tracklet transformer?
      float padLength = pad->getRowSize(trdTrklt.getPadRow());
      if (!((trkWork.getSigmaZ2() < (padLength * padLength / 12.f)) && (std::fabs(trdSP.getZ() - trkWork.getZ()) < padLength))) {
        tiltCorrUp = 0.f;
      }
      std::array<float, 2> trkltTRDYZ{trdSP.getY() - tiltCorrUp, zPosCorrUp};
      std::array<float, 3> trkltTRDCov;
      mRecoParam.recalcTrkltCov(tilt, trkWork.getSnp(), pad->getRowSize(trdTrklt.getPadRow()), trkltTRDCov);
      if (!trkWork.update(trkltTRDYZ, trkltTRDCov)) {
        LOG(debug) << "Failed to update track at TRD layer " << iLayer;
        return;
      }
    }
  }

  if (mDumpTrackPoints) {
    (*trackDataExtended).trkOuter = trkWork;
  }

  // go back through the TPC and store updated track positions
  bool outerParamStored = false;
  for (int iRow = param::NPadRows; iRow--;) {
    if (!mCache[iRow].clAvailable) {
      continue;
    }
    if (mProcessSeeds && !outerParamStored) {
      // for debug purposes we store the track parameters
      // of the refitted ITS-(TRD)-(TOF) track at the
      // outermose TPC cluster if we are processing all seeds
      // i.e. if we in any case also process the ITS-TPC only
      // part of the same track
      trackData.par = trkWork;
      outerParamStored = true;
    }
    if (!trkWork.rotate(mCache[iRow].clAngle)) {
      LOG(debug) << "Failed to rotate track during back propagation";
      return;
    }
    if (!propagator->PropagateToXBxByBz(trkWork, param::RowX[iRow], mParams->maxSnp, mParams->maxStep, mMatCorr)) {
      LOG(debug) << "Failed on back propagation";
      //printf("trkX(%.2f), clX(%.2f), clY(%.2f), clZ(%.2f), alphaTOF(%.2f)\n", trkWork.getX(), param::RowX[iRow], clTOFYZ[0], clTOFYZ[1], clTOFAlpha);
      return;
    }
    mCache[iRow].y[ExtIn] = trkWork.getY();
    mCache[iRow].z[ExtIn] = trkWork.getZ();
    mCache[iRow].sy2[ExtIn] = trkWork.getSigmaY2();
    mCache[iRow].szy[ExtIn] = trkWork.getSigmaZY();
    mCache[iRow].sz2[ExtIn] = trkWork.getSigmaZ2();
    mCache[iRow].snp[ExtIn] = trkWork.getSnp();
  }

  // calculate weighted mean at each pad row (assume for now y and z are uncorrelated) and store residuals to TPC clusters
  unsigned short deltaRow = 0;
  for (int iRow = 0; iRow < param::NPadRows; ++iRow) {
    if (!mCache[iRow].clAvailable) {
      ++deltaRow;
      continue;
    }
    float wTotY = 1.f / mCache[iRow].sy2[ExtOut] + 1.f / mCache[iRow].sy2[ExtIn];
    float wTotZ = 1.f / mCache[iRow].sz2[ExtOut] + 1.f / mCache[iRow].sz2[ExtIn];
    mCache[iRow].y[Int] = (mCache[iRow].y[ExtOut] / mCache[iRow].sy2[ExtOut] + mCache[iRow].y[ExtIn] / mCache[iRow].sy2[ExtIn]) / wTotY;
    mCache[iRow].z[Int] = (mCache[iRow].z[ExtOut] / mCache[iRow].sz2[ExtOut] + mCache[iRow].z[ExtIn] / mCache[iRow].sz2[ExtIn]) / wTotZ;

    // simple average w/o weighting for angle
    mCache[iRow].snp[Int] = (mCache[iRow].snp[ExtOut] + mCache[iRow].snp[ExtIn]) / 2.f;

    TPCClusterResiduals res;
    res.setDY(mCache[iRow].clY - mCache[iRow].y[Int]);
    res.setDZ(mCache[iRow].clZ - mCache[iRow].z[Int]);
    res.setY(mCache[iRow].y[Int]);
    res.setZ(mCache[iRow].z[Int]);
    res.setSnp(mCache[iRow].snp[Int]);
    res.sec = mCache[iRow].clSec;
    res.dRow = deltaRow;
    clusterResiduals.push_back(std::move(res));
    deltaRow = 1;
  }
  trackData.chi2TRD = gidTable[GTrackID::TRD].isIndexSet() ? mRecoCont->getITSTPCTRDTrack<o2::trd::TrackTRD>(gidTable[GTrackID::ITSTPCTRD]).getChi2() : 0;
  trackData.chi2TPC = trkTPC.getChi2();
  trackData.chi2ITS = trkITS.getChi2();
  trackData.nClsTPC = trkTPC.getNClusterReferences();
  trackData.nClsITS = trkITS.getNumberOfClusters();
  trackData.nTrkltsTRD = gidTable[GTrackID::TRD].isIndexSet() ? mRecoCont->getITSTPCTRDTrack<o2::trd::TrackTRD>(gidTable[GTrackID::ITSTPCTRD]).getNtracklets() : 0;
  trackData.clAvailTOF = gidTable[GTrackID::TOF].isIndexSet() ? 1 : 0;
  trackData.dEdxTPC = trkTPC.getdEdx().dEdxTotTPC;

  TrackParams params; // for refitted track parameters and flagging rejected clusters
  if (mParams->skipOutlierFiltering || validateTrack(trackData, params, clusterResiduals)) {
    // track is good
    int nClValidated = 0;
    int iRow = 0;
    for (unsigned int iCl = 0; iCl < clusterResiduals.size(); ++iCl) {
      iRow += clusterResiduals[iCl].dRow;
      if (params.flagRej[iCl]) {
        // skip masked cluster residual
        continue;
      }
      ++nClValidated;
      float tgPhi = clusterResiduals[iCl].snp / std::sqrt((1.f - clusterResiduals[iCl].snp) * (1.f + clusterResiduals[iCl].snp));
      mClRes.emplace_back(clusterResiduals[iCl].dy, clusterResiduals[iCl].dz, tgPhi, clusterResiduals[iCl].y, clusterResiduals[iCl].z, iRow, clusterResiduals[iCl].sec);
    }
    trackData.clIdx.setEntries(nClValidated);
    mTrackData.push_back(std::move(trackData));
    if (mDumpTrackPoints) {
      (*trackDataExtended).clIdx.setEntries(nClValidated);
      mTrackDataExtended.push_back(std::move(*trackDataExtended));
    }
    mGIDsSuccess.push_back(mGIDs[iSeed]);
    mTrackDataCompact.emplace_back(mClRes.size() - nClValidated, nClValidated, mGIDs[iSeed].getSource());
  }
  if (mParams->writeUnfiltered) {
    TrackData trkDataTmp = trackData;
    trkDataTmp.clIdx.setFirstEntry(mClResUnfiltered.size());
    trkDataTmp.clIdx.setEntries(clusterResiduals.size());
    mTrackDataUnfiltered.push_back(std::move(trkDataTmp));
    mClResUnfiltered.insert(mClResUnfiltered.end(), clusterResiduals.begin(), clusterResiduals.end());
  }
}

void TrackInterpolation::extrapolateTrack(int iSeed)
{
  // extrapolate ITS-only track through TPC and store residuals to TPC clusters in the output vectors
  LOGP(debug, "Starting track extrapolation for GID {}", mGIDs[iSeed].asString());
  const auto& gidTable = mGIDtables[iSeed];
  TrackData trackData;
  std::unique_ptr<TrackDataExtended> trackDataExtended;
  std::vector<TPCClusterResiduals> clusterResiduals;
  trackData.clIdx.setFirstEntry(mClRes.size());
  const auto& trkITS = mRecoCont->getITSTrack(gidTable[GTrackID::ITS]);
  const auto& trkTPC = mRecoCont->getTPCTrack(gidTable[GTrackID::TPC]);
  if (mDumpTrackPoints) {
    trackDataExtended = std::make_unique<TrackDataExtended>();
    (*trackDataExtended).gid = mGIDs[iSeed];
    (*trackDataExtended).clIdx.setFirstEntry(mClRes.size());
    (*trackDataExtended).trkITS = trkITS;
    (*trackDataExtended).trkTPC = trkTPC;
    auto nCl = trkITS.getNumberOfClusters();
    auto clEntry = trkITS.getFirstClusterEntry();
    for (int iCl = nCl - 1; iCl >= 0; iCl--) { // clusters are stored from outer to inner layers
      const auto& clsITS = mITSClustersArray[mITSTrackClusIdx[clEntry + iCl]];
      (*trackDataExtended).clsITS.push_back(clsITS);
    }
  }
  trackData.gid = mGIDs[iSeed];
  trackData.par = mSeeds[iSeed];

  auto& trkWork = mSeeds[iSeed];
  float clusterTimeBinOffset = mTrackTimes[iSeed] / mTPCTimeBinMUS;
  auto propagator = o2::base::Propagator::Instance();
  unsigned short rowPrev = 0; // used to calculate dRow of two consecutive cluster residuals
  unsigned short nMeasurements = 0;
  uint8_t clRowPrev = -1; // used to identify and skip split clusters on the same pad row
  for (int iCl = trkTPC.getNClusterReferences(); iCl--;) {
    uint8_t sector, row;
    uint32_t clusterIndexInRow;
    const auto& cl = trkTPC.getCluster(mTPCTracksClusIdx, iCl, *mTPCClusterIdxStruct, sector, row);
    if (clRowPrev == row) {
      // if there are split clusters we only take the first one on the pad row
      continue;
    } else {
      // this is the first cluster we see on this pad row
      clRowPrev = row;
    }
    float x = 0, y = 0, z = 0;
    mFastTransform->TransformIdeal(sector, row, cl.getPad(), cl.getTime(), x, y, z, clusterTimeBinOffset);
    if (!trkWork.rotate(o2::math_utils::sector2Angle(sector))) {
      return;
    }
    if (!propagator->PropagateToXBxByBz(trkWork, x, mParams->maxSnp, mParams->maxStep, mMatCorr)) {
      return;
    }
    TPCClusterResiduals res;
    res.setDY(y - trkWork.getY());
    res.setDZ(z - trkWork.getZ());
    res.setY(trkWork.getY());
    res.setZ(trkWork.getZ());
    res.setSnp(trkWork.getSnp());
    res.sec = sector;
    res.dRow = row - rowPrev;
    rowPrev = row;
    clusterResiduals.push_back(std::move(res));
    ++nMeasurements;
  }
  trackData.chi2TPC = trkTPC.getChi2();
  trackData.chi2ITS = trkITS.getChi2();
  trackData.nClsTPC = trkTPC.getNClusterReferences();
  trackData.nClsITS = trkITS.getNumberOfClusters();
  trackData.clIdx.setEntries(nMeasurements);
  trackData.dEdxTPC = trkTPC.getdEdx().dEdxTotTPC;
  if (mDumpTrackPoints) {
    (*trackDataExtended).trkOuter = trkWork;
  }

  TrackParams params; // for refitted track parameters and flagging rejected clusters
  if (mParams->skipOutlierFiltering || validateTrack(trackData, params, clusterResiduals)) {
    // track is good
    int nClValidated = 0;
    int iRow = 0;
    for (unsigned int iCl = 0; iCl < clusterResiduals.size(); ++iCl) {
      iRow += clusterResiduals[iCl].dRow;
      if (params.flagRej[iCl]) {
        // skip masked cluster residual
        continue;
      }
      ++nClValidated;
      float tgPhi = clusterResiduals[iCl].snp / std::sqrt((1.f - clusterResiduals[iCl].snp) * (1.f + clusterResiduals[iCl].snp));
      mClRes.emplace_back(clusterResiduals[iCl].dy, clusterResiduals[iCl].dz, tgPhi, clusterResiduals[iCl].y, clusterResiduals[iCl].z, iRow, clusterResiduals[iCl].sec);
    }
    trackData.clIdx.setEntries(nClValidated);
    mTrackData.push_back(std::move(trackData));
    mGIDsSuccess.push_back(mGIDs[iSeed]);
    mTrackDataCompact.emplace_back(mClRes.size() - nClValidated, nClValidated, mGIDs[iSeed].getSource());
    if (mDumpTrackPoints) {
      (*trackDataExtended).clIdx.setEntries(nClValidated);
      mTrackDataExtended.push_back(std::move(*trackDataExtended));
    }
  }
  if (mParams->writeUnfiltered) {
    TrackData trkDataTmp = trackData;
    trkDataTmp.clIdx.setFirstEntry(mClResUnfiltered.size());
    trkDataTmp.clIdx.setEntries(clusterResiduals.size());
    mTrackDataUnfiltered.push_back(std::move(trkDataTmp));
    mClResUnfiltered.insert(mClResUnfiltered.end(), clusterResiduals.begin(), clusterResiduals.end());
  }
}

bool TrackInterpolation::validateTrack(const TrackData& trk, TrackParams& params, const std::vector<TPCClusterResiduals>& clsRes) const
{
  if (clsRes.size() < mParams->minNCl) {
    // no enough clusters for this track to be considered
    LOG(debug) << "Skipping track with too few clusters: " << clsRes.size();
    return false;
  }

  bool resHelix = compareToHelix(trk, params, clsRes);
  if (!resHelix) {
    LOG(debug) << "Skipping track too far from helix approximation";
    return false;
  }
  if (fabsf(mBz) > 0.01 && fabsf(params.qpt) > mParams->maxQ2Pt) {
    LOG(debug) << "Skipping track with too high q/pT: " << params.qpt;
    return false;
  }
  if (!outlierFiltering(trk, params, clsRes)) {
    return false;
  }
  return true;
}

bool TrackInterpolation::compareToHelix(const TrackData& trk, TrackParams& params, const std::vector<TPCClusterResiduals>& clsRes) const
{
  std::array<float, param::NPadRows> residHelixY;
  std::array<float, param::NPadRows> residHelixZ;

  std::array<float, param::NPadRows> xLab;
  std::array<float, param::NPadRows> yLab;
  std::array<float, param::NPadRows> sPath;

  float curvature = fabsf(trk.par.getQ2Pt() * mBz * o2::constants::physics::LightSpeedCm2S * 1e-14f);
  int secFirst = clsRes[0].sec;
  float phiSect = (secFirst + .5f) * o2::constants::math::SectorSpanRad;
  float snPhi = sin(phiSect);
  float csPhi = cos(phiSect);
  sPath[0] = 0.f;

  int iRow = 0;
  int nCl = clsRes.size();
  for (unsigned int iP = 0; iP < nCl; ++iP) {
    iRow += clsRes[iP].dRow;
    float yTrk = clsRes[iP].y;
    // LOGF(info, "iRow(%i), yTrk(%f)", iRow, yTrk);
    xLab[iP] = param::RowX[iRow];
    if (clsRes[iP].sec != secFirst) {
      float phiSectCurrent = (clsRes[iP].sec + .5f) * o2::constants::math::SectorSpanRad;
      float cs = cos(phiSectCurrent - phiSect);
      float sn = sin(phiSectCurrent - phiSect);
      xLab[iP] = param::RowX[iRow] * cs - yTrk * sn;
      yLab[iP] = yTrk * cs + param::RowX[iRow] * sn;
    } else {
      xLab[iP] = param::RowX[iRow];
      yLab[iP] = yTrk;
    }
    // this is needed only later, but we retrieve it already now to save another loop
    params.zTrk[iP] = clsRes[iP].z;
    params.xTrk[iP] = param::RowX[iRow];
    params.dy[iP] = clsRes[iP].dy;
    params.dz[iP] = clsRes[iP].dz;
    // done retrieving values for later
    if (iP > 0) {
      float dx = xLab[iP] - xLab[iP - 1];
      float dy = yLab[iP] - yLab[iP - 1];
      float ds2 = dx * dx + dy * dy;
      float ds = sqrt(ds2); // circular path (linear approximation)
      // if the curvature of the track or the (approximated) chord length is too large the more exact formula is used:
      // chord length = 2r * asin(ds/(2r))
      // using the first two terms of the tailer expansion for asin(x) ~ x + x^3 / 6
      if (ds * curvature > 0.05) {
        ds *= (1.f + ds2 * curvature * curvature / 24.f);
      }
      sPath[iP] = sPath[iP - 1] + ds;
    }
  }
  if (fabsf(mBz) < 0.01) {
    // for B=0 we don't need to try a circular fit...
    return true;
  }
  float xcSec = 0.f;
  float ycSec = 0.f;
  float r = 0.f;
  TrackResiduals::fitCircle(nCl, xLab, yLab, xcSec, ycSec, r, residHelixY);
  // LOGF(info, "Done with circle fit. nCl(%i), xcSec(%f), ycSec(%f), r(%f).", nCl, xcSec, ycSec, r);
  /*
  for (int i=0; i<nCl; ++i) {
    LOGF(info, "i(%i), xLab(%f), yLab(%f).", i, xLab[i], yLab[i]);
  }
  */
  // determine curvature
  float phiI = TMath::ATan2(yLab[0], xLab[0]);
  float phiF = TMath::ATan2(yLab[nCl - 1], xLab[nCl - 1]);
  if (phiI < 0) {
    phiI += o2::constants::math::TwoPI;
  }
  if (phiF < 0) {
    phiF += o2::constants::math::TwoPI;
  }
  float dPhi = phiF - phiI;
  float curvSign = -1.f;
  if (dPhi > 0) {
    if (dPhi < o2::constants::math::PI) {
      curvSign = 1.f;
    }
  } else if (dPhi < -o2::constants::math::PI) {
    curvSign = 1.f;
  }
  params.qpt = std::copysign(1.f / (r * mBz * o2::constants::physics::LightSpeedCm2S * 1e-14f), curvSign);

  // calculate circle coordinates in the lab frame
  float xc = xcSec * csPhi - ycSec * snPhi;
  float yc = xcSec * snPhi + ycSec * csPhi;

  std::array<float, 2> pol1Z;
  TrackResiduals::fitPoly1(nCl, sPath, params.zTrk, pol1Z);

  params.tgl = pol1Z[0];

  // max deviations in both directions from helix fit in y and z
  float hMinY = 1e9f;
  float hMaxY = -1e9f;
  float hMinZ = 1e9f;
  float hMaxZ = -1e9f;
  // extract residuals in Z and fill track slopes in sector frame
  int secCurr = secFirst;
  iRow = 0;
  for (unsigned int iCl = 0; iCl < nCl; ++iCl) {
    iRow += clsRes[iCl].dRow;
    float resZ = params.zTrk[iCl] - (pol1Z[1] + sPath[iCl] * pol1Z[0]);
    residHelixZ[iCl] = resZ;
    if (resZ < hMinZ) {
      hMinZ = resZ;
    }
    if (resZ > hMaxZ) {
      hMaxZ = resZ;
    }
    if (residHelixY[iCl] < hMinY) {
      hMinY = residHelixY[iCl];
    }
    if (residHelixY[iCl] > hMaxY) {
      hMaxY = residHelixY[iCl];
    }
    int sec = clsRes[iCl].sec;
    if (sec != secCurr) {
      secCurr = sec;
      phiSect = (.5f + sec) * o2::constants::math::SectorSpanRad;
      snPhi = sin(phiSect);
      csPhi = cos(phiSect);
      xcSec = xc * csPhi + yc * snPhi; // recalculate circle center in the sector frame
    }
    float cstalp = (param::RowX[iRow] - xcSec) / r;
    if (fabsf(cstalp) > 1.f - sFloatEps) {
      // track cannot reach this pad row
      cstalp = std::copysign(1.f - sFloatEps, cstalp);
    }
    params.tglArr[iCl] = cstalp / sqrt((1 - cstalp) * (1 + cstalp)); // 1 / tan(acos(cstalp)) = cstalp / sqrt(1 - cstalp^2)

    // In B+ the slope of q- should increase with x. Just look on q * B
    if (params.qpt * mBz > 0) {
      params.tglArr[iCl] *= -1.f;
    }
  }
  // LOGF(info, "CompareToHelix: hMaxY(%f), hMinY(%f), hMaxZ(%f), hMinZ(%f). Max deviation allowed: y(%.2f), z(%.2f)", hMaxY, hMinY, hMaxZ, hMinZ, mParams->maxDevHelixY, mParams->maxDevHelixZ);
  // LOGF(info, "New pt/Q (%f), old pt/Q (%f)", 1./params.qpt, 1./trk.qPt);
  return fabsf(hMaxY - hMinY) < mParams->maxDevHelixY && fabsf(hMaxZ - hMinZ) < mParams->maxDevHelixZ;
}

bool TrackInterpolation::outlierFiltering(const TrackData& trk, TrackParams& params, const std::vector<TPCClusterResiduals>& clsRes) const
{
  if (clsRes.size() < mParams->nMALong) {
    LOG(debug) << "Skipping track with too few clusters for long moving average: " << clsRes.size();
    return false;
  }
  float rmsLong = checkResiduals(trk, params, clsRes);
  if (static_cast<float>(params.flagRej.count()) / clsRes.size() > mParams->maxRejFrac) {
    LOGP(debug, "Skipping track with too many clusters rejected: {} out of {}", params.flagRej.count(), clsRes.size());
    return false;
  }
  if (rmsLong > mParams->maxRMSLong) {
    LOG(debug) << "Skipping track with too large RMS: " << rmsLong;
    return false;
  }
  return true;
}

float TrackInterpolation::checkResiduals(const TrackData& trk, TrackParams& params, const std::vector<TPCClusterResiduals>& clsRes) const
{
  float rmsLong = 0.f;

  int nCl = clsRes.size();
  int iClFirst = 0;
  int iClLast = nCl - 1;
  int secStart = clsRes[0].sec;

  // arrays with differences / abs(differences) of points to their neighbourhood, initialized to zero
  std::array<float, param::NPadRows> yDiffLL{};
  std::array<float, param::NPadRows> zDiffLL{};
  std::array<float, param::NPadRows> absDevY{};
  std::array<float, param::NPadRows> absDevZ{};

  for (unsigned int iCl = 0; iCl < nCl; ++iCl) {
    if (iCl < iClLast && clsRes[iCl].sec == secStart) {
      continue;
    }
    // sector changed or last cluster reached
    // now run estimators for all points in the same sector
    int nClSec = iCl - iClFirst;
    if (iCl == iClLast) {
      ++nClSec;
    }
    diffToLocLine(nClSec, iClFirst, params.xTrk, params.dy, yDiffLL);
    diffToLocLine(nClSec, iClFirst, params.xTrk, params.dz, zDiffLL);
    iClFirst = iCl;
    secStart = clsRes[iCl].sec;
  }
  // store abs deviations
  int nAccY = 0;
  int nAccZ = 0;
  for (int iCl = nCl; iCl--;) {
    if (fabsf(yDiffLL[iCl]) > param::sEps) {
      absDevY[nAccY++] = fabsf(yDiffLL[iCl]);
    }
    if (fabsf(zDiffLL[iCl]) > param::sEps) {
      absDevZ[nAccZ++] = fabsf(zDiffLL[iCl]);
    }
  }
  if (nAccY < mParams->minNumberOfAcceptedResiduals || nAccZ < mParams->minNumberOfAcceptedResiduals) {
    // mask all clusters
    LOGP(debug, "Accepted {} clusters for dY {} clusters for dZ, but required at least {} for both", nAccY, nAccZ, mParams->minNumberOfAcceptedResiduals);
    params.flagRej.set();
    return 0.f;
  }
  // estimate rms on 90% of the smallest deviations
  int nKeepY = static_cast<int>(.9 * nAccY);
  int nKeepZ = static_cast<int>(.9 * nAccZ);
  std::nth_element(absDevY.begin(), absDevY.begin() + nKeepY, absDevY.begin() + nAccY);
  std::nth_element(absDevZ.begin(), absDevZ.begin() + nKeepZ, absDevZ.begin() + nAccZ);
  float rmsYkeep = 0.f;
  float rmsZkeep = 0.f;
  for (int i = nKeepY; i--;) {
    rmsYkeep += absDevY[i] * absDevY[i];
  }
  for (int i = nKeepZ; i--;) {
    rmsZkeep += absDevZ[i] * absDevZ[i];
  }
  rmsYkeep = std::sqrt(rmsYkeep / nKeepY);
  rmsZkeep = std::sqrt(rmsZkeep / nKeepZ);
  if (rmsYkeep < param::sEps || rmsZkeep < param::sEps) {
    LOG(warning) << "Too small RMS: " << rmsYkeep << "(y), " << rmsZkeep << "(z).";
    params.flagRej.set();
    return 0.f;
  }
  float rmsYkeepI = 1.f / rmsYkeep;
  float rmsZkeepI = 1.f / rmsZkeep;
  int nAcc = 0;
  std::array<float, param::NPadRows> yAcc;
  std::array<float, param::NPadRows> yDiffLong;
  for (int iCl = 0; iCl < nCl; ++iCl) {
    yDiffLL[iCl] *= rmsYkeepI;
    zDiffLL[iCl] *= rmsZkeepI;
    if (yDiffLL[iCl] * yDiffLL[iCl] + zDiffLL[iCl] * zDiffLL[iCl] > mParams->maxStdDevMA) {
      params.flagRej.set(iCl);
    } else {
      yAcc[nAcc++] = params.dy[iCl];
    }
  }
  if (nAcc > mParams->nMALong) {
    diffToMA(nAcc, yAcc, yDiffLong);
    float average = 0.f;
    float rms = 0.f;
    for (int i = 0; i < nAcc; ++i) {
      average += yDiffLong[i];
      rms += yDiffLong[i] * yDiffLong[i];
    }
    average /= nAcc;
    rmsLong = rms / nAcc - average * average;
    rmsLong = (rmsLong > 0) ? std::sqrt(rmsLong) : 0.f;
  }
  return rmsLong;
}

void TrackInterpolation::diffToLocLine(const int np, int idxOffset, const std::array<float, param::NPadRows>& x, const std::array<float, param::NPadRows>& y, std::array<float, param::NPadRows>& diffY) const
{
  // Calculate the difference between the points and the linear extrapolations from the neighbourhood.
  // Nothing more than multiple 1-d fits at once. Instead of building 4 sums (x, x^2, y, xy), 4 * nPoints sums are calculated at once
  // compare to TrackResiduals::fitPoly1() method

  // adding one entry to the vectors saves an additional if statement when calculating the cumulants
  std::vector<float> sumX1vec(np + 1);
  std::vector<float> sumX2vec(np + 1);
  std::vector<float> sumY1vec(np + 1);
  std::vector<float> sumXYvec(np + 1);
  auto sumX1 = &(sumX1vec[1]);
  auto sumX2 = &(sumX2vec[1]);
  auto sumY1 = &(sumY1vec[1]);
  auto sumXY = &(sumXYvec[1]);

  // accumulate sums for all points
  for (int iCl = 0; iCl < np; ++iCl) {
    int idx = iCl + idxOffset;
    sumX1[iCl] = sumX1[iCl - 1] + x[idx];
    sumX2[iCl] = sumX2[iCl - 1] + x[idx] * x[idx];
    sumY1[iCl] = sumY1[iCl - 1] + y[idx];
    sumXY[iCl] = sumXY[iCl - 1] + x[idx] * y[idx];
  }

  for (int iCl = 0; iCl < np; ++iCl) {
    int iClLeft = iCl - mParams->nMAShort;
    int iClRight = iCl + mParams->nMAShort;
    if (iClLeft < 0) {
      iClLeft = 0;
    }
    if (iClRight >= np) {
      iClRight = np - 1;
    }
    int nPoints = iClRight - iClLeft;
    if (nPoints < mParams->nMAShort) {
      continue;
    }
    float nPointsInv = 1.f / nPoints;
    int iClLeftP = iClLeft - 1;
    int iClCurrP = iCl - 1;
    // extract sum from iClLeft to iClRight from cumulants, excluding iCl from the fit
    float sX1 = sumX1[iClRight] - sumX1[iClLeftP] - (sumX1[iCl] - sumX1[iClCurrP]);
    float sX2 = sumX2[iClRight] - sumX2[iClLeftP] - (sumX2[iCl] - sumX2[iClCurrP]);
    float sY1 = sumY1[iClRight] - sumY1[iClLeftP] - (sumY1[iCl] - sumY1[iClCurrP]);
    float sXY = sumXY[iClRight] - sumXY[iClLeftP] - (sumXY[iCl] - sumXY[iClCurrP]);
    float det = sX2 - nPointsInv * sX1 * sX1;
    if (fabsf(det) < 1e-12f) {
      continue;
    }
    float slope = (sXY - nPointsInv * sX1 * sY1) / det;
    float offset = nPointsInv * sY1 - nPointsInv * slope * sX1;
    diffY[iCl + idxOffset] = y[iCl + idxOffset] - slope * x[iCl + idxOffset] - offset;
  }
}

void TrackInterpolation::diffToMA(const int np, const std::array<float, param::NPadRows>& y, std::array<float, param::NPadRows>& diffMA) const
{
  // Calculate
  std::vector<float> sumVec(np + 1);
  auto sum = &(sumVec[1]);
  for (int i = 0; i < np; ++i) {
    sum[i] = sum[i - 1] + y[i];
  }
  for (int i = 0; i < np; ++i) {
    diffMA[i] = 0;
    int iLeft = i - mParams->nMALong;
    int iRight = i + mParams->nMALong;
    if (iLeft < 0) {
      iLeft = 0;
    }
    if (iRight >= np) {
      iRight = np - 1;
    }
    int nPoints = iRight - iLeft;
    if (nPoints < mParams->nMALong) {
      // this cannot happen, since at least mParams->nMALong points are required as neighbours for this function to be called
      continue;
    }
    float movingAverage = (sum[iRight] - sum[iLeft - 1] - (sum[i] - sum[i - 1])) / nPoints;
    diffMA[i] = y[i] - movingAverage;
  }
}

void TrackInterpolation::reset()
{
  mTrackData.clear();
  mTrackDataCompact.clear();
  mTrackDataExtended.clear();
  mClRes.clear();
  mTrackDataUnfiltered.clear();
  mClResUnfiltered.clear();
  mGIDsSuccess.clear();
  for (auto& vec : mTrackIndices) {
    vec.clear();
  }
  mGIDs.clear();
  mGIDtables.clear();
  mTrackTimes.clear();
  mSeeds.clear();
}

//______________________________________________
void TrackInterpolation::setTPCVDrift(const o2::tpc::VDriftCorrFact& v)
{
  // Attention! For the refit we are using reference VDrift and TDriftOffest rather than high-rate calibrated, since we want to have fixed reference over the run
  if (v.refVDrift != mTPCVDriftRef) {
    mTPCVDriftRef = v.refVDrift;
    mTPCDriftTimeOffsetRef = v.refTimeOffset;
    LOGP(info, "Imposing reference VDrift={}/TDrift={} for TPC residuals extraction", mTPCVDriftRef, mTPCDriftTimeOffsetRef);
    o2::tpc::TPCFastTransformHelperO2::instance()->updateCalibration(*mFastTransform, 0, 1.0, mTPCVDriftRef, mTPCDriftTimeOffsetRef);
  }
}
