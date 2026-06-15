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
#include "ITSBase/GeometryTGeo.h"
#include "TPCBase/ParameterElectronics.h"
#include "TOFBase/Geo.h"
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
#include "CCDB/BasicCCDBManager.h"
#include "GPUO2InterfaceUtils.h"
#include "GPUO2InterfaceConfiguration.h"
#include "GPUO2InterfaceRefit.h"
#include "GPUParam.h"
#include "GPUParam.inc"
#include <set>
#include <algorithm>
#include <random>

using namespace o2::tpc;
using GTrackID = o2::dataformats::GlobalTrackID;
using DetID = o2::detectors::DetID;

bool UnbinnedResid::gInitDone = false;

float UnbinnedResid::getAlpha() const
{
  if (!isITS()) {
    return o2::math_utils::sector2Angle(sec % 18);
  }
  // ITS alpha repends on the chip ID
  checkInitDone();
  return o2::its::GeometryTGeo::Instance()->getSensorRefAlpha(channel);
}

float UnbinnedResid::getX() const
{
  if (isTPC()) {
    return param::RowX[row];
  }
  checkInitDone();
  if (isITS()) {
    return o2::its::GeometryTGeo::Instance()->getSensorRefX(channel); // ITS X repends on the chip ID
  }
  if (isTRD()) {
    auto geo = o2::trd::Geometry::instance();
    ROOT::Math::Impl::Transform3D<double>::Point local{geo->cdrHght() - 0.5 - 0.279, 0., 0.}; // see TrackletTransformer::transformTracklet
    return (geo->getMatrixT2L(channel) ^ local).X();
  }
  if (isTOF()) {
    int det[5];
    o2::tof::Geo::getVolumeIndices(channel + sec * o2::tof::Geo::NPADSXSECTOR, det);
    float pos[3] = {0.f, 0.f, 0.f};
    o2::tof::Geo::getPos(det, pos);
    float posl[3] = {pos[0], pos[1], pos[2]};
    o2::tof::Geo::rotateToSector(pos, sec);
    return pos[2]; // coordinates in sector frame: note that the rotation above puts z in pos[1], the radial coordinate in pos[2], and the tangent coordinate in pos[0] (this is to match the TOF residual system, where we don't use the radial component), so we swap their positions.
  }
  LOGP(fatal, "Did not recognize detector type: row:{}, sec:{}, channel:{}", row, sec, channel);
  return 0.;
}

void UnbinnedResid::checkInitDone()
{
  if (!gInitDone) {
    LOGP(warn, "geometry initialization was not done, doing this for the current timestamp");
    init();
    if (!gInitDone) {
      LOGP(fatal, "geometry initialization failed");
    }
  }
}

void UnbinnedResid::init(long timestamp)
{
  if (gInitDone) {
    LOGP(warn, "Initialization was already done");
    return;
  }
  if (!gGeoManager) {
    o2::ccdb::BasicCCDBManager::instance().getSpecific<TGeoManager>("GLO/Config/GeometryAligned", timestamp);
  }
  auto geoTRD = o2::trd::Geometry::instance();
  geoTRD->createPadPlaneArray();
  geoTRD->createClusterMatrixArray();
  gInitDone = true;
}

TrackInterpolation::~TrackInterpolation()
{
  finalize();
}

void TrackInterpolation::finalize()
{
  if (mDBGOut) {
    mDBGOut->Close();
    mDBGOut.reset();
  }
}

void TrackInterpolation::init(o2::dataformats::GlobalTrackID::mask_t src, o2::dataformats::GlobalTrackID::mask_t srcMap)
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
  mRecoParam.init(mBz);
  mGeoTRD = o2::trd::Geometry::instance();
  mParams = &SpacePointsCalibConfParam::Instance();

  mSourcesConfigured = src;
  mSourcesConfiguredMap = srcMap;
  mSingleSourcesConfigured = (mSourcesConfigured == mSourcesConfiguredMap);
  mTrackTypes.insert({GTrackID::ITSTPC, 0});
  mTrackTypes.insert({GTrackID::ITSTPCTRD, 1});
  mTrackTypes.insert({GTrackID::ITSTPCTOF, 2});
  mTrackTypes.insert({GTrackID::ITSTPCTRDTOF, 3});

  auto geom = o2::its::GeometryTGeo::Instance();
  geom->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L, o2::math_utils::TransformType::L2G));
  mTPCParam = o2::gpu::GPUO2InterfaceUtils::getFullParamShared(0.f, mNHBPerTF);

  if (mParams->writeValidationData) {
    std::string dbgnm = mNLanes == 1 ? "track_interpolation_dbg.root" : fmt::format("track_interpolation_dbg_{}.root", mLaneID);
    mDBGOut = std::make_unique<o2::utils::TreeStreamRedirector>(dbgnm.c_str(), "recreate");
  }

  mInitDone = true;
  LOGP(info, "Done initializing TrackInterpolation. Configured track input: {}. Track input specifically for map: {}",
       GTrackID::getSourcesNames(mSourcesConfigured), mSingleSourcesConfigured ? "identical" : GTrackID::getSourcesNames(mSourcesConfiguredMap));
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

GTrackID::Source TrackInterpolation::findValidSource(const GTrackID::mask_t mask, const GTrackID::Source src) const
{
  LOGP(debug, "Trying to find valid source for {} in {}", GTrackID::getSourceName(src), GTrackID::getSourcesNames(mask));
  if (src == GTrackID::ITSTPCTRDTOF) {
    if (mask[GTrackID::ITSTPCTRD]) {
      return GTrackID::ITSTPCTRD;
    } else if (mask[GTrackID::ITSTPC]) {
      return GTrackID::ITSTPC;
    } else {
      return GTrackID::NSources;
    }
  } else if (src == GTrackID::ITSTPCTRD || src == GTrackID::ITSTPCTOF) {
    if (mask[GTrackID::ITSTPC]) {
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
  GTrackID::mask_t allowedSources = GTrackID::getSourcesMask("ITS-TPC,ITS-TPC-TRD,ITS-TPC-TOF,ITS-TPC-TRD-TOF") & mSourcesConfigured;
  constexpr std::array<int, 3> SrcFast = {int(GTrackID::ITSTPCTRD), int(GTrackID::ITSTPCTOF), int(GTrackID::ITSTPCTRDTOF)};
  if (mParams->refitITS) {
    mITSRefitSeedID.resize(mRecoCont->getITSTracks().size(), -1);
  }

  for (int iv = 0; iv < nv; iv++) {
    LOGP(debug, "processing PV {} of {}", iv, nv);

    const auto& vtref = vtxRefs[iv];
    auto pv = pvvec[iv];
    if (mParams->minTOFTRDPVContributors > 0) { // we want only PVs constrained by fast detectors
      int nfound = 0;
      bool usePV = false;
      for (uint32_t is = 0; is < SrcFast.size() && !usePV; is++) {
        int src = SrcFast[is], idMin = vtref.getFirstEntryOfSource(src), idMax = idMin + vtref.getEntriesOfSource(src);
        for (int i = idMin; i < idMax; i++) {
          if (trackIndex[i].isPVContributor() && (++nfound == mParams->minTOFTRDPVContributors)) {
            usePV = true;
            break;
          }
        }
      }
      if (!usePV) {
        continue;
      }
    }

    for (int is = GTrackID::NSources; is--;) {
      if (!allowedSources[is]) {
        continue;
      }
      LOGP(debug, "Checking source {}", is);
      int idMin = vtref.getFirstEntryOfSource(is), idMax = idMin + vtref.getEntriesOfSource(is);
      for (int i = idMin; i < idMax; i++) {
        auto vid = trackIndex[i];
        auto vidOrig = vid; // in case only ITS-TPC tracks are configured vid might be overwritten. We need to remember it for the PID
        if (mParams->ignoreNonPVContrib && !vid.isPVContributor()) {
          continue;
        }
        if (vid.isAmbiguous()) {
          continue;
        }
        auto gidTable = mRecoCont->getSingleDetectorRefs(vid);
        if (!mSourcesConfigured[is]) {
          auto src = findValidSource(mSourcesConfigured, static_cast<GTrackID::Source>(vid.getSource()));
          if (src == GTrackID::ITSTPCTRD || src == GTrackID::ITSTPC) {
            LOGP(debug, "prepareInputTrackSample: Found valid source {}", GTrackID::getSourceName(src));
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
        mSeeds.back().setPID(mRecoCont->getTrackParam(vidOrig).getPID(), true);
        mGIDs.push_back(vid);
        mGIDtables.push_back(gidTable);
        mTrackTimes.push_back(pv.getTimeStamp().getTimeStamp());
        mTrackIndices[mTrackTypes[vid.getSource()]].push_back(nTrackSeeds++);
        mTrackPVID.push_back(iv);
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
  int nbOccTOT = o2::gpu::GPUO2InterfaceRefit::fillOccupancyMapGetSize(mNHBPerTF, mTPCParam.get());
  o2::gpu::GPUO2InterfaceUtils::paramUseExternalOccupancyMap(mTPCParam.get(), mNHBPerTF, mRecoCont->occupancyMapTPC.data(), nbOccTOT);
  mNTPCOccBinLength = mTPCParam->rec.tpc.occupancyMapTimeBins;
  mNTPCOccBinLengthInv = 1.f / mNTPCOccBinLength;
  {
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
  std::vector<uint32_t> trackIndices; // here we keep the GIDs for all track types in a single vector to use in loop
  std::shuffle(mTrackIndices[mTrackTypes[GTrackID::ITSTPCTRDTOF]].begin(), mTrackIndices[mTrackTypes[GTrackID::ITSTPCTRDTOF]].end(), g);
  std::shuffle(mTrackIndices[mTrackTypes[GTrackID::ITSTPCTRD]].begin(), mTrackIndices[mTrackTypes[GTrackID::ITSTPCTRD]].end(), g);
  std::shuffle(mTrackIndices[mTrackTypes[GTrackID::ITSTPCTOF]].begin(), mTrackIndices[mTrackTypes[GTrackID::ITSTPCTOF]].end(), g);
  std::shuffle(mTrackIndices[mTrackTypes[GTrackID::ITSTPC]].begin(), mTrackIndices[mTrackTypes[GTrackID::ITSTPC]].end(), g);
  trackIndices.insert(trackIndices.end(), mTrackIndices[mTrackTypes[GTrackID::ITSTPCTRDTOF]].begin(), mTrackIndices[mTrackTypes[GTrackID::ITSTPCTRDTOF]].end());
  trackIndices.insert(trackIndices.end(), mTrackIndices[mTrackTypes[GTrackID::ITSTPCTRD]].begin(), mTrackIndices[mTrackTypes[GTrackID::ITSTPCTRD]].end());
  trackIndices.insert(trackIndices.end(), mTrackIndices[mTrackTypes[GTrackID::ITSTPCTOF]].begin(), mTrackIndices[mTrackTypes[GTrackID::ITSTPCTOF]].end());
  trackIndices.insert(trackIndices.end(), mTrackIndices[mTrackTypes[GTrackID::ITSTPC]].begin(), mTrackIndices[mTrackTypes[GTrackID::ITSTPC]].end());
  int nSeeds = mSeeds.size(), lastChecked = 0;
  mParentID.clear();
  mParentID.resize(nSeeds, -1);

  int maxOutputTracks = (mMaxTracksPerTF >= 0) ? mMaxTracksPerTF + mAddTracksForMapPerTF : nSeeds;
  mTrackData.reserve(maxOutputTracks);
  mClRes.reserve(maxOutputTracks * param::NPadRows);
  mDetInfoRes.reserve(maxOutputTracks * param::NPadRows);
  bool maxTracksReached = false;
  for (int iSeed = 0; iSeed < nSeeds; ++iSeed) {
    if (mMaxTracksPerTF >= 0 && mTrackDataCompact.size() >= mMaxTracksPerTF + mAddTracksForMapPerTF) {
      LOG(info) << "Maximum number of tracks per TF reached. Skipping the remaining " << nSeeds - iSeed << " tracks.";
      break;
    }
    int seedIndex = trackIndices[iSeed];
    if (mParams->enableTrackDownsampling && !isTrackSelected(mSeeds[seedIndex])) {
      continue;
    }
    auto addPart = [this, seedIndex](GTrackID::Source src) {
      this->mGIDs.push_back(this->mGIDtables[seedIndex][src]);
      this->mGIDtables.push_back(this->mRecoCont->getSingleDetectorRefs(this->mGIDs.back()));
      this->mTrackTimes.push_back(this->mTrackTimes[seedIndex]);
      this->mSeeds.push_back(this->mSeeds[seedIndex]);
      this->mParentID.push_back(seedIndex); // store parent seed id
      this->mTrackPVID.push_back(this->mTrackPVID[seedIndex]);
    };

    GTrackID::mask_t partsAdded;
    if (!mSingleSourcesConfigured && !mSourcesConfiguredMap[mGIDs[seedIndex].getSource()]) {
      auto src = findValidSource(mSourcesConfiguredMap, static_cast<GTrackID::Source>(mGIDs[seedIndex].getSource()));
      if (src == GTrackID::ITSTPCTRD || src == GTrackID::ITSTPC) {
        LOGP(debug, "process {}: Found valid source {} for {} | nseeds:{} mSeeds:{} used: {}", iSeed, GTrackID::getSourceName(src), GTrackID::getSourceName(mGIDs[seedIndex].getSource()), nSeeds, mSeeds.size(), mTrackDataCompact.size());
        addPart(src);
      }
    }
    if (mMaxTracksPerTF >= 0 && mTrackDataCompact.size() >= mMaxTracksPerTF) {
      if (!maxTracksReached) {
        LOGP(info, "We already have reached mMaxTracksPerTF={}, but we continue to create seeds until mAddTracksForMapPerTF={} is also reached, iSeed: {} of {} inital seeds", mMaxTracksPerTF, mAddTracksForMapPerTF, iSeed, nSeeds);
      }
      maxTracksReached = true;
      continue;
    }
    if (mGIDs[seedIndex].includesDet(DetID::TRD) || mGIDs[seedIndex].includesDet(DetID::TOF)) {
      interpolateTrack(seedIndex);
      LOGP(debug, "interpolateTrack {} {}, accepted: {}", iSeed, GTrackID::getSourceName(mGIDs[seedIndex].getSource()), mTrackDataCompact.size());
      if (mProcessSeeds) {
        if (mGIDs[seedIndex].includesDet(DetID::TRD) && mGIDs[seedIndex].includesDet(DetID::TOF) && !partsAdded[GTrackID::ITSTPCTRD]) {
          addPart(GTrackID::ITSTPCTRD);
        }
        if (!partsAdded[GTrackID::ITSTPC]) {
          addPart(GTrackID::ITSTPC);
        }
      }
    } else {
      extrapolateTrack(seedIndex);
      LOGP(debug, "extrapolateTrack {} {}, accepted: {}", iSeed, GTrackID::getSourceName(mGIDs[seedIndex].getSource()), mTrackDataCompact.size());
    }
    lastChecked = iSeed;
  }
  std::vector<int> remSeeds;
  if (mSeeds.size() > ++lastChecked) {
    remSeeds.resize(mSeeds.size() - lastChecked);
    std::iota(remSeeds.begin(), remSeeds.end(), lastChecked);
    std::shuffle(remSeeds.begin(), remSeeds.end(), g);
    LOGP(info, "Up to {} tracks out of {} additional seeds will be processed in random order, of which {} are stripped versions, accepted seeds: {}",
         mAddTracksForMapPerTF > 0 ? mAddTracksForMapPerTF : remSeeds.size(),
         remSeeds.size(), mSeeds.size() - nSeeds, mTrackDataCompact.size());
  }
  int extraChecked = 0;
  for (int iSeed : remSeeds) {
    if (mAddTracksForMapPerTF > 0 && mTrackDataCompact.size() >= mMaxTracksPerTF + mAddTracksForMapPerTF) {
      LOGP(info, "Maximum number {} of additional tracks per TF reached. Skipping the remaining {} tracks", mAddTracksForMapPerTF, remSeeds.size() - extraChecked);
      break;
    }
    extraChecked++;
    if (mGIDs[iSeed].includesDet(DetID::TRD) || mGIDs[iSeed].includesDet(DetID::TOF)) {
      interpolateTrack(iSeed);
      LOGP(debug, "extra check {} of {}, seed {} interpolateTrack {}, used: {}", extraChecked, remSeeds.size(), iSeed, GTrackID::getSourceName(mGIDs[iSeed].getSource()), mTrackDataCompact.size());
    } else {
      LOGP(debug, "extra check {} of {}, seed {} extrapolateTrack {}, used: {}", extraChecked, remSeeds.size(), iSeed, GTrackID::getSourceName(mGIDs[iSeed].getSource()), mTrackDataCompact.size());
      extrapolateTrack(iSeed);
    }
  }
  LOGP(info, "Could process {} tracks successfully ({} rejected in refits, {} in propagation, {} as loopers), {} residuals were rejected, {} accepted",
       mTrackData.size(), mNRejRefit, mNRejProp, mNRejLoop, mRejectedResiduals, mClRes.size());
  mRejectedResiduals = 0;
  mNRejRefit = 0;
  mNRejProp = 0;
  mNRejLoop = 0;
}

void TrackInterpolation::interpolateTrack(int iSeed)
{
  LOGP(debug, "Starting track interpolation for GID {}", mGIDs[iSeed].asString());
  TrackData trackData;
  o2::trd::Tracklet64 trkl64;
  o2::trd::CalibratedTracklet trklCalib;
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
  if (mParams->refitITS && !refITSTrack(gidTable[GTrackID::ITS], iSeed)) {
    mNRejRefit++;
    return;
  }
  trackData.gid = mGIDs[iSeed];
  trackData.par = mSeeds[iSeed];
  auto trkWork = mSeeds[iSeed];
  o2::track::TrackPar trkInner{trkWork};
  // reset the cache array (sufficient to set cluster available to zero)
  for (auto& elem : mCache) {
    elem.clAvailable = 0;
  }
  trackData.clIdx.setFirstEntry(mClRes.size()); // reference the first cluster residual belonging to this track
  float clusterTimeBinOffset = mTrackTimes[iSeed] / mTPCTimeBinMUS;

  // store the TPC cluster positions in the cache, as well as dedx info
  std::array<std::pair<uint16_t, uint16_t>, constants::MAXGLOBALPADROW> mCacheDEDX{};
  std::array<short, constants::MAXGLOBALPADROW> multBins{};
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
    mCacheDEDX[row].first = clTPC.getQtot();
    mCacheDEDX[row].second = clTPC.getQmax();
    int imb = int(clTPC.getTime() * mNTPCOccBinLengthInv);
    if (imb < mTPCParam->occupancyMapSize) {
      multBins[row] = 1 + std::max(0, imb);
    }
  }

  // extrapolate seed through TPC and store track position at each pad row
  for (int iRow = 0; iRow < param::NPadRows; ++iRow) {
    if (!mCache[iRow].clAvailable) {
      continue;
    }
    if (!trkWork.rotate(mCache[iRow].clAngle)) {
      LOG(debug) << "Failed to rotate track during first extrapolation";
      mNRejProp++;
      return;
    }
    if (!propagator->PropagateToXBxByBz(trkWork, param::RowX[iRow], mParams->maxSnp, mParams->maxStep, mMatCorr)) {
      LOG(debug) << "Failed on first extrapolation";
      mNRejProp++;
      return;
    }
    mCache[iRow].y[ExtOut] = trkWork.getY();
    mCache[iRow].z[ExtOut] = trkWork.getZ();
    mCache[iRow].sy2[ExtOut] = trkWork.getSigmaY2();
    mCache[iRow].szy[ExtOut] = trkWork.getSigmaZY();
    mCache[iRow].sz2[ExtOut] = trkWork.getSigmaZ2();
    mCache[iRow].snp[ExtOut] = trkWork.getSnp();
    // printf("Track alpha at row %i: %.2f, Y(%.2f), Z(%.2f)\n", iRow, trkWork.getAlpha(), trkWork.getY(), trkWork.getZ());
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
      mNRejProp++;
      return;
    }
    float clTOFxyz[3] = {clTOF.getX(), clTOF.getY(), clTOF.getZ()};
    if (!clTOF.isInNominalSector()) {
      o2::tof::Geo::alignedToNominalSector(clTOFxyz, clTOFSec); // go from the aligned to nominal sector frame
    }
    std::array<float, 2> clTOFYZ{clTOFxyz[1], clTOFxyz[2]};
    std::array<float, 3> clTOFCov{mParams->sigYZ2TOF, 0.f, mParams->sigYZ2TOF}; // assume no correlation between y and z and equal cluster error sigma^2 = (3cm)^2 / 12
    if (!propagator->PropagateToXBxByBz(trkWork, clTOFxyz[0], mParams->maxSnp, mParams->maxStep, mMatCorr)) {
      LOG(debug) << "Failed final propagation to TOF radius";
      mNRejProp++;
      return;
    }
    // TODO: check if reset of covariance matrix is needed here (or, in case TOF point is not available at outermost TRD layer)
    if (!trkWork.update(clTOFYZ, clTOFCov)) {
      LOG(debug) << "Failed to update extrapolated ITS track with TOF cluster";
      // LOGF(info, "trkWork.y=%f, cl.y=%f, trkWork.z=%f, cl.z=%f", trkWork.getY(), clTOFYZ[0], trkWork.getZ(), clTOFYZ[1]);
      mNRejProp++;
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
      std::array<float, 2> trkltTRDYZ{};
      std::array<float, 3> trkltTRDCov{};
      int res = processTRDLayer(trkTRD, iLayer, trkWork, &trkltTRDYZ, &trkltTRDCov);
      if (res == -1) { // no TRD tracklet in this layer
        continue;
      }
      if (res < -1) { // failed to reach this layer
        return;
      }
      if (!trkWork.update(trkltTRDYZ, trkltTRDCov)) {
        LOG(debug) << "Failed to update track at TRD layer " << iLayer;
        mNRejProp++;
        return;
      }
    }
  }

  if (mDumpTrackPoints) {
    (*trackDataExtended).trkOuter = trkWork;
  }
  auto trkOuter = trkWork; // outer param

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
      mNRejProp++;
      return;
    }
    if (!propagator->PropagateToXBxByBz(trkWork, param::RowX[iRow], mParams->maxSnp, mParams->maxStep, mMatCorr)) {
      LOG(debug) << "Failed on back propagation";
      // printf("trkX(%.2f), clX(%.2f), clY(%.2f), clZ(%.2f), alphaTOF(%.2f)\n", trkWork.getX(), param::RowX[iRow], clTOFYZ[0], clTOFYZ[1], clTOFAlpha);
      mNRejProp++;
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

    const auto dY = mCache[iRow].clY - mCache[iRow].y[Int];
    const auto dZ = mCache[iRow].clZ - mCache[iRow].z[Int];
    const auto y = mCache[iRow].y[Int];
    const auto z = mCache[iRow].z[Int];
    const auto snp = mCache[iRow].snp[Int];
    const auto sec = mCache[iRow].clSec;
    clusterResiduals.emplace_back(dY, dZ, y, z, snp, sec, deltaRow);

    deltaRow = 1;
  }
  trackData.chi2TRD = gidTable[GTrackID::TRD].isIndexSet() ? mRecoCont->getITSTPCTRDTrack<o2::trd::TrackTRD>(gidTable[GTrackID::ITSTPCTRD]).getChi2() : 0;
  trackData.chi2TPC = trkTPC.getChi2();
  trackData.chi2ITS = trkITS.getChi2();
  trackData.nClsTPC = trkTPC.getNClusterReferences();
  trackData.nClsITS = trkITS.getNumberOfClusters();
  trackData.nTrkltsTRD = gidTable[GTrackID::TRD].isIndexSet() ? mRecoCont->getITSTPCTRDTrack<o2::trd::TrackTRD>(gidTable[GTrackID::ITSTPCTRD]).getNtracklets() : 0;

  double t0forTOF = 0.; // to be set if TOF is matched
  float t0forTOFwithinBC = 0.f;
  float t0forTOFres = 9999.f;

  if (gidTable[GTrackID::TOF].isIndexSet()) {
    const auto& tofMatch = mRecoCont->getTOFMatch(mGIDs[iSeed]);
    ULong64_t bclongtof = (tofMatch.getSignal() - 10000) * o2::tof::Geo::BC_TIME_INPS_INV;
    t0forTOF = tofMatch.getFT0Best(); // setting t0 for TOF
    t0forTOFwithinBC = t0forTOF - bclongtof * o2::tof::Geo::BC_TIME_INPS;
    t0forTOFres = tofMatch.getFT0BestRes();
    trackData.deltaTOF = tofMatch.getSignal() - t0forTOF - tofMatch.getLTIntegralOut().getTOF(trkTPC.getPID().getID());
    trackData.clAvailTOF = uint16_t(t0forTOFres);
  } else {
    trackData.clAvailTOF = 0;
  }
  trackData.dEdxTPC = trkTPC.getdEdx().dEdxTotTPC;

  mTrackValidation.clear(); // for refitted track parameters and flagging rejected clusters

  bool stored = false;
  trackData.filterFlag = mParams->skipOutlierFiltering ? -1 : validateTrack(trackData, mTrackValidation, clusterResiduals, true);
  if (trackData.filterFlag <= 0 || mParams->writeUnfiltered) {
    int nClValidated = 0;
    int iRow = 0;
    for (unsigned int iCl = 0; iCl < clusterResiduals.size(); ++iCl) {
      iRow += clusterResiduals[iCl].dRow;
      const auto rej = trackData.filterFlag < 0 ? false : mTrackValidation.points[iCl].flagRej;
      if (rej && !mParams->keepRejectedResiduals) { // skip masked cluster residual
        continue;
      }
      const float tgPhi = clusterResiduals[iCl].snp / std::sqrt((1.f - clusterResiduals[iCl].snp) * (1.f + clusterResiduals[iCl].snp));
      const auto dy = clusterResiduals[iCl].dy;
      const auto dz = clusterResiduals[iCl].dz;
      const auto y = clusterResiduals[iCl].y;
      const auto z = clusterResiduals[iCl].z;
      const auto sec = clusterResiduals[iCl].sec;
      if ((std::abs(dy) < param::MaxResid) && (std::abs(dz) < param::MaxResid) && (std::abs(y) < param::MaxY) && (std::abs(z) < param::MaxZ) && (std::abs(tgPhi) < param::MaxTgSlp)) {
        mClRes.emplace_back(dy, dz, tgPhi, y, z, iRow, sec, -1, rej);
        mDetInfoRes.emplace_back().setTPC(mCacheDEDX[iRow].first, mCacheDEDX[iRow].second); // qtot, qmax
        ++nClValidated;
      } else {
        ++mRejectedResiduals;
      }
    }
    trackData.clIdx.setEntries(nClValidated);

    // store multiplicity info
    for (int ist = 0; ist < NSTACKS; ist++) {
      int mltBinMin = 0x7ffff, mltBinMax = -1, prevBin = -1;
      for (int ir = STACKROWS[ist]; ir < STACKROWS[ist + 1]; ir++) {
        if (multBins[ir] != prevBin && multBins[ir] > 0) { // there is a cluster different from previous one
          prevBin = multBins[ir];
          if (multBins[ir] > mltBinMax) {
            mltBinMax = multBins[ir];
          }
          if (multBins[ir] < mltBinMin) {
            mltBinMin = multBins[ir];
          }
        }
      }
      if (--mltBinMin >= 0) { // we were offsetting bin IDs by 1!
        float avMlt = 0;
        for (int ib = mltBinMin; ib < mltBinMax; ib++) {
          avMlt += mTPCParam->occupancyMap[ib];
        }
        avMlt /= (mltBinMax - mltBinMin);
        trackData.setMultStack(avMlt, ist);
      }
    }

    bool stopPropagation = !mExtDetResid;
    if (!stopPropagation) {
      // do we have TRD residuals to add?
      trkWork = trkOuter;
      if (gidTable[GTrackID::TRD].isIndexSet()) {
        const auto& trkTRD = mRecoCont->getITSTPCTRDTrack<o2::trd::TrackTRD>(gidTable[GTrackID::ITSTPCTRD]);
        for (int iLayer = 0; iLayer < o2::trd::constants::NLAYER; iLayer++) {
          std::array<float, 2> trkltTRDYZ{};
          int res = processTRDLayer(trkTRD, iLayer, trkWork, &trkltTRDYZ, nullptr, &trackData, &trkl64, &trklCalib);
          if (res == -1) { // no traklet on this layer
            continue;
          }
          if (res < -1) { // failed to reach this layer
            stopPropagation = true;
            break;
          }

          float tgPhi = trkWork.getSnp() / std::sqrt((1.f - trkWork.getSnp()) * (1.f + trkWork.getSnp()));
          auto dy = trkltTRDYZ[0] - trkWork.getY();
          auto dz = trkltTRDYZ[1] - trkWork.getZ();
          if ((std::abs(dy) < param::MaxResid) && (std::abs(dz) < param::MaxResid) && (std::abs(trkWork.getY()) < param::MaxY) && (std::abs(trkWork.getZ()) < param::MaxZ) && (std::abs(tgPhi) < param::MaxTgSlp)) {
            mClRes.emplace_back(dy, dz, tgPhi, trkWork.getY(), trkWork.getZ(), 160 + iLayer, o2::math_utils::angle2Sector(trkWork.getAlpha()), (short)res);
            mDetInfoRes.emplace_back().setTRD(trkl64.getQ0(), trkl64.getQ1(), trkl64.getQ2(), trklCalib.getDy()); // q0,q1,q2,slope
            trackData.nExtDetResid++;
          }
        }
      }

      // do we have TOF residual to add?
      while (gidTable[GTrackID::TOF].isIndexSet() && !stopPropagation) {
        const auto& clTOF = mRecoCont->getTOFClusters()[gidTable[GTrackID::TOF]];
        float clTOFxyz[3] = {clTOF.getX(), clTOF.getY(), clTOF.getZ()};
        if (!clTOF.isInNominalSector()) {
          o2::tof::Geo::alignedToNominalSector(clTOFxyz, clTOF.getCount()); // go from the aligned to nominal sector frame
        }
        const float clTOFAlpha = o2::math_utils::sector2Angle(clTOF.getCount());
        if (trkWork.getAlpha() != clTOFAlpha && !trkWork.rotate(clTOFAlpha)) {
          LOG(debug) << "Failed to rotate into TOF cluster sector frame";
          stopPropagation = true;
          break;
        }
        if (!propagator->PropagateToXBxByBz(trkWork, clTOFxyz[0], mParams->maxSnp, mParams->maxStep, mMatCorr)) {
          LOG(debug) << "Failed final propagation to TOF radius";
          break;
        }

        float tgPhi = trkWork.getSnp() / std::sqrt((1.f - trkWork.getSnp()) * (1.f + trkWork.getSnp()));
        auto dy = clTOFxyz[1] - trkWork.getY();
        auto dz = clTOFxyz[2] - trkWork.getZ();
        // get seeding track time

        if ((std::abs(dy) < param::MaxResid) && (std::abs(dz) < param::MaxResid) && (std::abs(trkWork.getY()) < param::MaxY) && (std::abs(trkWork.getZ()) < param::MaxZ) && (std::abs(tgPhi) < param::MaxTgSlp)) {
          mClRes.emplace_back(dy, dz, tgPhi, trkWork.getY(), trkWork.getZ(), 170, clTOF.getCount(), clTOF.getPadInSector());
          // get seeding track time
          if (!gidTable[GTrackID::ITSTPC].isIndexSet()) {
            LOGP(fatal, "ITS-TPC seed index is not set for TOF track");
          }
          float tdif = static_cast<float>(clTOF.getTime() - t0forTOF); // time in \mus wrt interaction time0
          mDetInfoRes.emplace_back().setTOF(tdif * 1e-6);
          trackData.nExtDetResid++;
        }
        break;
      }

      // add ITS residuals
      while (!stopPropagation) {
        auto& trkWorkITS = trkInner; // this is ITS outer param
        auto nCl = trkITS.getNumberOfClusters();
        auto clEntry = trkITS.getFirstClusterEntry();
        auto geom = o2::its::GeometryTGeo::Instance();
        for (int iCl = 0; iCl < nCl; iCl++) { // clusters are stored from outer to inner layers
          const auto& cls = mITSClustersArray[mITSTrackClusIdx[clEntry + iCl]];
          int chip = cls.getSensorID();
          float chipX, chipAlpha;
          geom->getSensorXAlphaRefPlane(cls.getSensorID(), chipX, chipAlpha);
          if (!trkWorkITS.rotate(chipAlpha) || !propagator->PropagateToXBxByBz(trkWorkITS, chipX, mParams->maxSnp, mParams->maxStep, mMatCorr)) {
            LOGP(debug, "Failed final propagation to ITS X={} alpha={}", chipX, chipAlpha);
            stopPropagation = true;
            break;
          }
          float tgPhi = trkWorkITS.getSnp() / std::sqrt((1.f - trkWorkITS.getSnp()) * (1.f + trkWorkITS.getSnp()));
          auto dy = cls.getY() - trkWorkITS.getY();
          auto dz = cls.getZ() - trkWorkITS.getZ();
          if ((std::abs(dy) < param::MaxResid) && (std::abs(dz) < param::MaxResid) && (std::abs(trkWorkITS.getY()) < param::MaxY) && (std::abs(trkWorkITS.getZ()) < param::MaxZ) && (std::abs(tgPhi) < param::MaxTgSlp)) {
            mClRes.emplace_back(dy, dz, tgPhi, trkWorkITS.getY(), trkWorkITS.getZ(), 180 + geom->getLayer(cls.getSensorID()), -1, cls.getSensorID());
            mDetInfoRes.emplace_back(); // empty placeholder
            trackData.nExtDetResid++;
          }
        }
        if (!stopPropagation) { // add residual to PV
          const auto& pv = mRecoCont->getPrimaryVertices()[mTrackPVID[iSeed]];
          o2::math_utils::Point3D<float> vtx{pv.getX(), pv.getY(), pv.getZ()};
          if (!propagator->propagateToDCA(vtx, trkWorkITS, mBz, mParams->maxStep, mMatCorr)) {
            LOGP(debug, "Failed propagation to DCA to PV ({} {} {}), {}", pv.getX(), pv.getY(), pv.getZ(), trkWorkITS.asString());
            stopPropagation = true;
            break;
          }
          // rotate PV to the track frame
          float sn, cs, alpha = trkWorkITS.getAlpha();
          math_utils::detail::bringToPMPi(alpha);
          math_utils::detail::sincos<float>(alpha, sn, cs);
          float xv = vtx.X() * cs + vtx.Y() * sn, yv = -vtx.X() * sn + vtx.Y() * cs, zv = vtx.Z();
          auto dy = yv - trkWorkITS.getY();
          auto dz = zv - trkWorkITS.getZ();
          if ((std::abs(dy) < param::MaxResid) && (std::abs(dz) < param::MaxResid) && (std::abs(trkWorkITS.getY()) < param::MaxY) && (std::abs(trkWorkITS.getZ()) < param::MaxZ) && std::abs(xv) < param::MaxVtxX) {
            short compXV = static_cast<short>(xv * 0x7fff / param::MaxVtxX);
            mClRes.emplace_back(dy, dz, alpha / TMath::Pi(), trkWorkITS.getY(), trkWorkITS.getZ(), 190, -1, compXV);
            if (!gidTable[GTrackID::ITSTPC].isIndexSet()) {
              LOGP(fatal, "ITS-TPC seed index is not set for TOF track");
            }
            float tdif = pv.getTimeStamp().getTimeStamp() - mRecoCont->getTPCITSTrack(gidTable[GTrackID::ITSTPC]).getTimeMUS().getTimeStamp();
            mDetInfoRes.emplace_back().setPV(tdif); // time in \mus wrt seeding ITS-TPC track
            trackData.nExtDetResid++;
          }
        }
        break;
      }
    }

    mGIDsSuccess.push_back(mGIDs[iSeed]);
    mTrackDataCompact.emplace_back(trackData.clIdx.getFirstEntry(), trackData.multStack, nClValidated, mGIDs[iSeed].getSource(), trackData.nExtDetResid, trackData.filterFlag);
    mTrackData.push_back(std::move(trackData));
    stored = true;
    if (mDumpTrackPoints) {
      (*trackDataExtended).clIdx.setEntries(nClValidated);
      (*trackDataExtended).nExtDetResid = trackData.nExtDetResid;
      (*trackDataExtended).filterFlag = trackData.filterFlag;
      mTrackDataExtended.push_back(std::move(*trackDataExtended));
    }
  }
  if (mParams->writeValidationData && trackData.filterFlag >= 0 && mDBGOut) {
    (*mDBGOut) << "valdata" << "params=" << mTrackValidation << "trackData=" << (stored ? mTrackData.back() : trackData) << "\n";
  }
}

int TrackInterpolation::processTRDLayer(const o2::trd::TrackTRD& trkTRD, int iLayer, o2::track::TrackParCov& trkWork,
                                        std::array<float, 2>* trkltTRDYZ, std::array<float, 3>* trkltTRDCov, TrackData* trkData,
                                        o2::trd::Tracklet64* trk64, o2::trd::CalibratedTracklet* trkCalib)
{
  // return chamber ID (0:539) in case of successful processing, -1 if there is no TRD tracklet at given layer, -2 if processing failed
  int trkltIdx = trkTRD.getTrackletIndex(iLayer);
  if (trkltIdx < 0) {
    return -1; // no TRD tracklet in this layer
  }
  const auto& trdSP = mRecoCont->getTRDCalibratedTracklets()[trkltIdx];
  const auto& trdTrklt = mRecoCont->getTRDTracklets()[trkltIdx];
  auto trkltDet = trdTrklt.getDetector();
  auto trkltSec = trkltDet / (o2::trd::constants::NLAYER * o2::trd::constants::NSTACK);
  if (trkltSec != o2::math_utils::angle2Sector(trkWork.getAlpha())) {
    if (!trkWork.rotate(o2::math_utils::sector2Angle(trkltSec))) {
      LOG(debug) << "Track could not be rotated in TRD tracklet coordinate system in layer " << iLayer;
      return -2;
    }
  }
  if (!o2::base::Propagator::Instance()->PropagateToXBxByBz(trkWork, trdSP.getX(), mParams->maxSnp, mParams->maxStep, mMatCorr)) {
    LOG(debug) << "Failed propagation to TRD layer " << iLayer;
    return -2;
  }
  if (trkltTRDYZ) {
    const auto* pad = mGeoTRD->getPadPlane(trkltDet);
    float tilt = tan(TMath::DegToRad() * pad->getTiltingAngle()); // tilt is signed! and returned in degrees
    float tiltCorrUp = tilt * (trdSP.getZ() - trkWork.getZ());
    float zPosCorrUp = trdSP.getZ() + mRecoParam.getZCorrCoeffNRC() * trkWork.getTgl(); // maybe Z can be corrected on avarage already by the tracklet transformer?
    float padLength = pad->getRowSize(trdTrklt.getPadRow());
    if (!((trkWork.getSigmaZ2() < (padLength * padLength / 12.f)) && (std::abs(trdSP.getZ() - trkWork.getZ()) < padLength))) {
      tiltCorrUp = 0.f;
    }
    (*trkltTRDYZ)[0] = trdSP.getY() - tiltCorrUp;
    (*trkltTRDYZ)[1] = zPosCorrUp;
    if (trkltTRDCov) {
      mRecoParam.recalcTrkltCov(tilt, trkWork.getSnp(), pad->getRowSize(trdTrklt.getPadRow()), *trkltTRDCov);
    }
  }
  if (trkData) {
    auto slope = trdSP.getDy();
    if (std::abs(slope) < param::MaxTRDSlope) {
      trkData->TRDTrkltSlope[iLayer] = slope * 0x7fff / param::MaxTRDSlope;
    }
  }
  if (trk64) {
    *trk64 = trdTrklt;
  }
  if (trkCalib) {
    *trkCalib = trdSP;
  }
  return trkltDet;
}

void TrackInterpolation::extrapolateTrack(int iSeed)
{
  // extrapolate ITS-only track through TPC and store residuals to TPC clusters in the output vectors
  LOGP(debug, "Starting track extrapolation for GID {}", mGIDs[iSeed].asString());
  const auto& gidTable = mGIDtables[iSeed];
  TrackData trackData;
  o2::trd::Tracklet64 trkl64;
  o2::trd::CalibratedTracklet trklCalib;
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
  if (mParams->refitITS && !refITSTrack(gidTable[GTrackID::ITS], iSeed)) {
    mNRejRefit++;
    return;
  }
  trackData.gid = mGIDs[iSeed];
  trackData.par = mSeeds[iSeed];

  auto trkWork = mSeeds[iSeed];
  float clusterTimeBinOffset = mTrackTimes[iSeed] / mTPCTimeBinMUS;
  auto propagator = o2::base::Propagator::Instance();
  unsigned short rowPrev = 0; // used to calculate dRow of two consecutive cluster residuals
  unsigned short nMeasurements = 0;
  uint8_t clRowPrev = constants::MAXGLOBALPADROW; // used to identify and skip split clusters on the same pad row
  std::array<std::pair<uint16_t, uint16_t>, constants::MAXGLOBALPADROW> mCacheDEDX{};
  std::array<short, constants::MAXGLOBALPADROW> multBins{};
  for (int iCl = trkTPC.getNClusterReferences(); iCl--;) {
    uint8_t sector, row;
    uint32_t clusterIndexInRow;
    const auto& cl = trkTPC.getCluster(mTPCTracksClusIdx, iCl, *mTPCClusterIdxStruct, sector, row);
    if (clRowPrev == row) {
      // if there are split clusters we only take the first one on the pad row
      continue;
    } else if (clRowPrev < constants::MAXGLOBALPADROW && clRowPrev > row) {
      // we seem to be looping, abort this track
      LOGP(debug, "TPC track with pT={} GeV and {} clusters has cluster {} on row {} while the previous cluster was on row {}",
           mSeeds[iSeed].getPt(), trkTPC.getNClusterReferences(), iCl, row, clRowPrev);
      mNRejLoop++;
      return;
    } else {
      // this is the first cluster we see on this pad row
      clRowPrev = row;
    }
    float x = 0, y = 0, z = 0;
    mFastTransform->TransformIdeal(sector, row, cl.getPad(), cl.getTime(), x, y, z, clusterTimeBinOffset);
    if (!trkWork.rotate(o2::math_utils::sector2Angle(sector))) {
      mNRejProp++;
      return;
    }
    if (!propagator->PropagateToXBxByBz(trkWork, x, mParams->maxSnp, mParams->maxStep, mMatCorr)) {
      mNRejProp++;
      return;
    }

    const auto dY = y - trkWork.getY();
    const auto dZ = z - trkWork.getZ();
    const auto ty = trkWork.getY();
    const auto tz = trkWork.getZ();
    const auto snp = trkWork.getSnp();
    const auto sec = sector;
    clusterResiduals.emplace_back(dY, dZ, ty, tz, snp, sec, row - rowPrev);
    mCacheDEDX[row].first = cl.getQtot();
    mCacheDEDX[row].second = cl.getQmax();
    rowPrev = row;
    int imb = int(cl.getTime() * mNTPCOccBinLengthInv);
    if (imb < mTPCParam->occupancyMapSize) {
      multBins[row] = 1 + std::max(0, imb);
    }
    ++nMeasurements;
  }

  mTrackValidation.clear(); // for refitted track parameters and flagging rejected clusters
  if (clusterResiduals.size() > constants::MAXGLOBALPADROW) {
    LOGP(warn, "Extrapolated ITS-TPC track and found more residuals than possible ({})", clusterResiduals.size());
    mNRejLoop++;
    return;
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

  bool stored = false;
  trackData.filterFlag = mParams->skipOutlierFiltering ? -1 : validateTrack(trackData, mTrackValidation, clusterResiduals, false);
  if (trackData.filterFlag <= 0 || mParams->writeUnfiltered) {
    int nClValidated = 0, iRow = 0;
    unsigned int iCl = 0;
    for (iCl = 0; iCl < clusterResiduals.size(); ++iCl) {
      iRow += clusterResiduals[iCl].dRow;
      if (iRow >= param::NPadRows) { // RS why do we need this?
        continue;
      }
      const auto rej = trackData.filterFlag < 0 ? false : mTrackValidation.points[iCl].flagRej;
      if (rej && !mParams->keepRejectedResiduals) { // skip masked cluster residual
        continue;
      }
      const float tgPhi = clusterResiduals[iCl].snp / std::sqrt((1.f - clusterResiduals[iCl].snp) * (1.f + clusterResiduals[iCl].snp));
      const auto dy = clusterResiduals[iCl].dy;
      const auto dz = clusterResiduals[iCl].dz;
      const auto y = clusterResiduals[iCl].y;
      const auto z = clusterResiduals[iCl].z;
      if ((std::abs(dy) < param::MaxResid) && (std::abs(dz) < param::MaxResid) && (std::abs(y) < param::MaxY) && (std::abs(z) < param::MaxZ) && (std::abs(tgPhi) < param::MaxTgSlp)) {
        mClRes.emplace_back(dy, dz, tgPhi, y, z, iRow, clusterResiduals[iCl].sec, -1, rej);
        mDetInfoRes.emplace_back().setTPC(mCacheDEDX[iRow].first, mCacheDEDX[iRow].second); // qtot, qmax
        ++nClValidated;
      } else {
        ++mRejectedResiduals;
      }
    }
    trackData.clIdx.setEntries(nClValidated);

    // store multiplicity info
    for (int ist = 0; ist < NSTACKS; ist++) {
      int mltBinMin = 0x7ffff, mltBinMax = -1, prevBin = -1;
      for (int ir = STACKROWS[ist]; ir < STACKROWS[ist + 1]; ir++) {
        if (multBins[ir] != prevBin && multBins[ir] > 0) { // there is a cluster
          prevBin = multBins[ir];
          if (multBins[ir] > mltBinMax) {
            mltBinMax = multBins[ir];
          }
          if (multBins[ir] < mltBinMin) {
            mltBinMin = multBins[ir];
          }
        }
      }
      if (--mltBinMin >= 0) { // we were offsetting bin IDs by 1!
        float avMlt = 0;
        for (int ib = mltBinMin; ib < mltBinMax; ib++) {
          avMlt += mTPCParam->occupancyMap[ib];
        }
        avMlt /= (mltBinMax - mltBinMin);
        trackData.setMultStack(avMlt, ist);
      }
    }

    bool stopPropagation = !mExtDetResid;
    if (!stopPropagation) {
      // do we have TRD residuals to add?
      int iSeedFull = mParentID[iSeed] == -1 ? iSeed : mParentID[iSeed];
      auto gidFull = mGIDs[iSeedFull];
      const auto& gidTableFull = mGIDtables[iSeedFull];
      if (gidTableFull[GTrackID::TRD].isIndexSet()) {
        const auto& trkTRD = mRecoCont->getITSTPCTRDTrack<o2::trd::TrackTRD>(gidTableFull[GTrackID::ITSTPCTRD]);
        trackData.nTrkltsTRD = trkTRD.getNtracklets();
        for (int iLayer = 0; iLayer < o2::trd::constants::NLAYER; iLayer++) {
          std::array<float, 2> trkltTRDYZ{};
          int res = processTRDLayer(trkTRD, iLayer, trkWork, &trkltTRDYZ, nullptr, &trackData, &trkl64, &trklCalib);
          if (res == -1) { // no traklet on this layer
            continue;
          }
          if (res < -1) { // failed to reach this layer
            stopPropagation = true;
            break;
          }

          float tgPhi = trkWork.getSnp() / std::sqrt((1.f - trkWork.getSnp()) * (1.f + trkWork.getSnp()));
          auto dy = trkltTRDYZ[0] - trkWork.getY();
          auto dz = trkltTRDYZ[1] - trkWork.getZ();
          const auto sec = clusterResiduals[iCl].sec;
          if ((std::abs(dy) < param::MaxResid) && (std::abs(dz) < param::MaxResid) && (std::abs(trkWork.getY()) < param::MaxY) && (std::abs(trkWork.getZ()) < param::MaxZ) && (std::abs(tgPhi) < param::MaxTgSlp)) {
            mClRes.emplace_back(dy, dz, tgPhi, trkWork.getY(), trkWork.getZ(), 160 + iLayer, o2::math_utils::angle2Sector(trkWork.getAlpha()), (short)res);
            mDetInfoRes.emplace_back().setTRD(trkl64.getQ0(), trkl64.getQ1(), trkl64.getQ2(), trklCalib.getDy()); // q0,q1,q2,slope
            trackData.nExtDetResid++;
          }
        }
      }

      // do we have TOF residual to add?
      trackData.clAvailTOF = 0;
      while (gidTableFull[GTrackID::TOF].isIndexSet() && !stopPropagation) {
        const auto& tofMatch = mRecoCont->getTOFMatch(gidFull);
        ULong64_t bclongtof = (tofMatch.getSignal() - 10000) * o2::tof::Geo::BC_TIME_INPS_INV;
        double t0forTOF = tofMatch.getFT0Best(); // setting t0 for TOF
        float t0forTOFwithinBC = t0forTOF - bclongtof * o2::tof::Geo::BC_TIME_INPS;
        float t0forTOFres = tofMatch.getFT0BestRes();
        trackData.deltaTOF = tofMatch.getSignal() - t0forTOF - tofMatch.getLTIntegralOut().getTOF(trkTPC.getPID().getID());
        trackData.clAvailTOF = uint16_t(t0forTOFres);
        const auto& clTOF = mRecoCont->getTOFClusters()[gidTableFull[GTrackID::TOF]];
        const float clTOFAlpha = o2::math_utils::sector2Angle(clTOF.getCount());
        float clTOFxyz[3] = {clTOF.getX(), clTOF.getY(), clTOF.getZ()};
        if (!clTOF.isInNominalSector()) {
          o2::tof::Geo::alignedToNominalSector(clTOFxyz, clTOF.getCount()); // go from the aligned to nominal sector frame
        }
        if (trkWork.getAlpha() != clTOFAlpha && !trkWork.rotate(clTOFAlpha)) {
          LOG(debug) << "Failed to rotate into TOF cluster sector frame";
          stopPropagation = true;
          break;
        }
        if (!propagator->PropagateToXBxByBz(trkWork, clTOFxyz[0], mParams->maxSnp, mParams->maxStep, mMatCorr)) {
          LOG(debug) << "Failed final propagation to TOF radius";
          break;
        }

        float tgPhi = trkWork.getSnp() / std::sqrt((1.f - trkWork.getSnp()) * (1.f + trkWork.getSnp()));
        auto dy = clTOFxyz[1] - trkWork.getY();
        auto dz = clTOFxyz[2] - trkWork.getZ();
        if ((std::abs(dy) < param::MaxResid) && (std::abs(dz) < param::MaxResid) && (std::abs(trkWork.getY()) < param::MaxY) && (std::abs(trkWork.getZ()) < param::MaxZ) && (std::abs(tgPhi) < param::MaxTgSlp)) {
          mClRes.emplace_back(dy, dz, tgPhi, trkWork.getY(), trkWork.getZ(), 170, clTOF.getCount(), clTOF.getPadInSector());
          // get seeding track time
          if (!gidTableFull[GTrackID::ITSTPC].isIndexSet()) {
            LOGP(fatal, "ITS-TPC seed index is not set for TOF track");
          }

          float tdif = static_cast<float>(clTOF.getTime() - t0forTOF); // time in \mus wrt interaction time0
          mDetInfoRes.emplace_back().setTOF(tdif * 1e-6);              // time in \mus wrt seeding ITS-TPC track
          trackData.nExtDetResid++;
        }
        break;
      }

      // add ITS residuals
      while (!stopPropagation) {
        o2::track::TrackPar trkWorkITS{trackData.par}; // this is ITS outer param
        auto nCl = trkITS.getNumberOfClusters();
        auto clEntry = trkITS.getFirstClusterEntry();
        auto geom = o2::its::GeometryTGeo::Instance();
        for (int iCl = 0; iCl < nCl; iCl++) { // clusters are stored from outer to inner layers
          const auto& cls = mITSClustersArray[mITSTrackClusIdx[clEntry + iCl]];
          int chip = cls.getSensorID();
          float chipX, chipAlpha;
          geom->getSensorXAlphaRefPlane(cls.getSensorID(), chipX, chipAlpha);
          if (!trkWorkITS.rotate(chipAlpha) || !propagator->propagateToX(trkWorkITS, chipX, mBz, mParams->maxSnp, mParams->maxStep, mMatCorr)) {
            LOGP(debug, "Failed final propagation to ITS X={} alpha={}", chipX, chipAlpha);
            stopPropagation = true;
            break;
          }
          float tgPhi = trkWorkITS.getSnp() / std::sqrt((1.f - trkWorkITS.getSnp()) * (1.f + trkWorkITS.getSnp()));
          auto dy = cls.getY() - trkWorkITS.getY();
          auto dz = cls.getZ() - trkWorkITS.getZ();
          if ((std::abs(dy) < param::MaxResid) && (std::abs(dz) < param::MaxResid) && (std::abs(trkWorkITS.getY()) < param::MaxY) && (std::abs(trkWorkITS.getZ()) < param::MaxZ) && (std::abs(tgPhi) < param::MaxTgSlp)) {
            mClRes.emplace_back(dy, dz, tgPhi, trkWorkITS.getY(), trkWorkITS.getZ(), 180 + geom->getLayer(cls.getSensorID()), -1, cls.getSensorID());
            mDetInfoRes.emplace_back(); // empty placeholder
            trackData.nExtDetResid++;
          }
        }
        if (!stopPropagation) { // add residual to PV
          const auto& pv = mRecoCont->getPrimaryVertices()[mTrackPVID[iSeed]];
          o2::math_utils::Point3D<float> vtx{pv.getX(), pv.getY(), pv.getZ()};
          if (!propagator->propagateToDCA(vtx, trkWorkITS, mBz, mParams->maxStep, mMatCorr)) {
            LOGP(debug, "Failed propagation to DCA to PV ({} {} {}), {}", pv.getX(), pv.getY(), pv.getZ(), trkWorkITS.asString());
            stopPropagation = true;
            break;
          }
          // rotate PV to the track frame
          float sn, cs, alpha = trkWorkITS.getAlpha();
          math_utils::detail::bringToPMPi(alpha);
          math_utils::detail::sincos<float>(alpha, sn, cs);
          float xv = vtx.X() * cs + vtx.Y() * sn, yv = -vtx.X() * sn + vtx.Y() * cs, zv = vtx.Z();
          auto dy = yv - trkWorkITS.getY();
          auto dz = zv - trkWorkITS.getZ();
          if ((std::abs(dy) < param::MaxResid) && (std::abs(dz) < param::MaxResid) && (std::abs(trkWorkITS.getY()) < param::MaxY) && (std::abs(trkWorkITS.getZ()) < param::MaxZ) && std::abs(xv) < param::MaxVtxX) {
            short compXV = static_cast<short>(xv * 0x7fff / param::MaxVtxX);
            mClRes.emplace_back(dy, dz, alpha / TMath::Pi(), trkWorkITS.getY(), trkWorkITS.getZ(), 190, -1, compXV);
            if (!gidTableFull[GTrackID::ITSTPC].isIndexSet()) {
              LOGP(fatal, "ITS-TPC seed index is not set for TOF track");
            }
            float tdif = pv.getTimeStamp().getTimeStamp() - mRecoCont->getTPCITSTrack(gidTableFull[GTrackID::ITSTPC]).getTimeMUS().getTimeStamp();
            mDetInfoRes.emplace_back().setPV(tdif); // time in \mus wrt seeding ITS-TPC track
            trackData.nExtDetResid++;
          }
        }
        break;
      }
    }
    mTrackData.push_back(std::move(trackData));
    stored = true;
    mGIDsSuccess.push_back(mGIDs[iSeed]);
    mTrackDataCompact.emplace_back(trackData.clIdx.getFirstEntry(), trackData.multStack, nClValidated, mGIDs[iSeed].getSource(), trackData.nExtDetResid, trackData.filterFlag);
    if (mDumpTrackPoints) {
      (*trackDataExtended).clIdx.setEntries(nClValidated);
      (*trackDataExtended).nExtDetResid = trackData.nExtDetResid;
      (*trackDataExtended).filterFlag = trackData.filterFlag;
      mTrackDataExtended.push_back(std::move(*trackDataExtended));
    }
  }
  if (mParams->writeValidationData && trackData.filterFlag >= 0 && mDBGOut) {
    (*mDBGOut) << "valdata" << "params=" << mTrackValidation << "trackData=" << (stored ? mTrackData.back() : trackData) << "\n";
  }
}

int8_t TrackInterpolation::validateTrack(const TrackData& trk, TrackValidationData& params, const std::vector<TPCClusterResiduals>& clsRes, bool interpol)
{
  int8_t status = 0;
  while (true) {
    if (clsRes.size() < mParams->minNCl) {
      // no enough clusters for this track to be considered
      LOG(debug) << "Skipping track with too few clusters: " << clsRes.size();
      status |= 0x1;
      if (!mParams->keepRejectedResiduals) {
        break; // we don't keep de-validated tracks, no need to check further
      }
    }

    bool resHelix = compareToHelix(trk, params, clsRes);
    if (!resHelix && interpol) {
      LOG(debug) << "Skipping track too far from helix approximation";
      status |= 0x1 << 1;
      if (!mParams->keepRejectedResiduals) {
        break; // we don't keep de-validated tracks, no need to check further
      }
    }
    if (interpol && (std::abs(mBz) > 0.01 && std::abs(params.qpt) > mParams->maxQ2Pt)) {
      LOG(debug) << "Skipping track with too high q/pT: " << params.qpt;
      status |= 0x1 << 2;
      if (!mParams->keepRejectedResiduals) {
        break; // we don't keep de-validated tracks, no need to check further
      }
    }
    if (!outlierFiltering(trk, params, clsRes)) {
      status |= 0x1 << 3;
      if (!mParams->keepRejectedResiduals) {
        break; // we don't keep de-validated tracks, no need to check further
      }
    }
    break;
  }
  return status & 0x7f;
}

bool TrackInterpolation::compareToHelix(const TrackData& trk, TrackValidationData& params, const std::vector<TPCClusterResiduals>& clsRes)
{
  float curvature = std::abs(trk.par.getQ2Pt() * mBz * o2::constants::physics::LightSpeedCm2S * 1e-14f);
  int secFirst = clsRes[0].sec;
  float phiSect = (secFirst + .5f) * o2::constants::math::SectorSpanRad;
  float snPhi = sin(phiSect);
  float csPhi = cos(phiSect);

  int iRow = 0;
  int nCl = clsRes.size();
  for (unsigned int iP = 0; iP < nCl; ++iP) {
    auto& point = params.points.emplace_back();

    iRow += clsRes[iP].dRow;
    point.yTrk = clsRes[iP].y;
    point.sec = clsRes[iP].sec;
    if (clsRes[iP].sec != secFirst) {
      float phiSectCurrent = (clsRes[iP].sec + .5f) * o2::constants::math::SectorSpanRad;
      float cs = cos(phiSectCurrent - phiSect);
      float sn = sin(phiSectCurrent - phiSect);
      point.xLab = param::RowX[iRow] * cs - point.yTrk * sn;
      point.yLab = point.yTrk * cs + param::RowX[iRow] * sn;
    } else {
      point.xLab = param::RowX[iRow];
      point.yLab = point.yTrk;
    }
    // this is needed only later, but we retrieve it already now to save another loop
    point.zTrk = clsRes[iP].z;
    point.xTrk = param::RowX[iRow];
    point.dy = clsRes[iP].dy;
    point.dz = clsRes[iP].dz;
    // done retrieving values for later
    if (iP > 0) {
      float dx = point.xLab - params.points[iP - 1].xLab;
      float dy = point.yLab - params.points[iP - 1].yLab;
      float ds2 = dx * dx + dy * dy;
      float ds = sqrt(ds2); // circular path (linear approximation)
      // if the curvature of the track or the (approximated) chord length is too large the more exact formula is used:
      // chord length = 2r * asin(ds/(2r))
      // using the first two terms of the tailer expansion for asin(x) ~ x + x^3 / 6
      if (ds * curvature > 0.05) {
        ds *= (1.f + ds2 * curvature * curvature / 24.f);
      }
      point.sPath = params.points[iP - 1].sPath + ds;
    } else {
      point.sPath = 0;
    }
  }
  if (std::abs(mBz) < 0.01) {
    // for B=0 we don't need to try a circular fit...
    return true;
  }
  TrackResiduals::fitCircle(params);
  // determine curvature
  float phiI = TMath::ATan2(params.points.front().yLab, params.points.front().xLab);
  float phiF = TMath::ATan2(params.points.back().yLab, params.points.back().xLab);
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
  params.qpt = std::copysign(1.f / (params.r * mBz * o2::constants::physics::LightSpeedCm2S * 1e-14f), curvSign);

  TrackResiduals::fitPoly1(params);

  // max deviations in both directions from helix fit in y and z
  float hMinY = 1e9f;
  float hMaxY = -1e9f;
  float hMinZ = 1e9f;
  float hMaxZ = -1e9f;
  // extract residuals in Z and fill track slopes in sector frame
  int secCurr = -1;
  iRow = 0;
  float xcSec = 0;
  for (unsigned int iCl = 0; iCl < nCl; ++iCl) {
    iRow += clsRes[iCl].dRow;
    auto& pnt = params.points[iCl];
    pnt.residHelixZ = pnt.zTrk - (params.zOffs + pnt.sPath * params.tgl);
    if (pnt.residHelixZ < hMinZ) {
      hMinZ = pnt.residHelixZ;
    }
    if (pnt.residHelixZ > hMaxZ) {
      hMaxZ = pnt.residHelixZ;
    }
    if (pnt.residHelixY < hMinY) {
      hMinY = pnt.residHelixY;
    }
    if (pnt.residHelixY > hMaxY) {
      hMaxY = pnt.residHelixY;
    }
    int sec = clsRes[iCl].sec;
    if (sec != secCurr) {
      secCurr = sec;
      phiSect = (.5f + sec) * o2::constants::math::SectorSpanRad;
      snPhi = sin(phiSect);
      csPhi = cos(phiSect);
      xcSec = params.xcLab * csPhi + params.ycLab * snPhi; // recalculate circle center in the new sector frame
    }
    float cstalp = (param::RowX[iRow] - xcSec) / params.r;
    if (std::abs(cstalp) > 1.f - sFloatEps) {
      // track cannot reach this pad row
      cstalp = std::copysign(1.f - sFloatEps, cstalp);
    }
    pnt.tglArr = cstalp / sqrt((1 - cstalp) * (1 + cstalp)); // 1 / tan(acos(cstalp)) = cstalp / sqrt(1 - cstalp^2)

    // In B+ the slope of q- should increase with x. Just look on q * B
    if (params.qpt * mBz > 0) {
      pnt.tglArr = -pnt.tglArr;
    }
  }
  // LOGF(info, "CompareToHelix: hMaxY(%f), hMinY(%f), hMaxZ(%f), hMinZ(%f). Max deviation allowed: y(%.2f), z(%.2f)", hMaxY, hMinY, hMaxZ, hMinZ, mParams->maxDevHelixY, mParams->maxDevHelixZ);
  // LOGF(info, "New pt/Q (%f), old pt/Q (%f)", 1./params.qpt, 1./trk.qPt);
  return std::abs(hMaxY - hMinY) < mParams->maxDevHelixY && std::abs(hMaxZ - hMinZ) < mParams->maxDevHelixZ;
}

bool TrackInterpolation::outlierFiltering(const TrackData& trk, TrackValidationData& params, const std::vector<TPCClusterResiduals>& clsRes)
{
  if (clsRes.size() < mParams->nMALong) {
    LOG(debug) << "Skipping track with too few clusters for long moving average: " << clsRes.size();
    return false;
  }
  float rmsLong = checkResiduals(trk, params, clsRes);
  if (static_cast<float>(params.nRej) / clsRes.size() > mParams->maxRejFrac) {
    LOGP(debug, "Skipping track with too many clusters rejected: {} out of {}", params.nRej, clsRes.size());
    return false;
  }
  if (rmsLong > mParams->maxRMSLong) {
    LOG(debug) << "Skipping track with too large RMS: " << rmsLong;
    return false;
  }
  return true;
}

float TrackInterpolation::checkResiduals(const TrackData& trk, TrackValidationData& params, const std::vector<TPCClusterResiduals>& clsRes)
{
  float rmsLong = 0.f;

  int nCl = clsRes.size();
  int iClFirst = 0;
  int iClLast = nCl - 1;
  int secStart = clsRes[0].sec;

  auto rejectAll = [&params]() {
    for (auto& pnt : params.points) {
      pnt.flagRej = true;
    }
    params.nRej = params.points.size();
  };

  // arrays with differences / abs(differences) of points to their neighbourhood, initialized to zero
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
    diffToLocLine(params, iClFirst, nClSec);
    iClFirst = iCl;
    secStart = clsRes[iCl].sec;
  }
  // store abs deviations
  int nAccY = 0;
  int nAccZ = 0;
  for (int iCl = nCl; iCl--;) {
    const auto pnt = params.points[iCl];
    if (std::abs(pnt.diffYSmooth) > param::sEps) {
      absDevY[nAccY++] = std::abs(pnt.diffYSmooth);
    }
    if (std::abs(pnt.diffZSmooth) > param::sEps) {
      absDevZ[nAccZ++] = std::abs(pnt.diffZSmooth);
    }
  }
  if (nAccY < mParams->minNumberOfAcceptedResiduals || nAccZ < mParams->minNumberOfAcceptedResiduals) {
    // mask all clusters
    LOGP(debug, "Accepted {} clusters for dY {} clusters for dZ, but required at least {} for both", nAccY, nAccZ, mParams->minNumberOfAcceptedResiduals);
    rejectAll();
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
    rejectAll();
    return 0.f;
  }
  float rmsYkeepI = 1.f / rmsYkeep;
  float rmsZkeepI = 1.f / rmsZkeep;
  int nAcc = 0;
  std::array<float, param::NPadRows> yAcc;
  std::array<float, param::NPadRows> yDiffLong;
  for (int iCl = 0; iCl < nCl; ++iCl) {
    auto& pnt = params.points[iCl];
    auto yDiffScl = pnt.diffYSmooth * rmsYkeepI;
    auto zDiffScl = pnt.diffZSmooth * rmsZkeepI;
    if (yDiffScl * yDiffScl + zDiffScl * zDiffScl > mParams->maxStdDevMA) {
      pnt.flagRej = true;
      params.nRej++;
    } else {
      yAcc[nAcc++] = pnt.dy;
    }
  }
  if (nAcc > mParams->nMALong) {
    diffToMA(nAcc, yAcc, yDiffLong);
    float average = 0.f, rms = 0.f;
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

void TrackInterpolation::diffToLocLine(TrackValidationData& params, int start, int np)
{
  std::array<float, param::NPadRows + 1> sumX1{}, sumX2{}, sumY1{}, sumXY{}, sumZ1{}, sumXZ{};
  for (int i = 0; i < np; ++i) {
    const auto& pnt = params.points[start + i];
    const float x = pnt.xTrk, y = pnt.dy, z = pnt.dz;
    sumX1[i + 1] = sumX1[i] + x;
    sumX2[i + 1] = sumX2[i] + x * x;
    sumY1[i + 1] = sumY1[i] + y;
    sumXY[i + 1] = sumXY[i] + x * y;
    sumZ1[i + 1] = sumZ1[i] + z;
    sumXZ[i + 1] = sumXZ[i] + x * z;
  }

  for (int i = 0; i < np; ++i) {
    auto& pnt = params.points[start + i];

    const int iLeft = std::max(0, i - mParams->nMAShort);
    const int iRight = std::min(np - 1, i + mParams->nMAShort);

    const int nPoints = iRight - iLeft; // excluding current point

    if (nPoints < mParams->nMAShort) {
      continue;
    }

    const float nPointsInv = 1.f / nPoints;

    float sX1 = sumX1[iRight + 1] - sumX1[iLeft] - pnt.xTrk;
    float sX2 = sumX2[iRight + 1] - sumX2[iLeft] - pnt.xTrk * pnt.xTrk;
    float sY1 = sumY1[iRight + 1] - sumY1[iLeft] - pnt.dy;
    float sXY = sumXY[iRight + 1] - sumXY[iLeft] - pnt.xTrk * pnt.dy;
    float sZ1 = sumZ1[iRight + 1] - sumZ1[iLeft] - pnt.dz;
    float sXZ = sumXZ[iRight + 1] - sumXZ[iLeft] - pnt.xTrk * pnt.dz;

    const float det = sX2 - nPointsInv * sX1 * sX1;

    if (std::abs(det) < 1e-12f) {
      continue;
    }

    const float slopeY = (sXY - nPointsInv * sX1 * sY1) / det;
    const float offsetY = nPointsInv * (sY1 - slopeY * sX1);
    const float slopeZ = (sXZ - nPointsInv * sX1 * sZ1) / det;
    const float offsetZ = nPointsInv * (sZ1 - slopeZ * sX1);
    pnt.diffYSmooth = pnt.dy - (slopeY * pnt.xTrk + offsetY);
    pnt.diffZSmooth = pnt.dz - (slopeZ * pnt.xTrk + offsetZ);
  }
}

void TrackInterpolation::diffToMA(const int np, const std::array<float, param::NPadRows>& y, std::array<float, param::NPadRows>& diffMA)
{
  // Calculate
  std::array<float, param::NPadRows + 1> sum{};
  for (int i = 0; i < np; ++i) {
    sum[i + 1] = sum[i] + y[i];
  }
  for (int i = 0; i < np; ++i) {
    diffMA[i] = 0.f;
    int iLeft = std::max(0, i - mParams->nMALong);
    int iRight = std::min(np - 1, i + mParams->nMALong);
    int nPoints = iRight - iLeft;
    if (nPoints < mParams->nMALong) { // this cannot happen, since at least mParams->nMALong points are required as neighbours for this function to be called
      continue;
    }
    float movingAverage = (sum[iRight + 1] - sum[iLeft] - y[i]) / nPoints;
    diffMA[i] = y[i] - movingAverage;
  }
}

void TrackInterpolation::reset()
{
  mTrackData.clear();
  mTrackDataCompact.clear();
  mTrackDataExtended.clear();
  mClRes.clear();
  mDetInfoRes.clear();
  mGIDsSuccess.clear();
  for (auto& vec : mTrackIndices) {
    vec.clear();
  }
  mGIDs.clear();
  mGIDtables.clear();
  mTrackTimes.clear();
  mSeeds.clear();
  mITSRefitSeedID.clear();
  mTrackPVID.clear();
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

//______________________________________________
bool TrackInterpolation::refITSTrack(o2::dataformats::GlobalTrackID gid, int seedID)
{
  // refit ITS track outwards taking PID (unless already refitted) from the seed and reassign to the seed
  auto& seed = mSeeds[seedID];
  int refitID = mITSRefitSeedID[gid.getIndex()];
  if (refitID >= 0) { // track was already refitted
    if (mSeeds[refitID].getPID() == seed.getPID()) {
      seed = mSeeds[refitID];
    }
    return true;
  }
  const auto& trkITS = mRecoCont->getITSTrack(gid);
  // fetch clusters
  auto nCl = trkITS.getNumberOfClusters();
  auto clEntry = trkITS.getFirstClusterEntry();
  o2::track::TrackParCov track(trkITS); // start from the inner param
  track.resetCovariance();
  track.setCov(track.getQ2Pt() * track.getQ2Pt() * track.getCov()[o2::track::CovLabels::kSigQ2Pt2], o2::track::CovLabels::kSigQ2Pt2);
  track.setPID(seed.getPID());
  o2::track::TrackPar refLin(track); // and use it also as linearization reference
  auto geom = o2::its::GeometryTGeo::Instance();
  auto prop = o2::base::Propagator::Instance();
  for (int iCl = nCl - 1; iCl >= 0; iCl--) { // clusters are stored from outer to inner layers
    const auto& cls = mITSClustersArray[mITSTrackClusIdx[clEntry + iCl]];
    int chip = cls.getSensorID();
    float chipX, chipAlpha;
    geom->getSensorXAlphaRefPlane(cls.getSensorID(), chipX, chipAlpha);
    if (!track.rotate(chipAlpha, refLin, mBz)) {
      LOGP(debug, "failed to rotate ITS tracks to alpha={} for the refit: {}", chipAlpha, track.asString());
      return false;
    }
    if (!prop->propagateToX(track, refLin, cls.getX(), mBz, o2::base::PropagatorImpl<float>::MAX_SIN_PHI, o2::base::PropagatorImpl<float>::MAX_STEP, o2::base::PropagatorF::MatCorrType::USEMatCorrLUT)) {
      LOGP(debug, "failed to propagate ITS tracks to X={}: {}", cls.getX(), track.asString());
      return false;
    }
    std::array<float, 2> posTF{cls.getY(), cls.getZ()};
    std::array<float, 3> covTF{cls.getSigmaY2(), cls.getSigmaYZ(), cls.getSigmaZ2()};
    if (!track.update(posTF, covTF)) {
      LOGP(debug, "failed to update ITS tracks by cluster ({},{})/({},{},{})", track.asString(), cls.getY(), cls.getZ(), cls.getSigmaY2(), cls.getSigmaYZ(), cls.getSigmaZ2());
      return false;
    }
    if (mParams->shiftRefToCluster) {
      refLin.setY(posTF[0]);
      refLin.setZ(posTF[1]);
    }
  }
  seed = track;
  // memorize that this ITS track was already refitted
  mITSRefitSeedID[gid.getIndex()] = seedID;
  return true;
}
