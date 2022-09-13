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
#include "TPCBase/ParameterElectronics.h"
#include "DataFormatsTPC/TrackTPC.h"
#include "DataFormatsTPC/Defs.h"
#include "DataFormatsTRD/Constants.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "TRDBase/PadPlane.h"
#include "TMath.h"
#include "DataFormatsTPC/VDriftCorrFact.h"
#include <fairlogger/Logger.h>
#include <set>

using namespace o2::tpc;
using GTrackID = o2::dataformats::GlobalTrackID;
using DetID = o2::detectors::DetID;

void TrackInterpolation::init()
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

  mRecoParam.setBfield(o2::base::Propagator::Instance()->getNominalBz());
  mGeoTRD = o2::trd::Geometry::instance();

  mInitDone = true;
  LOG(info) << "Done initializing TrackInterpolation";
}

void TrackInterpolation::process(const o2::globaltracking::RecoContainer& inp, const std::vector<GTrackID>& gids, const std::vector<o2::globaltracking::RecoContainer::GlobalIDSet>& gidTables, std::vector<o2::track::TrackParCov>& seeds, const std::vector<float>& trkTimes)
{
  // main processing function

  if (!mInitDone) {
    LOG(error) << "Initialization not yet done. Aborting...";
    return;
  }

  // reset output vectors
  reset();

  // set the input containers
  mRecoCont = &inp;
  mGIDs = &gids;
  mGIDtables = &gidTables;
  mSeeds = &seeds;
  mTrackTimes = &trkTimes;
  mTPCTracksClusIdx = mRecoCont->getTPCTracksClusterRefs();
  mTPCClusterIdxStruct = &mRecoCont->getTPCClusters();

  int nSeeds = mSeeds->size();
  mTrackData.reserve(nSeeds);
  mClRes.reserve(nSeeds * param::NPadRows);

  for (int iSeed = 0; iSeed < nSeeds; ++iSeed) {
    if (gids[iSeed].includesDet(DetID::TRD) || gids[iSeed].includesDet(DetID::TOF)) {
      interpolateTrack(iSeed);
    } else {
      extrapolateTrack(iSeed);
    }
  }

  LOG(info) << "Could process " << mTrackData.size() << " tracks successfully";
}

void TrackInterpolation::interpolateTrack(int iSeed)
{
  TrackData trackData;
  auto propagator = o2::base::Propagator::Instance();
  const auto& gidTable = (*mGIDtables)[iSeed];
  const auto& trkTPC = mRecoCont->getTPCTrack(gidTable[GTrackID::TPC]);
  const auto& trkITS = mRecoCont->getITSTrack(gidTable[GTrackID::ITS]);
  auto& trkWork = (*mSeeds)[iSeed];
  // reset the cache array (sufficient to set cluster available to zero)
  for (auto& elem : mCache) {
    elem.clAvailable = 0;
  }
  trackData.clIdx.setFirstEntry(mClRes.size()); // reference the first cluster residual belonging to this track
  float clusterTimeBinOffset = (*mTrackTimes)[iSeed] / mTPCTimeBinMUS;

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
    if (!propagator->PropagateToXBxByBz(trkWork, param::RowX[iRow], mMaxSnp, mMaxStep, mMatCorr)) {
      LOG(debug) << "Failed on first extrapolation";
      return;
    }
    mCache[iRow].y[ExtOut] = trkWork.getY();
    mCache[iRow].z[ExtOut] = trkWork.getZ();
    mCache[iRow].sy2[ExtOut] = trkWork.getSigmaY2();
    mCache[iRow].szy[ExtOut] = trkWork.getSigmaZY();
    mCache[iRow].sz2[ExtOut] = trkWork.getSigmaZ2();
    mCache[iRow].snp[ExtOut] = trkWork.getSnp();
    mCache[iRow].tgl[ExtOut] = trkWork.getTgl();
    //printf("Track alpha at row %i: %.2f, Y(%.2f), Z(%.2f)\n", iRow, trkWork.getAlpha(), trkWork.getY(), trkWork.getZ());
  }

  // start from outermost cluster with outer refit and back propagation
  if (gidTable[GTrackID::TOF].isIndexSet()) {
    LOG(debug) << "TOF point available";
    const auto& clTOF = mRecoCont->getTOFClusters()[gidTable[GTrackID::TOF]];
    const int clTOFSec = clTOF.getCount();
    const float clTOFAlpha = o2::math_utils::sector2Angle(clTOFSec);
    if (!trkWork.rotate(clTOFAlpha)) {
      LOG(debug) << "Failed to rotate into TOF cluster sector frame";
      return;
    }
    float clTOFX = clTOF.getX();
    std::array<float, 2> clTOFYZ{clTOF.getY(), clTOF.getZ()};
    std::array<float, 3> clTOFCov{mSigYZ2TOF, 0.f, mSigYZ2TOF}; // assume no correlation between y and z and equal cluster error sigma^2 = (3cm)^2 / 12
    if (!propagator->PropagateToXBxByBz(trkWork, clTOFX, mMaxSnp, mMaxStep, mMatCorr)) {
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
    for (int iLayer = o2::trd::constants::NLAYER - 1; iLayer >= 0; --iLayer) {
      int trkltIdx = trkTRD.getTrackletIndex(iLayer);
      if (trkltIdx < 0) {
        // no TRD tracklet in this layer
        continue;
      }
      const auto& trdSP = mRecoCont->getTRDCalibratedTracklets()[trkltIdx];
      const auto& trdTrklt = mRecoCont->getTRDTracklets()[trkltIdx];
      auto trkltDet = trdTrklt.getDetector();
      auto trkltSec = trkltDet / (o2::trd::constants::NLAYER * o2::trd::constants::NSTACK);
      if (trkltSec != o2::math_utils::angle2Sector(trkWork.getAlpha())) {
        if (!trkWork.rotate(o2::math_utils::sector2Angle(trkltSec))) {
          LOG(debug) << "Track could not be rotated in TRD tracklet coordinate system in layer " << iLayer;
          return;
        }
      }
      if (!propagator->PropagateToXBxByBz(trkWork, trdSP.getX(), mMaxSnp, mMaxStep, mMatCorr)) {
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

  // go back through the TPC and store updated track positions
  for (int iRow = param::NPadRows; iRow--;) {
    if (!mCache[iRow].clAvailable) {
      continue;
    }
    if (!trkWork.rotate(mCache[iRow].clAngle)) {
      LOG(debug) << "Failed to rotate track during back propagation";
      return;
    }
    if (!propagator->PropagateToXBxByBz(trkWork, param::RowX[iRow], mMaxSnp, mMaxStep, mMatCorr)) {
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
    mCache[iRow].tgl[ExtIn] = trkWork.getTgl();
  }

  // calculate weighted mean at each pad row (assume for now y and z are uncorrelated) and store residuals to TPC clusters
  unsigned short deltaRow = 0;
  unsigned short nMeasurements = 0;
  for (int iRow = 0; iRow < param::NPadRows; ++iRow) {
    if (!mCache[iRow].clAvailable) {
      ++deltaRow;
      continue;
    }
    float wTotY = 1.f / mCache[iRow].sy2[ExtOut] + 1.f / mCache[iRow].sy2[ExtIn];
    float wTotZ = 1.f / mCache[iRow].sz2[ExtOut] + 1.f / mCache[iRow].sz2[ExtIn];
    mCache[iRow].y[Int] = (mCache[iRow].y[ExtOut] / mCache[iRow].sy2[ExtOut] + mCache[iRow].y[ExtIn] / mCache[iRow].sy2[ExtIn]) / wTotY;
    mCache[iRow].z[Int] = (mCache[iRow].z[ExtOut] / mCache[iRow].sz2[ExtOut] + mCache[iRow].z[ExtIn] / mCache[iRow].sz2[ExtIn]) / wTotZ;

    // simple average w/o weighting for angles
    mCache[iRow].snp[Int] = (mCache[iRow].snp[ExtOut] + mCache[iRow].snp[ExtIn]) / 2.f;
    mCache[iRow].tgl[Int] = (mCache[iRow].tgl[ExtOut] + mCache[iRow].tgl[ExtIn]) / 2.f;

    TPCClusterResiduals res;
    res.setDY(mCache[iRow].clY - mCache[iRow].y[Int]);
    res.setDZ(mCache[iRow].clZ - mCache[iRow].z[Int]);
    res.setY(mCache[iRow].y[Int]);
    res.setZ(mCache[iRow].z[Int]);
    res.setSnp(mCache[iRow].snp[Int]);
    res.setTgl(mCache[iRow].tgl[Int]);
    res.sec = mCache[iRow].clSec;
    res.dRow = deltaRow;
    mClRes.push_back(std::move(res));
    ++nMeasurements;
    deltaRow = 1;
  }

  trackData.gid = (*mGIDs)[iSeed];
  trackData.x = (*mSeeds)[iSeed].getX();
  trackData.alpha = (*mSeeds)[iSeed].getAlpha();
  for (int i = 0; i < o2::track::kNParams; ++i) {
    trackData.p[i] = (*mSeeds)[iSeed].getParam(i);
  }
  trackData.chi2TRD = gidTable[GTrackID::TRD].isIndexSet() ? mRecoCont->getITSTPCTRDTrack<o2::trd::TrackTRD>(gidTable[GTrackID::ITSTPCTRD]).getChi2() : 0;
  trackData.chi2TPC = trkTPC.getChi2();
  trackData.chi2ITS = trkITS.getChi2();
  trackData.nClsTPC = trkTPC.getNClusterReferences();
  trackData.nClsITS = trkITS.getNumberOfClusters();
  trackData.nTrkltsTRD = gidTable[GTrackID::TRD].isIndexSet() ? mRecoCont->getITSTPCTRDTrack<o2::trd::TrackTRD>(gidTable[GTrackID::ITSTPCTRD]).getNtracklets() : 0;
  trackData.clAvailTOF = gidTable[GTrackID::TOF].isIndexSet() ? 1 : 0;
  trackData.clIdx.setEntries(nMeasurements);

  mTrackData.push_back(std::move(trackData));
  mGIDsSuccess.push_back((*mGIDs)[iSeed]);
}

void TrackInterpolation::extrapolateTrack(int iSeed)
{
  // extrapolate ITS-only track through TPC and store residuals to TPC clusters in the output vectors
  const auto& gidTable = (*mGIDtables)[iSeed];
  TrackData trackData;
  trackData.clIdx.setFirstEntry(mClRes.size());
  const auto& trkITS = mRecoCont->getITSTrack(gidTable[GTrackID::ITS]);
  const auto& trkTPC = mRecoCont->getTPCTrack(gidTable[GTrackID::TPC]);
  auto& trkWork = (*mSeeds)[iSeed];
  float clusterTimeBinOffset = (*mTrackTimes)[iSeed] / mTPCTimeBinMUS;
  auto propagator = o2::base::Propagator::Instance();
  unsigned short rowPrev = 0;
  unsigned short nMeasurements = 0;
  for (int iCl = trkTPC.getNClusterReferences(); iCl--;) {
    uint8_t sector, row;
    uint32_t clusterIndexInRow;
    const auto& cl = trkTPC.getCluster(mTPCTracksClusIdx, iCl, *mTPCClusterIdxStruct, sector, row);
    float x = 0, y = 0, z = 0;
    mFastTransform->TransformIdeal(sector, row, cl.getPad(), cl.getTime(), x, y, z, clusterTimeBinOffset);
    if (!trkWork.rotate(o2::math_utils::sector2Angle(sector))) {
      return;
    }
    if (!propagator->PropagateToXBxByBz(trkWork, x, mMaxSnp, mMaxStep, mMatCorr)) {
      return;
    }
    TPCClusterResiduals res;
    res.setDY(y - trkWork.getY());
    res.setDY(z - trkWork.getZ());
    res.setY(trkWork.getY());
    res.setZ(trkWork.getZ());
    res.setSnp(trkWork.getSnp());
    res.setTgl(trkWork.getTgl());
    res.sec = sector;
    res.dRow = row - rowPrev;
    rowPrev = row;
    mClRes.push_back(std::move(res));
    ++nMeasurements;
  }
  trackData.gid = (*mGIDs)[iSeed];
  trackData.x = (*mSeeds)[iSeed].getX();
  trackData.alpha = (*mSeeds)[iSeed].getAlpha();
  for (int i = 0; i < o2::track::kNParams; ++i) {
    trackData.p[i] = (*mSeeds)[iSeed].getParam(i);
  }
  trackData.chi2TPC = trkTPC.getChi2();
  trackData.chi2ITS = trkITS.getChi2();
  trackData.nClsTPC = trkTPC.getNClusterReferences();
  trackData.nClsITS = trkITS.getNumberOfClusters();
  trackData.clIdx.setEntries(nMeasurements);

  mTrackData.push_back(std::move(trackData));
  mGIDsSuccess.push_back((*mGIDs)[iSeed]);
}

void TrackInterpolation::reset()
{
  mTrackData.clear();
  mClRes.clear();
  mGIDsSuccess.clear();
}

//______________________________________________
void TrackInterpolation::setTPCVDrift(const o2::tpc::VDriftCorrFact& v)
{
  mTPCVDrift = v.refVDrift * v.corrFact;
  // Attention! For the refit we are using reference VDrift rather than high-rate calibrated, since we want to have fixed reference over the run
  if (v.refVDrift != mTPCVDriftRef) {
    mTPCVDriftRef = v.refVDrift;
    LOGP(info, "Imposing reference VDrift={} for TPC residuals extraction", mTPCVDriftRef);
    o2::tpc::TPCFastTransformHelperO2::instance()->updateCalibration(*mFastTransform, 0, 1.0, mTPCVDriftRef);
  }
}
