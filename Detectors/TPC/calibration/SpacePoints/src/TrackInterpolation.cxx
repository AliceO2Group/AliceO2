// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
#include "ReconstructionDataFormats/GlobalTrackID.h"

#include <fairlogger/Logger.h>
#include <set>

using namespace o2::tpc;

void TrackInterpolation::init()
{
  // perform initialization
  LOG(INFO) << "Start initializing TrackInterpolation";
  if (mInitDone) {
    LOG(error) << "Initialization already performed.";
    return;
  }

  const auto& elParam = ParameterElectronics::Instance();
  mTPCTimeBinMUS = elParam.ZbinWidth;

  std::unique_ptr<TPCFastTransform> fastTransform = (TPCFastTransformHelperO2::instance()->create(0));
  mFastTransform = std::move(fastTransform);

  mInitDone = true;
  LOG(INFO) << "Done initializing TrackInterpolation";
}

void TrackInterpolation::process()
{
  // main processing function
  if (!mInitDone) {
    LOG(error) << "Initialization not yet done. Aborting...";
    return;
  }
  reset();

#ifdef TPC_RUN2
  // processing will not work if the run 2 geometry is defined in the parameter class SpacePointsCalibParam.h
  LOG(FATAL) << "Run 2 parameters compiled for the TPC geometry. Creating residual trees from Run 3 data will not work. Aborting...";
  return;
#endif

  std::set<unsigned int> tracksDone; // to store indices of ITS-TPC matched tracks that have been processed

  LOG(INFO) << "Processing " << mTOFMatchesArray.size() << " ITS-TPC-TOF matched tracks out of "
            << mITSTPCTracksArray.size() << " ITS-TPC matched tracks";

  // TODO reserve only a fraction of the needed space for all tracks? How many tracks pass on average the quality cuts with how many TPC clusters?
  mTrackData.reserve(mTOFMatchesArray.size());
  mClRes.reserve(mTOFMatchesArray.size() * param::NPadRows);

  int nTracksTPC = mTPCTracksArray.size();

  for (const auto& trkTOF : mTOFMatchesArray) {
    // process ITS-TPC-TOF matched tracks
    if (!trackPassesQualityCuts(mITSTPCTracksArray[trkTOF.getTrackIndex()])) {
      LOG(DEBUG) << "Abandoning track due to bad quality";
      continue;
    }
    mTrackData.emplace_back();
    if (!interpolateTrackITSTOF(trkTOF)) {
      LOG(DEBUG) << "Failed to interpolate ITS-TOF track";
      mTrackData.pop_back();
      continue;
    }
    mTrackData.back().nTracksInEvent = nTracksTPC;
    tracksDone.insert(trkTOF.getTrackIndex());
  }

  LOG(INFO) << "Could process " << tracksDone.size() << " ITS-TPC-TOF matched tracks successfully";

  if (mDoITSOnlyTracks) {
    size_t nTracksDoneITS = 0;
    size_t nTracksSkipped = 0;
    for (std::size_t iTrk = 0; iTrk < mITSTPCTracksArray.size(); ++iTrk) {
      // process ITS-TPC matched tracks that were not matched to TOF
      if (tracksDone.find(iTrk) != tracksDone.end()) {
        // track also has a matching cluster in TOF and has already been processed
        ++nTracksSkipped;
        continue;
      }
      const auto& trk = mITSTPCTracksArray[iTrk];
      if (!trackPassesQualityCuts(trk, false)) {
        continue;
      }
      mTrackData.emplace_back();
      const auto& trkTPC = mTPCTracksArray[trk.getRefTPC()];
      const auto& trkITS = mITSTracksArray[trk.getRefITS()];
      if (!extrapolateTrackITS(trkITS, trkTPC, trk.getTimeMUS().getTimeStamp(), trk.getRefTPC())) {
        mTrackData.pop_back();
        continue;
      }
      mTrackData.back().nTracksInEvent = nTracksTPC;
      ++nTracksDoneITS;
    }
    LOG(INFO) << "Could process " << nTracksDoneITS << " ITS-TPC matched tracks successfully";
    LOG(INFO) << "Skipped " << nTracksSkipped << " tracks, as they were successfully propagated to TOF";
  }
}

bool TrackInterpolation::trackPassesQualityCuts(const o2::dataformats::TrackTPCITS& matchITSTPC, bool hasOuterPoint) const
{
  // apply track quality cuts (assume different settings for track with and without points in TRD or TOF)
  const auto& trkTPC = mTPCTracksArray[matchITSTPC.getRefTPC()];
  const auto& trkITS = mITSTracksArray[matchITSTPC.getRefITS()];
  if (hasOuterPoint) {
    // track has a match in TRD or TOF
    if (trkTPC.getNClusterReferences() < param::MinTPCNCls ||
        trkITS.getNumberOfClusters() < param::MinITSNCls) {
      LOG(DEBUG) << "TPC clusters (" << trkTPC.getNClusterReferences() << "), ITS clusters(" << trkITS.getNumberOfClusters() << ")";
      return false;
    }
    if (trkTPC.getChi2() / trkTPC.getNClusterReferences() > param::MaxTPCChi2 ||
        trkITS.getChi2() / trkITS.getNumberOfClusters() > param::MaxITSChi2) {
      LOG(DEBUG) << "TPC reduced chi2 (" << trkTPC.getChi2() / trkTPC.getNClusterReferences() << "), ITS reduced chi2 (" << trkITS.getChi2() / trkITS.getNumberOfClusters() << ")";
      return false;
    }
  } else {
    // ITS-TPC only track
    if (trkTPC.getNClusterReferences() < param::MinTPCNClsNoOuterPoint ||
        trkITS.getNumberOfClusters() < param::MinITSNClsNoOuterPoint) {
      return false;
    }
    if (trkTPC.getChi2() / trkTPC.getNClusterReferences() > param::MaxTPCChi2 ||
        trkITS.getChi2() / trkITS.getNumberOfClusters() > param::MaxITSChi2) {
      return false;
    }
  }
  return true;
}

bool TrackInterpolation::interpolateTrackITSTOF(const o2::dataformats::MatchInfoTOF& matchTOF)
{
  // get TPC cluster residuals to ITS-TOF only tracks
  size_t trkIdx = mTrackData.size() - 1;
  auto propagator = o2::base::Propagator::Instance();
  const auto& matchITSTPC = mITSTPCTracksArray[matchTOF.getTrackIndex()];
  const auto& clTOF = mTOFClustersArray[matchTOF.getTOFClIndex()];
  //const int clTOFSec = (TMath::ATan2(-clTOF.getY(), -clTOF.getX()) + o2::constants::math::PI) * o2::constants::math::Rad2Deg * 0.05; // taken from TOF cluster class as there is no const getter for the sector
  const int clTOFSec = clTOF.getCount();
  const float clTOFAlpha = o2::math_utils::sector2Angle(clTOFSec);
  const auto& trkTPC = mTPCTracksArray[matchITSTPC.getRefTPC()];
  const auto& trkITS = mITSTracksArray[matchITSTPC.getRefITS()];
  auto trkWork = trkITS.getParamOut();
  // reset the cache array (sufficient to set )
  for (auto& elem : mCache) {
    elem.clAvailable = 0;
  }
  mTrackData[trkIdx].clIdx.setFirstEntry(mClRes.size()); // reference the first cluster residual belonging to this track
  //printf("=== New Track with pt = %.3f, nClsTPC = %i ===\n", trkWork.getQ2Pt(), trkTPC.getNClusterReferences());

  // store the TPC cluster positions in the cache
  float clusterTimeBinOffset = matchITSTPC.getTimeMUS().getTimeStamp() / mTPCTimeBinMUS;
  for (int iCl = trkTPC.getNClusterReferences(); iCl--;) {
    uint8_t sector, row;
    uint32_t clusterIndexInRow;
    const auto& clTPC = trkTPC.getCluster(mTPCTracksClusIdx, iCl, *mTPCClusterIdxStruct, sector, row);
    float clTPCX;
    std::array<float, 2> clTPCYZ;
    mFastTransform->TransformIdeal(sector, row, clTPC.getPad(), clTPC.getTime(), clTPCX, clTPCYZ[0], clTPCYZ[1], clusterTimeBinOffset);
    sector %= SECTORSPERSIDE;
    mCache[row].clAvailable = 1;
    mCache[row].clY = clTPCYZ[0];
    mCache[row].clZ = clTPCYZ[1];
    mCache[row].clAngle = o2::math_utils::sector2Angle(sector);
  }

  // first extrapolate through TPC and store track position at each pad row
  for (int iRow = 0; iRow < param::NPadRows; ++iRow) {
    if (!mCache[iRow].clAvailable) {
      continue;
    }
    if (!trkWork.rotate(mCache[iRow].clAngle)) {
      LOG(DEBUG) << "Failed to rotate track during first extrapolation";
      return false;
    }
    if (!propagator->PropagateToXBxByBz(trkWork, param::RowX[iRow], mMaxSnp, mMaxStep, mMatCorr)) {
      LOG(DEBUG) << "Failed on first extrapolation";
      return false;
    }
    mCache[iRow].y[ExtOut] = trkWork.getY();
    mCache[iRow].z[ExtOut] = trkWork.getZ();
    mCache[iRow].sy2[ExtOut] = trkWork.getSigmaY2();
    mCache[iRow].szy[ExtOut] = trkWork.getSigmaZY();
    mCache[iRow].sz2[ExtOut] = trkWork.getSigmaZ2();
    mCache[iRow].phi[ExtOut] = trkWork.getSnp();
    mCache[iRow].tgl[ExtOut] = trkWork.getTgl();
    //printf("Track alpha at row %i: %.2f, Y(%.2f), Z(%.2f)\n", iRow, trkWork.getAlpha(), trkWork.getY(), trkWork.getZ());
  }

  // now continue to TOF and update track with TOF cluster
  if (!trkWork.rotate(clTOFAlpha)) {
    LOG(DEBUG) << "Failed to rotate into TOF cluster sector frame";
    return false;
  }
  //float ca, sa;
  //o2::math_utils::sincos(clTOFAlpha, sa, ca);
  //float clTOFX = clTOF.getX() * ca + clTOF.getY() * sa;                                 // cluster x in sector coordinate frame
  //std::array<float, 2> clTOFYZ{ -clTOF.getX() * sa + clTOF.getY() * ca, clTOF.getZ() }; // cluster y and z in sector coordinate frame
  float clTOFX = clTOF.getX();
  std::array<float, 2> clTOFYZ{clTOF.getY(), clTOF.getZ()};
  std::array<float, 3> clTOFCov{mSigYZ2TOF, 0.f, mSigYZ2TOF}; // assume no correlation between y and z and equal cluster error sigma^2 = (3cm)^2 / 12
  if (!propagator->PropagateToXBxByBz(trkWork, clTOFX, mMaxSnp, mMaxStep, mMatCorr)) {
    LOG(DEBUG) << "Failed final propagation to TOF radius";
    return false;
  }
  if (!trkWork.update(clTOFYZ, clTOFCov)) {
    LOG(DEBUG) << "Failed to update extrapolated ITS track with TOF cluster";
    return false;
  }

  // go back through the TPC and store updated track positions
  for (int iRow = param::NPadRows; iRow--;) {
    if (!mCache[iRow].clAvailable) {
      continue;
    }
    if (!trkWork.rotate(mCache[iRow].clAngle)) {
      LOG(DEBUG) << "Failed to rotate track during back propagation";
      return false;
    }
    if (!propagator->PropagateToXBxByBz(trkWork, param::RowX[iRow], mMaxSnp, mMaxStep, mMatCorr)) {
      LOG(DEBUG) << "Failed on back propagation";
      //printf("trkX(%.2f), clX(%.2f), clY(%.2f), clZ(%.2f), alphaTOF(%.2f)\n", trkWork.getX(), param::RowX[iRow], clTOFYZ[0], clTOFYZ[1], clTOFAlpha);
      return false;
    }
    mCache[iRow].y[ExtIn] = trkWork.getY();
    mCache[iRow].z[ExtIn] = trkWork.getZ();
    mCache[iRow].sy2[ExtIn] = trkWork.getSigmaY2();
    mCache[iRow].szy[ExtIn] = trkWork.getSigmaZY();
    mCache[iRow].sz2[ExtIn] = trkWork.getSigmaZ2();
    mCache[iRow].phi[ExtIn] = trkWork.getSnp();
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
    mCache[iRow].phi[Int] = (mCache[iRow].phi[ExtOut] + mCache[iRow].phi[ExtIn]) / 2.f;
    mCache[iRow].tgl[Int] = (mCache[iRow].tgl[ExtOut] + mCache[iRow].tgl[ExtIn]) / 2.f;

    TPCClusterResiduals res;
    res.setDY(mCache[iRow].clY - mCache[iRow].y[Int]);
    res.setDZ(mCache[iRow].clZ - mCache[iRow].z[Int]);
    res.setY(mCache[iRow].y[Int]);
    res.setZ(mCache[iRow].z[Int]);
    res.setPhi(mCache[iRow].phi[Int]);
    res.setTgl(mCache[iRow].tgl[Int]);
    res.sec = o2::math_utils::angle2Sector(mCache[iRow].clAngle);
    res.dRow = deltaRow;
    res.row = iRow;
    mClRes.push_back(std::move(res));
    ++nMeasurements;
    deltaRow = 1;
  }

  mTrackData[trkIdx].trkId = matchITSTPC.getRefTPC();
  mTrackData[trkIdx].eta = trkTPC.getEta();
  mTrackData[trkIdx].phi = trkTPC.getSnp();
  mTrackData[trkIdx].qPt = trkTPC.getQ2Pt();
  mTrackData[trkIdx].chi2TPC = trkTPC.getChi2();
  mTrackData[trkIdx].chi2ITS = trkITS.getChi2();
  mTrackData[trkIdx].nClsTPC = trkTPC.getNClusterReferences();
  mTrackData[trkIdx].nClsITS = trkITS.getNumberOfClusters();
  mTrackData[trkIdx].clIdx.setEntries(nMeasurements);

  LOG(DEBUG) << "Track interpolation successfull";
  return true;
}

bool TrackInterpolation::extrapolateTrackITS(const o2::its::TrackITS& trkITS, const TrackTPC& trkTPC, float trkTime, int trkIdTPC)
{
  // extrapolate ITS-only track through TPC and store residuals to TPC clusters in the output vectors
  size_t trkIdx = mTrackData.size() - 1;
  mTrackData[trkIdx].clIdx.setFirstEntry(mClRes.size());
  auto trk = trkITS.getParamOut();
  float clusterTimeBinOffset = trkTime / mTPCTimeBinMUS;
  auto propagator = o2::base::Propagator::Instance();
  unsigned short rowPrev = 0;
  unsigned short nMeasurements = 0;
  for (int iCl = trkTPC.getNClusterReferences(); iCl--;) {
    uint8_t sector, row;
    uint32_t clusterIndexInRow;
    const auto& cl = trkTPC.getCluster(mTPCTracksClusIdx, iCl, *mTPCClusterIdxStruct, sector, row);
    float x = 0, y = 0, z = 0;
    mFastTransform->TransformIdeal(sector, row, cl.getPad(), cl.getTime(), x, y, z, clusterTimeBinOffset);
    if (!trk.rotate(o2::math_utils::sector2Angle(sector))) {
      return false;
    }
    if (!propagator->PropagateToXBxByBz(trk, x, mMaxSnp, mMaxStep, mMatCorr)) {
      return false;
    }
    TPCClusterResiduals res;
    res.setDY(y - trk.getY());
    res.setDY(z - trk.getZ());
    res.setY(trk.getY());
    res.setZ(trk.getZ());
    res.setPhi(trk.getSnp());
    res.setTgl(trk.getTgl());
    res.sec = o2::math_utils::angle2Sector(trk.getAlpha());
    res.dRow = row - rowPrev;
    res.row = row;
    rowPrev = row;
    mClRes.push_back(std::move(res));
    ++nMeasurements;
  }
  mTrackData[trkIdx].trkId = trkIdTPC;
  mTrackData[trkIdx].eta = trkTPC.getEta();
  mTrackData[trkIdx].phi = trkTPC.getSnp();
  mTrackData[trkIdx].qPt = trkTPC.getQ2Pt();
  mTrackData[trkIdx].chi2TPC = trkTPC.getChi2();
  mTrackData[trkIdx].chi2ITS = trkITS.getChi2();
  mTrackData[trkIdx].nClsTPC = trkTPC.getNClusterReferences();
  mTrackData[trkIdx].nClsITS = trkITS.getNumberOfClusters();
  mTrackData[trkIdx].clIdx.setEntries(nMeasurements);

  return true;
}

void TrackInterpolation::reset()
{
  mTrackData.clear();
  mClRes.clear();
}
