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
#include "DetectorsBase/Propagator.h"

#include <fairlogger/Logger.h>

using namespace o2::tpc;

void TrackInterpolation::init()
{
  // perform initialization
  attachInputTrees();

  const auto& elParam = ParameterElectronics::Instance();
  mTPCTimeBinMUS = elParam.ZbinWidth;

  std::unique_ptr<TPCFastTransform> fastTransform = (TPCFastTransformHelperO2::instance()->create(0));
  mFastTransform = std::move(fastTransform);

  if (mTreeOutTrackData && mTreeOutClusterRes) {
    prepareOutputTrees();
  }
}

void TrackInterpolation::process()
{
  // main processing function
  loadEntryForTrees(0); // TODO for now all input data is stored in trees with a single entry

  std::set<unsigned int> tracksDone; // to store indices of ITS-TPC matched tracks that have been processed

  LOG(info) << "Processing " << mTOFMatchVecInput->size() << " ITS-TPC-TOF matched tracks out of "
            << mITSTPCTrackVecInput->size() << " ITS-TPC matched tracks";

  // TODO reserve only a fraction of the needed space for all tracks? How many tracks pass on average the quality cuts with how many TPC clusters?
  mTrackData.reserve(mTOFMatchVecInput->size());
  mClRes.reserve(mTOFMatchVecInput->size() * param::NPadRows);

  int nTracksTPC = mTPCTrackVecInput->size();

  for (const auto& trkTOF : *mTOFMatchVecInput) {
    // process ITS-TPC-TOF matched tracks
    if (!trackPassesQualityCuts(mITSTPCTrackVecInput->at(trkTOF.first.getIndex()))) {
      continue;
    }
    mTrackData.emplace_back();
    if (!interpolateTrackITSTOF(trkTOF)) {
      mTrackData.pop_back();
      continue;
    }
    mTrackData.back().nTracksInEvent = nTracksTPC;
    tracksDone.insert(trkTOF.first.getIndex());
  }

  LOG(info) << "Could process " << tracksDone.size() << " ITS-TPC-TOF matched tracks successfully";

  if (mDoITSOnlyTracks) {
    size_t nTracksDoneITS = 0;
    size_t nTracksSkipped = 0;
    for (std::size_t iTrk = 0; iTrk < mITSTPCTrackVecInput->size(); ++iTrk) {
      // process ITS-TPC matched tracks that were not matched to TOF
      if (tracksDone.find(iTrk) != tracksDone.end()) {
        // track also has a matching cluster in TOF and has already been processed
        ++nTracksSkipped;
        continue;
      }
      const auto& trk = mITSTPCTrackVecInput->at(iTrk);
      if (!trackPassesQualityCuts(trk, false)) {
        continue;
      }
      mTrackData.emplace_back();
      const auto& trkIdTPC = trk.getRefTPC().getIndex();
      const auto& trkTPC = mTPCTrackVecInput->at(trkIdTPC);
      const auto& trkITS = mITSTrackVecInput->at(trk.getRefITS().getIndex());
      if (!extrapolateTrackITS(trkITS, trkTPC, trk.getTimeMUS().getTimeStamp(), trkIdTPC)) {
        mTrackData.pop_back();
        continue;
      }
      mTrackData.back().nTracksInEvent = nTracksTPC;
      ++nTracksDoneITS;
    }
    LOG(info) << "Could process " << nTracksDoneITS << " ITS-TPC matched tracks successfully";
    LOG(info) << "Skipped " << nTracksSkipped << " tracks, as they were successfully propagated to TOF";
  }

  if (mTreeOutTrackData) {
    mTreeOutTrackData->Fill();
  }
  if (mTreeOutClusterRes) {
    mTreeOutClusterRes->Fill();
  }
}

bool TrackInterpolation::trackPassesQualityCuts(const o2::dataformats::TrackTPCITS& matchITSTPC, bool hasOuterPoint) const
{
  // apply track quality cuts (assume different settings for track with and without points in TRD or TOF)
  const auto& trkTPC = mTPCTrackVecInput->at(matchITSTPC.getRefTPC().getIndex());
  const auto& trkITS = mITSTrackVecInput->at(matchITSTPC.getRefITS().getIndex());
  if (hasOuterPoint) {
    // track has a match in TRD or TOF
    if (trkTPC.getNClusterReferences() < param::MinTPCNCls ||
        trkITS.getNumberOfClusters() < param::MinITSNCls) {
      return false;
    }
    if (trkTPC.getChi2() / trkTPC.getNClusterReferences() > param::MaxTPCChi2 ||
        trkITS.getChi2() / trkITS.getNumberOfClusters() > param::MaxITSChi2) {
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

bool TrackInterpolation::interpolateTrackITSTOF(const std::pair<o2::dataformats::EvIndex<>, o2::dataformats::MatchInfoTOF>& matchTOF)
{
  // get TPC cluster residuals to ITS-TOF only tracks
  size_t trkIdx = mTrackData.size() - 1;
  auto propagator = o2::base::Propagator::Instance();
  const auto& matchITSTPC = mITSTPCTrackVecInput->at(matchTOF.first.getIndex());
  const auto& clTOF = mTOFClusterVecInput->at(matchTOF.second.getTOFClIndex());
  //const int clTOFSec = (TMath::ATan2(-clTOF.getY(), -clTOF.getX()) + o2::constants::math::PI) * o2::constants::math::Rad2Deg * 0.05; // taken from TOF cluster class as there is no const getter for the sector
  const int clTOFSec = clTOF.getCount();
  const float clTOFAlpha = o2::utils::Sector2Angle(clTOFSec);
  const auto& trkTPC = mTPCTrackVecInput->at(matchITSTPC.getRefTPC().getIndex());
  const auto& trkITS = mITSTrackVecInput->at(matchITSTPC.getRefITS().getIndex());
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
    trkTPC.getClusterReference(iCl, sector, row, clusterIndexInRow);
    const auto& clTPC = trkTPC.getCluster(iCl, *mTPCClusterIdxStruct, sector, row);
    float clTPCX;
    std::array<float, 2> clTPCYZ;
    mFastTransform->TransformIdeal(sector, row, clTPC.getPad(), clTPC.getTime(), clTPCX, clTPCYZ[0], clTPCYZ[1], clusterTimeBinOffset);
    sector %= SECTORSPERSIDE;
    mCache[row].clAvailable = 1;
    mCache[row].clY = clTPCYZ[0];
    mCache[row].clZ = clTPCYZ[1];
    mCache[row].clAngle = o2::utils::Sector2Angle(sector);
  }

  // first extrapolate through TPC and store track position at each pad row
  for (int iRow = 0; iRow < param::NPadRows; ++iRow) {
    if (!mCache[iRow].clAvailable) {
      continue;
    }
    if (!trkWork.rotate(mCache[iRow].clAngle)) {
      LOG(debug) << "Failed to rotate track during first extrapolation";
      return false;
    }
    if (!propagator->PropagateToXBxByBz(trkWork, param::RowX[iRow], o2::constants::physics::MassPionCharged, mMaxSnp, mMaxStep, mMatCorr)) {
      LOG(debug) << "Failed on first extrapolation";
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
    LOG(debug) << "Failed to rotate into TOF cluster sector frame";
    return false;
  }
  //float ca, sa;
  //o2::utils::sincosf(clTOFAlpha, sa, ca);
  //float clTOFX = clTOF.getX() * ca + clTOF.getY() * sa;                                 // cluster x in sector coordinate frame
  //std::array<float, 2> clTOFYZ{ -clTOF.getX() * sa + clTOF.getY() * ca, clTOF.getZ() }; // cluster y and z in sector coordinate frame
  float clTOFX = clTOF.getX();
  std::array<float, 2> clTOFYZ{clTOF.getY(), clTOF.getZ()};
  std::array<float, 3> clTOFCov{mSigYZ2TOF, 0.f, mSigYZ2TOF}; // assume no correlation between y and z and equal cluster error sigma^2 = (3cm)^2 / 12
  if (!propagator->PropagateToXBxByBz(trkWork, clTOFX, o2::constants::physics::MassPionCharged, mMaxSnp, mMaxStep, mMatCorr)) {
    LOG(debug) << "Failed final propagation to TOF radius";
    return false;
  }
  if (!trkWork.update(clTOFYZ, clTOFCov)) {
    LOG(debug) << "Failed to update extrapolated ITS track with TOF cluster";
    return false;
  }

  // go back through the TPC and store updated track positions
  for (int iRow = param::NPadRows; iRow--;) {
    if (!mCache[iRow].clAvailable) {
      continue;
    }
    if (!trkWork.rotate(mCache[iRow].clAngle)) {
      LOG(debug) << "Failed to rotate track during back propagation";
      return false;
    }
    if (!propagator->PropagateToXBxByBz(trkWork, param::RowX[iRow], o2::constants::physics::MassPionCharged, mMaxSnp, mMaxStep, mMatCorr)) {
      LOG(debug) << "Failed on back propagation";
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
    res.sec = o2::utils::Angle2Sector(mCache[iRow].clAngle);
    res.dRow = deltaRow;
    res.row = iRow;
    mClRes.push_back(std::move(res));
    ++nMeasurements;
    deltaRow = 1;
  }

  mTrackData[trkIdx].trkId = matchITSTPC.getRefTPC().getIndex();
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
    trkTPC.getClusterReference(iCl, sector, row, clusterIndexInRow);
    const auto& cl = trkTPC.getCluster(iCl, *mTPCClusterIdxStruct, sector, row);
    float x = 0, y = 0, z = 0;
    mFastTransform->TransformIdeal(sector, row, cl.getPad(), cl.getTime(), x, y, z, clusterTimeBinOffset);
    if (!trk.rotate(o2::utils::Sector2Angle(sector))) {
      return false;
    }
    if (!propagator->PropagateToXBxByBz(trk, x, o2::constants::physics::MassPionCharged, mMaxSnp, mMaxStep, mMatCorr)) {
      return false;
    }
    TPCClusterResiduals res;
    res.setDY(y - trk.getY());
    res.setDY(z - trk.getZ());
    res.setY(trk.getY());
    res.setZ(trk.getZ());
    res.setPhi(trk.getSnp());
    res.setTgl(trk.getTgl());
    res.sec = o2::utils::Angle2Sector(trk.getAlpha());
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

void TrackInterpolation::loadEntryForTrees(int iEntry)
{
  mTreeITSTPCTracks->GetEntry(iEntry);
  mTreeTPCTracks->GetEntry(iEntry);
  mTreeITSTracks->GetEntry(iEntry);
  mTreeITSClusters->GetEntry(iEntry);
  mTreeTOFMatches->GetEntry(iEntry);
  mTreeTOFClusters->GetEntry(iEntry);
  mTPCClusterReader->read(iEntry);
  mTPCClusterReader->fillIndex(*mTPCClusterIdxStructOwn.get(), mTPCClusterBufferOwn, mTPCClusterMCBufferOwn);
  mTPCClusterIdxStruct = mTPCClusterIdxStructOwn.get();
}

void TrackInterpolation::attachInputTrees()
{
  // access input for ITS-TPC matched tracks (needed for the references to ITS/TPC tracks)
  if (!mTreeITSTPCTracks) {
    LOG(fatal) << "The input tree for ITS-TPC matched tracks is not set!";
  }
  if (!mTreeITSTPCTracks->GetBranch(mITSTPCTrackBranchName.data())) {
    LOG(fatal) << "Did not find ITS-TPC matched tracks branch " << mITSTPCTrackBranchName << " in the input tree";
  }
  mTreeITSTPCTracks->SetBranchAddress(mITSTPCTrackBranchName.data(), &mITSTPCTrackVecInput);
  LOG(info) << "Attached ITS-TPC tracks " << mITSTPCTrackBranchName << " branch with "
            << mTreeITSTPCTracks->GetEntries() << " entries";
  // access input for TPC tracks (only used for cluster access)
  if (!mTreeTPCTracks) {
    LOG(fatal) << "The input tree for TPC tracks is not set!";
  }
  if (!mTreeTPCTracks->GetBranch(mTPCTrackBranchName.data())) {
    LOG(fatal) << "Did not find TPC tracks branch " << mTPCTrackBranchName << " in the input tree";
  }
  mTreeTPCTracks->SetBranchAddress(mTPCTrackBranchName.data(), &mTPCTrackVecInput);
  LOG(info) << "Attached TPC tracks " << mTPCTrackBranchName << " branch with "
            << mTreeTPCTracks->GetEntries() << " entries";
  // access input for TPC clusters
  if (!mTPCClusterReader) {
    LOG(fatal) << "TPC clusters reader is not set";
  }
  LOG(info) << "Attached TPC clusters reader with " << mTPCClusterReader->getTreeSize() << "entries";
  mTPCClusterIdxStructOwn = std::make_unique<ClusterNativeAccess>();
  // access input for ITS tracks
  if (!mTreeITSTracks) {
    LOG(fatal) << "The input tree for ITS tracks is not set!";
  }
  if (!mTreeITSTracks->GetBranch(mITSTrackBranchName.data())) {
    LOG(fatal) << "Did not find ITS tracks branch " << mITSTrackBranchName << " in the input tree";
  }
  mTreeITSTracks->SetBranchAddress(mITSTrackBranchName.data(), &mITSTrackVecInput);
  LOG(info) << "Attached ITS tracks " << mITSTrackBranchName << " branch with "
            << mTreeITSTracks->GetEntries() << " entries";
  // access input for ITS clusters
  if (!mTreeITSClusters) {
    LOG(fatal) << "The input tree for ITS clusters is not set!";
  }
  if (!mTreeITSClusters->GetBranch(mITSClusterBranchName.data())) {
    LOG(fatal) << "Did not find ITS clusters branch " << mITSClusterBranchName << " in the input tree";
  }
  mTreeITSClusters->SetBranchAddress(mITSClusterBranchName.data(), &mITSClusterVecInput);
  LOG(info) << "Attached ITS clusters " << mITSClusterBranchName << " branch with "
            << mTreeITSClusters->GetEntries() << " entries";
  // access input for TPC-TOF matching information
  if (!mTreeTOFMatches) {
    LOG(fatal) << "The input tree for with TOF matching information is not set!";
  }
  if (!mTreeTOFMatches->GetBranch(mTOFMatchingBranchName.data())) {
    LOG(fatal) << "Did not find TOF matches branch " << mTOFMatchingBranchName << " in the input tree";
  }
  mTreeTOFMatches->SetBranchAddress(mTOFMatchingBranchName.data(), &mTOFMatchVecInput);
  LOG(info) << "Attached TOF matches " << mTOFMatchingBranchName << " branch with "
            << mTreeTOFMatches->GetEntries() << " entries";
  // access input for TOF clusters
  if (!mTreeTOFClusters) {
    LOG(fatal) << "The input tree for with TOF clusters is not set!";
  }
  if (!mTreeTOFClusters->GetBranch(mTOFClusterBranchName.data())) {
    LOG(fatal) << "Did not find TOF clusters branch " << mTOFClusterBranchName << " in the input tree";
  }
  mTreeTOFClusters->SetBranchAddress(mTOFClusterBranchName.data(), &mTOFClusterVecInput);
  LOG(info) << "Attached TOF clusters " << mTOFClusterBranchName << " branch with "
            << mTreeTOFClusters->GetEntries() << " entries";

  // TODO: add MC information
}

void TrackInterpolation::prepareOutputTrees()
{
  mTreeOutTrackData->Branch("tracks", &mTrackDataPtr);
  mTreeOutClusterRes->Branch("residuals", &mClResPtr);
}

void TrackInterpolation::reset()
{
  mTrackData.clear();
  mClRes.clear();
}
