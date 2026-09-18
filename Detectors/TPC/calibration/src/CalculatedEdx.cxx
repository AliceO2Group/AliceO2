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

///
/// @file   CalculatedEdx.cxx
/// @author Tuba Gündem, tuba.gundem@cern.ch
///

#include "TPCCalibration/CalculatedEdx.h"
#include "TPCBase/PadPos.h"
#include "TPCBase/ROC.h"
#include "TPCBase/Mapper.h"
#include "DataFormatsTPC/ClusterNative.h"
#include "DetectorsBase/Propagator.h"
#include "CCDB/BasicCCDBManager.h"
#include "TPCBaseRecSim/CDBInterface.h"
#include "TPCReconstruction/TPCFastTransformHelperO2.h"
#include "CalibdEdxTrackTopologyPol.h"
#include "DataFormatsParameters/GRPMagField.h"
#include "GPUO2InterfaceUtils.h"
#include "GPUTPCGMMergedTrackHit.h"

#include <cstdlib>
#include <limits>

using namespace o2::tpc;

namespace
{
// sentinel for minChargeTotROC/minChargeMaxROC
constexpr float kNoValidCharge = std::numeric_limits<float>::max();

// dEdxSettings::sameRowClusterMethod == 1: one synthesized cluster summing a mergeable group's fragments'
// qTot/qMax and charge-weighted pad/time
o2::tpc::ClusterNative buildMergedClusterSum(const std::vector<o2::tpc::ClusterNative>& fragments)
{
  float weightedPadSum = 0.f;
  float weightedTimeSum = 0.f;
  float totalCharge = 0.f;
  uint16_t maxCharge = 0;

  const o2::tpc::ClusterNative& firstCluster = fragments[0];
  o2::tpc::ClusterNative combinedCluster = firstCluster;

  for (const auto& cl : fragments) {
    const float clPad = cl.getPad();
    const float clTime = cl.getTime();
    const uint16_t clqTot = cl.getQtot();
    const uint16_t clqMax = cl.qMax;

    weightedPadSum += clPad * clqTot;
    weightedTimeSum += clTime * clqTot;
    totalCharge += clqTot;
    maxCharge = std::max(maxCharge, clqMax);
  }

  if (totalCharge > o2::tpc::ClusterNative::maxRegularQtot) {
    combinedCluster.setSaturatedQtot(static_cast<uint32_t>(totalCharge));
  } else {
    combinedCluster.qTotPacked = static_cast<uint16_t>(totalCharge);
  }
  combinedCluster.qMax = maxCharge;
  combinedCluster.padPacked = static_cast<uint16_t>(weightedPadSum / totalCharge * o2::tpc::ClusterNative::scalePadPacked);
  combinedCluster.timeFlagsPacked = (static_cast<uint32_t>(weightedTimeSum / totalCharge * o2::tpc::ClusterNative::scaleTimePacked) & 0xFFFFFF) | (firstCluster.timeFlagsPacked & 0xFF000000);
  return combinedCluster;
}
} // namespace

const CalculatedEdx::RowFragment& CalculatedEdx::pickDominantFragment(const std::vector<RowFragment>& fragments)
{
  const RowFragment* dominant = &fragments[0];
  for (const auto& frag : fragments) {
    if (frag.cl.getQtot() > dominant->cl.getQtot()) {
      dominant = &frag;
    }
  }
  return *dominant;
}

CalculatedEdx::CalculatedEdx()
{
  mTPCCorrMapFull = TPCFastTransformHelperO2::instance()->create(0);
  rebuildTPCCorrMapPOD();
}

void CalculatedEdx::rebuildTPCCorrMapPOD()
{
  // re-flatten the regular mTPCCorrMapFull into the POD buffer setRefit()'s GPU refitter reads
  gpu::aligned_unique_buffer_ptr<gpu::TPCFastTransformPOD> buffer;
  gpu::TPCFastTransformPOD::create(buffer, *mTPCCorrMapFull);
  mTPCCorrMapBuffer = std::move(buffer);
  mTPCCorrMap = mTPCCorrMapBuffer.get();
}

void CalculatedEdx::setTPCCorrMap(const o2::gpu::TPCFastTransform& corrMap)
{
  if (mRefit) {
    // the existing refitter holds a pointer into the correction map buffer we are about to replace; drop it
    // rather than leave it dangling. Normal per-event usage (loadCalibsFromCCDB()/loadCalibsFromLocalCCDBFolder()
    // followed by setRefit()) re-creates it right after this call with the new map, so this is not an error.
    LOGP(warning, "CalculatedEdx::setTPCCorrMap() called after setRefit(); invalidating the existing refitter, call setRefit() again before using it.");
    mRefit.reset();
  }
  mTPCCorrMapFull = TPCFastTransformHelperO2::instance()->create(0, corrMap.getCorrection());
  rebuildTPCCorrMapPOD();
}

void CalculatedEdx::setTPCVDrift(const o2::tpc::VDriftCorrFact& v)
{
  if (mRefit) {
    // see setTPCCorrMap() above: drop the now-stale refitter instead of leaving it dangling
    LOGP(warning, "CalculatedEdx::setTPCVDrift() called after setRefit(); invalidating the existing refitter, call setRefit() again before using it.");
    mRefit.reset();
  }
  TPCFastTransformHelperO2::instance()->updateCalibration(*mTPCCorrMapFull, 0, v.corrFact, v.refVDrift, v.getTimeOffset());
  rebuildTPCCorrMapPOD();
}

void CalculatedEdx::setMembers(std::vector<o2::tpc::TPCClRefElem>* tpcTrackClIdxVecInput, const o2::tpc::ClusterNativeAccess& clIndex, std::vector<o2::tpc::TrackTPC>* vTPCTracksArrayInp)
{
  mTracks = vTPCTracksArrayInp;
  mTPCTrackClIdxVecInput = tpcTrackClIdxVecInput;
  mClusterIndex = &clIndex;
}

void CalculatedEdx::setRefit(const unsigned int nHbfPerTf)
{
  mTPCRefitterShMap.resize(mClusterIndex->nClustersTotal);
  auto sizeOcc = o2::gpu::GPUO2InterfaceRefit::fillOccupancyMapGetSize(nHbfPerTf, nullptr);
  mTPCRefitterOccMap.resize(sizeOcc);
  std::fill(mTPCRefitterOccMap.begin(), mTPCRefitterOccMap.end(), 0);
  o2::gpu::GPUO2InterfaceRefit::fillSharedClustersAndOccupancyMap(mClusterIndex, *mTracks, mTPCTrackClIdxVecInput->data(), mTPCRefitterShMap.data(), mTPCRefitterOccMap.data(), nHbfPerTf);
  mRefit = std::make_unique<o2::gpu::GPUO2InterfaceRefit>(mClusterIndex, mTPCCorrMap, mFieldNominalGPUBz, mTPCTrackClIdxVecInput->data(), nHbfPerTf, mTPCRefitterShMap.data(), mTPCRefitterOccMap.data(), mTPCRefitterOccMap.size());
}

void CalculatedEdx::fillMissingClusters(int missingClusters[4], const float minChargeTot[4], const float minChargeMax[4], int method, std::array<std::vector<float>, 5>& chargeTotROC, std::array<std::vector<float>, 5>& chargeMaxROC)
{
  if (method != 0 && method != 1) {
    LOGP(info, "Unrecognized subthreshold cluster treatment. Not adding virtual charges to the track!");
    return;
  }

  for (int roc = 0; roc < 4; roc++) {
    // minChargeTot/MaxROC[roc] is only ever updated from an accepted real cluster in that ROC
    // a region this track never accepted a single cluster in leaves it at kNoValidCharge, so skip
    // this ROC's fill entirely rather than injecting the sentinel as if it were a measured charge
    if (minChargeTot[roc] >= kNoValidCharge || minChargeMax[roc] >= kNoValidCharge) {
      continue;
    }
    const float chargeTot = (method == 1) ? minChargeTot[roc] / 2.f : minChargeTot[roc];
    const float chargeMax = (method == 1) ? minChargeMax[roc] / 2.f : minChargeMax[roc];
    for (int i = 0; i < missingClusters[roc]; i++) {

      chargeTotROC[roc].emplace_back(chargeTot);
      chargeTotROC[4].emplace_back(chargeTot);

      chargeMaxROC[roc].emplace_back(chargeMax);
      chargeMaxROC[4].emplace_back(chargeMax);
    }
  }
}

void CalculatedEdx::handleSameRowClusters(o2::tpc::TrackTPC& track, std::vector<std::pair<unsigned char, unsigned char>>& rowOrder, std::map<std::pair<unsigned char, unsigned char>, std::vector<int>>& clustersByRow, std::set<std::pair<unsigned char, unsigned char>>& mergeableRows, std::map<int, std::tuple<unsigned char, unsigned char, unsigned int>>& clusterReferencesByIndex)
{
  // get number of clusters
  const int nClusters = track.getNClusterReferences();

  // group clusters by (sector, row)
  for (int iCl = 0; iCl < nClusters; iCl++) {
    const o2::tpc::ClusterNative& cl = track.getCluster(*mTPCTrackClIdxVecInput, iCl, *mClusterIndex);

    unsigned char sectorIndex = 0;
    unsigned char rowIndex = 0;
    unsigned int clusterIndexNumb = 0;

    track.getClusterReference(*mTPCTrackClIdxVecInput, iCl, sectorIndex, rowIndex, clusterIndexNumb);

    const auto rowKey = std::make_pair(sectorIndex, rowIndex);
    if (clustersByRow.find(rowKey) == clustersByRow.end()) {
      rowOrder.emplace_back(rowKey);
    }

    // add the cluster index to the corresponding (sector, row) key in clustersByRow
    clustersByRow[rowKey].emplace_back(iCl);

    // store the reference data in clusterReferencesByIndex
    clusterReferencesByIndex[iCl] = std::make_tuple(sectorIndex, rowIndex, clusterIndexNumb);
  }

  // flag groups of several clusters that ended up in the same (sector, row) as eligible for
  // dEdxSettings::sameRowClusterMethod to merge but ONLY when they are close in pad and time
  for (const auto& [rowKey, clusterIndices] : clustersByRow) {
    if (clusterIndices.size() <= 1) {
      continue;
    }

    // proximity gate: skip the merge if any pair in the group is farther apart than the pad/time window
    float minPad = 1e9f, maxPad = -1e9f, minTime = 1e9f, maxTime = -1e9f;
    for (int clusterIdx : clusterIndices) {
      const o2::tpc::ClusterNative& cl = track.getCluster(*mTPCTrackClIdxVecInput, clusterIdx, *mClusterIndex);
      minPad = std::min(minPad, cl.getPad());
      maxPad = std::max(maxPad, cl.getPad());
      minTime = std::min(minTime, cl.getTime());
      maxTime = std::max(maxTime, cl.getTime());
    }
    if ((maxPad - minPad) > mSameRowMaxPadDiff || (maxTime - minTime) > mSameRowMaxTimeDiff) {
      continue; // looper legs / distinct crossings -> keep as separate samples
    }

    mergeableRows.insert(rowKey);
  }
}

void CalculatedEdx::handleSameRowClusters(const std::vector<o2::tpc::ClusterNative>& clusters, const ClInfoVec& clusterInfos, std::vector<std::pair<unsigned char, unsigned char>>& rowOrder, std::map<std::pair<unsigned char, unsigned char>, std::vector<int>>& clustersByRow, std::set<std::pair<unsigned char, unsigned char>>& mergeableRows)
{
  const int nClusters = static_cast<int>(clusters.size());

  // group clusters by (sector, row)
  for (int iCl = 0; iCl < nClusters; iCl++) {
    const auto rowKey = std::make_pair(clusterInfos[iCl].sectorIndex, clusterInfos[iCl].rowIndex);
    if (clustersByRow.find(rowKey) == clustersByRow.end()) {
      rowOrder.emplace_back(rowKey);
    }

    // add the cluster index to the corresponding (sector, row) key in clustersByRow
    clustersByRow[rowKey].emplace_back(iCl);
  }

  // flag groups of several clusters in the same (sector, row) as eligible for dEdxSettings::sameRowClusterMethod to merge, only when they are close in pad and time
  for (const auto& [rowKey, clusterIndices] : clustersByRow) {
    if (clusterIndices.size() <= 1) {
      continue;
    }

    // proximity gate: skip the merge if any pair in the group is farther apart than the pad/time window
    float minPad = 1e9f, maxPad = -1e9f, minTime = 1e9f, maxTime = -1e9f;
    for (int clusterIdx : clusterIndices) {
      const o2::tpc::ClusterNative& cl = clusters[clusterIdx];
      minPad = std::min(minPad, cl.getPad());
      maxPad = std::max(maxPad, cl.getPad());
      minTime = std::min(minTime, cl.getTime());
      maxTime = std::max(maxTime, cl.getTime());
    }
    if ((maxPad - minPad) > mSameRowMaxPadDiff || (maxTime - minTime) > mSameRowMaxTimeDiff) {
      continue; // looper legs / distinct crossings -> keep as separate samples
    }

    mergeableRows.insert(rowKey);
  }
}

void CalculatedEdx::gatherRowClusterData(o2::tpc::TrackTPC& track, std::vector<RowClusterData>& rowData, AverageOccupancy& averageOcc)
{
  rowData.clear();

  bool refitAbandoned = false;

  // handle same (sector, row) clusters
  std::vector<std::pair<unsigned char, unsigned char>> rowOrder;
  std::map<std::pair<unsigned char, unsigned char>, std::vector<int>> clustersByRow;
  std::set<std::pair<unsigned char, unsigned char>> mergeableRows;
  std::map<int, std::tuple<unsigned char, unsigned char, unsigned int>> clusterReferencesByIndex;

  handleSameRowClusters(track, rowOrder, clustersByRow, mergeableRows, clusterReferencesByIndex);

  rowData.reserve(rowOrder.size());

  // per-region occupancy, for the average occupancy output
  std::array<std::vector<unsigned int>, 4> occupancyROC;

  // for tracking missing clusters
  unsigned char rowIndexOld = 255;
  unsigned char sectorIndexOld = 255;

  // loop over the clusters in the track's row-traversal order (rowOrder)
  for (const auto& rowKey : rowOrder) {
    const auto& clusterIndices = clustersByRow.at(rowKey);
    const unsigned char rowIndex = rowKey.second;
    const unsigned char sectorIndex = rowKey.first;

    std::vector<o2::tpc::ClusterNative> fragmentClusters;
    std::vector<bool> fragmentIsShared;
    fragmentClusters.reserve(clusterIndices.size());
    fragmentIsShared.reserve(clusterIndices.size());
    for (const int clusterIdx : clusterIndices) {
      fragmentClusters.emplace_back(track.getCluster(*mTPCTrackClIdxVecInput, clusterIdx, *mClusterIndex));
      const auto& [fragSectorIndex, fragRowIndex, clusterIndexNumb] = clusterReferencesByIndex[clusterIdx];
      const unsigned int absoluteIndex = mClusterIndex->clusterOffset[fragSectorIndex][fragRowIndex] + clusterIndexNumb;
      fragmentIsShared.emplace_back(mRefit ? (mTPCRefitterShMap[absoluteIndex] & o2::gpu::GPUTPCGMMergedTrackHit::flagShared) : 0);
    }

    const bool mergeable = mergeableRows.count(rowKey) > 0;
    gatherRowClusterDataForRow(track, fragmentClusters, fragmentIsShared, sectorIndex, rowIndex, mergeable, rowIndexOld, sectorIndexOld, occupancyROC, rowData, refitAbandoned);
    rowIndexOld = rowIndex;
    sectorIndexOld = sectorIndex;
  }

  // calculate average cl occupancy for the track per TPC region; skip clusters where getOccupancy() had no data (sentinel -1)
  double* const averageOccROC[4] = {&averageOcc.IROC, &averageOcc.OROC1, &averageOcc.OROC2, &averageOcc.OROC3};
  for (int roc = 0; roc < 4; roc++) {
    unsigned int sumOcc = 0;
    size_t nValidOcc = 0;
    for (const unsigned int occ : occupancyROC[roc]) {
      if (occ != static_cast<unsigned int>(-1)) {
        sumOcc += occ;
        ++nValidOcc;
      }
    }
    if (nValidOcc > 0) {
      *averageOccROC[roc] = static_cast<double>(sumOcc) / nValidOcc;
    }
  }
}

bool CalculatedEdx::propagateTrackToX(o2::track::TrackParCov& track, float xPosition, unsigned char sectorIndex) const
{
  const o2::track::TrackParCov trackBackup = track;
  bool check = track.rotate(o2::math_utils::detail::sector2Angle<float>(sectorIndex));
  if (check) {
    check = o2::base::Propagator::Instance()->PropagateToXBxByBz(track, xPosition, 0.999f, 0.5f, o2::base::Propagator::MatCorrType::USEMatCorrLUT);
  }
  if (!check) {
    track = trackBackup;
    check = track.rotate(o2::math_utils::detail::sector2Angle<float>(sectorIndex));
    if (check) {
      check = o2::base::Propagator::Instance()->PropagateToXBxByBz(track, xPosition, 0.999f, 0.5f, o2::base::Propagator::MatCorrType::USEMatCorrNONE);
    }
  }
  if (!check) {
    track = trackBackup;
    check = track.rotateParam(o2::math_utils::detail::sector2Angle<float>(sectorIndex));
    if (check) {
      check = track.propagateParamTo(xPosition, mFieldNominalGPUBz);
    }
  }
  if (!check) {
    track = trackBackup;
  }
  return check;
}

void CalculatedEdx::gatherRowClusterDataForRow(o2::tpc::TrackTPC& track, const std::vector<o2::tpc::ClusterNative>& fragmentClusters, const std::vector<bool>& fragmentIsShared, unsigned char sectorIndex, unsigned char rowIndex, bool mergeable, unsigned char rowIndexOld, unsigned char sectorIndexOld, std::array<std::vector<unsigned int>, 4>& occupancyROC, std::vector<RowClusterData>& rowData, bool& refitAbandoned)
{
  RowClusterData row;
  row.sectorIndex = sectorIndex;
  row.rowIndex = rowIndex;
  row.mergeable = mergeable;

  // get region and stack
  const int region = Mapper::REGION[rowIndex];
  const CRU cru(Sector(sectorIndex), region);
  const auto stack = cru.gemStack();
  StackID stackID{sectorIndex, stack};
  const int stackNumber = static_cast<int>(stack);

  row.region = region;
  row.stack = stack;
  row.stackID = stackID;
  row.stackNumber = stackNumber;

  // per-fragment quantities: pad depends on the individual cluster's own pad, so threshold/gain/gainResidual/occupancy/isDeadRegion
  // all looked up by pad, are computed once per fragment here
  row.fragments.reserve(fragmentClusters.size());
  for (size_t iFrag = 0; iFrag < fragmentClusters.size(); ++iFrag) {
    const o2::tpc::ClusterNative& cl = fragmentClusters[iFrag];
    RowFragment frag;
    frag.cl = cl;
    frag.isShared = fragmentIsShared[iFrag];
    frag.pad = std::clamp(static_cast<unsigned int>(cl.getPad() + 0.5f), static_cast<unsigned int>(0), Mapper::PADSPERROW[region][Mapper::getLocalRowFromGlobalRow(rowIndex)] - 1); // the left side of the pad is defined at e.g. 3.5 and the right side at 4.5
    frag.occupancy = getOccupancy(cl.getTime());
    frag.threshold = mCalibCont.getZeroSupressionThreshold(sectorIndex, rowIndex, frag.pad);
    frag.gain = mCalibCont.getGain(sectorIndex, rowIndex, frag.pad);
    frag.gainResidual = mCalibCont.getResidualGain(sectorIndex, rowIndex, frag.pad);
    frag.isDeadRegion = mCalibCont.isDead(static_cast<unsigned int>(sectorIndex), static_cast<gpu::tpccf::Row>(rowIndex), static_cast<gpu::tpccf::Pad>(frag.pad));

    row.fragments.emplace_back(std::move(frag));
  }

  // dEdxSettings::sameRowClusterMethod==1's merged sample
  if (mergeable && fragmentClusters.size() > 1) {
    row.mergedFragment.cl = buildMergedClusterSum(fragmentClusters);
    row.mergedFragment.pad = std::clamp(static_cast<unsigned int>(row.mergedFragment.cl.getPad() + 0.5f), static_cast<unsigned int>(0), Mapper::PADSPERROW[region][Mapper::getLocalRowFromGlobalRow(rowIndex)] - 1);
    row.mergedFragment.occupancy = getOccupancy(row.mergedFragment.cl.getTime());
    row.mergedFragment.threshold = mCalibCont.getZeroSupressionThreshold(sectorIndex, rowIndex, row.mergedFragment.pad);
    row.mergedFragment.gain = mCalibCont.getGain(sectorIndex, rowIndex, row.mergedFragment.pad);
    row.mergedFragment.gainResidual = mCalibCont.getResidualGain(sectorIndex, rowIndex, row.mergedFragment.pad);
    row.mergedFragment.isDeadRegion = mCalibCont.isDead(static_cast<unsigned int>(sectorIndex), static_cast<gpu::tpccf::Row>(rowIndex), static_cast<gpu::tpccf::Pad>(row.mergedFragment.pad));
    row.mergedFragment.isShared = row.fragments[0].isShared; // preserves the historical "first fragment's isShared" choice for a merged sample
  }

  // occupancy sample(s) for this row's average occupancy contribution
  std::array<std::vector<unsigned int>, 4>::size_type occStackIdx;
  if (stack == GEMstack::IROCgem) {
    occStackIdx = 0;
  } else if (stack == GEMstack::OROC1gem) {
    occStackIdx = 1;
  } else if (stack == GEMstack::OROC2gem) {
    occStackIdx = 2;
  } else {
    occStackIdx = 3;
  }
  if (mergeable && fragmentClusters.size() > 1) {
    occupancyROC[occStackIdx].emplace_back(row.mergedFragment.occupancy);
  } else {
    for (const auto& frag : row.fragments) {
      occupancyROC[occStackIdx].emplace_back(frag.occupancy);
    }
  }

  // get the x position of the track
  const float xPosition = Mapper::instance().getPadCentre(PadPos(rowIndex, 0)).X();
  bool check = true;
  bool refitFellBack = false;
  if (mRefit) {
    if (!refitAbandoned) {
      // snapshot the track's state as it stood before this row's refit attempt (i.e. after the previous row's
      // successful refit) -- RefitTrackAsGPU() writes back into `track` even on failure, so on failure this is
      // the most recent known-good state to fall back from, not the track's pristine pre-loop state
      const o2::track::TrackParCov trackBeforeRefit = track;
      // refit this track
      mRefit->setTrackReferenceX(xPosition);
      // RefitTrackAsGPU() returns < 0 when it fails; reachedReference is false when the fit succeeded but the final move-to-reference step could not reach xPosition, so both trigger the fallback.
      bool reachedReference = true;
      check = (mRefit->RefitTrackAsGPU(track, false, true, &reachedReference) < 0) ? false : reachedReference;
      if (!check || std::isnan(track.getParam(1))) {
        refitAbandoned = true;
        static_cast<o2::track::TrackParCov&>(track) = trackBeforeRefit;
        check = propagateTrackToX(track, xPosition, sectorIndex);
        refitFellBack = check;
      }
    } else {
      // already abandoned refit for this track (the previous row's RefitTrackAsGPU() failed)
      // keep propagating incrementally from `track`'s current state
      check = propagateTrackToX(track, xPosition, sectorIndex);
      refitFellBack = check;
    }
  } else if (mPropagateTrack) {
    // propagate this track to the plane X=xk (cm) in the field "b" (kG)
    check = propagateTrackToX(track, xPosition, sectorIndex);
  } else if (mPropagateParams) {
    // propagate the params of the track instead of full propagation; same rollback rationale as mPropagateTrack above
    const o2::track::TrackParCov trackBackup = track;
    check = track.rotateParam(o2::math_utils::detail::sector2Angle<float>(sectorIndex));
    if (check) {
      check = track.propagateParamTo(xPosition, mFieldNominalGPUBz);
    }
    if (!check) {
      static_cast<o2::track::TrackParCov&>(track) = trackBackup;
    }
  }

  row.propagationFailed = (!check || std::isnan(track.getParam(1)));
  row.refitFellBack = refitFellBack && !row.propagationFailed;
  ++mNRowsProcessed;
  if (row.propagationFailed) {
    ++mNPropagationFailed;
  } else if (row.refitFellBack) {
    ++mNRefitFallback;
  }

  // snapshot of the track state after refit/propagation to this row; reused by calculatedEdxFromRowData() for every settings entry
  row.trackSnapshot = track;

  // number of rows skipped between this row and the previous entry in rowOrder
  row.sameSectorAsPrevRow = (sectorIndexOld == sectorIndex);
  row.missingClusters = (rowIndexOld == 255) ? 0 : (std::abs(static_cast<int>(rowIndex) - static_cast<int>(rowIndexOld)) - 1);

  // veto the gap as a subthreshold candidate if any of its missing row(s) would land on a dead channel or off the padrow edge
  row.missingClusterGapDeadOrEdge = false;
  if (row.missingClusters > 0 && row.missingClusters <= mMaxMissingCl) {
    const o2::gpu::GPUTPCGeometry gpuGeom;
    const RowClusterData& prevRow = rowData.back();
    // bracket the gap by its lower/upper real row, independent of the rowOrder direction
    const int rowLo = std::min<int>(rowIndex, rowIndexOld);
    const int rowHi = std::max<int>(rowIndex, rowIndexOld);
    const float padLo = (rowLo == static_cast<int>(rowIndexOld)) ? prevRow.fragments[0].cl.getPad() : row.fragments[0].cl.getPad();
    const float padHi = (rowHi == static_cast<int>(rowIndexOld)) ? prevRow.fragments[0].cl.getPad() : row.fragments[0].cl.getPad();
    const float yLo = gpuGeom.LinearPad2Y(sectorIndex, rowLo, padLo);
    const float yHi = gpuGeom.LinearPad2Y(sectorIndex, rowHi, padHi);
    for (int k = 1; k <= row.missingClusters; ++k) {
      const unsigned char missingRow = static_cast<unsigned char>(rowLo + k);
      const float frac = static_cast<float>(k) / (row.missingClusters + 1);
      const float missingPad = gpuGeom.LinearY2Pad(sectorIndex, missingRow, yLo + (yHi - yLo) * frac);
      if (missingPad < 0.f || missingPad >= gpuGeom.NPads(missingRow)) {
        row.missingClusterGapDeadOrEdge = true;
        break;
      }
      const int missingRegion = Mapper::REGION[missingRow];
      const unsigned char missingPadClamped = std::clamp(static_cast<unsigned int>(missingPad + 0.5f), static_cast<unsigned int>(0), Mapper::PADSPERROW[missingRegion][Mapper::getLocalRowFromGlobalRow(missingRow)] - 1);
      if (mCalibCont.isDead(static_cast<unsigned int>(sectorIndex), static_cast<gpu::tpccf::Row>(missingRow), static_cast<gpu::tpccf::Pad>(missingPadClamped))) {
        row.missingClusterGapDeadOrEdge = true;
        break;
      }
    }
  }

  rowData.emplace_back(std::move(row));
}

void CalculatedEdx::gatherRowClusterData(o2::tpc::TrackTPC& track, const std::vector<o2::tpc::ClusterNative>& clusters, const ClInfoVec& clusterInfos, std::vector<RowClusterData>& rowData, AverageOccupancy& averageOcc)
{
  rowData.clear();

  bool refitAbandoned = false;

  // handle same (sector, row) clusters
  std::vector<std::pair<unsigned char, unsigned char>> rowOrder;
  std::map<std::pair<unsigned char, unsigned char>, std::vector<int>> clustersByRow;
  std::set<std::pair<unsigned char, unsigned char>> mergeableRows;

  handleSameRowClusters(clusters, clusterInfos, rowOrder, clustersByRow, mergeableRows);

  rowData.reserve(rowOrder.size());

  // per-region occupancy, for the average occupancy output
  std::array<std::vector<unsigned int>, 4> occupancyROC;

  // for tracking missing clusters
  unsigned char rowIndexOld = 255;
  unsigned char sectorIndexOld = 255;

  // loop over the clusters in the rowOrder; the refit/propagation to a row is done exactly once here regardless of how many raw clusters the row has
  for (const auto& rowKey : rowOrder) {
    const auto& clusterIndices = clustersByRow.at(rowKey);
    const unsigned char rowIndex = rowKey.second;
    const unsigned char sectorIndex = rowKey.first;

    std::vector<o2::tpc::ClusterNative> fragmentClusters;
    std::vector<bool> fragmentIsShared;
    fragmentClusters.reserve(clusterIndices.size());
    fragmentIsShared.reserve(clusterIndices.size());
    for (const int clusterIdx : clusterIndices) {
      fragmentClusters.emplace_back(clusters[clusterIdx]);
      // isShared cannot be looked up from mTPCRefitterShMap for externally supplied clusters, so it is taken from the info directly
      fragmentIsShared.emplace_back(clusterInfos[clusterIdx].isShared);
    }

    const bool mergeable = mergeableRows.count(rowKey) > 0;
    gatherRowClusterDataForRow(track, fragmentClusters, fragmentIsShared, sectorIndex, rowIndex, mergeable, rowIndexOld, sectorIndexOld, occupancyROC, rowData, refitAbandoned);
    rowData.back().inputClusterIndices = clusterIndices; // positions in the externally supplied clusters vector grouped into this row
    rowIndexOld = rowIndex;
    sectorIndexOld = sectorIndex;
  }

  // calculate average cl occupancy for the track per TPC region; skip clusters where getOccupancy() had no data (sentinel -1)
  double* const averageOccROC[4] = {&averageOcc.IROC, &averageOcc.OROC1, &averageOcc.OROC2, &averageOcc.OROC3};
  for (int roc = 0; roc < 4; roc++) {
    unsigned int sumOcc = 0;
    size_t nValidOcc = 0;
    for (const unsigned int occ : occupancyROC[roc]) {
      if (occ != static_cast<unsigned int>(-1)) {
        sumOcc += occ;
        ++nValidOcc;
      }
    }
    if (nValidOcc > 0) {
      *averageOccROC[roc] = static_cast<double>(sumOcc) / nValidOcc;
    }
  }
}

void CalculatedEdx::calculatedEdxFromRowData(const std::vector<RowClusterData>& rowData, const dEdxSettings& settings, size_t settingsIndex, float trackTime0, const o2::tpc::TrackTPC& trackOrig, const AverageOccupancy& averageOcc, dEdxInfo& output, const MCCompLabel* mcLabel)
{
  // NHits and NHitsSubthreshold values per region
  int nClsROC[4] = {0, 0, 0, 0};
  int nClsSubThreshROC[4] = {0, 0, 0, 0};

  const unsigned short sameRowClusterMethod = (settings.sameRowClusterMethod <= 2) ? settings.sameRowClusterMethod : 0;
  if (settings.sameRowClusterMethod > 2) {
    LOGP(warning, "Unrecognized sameRowClusterMethod {} (expected 0, 1, or 2); treating same-row cluster groups as method 0 (do not merge)", settings.sameRowClusterMethod);
  }

  // corrected qTot and qMax values per region
  const int nType = 5;
  std::array<std::vector<float>, nType> chargeTotROC;
  std::array<std::vector<float>, nType> chargeMaxROC;
  for (int i = 0; i < nType; ++i) {
    chargeTotROC[i].reserve(Mapper::PADROWS);
    chargeMaxROC[i].reserve(Mapper::PADROWS);
  }

  // per-region (IROC, OROC1, OROC2, OROC3) running minimum charge among accepted clusters, used as the virtual charge for that region's subthreshold clusters below
  float minChargeTotROC[4] = {kNoValidCharge, kNoValidCharge, kNoValidCharge, kNoValidCharge};
  float minChargeMaxROC[4] = {kNoValidCharge, kNoValidCharge, kNoValidCharge, kNoValidCharge};

  o2::utils::TreeStreamRedirector* debugStreamer = nullptr;
  std::vector<unsigned int> occupancyVector;
  if (mDebug) {
    setStreamer(settings.debugRootFile.c_str());
    debugStreamer = mStreamers.at(settings.debugRootFile).get();
    ++mDebugTrackIndex;
    occupancyVector.reserve(rowData.size());
  }

  // a gap is not filled as a subthreshold cluster when the row that closes it sits within the outermost min(nRows/2, mSubThreshEdgeRows) rows
  const int edgeRowCut = std::min<int>(static_cast<int>(rowData.size()) / 2, mSubThreshEdgeRows);

  for (size_t iRowData = 0; iRowData < rowData.size(); ++iRowData) {
    const auto& row = rowData[iRowData];

    // one effective sample per row for this settings entry, per dEdxSettings::sameRowClusterMethod; pointers into
    // row.fragments/row.mergedFragment (both owned by rowData, alive for the whole calculatedEdxFromRowData() call)
    std::vector<const RowFragment*> samples;

    const bool doMerge = row.mergeable && row.fragments.size() > 1 && sameRowClusterMethod != 0;
    if (!doMerge) {
      samples.reserve(row.fragments.size());
      for (const auto& frag : row.fragments) {
        samples.push_back(&frag);
      }
    } else if (sameRowClusterMethod == 2) {
      samples.push_back(&pickDominantFragment(row.fragments));
    } else { // sameRowClusterMethod == 1: use the sum-merged sample gatherRowClusterDataForRow() already computed once for this row
      samples.push_back(&row.mergedFragment);
    }

    // ExcludeSamePadRowCl: true whenever the row-group HAD >1 raw fragment, regardless of whether this settings entry actually merged them
    const bool isCombined = row.fragments.size() > 1;

    // find missing clusters
    const int missingClusters = row.missingClusters;
    if ((missingClusters > 0) && (missingClusters <= mMaxMissingCl) && !row.missingClusterGapDeadOrEdge && (static_cast<int>(iRowData) >= edgeRowCut)) {
      if ((settings.clusterMask & ClusterFlags::ExcludeSectorBoundaries) == ClusterFlags::ExcludeSectorBoundaries) {
        if (row.sameSectorAsPrevRow) {
          if (row.stack == GEMstack::IROCgem) {
            nClsSubThreshROC[0] += missingClusters;
            nClsROC[0] += missingClusters;
          } else if (row.stack == GEMstack::OROC1gem) {
            nClsSubThreshROC[1] += missingClusters;
            nClsROC[1] += missingClusters;
          } else if (row.stack == GEMstack::OROC2gem) {
            nClsSubThreshROC[2] += missingClusters;
            nClsROC[2] += missingClusters;
          } else if (row.stack == GEMstack::OROC3gem) {
            nClsSubThreshROC[3] += missingClusters;
            nClsROC[3] += missingClusters;
          }
        }
      } else {
        if (row.stack == GEMstack::IROCgem) {
          nClsSubThreshROC[0] += missingClusters;
          nClsROC[0] += missingClusters;
        } else if (row.stack == GEMstack::OROC1gem) {
          nClsSubThreshROC[1] += missingClusters;
          nClsROC[1] += missingClusters;
        } else if (row.stack == GEMstack::OROC2gem) {
          nClsSubThreshROC[2] += missingClusters;
          nClsROC[2] += missingClusters;
        } else if (row.stack == GEMstack::OROC3gem) {
          nClsSubThreshROC[3] += missingClusters;
          nClsROC[3] += missingClusters;
        }
      }
    };

    for (const auto& sample : samples) {
      if (mDebug) {
        occupancyVector.emplace_back(sample->occupancy);
      }

      // get cluster values
      float chargeTot = sample->cl.getQtot();
      float chargeMax = sample->cl.getQmax();

      // corrections
      float effectiveLength = 1.0f;
      float effectiveLengthTot = 1.0f;
      float effectiveLengthMax = 1.0f;
      float gain = 1.0f;
      float gainResidual = 1.0f;
      float corrTot = 1.0f;
      float corrMax = 1.0f;
      float scCorr = 1.0f;

      int excludeCl = 0; // works as a bit mask
      const uint8_t flagsCl = sample->cl.getFlags();
      if (((settings.clusterMask & ClusterFlags::ExcludeSingleCl) == ClusterFlags::ExcludeSingleCl) && ((flagsCl & ClusterNative::flagSingle) == ClusterNative::flagSingle)) {
        excludeCl += 0b001; // 1 for single cluster
      }
      if (((settings.clusterMask & ClusterFlags::ExcludeSplitPadCl) == ClusterFlags::ExcludeSplitPadCl) && ((flagsCl & ClusterNative::flagSplitPad) == ClusterNative::flagSplitPad)) {
        excludeCl += 0b010; // 2 for split pad cluster
      }
      if (((settings.clusterMask & ClusterFlags::ExcludeSplitTimeCl) == ClusterFlags::ExcludeSplitTimeCl) && ((flagsCl & ClusterNative::flagSplitTime) == ClusterNative::flagSplitTime)) {
        excludeCl += 0b0100; // 4 for split time cluster
      }
      if (((settings.clusterMask & ClusterFlags::ExcludeSplitCl) == ClusterFlags::ExcludeSplitCl) && (((flagsCl & ClusterNative::flagSplitPad) == ClusterNative::flagSplitPad) || ((flagsCl & ClusterNative::flagSplitTime) == ClusterNative::flagSplitTime))) {
        excludeCl += 0b01000; // 8 for split cluster
      }
      if (((settings.clusterMask & ClusterFlags::ExcludeEdgeCl) == ClusterFlags::ExcludeEdgeCl) && ((flagsCl & ClusterNative::flagEdge) == ClusterNative::flagEdge)) {
        excludeCl += 0b010000; // 16 for edge cluster
      }
      if (((settings.clusterMask & ClusterFlags::ExcludeSharedCl) == ClusterFlags::ExcludeSharedCl) && sample->isShared) {
        excludeCl += 0b0100000; // 32 for shared cluster
      }
      if (((settings.clusterMask & ClusterFlags::ExcludeSamePadRowCl) == ClusterFlags::ExcludeSamePadRowCl) && isCombined) {
        excludeCl += 0b01000000; // 64 for combined cluster
      }
      if ((settings.stackBoundaryMethod == 1 || settings.stackBoundaryMethod == 2) && isInStackBoundaries(row.stackNumber, row.rowIndex, settings.stackBoundaryMethod)) {
        excludeCl += 0b010000000; // 128 for stack boundary cluster
      }
      if (sample->isDeadRegion) {
        excludeCl += 0b0100000000; // 256 for dead region
      }
      if (row.propagationFailed) {
        excludeCl += 0b01000000000; // 512 for failure of track propagation or refit
      }

      // get effective length
      if ((settings.correctionMask & CorrectionFlags::TopologySimple) == CorrectionFlags::TopologySimple) {
        effectiveLength = getTrackTopologyCorrection(row.trackSnapshot, row.region);
        chargeTot /= effectiveLength;
        chargeMax /= effectiveLength;
      };

      const bool gainFullApplied = (settings.correctionMask & CorrectionFlags::GainFull) == CorrectionFlags::GainFull;
      float topoChargeTot = chargeTot;
      float topoChargeMax = chargeMax;
      if (gainFullApplied) {
        gain = sample->gain;
        chargeTot /= gain;
        chargeMax /= gain;
      } else {
        topoChargeTot *= sample->gain;
        topoChargeMax *= sample->gain;
      }

      // topology correction
      if ((settings.correctionMask & CorrectionFlags::TopologyPol) == CorrectionFlags::TopologyPol) {
        effectiveLengthTot = getTrackTopologyCorrectionPol(row.trackSnapshot, sample->cl, row.region, topoChargeTot, ChargeType::Tot, sample->threshold);
        effectiveLengthMax = getTrackTopologyCorrectionPol(row.trackSnapshot, sample->cl, row.region, topoChargeMax, ChargeType::Max, sample->threshold);
        chargeTot /= effectiveLengthTot;
        chargeMax /= effectiveLengthMax;
      };

      // residual dE/dx correction on tgl and sector plane
      if ((settings.correctionMask & CorrectionFlags::dEdxResidual) == CorrectionFlags::dEdxResidual) {
        corrTot = mCalibCont.getResidualCorrection(row.stackID, ChargeType::Tot, row.trackSnapshot.getTgl(), row.trackSnapshot.getSnp());
        corrMax = mCalibCont.getResidualCorrection(row.stackID, ChargeType::Max, row.trackSnapshot.getTgl(), row.trackSnapshot.getSnp());
        if (corrTot > 0) {
          chargeTot /= corrTot;
        };
        if (corrMax > 0) {
          chargeMax /= corrMax;
        };
      };

      // residual gain map
      if ((settings.correctionMask & CorrectionFlags::GainResidual) == CorrectionFlags::GainResidual) {
        gainResidual = sample->gainResidual;
        chargeTot /= gainResidual;
        chargeMax /= gainResidual;
      };

      // space-charge dEdx corrections
      const float time = sample->cl.getTime() - trackTime0; // ToDo: get correct time from ITS-TPC track if possible
      if ((settings.correctionMask & CorrectionFlags::dEdxSC) == CorrectionFlags::dEdxSC) {
        scCorr = mSCdEdxCorrection.getCorrection(time, row.sectorIndex, row.rowIndex, sample->pad);
        if (scCorr > 0) {
          chargeTot /= scCorr;
        };
        if (scCorr > 0) {
          chargeMax /= scCorr;
        };
      }

      // for debugging
      if (mDebug) {
        const o2::gpu::GPUTPCGeometry gpuGeom;
        const float localX = gpuGeom.Row2X(row.rowIndex);
        const float localY = gpuGeom.LinearPad2Y(row.sectorIndex, row.rowIndex, sample->cl.getPad());
        const LocalPosition2D l2D{localX, localY};
        const auto g2D = Mapper::LocalToGlobal(l2D, Sector(row.sectorIndex));
        const float globalX = g2D.x();
        const float globalY = g2D.y();

        // slice to the base parametrization (X, alpha, params, covariance) instead of the full TrackTPC, since only the parametrization changes cluster-to-cluster after refit/propagation
        const o2::track::TrackParCov trackParam = row.trackSnapshot;

        (*debugStreamer) << "dEdxDebugCl"
                         << "trackIndex=" << mDebugTrackIndex
                         << "trackParam=" << trackParam
                         << "cl=" << sample->cl
                         << "chargeTot=" << chargeTot
                         << "chargeMax=" << chargeMax
                         << "excludeCl=" << excludeCl
                         << "region=" << row.region
                         << "rowIndex=" << row.rowIndex
                         << "sectorIndex=" << row.sectorIndex
                         << "stack=" << row.stackNumber
                         << "localX=" << localX
                         << "localY=" << localY
                         << "globalX=" << globalX
                         << "globalY=" << globalY
                         << "isShared=" << sample->isShared
                         << "isCombined=" << isCombined
                         << "refitFellBack=" << row.refitFellBack
                         << "topologyCorr=" << effectiveLength
                         << "topologyCorrTot=" << effectiveLengthTot
                         << "topologyCorrMax=" << effectiveLengthMax
                         << "gain=" << gain
                         << "gainResidual=" << gainResidual
                         << "residualCorrTot=" << corrTot
                         << "residualCorrMax=" << corrMax
                         << "scCorr=" << scCorr
                         << "occupancy=" << sample->occupancy
                         << "inputClusterIndices=" << row.inputClusterIndices
                         << "\n";
      };

      if (excludeCl != 0) {
        continue;
      }

      // set the region's min charge, only from clusters actually included in the dEdx calculation,
      // so excluded clusters (dead region, edge, failed propagation, ...) don't bias the virtual charge used for subthreshold filling
      if (chargeTot < minChargeTotROC[row.stackNumber]) {
        minChargeTotROC[row.stackNumber] = chargeTot;
      };

      if (chargeMax < minChargeMaxROC[row.stackNumber]) {
        minChargeMaxROC[row.stackNumber] = chargeMax;
      };

      if (row.stack == GEMstack::IROCgem) {
        chargeTotROC[0].emplace_back(chargeTot);
        chargeMaxROC[0].emplace_back(chargeMax);
        nClsROC[0]++;
      } else if (row.stack == GEMstack::OROC1gem) {
        chargeTotROC[1].emplace_back(chargeTot);
        chargeMaxROC[1].emplace_back(chargeMax);
        nClsROC[1]++;
      } else if (row.stack == GEMstack::OROC2gem) {
        chargeTotROC[2].emplace_back(chargeTot);
        chargeMaxROC[2].emplace_back(chargeMax);
        nClsROC[2]++;
      } else if (row.stack == GEMstack::OROC3gem) {
        chargeTotROC[3].emplace_back(chargeTot);
        chargeMaxROC[3].emplace_back(chargeMax);
        nClsROC[3]++;
      };

      chargeTotROC[4].emplace_back(chargeTot);
      chargeMaxROC[4].emplace_back(chargeMax);
    }
  }

  // fill subthreshold clusters if not excluded
  if (((settings.clusterMask & ClusterFlags::ExcludeSubthresholdCl) == ClusterFlags::None)) {
    float cappedMinChargeTotROC[4], cappedMinChargeMaxROC[4];
    for (int roc = 0; roc < 4; roc++) {
      cappedMinChargeTotROC[roc] = (minChargeTotROC[roc] >= kNoValidCharge) ? minChargeTotROC[roc] : std::min(minChargeTotROC[roc], settings.maxSubthresholdChargeTot);
      cappedMinChargeMaxROC[roc] = (minChargeMaxROC[roc] >= kNoValidCharge) ? minChargeMaxROC[roc] : std::min(minChargeMaxROC[roc], settings.maxSubthresholdChargeMax);
      // a ROC with no valid accepted-cluster charge at all makes fillMissingClusters() below skip it entirely
      // (nothing is pushed into chargeTotROC/chargeMaxROC for it) -- so the gaps counted for this ROC earlier
      // in the row loop must be un-counted here too, or NHits*/NHitsSubThreshold* and the
      // mNSubThresholdFilledPerSettings diagnostic would report fills that never actually happened
      if (minChargeTotROC[roc] >= kNoValidCharge || minChargeMaxROC[roc] >= kNoValidCharge) {
        nClsROC[roc] -= nClsSubThreshROC[roc];
        nClsSubThreshROC[roc] = 0;
      }
    }
    fillMissingClusters(nClsSubThreshROC, cappedMinChargeTotROC, cappedMinChargeMaxROC, settings.subthresholdMethod, chargeTotROC, chargeMaxROC);
    if (mNSubThresholdFilledPerSettings.size() <= settingsIndex) {
      mNSubThresholdFilledPerSettings.resize(settingsIndex + 1, 0);
    }
    mNSubThresholdFilledPerSettings[settingsIndex] += nClsSubThreshROC[0] + nClsSubThreshROC[1] + nClsSubThreshROC[2] + nClsSubThreshROC[3];
  }

  // number of clusters
  output.NHitsSubThresholdIROC = nClsROC[0];
  output.NHitsSubThresholdOROC1 = nClsROC[1];
  output.NHitsSubThresholdOROC2 = nClsROC[2];
  output.NHitsSubThresholdOROC3 = nClsROC[3];

  // the gaps found above are always treated as subthreshold clusters (except a ROC with no valid charge to fill
  // them with at all, un-counted above so it isn't double-reported as both "hit" and "subthreshold hit")
  output.NHitsIROC = nClsROC[0] - nClsSubThreshROC[0];
  output.NHitsOROC1 = nClsROC[1] - nClsSubThreshROC[1];
  output.NHitsOROC2 = nClsROC[2] - nClsSubThreshROC[2];
  output.NHitsOROC3 = nClsROC[3] - nClsSubThreshROC[3];

  // copy corrected cluster charges
  auto chargeTotVector = mDebug ? chargeTotROC[4] : std::vector<float>();
  auto chargeMaxVector = mDebug ? chargeMaxROC[4] : std::vector<float>();

  // calculate dEdx
  output.dEdxTotIROC = getTruncMean(chargeTotROC[0], settings.low, settings.high);
  output.dEdxTotOROC1 = getTruncMean(chargeTotROC[1], settings.low, settings.high);
  output.dEdxTotOROC2 = getTruncMean(chargeTotROC[2], settings.low, settings.high);
  output.dEdxTotOROC3 = getTruncMean(chargeTotROC[3], settings.low, settings.high);
  output.dEdxTotTPC = getTruncMean(chargeTotROC[4], settings.low, settings.high);

  output.dEdxMaxIROC = getTruncMean(chargeMaxROC[0], settings.low, settings.high);
  output.dEdxMaxOROC1 = getTruncMean(chargeMaxROC[1], settings.low, settings.high);
  output.dEdxMaxOROC2 = getTruncMean(chargeMaxROC[2], settings.low, settings.high);
  output.dEdxMaxOROC3 = getTruncMean(chargeMaxROC[3], settings.low, settings.high);
  output.dEdxMaxTPC = getTruncMean(chargeMaxROC[4], settings.low, settings.high);

  // for debugging: one row per track, with the track as it was before refit/propagation touched it, per-cluster rows were already written to the "dEdxDebugCl" tree above (each with its own propagated track parameters) and can be grouped back to this row via trackIndex
  if (mDebug) {
    float minChargeTot = minChargeTotROC[0], minChargeMax = minChargeMaxROC[0];
    for (int roc = 1; roc < 4; roc++) {
      minChargeTot = (minChargeTotROC[roc] < minChargeTot) ? minChargeTotROC[roc] : minChargeTot;
      minChargeMax = (minChargeMaxROC[roc] < minChargeMax) ? minChargeMaxROC[roc] : minChargeMax;
    }
    const MCCompLabel label = mcLabel ? *mcLabel : MCCompLabel{};
    (*debugStreamer) << "dEdxDebugTrack"
                     << "trackIndex=" << mDebugTrackIndex
                     << "track=" << trackOrig
                     << "output=" << output
                     << "averageOcc=" << averageOcc
                     << "nCl=" << rowData.size()
                     << "minChargeTot=" << minChargeTot
                     << "minChargeMax=" << minChargeMax
                     << "chargeTotVector=" << chargeTotVector
                     << "chargeMaxVector=" << chargeMaxVector
                     << "occupancy=" << occupancyVector
                     << "mcLabel=" << label
                     << "\n";
  }
}

void CalculatedEdx::calculatedEdxMultipleSettings(o2::tpc::TrackTPC& track, std::vector<dEdxInfo>& outputs, AverageOccupancy& averageOcc, const std::vector<dEdxSettings>& settingsList, const MCCompLabel* mcLabel)
{
  outputs.clear();
  if (settingsList.empty()) {
    return;
  }

  o2::tpc::TrackTPC trackOrig;
  if (mDebug) {
    trackOrig = track; // pristine track, before refit/propagation mutates it cluster-by-cluster below
  }
  const float trackTime0 = track.getTime0(); // unaffected by refit/propagation, so it is the same for every row and every settings entry

  // gather the per-row cluster/track data once, performing the refit/propagation to each cluster row exactly once; this also fills averageOcc, which does not depend on the dEdx settings and is therefore shared by every settings entry
  std::vector<RowClusterData> rowData;
  gatherRowClusterData(track, rowData, averageOcc);

  // evaluate each settings entry against the shared row data
  outputs.resize(settingsList.size());
  for (size_t i = 0; i < settingsList.size(); ++i) {
    calculatedEdxFromRowData(rowData, settingsList[i], i, trackTime0, trackOrig, averageOcc, outputs[i], mcLabel);
  }
}

void CalculatedEdx::calculatedEdx(o2::tpc::TrackTPC& track, const std::vector<o2::tpc::ClusterNative>& clusters, const ClInfoVec& clusterInfos, dEdxInfo& output, AverageOccupancy& averageOcc, float low, float high, CorrectionFlags correctionMask, ClusterFlags clusterMask, int subthresholdMethod, int stackBoundaryMethod, const char* debugRootFile, float maxSubthresholdChargeTot, float maxSubthresholdChargeMax, int sameRowClusterMethod)
{
  dEdxSettings settings;
  settings.low = low;
  settings.high = high;
  settings.correctionMask = correctionMask;
  settings.clusterMask = clusterMask;
  settings.subthresholdMethod = subthresholdMethod;
  settings.stackBoundaryMethod = stackBoundaryMethod;
  settings.debugRootFile = debugRootFile;
  settings.maxSubthresholdChargeTot = maxSubthresholdChargeTot;
  settings.maxSubthresholdChargeMax = maxSubthresholdChargeMax;
  settings.sameRowClusterMethod = sameRowClusterMethod;

  o2::tpc::TrackTPC trackOrig;
  if (mDebug) {
    trackOrig = track; // pristine track, before refit/propagation mutates it cluster-by-cluster below
  }
  const float trackTime0 = track.getTime0();

  std::vector<RowClusterData> rowData;
  gatherRowClusterData(track, clusters, clusterInfos, rowData, averageOcc);

  calculatedEdxFromRowData(rowData, settings, 0, trackTime0, trackOrig, averageOcc, output);
}

void CalculatedEdx::calculatedEdxMultipleSettings(o2::tpc::TrackTPC& track, const std::vector<o2::tpc::ClusterNative>& clusters, const ClInfoVec& clusterInfos, std::vector<dEdxInfo>& outputs, AverageOccupancy& averageOcc, const std::vector<dEdxSettings>& settingsList, const MCCompLabel* mcLabel)
{
  outputs.clear();
  if (settingsList.empty()) {
    return;
  }

  o2::tpc::TrackTPC trackOrig;
  if (mDebug) {
    trackOrig = track; // pristine track, before refit/propagation mutates it cluster-by-cluster below
  }
  const float trackTime0 = track.getTime0(); // unaffected by refit/propagation, so it is the same for every row and every settings entry

  // gather the per-row cluster/track data once, performing the refit/propagation to each cluster row exactly once; this also fills averageOcc, which does not depend on the dEdx settings and is therefore shared by every settings entry
  std::vector<RowClusterData> rowData;
  gatherRowClusterData(track, clusters, clusterInfos, rowData, averageOcc);

  // evaluate each settings entry against the shared row data
  outputs.resize(settingsList.size());
  for (size_t i = 0; i < settingsList.size(); ++i) {
    calculatedEdxFromRowData(rowData, settingsList[i], i, trackTime0, trackOrig, averageOcc, outputs[i], mcLabel);
  }
}

void CalculatedEdx::calculatedEdx(o2::tpc::TrackTPC& track, dEdxInfo& output, AverageOccupancy& averageOcc, float low, float high, CorrectionFlags correctionMask, ClusterFlags clusterMask, int subthresholdMethod, int stackBoundaryMethod, const char* debugRootFile, float maxSubthresholdChargeTot, float maxSubthresholdChargeMax, int sameRowClusterMethod)
{
  dEdxSettings settings;
  settings.low = low;
  settings.high = high;
  settings.correctionMask = correctionMask;
  settings.clusterMask = clusterMask;
  settings.subthresholdMethod = subthresholdMethod;
  settings.stackBoundaryMethod = stackBoundaryMethod;
  settings.debugRootFile = debugRootFile;
  settings.maxSubthresholdChargeTot = maxSubthresholdChargeTot;
  settings.maxSubthresholdChargeMax = maxSubthresholdChargeMax;
  settings.sameRowClusterMethod = sameRowClusterMethod;

  o2::tpc::TrackTPC trackOrig;
  if (mDebug) {
    trackOrig = track; // pristine track, before refit/propagation mutates it cluster-by-cluster below
  }
  const float trackTime0 = track.getTime0();

  std::vector<RowClusterData> rowData;
  gatherRowClusterData(track, rowData, averageOcc);

  calculatedEdxFromRowData(rowData, settings, 0, trackTime0, trackOrig, averageOcc, output);
}

float CalculatedEdx::getTruncMean(std::vector<float>& charge, float low, float high) const
{
  // sort the charge vector
  std::sort(charge.begin(), charge.end());

  // calculate truncated mean
  int nCl = 0;
  float sum = 0;
  size_t firstCl = charge.size() * low;
  size_t lastCl = charge.size() * high;

  for (size_t iCl = firstCl; iCl < lastCl; ++iCl) {
    sum += charge[iCl];
    ++nCl;
  }

  if (nCl > 0) {
    sum /= nCl;
  }
  // if nCl == 0 (charge was empty, or too few entries for low/high to select any index), sum stays 0
  return sum;
}

float CalculatedEdx::getTrackTopologyCorrection(const o2::tpc::TrackTPC& track, const unsigned int region) const
{
  const float padLength = Mapper::instance().getPadRegionInfo(region).getPadHeight();
  const float snp = track.getSnp();
  const float tgl = track.getTgl();
  const float snp2 = snp * snp;
  const float tgl2 = tgl * tgl;
  // calculate the trace length of the track over the pad
  const float effectiveLength = padLength * std::sqrt((1 + tgl2) / (1 - snp2));
  return effectiveLength;
}

float CalculatedEdx::getTrackTopologyCorrectionPol(const o2::tpc::TrackTPC& track, const o2::tpc::ClusterNative& cl, const unsigned int region, const float charge, ChargeType chargeType, const float threshold) const
{
  const float snp = std::abs(track.getSnp());
  const float tgl = track.getTgl();
  constexpr float maxSnp2 = 0.99f;
  float snp2 = snp * snp;
  if (snp2 > maxSnp2) {
    snp2 = maxSnp2;
  }
  const float tgl2 = tgl * tgl;
  const float sec2 = 1.f / (1.f - snp2);
  const float tanTheta = std::sqrt(tgl2 * sec2);

  const float z = std::abs(track.getParam(1));
  const float padTmp = cl.getPad();
  const float absRelPad = std::abs(padTmp - int(padTmp + 0.5f));
  const float relTime = cl.getTime() - int(cl.getTime() + 0.5f);

  const float effectiveLength = mCalibCont.getTopologyCorrection(region, chargeType, tanTheta, snp, z, absRelPad, relTime, threshold, charge);
  return effectiveLength;
}

unsigned int CalculatedEdx::getOccupancy(float clTime) const
{
  // occupancy is only meaningful when the refit method is used, since mTPCRefitterOccMap is only filled by setRefit()
  const int nTimeBinsPerOccupBin = 16;
  const int iBinOcc = clTime / nTimeBinsPerOccupBin + 2;
  if (!mRefit || iBinOcc < 0 || static_cast<size_t>(iBinOcc) >= mTPCRefitterOccMap.size()) {
    return -1;
  }
  return mTPCRefitterOccMap[iBinOcc];
}

bool CalculatedEdx::isInStackBoundaries(int stackNumber, unsigned char rowIndex, int stackBoundaryMethod)
{
  // retrieve boundaries for the given stack
  const auto& boundaries = mStackBoundaries[stackNumber];
  // check direct match for method 1 or 2
  for (unsigned char boundary : boundaries) {
    if (rowIndex == boundary) {
      return true;
    }
  }
  // additional checks for method 2
  if (stackBoundaryMethod == 2) {
    if (rowIndex == boundaries[0] + 1 || rowIndex == boundaries[1] - 1) {
      return true;
    }
  }
  return false;
}

void CalculatedEdx::loadCalibsFromCCDB(long runNumberOrTimeStamp, const bool isMC, const bool loadSCCorrMap, const bool loadSCCorrMapForRefit, const bool loadVDriftForRefit)
{
  // setup CCDB manager
  auto& cm = o2::ccdb::BasicCCDBManager::instance();
  cm.setURL("http://alice-ccdb.cern.ch/");

  auto tRun = runNumberOrTimeStamp;
  if (runNumberOrTimeStamp < 10000000) {
    auto runDuration = cm.getRunDuration(runNumberOrTimeStamp);
    tRun = runDuration.first + (runDuration.second - runDuration.first) / 2; // time stamp for the middle of the run duration
  }
  LOGP(info, "Timestamp: {}", tRun);
  cm.setTimestamp(tRun);

  // set the track topology correction
  o2::tpc::CalibdEdxTrackTopologyPolContainer* calibTrackTopologyContainer = cm.getForTimeStamp<o2::tpc::CalibdEdxTrackTopologyPolContainer>(o2::tpc::CDBTypeMap.at(o2::tpc::CDBType::CalTopologyGain), tRun);
  o2::tpc::CalibdEdxTrackTopologyPol calibTrackTopology;
  calibTrackTopology.setFromContainer(*calibTrackTopologyContainer);
  mCalibCont.setPolTopologyCorrection(calibTrackTopology);

  // set the gain map
  o2::tpc::CalDet<float>* gainMap = cm.getForTimeStamp<o2::tpc::CalDet<float>>(o2::tpc::CDBTypeMap.at(o2::tpc::CDBType::CalPadGainFull), tRun);
  const o2::tpc::CalDet<float> gainMapResidual = (*cm.getForTimeStamp<std::unordered_map<std::string, o2::tpc::CalDet<float>>>(o2::tpc::CDBTypeMap.at(o2::tpc::CDBType::CalPadGainResidual), tRun))["GainMap"];

  const float minGain = 0;
  const float maxGain = 2;
  mCalibCont.setGainMap(*gainMap, minGain, maxGain);
  mCalibCont.setGainMapResidual(gainMapResidual);

  // set the residual dEdx correction
  o2::tpc::CalibdEdxCorrection* residualObj = isMC ? cm.getForTimeStamp<o2::tpc::CalibdEdxCorrection>(o2::tpc::CDBTypeMap.at(o2::tpc::CDBType::CalTimeGainMC), tRun) : cm.getForTimeStamp<o2::tpc::CalibdEdxCorrection>(o2::tpc::CDBTypeMap.at(o2::tpc::CDBType::CalTimeGain), tRun);

  const auto* residualCorr = static_cast<o2::tpc::CalibdEdxCorrection*>(residualObj);
  mCalibCont.setResidualCorrection(*residualCorr);

  // set the zero supression threshold map
  std::unordered_map<std::string, o2::tpc::CalDet<float>>* zeroSupressionThresholdMap = cm.getForTimeStamp<std::unordered_map<std::string, o2::tpc::CalDet<float>>>(o2::tpc::CDBTypeMap.at(o2::tpc::CDBType::ConfigFEEPad), tRun);
  mCalibCont.setZeroSupresssionThreshold(zeroSupressionThresholdMap->at("ThresholdMap"));

  // set the magnetic field
  auto magField = cm.get<o2::parameters::GRPMagField>("GLO/Config/GRPMagField");
  o2::base::Propagator::initFieldFromGRP(magField);
  float bz = GPUO2InterfaceUtils::getNominalGPUBz(*magField);
  LOGP(info, "Magnetic field: {}", bz);
  setFieldNominalGPUBz(bz);

  // set the propagator
  auto propagator = o2::base::Propagator::Instance();
  const o2::base::MatLayerCylSet* matLut = o2::base::MatLayerCylSet::rectifyPtrFromFile(cm.get<o2::base::MatLayerCylSet>("GLO/Param/MatLUT"));
  propagator->setMatLUT(matLut);

  // load the space-charge correction maps
  if (loadSCCorrMap || loadSCCorrMapForRefit) {
    auto avgMap = isMC ? cm.getForTimeStamp<o2::gpu::TPCFastTransform>(o2::tpc::CDBTypeMap.at(o2::tpc::CDBType::CalCorrMapMC), tRun) : cm.getForTimeStamp<o2::gpu::TPCFastTransform>(o2::tpc::CDBTypeMap.at(o2::tpc::CDBType::CalCorrMap), tRun);
    avgMap->rectifyAfterReadingFromFile();

    if (loadSCCorrMap) {
      auto derMap = isMC ? cm.getForTimeStamp<o2::gpu::TPCFastTransform>(o2::tpc::CDBTypeMap.at(o2::tpc::CDBType::CalCorrDerivMapMC), tRun) : cm.getForTimeStamp<o2::gpu::TPCFastTransform>(o2::tpc::CDBTypeMap.at(o2::tpc::CDBType::CalCorrDerivMap), tRun);
      derMap->rectifyAfterReadingFromFile();
      mSCdEdxCorrection.setCorrectionMaps(avgMap, derMap);
    }

    if (loadSCCorrMapForRefit) {
      // feed the space-charge-corrected map into the refit transform
      setTPCCorrMap(*avgMap);
      LOGP(info, "refit transform: using CCDB space-charge correction map {}",
           o2::tpc::CDBTypeMap.at(isMC ? o2::tpc::CDBType::CalCorrMapMC : o2::tpc::CDBType::CalCorrMap));
    }
  }

  // apply the calibrated drift velocity + time offset to the refit transform
  if (loadVDriftForRefit) {
    const bool prevFatalWhenNull = cm.getFatalWhenNull();
    cm.setFatalWhenNull(false);
    if (auto* vd = cm.getForTimeStamp<o2::tpc::VDriftCorrFact>(o2::tpc::CDBTypeMap.at(o2::tpc::CDBType::CalVDriftTgl), tRun)) {
      setTPCVDrift(*vd);
      LOGP(info, "refit transform vDrift calib: corrFact={:.5f} refVDrift={:.5f} timeOffset={:.4f}us", vd->corrFact, vd->refVDrift, vd->getTimeOffset());
    } else {
      LOGP(warning, "no TPC/Calib/VDriftTgl at ts {} -- refit transform stays at nominal vDrift/t0", tRun);
    }
    cm.setFatalWhenNull(prevFatalWhenNull);
  }

  // set the dead channel map
  o2::tpc::DeadChannelMapCreator deadCMCreator;
  deadCMCreator.init();
  deadCMCreator.load(tRun);
  const o2::tpc::CalDet<bool>& deadMap = deadCMCreator.getDeadChannelMap();
  mCalibCont.setDeadChannelMap(deadMap);
}

void CalculatedEdx::loadCalibsFromLocalCCDBFolder(const char* localCCDBFolder)
{
  setTrackTopologyCorrectionFromFile(localCCDBFolder, "/TPC/Calib/TopologyGainPiecewise/snapshot.root", "ccdb_object");
  setGainMapFromFile(localCCDBFolder, "/TPC/Calib/PadGainFull/snapshot.root", "ccdb_object");
  setGainMapResidualFromFile(localCCDBFolder, "/TPC/Calib/PadGainResidual/snapshot.root", "ccdb_object");
  setResidualCorrectionFromFile(localCCDBFolder, "/TPC/Calib/TimeGain/snapshot.root", "ccdb_object");
  setZeroSuppressionThresholdFromFile(localCCDBFolder, "/TPC/Config/FEEPad/snapshot.root", "ccdb_object");
  setMagneticFieldFromFile(localCCDBFolder, "/GLO/Config/GRPMagField/snapshot.root", "ccdb_object");
  setPropagatorFromFile(localCCDBFolder, "/GLO/Param/MatLUT/snapshot.root", "ccdb_object");
  setVDriftFromFile(localCCDBFolder, "/TPC/Calib/VDriftTgl/snapshot.root", "ccdb_object"); // optional: skipped if absent
}

void CalculatedEdx::setVDriftFromFile(const char* folder, const char* file, const char* object)
{
  std::unique_ptr<TFile> vdFile(TFile::Open(fmt::format("{}{}", folder, file).data()));
  if (!vdFile || vdFile->IsZombie()) {
    LOGP(warning, "no {}{} -- refit transform stays at nominal vDrift/t0", folder, file);
    return;
  }
  if (auto* vd = (o2::tpc::VDriftCorrFact*)vdFile->Get(object)) {
    setTPCVDrift(*vd);
    LOGP(info, "refit transform vDrift calib from {}: corrFact={:.5f} refVDrift={:.5f} timeOffset={:.4f}us", vdFile->GetName(), vd->corrFact, vd->refVDrift, vd->getTimeOffset());
  }
}

void CalculatedEdx::setTrackTopologyCorrectionFromFile(const char* folder, const char* file, const char* object)
{
  o2::tpc::CalibdEdxTrackTopologyPol calibTrackTopology;
  calibTrackTopology.loadFromFile(fmt::format("{}{}", folder, file).data(), object);
  mCalibCont.setPolTopologyCorrection(calibTrackTopology);
}

void CalculatedEdx::setGainMapFromFile(const char* folder, const char* file, const char* object)
{
  std::unique_ptr<TFile> gainMapFile(TFile::Open(fmt::format("{}{}", folder, file).data()));
  if (!gainMapFile->IsZombie()) {
    LOGP(info, "Using file: {}", gainMapFile->GetName());
    o2::tpc::CalDet<float>* gainMap = (o2::tpc::CalDet<float>*)gainMapFile->Get(object);
    mCalibCont.setGainMap(*gainMap, 0., 2.);
  }
}

void CalculatedEdx::setGainMapResidualFromFile(const char* folder, const char* file, const char* object)
{
  std::unique_ptr<TFile> gainMapResidualFile(TFile::Open(fmt::format("{}{}", folder, file).data()));
  if (!gainMapResidualFile->IsZombie()) {
    LOGP(info, "Using file: {}", gainMapResidualFile->GetName());
    std::unordered_map<std::string, o2::tpc::CalDet<float>>* gainMapResidual = (std::unordered_map<std::string, o2::tpc::CalDet<float>>*)gainMapResidualFile->Get(object);
    mCalibCont.setGainMapResidual(gainMapResidual->at("GainMap"));
  }
}

void CalculatedEdx::setResidualCorrectionFromFile(const char* folder, const char* file, const char* object)
{
  std::unique_ptr<TFile> calibdEdxResidualFile(TFile::Open(fmt::format("{}{}", folder, file).data()));
  if (!calibdEdxResidualFile->IsZombie()) {
    LOGP(info, "Using file: {}", calibdEdxResidualFile->GetName());
    o2::tpc::CalibdEdxCorrection* calibdEdxResidual = (o2::tpc::CalibdEdxCorrection*)calibdEdxResidualFile->Get(object);
    mCalibCont.setResidualCorrection(*calibdEdxResidual);
  }
}

void CalculatedEdx::setZeroSuppressionThresholdFromFile(const char* folder, const char* file, const char* object)
{
  std::unique_ptr<TFile> zeroSuppressionFile(TFile::Open(fmt::format("{}{}", folder, file).data()));
  if (!zeroSuppressionFile->IsZombie()) {
    LOGP(info, "Using file: {}", zeroSuppressionFile->GetName());
    std::unordered_map<std::string, o2::tpc::CalDet<float>>* zeroSupressionThresholdMap = (std::unordered_map<std::string, o2::tpc::CalDet<float>>*)zeroSuppressionFile->Get(object);
    mCalibCont.setZeroSupresssionThreshold(zeroSupressionThresholdMap->at("ThresholdMap"));
  }
}

void CalculatedEdx::setMagneticFieldFromFile(const char* folder, const char* file, const char* object)
{
  std::unique_ptr<TFile> magFile(TFile::Open(fmt::format("{}{}", folder, file).data()));
  if (!magFile->IsZombie()) {
    LOGP(info, "Using file: {}", magFile->GetName());
    o2::parameters::GRPMagField* magField = (o2::parameters::GRPMagField*)magFile->Get(object);
    o2::base::Propagator::initFieldFromGRP(magField);
    float bz = GPUO2InterfaceUtils::getNominalGPUBz(*magField);
    LOGP(info, "Magnetic field: {}", bz);
    setFieldNominalGPUBz(bz);
  }
}

void CalculatedEdx::setPropagatorFromFile(const char* folder, const char* file, const char* object)
{
  auto propagator = o2::base::Propagator::Instance();
  std::unique_ptr<TFile> matLutFile(TFile::Open(fmt::format("{}{}", folder, file).data()));
  if (!matLutFile->IsZombie()) {
    LOGP(info, "Using file: {}", matLutFile->GetName());
    o2::base::MatLayerCylSet* matLut = o2::base::MatLayerCylSet::rectifyPtrFromFile((o2::base::MatLayerCylSet*)matLutFile->Get(object));
    propagator->setMatLUT(matLut);
  }
}