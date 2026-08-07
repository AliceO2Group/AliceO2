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

using namespace o2::tpc;

CalculatedEdx::CalculatedEdx()
{
  gpu::aligned_unique_buffer_ptr<gpu::TPCFastTransformPOD> buffer;
  gpu::TPCFastTransformPOD::create(buffer, *TPCFastTransformHelperO2::instance()->create(0));
  mTPCCorrMapBuffer = std::move(buffer);
  mTPCCorrMap = mTPCCorrMapBuffer.get();
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

void CalculatedEdx::handleSameRowClusters(o2::tpc::TrackTPC& track, std::vector<std::pair<unsigned char, unsigned char>>& rowOrder, std::map<std::pair<unsigned char, unsigned char>, std::vector<int>>& clustersByRow, std::map<std::pair<unsigned char, unsigned char>, o2::tpc::ClusterNative>& combinedClustersByRow, std::map<int, std::tuple<unsigned char, unsigned char, unsigned int>>& clusterReferencesByIndex)
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

  // combine clusters in the same (sector, row) and store the result
  for (const auto& [rowKey, clusterIndices] : clustersByRow) {
    if (clusterIndices.size() > 1) { // only combine if there are multiple clusters in the same row

      // initialize variables for the combined cluster properties
      float weightedPadSum = 0;
      float weightedTimeSum = 0;
      float totalCharge = 0;
      uint16_t maxCharge = 0;

      // use the first cluster as a template for other fields
      const o2::tpc::ClusterNative& firstCluster = track.getCluster(*mTPCTrackClIdxVecInput, clusterIndices[0], *mClusterIndex);
      o2::tpc::ClusterNative combinedCluster = firstCluster;

      // iterate over all the clusters in the current row to combine their properties
      for (int clusterIdx : clusterIndices) {
        const o2::tpc::ClusterNative& cl = track.getCluster(*mTPCTrackClIdxVecInput, clusterIdx, *mClusterIndex);

        float clPad = cl.getPad();
        float clTime = cl.getTime();
        uint16_t clqTot = cl.getQtot();
        uint16_t clqMax = cl.qMax;

        // calculate weighted sums for pad and time
        weightedPadSum += clPad * clqTot;
        weightedTimeSum += clTime * clqTot;
        totalCharge += clqTot;
        maxCharge = std::max(maxCharge, clqMax);
      }

      // finalize the combined cluster properties
      if (totalCharge > o2::tpc::ClusterNative::maxRegularQtot) {
        combinedCluster.setSaturatedQtot(static_cast<uint32_t>(totalCharge));
      } else {
        combinedCluster.qTotPacked = static_cast<uint16_t>(totalCharge);
      }
      combinedCluster.qMax = maxCharge;
      combinedCluster.padPacked = static_cast<uint16_t>(weightedPadSum / totalCharge * o2::tpc::ClusterNative::scalePadPacked);
      combinedCluster.timeFlagsPacked = (static_cast<uint32_t>(weightedTimeSum / totalCharge * o2::tpc::ClusterNative::scaleTimePacked) & 0xFFFFFF) | (firstCluster.timeFlagsPacked & 0xFF000000);

      // store the combined cluster in the result map for the (sector, row)
      combinedClustersByRow[rowKey] = combinedCluster;
    }
  }
}

void CalculatedEdx::gatherRowClusterData(o2::tpc::TrackTPC& track, std::vector<RowClusterData>& rowData, AverageOccupancy& averageOcc)
{
  rowData.clear();

  // handle same (sector, row) clusters
  std::vector<std::pair<unsigned char, unsigned char>> rowOrder;
  std::map<std::pair<unsigned char, unsigned char>, std::vector<int>> clustersByRow;
  std::map<std::pair<unsigned char, unsigned char>, o2::tpc::ClusterNative> combinedClustersByRow;
  std::map<int, std::tuple<unsigned char, unsigned char, unsigned int>> clusterReferencesByIndex;

  handleSameRowClusters(track, rowOrder, clustersByRow, combinedClustersByRow, clusterReferencesByIndex);

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
    int clusterIdx = clusterIndices[0];
    const o2::tpc::ClusterNative& clConst = track.getCluster(*mTPCTrackClIdxVecInput, clusterIdx, *mClusterIndex);
    const auto& [sectorIndex, rowIndexRef, clusterIndexNumb] = clusterReferencesByIndex[clusterIdx];
    bool isCombined = false;

    o2::tpc::ClusterNative cl = clConst;
    if (clusterIndices.size() > 1) {
      cl = combinedClustersByRow[rowKey];
      isCombined = true;
    }

    RowClusterData row;
    row.cl = cl;
    row.clPad = cl.getPad();
    row.clTime = cl.getTime();
    row.chargeTot = cl.getQtot();
    row.chargeMax = cl.getQmax();
    row.sectorIndex = sectorIndex;
    row.rowIndex = rowIndex;
    row.isCombined = isCombined;

    const unsigned int occupancy = getOccupancy(row.clTime);
    row.occupancy = occupancy;

    // check if the cluster is shared
    const unsigned int absoluteIndex = mClusterIndex->clusterOffset[sectorIndex][rowIndex] + clusterIndexNumb;
    row.isShared = mRefit ? (mTPCRefitterShMap[absoluteIndex] & o2::gpu::GPUTPCGMMergedTrackHit::flagShared) : 0;

    // get region, pad, stack and stack ID
    const int region = Mapper::REGION[rowIndex];
    const unsigned char pad = std::clamp(static_cast<unsigned int>(row.clPad + 0.5f), static_cast<unsigned int>(0), Mapper::PADSPERROW[region][Mapper::getLocalRowFromGlobalRow(rowIndex)] - 1); // the left side of the pad is defined at e.g. 3.5 and the right side at 4.5
    const CRU cru(Sector(sectorIndex), region);
    const auto stack = cru.gemStack();
    StackID stackID{sectorIndex, stack};
    const int stackNumber = static_cast<int>(stack);

    row.region = region;
    row.pad = pad;
    row.stack = stack;
    row.stackID = stackID;
    row.stackNumber = stackNumber;

    if (stack == GEMstack::IROCgem) {
      occupancyROC[0].emplace_back(occupancy);
    } else if (stack == GEMstack::OROC1gem) {
      occupancyROC[1].emplace_back(occupancy);
    } else if (stack == GEMstack::OROC2gem) {
      occupancyROC[2].emplace_back(occupancy);
    } else if (stack == GEMstack::OROC3gem) {
      occupancyROC[3].emplace_back(occupancy);
    }

    row.isDeadRegion = mCalibCont.isDead(static_cast<unsigned int>(sectorIndex), static_cast<gpu::tpccf::Row>(rowIndex), static_cast<gpu::tpccf::Pad>(pad));

    // get the x position of the track
    const float xPosition = Mapper::instance().getPadCentre(PadPos(rowIndex, 0)).X();
    bool check = true;
    if (mRefit) {
      // refit this track
      mRefit->setTrackReferenceX(xPosition);
      check = (mRefit->RefitTrackAsGPU(track, false, true) < 0) ? false : true;
    } else if (mPropagateTrack) {
      // propagate this track to the plane X=xk (cm) in the field "b" (kG); snapshot the fit state first and roll it back before each fallback attempt below. A failure here must not leave the track frozen at this row's stale X
      // so retry the same target without material corrections and if that also fails, fall back to the simple analytic parameter-only propagation used by mPropagateParams only if all three fail, give up and roll back, marking the row as propagationFailed below
      const o2::track::TrackParCov trackBackup = track;
      check = track.rotate(o2::math_utils::detail::sector2Angle<float>(sectorIndex));
      if (check) {
        check = o2::base::Propagator::Instance()->PropagateToXBxByBz(track, xPosition, 0.999f, 0.5f, o2::base::Propagator::MatCorrType::USEMatCorrLUT);
      }
      if (!check) {
        static_cast<o2::track::TrackParCov&>(track) = trackBackup;
        check = track.rotate(o2::math_utils::detail::sector2Angle<float>(sectorIndex));
        if (check) {
          check = o2::base::Propagator::Instance()->PropagateToXBxByBz(track, xPosition, 0.999f, 0.5f, o2::base::Propagator::MatCorrType::USEMatCorrNONE);
        }
      }
      if (!check) {
        static_cast<o2::track::TrackParCov&>(track) = trackBackup;
        check = track.rotateParam(o2::math_utils::detail::sector2Angle<float>(sectorIndex));
        if (check) {
          check = track.propagateParamTo(xPosition, mFieldNominalGPUBz);
        }
      }
      if (!check) {
        static_cast<o2::track::TrackParCov&>(track) = trackBackup;
      }
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
    ++mNRowsProcessed;
    if (row.propagationFailed) {
      ++mNPropagationFailed;
    }

    // snapshot of the track state after refit/propagation to this row; reused by calculatedEdxFromRowData() for every settings entry
    row.trackSnapshot = track;

    // get threshold and gain
    row.threshold = mCalibCont.getZeroSupressionThreshold(sectorIndex, rowIndex, pad);
    row.gain = mCalibCont.getGain(sectorIndex, rowIndex, pad);
    row.gainResidual = mCalibCont.getResidualGain(sectorIndex, rowIndex, pad);

    // number of rows skipped since the previous row in rowOrder
    row.missingClusters = rowIndex - rowIndexOld - 1;
    row.sameSectorAsPrevRow = (sectorIndexOld == sectorIndex);

    // veto the gap as a subthreshold candidate if any of its missing row(s) would land on a dead channel or off the padrow edge
    row.missingClusterGapDeadOrEdge = false;
    if (row.missingClusters > 0 && row.missingClusters <= mMaxMissingCl && row.sameSectorAsPrevRow) {
      const o2::gpu::GPUTPCGeometry gpuGeom;
      const RowClusterData& prevRow = rowData.back();
      const float yPrev = gpuGeom.LinearPad2Y(sectorIndex, prevRow.rowIndex, prevRow.clPad);
      const float yCur = gpuGeom.LinearPad2Y(sectorIndex, rowIndex, row.clPad);
      for (int k = 1; k <= row.missingClusters; ++k) {
        const unsigned char missingRow = prevRow.rowIndex + k;
        const float frac = static_cast<float>(k) / (row.missingClusters + 1);
        const float missingPad = gpuGeom.LinearY2Pad(sectorIndex, missingRow, yPrev + (yCur - yPrev) * frac);
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

    rowIndexOld = rowIndex;
    sectorIndexOld = sectorIndex;

    rowData.emplace_back(std::move(row));
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

void CalculatedEdx::calculatedEdxFromRowData(const std::vector<RowClusterData>& rowData, const dEdxSettings& settings, size_t settingsIndex, float trackTime0, const o2::tpc::TrackTPC& trackOrig, const AverageOccupancy& averageOcc, dEdxInfo& output)
{
  // NHits and NHitsSubthreshold values per region
  int nClsROC[4] = {0, 0, 0, 0};
  int nClsSubThreshROC[4] = {0, 0, 0, 0};

  // corrected qTot and qMax values per region
  const int nType = 5;
  std::array<std::vector<float>, nType> chargeTotROC;
  std::array<std::vector<float>, nType> chargeMaxROC;
  for (int i = 0; i < nType; ++i) {
    chargeTotROC[i].reserve(Mapper::PADROWS);
    chargeMaxROC[i].reserve(Mapper::PADROWS);
  }

  // per-region (IROC, OROC1, OROC2, OROC3) running minimum charge among accepted clusters, used as the virtual charge for that region's subthreshold clusters below
  float minChargeTotROC[4] = {100000.f, 100000.f, 100000.f, 100000.f};
  float minChargeMaxROC[4] = {100000.f, 100000.f, 100000.f, 100000.f};

  o2::utils::TreeStreamRedirector* debugStreamer = nullptr;
  std::vector<unsigned int> occupancyVector;
  if (mDebug) {
    setStreamer(settings.debugRootFile.c_str());
    debugStreamer = mStreamers.at(settings.debugRootFile).get();
    ++mDebugTrackIndex;
    occupancyVector.reserve(rowData.size());
  }

  for (const auto& row : rowData) {
    if (mDebug) {
      occupancyVector.emplace_back(row.occupancy);
    }

    // get cluster values
    float chargeTot = row.chargeTot;
    float chargeMax = row.chargeMax;

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
    const uint8_t flagsCl = row.cl.getFlags();
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
    if (((settings.clusterMask & ClusterFlags::ExcludeSharedCl) == ClusterFlags::ExcludeSharedCl) && row.isShared) {
      excludeCl += 0b0100000; // 32 for shared cluster
    }
    if (((settings.clusterMask & ClusterFlags::ExcludeSamePadRowCl) == ClusterFlags::ExcludeSamePadRowCl) && row.isCombined) {
      excludeCl += 0b01000000; // 64 for combined cluster
    }
    if ((settings.stackBoundaryMethod == 1 || settings.stackBoundaryMethod == 2) && isInStackBoundaries(row.stackNumber, row.rowIndex, settings.stackBoundaryMethod)) {
      excludeCl += 0b010000000; // 128 for stack boundary cluster
    }
    if (row.isDeadRegion) {
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
    if ((settings.correctionMask & CorrectionFlags::TopologyPol) == CorrectionFlags::TopologyPol) {
      effectiveLengthTot = getTrackTopologyCorrectionPol(row.trackSnapshot, row.cl, row.region, chargeTot, ChargeType::Tot, row.threshold);
      effectiveLengthMax = getTrackTopologyCorrectionPol(row.trackSnapshot, row.cl, row.region, chargeMax, ChargeType::Max, row.threshold);
      chargeTot /= effectiveLengthTot;
      chargeMax /= effectiveLengthMax;
    };

    // get gain
    if ((settings.correctionMask & CorrectionFlags::GainFull) == CorrectionFlags::GainFull) {
      gain = row.gain;
    };
    if ((settings.correctionMask & CorrectionFlags::GainResidual) == CorrectionFlags::GainResidual) {
      gainResidual = row.gainResidual;
    };
    chargeTot /= gain * gainResidual;
    chargeMax /= gain * gainResidual;

    // get dEdx correction on tgl and sector plane
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

    // space-charge dEdx corrections
    const float time = row.clTime - trackTime0; // ToDo: get correct time from ITS-TPC track if possible
    if ((settings.correctionMask & CorrectionFlags::dEdxSC) == CorrectionFlags::dEdxSC) {
      scCorr = mSCdEdxCorrection.getCorrection(time, row.sectorIndex, row.rowIndex, row.pad);
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
      const float localY = gpuGeom.LinearPad2Y(row.sectorIndex, row.rowIndex, row.clPad);
      const LocalPosition2D l2D{localX, localY};
      const auto g2D = Mapper::LocalToGlobal(l2D, Sector(row.sectorIndex));
      const float globalX = g2D.x();
      const float globalY = g2D.y();

      // slice to the base parametrization (X, alpha, params, covariance) instead of the full TrackTPC, since only the parametrization changes cluster-to-cluster after refit/propagation
      const o2::track::TrackParCov trackParam = row.trackSnapshot;

      (*debugStreamer) << "dEdxDebugCl"
                       << "trackIndex=" << mDebugTrackIndex
                       << "trackParam=" << trackParam
                       << "cl=" << row.cl
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
                       << "isShared=" << row.isShared
                       << "isCombined=" << row.isCombined
                       << "topologyCorr=" << effectiveLength
                       << "topologyCorrTot=" << effectiveLengthTot
                       << "topologyCorrMax=" << effectiveLengthMax
                       << "gain=" << gain
                       << "gainResidual=" << gainResidual
                       << "residualCorrTot=" << corrTot
                       << "residualCorrMax=" << corrMax
                       << "scCorr=" << scCorr
                       << "occupancy=" << row.occupancy
                       << "\n";
    };

    // find missing clusters
    const int missingClusters = row.missingClusters;
    if ((missingClusters > 0) && (missingClusters <= mMaxMissingCl) && !row.missingClusterGapDeadOrEdge) {
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

  // number of clusters
  output.NHitsSubThresholdIROC = nClsROC[0];
  output.NHitsSubThresholdOROC1 = nClsROC[1];
  output.NHitsSubThresholdOROC2 = nClsROC[2];
  output.NHitsSubThresholdOROC3 = nClsROC[3];

  // the gaps found above are always treated as subthreshold clusters
  output.NHitsIROC = nClsROC[0] - nClsSubThreshROC[0];
  output.NHitsOROC1 = nClsROC[1] - nClsSubThreshROC[1];
  output.NHitsOROC2 = nClsROC[2] - nClsSubThreshROC[2];
  output.NHitsOROC3 = nClsROC[3] - nClsSubThreshROC[3];

  // fill subthreshold clusters if not excluded
  if (((settings.clusterMask & ClusterFlags::ExcludeSubthresholdCl) == ClusterFlags::None)) {
    float cappedMinChargeTotROC[4], cappedMinChargeMaxROC[4];
    for (int roc = 0; roc < 4; roc++) {
      cappedMinChargeTotROC[roc] = std::min(minChargeTotROC[roc], settings.maxSubthresholdChargeTot);
      cappedMinChargeMaxROC[roc] = std::min(minChargeMaxROC[roc], settings.maxSubthresholdChargeMax);
    }
    fillMissingClusters(nClsSubThreshROC, cappedMinChargeTotROC, cappedMinChargeMaxROC, settings.subthresholdMethod, chargeTotROC, chargeMaxROC);
    if (mNSubThresholdFilledPerSettings.size() <= settingsIndex) {
      mNSubThresholdFilledPerSettings.resize(settingsIndex + 1, 0);
    }
    mNSubThresholdFilledPerSettings[settingsIndex] += nClsSubThreshROC[0] + nClsSubThreshROC[1] + nClsSubThreshROC[2] + nClsSubThreshROC[3];
  }

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
                     << "\n";
  }
}

void CalculatedEdx::calculatedEdxMultipleSettings(o2::tpc::TrackTPC& track, std::vector<dEdxInfo>& outputs, AverageOccupancy& averageOcc, const std::vector<dEdxSettings>& settingsList)
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
    calculatedEdxFromRowData(rowData, settingsList[i], i, trackTime0, trackOrig, averageOcc, outputs[i]);
  }
}

void CalculatedEdx::calculatedEdx(o2::tpc::TrackTPC& track, dEdxInfo& output, AverageOccupancy& averageOcc, float low, float high, CorrectionFlags correctionMask, ClusterFlags clusterMask, int subthresholdMethod, int stackBoundaryMethod, const char* debugRootFile, float maxSubthresholdChargeTot, float maxSubthresholdChargeMax)
{
  // NHits and NHitsSubthreshold values per region
  int nClsROC[4] = {0, 0, 0, 0};
  int nClsSubThreshROC[4] = {0, 0, 0, 0};

  // corrected qTot and qMax values per region
  const int nType = 5;
  std::array<std::vector<float>, nType> chargeTotROC;
  std::array<std::vector<float>, nType> chargeMaxROC;
  for (int i = 0; i < nType; ++i) {
    chargeTotROC[i].reserve(Mapper::PADROWS);
    chargeMaxROC[i].reserve(Mapper::PADROWS);
  }

  // occupancy vector for a track (all clusters, for debugging) and per-region (for the average occupancy output)
  std::vector<unsigned int> occupancyVector;
  std::array<std::vector<unsigned int>, 4> occupancyROC;

  // for tracking missing clusters
  unsigned char rowIndexOld = 255;
  unsigned char sectorIndexOld = 255;
  float clPadOld = 0.f; // previous row's cluster pad, used by the dead/edge gap veto below
  // per-region running minimum charge
  float minChargeTotROC[4] = {100000.f, 100000.f, 100000.f, 100000.f};
  float minChargeMaxROC[4] = {100000.f, 100000.f, 100000.f, 100000.f};

  // corrections
  float effectiveLength = 1.0f;
  float effectiveLengthTot = 1.0f;
  float effectiveLengthMax = 1.0f;
  float gain = 1.0f;
  float gainResidual = 1.0f;
  float corrTot = 1.0f;
  float corrMax = 1.0f;
  float scCorr = 1.0f;

  // handle same (sector, row) clusters
  std::vector<std::pair<unsigned char, unsigned char>> rowOrder;
  std::map<std::pair<unsigned char, unsigned char>, std::vector<int>> clustersByRow;
  std::map<std::pair<unsigned char, unsigned char>, o2::tpc::ClusterNative> combinedClustersByRow;
  std::map<int, std::tuple<unsigned char, unsigned char, unsigned int>> clusterReferencesByIndex;

  handleSameRowClusters(track, rowOrder, clustersByRow, combinedClustersByRow, clusterReferencesByIndex);

  o2::utils::TreeStreamRedirector* debugStreamer = nullptr;
  o2::tpc::TrackTPC trackOrig;
  if (mDebug) {
    setStreamer(debugRootFile);
    debugStreamer = mStreamers.at(debugRootFile).get();
    ++mDebugTrackIndex;
    trackOrig = track; // pristine track, before refit/propagation mutates it cluster-by-cluster below
  }

  // loop over the clusters in the track's true physical row-traversal order (rowOrder)
  for (const auto& rowKey : rowOrder) {
    const auto& clusterIndices = clustersByRow.at(rowKey);
    const unsigned char rowIndex = rowKey.second;
    int clusterIdx = clusterIndices[0];
    const o2::tpc::ClusterNative& clConst = track.getCluster(*mTPCTrackClIdxVecInput, clusterIdx, *mClusterIndex);
    const auto& [sectorIndex, rowIndexRef, clusterIndexNumb] = clusterReferencesByIndex[clusterIdx];
    bool isCombined = false;

    o2::tpc::ClusterNative cl = clConst;

    if (clusterIndices.size() > 1) {
      cl = combinedClustersByRow[rowKey];
      isCombined = true;
    }

    // get cluster values
    float chargeTot = cl.getQtot();
    float chargeMax = cl.getQmax();
    const float clPad = cl.getPad();
    const float clTime = cl.getTime();
    const uint8_t flagsCl = cl.getFlags();
    unsigned int occupancy = getOccupancy(clTime);
    occupancyVector.emplace_back(occupancy);

    // check if the cluster is shared
    const unsigned int absoluteIndex = mClusterIndex->clusterOffset[sectorIndex][rowIndex] + clusterIndexNumb;
    const bool isShared = mRefit ? (mTPCRefitterShMap[absoluteIndex] & o2::gpu::GPUTPCGMMergedTrackHit::flagShared) : 0;

    // get region, pad, stack and stack ID
    const int region = Mapper::REGION[rowIndex];
    const unsigned char pad = std::clamp(static_cast<unsigned int>(clPad + 0.5f), static_cast<unsigned int>(0), Mapper::PADSPERROW[region][Mapper::getLocalRowFromGlobalRow(rowIndex)] - 1); // the left side of the pad is defined at e.g. 3.5 and the right side at 4.5
    const CRU cru(Sector(sectorIndex), region);
    const auto stack = cru.gemStack();
    StackID stackID{sectorIndex, stack};
    // the stack number for debugging
    const int stackNumber = static_cast<int>(stack);

    if (stack == GEMstack::IROCgem) {
      occupancyROC[0].emplace_back(occupancy);
    } else if (stack == GEMstack::OROC1gem) {
      occupancyROC[1].emplace_back(occupancy);
    } else if (stack == GEMstack::OROC2gem) {
      occupancyROC[2].emplace_back(occupancy);
    } else if (stack == GEMstack::OROC3gem) {
      occupancyROC[3].emplace_back(occupancy);
    }

    int excludeCl = 0; // works as a bit mask
    if (((clusterMask & ClusterFlags::ExcludeSingleCl) == ClusterFlags::ExcludeSingleCl) && ((flagsCl & ClusterNative::flagSingle) == ClusterNative::flagSingle)) {
      excludeCl += 0b001; // 1 for single cluster
    }
    if (((clusterMask & ClusterFlags::ExcludeSplitPadCl) == ClusterFlags::ExcludeSplitPadCl) && ((flagsCl & ClusterNative::flagSplitPad) == ClusterNative::flagSplitPad)) {
      excludeCl += 0b010; // 2 for split pad cluster
    }
    if (((clusterMask & ClusterFlags::ExcludeSplitTimeCl) == ClusterFlags::ExcludeSplitTimeCl) && ((flagsCl & ClusterNative::flagSplitTime) == ClusterNative::flagSplitTime)) {
      excludeCl += 0b0100; // 4 for split time cluster
    }
    if (((clusterMask & ClusterFlags::ExcludeSplitCl) == ClusterFlags::ExcludeSplitCl) && (((flagsCl & ClusterNative::flagSplitPad) == ClusterNative::flagSplitPad) || ((flagsCl & ClusterNative::flagSplitTime) == ClusterNative::flagSplitTime))) {
      excludeCl += 0b01000; // 8 for split cluster
    }
    if (((clusterMask & ClusterFlags::ExcludeEdgeCl) == ClusterFlags::ExcludeEdgeCl) && ((flagsCl & ClusterNative::flagEdge) == ClusterNative::flagEdge)) {
      excludeCl += 0b010000; // 16 for edge cluster
    }
    if (((clusterMask & ClusterFlags::ExcludeSharedCl) == ClusterFlags::ExcludeSharedCl) && isShared) {
      excludeCl += 0b0100000; // 32 for shared cluster
    }
    if (((clusterMask & ClusterFlags::ExcludeSamePadRowCl) == ClusterFlags::ExcludeSamePadRowCl) && isCombined) {
      excludeCl += 0b01000000; // 64 for combined cluster
    }
    if ((stackBoundaryMethod == 1 || stackBoundaryMethod == 2) && isInStackBoundaries(stackNumber, rowIndex, stackBoundaryMethod)) {
      excludeCl += 0b010000000; // 128 for stack boundary cluster
    }
    if (mCalibCont.isDead(static_cast<unsigned int>(sectorIndex), static_cast<gpu::tpccf::Row>(rowIndex), static_cast<gpu::tpccf::Pad>(pad))) {
      excludeCl += 0b0100000000; // 256 for dead region
    }

    // get the x position of the track
    const float xPosition = Mapper::instance().getPadCentre(PadPos(rowIndex, 0)).X();
    bool check = true;
    if (mRefit) {
      // refit this track
      mRefit->setTrackReferenceX(xPosition);
      check = (mRefit->RefitTrackAsGPU(track, false, true) < 0) ? false : true;
    } else if (mPropagateTrack) {
      // propagate this track to the plane X=xk (cm) in the field "b" (kG); snapshot the fit state first and roll it back before each fallback attempt below. A failure here must not leave the track frozen at this row's stale X
      // so retry the same target without material corrections and if that also fails, fall back to the simple analytic parameter-only propagation used by mPropagateParams only if all three fail, give up and roll back, marking the row as propagationFailed below
      const o2::track::TrackParCov trackBackup = track;
      check = track.rotate(o2::math_utils::detail::sector2Angle<float>(sectorIndex));
      if (check) {
        check = o2::base::Propagator::Instance()->PropagateToXBxByBz(track, xPosition, 0.999f, 0.5f, o2::base::Propagator::MatCorrType::USEMatCorrLUT);
      }
      if (!check) {
        static_cast<o2::track::TrackParCov&>(track) = trackBackup;
        check = track.rotate(o2::math_utils::detail::sector2Angle<float>(sectorIndex));
        if (check) {
          check = o2::base::Propagator::Instance()->PropagateToXBxByBz(track, xPosition, 0.999f, 0.5f, o2::base::Propagator::MatCorrType::USEMatCorrNONE);
        }
      }
      if (!check) {
        static_cast<o2::track::TrackParCov&>(track) = trackBackup;
        check = track.rotateParam(o2::math_utils::detail::sector2Angle<float>(sectorIndex));
        if (check) {
          check = track.propagateParamTo(xPosition, mFieldNominalGPUBz);
        }
      }
      if (!check) {
        static_cast<o2::track::TrackParCov&>(track) = trackBackup;
      }
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

    if (!check || std::isnan(track.getParam(1))) {
      excludeCl += 0b01000000000; // 512 for failure of track propagation or refit
    }

    // get threshold
    const float threshold = mCalibCont.getZeroSupressionThreshold(sectorIndex, rowIndex, pad);

    // get effective length
    if ((correctionMask & CorrectionFlags::TopologySimple) == CorrectionFlags::TopologySimple) {
      effectiveLength = getTrackTopologyCorrection(track, region);
      chargeTot /= effectiveLength;
      chargeMax /= effectiveLength;
    };
    if ((correctionMask & CorrectionFlags::TopologyPol) == CorrectionFlags::TopologyPol) {
      effectiveLengthTot = getTrackTopologyCorrectionPol(track, cl, region, chargeTot, ChargeType::Tot, threshold);
      effectiveLengthMax = getTrackTopologyCorrectionPol(track, cl, region, chargeMax, ChargeType::Max, threshold);
      chargeTot /= effectiveLengthTot;
      chargeMax /= effectiveLengthMax;
    };

    // get gain
    if ((correctionMask & CorrectionFlags::GainFull) == CorrectionFlags::GainFull) {
      gain = mCalibCont.getGain(sectorIndex, rowIndex, pad);
    };
    if ((correctionMask & CorrectionFlags::GainResidual) == CorrectionFlags::GainResidual) {
      gainResidual = mCalibCont.getResidualGain(sectorIndex, rowIndex, pad);
    };
    chargeTot /= gain * gainResidual;
    chargeMax /= gain * gainResidual;

    // get dEdx correction on tgl and sector plane
    if ((correctionMask & CorrectionFlags::dEdxResidual) == CorrectionFlags::dEdxResidual) {
      corrTot = mCalibCont.getResidualCorrection(stackID, ChargeType::Tot, track.getTgl(), track.getSnp());
      corrMax = mCalibCont.getResidualCorrection(stackID, ChargeType::Max, track.getTgl(), track.getSnp());
      if (corrTot > 0) {
        chargeTot /= corrTot;
      };
      if (corrMax > 0) {
        chargeMax /= corrMax;
      };
    };

    // space-charge dEdx corrections
    const float time = clTime - track.getTime0(); // ToDo: get correct time from ITS-TPC track if possible
    if ((correctionMask & CorrectionFlags::dEdxSC) == CorrectionFlags::dEdxSC) {
      scCorr = mSCdEdxCorrection.getCorrection(time, sectorIndex, rowIndex, pad);
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
      const float localX = gpuGeom.Row2X(rowIndex);
      const float localY = gpuGeom.LinearPad2Y(sectorIndex, rowIndex, clPad);
      const LocalPosition2D l2D{localX, localY};
      const auto g2D = Mapper::LocalToGlobal(l2D, Sector(sectorIndex));
      const float globalX = g2D.x();
      const float globalY = g2D.y();

      // slice to the base parametrization instead of the full TrackTPC, since only the parametrization changes cluster-to-cluster after refit/propagation
      const o2::track::TrackParCov trackParam = track;

      // one row per cluster, tagged with the running track index so rows can be grouped back to the track's
      // "dEdxDebugTrack" row; trackParam holds the parameters after refit/propagation to this cluster's row,
      // so they differ from cluster to cluster (and from the pristine track stored in "dEdxDebugTrack")
      (*debugStreamer) << "dEdxDebugCl"
                       << "trackIndex=" << mDebugTrackIndex
                       << "trackParam=" << trackParam
                       << "cl=" << cl
                       << "chargeTot=" << chargeTot
                       << "chargeMax=" << chargeMax
                       << "excludeCl=" << excludeCl
                       << "region=" << region
                       << "rowIndex=" << rowIndex
                       << "sectorIndex=" << sectorIndex
                       << "stack=" << stackNumber
                       << "localX=" << localX
                       << "localY=" << localY
                       << "globalX=" << globalX
                       << "globalY=" << globalY
                       << "isShared=" << isShared
                       << "isCombined=" << isCombined
                       << "topologyCorr=" << effectiveLength
                       << "topologyCorrTot=" << effectiveLengthTot
                       << "topologyCorrMax=" << effectiveLengthMax
                       << "gain=" << gain
                       << "gainResidual=" << gainResidual
                       << "residualCorrTot=" << corrTot
                       << "residualCorrMax=" << corrMax
                       << "scCorr=" << scCorr
                       << "occupancy=" << occupancy
                       << "\n";
    };

    // find missing clusters - deliberately evaluated before (independent of) this row's own excludeCl status
    int missingClusters = rowIndex - rowIndexOld - 1;

    // veto the gap as a subthreshold candidate if any of its missing row(s) would land on a dead channel or off the padrow edge
    bool missingClusterGapDeadOrEdge = false;
    if (missingClusters > 0 && missingClusters <= mMaxMissingCl && sectorIndexOld == sectorIndex) {
      const o2::gpu::GPUTPCGeometry gpuGeom;
      const float yPrev = gpuGeom.LinearPad2Y(sectorIndex, rowIndexOld, clPadOld);
      const float yCur = gpuGeom.LinearPad2Y(sectorIndex, rowIndex, clPad);
      for (int k = 1; k <= missingClusters; ++k) {
        const unsigned char missingRow = rowIndexOld + k;
        const float frac = static_cast<float>(k) / (missingClusters + 1);
        const float missingPad = gpuGeom.LinearY2Pad(sectorIndex, missingRow, yPrev + (yCur - yPrev) * frac);
        if (missingPad < 0.f || missingPad >= gpuGeom.NPads(missingRow)) {
          missingClusterGapDeadOrEdge = true;
          break;
        }
        const int missingRegion = Mapper::REGION[missingRow];
        const unsigned char missingPadClamped = std::clamp(static_cast<unsigned int>(missingPad + 0.5f), static_cast<unsigned int>(0), Mapper::PADSPERROW[missingRegion][Mapper::getLocalRowFromGlobalRow(missingRow)] - 1);
        if (mCalibCont.isDead(static_cast<unsigned int>(sectorIndex), static_cast<gpu::tpccf::Row>(missingRow), static_cast<gpu::tpccf::Pad>(missingPadClamped))) {
          missingClusterGapDeadOrEdge = true;
          break;
        }
      }
    }

    if ((missingClusters > 0) && (missingClusters <= mMaxMissingCl) && !missingClusterGapDeadOrEdge) {
      if ((clusterMask & ClusterFlags::ExcludeSectorBoundaries) == ClusterFlags::ExcludeSectorBoundaries) {
        if (sectorIndexOld == sectorIndex) {
          if (stack == GEMstack::IROCgem) {
            nClsSubThreshROC[0] += missingClusters;
            nClsROC[0] += missingClusters;
          } else if (stack == GEMstack::OROC1gem) {
            nClsSubThreshROC[1] += missingClusters;
            nClsROC[1] += missingClusters;
          } else if (stack == GEMstack::OROC2gem) {
            nClsSubThreshROC[2] += missingClusters;
            nClsROC[2] += missingClusters;
          } else if (stack == GEMstack::OROC3gem) {
            nClsSubThreshROC[3] += missingClusters;
            nClsROC[3] += missingClusters;
          }
        }
      } else {
        if (stack == GEMstack::IROCgem) {
          nClsSubThreshROC[0] += missingClusters;
          nClsROC[0] += missingClusters;
        } else if (stack == GEMstack::OROC1gem) {
          nClsSubThreshROC[1] += missingClusters;
          nClsROC[1] += missingClusters;
        } else if (stack == GEMstack::OROC2gem) {
          nClsSubThreshROC[2] += missingClusters;
          nClsROC[2] += missingClusters;
        } else if (stack == GEMstack::OROC3gem) {
          nClsSubThreshROC[3] += missingClusters;
          nClsROC[3] += missingClusters;
        }
      }
    };

    if (excludeCl != 0) {
      // to avoid counting the skipped cluster itself as a real or subthreshold hit
      rowIndexOld = rowIndex;
      sectorIndexOld = sectorIndex;
      clPadOld = clPad;
      continue;
    }

    // set the region's min charge, only from clusters actually included in the dEdx calculation,
    // so excluded clusters (dead region, edge, failed propagation, ...) don't bias the virtual charge used for subthreshold filling
    if (chargeTot < minChargeTotROC[stackNumber]) {
      minChargeTotROC[stackNumber] = chargeTot;
    };

    if (chargeMax < minChargeMaxROC[stackNumber]) {
      minChargeMaxROC[stackNumber] = chargeMax;
    };

    if (stack == GEMstack::IROCgem) {
      chargeTotROC[0].emplace_back(chargeTot);
      chargeMaxROC[0].emplace_back(chargeMax);
      nClsROC[0]++;
    } else if (stack == GEMstack::OROC1gem) {
      chargeTotROC[1].emplace_back(chargeTot);
      chargeMaxROC[1].emplace_back(chargeMax);
      nClsROC[1]++;
    } else if (stack == GEMstack::OROC2gem) {
      chargeTotROC[2].emplace_back(chargeTot);
      chargeMaxROC[2].emplace_back(chargeMax);
      nClsROC[2]++;
    } else if (stack == GEMstack::OROC3gem) {
      chargeTotROC[3].emplace_back(chargeTot);
      chargeMaxROC[3].emplace_back(chargeMax);
      nClsROC[3]++;
    };

    chargeTotROC[4].emplace_back(chargeTot);
    chargeMaxROC[4].emplace_back(chargeMax);

    rowIndexOld = rowIndex;
    sectorIndexOld = sectorIndex;
    clPadOld = clPad;
  }

  // number of clusters
  output.NHitsSubThresholdIROC = nClsROC[0];
  output.NHitsSubThresholdOROC1 = nClsROC[1];
  output.NHitsSubThresholdOROC2 = nClsROC[2];
  output.NHitsSubThresholdOROC3 = nClsROC[3];

  output.NHitsIROC = nClsROC[0] - nClsSubThreshROC[0];
  output.NHitsOROC1 = nClsROC[1] - nClsSubThreshROC[1];
  output.NHitsOROC2 = nClsROC[2] - nClsSubThreshROC[2];
  output.NHitsOROC3 = nClsROC[3] - nClsSubThreshROC[3];

  // fill subthreshold clusters
  if (((clusterMask & ClusterFlags::ExcludeSubthresholdCl) == ClusterFlags::None)) {
    float cappedMinChargeTotROC[4], cappedMinChargeMaxROC[4];
    for (int roc = 0; roc < 4; roc++) {
      cappedMinChargeTotROC[roc] = std::min(minChargeTotROC[roc], maxSubthresholdChargeTot);
      cappedMinChargeMaxROC[roc] = std::min(minChargeMaxROC[roc], maxSubthresholdChargeMax);
    }
    fillMissingClusters(nClsSubThreshROC, cappedMinChargeTotROC, cappedMinChargeMaxROC, subthresholdMethod, chargeTotROC, chargeMaxROC);
  }

  // copy corrected cluster charges
  auto chargeTotVector = mDebug ? chargeTotROC[4] : std::vector<float>();
  auto chargeMaxVector = mDebug ? chargeMaxROC[4] : std::vector<float>();

  // calculate dEdx
  output.dEdxTotIROC = getTruncMean(chargeTotROC[0], low, high);
  output.dEdxTotOROC1 = getTruncMean(chargeTotROC[1], low, high);
  output.dEdxTotOROC2 = getTruncMean(chargeTotROC[2], low, high);
  output.dEdxTotOROC3 = getTruncMean(chargeTotROC[3], low, high);
  output.dEdxTotTPC = getTruncMean(chargeTotROC[4], low, high);

  output.dEdxMaxIROC = getTruncMean(chargeMaxROC[0], low, high);
  output.dEdxMaxOROC1 = getTruncMean(chargeMaxROC[1], low, high);
  output.dEdxMaxOROC2 = getTruncMean(chargeMaxROC[2], low, high);
  output.dEdxMaxOROC3 = getTruncMean(chargeMaxROC[3], low, high);
  output.dEdxMaxTPC = getTruncMean(chargeMaxROC[4], low, high);

  // calculate average cl occupancy for the track per TPC region; skip clusters where getOccupancy() had no data (sentinel -1),
  // otherwise a single such entry would poison the sum via unsigned overflow
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

  // for debugging: one row per track, with the track as it was before refit/propagation touched it,
  // per-cluster rows were already written to the "dEdxDebugCl" tree above (each with its own propagated track parameters) and can be grouped back to this row via trackIndex
  if (mDebug) {
    // minChargeTot/Max are now tracked per region (see minChargeTotROC/minChargeMaxROC above); report the
    // smallest of the four here so the "dEdxDebugTrack" tree keeps its existing single-scalar branches
    float minChargeTot = minChargeTotROC[0], minChargeMax = minChargeMaxROC[0];
    for (int roc = 1; roc < 4; roc++) {
      minChargeTot = (minChargeTotROC[roc] < minChargeTot) ? minChargeTotROC[roc] : minChargeTot;
      minChargeMax = (minChargeMaxROC[roc] < minChargeMax) ? minChargeMaxROC[roc] : minChargeMax;
    }
    (*debugStreamer) << "dEdxDebugTrack"
                     << "trackIndex=" << mDebugTrackIndex
                     << "track=" << trackOrig
                     << "output=" << output
                     << "averageOcc=" << averageOcc
                     << "nCl=" << clustersByRow.size()
                     << "minChargeTot=" << minChargeTot
                     << "minChargeMax=" << minChargeMax
                     << "chargeTotVector=" << chargeTotVector
                     << "chargeMaxVector=" << chargeMaxVector
                     << "occupancy=" << occupancyVector
                     << "\n";
  }
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
  const float snp2 = snp * snp;
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

void CalculatedEdx::loadCalibsFromCCDB(long runNumberOrTimeStamp, const bool isMC, const bool loadSCCorrMap)
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

  // load sc correction maps; skip if not needed
  if (loadSCCorrMap) {
    auto avgMap = isMC ? cm.getForTimeStamp<o2::gpu::TPCFastTransform>(o2::tpc::CDBTypeMap.at(o2::tpc::CDBType::CalCorrMapMC), tRun) : cm.getForTimeStamp<o2::gpu::TPCFastTransform>(o2::tpc::CDBTypeMap.at(o2::tpc::CDBType::CalCorrMap), tRun);
    avgMap->rectifyAfterReadingFromFile();

    auto derMap = isMC ? cm.getForTimeStamp<o2::gpu::TPCFastTransform>(o2::tpc::CDBTypeMap.at(o2::tpc::CDBType::CalCorrDerivMapMC), tRun) : cm.getForTimeStamp<o2::gpu::TPCFastTransform>(o2::tpc::CDBTypeMap.at(o2::tpc::CDBType::CalCorrDerivMap), tRun);
    derMap->rectifyAfterReadingFromFile();

    mSCdEdxCorrection.setCorrectionMaps(avgMap, derMap);
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