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
#include "TPCBase/CDBInterface.h"
#include "TPCReconstruction/TPCFastTransformHelperO2.h"
#include "CalibdEdxTrackTopologyPol.h"
#include "DataFormatsParameters/GRPMagField.h"
#include "GPUO2InterfaceUtils.h"

using namespace o2::tpc;

CalculatedEdx::CalculatedEdx()
{
  mTPCCorrMapsHelper.setOwner(true);
  mTPCCorrMapsHelper.setCorrMap(TPCFastTransformHelperO2::instance()->create(0));
}

void CalculatedEdx::setMembers(std::vector<o2::tpc::TPCClRefElem>* tpcTrackClIdxVecInput, const o2::tpc::ClusterNativeAccess& clIndex, std::vector<o2::tpc::TrackTPC>* vTPCTracksArrayInp)
{
  mTracks = vTPCTracksArrayInp;
  mTPCTrackClIdxVecInput = tpcTrackClIdxVecInput;
  mClusterIndex = &clIndex;
}

void CalculatedEdx::setRefit(const unsigned int nHbfPerTf)
{
  mTPCRefitterShMap.reserve(mClusterIndex->nClustersTotal);
  auto sizeOcc = o2::gpu::GPUO2InterfaceRefit::fillOccupancyMapGetSize(nHbfPerTf, nullptr);
  LOGP(info, "Occupancy map size: {}", sizeOcc);
  mTPCRefitterOccMap.resize(sizeOcc);
  o2::gpu::GPUO2InterfaceRefit::fillSharedClustersAndOccupancyMap(mClusterIndex, *mTracks, mTPCTrackClIdxVecInput->data(), mTPCRefitterShMap.data(), mTPCRefitterOccMap.data(), nHbfPerTf);
  mRefit = std::make_unique<o2::gpu::GPUO2InterfaceRefit>(mClusterIndex, &mTPCCorrMapsHelper, mFieldNominalGPUBz, mTPCTrackClIdxVecInput->data(), nHbfPerTf, mTPCRefitterShMap.data(), mTPCRefitterOccMap.data(), mTPCRefitterOccMap.size());
}

void CalculatedEdx::fillMissingClusters(int missingClusters[4], float minChargeTot, float minChargeMax, int method)
{
  if (method != 0 && method != 1) {
    LOGP(info, "Unrecognized subthreshold cluster treatment. Not adding virtual charges to the track!");
    return;
  }

  for (int roc = 0; roc < 4; roc++) {
    for (int i = 0; i < missingClusters[roc]; i++) {
      float chargeTot = (method == 1) ? minChargeTot / 2.f : minChargeTot;
      float chargeMax = (method == 1) ? minChargeMax / 2.f : minChargeMax;

      mChargeTotROC[roc].emplace_back(chargeTot);
      mChargeTotROC[4].emplace_back(chargeTot);

      mChargeMaxROC[roc].emplace_back(chargeMax);
      mChargeMaxROC[4].emplace_back(chargeMax);
    }
  }
}

void CalculatedEdx::calculatedEdx(o2::tpc::TrackTPC& track, dEdxInfo& output, float low, float high, CorrectionFlags correctionMask, ClusterFlags clusterMask, int subthresholdMethod, const char* debugRootFile)
{
  // get number of clusters
  const int nClusters = track.getNClusterReferences();

  int nClsROC[4] = {0, 0, 0, 0};
  int nClsSubThreshROC[4] = {0, 0, 0, 0};

  mChargeTotROC[0].clear();
  mChargeTotROC[1].clear();
  mChargeTotROC[2].clear();
  mChargeTotROC[3].clear();
  mChargeTotROC[4].clear();

  mChargeMaxROC[0].clear();
  mChargeMaxROC[1].clear();
  mChargeMaxROC[2].clear();
  mChargeMaxROC[3].clear();
  mChargeMaxROC[4].clear();

  // debug vectors
  std::vector<int> excludeClVector;
  std::vector<int> regionVector;
  std::vector<unsigned char> rowIndexVector;
  std::vector<unsigned char> padVector;
  std::vector<unsigned char> sectorVector;
  std::vector<int> stackVector;
  std::vector<float> localXVector;
  std::vector<float> localYVector;
  std::vector<float> offsPadVector;

  std::vector<float> topologyCorrVector;
  std::vector<float> topologyCorrTotVector;
  std::vector<float> topologyCorrMaxVector;
  std::vector<float> gainVector;
  std::vector<float> gainResidualVector;
  std::vector<float> residualCorrTotVector;
  std::vector<float> residualCorrMaxVector;

  std::vector<o2::tpc::TrackTPC> trackVector;
  std::vector<o2::tpc::ClusterNative> clVector;

  if (mDebug) {
    excludeClVector.reserve(nClusters);
    regionVector.reserve(nClusters);
    rowIndexVector.reserve(nClusters);
    padVector.reserve(nClusters);
    stackVector.reserve(nClusters);
    sectorVector.reserve(nClusters);
    localXVector.reserve(nClusters);
    localYVector.reserve(nClusters);
    offsPadVector.reserve(nClusters);
    topologyCorrVector.reserve(nClusters);
    topologyCorrTotVector.reserve(nClusters);
    topologyCorrMaxVector.reserve(nClusters);
    gainVector.reserve(nClusters);
    gainResidualVector.reserve(nClusters);
    residualCorrTotVector.reserve(nClusters);
    residualCorrMaxVector.reserve(nClusters);
    trackVector.reserve(nClusters);
    clVector.reserve(nClusters);
  }

  // for missing clusters
  unsigned char rowIndexOld = 0;
  unsigned char sectorIndexOld = 0;
  float minChargeTot = 100000.f;
  float minChargeMax = 100000.f;

  // loop over the clusters
  for (int iCl = 0; iCl < nClusters; iCl++) {

    const o2::tpc::ClusterNative& cl = track.getCluster(*mTPCTrackClIdxVecInput, iCl, *mClusterIndex);

    unsigned char sectorIndex = 0;
    unsigned char rowIndex = 0;
    unsigned int clusterIndexNumb = 0;

    // set sectorIndex, rowIndex, clusterIndexNumb
    track.getClusterReference(*mTPCTrackClIdxVecInput, iCl, sectorIndex, rowIndex, clusterIndexNumb);

    // get region, pad, stack and stack ID
    const int region = Mapper::REGION[rowIndex];
    const unsigned char pad = std::clamp(static_cast<unsigned int>(cl.getPad() + 0.5f), static_cast<unsigned int>(0), Mapper::PADSPERROW[region][Mapper::getLocalRowFromGlobalRow(rowIndex)] - 1); // the left side of the pad is defined at e.g. 3.5 and the right side at 4.5
    const CRU cru(Sector(sectorIndex), region);
    const auto stack = cru.gemStack();
    StackID stackID{sectorIndex, stack};
    // the stack number for debugging
    const int stackNumber = static_cast<int>(stack);

    // get local coordinates, offset and flags
    const float localX = o2::tpc::Mapper::instance().getPadCentre(PadPos(rowIndex, pad)).X();
    const float localY = Mapper::instance().getPadCentre(PadPos(rowIndex, pad)).Y();
    const float offsPad = (cl.getPad() - pad) * o2::tpc::Mapper::instance().getPadRegionInfo(Mapper::REGION[rowIndex]).getPadWidth();
    const auto flagsCl = cl.getFlags();

    int excludeCl = 0; // works as a bit mask
    if (((clusterMask & ClusterFlags::ExcludeSingleCl) == ClusterFlags::ExcludeSingleCl) && ((flagsCl & ClusterNative::flagSingle) == ClusterNative::flagSingle)) {
      excludeCl += 0b001; // 1 for single cluster
    }
    if (((clusterMask & ClusterFlags::ExcludeSplitCl) == ClusterFlags::ExcludeSplitCl) && (((flagsCl & ClusterNative::flagSplitPad) == ClusterNative::flagSplitPad) || ((flagsCl & ClusterNative::flagSplitTime) == ClusterNative::flagSplitTime))) {
      excludeCl += 0b010; // 2 for split cluster
    }
    if (((clusterMask & ClusterFlags::ExcludeEdgeCl) == ClusterFlags::ExcludeEdgeCl) && ((flagsCl & ClusterNative::flagEdge) == ClusterNative::flagEdge)) {
      excludeCl += 0b100; // 4 for edge cluster
    }

    // get the x position of the track
    const float xPosition = Mapper::instance().getPadCentre(PadPos(rowIndex, 0)).X();

    bool check = true;
    if (!mPropagateTrack) {
      if (mRefit == nullptr) {
        LOGP(error, "mRefit is a nullptr, call the function setRefit() before looping over the tracks.");
      }
      mRefit->setTrackReferenceX(xPosition);
      check = (mRefit->RefitTrackAsGPU(track, false, true) < 0) ? false : true;
    } else {
      // propagate this track to the plane X=xk (cm) in the field "b" (kG)
      track.rotate(o2::math_utils::detail::sector2Angle<float>(sectorIndex));
      check = o2::base::Propagator::Instance()->PropagateToXBxByBz(track, xPosition, 0.999f, 2., o2::base::Propagator::MatCorrType::USEMatCorrLUT);
    }

    if (!check || std::isnan(track.getParam(1))) {
      excludeCl += 0b1000; // 8 for failure of track propagation or refit
    }

    if (excludeCl != 0) {
      // for debugging
      if (mDebug) {
        excludeClVector.emplace_back(excludeCl);
        regionVector.emplace_back(region);
        rowIndexVector.emplace_back(rowIndex);
        padVector.emplace_back(pad);
        sectorVector.emplace_back(sectorIndex);
        stackVector.emplace_back(stackNumber);
        localXVector.emplace_back(localX);
        localYVector.emplace_back(localY);
        offsPadVector.emplace_back(offsPad);
        trackVector.emplace_back(track);
        clVector.emplace_back(cl);

        topologyCorrVector.emplace_back(-999.f);
        topologyCorrTotVector.emplace_back(-999.f);
        topologyCorrMaxVector.emplace_back(-999.f);
        gainVector.emplace_back(-999.f);
        gainResidualVector.emplace_back(-999.f);
        residualCorrTotVector.emplace_back(-999.f);
        residualCorrMaxVector.emplace_back(-999.f);
      }
      // to avoid counting the skipped cluster as a subthreshold cluster
      rowIndexOld = rowIndex;
      sectorIndexOld = sectorIndex;
      continue;
    }

    // get charge values
    float chargeTot = cl.qTot;
    float chargeMax = cl.qMax;

    // get threshold
    const float threshold = mCalibCont.getZeroSupressionThreshold(sectorIndex, rowIndex, pad);

    // find missing clusters
    int missingClusters = rowIndexOld - rowIndex - 1;
    if ((missingClusters > 0) && (missingClusters <= mMaxMissingCl)) {
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
    rowIndexOld = rowIndex;
    sectorIndexOld = sectorIndex;

    // get effective length
    float effectiveLength = 1.0f;
    float effectiveLengthTot = 1.0f;
    float effectiveLengthMax = 1.0f;
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
    float gain = 1.0f;
    float gainResidual = 1.0f;
    if ((correctionMask & CorrectionFlags::GainFull) == CorrectionFlags::GainFull) {
      gain = mCalibCont.getGain(sectorIndex, rowIndex, pad);
    };
    if ((correctionMask & CorrectionFlags::GainResidual) == CorrectionFlags::GainResidual) {
      gainResidual = mCalibCont.getResidualGain(sectorIndex, rowIndex, pad);
    };
    chargeTot /= gain * gainResidual;
    chargeMax /= gain * gainResidual;

    // get dEdx correction on tgl and sector plane
    float corrTot = 1.0f;
    float corrMax = 1.0f;
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

    // set the min charge
    if (chargeTot < minChargeTot) {
      minChargeTot = chargeTot;
    };

    if (chargeMax < minChargeMax) {
      minChargeMax = chargeMax;
    };

    if (stack == GEMstack::IROCgem) {
      mChargeTotROC[0].emplace_back(chargeTot);
      mChargeMaxROC[0].emplace_back(chargeMax);
      nClsROC[0]++;
    } else if (stack == GEMstack::OROC1gem) {
      mChargeTotROC[1].emplace_back(chargeTot);
      mChargeMaxROC[1].emplace_back(chargeMax);
      nClsROC[1]++;
    } else if (stack == GEMstack::OROC2gem) {
      mChargeTotROC[2].emplace_back(chargeTot);
      mChargeMaxROC[2].emplace_back(chargeMax);
      nClsROC[2]++;
    } else if (stack == GEMstack::OROC3gem) {
      mChargeTotROC[3].emplace_back(chargeTot);
      mChargeMaxROC[3].emplace_back(chargeMax);
      nClsROC[3]++;
    };

    mChargeTotROC[4].emplace_back(chargeTot);
    mChargeMaxROC[4].emplace_back(chargeMax);

    // for debugging
    if (mDebug) {
      excludeClVector.emplace_back(0); // cl is successfully processed
      regionVector.emplace_back(region);
      rowIndexVector.emplace_back(rowIndex);
      padVector.emplace_back(pad);
      sectorVector.emplace_back(sectorIndex);
      stackVector.emplace_back(stackNumber);
      localXVector.emplace_back(localX);
      localYVector.emplace_back(localY);
      offsPadVector.emplace_back(offsPad);
      trackVector.emplace_back(track);
      clVector.emplace_back(cl);

      topologyCorrVector.emplace_back(effectiveLength);
      topologyCorrTotVector.emplace_back(effectiveLengthTot);
      topologyCorrMaxVector.emplace_back(effectiveLengthMax);
      gainVector.emplace_back(gain);
      gainResidualVector.emplace_back(gainResidual);
      residualCorrTotVector.emplace_back(corrTot);
      residualCorrMaxVector.emplace_back(corrMax);
    };
  }

  // number of clusters
  output.NHitsSubThresholdIROC = nClsROC[0];
  output.NHitsSubThresholdOROC1 = nClsROC[1];
  output.NHitsSubThresholdOROC2 = nClsROC[2];
  output.NHitsSubThresholdOROC3 = nClsROC[3];

  // check if the lost clusters are subthreshold clusters based on the charge thresholds
  if (minChargeTot <= mMinChargeTotThreshold && minChargeMax <= mMinChargeMaxThreshold) {
    output.NHitsIROC = nClsROC[0] - nClsSubThreshROC[0];
    output.NHitsOROC1 = nClsROC[1] - nClsSubThreshROC[1];
    output.NHitsOROC2 = nClsROC[2] - nClsSubThreshROC[2];
    output.NHitsOROC3 = nClsROC[3] - nClsSubThreshROC[3];

    // fill subthreshold clusters if not excluded
    if (((clusterMask & ClusterFlags::ExcludeSubthresholdCl) == ClusterFlags::None)) {
      fillMissingClusters(nClsSubThreshROC, minChargeTot, minChargeMax, subthresholdMethod);
    }
  } else {
    output.NHitsIROC = nClsROC[0];
    output.NHitsOROC1 = nClsROC[1];
    output.NHitsOROC2 = nClsROC[2];
    output.NHitsOROC3 = nClsROC[3];
  }

  // calculate dEdx
  output.dEdxTotIROC = getTruncMean(mChargeTotROC[0], low, high);
  output.dEdxTotOROC1 = getTruncMean(mChargeTotROC[1], low, high);
  output.dEdxTotOROC2 = getTruncMean(mChargeTotROC[2], low, high);
  output.dEdxTotOROC3 = getTruncMean(mChargeTotROC[3], low, high);
  output.dEdxTotTPC = getTruncMean(mChargeTotROC[4], low, high);

  output.dEdxMaxIROC = getTruncMean(mChargeMaxROC[0], low, high);
  output.dEdxMaxOROC1 = getTruncMean(mChargeMaxROC[1], low, high);
  output.dEdxMaxOROC2 = getTruncMean(mChargeMaxROC[2], low, high);
  output.dEdxMaxOROC3 = getTruncMean(mChargeMaxROC[3], low, high);
  output.dEdxMaxTPC = getTruncMean(mChargeMaxROC[4], low, high);

  // for debugging
  if (mDebug) {
    if (mStreamer == nullptr) {
      setStreamer(debugRootFile);
    }

    (*mStreamer) << "dEdxDebug"
                 << "Ncl=" << nClusters
                 << "excludeClVector=" << excludeClVector
                 << "regionVector=" << regionVector
                 << "rowIndexVector=" << rowIndexVector
                 << "padVector=" << padVector
                 << "sectorVector=" << sectorVector
                 << "stackVector=" << stackVector
                 << "topologyCorrVector=" << topologyCorrVector
                 << "topologyCorrTotVector=" << topologyCorrTotVector
                 << "topologyCorrMaxVector=" << topologyCorrMaxVector
                 << "gainVector=" << gainVector
                 << "gainResidualVector=" << gainResidualVector
                 << "residualCorrTotVector=" << residualCorrTotVector
                 << "residualCorrMaxVector=" << residualCorrMaxVector
                 << "localXVector=" << localXVector
                 << "localYVector=" << localYVector
                 << "offsPadVector=" << offsPadVector
                 << "trackVector=" << trackVector
                 << "clVector=" << clVector
                 << "minChargeTot=" << minChargeTot
                 << "minChargeMax=" << minChargeMax
                 << "output=" << output
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

void CalculatedEdx::loadCalibsFromCCDB(long runNumberOrTimeStamp)
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
  o2::tpc::CalibdEdxCorrection* residualObj = cm.getForTimeStamp<o2::tpc::CalibdEdxCorrection>(o2::tpc::CDBTypeMap.at(o2::tpc::CDBType::CalTimeGain), tRun);

  const auto* residualCorr = static_cast<o2::tpc::CalibdEdxCorrection*>(residualObj);
  mCalibCont.setResidualCorrection(*residualCorr);

  // set the zero supression threshold map
  std::unordered_map<string, o2::tpc::CalDet<float>>* zeroSupressionThresholdMap = cm.getForTimeStamp<std::unordered_map<string, o2::tpc::CalDet<float>>>(o2::tpc::CDBTypeMap.at(o2::tpc::CDBType::ConfigFEEPad), tRun);
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
    std::unordered_map<string, o2::tpc::CalDet<float>>* gainMapResidual = (std::unordered_map<string, o2::tpc::CalDet<float>>*)gainMapResidualFile->Get(object);
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
    std::unordered_map<string, o2::tpc::CalDet<float>>* zeroSupressionThresholdMap = (std::unordered_map<string, o2::tpc::CalDet<float>>*)zeroSuppressionFile->Get(object);
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