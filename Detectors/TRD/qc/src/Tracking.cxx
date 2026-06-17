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

/// \file Tracking.cxx
/// \brief Check the performance of the TRD in global tracking
/// \author Ole Schmidt

#include "GPUO2InterfaceConfiguration.h"
#include "TRDQC/Tracking.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"
#include "DetectorsBase/GeometryManager.h"
#include "TRDBase/Geometry.h"
#include "TRDBase/PadPlane.h"
#include <fairlogger/Logger.h>

using namespace o2::trd;
using namespace o2::trd::constants;

void Tracking::init()
{
  mRecoParam.init(o2::base::Propagator::Instance()->getNominalBz());
}

void Tracking::setInput(const o2::globaltracking::RecoContainer& input)
{
  mTracksTPC = input.getTPCTracks();
  mTracksITSTPC = input.getTPCITSTracks();
  mTracksITSTPCTRD = input.getITSTPCTRDTracks<TrackTRD>();
  mTracksTPCTRD = input.getTPCTRDTracks<TrackTRD>();
  mTrackletsRaw = input.getTRDTracklets();
  mTrackletsCalib = input.getTRDCalibratedTracklets();
  mTrackTriggerRecordsITSTPCTRD = input.getITSTPCTRDTriggers();
  mTrackTriggerRecordsTPCTRD = input.getTPCTRDTriggers();
}

void Tracking::run()
{
  mCurrentTriggerRecord = 0;
  mCurrentTrackId = 0;
  for (const auto& trkTrd : mTracksTPCTRD) {
    checkTrack(trkTrd, true);
    mCurrentTrackId++;
  }
  mCurrentTriggerRecord = 0;
  mCurrentTrackId = 0;
  for (const auto& trkTrd : mTracksITSTPCTRD) {
    checkTrack(trkTrd, false);
    mCurrentTrackId++;
  }
}

void Tracking::checkTrack(const TrackTRD& trkTrd, bool isTPCTRD)
{
  auto propagator = o2::base::Propagator::Instance();
  auto id = trkTrd.getRefGlobalTrackId();
  TrackQC qcStruct;

  qcStruct.refGlobalTrackId = id;
  qcStruct.trackTRD = trkTrd;

  LOGF(debug, "Got track with %i tracklets and ID %i", trkTrd.getNtracklets(), (int)id);
  o2::track::TrackParCov trk = isTPCTRD ? mTracksTPC[id].getParamOut() : mTracksITSTPC[id].getParamOut();
  qcStruct.trackSeed = trk;
  if (mPID) {
    qcStruct.dEdxTotTPC = isTPCTRD ? mTracksTPC[id].getdEdx().dEdxTotTPC : mTracksTPC[mTracksITSTPC[id].getRefTPC()].getdEdx().dEdxTotTPC;
  }

  // find corresponding track trigger record to get track timing
  int triggeredBC = 0;
  for (; mCurrentTriggerRecord < (isTPCTRD ? mTrackTriggerRecordsTPCTRD.size() : mTrackTriggerRecordsITSTPCTRD.size()); mCurrentTriggerRecord++) {
    auto& tRecord = (isTPCTRD ? mTrackTriggerRecordsTPCTRD[mCurrentTriggerRecord] : mTrackTriggerRecordsITSTPCTRD[mCurrentTriggerRecord]);
    if (mCurrentTrackId >= tRecord.getFirstTrack() && mCurrentTrackId < tRecord.getFirstTrack() + tRecord.getNumberOfTracks()) {
      triggeredBC = tRecord.getBCData().differenceInBC({0, mFirstOrbit});
      break;
    }
  }

  // Find most probable BCs and RMS for pile-up correction and error. Same BC is assumed for all tracklets
  float tCorrPileUp = 0.;
  float tErrPileUp2 = 0;
  float maxProb = 0.f;
  // The uncertainty is the RMS wrt the default correction of all possible corrections weighted by their probability
  float sumCorr = 0.f;
  float sumCorr2 = 0.f;
  float sumProb = 0.f;
  for (int iBC = 0; iBC < mTriggeredBCFT0.size(); iBC++) {
    int deltaBC = roundf(mTriggeredBCFT0[iBC] - triggeredBC);
    if (deltaBC <= mRecoParam.getPileUpRangeBefore()) {
      continue;
    }
    if (deltaBC >= mRecoParam.getPileUpRangeAfter()) {
      break;
    }
    // collect the charges
    std::array<int, 6> q0;
    std::array<int, 6> q1;
    for (int iLy = 0; iLy < NLAYER; iLy++) {
      int trkltId = trkTrd.getTrackletIndex(iLy);
      if (trkltId < 0) {
        q0[iLy] = -1;
        q1[iLy] = -1;
      } else {
        q0[iLy] = mTrackletsRaw[trkltId].getQ0();
        q1[iLy] = mTrackletsRaw[trkltId].getQ1();
      }
    }
    // get pile-up probability
    float probBC = mRecoParam.getPileUpProbTrack(deltaBC, q0, q1);
    sumCorr += probBC * deltaBC;
    sumCorr2 += probBC * deltaBC * deltaBC;
    sumProb += probBC;
    if (probBC > maxProb) {
      maxProb = probBC;
      tCorrPileUp = -deltaBC;
    }
  }
  if (sumProb > 1e-6) {
    tErrPileUp2 = sumCorr2 / sumProb - 2 * tCorrPileUp * sumCorr / sumProb + tCorrPileUp * tCorrPileUp;
  }

  for (int iLayer = 0; iLayer < NLAYER; ++iLayer) {
    int trkltId = trkTrd.getTrackletIndex(iLayer);
    if (trkltId < 0) {
      continue;
    }
    const auto& tracklet = mTrackletsRaw[trkltId];
    qcStruct.trklt64[iLayer] = tracklet;
    qcStruct.trkltCalib[iLayer] = mTrackletsCalib[trkltId];
    int trkltDet = tracklet.getDetector();
    int trkltSec = trkltDet / (NLAYER * NSTACK);
    if (trkltSec != o2::math_utils::angle2Sector(trk.getAlpha())) {
      if (!trk.rotate(o2::math_utils::sector2Angle(trkltSec))) {
        LOGF(debug, "Track could not be rotated in tracklet coordinate system");
        break;
      }
    }
    if (!propagator->PropagateToXBxByBz(trk, mTrackletsCalib[trkltId].getX(), mMaxSnp, mMaxStep, mMatCorr)) {
      LOGF(debug, "Track propagation failed in layer %i (pt=%f, xTrk=%f, xToGo=%f)", iLayer, trk.getPt(), trk.getX(), mTrackletsCalib[trkltId].getX());
      break;
    }
    const PadPlane* pad = Geometry::instance()->getPadPlane(trkltDet);
    float tilt = tan(TMath::DegToRad() * pad->getTiltingAngle()); // tilt is signed! and returned in degrees
    float tiltCorrUp = tilt * (mTrackletsCalib[trkltId].getZ() - trk.getZ());
    float dyTiltCorr = tilt * trk.getTgl() * Geometry::instance()->cdrHght();
    float zPosCorrUp = mTrackletsCalib[trkltId].getZ() + mRecoParam.getZCorrCoeffNRC() * trk.getTgl();
    float padLength = pad->getRowSize(tracklet.getPadRow());
    if (!((trk.getSigmaZ2() < (padLength * padLength / 12.f)) && (std::fabs(mTrackletsCalib[trkltId].getZ() - trk.getZ()) < padLength))) {
      tiltCorrUp = 0.f;
    }

    // conversion from slope in pad per time bin to slope in cm per BC = tracklets[trkltIdx].getSlopeFloat() * padWidth / BCperTimeBin
    float slopeFactor = mTrackletsRaw[trkltId].getSlopeFloat() * pad->getWidthIPad() / 4.f;
    float yCorrPileUp = tCorrPileUp * slopeFactor;
    float yAddErrPileUp2 = tErrPileUp2 * slopeFactor * slopeFactor;

    float angularPull = (mTrackletsCalib[trkltId].getDy() + dyTiltCorr - mRecoParam.convertAngleToDy(trk.getSnp())) / std::sqrt(mRecoParam.getDyRes(trk.getSnp(), 0));

    std::array<float, 2> trkltPosUp{mTrackletsCalib[trkltId].getY() - tiltCorrUp + yCorrPileUp, zPosCorrUp};
    std::array<float, 3> trkltCovUp;
    mRecoParam.recalcTrkltCov(tilt, trk.getSnp(), pad->getRowSize(tracklet.getPadRow()), trkltCovUp, angularPull, 0);
    trkltCovUp[0] += yAddErrPileUp2;
    auto chi2trklt = trk.getPredictedChi2(trkltPosUp, trkltCovUp);

    qcStruct.trackProp[iLayer] = trk;
    qcStruct.trackletY[iLayer] = trkltPosUp[0];
    qcStruct.trackletZ[iLayer] = trkltPosUp[1];
    qcStruct.trackletChi2[iLayer] = chi2trklt;

    /// Corrected Tracklets
    // To correct for longer drift lengths in the drfit chamber and hence more energy deposition,
    // we calculate the variation to reference for the tracklets (exrapolated from the track fit).
    // \sqrt{ (dx/dx)^2 + (dy/dx)^2 + (dz/dx)^2}
    auto tphi = trk.getSnp() / std::sqrt((1.f - trk.getSnp()) * (1.f + trk.getSnp()));
    auto trackletLength = std::sqrt(1.f + tphi * tphi + trk.getTgl() * trk.getTgl());
    auto cor = mLocalGain.getValue(tracklet.getHCID() / 2, tracklet.getPadCol(mApplyShift), tracklet.getPadRow()) * trackletLength;
    float q0{tracklet.getQ0() / cor}, q1{tracklet.getQ1() / cor}, q2{tracklet.getQ2() / cor};

    // z-row merging
    if (trkTrd.getIsCrossingNeighbor(iLayer) && trkTrd.getHasNeighbor()) {
      for (const auto& trklt : mTrackletsRaw) {
        if (tracklet.getTrackletWord() == trklt.getTrackletWord()) { // skip original tracklet
          continue;
        }
        if (std::abs(tracklet.getPadCol(mApplyShift) - trklt.getPadCol(mApplyShift)) <= 1 && std::abs(tracklet.getPadRow() - trklt.getPadRow()) == 1) {
          // Add charge information
          auto cor = mLocalGain.getValue(trklt.getHCID() / 2, trklt.getPadCol(mApplyShift), trklt.getPadRow()) * trackletLength;
          q0 += trklt.getQ0() / cor;
          q1 += trklt.getQ1() / cor;
          q2 += trklt.getQ2() / cor;
          break;
        }
      }
    }

    qcStruct.trackletCorCharges[iLayer] = {q0, q1, q2};
  }
  mTrackQC.push_back(qcStruct);
}
