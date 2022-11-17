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
  mRecoParam.setBfield(o2::base::Propagator::Instance()->getNominalBz());
}

void Tracking::setInput(const o2::globaltracking::RecoContainer& input)
{
  // mRecoCont = &input;
  mTracksTPC = input.getTPCTracks();
  mTracksITSTPC = input.getTPCITSTracks();
  mTracksITSTPCTRD = input.getITSTPCTRDTracks<TrackTRD>();
  mTracksTPCTRD = input.getTPCTRDTracks<TrackTRD>();
  mTrackletsRaw = input.getTRDTracklets();
  mTrackletsCalib = input.getTRDCalibratedTracklets();
}

void Tracking::run()
{
  for (const auto& trkTrd : mTracksTPCTRD) {
    checkTrack(trkTrd, true);
  }
  for (const auto& trkTrd : mTracksITSTPCTRD) {
    checkTrack(trkTrd, false);
  }
}

void Tracking::checkTrack(const TrackTRD& trkTrd, bool isTPCTRD)
{
  auto propagator = o2::base::Propagator::Instance();
  auto id = trkTrd.getRefGlobalTrackId();
  TrackQC qcStruct;
  qcStruct.type = isTPCTRD ? 0 : 1;
  qcStruct.refGlobalTrackId = id;
  qcStruct.nTracklets = trkTrd.getNtracklets();
  qcStruct.nLayers = trkTrd.getNlayersFindable();
  qcStruct.chi2 = trkTrd.getChi2();
  qcStruct.reducedChi2 = trkTrd.getReducedChi2();
  qcStruct.p = trkTrd.getP();
  qcStruct.pt = trkTrd.getPt();
  qcStruct.ptSigma2 = trkTrd.getSigma1Pt2();
  qcStruct.hasNeighbor = trkTrd.getHasNeighbor();
  qcStruct.hasPadrowCrossing = trkTrd.getHasPadrowCrossing();

  LOGF(debug, "Got track with %i tracklets and ID %i", trkTrd.getNtracklets(), id);
  const auto& trkSeed = isTPCTRD ? mTracksTPC[id].getParamOut() : mTracksITSTPC[id].getParamOut();
  qcStruct.dEdxTotTPC = isTPCTRD ? mTracksTPC[id].getdEdx().dEdxTotTPC : mTracksTPC[mTracksITSTPC[id].getRefTPC()].getdEdx().dEdxTotTPC;
  auto trk = trkSeed;
  for (int iLayer = 0; iLayer < NLAYER; ++iLayer) {
    qcStruct.isCrossingNeighbor[iLayer] = trkTrd.getIsCrossingNeighbor(iLayer);
    qcStruct.findable[iLayer] = trkTrd.getIsFindable(iLayer);
    int trkltId = trkTrd.getTrackletIndex(iLayer);
    if (trkltId < 0) {
      continue;
    }
    int trkltDet = mTrackletsRaw[trkltId].getDetector();
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
    float zPosCorrUp = mTrackletsCalib[trkltId].getZ() + mRecoParam.getZCorrCoeffNRC() * trk.getTgl();
    float padLength = pad->getRowSize(mTrackletsRaw[trkltId].getPadRow());
    if (!((trk.getSigmaZ2() < (padLength * padLength / 12.f)) && (std::fabs(mTrackletsCalib[trkltId].getZ() - trk.getZ()) < padLength))) {
      tiltCorrUp = 0.f;
    }

    std::array<float, 2> trkltPosUp{mTrackletsCalib[trkltId].getY() - tiltCorrUp, zPosCorrUp};
    std::array<float, 3> trkltCovUp;
    mRecoParam.recalcTrkltCov(tilt, trk.getSnp(), pad->getRowSize(mTrackletsRaw[trkltId].getPadRow()), trkltCovUp);
    auto chi2trklt = trk.getPredictedChi2(trkltPosUp, trkltCovUp);

    qcStruct.trackX[iLayer] = trk.getX();
    qcStruct.trackY[iLayer] = trk.getY();
    qcStruct.trackZ[iLayer] = trk.getZ();
    qcStruct.trackSnp[iLayer] = trk.getSnp();
    qcStruct.trackTgl[iLayer] = trk.getTgl();
    qcStruct.trackQpt[iLayer] = trk.getQ2Pt();
    qcStruct.trackPhi[iLayer] = trk.getPhi();
    qcStruct.trackEta[iLayer] = trk.getEta();
    qcStruct.trackletYraw[iLayer] = mTrackletsCalib[trkltId].getY();
    qcStruct.trackletZraw[iLayer] = mTrackletsCalib[trkltId].getZ();
    qcStruct.trackletY[iLayer] = trkltPosUp[0];
    qcStruct.trackletZ[iLayer] = trkltPosUp[1];
    qcStruct.trackletDy[iLayer] = mTrackletsCalib[trkltId].getDy();
    qcStruct.trackletSlope[iLayer] = mTrackletsRaw[trkltId].getSlope();
    qcStruct.trackletSlopeSigned[iLayer] = mTrackletsRaw[trkltId].getSlopeBinSigned();
    qcStruct.trackletPosition[iLayer] = mTrackletsRaw[trkltId].getPosition();
    qcStruct.trackletPositionSigned[iLayer] = mTrackletsRaw[trkltId].getPositionBinSigned();
    qcStruct.trackletDet[iLayer] = trkltDet;
    qcStruct.trackletHCId[iLayer] = mTrackletsRaw[trkltId].getHCID();
    qcStruct.trackletRob[iLayer] = mTrackletsRaw[trkltId].getROB();
    qcStruct.trackletMcm[iLayer] = mTrackletsRaw[trkltId].getMCM();
    qcStruct.trackletChi2[iLayer] = chi2trklt;
    qcStruct.trackletCharges[iLayer] = {
      mTrackletsRaw[trkltId].getQ0(),
      mTrackletsRaw[trkltId].getQ1(),
      mTrackletsRaw[trkltId].getQ2(),
    };
  }
  mTrackQC.push_back(qcStruct);
}
