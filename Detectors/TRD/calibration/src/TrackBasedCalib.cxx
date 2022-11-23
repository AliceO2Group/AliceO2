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

/// \file TrackBasedCalib.cxx
/// \brief Provides information required for TRD calibration which is based on the global tracking
/// \author Ole Schmidt

#include "TRDCalibration/TrackBasedCalib.h"
#include "TRDCalibration/CalibrationParams.h"
#include "DataFormatsTRD/Constants.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"
#include "DetectorsBase/GeometryManager.h"
#include "TRDBase/Geometry.h"
#include "TRDBase/PadPlane.h"
#include "CommonUtils/NameConf.h"
#include <fairlogger/Logger.h>

using namespace o2::trd;
using namespace o2::trd::constants;

void TrackBasedCalib::reset()
{
  mAngResHistos.reset();
}

void TrackBasedCalib::init()
{
  mRecoParam.setBfield(o2::base::Propagator::Instance()->getNominalBz());
}

void TrackBasedCalib::setInput(const o2::globaltracking::RecoContainer& input)
{
  mTracksInITSTPCTRD = input.getITSTPCTRDTracks<TrackTRD>();
  mTracksInTPCTRD = input.getTPCTRDTracks<TrackTRD>();
  mTrackletsRaw = input.getTRDTracklets();
  mTrackletsCalib = input.getTRDCalibratedTracklets();
}

void TrackBasedCalib::calculateGainCalibObjs()
{
  /*
  This method could gather the information required for the gain calibration.
  The global tracks contain dEdx information from the TPC and the TRD tracklets
  contain ADC data stored in different charge windows.
  The calibration object for the gain calibration needs to be defined
  The tracks probably don't need to be copied, since the track parameters
  don't need ot be transported.
  */
}

void TrackBasedCalib::calculateAngResHistos()
{
  if (mTrackletsRaw.size() != mTrackletsCalib.size()) {
    LOG(error) << "TRD raw tracklet container size differs from calibrated tracklet container size";
    return;
  }

  if (!mNoiseCalib) {
    LOG(alarm) << "No MCM noise map available. Please upload valid object to CCDB.";
  }

  LOGF(info, "As input tracks are available: %lu ITS-TPC-TRD tracks and %lu TPC-TRD tracks", mTracksInITSTPCTRD.size(), mTracksInTPCTRD.size());

  int nTracksSuccessITSTPCTRD = doTrdOnlyTrackFits(mTracksInITSTPCTRD);
  int nTracksSuccessTPCTRD = doTrdOnlyTrackFits(mTracksInTPCTRD);

  LOGF(info, "Successfully processed %i tracks (%i from ITS-TPC-TRD and %i from TPC-TRD) and collected %lu angular residuals",
       nTracksSuccessITSTPCTRD + nTracksSuccessTPCTRD, nTracksSuccessITSTPCTRD, nTracksSuccessTPCTRD, mAngResHistos.getNEntries());
  // mAngResHistos.print();
}

int TrackBasedCalib::doTrdOnlyTrackFits(gsl::span<const TrackTRD>& tracks)
{
  auto& params = TRDCalibParams::Instance();
  int nTracksSuccess = 0;
  for (const auto& trkIn : tracks) {
    if (trkIn.getNtracklets() < params.nTrackletsMin) {
      // with less than 3 tracklets the TRD-only refit not meaningful
      continue;
    }
    auto trkWork = trkIn; // input is const, so we need to create a copy
    bool trackFailed = false;

    trkWork.setChi2(0.f);
    trkWork.resetCovariance(20);

    if (std::isnan(trkWork.getSnp())) {
      LOG(alarm) << "Track with invalid parameters found: " << trkWork.getRefGlobalTrackId();
      continue;
    }

    // first inward propagation (TRD track fit)
    int currLayer = NLAYER;
    for (int iLayer = NLAYER - 1; iLayer >= 0; --iLayer) {
      if (trkWork.getTrackletIndex(iLayer) == -1) {
        continue;
      }
      if (mNoiseCalib && mNoiseCalib->isTrackletFromNoisyMCM(mTrackletsRaw[trkWork.getTrackletIndex(iLayer)])) {
        // ignore tracklets which originate from noisy MCMs
        continue;
      }
      if (propagateAndUpdate(trkWork, iLayer, true)) {
        trackFailed = true;
        break;
      }
      currLayer = iLayer;
    }
    if (trackFailed) {
      continue;
    }

    // outward propagation (smoothing)
    for (int iLayer = currLayer + 1; iLayer < NLAYER; ++iLayer) {
      if (trkWork.getTrackletIndex(iLayer) == -1) {
        continue;
      }
      if (mNoiseCalib && mNoiseCalib->isTrackletFromNoisyMCM(mTrackletsRaw[trkWork.getTrackletIndex(iLayer)])) {
        // ignore tracklets which originate from noisy MCMs
        continue;
      }
      if (propagateAndUpdate(trkWork, iLayer, true)) {
        trackFailed = true;
        break;
      }
      currLayer = iLayer;
    }
    if (trackFailed) {
      continue;
    }

    // second inward propagation (collect angular differences between tracklets + TRD track)
    for (int iLayer = currLayer; iLayer >= 0; --iLayer) {
      if (trkWork.getTrackletIndex(iLayer) == -1) {
        continue;
      }
      if (mNoiseCalib && mNoiseCalib->isTrackletFromNoisyMCM(mTrackletsRaw[trkWork.getTrackletIndex(iLayer)])) {
        // ignore tracklets which originate from noisy MCMs
        continue;
      }
      if (propagateAndUpdate(trkWork, iLayer, false)) {
        trackFailed = true;
        break;
      }

      if (trkWork.getReducedChi2() > params.chi2RedMax) {
        // set an upper bound on acceptable tracks we use for qc
        continue;
      }

      float trkAngle = o2::math_utils::asin(trkWork.getSnp()) * TMath::RadToDeg();
      float trkltAngle = o2::math_utils::atan(mTrackletsCalib[trkWork.getTrackletIndex(iLayer)].getDy() / Geometry::cdrHght()) * TMath::RadToDeg();
      float angleDeviation = trkltAngle - trkAngle;
      if (mAngResHistos.addEntry(angleDeviation, trkAngle, mTrackletsRaw[trkWork.getTrackletIndex(iLayer)].getDetector())) {
        // track impact angle out of histogram range
        continue;
      }
    }

    // here we can count the number of successfully processed tracks
    ++nTracksSuccess;
  } // end of track loop
  return nTracksSuccess;
}

bool TrackBasedCalib::propagateAndUpdate(TrackTRD& trk, int iLayer, bool doUpdate) const
{
  // Propagates the track to TRD layer iLayer and updates the track
  // parameters (if requested)
  // returns 0 in case of success

  auto propagator = o2::base::Propagator::Instance();

  int trkltId = trk.getTrackletIndex(iLayer);
  int trkltDet = mTrackletsRaw[trkltId].getDetector();
  int trkltSec = trkltDet / (NLAYER * NSTACK);

  if (trkltSec != o2::math_utils::angle2Sector(trk.getAlpha())) {
    if (!trk.rotate(o2::math_utils::sector2Angle(trkltSec))) {
      LOGF(debug, "Track could not be rotated in tracklet coordinate system");
      return 1;
    }
  }

  if (!propagator->PropagateToXBxByBz(trk, mTrackletsCalib[trkltId].getX(), mMaxSnp, mMaxStep, mMatCorr)) {
    LOGF(debug, "Track propagation failed in layer %i (pt=%f, xTrk=%f, xToGo=%f)", iLayer, trk.getPt(), trk.getX(), mTrackletsCalib[trkltId].getX());
    return 1;
  }

  if (!doUpdate) {
    // nothing more to be done
    return 0;
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

  if (!trk.update(trkltPosUp, trkltCovUp)) {
    LOGF(info, "Failed to update track with space point in layer %i", iLayer);
    return 1;
  }
  return 0;
}
