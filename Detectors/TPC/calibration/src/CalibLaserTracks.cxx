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

/// \file CalibLaserTracks.cxx
/// \brief calibration using laser tracks
/// \author Jens Wiechula, Jens.Wiechula@ikf.uni-frankfurt.de

#include "MathUtils/Utils.h"
#include "TPCBase/ParameterGas.h"
#include "TPCBase/ParameterElectronics.h"
#include "TPCCalibration/CalibLaserTracks.h"
#include "TLinearFitter.h"

using namespace o2::tpc;
void CalibLaserTracks::fill(std::vector<TrackTPC> const& tracks)
{
  fill(gsl::span(tracks.data(), tracks.size()));
}

//______________________________________________________________________________
void CalibLaserTracks::fill(const gsl::span<const TrackTPC> tracks)
{
  for (const auto& track : tracks) {
    processTrack(track);
  }

  endTF();
}

//______________________________________________________________________________
void CalibLaserTracks::processTrack(const TrackTPC& track)
{
  if (track.hasBothSidesClusters()) {
    return;
  }

  auto parOutAtLtr = track.getOuterParam();

  // track should have been alreay propagated close to the laser mirrors
  if (parOutAtLtr.getX() < 220) {
    return;
  }

  // recalculate z position based on trigger or CE position
  float zTrack = parOutAtLtr.getZ();

  // TODO: calculation has to be improved
  if (mTriggerPos < 0) {
    // use CE for time 0
    const float zOffset = (track.getTime0() + mTriggerPos) * mZbinWidth * mDriftV + 250;
    //printf("time0: %.2f, trigger pos: %d, zTrack: %.2f, zOffset: %.2f\n", track.getTime0(), mTriggerPos, zTrack, zOffset);
    zTrack += zOffset;
    parOutAtLtr.setZ(zTrack);
  } else if (mTriggerPos > 0) {
  }

  if (std::abs(zTrack) > 300) {
    return;
  }

  // try association with ideal laser track and rotate parameters
  const int side = track.hasCSideClusters();
  const int laserTrackID = findLaserTrackID(parOutAtLtr, side);
  if (laserTrackID < 0 || laserTrackID >= LaserTrack::NumberOfTracks) {
    return;
  }

  LaserTrack ltr = mLaserTracks.getTrack(laserTrackID);
  parOutAtLtr.rotateParam(ltr.getAlpha());
  parOutAtLtr.propagateParamTo(ltr.getX(), mBz);

  mZmatchPairsTF.emplace_back(TimePair{ltr.getZ(), parOutAtLtr.getZ(), mTFtime});

  if (mWriteDebugTree) {
    if (!mDebugStream) {
      mDebugStream = std::make_unique<o2::utils::TreeStreamRedirector>("CalibLaserTracks_debug.root", "recreate");
    }

    *mDebugStream << "ltrMatch"
                  << "ltr=" << ltr              // matched ideal laser track
                  << "trOutLtr=" << parOutAtLtr // track rotated and propagated to ideal track position
                  << "\n";
  }
}

//______________________________________________________________________________
int CalibLaserTracks::findLaserTrackID(TrackPar outerParam, int side)
{
  //const auto phisec = getPhiNearbySectorEdge(outerParam);
  const auto phisec = getPhiNearbyLaserRod(outerParam, side);
  if (!outerParam.rotateParam(phisec)) {
    return -1;
  }

  if (side < 0) {
    side = outerParam.getZ() < 0;
  }
  const int rod = std::nearbyint((phisec - LaserTrack::FirstRodPhi[side]) / LaserTrack::RodDistancePhi);
  //printf("\n\nside: %i\n", side);
  const auto xyzGlo = outerParam.getXYZGlo();
  auto phi = std::atan2(xyzGlo.Y(), xyzGlo.X());
  o2::math_utils::bringTo02Pi(phi);
  //printf("rod:  %d (phisec: %.2f, phi: %.2f\n", rod, phisec, phi);
  int bundle = -1;
  int beam = -1;
  float mindist = 1000;

  const auto outerParamZ = std::abs(outerParam.getZ());
  //printf("outerParamZ: %.2f\n", outerParamZ);
  for (size_t i = 0; i < LaserTrack::CoarseBundleZPos.size(); ++i) {
    const float dist = std::abs(outerParamZ - LaserTrack::CoarseBundleZPos[i]);
    //printf("laserZ: %.2f (%.2f, %.2f)\n", LaserTrack::CoarseBundleZPos[i], dist, mindist);
    if (dist < mindist) {
      mindist = dist;
      bundle = i;
    }
  }

  if (bundle < 0) {
    return -1;
  }

  //printf("bundle: %i\n", bundle);

  const auto outerParamsInBundle = mLaserTracks.getTracksInBundle(side, rod, bundle);
  mindist = 1000;
  for (int i = 0; i < outerParamsInBundle.size(); ++i) {
    const auto louterParam = outerParamsInBundle[i];
    if (i == 0) {
      outerParam.propagateParamTo(louterParam.getX(), mBz);
    }
    const float dist = std::abs(outerParam.getSnp() - louterParam.getSnp());
    if (dist < mindist) {
      mindist = dist;
      beam = i;
    }
  }

  //printf("beam: %i (%.4f)\n", beam, mindist);
  if (mindist > 0.01) {
    return -1;
  }

  const int trackID = LaserTrack::NumberOfTracks / 2 * side +
                      LaserTrack::BundlesPerRod * LaserTrack::TracksPerBundle * rod +
                      LaserTrack::TracksPerBundle * bundle +
                      beam;

  //printf("trackID: %d\n", trackID);

  return trackID;
}

//______________________________________________________________________________
float CalibLaserTracks::getPhiNearbySectorEdge(const TrackPar& param)
{
  // rotate to nearest laser bundle
  const auto xyzGlo = param.getXYZGlo();
  auto phi = std::atan2(xyzGlo.Y(), xyzGlo.X());
  o2::math_utils::bringTo02PiGen(phi);
  const auto phisec = std::nearbyint(phi / LaserTrack::SectorSpanRad) * LaserTrack::SectorSpanRad;
  //printf("%.2f : %.2f\n", phi, phisec);
  return (phisec);
}

//______________________________________________________________________________
float CalibLaserTracks::getPhiNearbyLaserRod(const TrackPar& param, int side)
{
  const auto xyzGlo = param.getXYZGlo();
  auto phi = std::atan2(xyzGlo.Y(), xyzGlo.X()) - LaserTrack::FirstRodPhi[side % 2];
  o2::math_utils::bringTo02PiGen(phi);
  phi = std::nearbyint(phi / LaserTrack::RodDistancePhi) * LaserTrack::RodDistancePhi + LaserTrack::FirstRodPhi[side % 2];
  o2::math_utils::bringTo02PiGen(phi);
  //printf("%.2f : %.2f\n", phi, phisec);
  return phi;
}

//______________________________________________________________________________
void CalibLaserTracks::updateParameters()
{
  const auto& gasParam = ParameterGas::Instance();
  const auto& electronicsParam = ParameterElectronics::Instance();
  mDriftV = gasParam.DriftV;
  mZbinWidth = electronicsParam.ZbinWidth;
}

//______________________________________________________________________________
void CalibLaserTracks::merge(const CalibLaserTracks* other)
{
  if (!other) {
    return;
  }
  mZmatchPairs.insert(mZmatchPairs.end(), mZmatchPairsTF.begin(), mZmatchPairsTF.end());
  mDVperTF.insert(mDVperTF.end(), other->mDVperTF.begin(), other->mDVperTF.end());

  sort(mZmatchPairs);
  sort(mDVperTF);
}

//______________________________________________________________________________
void CalibLaserTracks::endTF()
{
  if (!mZmatchPairsTF.size()) {
    return;
  }

  mZmatchPairs.insert(mZmatchPairs.end(), mZmatchPairsTF.begin(), mZmatchPairsTF.end());

  auto fitResult = fit(mZmatchPairsTF);
  mDVperTF.emplace_back(fitResult);

  if (mDebugStream) {
    (*mDebugStream) << "tfData"
                    << "tfTime=" << mTFtime
                    << "zPairs=" << mZmatchPairsTF
                    << "dvFit=" << fitResult
                    << "\n";
  }

  mZmatchPairsTF.clear();
}

//______________________________________________________________________________
void CalibLaserTracks::finalize()
{
  mDVall = fit(mZmatchPairs);

  if (mDebugStream) {
    (*mDebugStream) << "finalData"
                    << "zPairs=" << mZmatchPairs
                    << "fitPairs=" << mDVperTF
                    << "fullFit=" << mDVall
                    << "\n";
  }

  sort(mZmatchPairs);
  sort(mDVperTF);
}

//______________________________________________________________________________
TimePair CalibLaserTracks::fit(const std::vector<TimePair>& trackMatches) const
{
  if (!trackMatches.size()) {
    return TimePair();
  }

  static TLinearFitter fit(2, "pol1");
  fit.StoreData(false);
  fit.ClearPoints();

  float meanTime = 0;
  for (const auto& point : trackMatches) {
    double x = point.x1;
    double y = point.x2;
    fit.AddPoint(&x, y);

    meanTime += point.time;
  }

  meanTime /= float(trackMatches.size());

  const float robustFraction = 0.9;
  const int minPoints = 6;

  if (trackMatches.size() < size_t(minPoints / robustFraction)) {
    return TimePair({0, 0, meanTime});
  }

  //fit.EvalRobust(robustFraction);
  fit.Eval();

  TimePair retVal;
  retVal.x1 = float(fit.GetParameter(0));
  retVal.x2 = float(fit.GetParameter(1));
  retVal.time = meanTime;

  return retVal;
}

//______________________________________________________________________________
void CalibLaserTracks::sort(std::vector<TimePair>& trackMatches)
{
  std::sort(mZmatchPairs.begin(), mZmatchPairs.end(), [](const auto& first, const auto& second) { return first.time < second.time; });
}

//______________________________________________________________________________
void CalibLaserTracks::print() const
{
}
