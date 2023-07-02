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
#include <chrono>

using namespace o2::tpc;
void CalibLaserTracks::fill(std::vector<TrackTPC> const& tracks)
{
  fill(gsl::span(tracks.data(), tracks.size()));
}

//______________________________________________________________________________
void CalibLaserTracks::fill(const gsl::span<const TrackTPC> tracks)
{
  // ===| clean up TF data |===
  mZmatchPairsTFA.clear();
  mZmatchPairsTFC.clear();
  mCalibDataTF.reset();

  if (!tracks.size()) {
    return;
  }

  mCalibData.nTrackTF.emplace_back();
  mCalibDataTF.nTrackTF.emplace_back();

  // ===| associate tracks with ideal laser track positions |===
  for (const auto& track : tracks) {
    processTrack(track);
  }

  // ===| set TF start and end times |===
  if (mCalibData.firstTime == 0 && mCalibData.lastTime == 0) {
    mCalibData.firstTime = mTFstart;
  }

  auto tfEnd = mTFend;
  if (tfEnd == 0) {
    tfEnd = mTFstart;
  }
  mCalibData.lastTime = tfEnd;

  mCalibDataTF.firstTime = mTFstart;
  mCalibDataTF.lastTime = tfEnd;

  // ===| TF counters |===
  ++mCalibData.processedTFs;
  ++mCalibDataTF.processedTFs;

  // ===| finalize TF processing |===
  endTF();
}

//______________________________________________________________________________
void CalibLaserTracks::processTrack(const TrackTPC& track)
{
  if (track.hasBothSidesClusters()) {
    return;
  }

  // use outer parameters which are closest to the laser mirrors
  auto parOutAtLtr = track.getOuterParam();

  // track should have been alreay propagated close to the laser mirrors
  if (parOutAtLtr.getX() < 220) {
    return;
  }

  // recalculate z position based on trigger or CE position if needed
  float zTrack = parOutAtLtr.getZ();

  // TODO: calculation has to be improved
  if (mTriggerPos < 0) {
    // use CE for time 0
    const float zOffset = (track.getTime0() + mTriggerPos) * mZbinWidth * mDriftV + 250;
    // printf("time0: %.2f, trigger pos: %d, zTrack: %.2f, zOffset: %.2f\n", track.getTime0(), mTriggerPos, zTrack, zOffset);
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

  auto ltr = mLaserTracks.getTrack(laserTrackID);
  parOutAtLtr.rotateParam(ltr.getAlpha());
  parOutAtLtr.propagateParamTo(ltr.getX(), mBz);

  if (ltr.getSide() == 0) {
    mZmatchPairsA.emplace_back(TimePair{ltr.getZ(), parOutAtLtr.getZ(), mTFstart});
    mZmatchPairsTFA.emplace_back(TimePair{ltr.getZ(), parOutAtLtr.getZ(), mTFstart});
  } else {
    mZmatchPairsC.emplace_back(TimePair{ltr.getZ(), parOutAtLtr.getZ(), mTFstart});
    mZmatchPairsTFC.emplace_back(TimePair{ltr.getZ(), parOutAtLtr.getZ(), mTFstart});
  }

  mCalibData.matchedLtrIDs.emplace_back(laserTrackID);
  mCalibDataTF.matchedLtrIDs.emplace_back(laserTrackID);

  const auto dEdx = track.getdEdx().dEdxTotTPC;
  mCalibData.dEdx.emplace_back(dEdx);
  mCalibDataTF.dEdx.emplace_back(dEdx);

  ++mCalibData.nTrackTF.back();
  ++mCalibDataTF.nTrackTF.back();

  // ===| debug output |========================================================
  if (mWriteDebugTree) {
    if (!mDebugStream) {
      mDebugStream = std::make_unique<o2::utils::TreeStreamRedirector>(mDebugOutputName.data(), "recreate");
    }

    auto writeTrack = track;
    *mDebugStream << "ltrMatch"
                  << "tfStart=" << mTFstart
                  << "tfEnd=" << mTFend
                  << "ltr=" << ltr              // matched ideal laser track
                  << "trOutLtr=" << parOutAtLtr // track rotated and propagated to ideal track position
                  << "TPCTracks=" << writeTrack // original TPC track
                  << "\n";
  }
}

//______________________________________________________________________________
int CalibLaserTracks::findLaserTrackID(TrackPar outerParam, int side)
{
  // ===| rotate outer param to closes laser rod |===
  const auto phisec = getPhiNearbyLaserRod(outerParam, side);
  if (!outerParam.rotateParam(phisec)) {
    return -1;
  }

  if (side < 0) {
    side = outerParam.getZ() < 0;
  }

  // ===| laser rod |===
  const int rod = std::nearbyint((phisec - LaserTrack::FirstRodPhi[side]) / LaserTrack::RodDistancePhi);

  // ===| laser bundle |===
  float mindist = 1000;

  const auto outerParamZ = std::abs(outerParam.getZ());

  int bundle = -1;
  for (size_t i = 0; i < LaserTrack::CoarseBundleZPos.size(); ++i) {
    const float dist = std::abs(outerParamZ - LaserTrack::CoarseBundleZPos[i]);
    if (dist < mindist) {
      mindist = dist;
      bundle = i;
    }
  }

  if (bundle < 0) {
    return -1;
  }

  // ===| laser beam |===
  const auto outerParamsInBundle = mLaserTracks.getTracksInBundle(side, rod, bundle);
  mindist = 1000;

  int beam = -1;
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

  if (mindist > 0.01) {
    return -1;
  }

  // ===| track ID from side, rod, bundle and beam |===
  const int trackID = LaserTrack::NumberOfTracks / 2 * side +
                      LaserTrack::BundlesPerRod * LaserTrack::TracksPerBundle * rod +
                      LaserTrack::TracksPerBundle * bundle +
                      beam;

  return trackID;
}

//______________________________________________________________________________
float CalibLaserTracks::getPhiNearbyLaserRod(const TrackPar& param, int side)
{
  const auto xyzGlo = param.getXYZGlo();
  const auto phiTrack = o2::math_utils::to02PiGen(std::atan2(xyzGlo.Y(), xyzGlo.X()) - LaserTrack::FirstRodPhi[side % 2]);
  const auto phiRod = o2::math_utils::to02PiGen(std::nearbyint(phiTrack / LaserTrack::RodDistancePhi) * LaserTrack::RodDistancePhi + LaserTrack::FirstRodPhi[side % 2]);

  return phiRod;
}

//______________________________________________________________________________
bool CalibLaserTracks::hasNearbyLaserRod(const TrackPar& param, int side)
{
  const auto xyzGlo = param.getXYZGlo();
  const auto phiTrack = o2::math_utils::to02PiGen(std::atan2(xyzGlo.Y(), xyzGlo.X()) - LaserTrack::FirstRodPhi[side % 2]);
  const auto phiRod = o2::math_utils::to02PiGen(std::nearbyint(phiTrack / LaserTrack::RodDistancePhi) * LaserTrack::RodDistancePhi);

  return std::abs(phiRod - phiTrack) < LaserTrack::SectorSpanRad / 4.;
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
  mCalibData.processedTFs += other->mCalibData.processedTFs;

  const auto sizeAthis = mZmatchPairsA.size();
  const auto sizeCthis = mZmatchPairsC.size();
  const auto sizeAother = other->mZmatchPairsA.size();
  const auto sizeCother = other->mZmatchPairsC.size();

  mZmatchPairsA.insert(mZmatchPairsA.end(), other->mZmatchPairsA.begin(), other->mZmatchPairsA.end());
  mZmatchPairsC.insert(mZmatchPairsC.end(), other->mZmatchPairsC.begin(), other->mZmatchPairsC.end());

  auto& ltrIDs = mCalibData.matchedLtrIDs;
  auto& ltrIDsOther = other->mCalibData.matchedLtrIDs;
  ltrIDs.insert(ltrIDs.end(), ltrIDsOther.begin(), ltrIDsOther.end());

  auto& nTrk = mCalibData.nTrackTF;
  auto& nTrkOther = other->mCalibData.nTrackTF;
  nTrk.insert(nTrk.end(), nTrkOther.begin(), nTrkOther.end());

  auto& dEdx = mCalibData.dEdx;
  auto& dEdxOther = other->mCalibData.dEdx;
  dEdx.insert(dEdx.end(), dEdxOther.begin(), dEdxOther.end());

  mCalibData.firstTime = std::min(mCalibData.firstTime, other->mCalibData.firstTime);
  mCalibData.lastTime = std::max(mCalibData.lastTime, other->mCalibData.lastTime);

  sort(mZmatchPairsA);
  sort(mZmatchPairsC);

  LOGP(info, "Merged CalibLaserTracks with mached pairs {} / {} + {} / {} = {} / {} (this +_other A- / C-Side)", sizeAthis, sizeCthis, sizeAother, sizeCother, mZmatchPairsA.size(), mZmatchPairsC.size());
}

//______________________________________________________________________________
void CalibLaserTracks::endTF()
{
  LOGP(info, "Ending time frame {} - {} with {} / {} matched laser tracks (total: {} / {}) on the A / C-Side", mTFstart, mTFend, mZmatchPairsTFA.size(), mZmatchPairsTFC.size(), mZmatchPairsA.size(), mZmatchPairsC.size());
  fillCalibData(mCalibDataTF, mZmatchPairsTFA, mZmatchPairsTFC);

  if (mDebugStream) {
    (*mDebugStream) << "tfData"
                    << "tfStart=" << mTFstart
                    << "tfEnf=" << mTFend
                    << "zPairsA=" << mZmatchPairsTFA
                    << "zPairsC=" << mZmatchPairsTFC
                    << "calibData=" << mCalibDataTF
                    << "\n";
  }
}

//______________________________________________________________________________
void CalibLaserTracks::finalize()
{
  mFinalized = true;
  sort(mZmatchPairsA);
  sort(mZmatchPairsC);

  fillCalibData(mCalibData, mZmatchPairsA, mZmatchPairsC);

  // auto& ltrIDs = mCalibData.matchedLtrIDs;
  // std::sort(ltrIDs.begin(), ltrIDs.end());
  // ltrIDs.erase(std::unique(ltrIDs.begin(), ltrIDs.end()), ltrIDs.end());

  if (mDebugStream) {
    (*mDebugStream) << "finalData"
                    << "zPairsA=" << mZmatchPairsA
                    << "zPairsC=" << mZmatchPairsC
                    << "calibData=" << mCalibData
                    << "\n";

    mDebugStream->Close();
  }
}

//______________________________________________________________________________
void CalibLaserTracks::fillCalibData(LtrCalibData& calibData, const std::vector<TimePair>& pairsA, const std::vector<TimePair>& pairsC)
{
  auto dvA = fit(pairsA);
  auto dvC = fit(pairsC);
  calibData.creationTime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
  calibData.refVDrift = mDriftV;
  calibData.dvOffsetA = dvA.x1;
  calibData.dvCorrectionA = dvA.x2;
  calibData.nTracksA = uint16_t(pairsA.size());

  calibData.dvOffsetC = dvC.x1;
  calibData.dvCorrectionC = dvC.x2;
  calibData.nTracksC = uint16_t(pairsC.size());
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

  uint64_t meanTime = 0;
  for (const auto& point : trackMatches) {
    double x = point.x1;
    double y = point.x2;
    fit.AddPoint(&x, y);

    meanTime += point.time;
  }

  meanTime /= uint64_t(trackMatches.size());

  const float robustFraction = 0.9;
  const int minPoints = 6;

  if (trackMatches.size() < size_t(minPoints / robustFraction)) {
    return TimePair({0, 0, meanTime});
  }

  // fit.EvalRobust(robustFraction);
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
  std::sort(trackMatches.begin(), trackMatches.end(), [](const auto& first, const auto& second) { return first.time < second.time; });
}

//______________________________________________________________________________
void CalibLaserTracks::print() const
{
  if (mFinalized) {
    LOGP(info,
         "Processed {} TFs from {} - {}; found tracks: {} / {}; T0 offsets: {} / {}; dv correction factors: {} / {} for A- / C-Side, reference: {}",
         mCalibData.processedTFs,
         mCalibData.firstTime,
         mCalibData.lastTime,
         mCalibData.nTracksA,
         mCalibData.nTracksC,
         mCalibData.dvOffsetA,
         mCalibData.dvOffsetC,
         mCalibData.dvCorrectionA,
         mCalibData.dvCorrectionC,
         mCalibData.refVDrift);
  } else {
    LOGP(info,
         "Processed {} TFs from {} - {}; **Not finalized**",
         mCalibData.processedTFs,
         mCalibData.firstTime,
         mCalibData.lastTime);

    LOGP(info,
         "Last processed TF from {} - {}; found tracks: {} / {}; T0 offsets: {} / {}; dv correction factors: {} / {} for A- / C-Side, reference: {}",
         mCalibDataTF.firstTime,
         mCalibDataTF.lastTime,
         mCalibDataTF.nTracksA,
         mCalibDataTF.nTracksC,
         mCalibDataTF.dvOffsetA,
         mCalibDataTF.dvOffsetC,
         mCalibDataTF.dvCorrectionA,
         mCalibDataTF.dvCorrectionC,
         mCalibDataTF.refVDrift);
  }
}
