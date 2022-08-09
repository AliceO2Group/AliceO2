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

/// @file AlignPointHelper.cxx

#include <cmath>

#include <Rtypes.h>
#include "Framework/Logger.h"
#include "DataFormatsMFT/TrackMFT.h"
#include "MFTAlignment/AlignSensorHelper.h"
#include "MFTAlignment/AlignPointHelper.h"
#include "MFTTracking/IOUtils.h"
#include "ITSMFTBase/SegmentationAlpide.h"

using namespace o2::mft;

ClassImp(o2::mft::AlignPointHelper);

//__________________________________________________________________________
AlignPointHelper::AlignPointHelper()
  : mIsAlignPointSet(false),
    mIsGlobalDerivativeDone(false),
    mIsLocalDerivativeDone(false),
    mIsTrackInitialParamSet(false),
    mIsClusterOk(false),
    mGeometry(nullptr),
    mDictionary(nullptr),
    mChipHelper(nullptr),
    mGlobalRecoPosition(0., 0., 0.),
    mLocalRecoPosition(0., 0., 0.),
    mLocalMeasuredPosition(0., 0., 0.),
    mLocalMeasuredPositionSigma(0., 0., 0),
    mGlobalMeasuredPosition(0., 0., 0.),
    mLocalResidual(0., 0., 0.)
{

  mGeometry = o2::mft::GeometryTGeo::Instance();
  mGeometry->fillMatrixCache(
    o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L,
                             o2::math_utils::TransformType::L2G));

  resetLocalDerivatives();
  resetGlobalDerivatives();
  resetTrackInitialParam();
  resetAlignPoint();

  mChipHelper = std::make_unique<AlignSensorHelper>();
  LOGF(debug, "AlignPointHelper instantiated");
}

//__________________________________________________________________________
void AlignPointHelper::computeLocalDerivatives()
{
  resetLocalDerivatives();
  if (mChipHelper == nullptr) {
    LOGF(error,
         "AlignPointHelper::computeLocalDerivatives() - no AlignSensorHelper found !");
    return;
  }
  if (!mIsTrackInitialParamSet) {
    LOGF(error,
         "AlignPointHelper::computeLocalDerivatives() - no initial track param found !");
    return;
  }
  if (!mIsAlignPointSet) {
    LOGF(error,
         "AlignPointHelper::computeLocalDerivatives() - no align point coordinates set !");
    return;
  }
  bool success = true;
  success &= computeLocalDerivativeX();
  success &= computeLocalDerivativeY();
  success &= computeLocalDerivativeZ();
  mIsLocalDerivativeDone = success;
}

//__________________________________________________________________________
void AlignPointHelper::computeGlobalDerivatives()
{
  resetGlobalDerivatives();
  if (mChipHelper == nullptr) {
    LOGF(error,
         "AlignPointHelper::computeGlobalDerivatives() - no AlignSensorHelper found !");
    return;
  }
  if (!mIsTrackInitialParamSet) {
    LOGF(error,
         "AlignPointHelper::computeGlobalDerivatives() - no initial track param found !");
    return;
  }
  if (!mIsAlignPointSet) {
    LOGF(error, "AlignPointHelper::computeGlobalDerivatives() - no align point coordinates set !");
    return;
  }
  bool success = true;
  success &= computeGlobalDerivativeX();
  success &= computeGlobalDerivativeY();
  success &= computeGlobalDerivativeZ();
  mIsGlobalDerivativeDone = success;
}

//__________________________________________________________________________
UShort_t AlignPointHelper::getSensorId() const
{
  if (mChipHelper == nullptr) {
    LOGF(error,
         "AlignPointHelper::getSensorId() - no AlignSensorHelper found !");
    return 0;
  }
  if (!mIsAlignPointSet) {
    LOGF(error,
         "AlignPointHelper::getSensorId() - no align point coordinates set !");
    return 0;
  }
  return mChipHelper->chipIndexInMft();
}

//__________________________________________________________________________
UShort_t AlignPointHelper::half() const
{
  if (mChipHelper == nullptr) {
    LOGF(error,
         "AlignPointHelper::half() - no AlignSensorHelper found !");
    return 0;
  }
  if (!mIsAlignPointSet) {
    LOGF(error,
         "AlignPointHelper::half() - no align point coordinates set !");
    return 0;
  }
  return mChipHelper->half();
}

//__________________________________________________________________________
UShort_t AlignPointHelper::disk() const
{
  if (mChipHelper == nullptr) {
    LOGF(error,
         "AlignPointHelper::disk() - no AlignSensorHelper found !");
    return 0;
  }
  if (!mIsAlignPointSet) {
    LOGF(error,
         "AlignPointHelper::disk() - no align point coordinates set !");
    return 0;
  }
  return mChipHelper->disk();
}

//__________________________________________________________________________
UShort_t AlignPointHelper::layer() const
{
  if (mChipHelper == nullptr) {
    LOGF(error,
         "AlignPointHelper::layer() - no AlignSensorHelper found !");
    return 0;
  }
  if (!mIsAlignPointSet) {
    LOGF(error,
         "AlignPointHelper::layer() - no align point coordinates set !");
    return 0;
  }
  return mChipHelper->layer();
}

//__________________________________________________________________________
void AlignPointHelper::resetAlignPoint()
{
  mGlobalRecoPosition.SetXYZ(0., 0., 0.);
  mLocalRecoPosition.SetXYZ(0., 0., 0.);

  mLocalMeasuredPosition.SetXYZ(0., 0., 0.);
  mLocalMeasuredPositionSigma.SetXYZ(
    o2::mft::ioutils::DefClusErrorRow,
    o2::itsmft::SegmentationAlpide::SensorLayerThicknessEff * 0.5,
    o2::mft::ioutils::DefClusErrorCol);
  mGlobalMeasuredPosition.SetXYZ(0., 0., 0.);

  mLocalResidual.SetXYZ(0., 0., 0.);

  mIsAlignPointSet = false;
}

//__________________________________________________________________________
void AlignPointHelper::resetTrackInitialParam()
{
  mTrackInitialParam.X0 = 0.;
  mTrackInitialParam.Y0 = 0.;
  mTrackInitialParam.Z0 = 0.;
  mTrackInitialParam.Tx = 0.;
  mTrackInitialParam.Ty = 0.;

  mIsTrackInitialParamSet = false;
}

//__________________________________________________________________________
void AlignPointHelper::convertCompactClusters(gsl::span<const itsmft::CompClusterExt> clusters,
                                              gsl::span<const unsigned char>::iterator& pattIt,
                                              std::vector<o2::BaseCluster<double>>& outputLocalClusters,
                                              std::vector<o2::BaseCluster<double>>& outputGlobalClusters)
{
  // use this version of convertCompactClusters() in a workflow

  if (mDictionary == nullptr) {
    LOGF(error,
         "AlignPointHelper::convertCompactClusters() - no dictionary found !");
    return;
  }
  if (mGeometry == nullptr) {
    mGeometry = o2::mft::GeometryTGeo::Instance();
  }
  mGeometry->fillMatrixCache(
    o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L,
                             o2::math_utils::TransformType::L2G));
  outputLocalClusters.clear();
  outputGlobalClusters.clear();
  // inspired from Detectors/ITSMFT/MFT/tracking/src/IOUtils.cxx
  for (auto& mftCluster : clusters) {
    auto chipID = mftCluster.getChipID();
    auto pattID = mftCluster.getPatternID();
    // Dummy COG errors (about half pixel size)
    double sigmaX = o2::mft::ioutils::DefClusErrorRow;
    double sigmaZ = o2::mft::ioutils::DefClusErrorCol;

    o2::math_utils::Point3D<double> locXYZ;
    if (pattID != o2::itsmft::CompCluster::InvalidPatternID) {
      // ALPIDE local Y coordinate => MFT global X coordinate (ALPIDE rows)
      sigmaX = mDictionary->getErrX(pattID);
      // ALPIDE local Z coordinate => MFT global Y coordinate (ALPIDE columns)
      sigmaZ = mDictionary->getErrZ(pattID);

      if (!mDictionary->isGroup(pattID)) {
        locXYZ = mDictionary->getClusterCoordinates(mftCluster);
      } else {
        o2::itsmft::ClusterPattern cPattern(pattIt);
        locXYZ = mDictionary->getClusterCoordinates(mftCluster, cPattern);
      }
    } else {
      o2::itsmft::ClusterPattern cPattern(pattIt);
      locXYZ = mDictionary->getClusterCoordinates(mftCluster, cPattern, false);
    }
    auto gloXYZ = mGeometry->getMatrixL2G(chipID) * locXYZ;

    auto& locCl3d = outputLocalClusters.emplace_back(chipID, locXYZ); // local
    locCl3d.setErrors(sigmaX, o2::itsmft::SegmentationAlpide::SensorLayerThicknessEff * 0.5, sigmaZ);

    auto& gloCl3d = outputGlobalClusters.emplace_back(chipID, gloXYZ); // global
    gloCl3d.setErrors(sigmaX, sigmaZ, o2::itsmft::SegmentationAlpide::SensorLayerThicknessEff * 0.5);
  }
  LOGF(debug,
       "AlignPointHelper::convertCompactClusters() - output vector size %d",
       outputLocalClusters.size());
}

//__________________________________________________________________________
void AlignPointHelper::convertCompactClusters(const std::vector<o2::itsmft::CompClusterExt>& clusters,
                                              std::vector<unsigned char>::iterator& pattIt,
                                              std::vector<o2::BaseCluster<double>>& outputLocalClusters,
                                              std::vector<o2::BaseCluster<double>>& outputGlobalClusters)
{
  // use this version of convertCompactClusters() in a macro

  if (mDictionary == nullptr) {
    LOGF(error,
         "AlignPointHelper::convertCompactClusters() - no dictionary found !");
    return;
  }
  if (mGeometry == nullptr) {
    mGeometry = o2::mft::GeometryTGeo::Instance();
  }
  mGeometry->fillMatrixCache(
    o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L,
                             o2::math_utils::TransformType::L2G));
  outputLocalClusters.clear();
  outputGlobalClusters.clear();
  // inspired from Detectors/ITSMFT/MFT/tracking/src/IOUtils.cxx
  for (auto& mftCluster : clusters) {
    auto chipID = mftCluster.getChipID();
    auto pattID = mftCluster.getPatternID();
    // Dummy COG errors (about half pixel size)
    double sigmaX = o2::mft::ioutils::DefClusErrorRow;
    double sigmaZ = o2::mft::ioutils::DefClusErrorCol;

    o2::math_utils::Point3D<double> locXYZ;
    if (pattID != o2::itsmft::CompCluster::InvalidPatternID) {
      // ALPIDE local Y coordinate => MFT global X coordinate (ALPIDE rows)
      sigmaX = mDictionary->getErrX(pattID);
      // ALPIDE local Z coordinate => MFT global Y coordinate (ALPIDE columns)
      sigmaZ = mDictionary->getErrZ(pattID);

      if (!mDictionary->isGroup(pattID)) {
        locXYZ = mDictionary->getClusterCoordinates(mftCluster);
      } else {
        o2::itsmft::ClusterPattern cPattern(pattIt);
        locXYZ = mDictionary->getClusterCoordinates(mftCluster, cPattern);
      }
    } else {
      o2::itsmft::ClusterPattern cPattern(pattIt);
      locXYZ = mDictionary->getClusterCoordinates(mftCluster, cPattern, false);
    }
    auto gloXYZ = mGeometry->getMatrixL2G(chipID) * locXYZ;

    auto& locCl3d = outputLocalClusters.emplace_back(chipID, locXYZ); // local
    locCl3d.setErrors(sigmaX, o2::itsmft::SegmentationAlpide::SensorLayerThicknessEff * 0.5, sigmaZ);

    auto& gloCl3d = outputGlobalClusters.emplace_back(chipID, gloXYZ); // global
    gloCl3d.setErrors(sigmaX, sigmaZ, o2::itsmft::SegmentationAlpide::SensorLayerThicknessEff * 0.5);
  }
  LOGF(debug,
       "AlignPointHelper::convertCompactClusters() - output vector size %d",
       outputLocalClusters.size());
}

//__________________________________________________________________________
void AlignPointHelper::recordTrackInitialParam(o2::mft::TrackMFT& mftTrack)
{
  mIsTrackInitialParamSet = false;
  mTrackInitialParam.X0 = mftTrack.getX();
  mTrackInitialParam.Y0 = mftTrack.getY();
  mTrackInitialParam.Z0 = mftTrack.getZ();
  double phi = mftTrack.getPhi();
  double tanLambda = mftTrack.getTanl();
  mTrackInitialParam.Tx = std::cos(phi) / tanLambda;
  mTrackInitialParam.Ty = std::sin(phi) / tanLambda;
  LOGF(debug,
       "AlignPointHelper::recordTrackInitialParam - x0 = %.3e, y0 = %.3e, z0 = %.3e, Tx = %.3e, Ty = %.3e",
       mTrackInitialParam.X0, mTrackInitialParam.Y0, mTrackInitialParam.Z0,
       mTrackInitialParam.Tx, mTrackInitialParam.Ty);

  mIsTrackInitialParamSet = true;
}

//__________________________________________________________________________
void AlignPointHelper::setGlobalRecoPosition(o2::mft::TrackMFT& mftTrack)
{
  mIsAlignPointSet = false;
  LOGF(debug,
       "AlignPointHelper::setGlobalRecoPosition() - track x = %.3e, y = %.3e, z = %.3e",
       mftTrack.getX(), mftTrack.getY(), mftTrack.getZ());
  mGlobalRecoPosition.SetXYZ(mftTrack.getX(), mftTrack.getY(), mftTrack.getZ());
  mIsAlignPointSet = true;
  if (isnan(mGlobalRecoPosition.X()) || isnan(mGlobalRecoPosition.Y()) || isnan(mGlobalRecoPosition.Z())) {
    LOGF(error,
         "AlignPointHelper::setGlobalRecoPosition() - track x = %.3e, y = %.3e, z = %.3e, point x = %.3e, y = %.3e, z = %.3e",
         mftTrack.getX(), mftTrack.getY(), mftTrack.getZ(),
         mGlobalRecoPosition.X(), mGlobalRecoPosition.Y(), mGlobalRecoPosition.Z());
    mIsAlignPointSet = false;
  }
}

//__________________________________________________________________________
void AlignPointHelper::setMeasuredPosition(const o2::BaseCluster<double>& localCluster,
                                           const o2::BaseCluster<double>& globalCluster)
{
  auto chipID = localCluster.getSensorID();

  mIsClusterOk = true;

  mLocalMeasuredPosition.SetXYZ(
    localCluster.getX(), localCluster.getY(), localCluster.getZ());
  if (isnan(mLocalMeasuredPosition.X()) || isnan(mLocalMeasuredPosition.Y()) || isnan(mLocalMeasuredPosition.Z())) {
    LOGF(error,
         "AlignPointHelper::setMeasuredPosition() - sr %4d local x = %.3e, y = %.3e, z = %.3e",
         chipID,
         mLocalMeasuredPosition.X(), mLocalMeasuredPosition.Y(), mLocalMeasuredPosition.Z());
    mIsClusterOk = false;
    return;
  }

  mGlobalMeasuredPosition.SetXYZ(
    globalCluster.getX(), globalCluster.getY(), globalCluster.getZ());
  if (isnan(mGlobalMeasuredPosition.X()) || isnan(mGlobalMeasuredPosition.Y()) || isnan(mGlobalMeasuredPosition.Z())) {
    LOGF(error,
         "AlignPointHelper::setMeasuredPosition() - sr %4d global x = %.3e, y = %.3e, z = %.3e",
         chipID,
         mGlobalMeasuredPosition.X(), mGlobalMeasuredPosition.Y(), mGlobalMeasuredPosition.Z());
    mIsClusterOk = false;
    return;
  }

  mIsAlignPointSet &= mChipHelper->setSensor(chipID);
}

//__________________________________________________________________________
void AlignPointHelper::setLocalResidual()
{
  if (mGeometry == nullptr) {
    mGeometry = o2::mft::GeometryTGeo::Instance();
  }

  if (mIsAlignPointSet) {
    mGeometry->fillMatrixCache(
      o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L,
                               o2::math_utils::TransformType::L2G));
    mLocalRecoPosition = mGeometry->getMatrixL2G(getSensorId()).ApplyInverse(mGlobalRecoPosition);
    mLocalResidual.SetXYZ(
      mLocalMeasuredPosition.X() - mLocalRecoPosition.X(),
      mLocalMeasuredPosition.Y() - mLocalRecoPosition.Y(),
      mLocalMeasuredPosition.Z() - mLocalRecoPosition.Z());
  } else {
    LOGF(error,
         "AlignPointHelper::setLocalResidual() - no align point coordinates set !");
  }
}

//__________________________________________________________________________
void AlignPointHelper::setGlobalResidual()
{
  if (mIsAlignPointSet) {
    mGlobalResidual.SetXYZ(
      mGlobalMeasuredPosition.X() - mGlobalRecoPosition.X(),
      mGlobalMeasuredPosition.Y() - mGlobalRecoPosition.Y(),
      mGlobalMeasuredPosition.Z() - mGlobalRecoPosition.Z());
  } else {
    LOGF(error,
         "AlignPointHelper::setGlobalResidual() - no align point coordinates set !");
  }
}

//__________________________________________________________________________
void AlignPointHelper::resetLocalDerivatives()
{
  mLocalDerivativeX.reset();
  mLocalDerivativeY.reset();
  mLocalDerivativeZ.reset();

  mIsLocalDerivativeDone = false;
}

//__________________________________________________________________________
void AlignPointHelper::resetGlobalDerivatives()
{
  mGlobalDerivativeX.reset();
  mGlobalDerivativeY.reset();
  mGlobalDerivativeZ.reset();

  mIsGlobalDerivativeDone = false;
}

//__________________________________________________________________________
bool AlignPointHelper::computeLocalDerivativeX()
{
  if (mChipHelper->isTransformExtracted()) {
    mLocalDerivativeX.mdX0 = mChipHelper->cosRy() * mChipHelper->cosRz();

    mLocalDerivativeX.mdTx = (mGlobalRecoPosition.Z() - mTrackInitialParam.Z0) *
                             mChipHelper->cosRy() * mChipHelper->cosRz();

    mLocalDerivativeX.mdY0 = (mChipHelper->sinRx() * mChipHelper->sinRy() * mChipHelper->cosRz()) +
                             (mChipHelper->cosRx() * mChipHelper->sinRz());

    mLocalDerivativeX.mdTy = (mGlobalRecoPosition.Z() - mTrackInitialParam.Z0) *
                             ((mChipHelper->sinRx() * mChipHelper->sinRy() * mChipHelper->cosRz()) +
                              (mChipHelper->cosRx() * mChipHelper->sinRz()));
    LOGF(debug,
         "computeLocalDerivativeX(): dX0 = %.3e, dTx = %.3e, dY0 = %.3e, dTy = %.3e",
         mLocalDerivativeX.mdX0,
         mLocalDerivativeX.mdTx,
         mLocalDerivativeX.mdY0,
         mLocalDerivativeX.mdTy);
    return true;
  } else {
    LOGF(error,
         "AlignPointHelper::computeLocalDerivativeX() - no sensor transform found !");
    return false;
  }
}

//__________________________________________________________________________
bool AlignPointHelper::computeLocalDerivativeY()
{
  if (mChipHelper->isTransformExtracted()) {
    mLocalDerivativeY.mdX0 = (-1.) * mChipHelper->cosRy() * mChipHelper->sinRz();

    mLocalDerivativeY.mdTx = (-1.) * (mGlobalRecoPosition.Z() - mTrackInitialParam.Z0) *
                             mChipHelper->cosRy() * mChipHelper->sinRz();

    mLocalDerivativeY.mdY0 = (mChipHelper->cosRx() * mChipHelper->cosRz()) -
                             (mChipHelper->sinRx() * mChipHelper->sinRy() * mChipHelper->sinRz());

    mLocalDerivativeY.mdTy = (mGlobalRecoPosition.Z() - mTrackInitialParam.Z0) *
                             ((mChipHelper->cosRx() * mChipHelper->cosRz()) -
                              (mChipHelper->sinRx() * mChipHelper->sinRy() * mChipHelper->sinRz()));
    LOGF(debug,
         "computeLocalDerivativeY(): dX0 = %.3e, dTx = %.3e, dY0 = %.3e, dTy = %.3e",
         mLocalDerivativeY.mdX0,
         mLocalDerivativeY.mdTx,
         mLocalDerivativeY.mdY0,
         mLocalDerivativeY.mdTy);
    return true;
  } else {
    LOGF(error,
         "AlignPointHelper::computeLocalDerivativeY() - no sensor transform found !");
    return false;
  }
}

//__________________________________________________________________________
bool AlignPointHelper::computeLocalDerivativeZ()
{
  if (mChipHelper->isTransformExtracted()) {
    mLocalDerivativeZ.mdX0 = mChipHelper->sinRy();

    mLocalDerivativeZ.mdTx = (mGlobalRecoPosition.Z() - mTrackInitialParam.Z0) * mChipHelper->sinRy();

    mLocalDerivativeZ.mdY0 = (-1.) * mChipHelper->sinRx() * mChipHelper->cosRy();

    mLocalDerivativeZ.mdTy = (-1.) * (mGlobalRecoPosition.Z() - mTrackInitialParam.Z0) * mChipHelper->sinRx() * mChipHelper->cosRy();
    LOGF(debug,
         "computeLocalDerivativeZ(): dX0 = %.3e, dTx = %.3e, dY0 = %.3e, dTy = %.3e",
         mLocalDerivativeZ.mdX0,
         mLocalDerivativeZ.mdTx,
         mLocalDerivativeZ.mdY0,
         mLocalDerivativeZ.mdTy);
    return true;
  } else {
    LOGF(error,
         "AlignPointHelper::computeLocalDerivativeZ() - no sensor transform found !");
    return false;
  }
}

//__________________________________________________________________________
bool AlignPointHelper::computeGlobalDerivativeX()
{
  if (mChipHelper->isTransformExtracted()) {
    mGlobalDerivativeX.mdDeltaX = (-1.) * mChipHelper->cosRy() * mChipHelper->cosRz();

    mGlobalDerivativeX.mdDeltaY = (-1) * ((mChipHelper->sinRx() * mChipHelper->sinRy() * mChipHelper->cosRz()) +
                                          (mChipHelper->cosRx() * mChipHelper->sinRz()));

    mGlobalDerivativeX.mdDeltaZ = (mTrackInitialParam.Tx * mChipHelper->cosRy() * mChipHelper->cosRz()) +
                                  (mTrackInitialParam.Ty * ((mChipHelper->sinRx() * mChipHelper->sinRy() * mChipHelper->cosRz()) +
                                                            (mChipHelper->cosRx() * mChipHelper->sinRz())));

    mGlobalDerivativeX.mdDeltaRz = ((-1.) * mChipHelper->cosRy() * mChipHelper->sinRz() *
                                    (mGlobalRecoPosition.X() - mChipHelper->translateX())) +
                                   ((mGlobalRecoPosition.Y() - mChipHelper->translateY()) *
                                    ((mChipHelper->cosRx() * mChipHelper->cosRz()) -
                                     (mChipHelper->sinRx() * mChipHelper->sinRy() * mChipHelper->sinRz())));
    LOGF(debug,
         "computeGlobalDerivativeX(): dx = %.3e, dy = %.3e, dz = %.3e, dRz = %.3e",
         mGlobalDerivativeX.mdDeltaX,
         mGlobalDerivativeX.mdDeltaY,
         mGlobalDerivativeX.mdDeltaZ,
         mGlobalDerivativeX.mdDeltaRz);
    return true;
  } else {
    LOGF(error,
         "AlignPointHelper::computeGlobalDerivativeX() - no sensor transform found !");
    return false;
  }
}

//__________________________________________________________________________
bool AlignPointHelper::computeGlobalDerivativeY()
{
  if (mChipHelper->isTransformExtracted()) {
    mGlobalDerivativeY.mdDeltaX = mChipHelper->cosRy() * mChipHelper->sinRz();

    mGlobalDerivativeY.mdDeltaY = (mChipHelper->sinRx() * mChipHelper->sinRy() * mChipHelper->sinRz()) -
                                  (mChipHelper->cosRx() * mChipHelper->cosRz());

    mGlobalDerivativeY.mdDeltaZ = ((-1.) * mTrackInitialParam.Tx * mChipHelper->cosRy() * mChipHelper->sinRz()) +
                                  (mTrackInitialParam.Ty *
                                   ((mChipHelper->cosRx() * mChipHelper->cosRz()) -
                                    (mChipHelper->sinRx() * mChipHelper->sinRy() * mChipHelper->sinRz())));

    mGlobalDerivativeY.mdDeltaRz = ((-1.) * (mGlobalRecoPosition.X() - mChipHelper->translateX()) * mChipHelper->cosRy() * mChipHelper->cosRz()) -
                                   ((mGlobalRecoPosition.Y() - mChipHelper->translateY()) *
                                    ((mChipHelper->cosRx() * mChipHelper->sinRz()) +
                                     (mChipHelper->sinRx() * mChipHelper->sinRy() + mChipHelper->cosRz())));
    LOGF(debug,
         "computeGlobalDerivativeY(): dx = %.3e, dy = %.3e, dz = %.3e, dRz = %.3e",
         mGlobalDerivativeY.mdDeltaX,
         mGlobalDerivativeY.mdDeltaY,
         mGlobalDerivativeY.mdDeltaZ,
         mGlobalDerivativeY.mdDeltaRz);
    return true;
  } else {
    LOGF(error,
         "AlignPointHelper::computeGlobalDerivativeY() - no sensor transform found !");
    return false;
  }
}

//__________________________________________________________________________
bool AlignPointHelper::computeGlobalDerivativeZ()
{
  if (mChipHelper->isTransformExtracted()) {
    mGlobalDerivativeZ.mdDeltaX = (-1.) * mChipHelper->sinRy();

    mGlobalDerivativeZ.mdDeltaY = mChipHelper->sinRx() * mChipHelper->cosRy();

    mGlobalDerivativeZ.mdDeltaZ = (mTrackInitialParam.Tx * mChipHelper->sinRy()) -
                                  (mTrackInitialParam.Ty * mChipHelper->sinRx() * mChipHelper->cosRy());

    mGlobalDerivativeZ.mdDeltaRz = 0;
    LOGF(debug,
         "computeGlobalDerivativeZ(): dx = %.3e, dy = %.3e, dz = %.3e, dRz = %.3e",
         mGlobalDerivativeZ.mdDeltaX,
         mGlobalDerivativeZ.mdDeltaY,
         mGlobalDerivativeZ.mdDeltaZ,
         mGlobalDerivativeZ.mdDeltaRz);
    return true;
  } else {
    LOGF(error,
         "AlignPointHelper::computeGlobalDerivativeZ() - no sensor transform found !");
    return false;
  }
}
