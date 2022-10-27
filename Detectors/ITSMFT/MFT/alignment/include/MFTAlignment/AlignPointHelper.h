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

/// \file AlignPointHelper.h
/// \author arakotoz@cern.ch
/// \brief Compute the local and global derivatives at an alignment point (track position, cluster position)

#ifndef ALICEO2_MFT_ALIGN_POINT_HELPER_H
#define ALICEO2_MFT_ALIGN_POINT_HELPER_H

#include <gsl/gsl>

#include "Framework/ProcessingContext.h"
#include "MathUtils/Cartesian.h"
#include "MFTBase/Geometry.h"
#include "MFTBase/GeometryTGeo.h"
#include "ReconstructionDataFormats/BaseCluster.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITSMFT/TopologyDictionary.h"
#include "MFTAlignment/AlignSensorHelper.h"
#include "MFTBase/GeometryTGeo.h"

namespace o2
{
namespace mft
{

class TrackMFT;

/// \class GlobalDerivative
/// \brief Simple container of global derivatives
class GlobalDerivative
{
  friend class AlignPointHelper;

 public:
  /// \brief constructor
  GlobalDerivative() = default;

  /// \brief destructor
  virtual ~GlobalDerivative() = default;

  /// \brief reset all data members to default value (zero)
  void reset()
  {
    mdDeltaX = 0.;
    mdDeltaY = 0.;
    mdDeltaZ = 0.;
    mdDeltaRz = 0.;
  }

  // simple getters

  double dDeltaX() const { return mdDeltaX; }
  double dDeltaY() const { return mdDeltaY; }
  double dDeltaZ() const { return mdDeltaZ; }
  double dDeltaRz() const { return mdDeltaRz; }

 protected:
  double mdDeltaX = 0.;  ///< derivative w.r.t. delta translation along global x-axis
  double mdDeltaY = 0.;  ///< derivative w.r.t. delta translation along global y-axis
  double mdDeltaZ = 0.;  ///< derivative w.r.t. delta translation along global z-axis
  double mdDeltaRz = 0.; ///< derivative w.r.t. delta rotation angle around global z-axis
};

/// \class LocalDerivative
/// \brief Simple container of local derivatives
class LocalDerivative
{
  friend class AlignPointHelper;

 public:
  /// \brief constructor
  LocalDerivative() = default;

  /// \brief destructor
  virtual ~LocalDerivative() = default;

  // simple getters

  double dX0() const { return mdX0; }
  double dTx() const { return mdTx; }
  double dY0() const { return mdY0; }
  double dTy() const { return mdTy; }

  /// \brief reset all data members to default value (zero)
  void reset()
  {
    mdX0 = 0.;
    mdTx = 0.;
    mdY0 = 0.;
    mdTy = 0.;
  }

 protected:
  double mdX0 = 0.; ///< derivative w.r.t. track param. x0
  double mdTx = 0.; ///< derivative w.r.t. track param. tx
  double mdY0 = 0.; ///< derivative w.r.t. track param. y0
  double mdTy = 0.; ///< derivative w.r.t. track param. ty
};

/*! \class AlignPointHelper
    \brief Container of a single alignment point and methods to fill it
    \details An alignment point is defined by the track crossing point coordinates at the z
    of this plane, the cluster coordinates, the value of the local derivarives and the global
    derivatives at that point, the sensor id of this cluster. This class also offers to
    compute the track-cluster residual at this point.
*/
class AlignPointHelper
{
 public:
  /// \brief constructor
  AlignPointHelper();

  /// \brief destructor
  virtual ~AlignPointHelper() = default;

  /// \brief simple structure to organise the storage of track parameters at inital z0 plane
  struct TrackParam {
    Double_t X0, Y0, Z0, Tx, Ty;
  };

  /// \brief method to call the computation of all three compnonents of the local derivative
  void computeLocalDerivatives();

  /// \brief method to call the computation of all three components of the global derivative
  void computeGlobalDerivatives();

  // simple getters

  UShort_t getSensorId() const;
  UShort_t half() const;
  UShort_t disk() const;
  UShort_t layer() const;

  // simple getters

  bool isAlignPointSet() const { return mIsAlignPointSet; }
  bool isGlobalDerivativeDone() const { return mIsGlobalDerivativeDone; }
  bool isLocalDerivativeDone() const { return mIsLocalDerivativeDone; }
  bool isClusterOk() const { return mIsClusterOk; }

  // simple getters

  GlobalDerivative globalDerivativeX() const { return mGlobalDerivativeX; }
  GlobalDerivative globalDerivativeY() const { return mGlobalDerivativeY; }
  GlobalDerivative globalDerivativeZ() const { return mGlobalDerivativeZ; }
  LocalDerivative localDerivativeX() const { return mLocalDerivativeX; }
  LocalDerivative localDerivativeY() const { return mLocalDerivativeY; }
  LocalDerivative localDerivativeZ() const { return mLocalDerivativeZ; }

  // simple getters

  o2::math_utils::Point3D<double> getLocalMeasuredPosition() const
  {
    return mLocalMeasuredPosition;
  }
  o2::math_utils::Point3D<double> getLocalMeasuredPositionSigma() const
  {
    return mLocalMeasuredPositionSigma;
  }
  o2::math_utils::Point3D<double> getLocalResidual() const
  {
    return mLocalResidual;
  }
  o2::math_utils::Point3D<double> getGlobalResidual() const
  {
    return mGlobalResidual;
  }
  o2::math_utils::Point3D<double> getGlobalMeasuredPosition() const
  {
    return mGlobalMeasuredPosition;
  }
  o2::math_utils::Point3D<double> getGlobalRecoPosition() const
  {
    return mGlobalRecoPosition;
  }
  o2::math_utils::Point3D<double> getLocalRecoPosition() const
  {
    return mLocalRecoPosition;
  }
  TrackParam getTrackInitialParam() const { return mTrackInitialParam; }

  /// \brief reset all quantities that define an alignment point to their default value
  void resetAlignPoint();

  /// \brief reset all track parameters to their default value (zero)
  void resetTrackInitialParam();

  /// \brief convert compact clusters (pixel coordinates in row, col) from workflow to base clusters with 3D position (local, global coordinates)
  void convertCompactClusters(gsl::span<const itsmft::CompClusterExt> clusters,
                              gsl::span<const unsigned char>::iterator& pattIt,
                              std::vector<o2::BaseCluster<double>>& outputLocalClusters,
                              std::vector<o2::BaseCluster<double>>& outputGlobalClusters);

  /// \brief convert compact clusters (pixel coordinates in row, col) from ROOT file to base clusters with 3D position (local, global coordinates)
  void convertCompactClusters(const std::vector<o2::itsmft::CompClusterExt>& clusters,
                              std::vector<unsigned char>::iterator& pattIt,
                              std::vector<o2::BaseCluster<double>>& outputLocalClusters,
                              std::vector<o2::BaseCluster<double>>& outputGlobalClusters);

  /// \brief store the track parameters at the initial z0 plane
  void recordTrackInitialParam(o2::mft::TrackMFT& mftTrack);

  /// \brief set cluster pattern dictionary (needed to compute cluster coordinates)
  void setClusterDictionary(const o2::itsmft::TopologyDictionary* d) { mDictionary = d; }

  // setters

  void setGlobalRecoPosition(o2::mft::TrackMFT& mftTrack);
  void setMeasuredPosition(const o2::BaseCluster<double>& localCluster,
                           const o2::BaseCluster<double>& globalCluster);
  void setLocalResidual();
  void setGlobalResidual();

 protected:
  bool mIsAlignPointSet;        ///< boolean to indicate if mGlobalRecoPosition and mLocalMeasuredPosition are set
  bool mIsGlobalDerivativeDone; ///< boolean to indicate if the global derivatives computaion is done
  bool mIsLocalDerivativeDone;  ///< boolean to indicate if the local derivatives computation is done
  bool mIsTrackInitialParamSet; ///< boolean to indicate if the initial track parameters are recorded
  bool mIsClusterOk;            ///< boolean to check if cluster was exploitable to get coordinates

  o2::mft::GeometryTGeo* mGeometry;                        ///< MFT geometry
  const o2::itsmft::TopologyDictionary* mDictionary;       ///< cluster patterns dictionary
  std::unique_ptr<o2::mft::AlignSensorHelper> mChipHelper; ///< utility to access the sensor transform used in the computation of the derivatives

  LocalDerivative mLocalDerivativeX; ///< first (X) component of the local derivatives
  LocalDerivative mLocalDerivativeY; ///< second (Y) component of the local derivatives
  LocalDerivative mLocalDerivativeZ; ///< last (Z) component of the local derivatives

  GlobalDerivative mGlobalDerivativeX; ///< first (X) component of the global derivatives
  GlobalDerivative mGlobalDerivativeY; ///< second (Y) component of the global derivatives
  GlobalDerivative mGlobalDerivativeZ; ///< last (Z) component of the global derivatives

  TrackParam mTrackInitialParam; ///< Track parameters at the initial reference plane z = z0

  o2::math_utils::Point3D<double> mGlobalRecoPosition; ///< Current cartesian position (cm, in Global ref. system) of the reconstructed track analytically propagated to the z position of the cluster
  o2::math_utils::Point3D<double> mLocalRecoPosition;  ///< Current cartesian position (cm, in Local ref. system) of the reconstructed track analytically propagated to the z position of the cluster

  o2::math_utils::Point3D<double> mLocalMeasuredPosition;      ///< Current cartesian position (cm, in Local ref. system) of the cluster
  o2::math_utils::Point3D<double> mLocalMeasuredPositionSigma; ///< Estimated error on local position measurement
  o2::math_utils::Point3D<double> mGlobalMeasuredPosition;     ///< Current cartesian position (cm, in Global ref. system) of the cluster

  o2::math_utils::Point3D<double> mLocalResidual;  ///< residual between track x-ing point and cluster in local ref. system
  o2::math_utils::Point3D<double> mGlobalResidual; ///< residual between track x-ing point and cluster in global ref. system

 protected:
  ///\brief reset all elements to zero for the local derivatives
  void resetLocalDerivatives();

  ///\brief reset all elements to zero for the global derivatives
  void resetGlobalDerivatives();

  /// \brief compute first (X) component of the local derivatives
  bool computeLocalDerivativeX();

  /// \brief compute second (Y) component of the local derivatives
  bool computeLocalDerivativeY();

  /// \brief compute last (Z) component of the local derivatives
  bool computeLocalDerivativeZ();

  /// \brief compute first (X) component of the global derivatives
  bool computeGlobalDerivativeX();

  /// \brief compute second (Y) component of the global derivatives
  bool computeGlobalDerivativeY();

  /// \brief compute last (Z) component of the global derivatives
  bool computeGlobalDerivativeZ();

  ClassDef(AlignPointHelper, 0);
};

} // namespace mft
} // namespace o2
#endif
