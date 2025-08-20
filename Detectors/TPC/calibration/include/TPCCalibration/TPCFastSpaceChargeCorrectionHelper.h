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

/// \file TPCFastSpaceChargeCorrectionHelper.h
/// \brief class to create TPC fast space charge correction
/// \author Sergey Gorbunov
///
/// Usage:
///
///  std::unique_ptr<TPCFastTransform> fastTransform = TPCFastSpaceChargeCorrectionHelper::instance()->create( 0 );
///

#ifndef ALICEO2_TPC_TPCFastSpaceChargeCorrectionHelper_H_
#define ALICEO2_TPC_TPCFastSpaceChargeCorrectionHelper_H_

#include "Rtypes.h"
#include <functional>
#include "TPCFastSpaceChargeCorrectionMap.h"
#include "TPCFastSpaceChargeCorrection.h"
#include "TPCFastTransformGeo.h"
#include "SpacePoints/TrackResiduals.h"

class TTree;

namespace o2
{
namespace tpc
{

class TrackResiduals;

using namespace o2::gpu;

class TPCFastSpaceChargeCorrectionHelper
{
 public:
  /// _____________  Constructors / destructors __________________________

  /// Default constructor
  TPCFastSpaceChargeCorrectionHelper() = default;

  /// Copy constructor: disabled
  TPCFastSpaceChargeCorrectionHelper(const TPCFastSpaceChargeCorrectionHelper&) = delete;

  /// Assignment operator: disabled
  TPCFastSpaceChargeCorrectionHelper& operator=(const TPCFastSpaceChargeCorrectionHelper&) = delete;

  /// Destructor
  ~TPCFastSpaceChargeCorrectionHelper() = default;

  /// Singleton
  static TPCFastSpaceChargeCorrectionHelper* instance();

  /// _______________  Settings   ________________________

  /// sets number of threads to use
  void setNthreads(int n);

  /// sets number of threads to N cpu cores
  void setNthreadsToMaximum();

  /// get the number of threads
  int getNthreads() const { return mNthreads; }

  /// _______________  Main functionality  ________________________

  /// creates TPCFastSpaceChargeCorrection object from a continious space charge correction in local coordinates
  std::unique_ptr<TPCFastSpaceChargeCorrection> createFromLocalCorrection(
    std::function<void(int roc, int irow, double y, double z, double& dx, double& dy, double& dz)> correctionLocal,
    const int nKnotsY = 10, const int nKnotsZ = 20);

  /// creates TPCFastSpaceChargeCorrection object from a continious space charge correction in global coordinates
  std::unique_ptr<TPCFastSpaceChargeCorrection> createFromGlobalCorrection(
    std::function<void(int roc, double gx, double gy, double gz,
                       double& dgx, double& dgy, double& dgz)>
      correctionGlobal,
    const int nKnotsY = 10, const int nKnotsZ = 20);

  /// Create SpaceCharge correction out of the voxel tree
  /// \param trackResiduals TrackResiduals object
  /// \param voxResTree TTree with voxel residuals
  /// \param voxResTreeInverse TTree with inverse voxel residuals
  /// \param useSmoothed if true, use smoothed residuals
  /// \param invertSigns if true, invert the signs of the residuals
  /// \param fitPointsDirect debug: pointer to the data used for the direct correction
  /// \param fitPointsInverse debug: pointer to the data used for the inverse correction
  /// \return pointer to the created TPCFastSpaceChargeCorrection object
  /// \note voxel trees wont be changed. They are read as non-const because of the ROOT::TTreeProcessorMT interface
  ///
  std::unique_ptr<o2::gpu::TPCFastSpaceChargeCorrection> createFromTrackResiduals(
    const o2::tpc::TrackResiduals& trackResiduals, TTree* voxResTree, TTree* voxResTreeInverse, //
    bool useSmoothed, bool invertSigns,                                                         //
    TPCFastSpaceChargeCorrectionMap* fitPointsDirect = nullptr,
    TPCFastSpaceChargeCorrectionMap* fitPointsInverse = nullptr);

  /// _______________  Utilities   ________________________

  const TPCFastTransformGeo& getGeometry() { return mGeo; }

  TPCFastSpaceChargeCorrectionMap& getCorrectionMap() { return mCorrectionMap; }

  void testGeometry(const TPCFastTransformGeo& geo) const;

  /// initialise inverse transformation
  void initInverse(o2::gpu::TPCFastSpaceChargeCorrection& correction, bool prn);

  /// initialise inverse transformation from linear combination of several input corrections
  void initInverse(std::vector<o2::gpu::TPCFastSpaceChargeCorrection*>& corrections, const std::vector<float>& scaling, bool prn);

  /// merge several corrections
  /// \param mainCorrection main correction
  /// \param scale scaling factor for the main correction
  /// \param additionalCorrections vector of pairs of additional corrections and their scaling factors
  /// \param prn printout flag
  /// \return main correction merged with additional corrections
  void mergeCorrections(
    o2::gpu::TPCFastSpaceChargeCorrection& mainCorrection, float scale,
    const std::vector<std::pair<const o2::gpu::TPCFastSpaceChargeCorrection*, float>>& additionalCorrections, bool prn);

  /// how far the voxel mean is allowed to be outside of the voxel (1.1 means 10%)
  void setVoxelMeanValidityRange(double range)
  {
    mVoxelMeanValidityRange = range;
  }

  double getVoxelMeanValidityRange() const { return mVoxelMeanValidityRange; }

  /// debug: if true, use voxel centers instead of the fitted positions for correction
  void setDebugUseVoxelCenters();

  bool isDebugUseVoxelCenters() const { return mDebugUseVoxelCenters; }

  /// debug: if true, mirror the data from the A side to the C side of the TPC
  void setDebugMirrorAdata2C();

  bool isDebugMirrorAdata2C() const { return mDebugMirrorAdata2C; }

 private:
  /// geometry initialization
  void initGeometry();

  void fillSpaceChargeCorrectionFromMap(TPCFastSpaceChargeCorrection& correction, bool processingInverseCorrection);

  static TPCFastSpaceChargeCorrectionHelper* sInstance; ///< singleton instance
  bool mIsInitialized = 0;                              ///< initialization flag
  int mNthreads = 1;                                    ///< n of threads to use
  TPCFastTransformGeo mGeo;                             ///< geometry parameters

  TPCFastSpaceChargeCorrectionMap mCorrectionMap{0, 0};

  double mVoxelMeanValidityRange{1.1}; ///< debug: how far the voxel mean is allowed to be outside of the voxel (1.1 means 10%)

  bool mDebugUseVoxelCenters{false}; ///< debug: if true, use voxel centers instead of the fitted positions for correction
  bool mDebugMirrorAdata2C{false};   ///< debug: if true, mirror the data from the A side to the C side of the TPC

  ClassDefNV(TPCFastSpaceChargeCorrectionHelper, 0);
};

} // namespace tpc
} // namespace o2
#endif
