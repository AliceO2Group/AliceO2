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

  /// _______________  Main functionality  ________________________

  /// set space charge correction in the global coordinates
  /// as a continious function
  void setGlobalSpaceChargeCorrection(
    std::function<void(int roc, double gx, double gy, double gz,
                       double& dgx, double& dgy, double& dgz)>
      correctionGlobal);

  template <typename F>
  void setGlobalSpaceChargeCorrection(F&& correctionGlobal)
  {
    std::function<void(int roc, double gx, double gy, double gz,
                       double& dgx, double& dgy, double& dgz)>
      f = correctionGlobal;
    setGlobalSpaceChargeCorrection(f);
  }

  /// set space charge correction in the local coordinates
  /// as a continious function
  template <typename F>
  void setLocalSpaceChargeCorrection(F&& correctionLocal)
  {
    std::function<void(int roc, int irow, double y, double z, double& dx, double& dy, double& dz)> f = correctionLocal;
    setLocalSpaceChargeCorrection(f);
  }

  /// set space charge correction in the local coordinates
  /// as a continious function
  void setLocalSpaceChargeCorrection(
    std::function<void(int roc, int irow, double y, double z, double& dx, double& dy, double& dz)> correctionLocal);

  /// creates TPCFastSpaceChargeCorrection object
  std::unique_ptr<TPCFastSpaceChargeCorrection> create(const int nKnotsY = 10, const int nKnotsZ = 20);

  /// _______________  Utilities   ________________________

  const TPCFastTransformGeo& getGeometry() { return mGeo; }

  TPCFastSpaceChargeCorrectionMap& getCorrectionMap() { return mCorrectionMap; }

  void fillSpaceChargeCorrectionFromMap(TPCFastSpaceChargeCorrection& correction);

  void testGeometry(const TPCFastTransformGeo& geo) const;

  /// Create SpaceCharge correction out of the voxel tree
  std::unique_ptr<o2::gpu::TPCFastSpaceChargeCorrection> createSpaceChargeCorrection(
    const o2::tpc::TrackResiduals& trackResiduals, TTree* voxResTree);

 private:
  /// geometry initialization
  void initGeometry();

  /// get space charge correction in internal TPCFastTransform coordinates u,v->dx,du,dv
  void getSpaceChargeCorrection(const TPCFastSpaceChargeCorrection& correction, int slice, int row, o2::gpu::TPCFastSpaceChargeCorrectionMap::CorrectionPoint p, double& su, double& sv, double& dx, double& du, double& dv);

  static TPCFastSpaceChargeCorrectionHelper* sInstance; ///< singleton instance
  bool mIsInitialized = 0;                              ///< initialization flag
  TPCFastTransformGeo mGeo;                             ///< geometry parameters

  TPCFastSpaceChargeCorrectionMap mCorrectionMap{0, 0};

  ClassDefNV(TPCFastSpaceChargeCorrectionHelper, 0);
};

} // namespace tpc
} // namespace o2
#endif
