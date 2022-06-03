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

/// \file TPCFastTransformHelperO2.h
/// \brief class to create TPC fast transformation
/// \author Sergey Gorbunov
///
/// Usage:
///
///  std::unique_ptr<TPCFastTransform> fastTransform = TPCFastTransformHelperO2::instance()->create( 0 );
///

#ifndef ALICEO2_TPC_TPCFASTTRANSFORMHELPERO2_H_
#define ALICEO2_TPC_TPCFASTTRANSFORMHELPERO2_H_

#include "TPCFastTransform.h"
#include "Rtypes.h"
#include <functional>
#include "TPCFastSpaceChargeCorrectionMap.h"

namespace o2
{
namespace tpc
{

using namespace o2::gpu;

class TPCFastTransformHelperO2
{
 public:
  /// _____________  Constructors / destructors __________________________

  /// Default constructor
  TPCFastTransformHelperO2() = default;

  /// Copy constructor: disabled
  TPCFastTransformHelperO2(const TPCFastTransformHelperO2&) = delete;

  /// Assignment operator: disabled
  TPCFastTransformHelperO2& operator=(const TPCFastTransformHelperO2&) = delete;

  /// Destructor
  ~TPCFastTransformHelperO2() = default;

  /// Singleton
  static TPCFastTransformHelperO2* instance();

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

  /// creates TPCFastTransform object
  std::unique_ptr<TPCFastTransform> create(Long_t TimeStamp);

  /// Updates the transformation with the new time stamp
  int updateCalibration(TPCFastTransform& transform, Long_t TimeStamp);

  /// _______________  Utilities   ________________________

  const TPCFastTransformGeo& getGeometry() { return mGeo; }

  void testGeometry(const TPCFastTransformGeo& fastTransform) const;

  TPCFastSpaceChargeCorrectionMap& getCorrectionMap() { return mCorrectionMap; }

 private:
  /// initialization
  void init();

  /// get space charge correction in internal TPCFastTransform coordinates u,v->dx,du,dv
  void getSpaceChargeCorrection(int slice, int row, o2::gpu::TPCFastSpaceChargeCorrectionMap::CorrectionPoint p, double& su, double& sv, double& dx, double& du, double& dv);

  static TPCFastTransformHelperO2* sInstance; ///< singleton instance
  bool mIsInitialized = 0;                    ///< initialization flag
  TPCFastTransformGeo mGeo;                   ///< geometry parameters

  TPCFastSpaceChargeCorrectionMap mCorrectionMap{0, 0};

  ClassDefNV(TPCFastTransformHelperO2, 3);
};
} // namespace tpc
} // namespace o2
#endif
