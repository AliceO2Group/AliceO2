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

  /// creates TPCFastTransform object
  std::unique_ptr<TPCFastTransform> create(Long_t TimeStamp);

  /// creates TPCFastTransform object
  std::unique_ptr<TPCFastTransform> create(Long_t TimeStamp, const TPCFastSpaceChargeCorrection& correction);

  /// Updates the transformation with the new time stamp
  int updateCalibration(TPCFastTransform& transform, Long_t TimeStamp, float vDriftFactor = 1.f, float vDriftRef = 0.f, float driftTimeOffset = 0.f);

  /// _______________  Utilities   ________________________

  const TPCFastTransformGeo& getGeometry() { return mGeo; }

  void testGeometry(const TPCFastTransformGeo& fastTransform) const;

 private:
  /// initialization
  void init();

  static TPCFastTransformHelperO2* sInstance; ///< singleton instance
  bool mIsInitialized = 0;                    ///< initialization flag
  TPCFastTransformGeo mGeo;                   ///< geometry parameters

  ClassDefNV(TPCFastTransformHelperO2, 3);
};
} // namespace tpc
} // namespace o2
#endif
