// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

namespace o2
{
namespace TPC
{

using namespace o2::gpu;

class TPCFastTransformHelperO2
{
 public:
  /// _____________  Constructors / destructors __________________________

  /// Default constructor
  TPCFastTransformHelperO2();

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

  /// Updates the transformation with the new time stamp
  int updateCalibration(TPCFastTransform& transform, Long_t TimeStamp);

  /// _______________  Utilities   ________________________

  void testGeometry(const TPCFastTransform& fastTransform) const;

 private:
  /// initialization
  void init();

  static TPCFastTransformHelperO2* sInstance; // singleton instance
  bool mIsInitialized;                        ///< initialization flag

  ClassDefNV(TPCFastTransformHelperO2, 1);
};
} // namespace TPC
} // namespace o2
#endif
