// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file TPCFastTransformManagerO2.h
/// \brief class to create TPC fast transformation
/// \author Sergey Gorbunov
///
/// Usage: 
/// 
///  TPCFastTransformManagerO2 manager;
///  TPCFastTransform fastTransform;
///  manager.create( fastTransform, 0 );
///

#ifndef ALICEO2_TPC_TPCFASTTRANSFORMMANAGERO2_H_
#define ALICEO2_TPC_TPCFASTTRANSFORMMANAGERO2_H_

#include "TPCFastTransform.h"
#include "Rtypes.h"

namespace o2
{
namespace TPC
{

using namespace ali_tpc_common::tpc_fast_transformation;

class TPCFastTransformManagerO2
{
public:

 /// _____________  Constructors / destructors __________________________
 
  /// Default constructor
  TPCFastTransformManagerO2();

  /// Copy constructor: disabled
  TPCFastTransformManagerO2(const TPCFastTransformManagerO2& ) = delete;
 
  /// Assignment operator: disabled 
  TPCFastTransformManagerO2 &operator=(const TPCFastTransformManagerO2 &) = delete;
     
  /// Destructor
  ~TPCFastTransformManagerO2() = default;

  /// _______________  Main functionality  ________________________

  /// Initializes TPCFastTransform object
  int  create( TPCFastTransform &transform, Long_t TimeStamp );

  /// Updates the transformation with the new time stamp 
  int updateCalibration( TPCFastTransform &transform, Long_t TimeStamp );
  
  /// _______________  Utilities   ________________________
 
  void testGeometry(const TPCFastTransform& fastTransform) const;

 private:

  int mLastTimeBin;                 ///< last calibrated time bin
};


}
}
#endif
