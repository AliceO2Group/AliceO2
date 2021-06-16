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

/// \file  TPCFastTransformManager.h
/// \brief Definition of TPCFastTransformManager class
///
/// \author  Sergey Gorbunov <sergey.gorbunov@cern.ch>

#ifndef ALICEO2_GPUCOMMON_TPCFASTTRANSFORMATION_TPCFASTTRANSFORMMANAGER_H
#define ALICEO2_GPUCOMMON_TPCFASTTRANSFORMATION_TPCFASTTRANSFORMMANAGER_H

#include <cmath>

#include "GPUCommonDef.h"
#include "Rtypes.h"
#include "TString.h"
#include "AliTPCTransform.h"

namespace GPUCA_NAMESPACE
{
namespace gpu
{
class TPCFastTransform;

///
/// The TPCFastTransformManager class is to initialize TPCFastTransformation object
///

class TPCFastTransformManager
{
 public:
  /// _____________  Constructors / destructors __________________________

  /// Default constructor
  TPCFastTransformManager();

  /// Copy constructor: disabled
  TPCFastTransformManager(const TPCFastTransformManager&) CON_DELETE;

  /// Assignment operator: disabled
  TPCFastTransformManager& operator=(const TPCFastTransformManager&) CON_DELETE;

  /// Destructor
  ~TPCFastTransformManager() CON_DEFAULT;

  /// _______________  Main functionality  ________________________

  /// Initializes TPCFastTransform object
  int create(TPCFastTransform& spline, AliTPCTransform* transform, Long_t TimeStamp);

  /// Updates the transformation with the new time stamp
  Int_t updateCalibration(TPCFastTransform& spline, Long_t TimeStamp);

  /// _______________  Utilities   ________________________

  AliTPCTransform* getOriginalTransform() { return mOrigTransform; }

  ///  Gives error string
  const char* getLastError() const { return mError.Data(); }

 private:
  /// Stores an error message
  int storeError(Int_t code, const char* msg);

  TString mError;                  ///< error string
  AliTPCTransform* mOrigTransform; ///< transient
  int fLastTimeBin;                ///< last calibrated time bin
};

inline int TPCFastTransformManager::storeError(int code, const char* msg)
{
  mError = msg;
  return code;
}
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
