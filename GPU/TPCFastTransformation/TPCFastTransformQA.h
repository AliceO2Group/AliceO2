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

#ifndef ALICEO2_GPUCOMMON_TPCFASTTRANSFORMATION_TPCFASTTRANSFORMQA_H
#define ALICEO2_GPUCOMMON_TPCFASTTRANSFORMATION_TPCFASTTRANSFORMQA_H

#include "GPUCommonDef.h"
#include "TPCFastTransformManager.h"
#include "GPUCommonLogger.h"

#include <cmath>
#include <iostream>

#include "Rtypes.h"
#include "TString.h"
#include "AliTPCTransform.h"

namespace GPUCA_NAMESPACE
{
namespace gpu
{

///
/// The TPCFastTransformQA class does performance check for TPCFastTransformation object
///

class TPCFastTransformQA
{
 public:
  /// _____________  Constructors / destructors __________________________

  /// Default constructor
  TPCFastTransformQA();

  /// Copy constructor: disabled
  TPCFastTransformQA(const TPCFastTransformQA&) CON_DELETE;

  /// Assignment operator: disabled
  TPCFastTransformQA& operator=(const TPCFastTransformQA&) CON_DELETE;

  /// Destructor
  ~TPCFastTransformQA() CON_DEFAULT;

  /// _______________  Main functionality  ________________________

  /// create fast transformation and perform a quality check
  int doQA(Long_t TimeStamp);

  /// create perform quality check
  int doQA(const TPCFastTransform& fastTransform);

 private:
  /// Stores an error message
  int storeError(Int_t code, const char* msg);
  TString mError; ///< error string
};

inline int TPCFastTransformQA::storeError(int code, const char* msg)
{
  mError = msg;
  LOG(info) << msg;
  return code;
}
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
