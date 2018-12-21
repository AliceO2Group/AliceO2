// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file  TPCFastTransformManager.h
/// \brief Definition of TPCFastTransformManager class
///
/// \author  Sergey Gorbunov <sergey.gorbunov@cern.ch>


#ifndef ALICE_ALITPCOMMON_TPCFASTTRANSFORMATION_TPCFASTTRANSFORMQA_H
#define ALICE_ALITPCOMMON_TPCFASTTRANSFORMATION_TPCFASTTRANSFORMQA_H

#include "AliTPCCommonDef.h"
#include "TPCFastTransformManager.h"
#include <cmath>
#include <iostream>

#include "Rtypes.h"
#include "TString.h"
#include "AliTPCTransform.h"

namespace ali_tpc_common {
namespace tpc_fast_transformation {

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
  TPCFastTransformQA(const TPCFastTransformQA& ) CON_DELETE;
 
  /// Assignment operator: disabled 
  TPCFastTransformQA &operator=(const TPCFastTransformQA &) CON_DELETE;
     
  /// Destructor
  ~TPCFastTransformQA() CON_DEFAULT;

  /// _______________  Main functionality  ________________________

  /// create fast transformation and perform a quality check
  int doQA( Long_t TimeStamp );
 
  /// create perform quality check
  int doQA( const TPCFastTransform &fastTransform );

 private:
  
  /// Stores an error message
  int storeError(Int_t code, const char *msg);
  TString mError; ///< error string

};

inline int TPCFastTransformQA::storeError(int code, const char *msg)
{
  mError = msg;
  std::cout<<msg<<std::endl;
  return code;
}

}} // namespaces

#endif
