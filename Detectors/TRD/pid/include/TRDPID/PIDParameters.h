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

// \brief Collect all possible configurable parameters for the PID task

#ifndef O2_TRD_PID_PARAMS_H
#define O2_TRD_PID_PARAMS_H

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"

namespace o2
{
namespace trd
{

/// PID parameters.
struct TRDPIDParams : public o2::conf::ConfigurableParamHelper<TRDPIDParams> {
  unsigned int numOrtThreads = 1;           ///< ONNX Session threads
  unsigned int graphOptimizationLevel = 99; ///< ONNX GraphOptimization Level
                                            /// 0=Disable All, 1=Enable Basic, 2=Enable Extended, 99=Enable ALL

  // boilerplate
  O2ParamDef(TRDPIDParams, "TRDPIDParams");
};

} // namespace trd
} // namespace o2

#endif
