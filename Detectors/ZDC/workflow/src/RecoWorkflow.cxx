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

/// @file   RecoWorkflow.cxx

#include "ZDCWorkflow/RecoWorkflow.h"
#include "ZDCWorkflow/DigitReaderSpec.h"
#include "ZDCWorkflow/ZDCRecoWriterDPLSpec.h"
#include "ZDCWorkflow/DigitRecoSpec.h"
#include "ZDCCalib/BaselineCalibEPNSpec.h"

namespace o2
{
namespace zdc
{

framework::WorkflowSpec getRecoWorkflow(const bool useMC, const bool disableRootInp, const bool disableRootOut, const int verbosity, const bool enableDebugOut,
                                        const bool enableZDCTDCCorr, const bool enableZDCEnergyParam, const bool enableZDCTowerParam, const bool enableBaselineParam)
{
  framework::WorkflowSpec specs;
  if (!disableRootInp) {
    specs.emplace_back(o2::zdc::getDigitReaderSpec(useMC));
  }
  specs.emplace_back(o2::zdc::getDigitRecoSpec(verbosity, enableDebugOut, enableZDCTDCCorr, enableZDCEnergyParam, enableZDCTowerParam, enableBaselineParam));
  specs.emplace_back(o2::zdc::getBaselineCalibEPNSpec());
  if (!disableRootOut) {
    specs.emplace_back(o2::zdc::getZDCRecoWriterDPLSpec());
  }
  //   specs.emplace_back(o2::zdc::getReconstructionSpec(useMC));
  //   if (!disableRootOut) {
  //     specs.emplace_back(o2::zdc::getRecPointWriterSpec(useMC));
  //   }
  return specs;
}

} // namespace zdc
} // namespace o2
