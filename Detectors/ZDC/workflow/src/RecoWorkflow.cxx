// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   RecoWorkflow.cxx

#include "ZDCWorkflow/RecoWorkflow.h"
#include "ZDCWorkflow/DigitReaderSpec.h"
#include "ZDCWorkflow/ZDCRecoWriterDPLSpec.h"
#include "ZDCWorkflow/DigitRecoSpec.h"

namespace o2
{
namespace zdc
{

framework::WorkflowSpec getRecoWorkflow(const bool useMC, const bool disableRootInp, const bool disableRootOut, const int verbosity, const bool enableDebugOut, const std::string ccdbURL)
{
  framework::WorkflowSpec specs;
  if (!disableRootInp) {
    specs.emplace_back(o2::zdc::getDigitReaderSpec(useMC));
  }
  specs.emplace_back(o2::zdc::getDigitRecoSpec(verbosity, enableDebugOut, ccdbURL));
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
