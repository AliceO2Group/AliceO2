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

#include "FT0Workflow/RecoWorkflow.h"

#include "FT0Workflow/DigitReaderSpec.h"
#include "FT0Workflow/RecPointWriterSpec.h"
#include "FT0Workflow/ReconstructionSpec.h"

namespace o2
{
namespace fit
{

framework::WorkflowSpec getRecoWorkflow(bool useMC, bool disableRootInp, bool disableRootOut)
{
  framework::WorkflowSpec specs;
  if (!disableRootInp) {
    specs.emplace_back(o2::ft0::getDigitReaderSpec(useMC));
  }
  specs.emplace_back(o2::ft0::getReconstructionSpec(useMC));
  if (!disableRootOut) {
    specs.emplace_back(o2::ft0::getRecPointWriterSpec(useMC));
  }
  return specs;
}

} // namespace fit
} // namespace o2
