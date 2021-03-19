// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   FT0Workflow.cxx

#include "FT0Workflow/FT0Workflow.h"
#include "FT0Workflow/FT0DataProcessDPLSpec.h"
#include "FT0Workflow/FT0DataReaderDPLSpec.h"
#include "FT0Workflow/FT0DigitWriterSpec.h"
#include "FT0Workflow/RawReaderFT0.h"
namespace o2
{
namespace ft0
{

framework::WorkflowSpec getFT0Workflow(bool isExtendedMode, bool useProcess,
                                       bool dumpProcessor, bool dumpReader,
                                       bool disableRootOut, bool disableTrgInput)
{
  LOG(INFO) << "framework::WorkflowSpec getFT0Workflow";
  framework::WorkflowSpec specs;
  if (isExtendedMode) {
    specs.emplace_back(o2::ft0::getFT0DataReaderDPLSpec(RawReaderFT0ext{dumpReader}));
  } else {
    if (disableTrgInput) {
      specs.emplace_back(o2::ft0::getFT0DataReaderDPLSpec(RawReaderFT0<false>{dumpReader}));
    } else {
      specs.emplace_back(o2::ft0::getFT0DataReaderDPLSpec(RawReaderFT0<true>{dumpReader}));
    }
  }
  if (useProcess) {
    specs.emplace_back(o2::ft0::getFT0DataProcessDPLSpec(dumpProcessor));
  }
  if (!disableRootOut) {
    specs.emplace_back(o2::ft0::getFT0DigitWriterSpec(false, !disableTrgInput));
  }
  return specs;
}

} // namespace ft0
} // namespace o2
