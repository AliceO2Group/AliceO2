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
                                       bool disableRootOut, bool askSTFDist)
{
  LOG(INFO) << "framework::WorkflowSpec getFT0Workflow";
  framework::WorkflowSpec specs;
  if (isExtendedMode) {
    specs.emplace_back(o2::ft0::getFT0DataReaderDPLSpec(RawReaderFT0ext{dumpReader}, askSTFDist));
  } else {
    specs.emplace_back(o2::ft0::getFT0DataReaderDPLSpec(RawReaderFT0<false>{dumpReader}, askSTFDist));
  }
  if (useProcess) {
    specs.emplace_back(o2::ft0::getFT0DataProcessDPLSpec(dumpProcessor));
  }
  if (!disableRootOut) {
    specs.emplace_back(o2::ft0::getFT0DigitWriterSpec(false, false));
  }
  return specs;
}

} // namespace ft0
} // namespace o2
