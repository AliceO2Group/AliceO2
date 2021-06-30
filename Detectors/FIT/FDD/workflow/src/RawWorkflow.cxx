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

/// @file   RawWorkflow.cxx

#include "FDDWorkflow/RawWorkflow.h"
#include "FDDWorkflow/RawDataProcessSpec.h"
#include "FDDWorkflow/RawDataReaderSpec.h"
#include "FDDWorkflow/DigitWriterSpec.h"
#include "FDDWorkflow/RawReaderFDD.h"
namespace o2
{
namespace fdd
{

framework::WorkflowSpec getFDDRawWorkflow(bool useProcess,
                                          bool dumpProcessor, bool dumpReader,
                                          bool disableRootOut)
{
  LOG(INFO) << "framework::WorkflowSpec getFDDWorkflow";
  framework::WorkflowSpec specs;
  specs.emplace_back(o2::fdd::getFDDRawDataReaderSpec(RawReaderFDD{dumpReader}));

  if (useProcess) {
    specs.emplace_back(o2::fdd::getFDDRawDataProcessSpec(dumpProcessor));
  }
  if (!disableRootOut) {
    specs.emplace_back(o2::fdd::getFDDDigitWriterSpec(false, false));
  }
  return specs;
}

} // namespace fdd
} // namespace o2
