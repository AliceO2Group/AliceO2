// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   ZDCWorkflow.cxx

#include "ZDCWorkflow/ZDCWorkflow.h"
//#include "ZDCWorkflow/ZDCDataProcessDPLSpec.h"
#include "ZDCWorkflow/ZDCDataReaderDPLSpec.h"
#include "ZDCWorkflow/ZDCDigitWriterDPLSpec.h"
#include "ZDCWorkflow/RawReaderZDC.h"
namespace o2
{
namespace zdc
{

framework::WorkflowSpec getZDCWorkflow(bool useProcess,
                                       bool dumpProcessor, bool dumpReader,
                                       bool disableRootOut)
{
  LOG(INFO) << "framework::WorkflowSpec getZDCWorkflow";
  framework::WorkflowSpec specs;
  specs.emplace_back(o2::zdc::getZDCDataReaderDPLSpec(RawReaderZDC{dumpReader}));
  //  if (useProcess) {
  //    specs.emplace_back(o2::zdc::getZDCDataProcessDPLSpec(dumpProcessor));
  //  }
  if (!disableRootOut) {
    specs.emplace_back(o2::zdc::getZDCDigitWriterDPLSpec());
  }
  return specs;
}

} // namespace zdc
} // namespace o2
