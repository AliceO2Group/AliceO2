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

/// @file   ZDCDataReaderDPLSpec.h

#ifndef O2_ZDCDATAREADERDPLSPEC_H
#define O2_ZDCDATAREADERDPLSPEC_H

#include "CommonUtils/NameConf.h"
#include "CCDB/BasicCCDBManager.h"
#include "CCDB/CCDBTimeStampUtils.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "Framework/CallbackService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/Lifetime.h"
#include "Framework/Output.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/SerializationMethods.h"
#include "Framework/InputRecordWalker.h"
#include "DPLUtils/DPLRawParser.h"
#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include "Framework/InputSpec.h"
#include "CommonUtils/ConfigurableParam.h"
#include "ZDCBase/Constants.h"
#include "ZDCBase/ModuleConfig.h"
#include "ZDCRaw/RawReaderZDC.h"
#include <iostream>
#include <vector>
#include <gsl/span>

using namespace o2::framework;

namespace o2
{
namespace zdc
{
class ZDCDataReaderDPLSpec : public Task
{
 public:
  ZDCDataReaderDPLSpec() = default;
  ZDCDataReaderDPLSpec(const RawReaderZDC& rawReader);
  ~ZDCDataReaderDPLSpec() override = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;

 private:
  std::string mccdbHost = o2::base::NameConf::getCCDBServer();
  int mVerbosity = 0;
  RawReaderZDC mRawReader;
};

framework::DataProcessorSpec getZDCDataReaderDPLSpec(const RawReaderZDC& rawReader, const bool askSTFDist);

} // namespace zdc
} // namespace o2

#endif /* O2_ZDCDATAREADERDPL_H */
