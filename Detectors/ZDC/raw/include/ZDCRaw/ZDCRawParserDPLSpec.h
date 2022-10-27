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

/// @file   ZDCRawParserDPLSpec.h

#ifndef O2_ZDCRAWPARSERDPLSPEC_H
#define O2_ZDCRAWPARSERDPLSPEC_H

#include <TStopwatch.h>
#include "Framework/Logger.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/DataAllocator.h"
#include "Framework/DataSpecUtils.h"
#include "Framework/Task.h"
#include "CommonUtils/NameConf.h"
#include "ZDCBase/Constants.h"
#include "ZDCBase/ModuleConfig.h"
#include "ZDCRaw/DumpRaw.h"

namespace o2
{
namespace zdc
{
class ZDCRawParserDPLSpec : public o2::framework::Task
{
 public:
  ZDCRawParserDPLSpec();
  ZDCRawParserDPLSpec(const int verbosity);
  ~ZDCRawParserDPLSpec() override = default;
  void init(o2::framework::InitContext& ic) final;
  void run(o2::framework::ProcessingContext& pc) final;
  void endOfStream(o2::framework::EndOfStreamContext& ec) final;

 private:
  int mVerbosity = DbgZero;  // Verbosity level
  bool mInitialized = false; // Connect once to CCDB during initialization
  DumpRaw mWorker;           // Baseline calibration object
  TStopwatch mTimer;
};

framework::DataProcessorSpec getZDCRawParserDPLSpec();

} // namespace zdc
} // namespace o2

#endif /* O2_ZDCRAWPARSERDPLSPEC_H */
