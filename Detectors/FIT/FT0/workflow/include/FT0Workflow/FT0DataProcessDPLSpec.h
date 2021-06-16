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

/// @file   FT0DataProcessDPLSpec.h

#ifndef O2_FT0DATAPROCESSDPLSPEC_H
#define O2_FT0DATAPROCESSDPLSPEC_H

#include "Framework/CallbackService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/Lifetime.h"
#include "Framework/Output.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/SerializationMethods.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"

#include "FT0Raw/DigitBlockFT0.h"
#include "DataFormatsFT0/Digit.h"
#include "DataFormatsFT0/ChannelData.h"

#include <iostream>
#include <vector>
#include <gsl/span>

using namespace o2::framework;

namespace o2
{
namespace ft0
{

class FT0DataProcessDPLSpec : public Task
{
 public:
  FT0DataProcessDPLSpec(bool dumpEventBlocks) : mDumpEventBlocks(dumpEventBlocks) {}
  ~FT0DataProcessDPLSpec() override = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;

 private:
  bool mDumpEventBlocks;

  o2::header::DataOrigin mOrigin = o2::header::gDataOriginFT0;
};

framework::DataProcessorSpec getFT0DataProcessDPLSpec(bool dumpProcessor);

} // namespace ft0
} // namespace o2

#endif /* O2_FT0DATAPROCESSDPL_H */
