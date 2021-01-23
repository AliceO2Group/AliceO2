// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   RawDataProcessSpec.h

#ifndef O2_FDD_RAWDATAPROCESSSPEC_H
#define O2_FDD_RAWDATAPROCESSSPEC_H

#include "Framework/CallbackService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/Lifetime.h"
#include "Framework/Output.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/SerializationMethods.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"

#include "FDDRaw/DigitBlockFDD.h"
#include "DataFormatsFDD/Digit.h"
#include "DataFormatsFDD/ChannelData.h"

#include <iostream>
#include <vector>
#include <gsl/span>

using namespace o2::framework;

namespace o2
{
namespace fdd
{

class RawDataProcessSpec : public Task
{
 public:
  RawDataProcessSpec(bool dumpEventBlocks) : mDumpEventBlocks(dumpEventBlocks) {}
  ~RawDataProcessSpec() override = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;

 private:
  bool mDumpEventBlocks;

  o2::header::DataOrigin mOrigin = o2::header::gDataOriginFDD;
};

framework::DataProcessorSpec getFDDRawDataProcessSpec(bool dumpProcessor);

} // namespace fdd
} // namespace o2

#endif /* O2_FDDDATAPROCESSDPL_H */
