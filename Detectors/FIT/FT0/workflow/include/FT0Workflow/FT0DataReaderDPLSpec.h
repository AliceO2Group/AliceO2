// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   FT0DataReaderDPLSpec.h

#ifndef O2_FT0DATAREADERDPLSPEC_H
#define O2_FT0DATAREADERDPLSPEC_H

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "Framework/CallbackService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/Lifetime.h"
#include "Framework/Output.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/SerializationMethods.h"
#include "DPLUtils/DPLRawParser.h"
//#include "FT0Raw/RawReaderFT0.h"
//#include "FT0Raw/DigitBlockFT0.h"

//#include "DataFormatsFT0/Digit.h"
//#include "DataFormatsFT0/ChannelData.h"
#include <iostream>
#include <vector>
#include <gsl/span>
using namespace o2::framework;

namespace o2
{
namespace ft0
{
template <typename RawReader>
class FT0DataReaderDPLSpec : public Task
{
 public:
  FT0DataReaderDPLSpec(bool dumpEventBlocks) : mDumpEventBlocks(dumpEventBlocks) {}
  ~FT0DataReaderDPLSpec() override = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;

 private:
//  bool mDumpEventBlocks;
//  bool mIsExtendedMode;
  RawReader mRawReader;
  o2::header::DataOrigin mOrigin = o2::header::gDataOriginFT0;
};
//framework::AlgorithmSpec getAlgorithmSpec(bool dumpReader, bool isExtendedMode);
framework::DataProcessorSpec getFT0DataReaderDPLSpec(bool dumpReader);

} // namespace ft0
} // namespace o2

#endif /* O2_FT0DATAREADERDPL_H */
