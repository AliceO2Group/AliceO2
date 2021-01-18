// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

///
/// \file    RawFileReaderSpec.cxx
/// \author  Antonio Franco
///
/// \brief This is an executable that reads a data file from disk and sends the individual CRU pages via DPL.
///
/// This is an executable that reads a data file from disk and sends the individual CRU pages via the Data Processing Layer.
//

#include <random>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <array>
#include <functional>
#include <vector>

#include "Framework/CallbackService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Lifetime.h"
#include "Framework/Output.h"
#include "Framework/Task.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/Logger.h"


#include "DPLUtils/DPLRawParser.h"
#include "Headers/RAWDataHeader.h"
#include "DetectorsRaw/RDHUtils.h"

#include "HMPIDWorkflow/RawFileReaderSpec.h"

namespace o2
{
namespace hmpid
{

using RDH = o2::header::RDHAny;
using namespace o2;
using namespace o2::framework;


void RawFileReaderTask::init(framework::InitContext& ic)
{
  // Get the input file and other options from the context
  LOG(INFO) << "Raw file reader Init";

  auto inputFileName = ic.options().get<std::string>("infile");
  mInputFile.open(inputFileName, std::ios::binary);
  if (!mInputFile.is_open()) {
    throw std::invalid_argument("Cannot open input file \"" + inputFileName + "\"");
  }

  // define the callback to close the file
  auto stop = [this]() {
    LOG(INFO) << "stop file reader";
    this->mInputFile.close();
  };
  ic.services().get<CallbackService>().set(CallbackService::Id::Stop, stop);
}

//_________________________________________________________________________________________________
void RawFileReaderTask::run(framework::ProcessingContext& pc)
{
  // send one RDH block via DPL
  RDH rdh;
  char* buf{nullptr};
  size_t bufSize{0};

  while (true) {

    // read the next RDH, stop if no more data is available
    mInputFile.read((char*)(&rdh), sizeof(RDH));
    if (mInputFile.fail()) {
      pc.services().get<ControlService>().endOfStream();
      return; // probably reached eof
    }

    // check that the RDH version is ok (only RDH versions from 4 to 6 are supported at the moment)
    auto rdhVersion = o2::raw::RDHUtils::getVersion(rdh);
    auto rdhHeaderSize = o2::raw::RDHUtils::getHeaderSize(rdh);
    if (rdhVersion != 6 || rdhHeaderSize != 64) {
      return;
    }

    // get the frame size from the RDH offsetToNext field
    auto frameSize = o2::raw::RDHUtils::getOffsetToNext(rdh);

    // stop if the frame size is too small
    if (frameSize < rdhHeaderSize) {
      LOG(WARNING) << " FrameSize too small: " << frameSize;
      pc.services().get<ControlService>().endOfStream();
      return;
    }

    // allocate the output buffer
    buf = (char*)realloc(buf, frameSize);
    if (buf == nullptr) {
      LOG(ERROR) << " Failed to allocate buffer";
      pc.services().get<ControlService>().endOfStream();
      return;
    }

    // copy the RDH into the output buffer
    memcpy(buf, &rdh, rdhHeaderSize);

    // read the frame payload into the output buffer
    mInputFile.read(buf + rdhHeaderSize, frameSize - rdhHeaderSize);
    // stop if data cannot be read completely
    if (mInputFile.fail()) {
      LOG(ERROR) << "Fail to read the payload ";
      free(buf);
      pc.services().get<ControlService>().endOfStream();
      return; // probably reached eof
    }
    // create the output message
    auto freefct = [](void* data, void* /*hint*/) { free(data); };
    pc.outputs().adoptChunk(Output{"HMP", "rawfile"}, buf, bufSize, freefct, nullptr);
  }
}

//_________________________________________________________________________________________________
// clang-format off
o2::framework::DataProcessorSpec getRawFileReaderSpec()
{
  return DataProcessorSpec{
    "RawFileReader",
    Inputs{},
    Outputs{OutputSpec{"HMP", "rawfile", 0, Lifetime::Timeframe}},
    AlgorithmSpec{adaptFromTask<RawFileReaderTask>()},
    Options{{"infile", VariantType::String, "", {"input file name"}}}
    };
}
// clang-format on


} // end namespace mch
} // end namespace o2
