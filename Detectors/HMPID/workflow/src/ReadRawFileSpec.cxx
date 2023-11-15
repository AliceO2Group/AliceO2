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

/// \file   ReadRawFileSpec.cxx
/// \author Antonio Franco - INFN Bari
/// \version 1.0
/// \date 01 feb 2021
/// \brief Implementation of a data processor to read a raw file and produce Raw Pages stream
///

#include <random>
#include <iostream>
#include <fstream>
#include <stdexcept>

#include "Framework/CallbackService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Lifetime.h"
#include "Framework/Output.h"
#include "Framework/Task.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/Logger.h"

#include "Headers/RAWDataHeader.h"
#include "DetectorsRaw/RDHUtils.h"
#include "DPLUtils/DPLRawParser.h"

#include "HMPIDWorkflow/ReadRawFileSpec.h"

namespace o2
{
namespace hmpid
{

using namespace o2;
using namespace o2::framework;
using RDH = o2::header::RDHAny;

void RawFileReaderTask::init(framework::InitContext& ic)
{
  LOG(info) << "Raw file reader init ";

  // read input parameters
  mPrint = ic.options().get<bool>("print");
  std::string inFileName = ic.options().get<std::string>("raw-file");
  mInputFile.open(inFileName, std::ios::binary);
  if (!mInputFile.is_open()) {
    throw std::invalid_argument("Cannot open input file \"" + inFileName + "\"");
  }

  auto stop = [this]() {
    LOG(info) << "stop file reader"; // close the input file
    this->mInputFile.close();
  };
  ic.services().get<CallbackService>().set<CallbackService::Id::Stop>(stop);

  mExTimer.start();
  return;
}

void RawFileReaderTask::run(framework::ProcessingContext& pc)
{

  RDH rdh;
  char* outBuffer{nullptr};
  size_t bufSize{0};
  int numberOfFrames = 0;
  LOG(info) << "Sleep 1 sec for sync";
  sleep(1);

  while (true) {
    // usleep(100);
    mInputFile.read((char*)(&rdh), sizeof(RDH)); // read the next RDH, stop if no more data is available
    if (mInputFile.fail()) {
      free(outBuffer);
      mInputFile.close();
      pc.services().get<ControlService>().endOfStream();
      pc.services().get<o2::framework::ControlService>().readyToQuit(framework::QuitRequest::Me);
      break;
    }
    auto rdhVersion = o2::raw::RDHUtils::getVersion(rdh);
    auto rdhHeaderSize = o2::raw::RDHUtils::getHeaderSize(rdh);
    LOG(debug) << "header_version=" << (int)rdhVersion;
    if (rdhVersion < 6 || rdhHeaderSize != 64) {
      LOG(info) << "Old or corrupted raw file, abort !";
      return;
    }
    auto frameSize = o2::raw::RDHUtils::getOffsetToNext(rdh); // get the frame size
    LOG(debug) << "frameSize=" << frameSize;
    if (frameSize < rdhHeaderSize) { // stop if the frame size is too small
      LOG(info) << "Wrong Frame size - frameSize too small: " << frameSize;
      pc.services().get<ControlService>().endOfStream();
      return;
    }
    numberOfFrames++;
    LOG(debug) << "Process page " << numberOfFrames << " dim = " << frameSize;

    outBuffer = (char*)realloc(outBuffer, bufSize + frameSize); // allocate the buffer
    if (outBuffer == nullptr) {
      LOG(info) << "Buffer allocation error. Abort !";
      pc.services().get<ControlService>().endOfStream();
      return;
    }
    memcpy(outBuffer, &rdh, rdhHeaderSize); // fill the buffer
    mInputFile.read(outBuffer + rdhHeaderSize, frameSize - rdhHeaderSize);
    if (mInputFile.fail()) { // Could be EoF
      free(outBuffer);
      pc.services().get<ControlService>().endOfStream();
      pc.services().get<o2::framework::ControlService>().readyToQuit(framework::QuitRequest::Me);
      break; // probably reached eof
    }
    bufSize = frameSize; // Set the buffer pointer
    pc.outputs().snapshot(Output{"HMP", "RAWDATA"}, outBuffer, bufSize);
    // std::cout << mExTimer.mTimer.CpuTime() << " " << mExTimer.mLastLogTime << std::endl;
    mExTimer.elapseMes("... Reading... Number of Pages = " + std::to_string(numberOfFrames));
  } // while (true)

  mExTimer.logMes("End of file !  Number of frames processed = " + std::to_string(numberOfFrames));
  mExTimer.stop();
  return;
}

//_________________________________________________________________________________________________
// clang-format off
o2::framework::DataProcessorSpec getReadRawFileSpec(std::string inputSpec)
{
  std::vector<o2::framework::InputSpec> inputs;
  return DataProcessorSpec{
    "HMP-ReadRawFile",
    inputs,
    Outputs{OutputSpec{"HMP", "RAWDATA", 0, Lifetime::Timeframe}},
    AlgorithmSpec{adaptFromTask<RawFileReaderTask>()},
    Options{{"raw-file", VariantType::String, "", {"Raw input file name"}},
            {"print", VariantType::Bool, false, {"verbose output"}}}};
}
// clang-format on

} // end namespace hmpid
} // end namespace o2
