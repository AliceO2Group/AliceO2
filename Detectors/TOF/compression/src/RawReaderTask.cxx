// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   RawReaderTask.cxx
/// @author Roberto Preghenella
/// @since  2019-12-18
/// @brief  TOF raw reader task

#include "TOFCompression/RawReaderTask.h"
#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Headers/RAWDataHeader.h"

using namespace o2::framework;

namespace o2
{
namespace tof
{

void RawReaderTask::init(InitContext& ic)
{
  LOG(INFO) << "RawReader init";
  auto filename = ic.options().get<std::string>("tof-raw-filename");
  mBuffer.reserve(1048576);

  /** open file **/
  if (mFile.is_open()) {
    LOG(WARNING) << "a file was already open, closing";
    mFile.close();
  }
  mFile.open(filename.c_str(), std::fstream::in | std::fstream::binary);
  if (!mFile.is_open()) {
    LOG(ERROR) << "cannot open input file: " << filename;
    mStatus = true;
    return;
  }
}

void RawReaderTask::run(ProcessingContext& pc)
{
  LOG(DEBUG) << "RawReader run";

  /** check status **/
  if (mStatus) {
    pc.services().get<ControlService>().endOfStream();
    pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
    return;
  }

  /** read full HBF **/
  int headerSize = sizeof(o2::header::RAWDataHeader);
  char* inputPointer = mBuffer.data();
  mFile.read(inputPointer, headerSize);
  inputPointer += headerSize;
  auto rdh = reinterpret_cast<o2::header::RAWDataHeader*>(inputPointer);
  while (!rdh->stop) {
    auto dataSize = rdh->offsetToNext - headerSize;
    mFile.read(inputPointer, dataSize);
    inputPointer += dataSize;
    mFile.read(inputPointer, headerSize);
    rdh = reinterpret_cast<o2::header::RAWDataHeader*>(inputPointer);
    inputPointer += headerSize;
  }
  mBuffer.resize(inputPointer - mBuffer.data());

  auto freefct = [](void* data, void* hint) {}; // simply ignore the cleanup for the test
  pc.outputs().adoptChunk(Output{o2::header::gDataOriginTOF, "RAWDATAFRAME", 0, Lifetime::Timeframe}, mBuffer.data(), mBuffer.size(), freefct, nullptr);

  /** check eof **/
  if (mFile.eof()) {
    LOG(WARNING) << "nothig else to read";
    mFile.close();
    mStatus = true;
  }
}

DataProcessorSpec RawReaderTask::getSpec()
{
  std::vector<InputSpec> inputs;
  std::vector<OutputSpec> outputs;

  return DataProcessorSpec{
    "tof-raw-reader",
    Inputs{},                                                                                // inputs
    Outputs{OutputSpec(o2::header::gDataOriginTOF, "RAWDATAFRAME", 0, Lifetime::Timeframe)}, // outputs
    AlgorithmSpec{adaptFromTask<RawReaderTask>()},                                           // call constructor + execute init (check)
    Options{
      {"tof-raw-filename", VariantType::String, "", {"Name of the raw input file"}}}};
}

} // namespace tof
} // namespace o2
