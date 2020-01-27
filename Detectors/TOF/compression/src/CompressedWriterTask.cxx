// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   CompressedWriterTask.cxx
/// @author Roberto Preghenella
/// @since  2019-12-18
/// @brief  TOF compressed data writer task

#include "TOFCompression/CompressedWriterTask.h"
#include "TOFCompression/RawDataFrame.h"
#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"

using namespace o2::framework;

namespace o2
{
namespace tof
{

void CompressedWriterTask::init(InitContext& ic)
{
  LOG(INFO) << "CompressedWriter init";
  auto filename = ic.options().get<std::string>("tof-compressed-filename");

  /** open file **/
  if (mFile.is_open()) {
    LOG(WARNING) << "a file was already open, closing";
    mFile.close();
  }
  mFile.open(filename.c_str(), std::fstream::out | std::fstream::binary);
  if (!mFile.is_open()) {
    LOG(ERROR) << "cannot open output file: " << filename;
    mStatus = true;
    return;
  }

  auto finishFunction = [this]() {
    LOG(INFO) << "CompressedWriter finish";
    mFile.close();
  };
  ic.services().get<CallbackService>().set(CallbackService::Id::Stop, finishFunction);
}

void CompressedWriterTask::run(ProcessingContext& pc)
{
  LOG(DEBUG) << "CompressedWriter run";

  /** check status **/
  if (mStatus) {
    pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
    return;
  }

  /** receive input **/
  auto dataFrame = pc.inputs().get<RawDataFrame*>("dataframe");

  /** write to file **/
  mFile.write(dataFrame->mBuffer, dataFrame->mSize);
}

DataProcessorSpec CompressedWriterTask::getSpec()
{
  std::vector<InputSpec> inputs;
  std::vector<OutputSpec> outputs;

  return DataProcessorSpec{
    "tof-compressed-writer",
    Inputs{InputSpec("dataframe", o2::header::gDataOriginTOF, "CMPDATAFRAME", 0, Lifetime::Timeframe)}, // inputs
    Outputs{},                                                                                          // outputs
    AlgorithmSpec{adaptFromTask<CompressedWriterTask>()},                                               // call constructor + execute init (check)
    Options{
      {"tof-compressed-filename", VariantType::String, "/dev/null", {"Name of the compressed output file"}}}};
}

} // namespace tof
} // namespace o2
