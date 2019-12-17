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

using namespace o2::framework;

namespace o2
{
namespace tof
{

void RawReaderTask::init(InitContext& ic)
{
  LOG(INFO) << "RawReader init";
  auto filename = ic.options().get<std::string>("tof-raw-filename");

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

  /** read file **/
  mFile.read(mDataFrame.mBuffer, mDataFrame.mSize);

  /** push the data **/
  pc.outputs().snapshot(Output{"TOF", "RAWDATAFRAME", 0, Lifetime::Timeframe}, mDataFrame);

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
    Inputs{},                                                           // inputs
    Outputs{OutputSpec("TOF", "RAWDATAFRAME", 0, Lifetime::Timeframe)}, // outputs
    AlgorithmSpec{adaptFromTask<RawReaderTask>()},                      // call constructor + execute init (check)
    Options{
      {"tof-raw-filename", VariantType::String, "", {"Name of the raw input file"}}}};
}

} // namespace tof
} // namespace o2
