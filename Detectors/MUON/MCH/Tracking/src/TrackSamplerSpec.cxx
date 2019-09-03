// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file TrackSamplerSpec.cxx
/// \brief Implementation of a data processor to read and send tracks
///
/// \author Philippe Pillot, Subatech

#include "TrackSamplerSpec.h"

#include <iostream>
#include <fstream>

#include <stdexcept>

#include "Framework/CallbackService.h"
#include "Framework/ControlService.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Lifetime.h"
#include "Framework/Output.h"
#include "Framework/Task.h"

namespace o2
{
namespace mch
{

using namespace std;
using namespace o2::framework;

class TrackSamplerTask
{
 public:
  //_________________________________________________________________________________________________
  void init(framework::InitContext& ic)
  {
    /// Get the input file from the context
    LOG(INFO) << "initializing track sampler";

    auto inputFileName = ic.options().get<std::string>("infile");
    mInputFile.open(inputFileName, ios::binary);
    if (!mInputFile.is_open()) {
      throw invalid_argument("Cannot open input file" + inputFileName);
    }

    auto stop = [this]() {
      /// close the input file
      LOG(INFO) << "stop track sampler";
      this->mInputFile.close();
    };
    ic.services().get<CallbackService>().set(CallbackService::Id::Stop, stop);
  }

  //_________________________________________________________________________________________________
  void run(framework::ProcessingContext& pc)
  {
    /// send the tracks with attached clusters of the current event

    int event(0);
    mInputFile.read(reinterpret_cast<char*>(&event), SSizeOfInt);
    if (mInputFile.fail()) {
      return; // probably reached eof
    }

    int size(0);
    mInputFile.read(reinterpret_cast<char*>(&size), SSizeOfInt);
    if (size == 0) {
      LOG(INFO) << "event " << event + 1 << " is empty";
      return; // skip empty event
    }

    // create the output message
    auto msgOut = pc.outputs().make<char>(Output{"MCH", "TRACKS", 0, Lifetime::Timeframe}, SHeaderSize + size);
    if (msgOut.size() != SHeaderSize + size) {
      throw length_error("incorrect message payload");
    }

    auto bufferPtr = msgOut.data();

    // fill header info
    memcpy(bufferPtr, &event, SSizeOfInt);
    bufferPtr += SSizeOfInt;
    memcpy(bufferPtr, &size, SSizeOfInt);
    bufferPtr += SSizeOfInt;

    // fill tracks and clusters info
    mInputFile.read(bufferPtr, size);
  }

 private:
  static constexpr int SSizeOfInt = sizeof(int);
  static constexpr int SHeaderSize = 2 * SSizeOfInt;

  std::ifstream mInputFile{}; ///< input file
};

//_________________________________________________________________________________________________
o2::framework::DataProcessorSpec getTrackSamplerSpec()
{
  return DataProcessorSpec{
    "TrackSampler",
    Inputs{},
    Outputs{OutputSpec{"MCH", "TRACKS", 0, Lifetime::Timeframe}},
    AlgorithmSpec{adaptFromTask<TrackSamplerTask>()},
    Options{{"infile", VariantType::String, "", {"input filename"}}}};
}

} // end namespace mch
} // end namespace o2
