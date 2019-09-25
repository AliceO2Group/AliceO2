// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file DigitSamplerSpec.cxx
/// \brief Implementation of a data processor to read and send digits
///
/// \author Philippe Pillot, Subatech

#include "DigitSamplerSpec.h"

#include <iostream>
#include <fstream>

#include <stdexcept>

#include "Framework/CallbackService.h"
#include "Framework/ControlService.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Lifetime.h"
#include "Framework/Output.h"
#include "Framework/Task.h"

#include "MCHBase/DigitBlock.h"

namespace o2
{
namespace mch
{

using namespace std;
using namespace o2::framework;
using namespace o2::mch;

class DigitSamplerTask
{
 public:
  //_________________________________________________________________________________________________
  void init(framework::InitContext& ic)
  {
    /// Get the input file and other options from the context
    LOG(INFO) << "initializing digit sampler";

    auto inputFileName = ic.options().get<std::string>("infile");
    mInputFile.open(inputFileName, ios::binary);
    if (!mInputFile.is_open()) {
      throw invalid_argument("Cannot open input file " + inputFileName);
    }

    auto stop = [this]() {
      /// close the input file
      LOG(INFO) << "stop digit sampler";
      this->mInputFile.close();
    };
    ic.services().get<CallbackService>().set(CallbackService::Id::Stop, stop);

    mPrint = ic.options().get<bool>("print");
  }

  //_________________________________________________________________________________________________
  void run(framework::ProcessingContext& pc)
  {
    /// send the digits of the current event

    DigitBlock digitBlock{};
    mInputFile.read(reinterpret_cast<char*>(&digitBlock), SSizeOfDigitBlock);
    if (mInputFile.fail()) {
      return; // probably reached eof
    }

    if (digitBlock.header.fRecordWidth != SSizeOfDigitStruct) {
      throw length_error("incorrect size of digits. Content changed?");
    }

    // create the output message
    auto size = digitBlock.header.fNrecords * SSizeOfDigitStruct;
    auto msgOut = pc.outputs().make<char>(Output{"MCH", "DIGITS", 0, Lifetime::Timeframe}, SSizeOfDigitBlock + size);
    if (msgOut.size() != SSizeOfDigitBlock + size) {
      throw length_error("incorrect message payload");
    }

    auto bufferPtr = msgOut.data();

    // fill header info
    memcpy(bufferPtr, &digitBlock, SSizeOfDigitBlock);
    bufferPtr += SSizeOfDigitBlock;

    // fill digits info
    if (size > 0) {
      mInputFile.read(bufferPtr, size);
    } else {
      LOG(INFO) << "event is empty";
    }

    if (mPrint) {
      LOG(INFO) << "block: " << *reinterpret_cast<const DigitBlock*>(msgOut.data());
    }
  }

 private:
  static constexpr uint32_t SSizeOfDigitBlock = sizeof(DigitBlock);
  static constexpr uint32_t SSizeOfDigitStruct = sizeof(DigitStruct);

  std::ifstream mInputFile{}; ///< input file
  bool mPrint = false;        ///< print digits
};

//_________________________________________________________________________________________________
o2::framework::DataProcessorSpec getDigitSamplerSpec()
{
  return DataProcessorSpec{
    "DigitSampler",
    Inputs{},
    Outputs{OutputSpec{"MCH", "DIGITS", 0, Lifetime::Timeframe}},
    AlgorithmSpec{adaptFromTask<DigitSamplerTask>()},
    Options{{"infile", VariantType::String, "", {"input file name"}},
            {"print", VariantType::Bool, false, {"print digits"}}}};
}

} // end namespace mch
} // end namespace o2
