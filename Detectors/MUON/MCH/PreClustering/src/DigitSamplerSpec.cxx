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

#include <string>
#include <stdexcept>

#include "Framework/CallbackService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Lifetime.h"
#include "Framework/Output.h"
#include "Framework/Task.h"

#include "MCHBase/Digit.h"

#include "MCHMappingFactory/CreateSegmentation.h"

namespace o2
{
namespace mch
{

using namespace std;
using namespace o2::framework;

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

    mUseRun2DigitUID = ic.options().get<bool>("useRun2DigitUID");

    auto stop = [this]() {
      /// close the input file
      LOG(INFO) << "stop digit sampler";
      this->mInputFile.close();
    };
    ic.services().get<CallbackService>().set(CallbackService::Id::Stop, stop);
  }

  //_________________________________________________________________________________________________
  void run(framework::ProcessingContext& pc)
  {
    /// send the digits of the current event

    int nDigits(0);
    mInputFile.read(reinterpret_cast<char*>(&nDigits), SSizeOfInt);
    if (mInputFile.fail()) {
      pc.services().get<ControlService>().endOfStream();
      return; // probably reached eof
    }

    // create the output message
    auto size = nDigits * SSizeOfDigit;
    auto msgOut = pc.outputs().make<char>(Output{"MCH", "DIGITS", 0, Lifetime::Timeframe}, SSizeOfInt + size);
    if (msgOut.size() != SSizeOfInt + size) {
      throw length_error("incorrect message payload");
    }

    auto bufferPtr = msgOut.data();

    // fill number of digits
    memcpy(bufferPtr, &nDigits, SSizeOfInt);
    bufferPtr += SSizeOfInt;

    // fill digits in O2 format, if any
    if (size > 0) {
      mInputFile.read(bufferPtr, size);
      if (mUseRun2DigitUID) {
        convertDigitUID2PadID(reinterpret_cast<Digit*>(bufferPtr), nDigits);
      }
    } else {
      LOG(INFO) << "event is empty";
    }
  }

 private:
  //_________________________________________________________________________________________________
  void convertDigitUID2PadID(Digit* digits, int nDigits)
  {
    /// convert the digit UID in run2 format into a pad ID (i.e. index) in O2 mapping

    for (int iDigit = 0; iDigit < nDigits; ++iDigit) {

      int deID = digits[iDigit].getDetID();
      int digitID = digits[iDigit].getPadID();
      int manuID = (digitID & 0xFFF000) >> 12;
      int manuCh = (digitID & 0x3F000000) >> 24;

      int padID = mapping::segmentation(deID).findPadByFEE(manuID, manuCh);
      if (padID < 0) {
        throw runtime_error(std::string("digitID ") + digitID + " does not exist in the mapping");
      }

      digits[iDigit].setPadID(padID);
    }
  }

  static constexpr uint32_t SSizeOfInt = sizeof(int);
  static constexpr uint32_t SSizeOfDigit = sizeof(Digit);

  std::ifstream mInputFile{};    ///< input file
  bool mUseRun2DigitUID = false; ///< true if Digit.mPadID = digit UID in run2 format
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
            {"useRun2DigitUID", VariantType::Bool, false, {"mPadID = digit UID in run2 format"}}}};
}

} // end namespace mch
} // end namespace o2
