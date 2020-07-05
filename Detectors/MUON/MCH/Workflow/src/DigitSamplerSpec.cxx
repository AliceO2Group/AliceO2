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
#include <vector>

#include <gsl/span>

#include "Framework/CallbackService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Lifetime.h"
#include "Framework/Output.h"
#include "Framework/Task.h"

#include "MCHBase/Digit.h"

#include "MCHMappingInterface/Segmentation.h"

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
    mPrint = ic.options().get<bool>("print");
    mNevents = ic.options().get<int>("nevents");
    mEvent = ic.options().get<int>("event");

    auto stop = [this]() {
      // close the input file
      LOG(INFO) << "stop digit sampler";
      this->mInputFile.close();
    };
    ic.services().get<CallbackService>().set(CallbackService::Id::Stop, stop);
  }

  //_________________________________________________________________________________________________
  void run(framework::ProcessingContext& pc)
  {
    /// check the number of processed events

    if (mNevents == 0) {
      pc.services().get<ControlService>().endOfStream();
      return;
    } else if (mNevents > 0) {
      mNevents -= 1;
    }

    // send the digits of the current event
    int nDigits(0);
    mInputFile.read(reinterpret_cast<char*>(&nDigits), sizeof(int));
    if (mInputFile.fail()) {
      pc.services().get<ControlService>().endOfStream();
      return; // probably reached eof
    }

    // send only the requested event, if any
    if (mEvent >= 0) {
      ++mCurrentEvent;
      if (mCurrentEvent < mEvent) {
        std::vector<Digit> digits(nDigits);
        mInputFile.read(reinterpret_cast<char*>(digits.data()), nDigits * sizeof(Digit));
        return;
      } else if (mCurrentEvent > mEvent) {
        pc.services().get<ControlService>().endOfStream();
        return;
      }
    }

    // create the output message
    auto digits = pc.outputs().make<Digit>(Output{"MCH", "DIGITS", 0, Lifetime::Timeframe}, nDigits);
    if (digits.size() != nDigits) {
      throw length_error("incorrect message payload");
    }

    // fill digits in O2 format, if any
    if (nDigits > 0) {
      mInputFile.read(reinterpret_cast<char*>(digits.data()), digits.size_bytes());
      if (mUseRun2DigitUID) {
        convertDigitUID2PadID(digits);
      }
    } else {
      LOG(INFO) << "event is empty";
    }
  }

 private:
  //_________________________________________________________________________________________________
  void convertDigitUID2PadID(gsl::span<Digit> digits)
  {
    /// convert the digit UID in run2 format into a pad ID (i.e. index) in O2 mapping

    for (auto& digit : digits) {

      int deID = digit.getDetID();
      int digitID = digit.getPadID();
      int manuID = (digitID & 0xFFF000) >> 12;
      int manuCh = (digitID & 0x3F000000) >> 24;

      int padID = mapping::segmentation(deID).findPadByFEE(manuID, manuCh);
      if (mPrint) {
        cout << deID << "  " << digitID << "  " << manuID << "  " << manuCh << "  " << padID << endl;
      }
      if (padID < 0) {
        throw runtime_error(std::string("digitID ") + digitID + " does not exist in the mapping");
      }

      digit.setPadID(padID);
    }
  }

  std::ifstream mInputFile{};    ///< input file
  bool mUseRun2DigitUID = false; ///< true if Digit.mPadID = digit UID in run2 format
  bool mPrint = false;           ///< print digits to terminal
  int mNevents = -1;             ///< number of events to process
  int mEvent = -1;               ///< if mEvent >= 0, process only this event
  int mCurrentEvent = -1;        ///< current event number
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
            {"useRun2DigitUID", VariantType::Bool, false, {"mPadID = digit UID in run2 format"}},
            {"print", VariantType::Bool, false, {"print digits"}},
            {"nevents", VariantType::Int, -1, {"number of events to process (-1 = all events in the file)"}},
            {"event", VariantType::Int, -1, {"event to process"}}}};
}

} // end namespace mch
} // end namespace o2
