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
#include "Framework/Logger.h"

#include "DataFormatsMCH/ROFRecord.h"
#include "DataFormatsMCH/Digit.h"

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
    LOG(info) << "initializing digit sampler";

    auto inputFileName = ic.options().get<std::string>("infile");
    mInputFile.open(inputFileName, ios::binary);
    if (!mInputFile.is_open()) {
      throw invalid_argument("Cannot open input file " + inputFileName);
    }
    if (mInputFile.peek() == EOF) {
      throw length_error("input file is empty");
    }

    mUseRun2DigitUID = ic.options().get<bool>("useRun2DigitUID");
    mPrint = ic.options().get<bool>("print");
    mNevents = ic.options().get<int>("nevents");
    mEvent = ic.options().get<int>("event");
    mNEventsPerTF = ic.options().get<int>("nEventsPerTF");
    if (mNEventsPerTF < 1) {
      throw invalid_argument("number of events per time frame must be >= 1");
    }

    auto stop = [this]() {
      // close the input file
      LOG(info) << "stop digit sampler";
      this->mInputFile.close();
    };
    ic.services().get<CallbackService>().set(CallbackService::Id::Stop, stop);
  }

  //_________________________________________________________________________________________________
  void run(framework::ProcessingContext& pc)
  {
    /// send the digits of the next event(s) in the current TF

    // skip events until the requested one, if any
    while (mCurrentEvent < mEvent && mInputFile.peek() != EOF) {
      LOG(info) << "skipping event " << mCurrentEvent;
      skipOneEvent();
      ++mCurrentEvent;
    }

    // stop if requested event(s) already processed or eof reached
    if (mNevents == 0 || (mEvent >= 0 && mCurrentEvent > mEvent) || mInputFile.peek() == EOF) {
      pc.services().get<ControlService>().endOfStream();
      // pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
      return;
    }

    // create the output messages
    auto& rofs = pc.outputs().make<std::vector<ROFRecord>>(OutputRef{"rofs"});
    auto& digits = pc.outputs().make<std::vector<Digit>>(OutputRef{"digits"});

    if (mCurrentEvent == mEvent) {

      // send only the requested event
      int nDigits = readOneEvent(digits);
      rofs.emplace_back(o2::InteractionRecord{0, static_cast<uint32_t>(mCurrentEvent)},
                        digits.size() - nDigits, nDigits);
      ++mCurrentEvent;

    } else {

      // or loop over the requested number of events (or until eof) and send all of them
      rofs.reserve(mNEventsPerTF);
      for (int iEvt = 0; iEvt < mNEventsPerTF && mNevents != 0 && mInputFile.peek() != EOF; ++iEvt, --mNevents) {
        int nDigits = readOneEvent(digits);
        rofs.emplace_back(o2::InteractionRecord{0, static_cast<uint32_t>(mCurrentEvent)},
                          digits.size() - nDigits, nDigits);
        ++mCurrentEvent;
      }
      rofs.shrink_to_fit(); // fix error "could not set used size: boost::interprocess::bad_alloc" for some sizes
    }

    // convert the digits UID if needed
    if (mUseRun2DigitUID) {
      convertDigitUID2PadID(digits);
    }
  }

 private:
  //_________________________________________________________________________________________________
  void skipOneEvent()
  {
    /// drop one event from the input file

    // get the number of digits
    int nDigits(-1);
    mInputFile.read(reinterpret_cast<char*>(&nDigits), sizeof(int));
    if (mInputFile.fail() || nDigits < 0) {
      throw length_error("invalid input");
    }

    // skip the digits if any
    if (nDigits > 0) {
      mInputFile.seekg(nDigits * sizeof(Digit), std::ios::cur);
      if (mInputFile.fail()) {
        throw length_error("invalid input");
      }
    }
  }

  //_________________________________________________________________________________________________
  int readOneEvent(std::vector<Digit, o2::pmr::polymorphic_allocator<Digit>>& digits)
  {
    /// fill the internal buffer with the digits of the current event

    // get the number of digits
    int nDigits(-1);
    mInputFile.read(reinterpret_cast<char*>(&nDigits), sizeof(int));
    if (mInputFile.fail() || nDigits < 0) {
      throw length_error("invalid input");
    }

    // get the digits if any
    if (nDigits > 0) {
      auto digitOffset = digits.size();
      digits.resize(digitOffset + nDigits);
      mInputFile.read(reinterpret_cast<char*>(&digits[digitOffset]), nDigits * sizeof(Digit));
      if (mInputFile.fail()) {
        throw length_error("invalid input");
      }
    } else {
      LOG(info) << "event is empty";
    }

    return nDigits;
  }

  //_________________________________________________________________________________________________
  void convertDigitUID2PadID(std::vector<Digit, o2::pmr::polymorphic_allocator<Digit>>& digits)
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
  int mCurrentEvent = 0;         ///< current event number
  int mNEventsPerTF = 1;         ///< number of events per time frame
};

//_________________________________________________________________________________________________
o2::framework::DataProcessorSpec getDigitSamplerSpec(const char* specName)
{
  return DataProcessorSpec{
    specName,
    Inputs{},
    Outputs{OutputSpec{{"rofs"}, "MCH", "DIGITROFS", 0, Lifetime::Timeframe},
            OutputSpec{{"digits"}, "MCH", "DIGITS", 0, Lifetime::Timeframe}},
    AlgorithmSpec{adaptFromTask<DigitSamplerTask>()},
    Options{{"infile", VariantType::String, "", {"input file name"}},
            {"useRun2DigitUID", VariantType::Bool, false, {"mPadID = digit UID in run2 format"}},
            {"print", VariantType::Bool, false, {"print digits"}},
            {"nevents", VariantType::Int, -1, {"number of events to process (-1 = all events in the file)"}},
            {"event", VariantType::Int, -1, {"event to process"}},
            {"nEventsPerTF", VariantType::Int, 1, {"number of events per time frame"}}}};
}

} // end namespace mch
} // end namespace o2
