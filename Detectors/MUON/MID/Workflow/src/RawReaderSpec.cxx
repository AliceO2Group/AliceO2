// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MID/Workflow/src/RawReaderSpec.cxx
/// \brief  Data processor spec for MID raw reader device
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   27 September 2019

#include "MIDWorkflow/RawReaderSpec.h"

#include <fstream>
#include <chrono>
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include <Framework/Logger.h>
#include "Framework/Output.h"
#include "Framework/Task.h"
#include "DataFormatsMID/ColumnData.h"
#include "MIDRaw/CrateParameters.h"
#include "MIDRaw/Decoder.h"
#include "MIDRaw/RawFileReader.h"

namespace of = o2::framework;

namespace o2
{
namespace mid
{

class RawReaderDeviceDPL
{
 public:
  void init(of::InitContext& ic)
  {
    auto filename = ic.options().get<std::string>("mid-raw-infile");
    auto readContinuous = ic.options().get<bool>("mid-read-continuous");
    if (!mRawFileReader.init(filename.c_str(), readContinuous)) {
      return;
    }

    mHBperTimeframe = ic.options().get<int>("mid-hb-per-timeframe");

    auto stop = [this]() {
      LOG(INFO) << "Capacities: ROFRecords: " << mDecoder.getROFRecords().capacity() << "  data: " << mDecoder.getData().capacity();
      LOG(INFO) << "Read " << mHBCounter << " HBs in " << mTimer.count() << " s";
      double scaleFactor = 1.e6 / mNROFs;
      LOG(INFO) << "Processing time / " << mNROFs << " ROFs: full: " << mTimer.count() * scaleFactor << " us  decoding: " << mTimerAlgo.count() * scaleFactor << " us";
    };
    ic.services().get<of::CallbackService>().set(of::CallbackService::Id::Stop, stop);
  }

  void run(of::ProcessingContext& pc)
  {
    if (mRawFileReader.getState() != 0) {
      return;
    }

    auto tStart = std::chrono::high_resolution_clock::now();

    size_t nHBs = 0;

    while (nHBs < mHBperTimeframe && mRawFileReader.readAllGBTs(false)) {
      nHBs += mRawFileReader.getNumberOfHBs(0);
    }
    mHBCounter += nHBs;

    auto tAlgoStart = std::chrono::high_resolution_clock::now();
    mDecoder.process(mRawFileReader.getData());
    mTimerAlgo += std::chrono::high_resolution_clock::now() - tAlgoStart;

    mRawFileReader.clear();

    pc.outputs().snapshot(of::Output{"MID", "DATA", 0, of::Lifetime::Timeframe}, mDecoder.getData());
    pc.outputs().snapshot(of::Output{"MID", "DATAROF", 0, of::Lifetime::Timeframe}, mDecoder.getROFRecords());

    mTimer += std::chrono::high_resolution_clock::now() - tStart;
    mNROFs += mDecoder.getROFRecords().size();

    if (mRawFileReader.getState() == 1) {
      pc.services().get<of::ControlService>().endOfStream();
    }
  }

 private:
  Decoder mDecoder{};
  RawFileReader<raw::RawUnit> mRawFileReader{};
  int mHBperTimeframe{256};
  unsigned long int mHBCounter{0};
  std::chrono::duration<double> mTimer{0};     ///< full timer
  std::chrono::duration<double> mTimerAlgo{0}; ///< algorithm timer
  unsigned int mNROFs{0};                      /// Total number of processed ROFs
};

framework::DataProcessorSpec getRawReaderSpec()
{
  return of::DataProcessorSpec{
    "MIDRawReader",
    of::Inputs{},
    of::Outputs{of::OutputSpec{"MID", "DATA"}, of::OutputSpec{"MID", "DATAROF"}},
    of::AlgorithmSpec{of::adaptFromTask<o2::mid::RawReaderDeviceDPL>()},
    of::Options{
      {"mid-raw-infile", of::VariantType::String, "mid_raw.dat", {"Raw input file"}},
      {"mid-hb-per-timeframe", of::VariantType::Int, 256, {"Number of heart beats per timeframe"}},
      {"mid-read-continuous", of::VariantType::Bool, false, {"Rewind the file when reaching the end so to simulate a continuous readout"}}}};
}
} // namespace mid
} // namespace o2
