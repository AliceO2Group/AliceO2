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
#include "Framework/ControlService.h"
#include <Framework/Logger.h>
#include "Framework/Output.h"
#include "Framework/Task.h"
#include "DataFormatsMID/ColumnData.h"
#include "MIDRaw/Decoder.h"

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
    mFile.open(filename.c_str());
    if (!mFile.is_open()) {
      LOG(ERROR) << "Cannot open the " << filename << " file !";
      mState = 1;
      return;
    }
    mBytes.reserve(2 * raw::sMaxBufferSize);
    mDecoder.init();

    mHBperTimeframe = ic.options().get<int>("mid-hb-per-timeframe");

    mState = 0;

    auto stop = [this]() {
      LOG(INFO) << "Capacities: ROFRecords: " << mDecoder.getROFRecords().capacity() << "  data: " << mDecoder.getData().capacity();
      LOG(INFO) << "Read " << mHBCounter << " HBs in " << mElapsedTime.count() << " s";
    };
    ic.services().get<of::CallbackService>().set(of::CallbackService::Id::Stop, stop);
  }

  void run(of::ProcessingContext& pc)
  {
    if (mState != 0) {
      return;
    }

    auto tStart = std::chrono::high_resolution_clock::now();

    readTF();

    mDecoder.process(mBytes);

    pc.outputs().snapshot(of::Output{"MID", "DATA", 0, of::Lifetime::Timeframe}, mDecoder.getData());
    pc.outputs().snapshot(of::Output{"MID", "DATAROF", 0, of::Lifetime::Timeframe}, mDecoder.getROFRecords());
    LOG(DEBUG) << "Sent " << mDecoder.getData().size() << " column data";

    mElapsedTime += std::chrono::high_resolution_clock::now() - tStart;

    if (mState == 1) {
      pc.services().get<of::ControlService>().readyToQuit(of::QuitRequest::Me);
    }
  }

 private:
  void read(size_t nBytes)
  {
    size_t currentIndex = mBytes.size();
    mBytes.resize(currentIndex + nBytes / raw::sElementSizeInBytes);
    mFile.read(reinterpret_cast<char*>(&(mBytes[currentIndex])), nBytes);
  }

  bool readTF()
  {
    mBytes.clear();
    while (readBlock()) {
      ++mHBCounter;
      if (mHBCounter % mHBperTimeframe == 0) {
        return true;
      }
    }
    return false;
  }

  bool readBlock()
  {
    bool stop = false;
    while (!stop) {
      // Read header
      read(raw::sHeaderSizeInBytes);

      // The check on the eof needs to be placed here and not at the beginning of the function.
      // The reason is that the eof flag is set if we try to read after the eof
      // But, since we know the size, we read up to the last character.
      // So we turn on the eof flag only if we try to read past the last data.
      // Of course, we resized the mBytes before trying to read.
      // Since we read 0, we need to remove the last bytes
      if (mFile.eof()) {
        mBytes.resize(mBytes.size() - raw::sHeaderSizeInElements);
        mState = 1;
        return false;
      }
      const header::RAWDataHeader* rdh = reinterpret_cast<const header::RAWDataHeader*>(&(mBytes[mBytes.size() - raw::sHeaderSizeInElements]));
      stop = rdh->stop;
      if (rdh->offsetToNext > raw::sHeaderSizeInBytes) {
        read(rdh->offsetToNext - raw::sHeaderSizeInBytes);
      }
    }

    return true;
  }

  Decoder mDecoder{};
  std::ifstream mFile{};
  std::vector<raw::RawUnit> mBytes{};
  int mHBperTimeframe{256};
  unsigned long int mHBCounter{0};
  int mState{0};
  std::chrono::duration<double> mElapsedTime{0}; ///< timer
};

framework::DataProcessorSpec getRawReaderSpec()
{
  return of::DataProcessorSpec{
    "MIDRawReader",
    of::Inputs{},
    of::Outputs{of::OutputSpec{"MID", "DATA"}, of::OutputSpec{"MID", "DATAROF"}},
    of::AlgorithmSpec{of::adaptFromTask<o2::mid::RawReaderDeviceDPL>()},
    of::Options{
      {"mid-raw-infile", of::VariantType::String, "mid_raw.dat", {"Name of the input file"}},
      {"mid-hb-per-timeframe", of::VariantType::Int, 256, {"Number of heart beats per timeframe"}}}};
}
} // namespace mid
} // namespace o2