// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

///
/// \file    cru-page-reader-workflow.cxx
/// \author  Andrea Ferrero
///
/// \brief This is an executable that reads a data file from disk and sends the individual CRU pages via DPL.
///
/// This is an executable that reads a data file from disk and sends the individual CRU pages via the Data Processing Layer.
/// It can be used as a data source for O2 development. For example, one can do:
/// \code{.sh}
/// o2-mch-cru-page-reader-workflow --infile=some_data_file | o2-mch-raw-to-digits-workflow
/// \endcode
///

#include <random>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include "Framework/CallbackService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Lifetime.h"
#include "Framework/Output.h"
#include "Framework/Task.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/runDataProcessing.h"

#include "DPLUtils/DPLRawParser.h"
#include "Headers/RAWDataHeader.h"
#include "DetectorsRaw/RDHUtils.h"

using namespace o2;
using namespace o2::framework;

namespace o2
{
namespace mch
{
namespace raw
{

using RDH = o2::header::RDHAny;

class FileReaderTask
{
 public:
  //_________________________________________________________________________________________________
  void init(framework::InitContext& ic)
  {
    /// Get the input file and other options from the context
    LOG(INFO) << "initializing file reader";
    mFrameMax = ic.options().get<int>("nframes");
    mPrint = ic.options().get<bool>("print");
    mFullHBF = ic.options().get<bool>("full-hbf");

    auto inputFileName = ic.options().get<std::string>("infile");
    mInputFile.open(inputFileName, std::ios::binary);
    if (!mInputFile.is_open()) {
      throw std::invalid_argument("Cannot open input file \"" + inputFileName + "\"");
    }

    auto stop = [this]() {
      /// close the input file
      LOG(INFO) << "stop file reader";
      this->mInputFile.close();
    };
    ic.services().get<CallbackService>().set(CallbackService::Id::Stop, stop);
  }

  //_________________________________________________________________________________________________
  void run(framework::ProcessingContext& pc)
  {
    /// send one RDH block via DPL
    RDH rdh;
    char* buf{nullptr};
    size_t bufSize{0};

    while (true) {

      // stop if the required number of frames has been reached
      if (mFrameMax == 0) {
        pc.services().get<ControlService>().endOfStream();
        return;
      }

      if (mPrint) {
        printf("mFrameMax: %d\n", mFrameMax);
      }
      if (mFrameMax > 0) {
        mFrameMax -= 1;
      }

      // read the next RDH, stop if no more data is available
      mInputFile.read((char*)(&rdh), sizeof(RDH));
      if (mInputFile.fail()) {
        if (mPrint) {
          std::cout << "end of file reached" << std::endl;
        }
        pc.services().get<ControlService>().endOfStream();
        return; // probably reached eof
      }

      // check that the RDH version is ok (only RDH versions from 4 to 6 are supported at the moment)
      auto rdhVersion = o2::raw::RDHUtils::getVersion(rdh);
      auto rdhHeaderSize = o2::raw::RDHUtils::getHeaderSize(rdh);
      if (mPrint) {
        std::cout << "header_version=" << (int)rdhVersion << std::endl;
      }
      if (rdhVersion < 4 || rdhVersion > 6 || rdhHeaderSize != 64) {
        return;
      }

      // get the frame size from the RDH offsetToNext field
      auto frameSize = o2::raw::RDHUtils::getOffsetToNext(rdh);
      if (mPrint) {
        std::cout << "frameSize=" << frameSize << std::endl;
      }

      // stop if the frame size is too small
      if (frameSize < rdhHeaderSize) {
        std::cout << mFrameMax << " - frameSize too small: " << frameSize << std::endl;
        pc.services().get<ControlService>().endOfStream();
        return;
      }

      // allocate the output buffer
      buf = (char*)realloc(buf, bufSize + frameSize);
      if (buf == nullptr) {
        std::cout << mFrameMax << " - failed to allocate buffer" << std::endl;
        pc.services().get<ControlService>().endOfStream();
        return;
      }

      // copy the RDH into the output buffer
      memcpy(buf + bufSize, &rdh, rdhHeaderSize);

      // read the frame payload into the output buffer
      mInputFile.read(buf + bufSize + rdhHeaderSize, frameSize - rdhHeaderSize);

      // stop if data cannot be read completely
      if (mInputFile.fail()) {
        if (mPrint) {
          std::cout << "end of file reached" << std::endl;
        }
        free(buf);
        pc.services().get<ControlService>().endOfStream();
        return; // probably reached eof
      }

      // increment the total buffer size
      bufSize += frameSize;

      auto stopBit = o2::raw::RDHUtils::getStop(rdh);

      // when requesting full HBframes, the output message is sent only when the stop RDH is reached
      // otherwise we send one message for each CRU page
      if ((stopBit != 0) || (mFullHBF == false)) {
        // create the output message
        auto freefct = [](void* data, void* /*hint*/) { free(data); };
        pc.outputs().adoptChunk(Output{"ROUT", "RAWDATA"}, buf, bufSize, freefct, nullptr);

        // stop the readout loop
        break;
      }
    } // while (true)
  }

 private:
  std::ifstream mInputFile{}; ///< input file
  int mFrameMax;              ///< number of frames to process
  bool mFullHBF;              ///< send full HeartBeat frames
  bool mPrint = false;        ///< print debug messages
};

//_________________________________________________________________________________________________
// clang-format off
o2::framework::DataProcessorSpec getFileReaderSpec()
{
  return DataProcessorSpec{
    "FileReader",
    Inputs{},
    Outputs{OutputSpec{"ROUT", "RAWDATA", 0, Lifetime::Timeframe}},
    AlgorithmSpec{adaptFromTask<FileReaderTask>()},
    Options{{"infile", VariantType::String, "", {"input file name"}},
            {"nframes", VariantType::Int, -1, {"number of frames to process"}},
            {"full-hbf", VariantType::Bool, false, {"send full HeartBeat frames"}},
            {"print", VariantType::Bool, false, {"verbose output"}}}};
}
// clang-format on

} // end namespace raw
} // end namespace mch
} // end namespace o2

using namespace o2;
using namespace o2::framework;

WorkflowSpec defineDataProcessing(const ConfigContext&)
{
  WorkflowSpec specs;

  // The producer to generate some data in the workflow
  DataProcessorSpec producer = mch::raw::getFileReaderSpec();
  specs.push_back(producer);

  return specs;
}
