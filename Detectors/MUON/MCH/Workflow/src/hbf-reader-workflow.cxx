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
/// \file    hbf-reader-workflow.cxx
/// \author  Andrea Ferrero
///
/// \brief This is an executable that reads a data file from disk and sends the individual HB frames via DPL.
///
/// This is an executable that reads a data file from disk and sends the individual HB frames via the Data Processing Layer.
/// It can be used as a data source for O2 development. For example, one can do:
/// \code{.sh}
/// o2-mch-hbf-reader-workflow --infile=some_data_file | o2-mch-raw-to-digits-workflow
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

namespace o2::header
{
extern std::ostream& operator<<(std::ostream&, const o2::header::RAWDataHeaderV4&);
}

namespace o2
{
namespace mch
{
namespace raw
{

/// RAWDataHeaderV4V5V6
/// RDH structure with fields common to V4, V5 and V6
/// The common fields are enough to determine the size of the payload
/// and the offset of the next HB frame
/// Description of the fields can be found here
/// https://gitlab.cern.ch/AliceO2Group/wp6-doc/-/blob/master/rdh/RDHv6.md
//
///
///       63     56      48      40      32      24      16       8       0
///       |---------------|---------------|---------------|---------------|
///
///       | reserved              | priori|               |    header     |
/// 0     | reserve zero  |Source | ty bit|    FEE id     | size  |version|
///
/// 1     |ep | cru id    |pcount|link id |  memory size  |offset nxt pack|
///
struct RAWDataHeaderV4V5V6 {
  union {
    // default value
    uint64_t word0 = 0x00000000ffff4006;
    //                       | |     | version 6
    //                       | |   | 8x64 bit words = 64 (0x40) byte
    //                       | | invalid FEE id
    //                       | priority bit 0
    struct {
      uint64_t version : 8;       /// bit  0 to  7: header version
      uint64_t headerSize : 8;    /// bit  8 to 15: header size
      uint64_t feeId : 16;        /// bit 16 to 31: FEE identifier
      uint64_t priority : 8;      /// bit 32 to 39: priority bit
      uint64_t sourceID : 8;      /// bit 40 to 47: source ID
      uint64_t zero0 : 16;        /// bit 48 to 63: zeroed
    };                            ///
  };                              ///
  union {                         ///
    uint64_t word1 = 0x0;         /// data written by the CRU
    struct {                      ///
      uint32_t offsetToNext : 16; /// bit 64 to 79:  offset to next packet in memory
      uint32_t memorySize : 16;   /// bit 80 to 95:  memory size
      uint32_t linkID : 8;        /// bit 96 to 103: link id
      uint32_t packetCounter : 8; /// bit 104 to 111: packet counter
      uint16_t cruID : 12;        /// bit 112 to 123: CRU ID
      uint32_t endPointID : 4;    /// bit 124 to 127: DATAPATH WRAPPER ID: number used to
    };                            ///                 identify one of the 2 End Points [0/1]
  };                              ///
  union {                         ///
    uint64_t word2 = 0x0;         ///
  };                              ///
  union {                         ///
    uint64_t word3 = 0x0;         /// bit  0 to 63: zeroed
  };                              ///
  union {                         ///
    uint64_t word4 = 0x0;         ///
  };                              ///
  union {                         ///
    uint64_t word5 = 0x0;         /// bit  0 to 63: zeroed
  };                              ///
  union {                         ///
    uint64_t word6 = 0x0;         ///
  };                              ///
  union {                         ///
    uint64_t word7 = 0x0;         /// bit  0 to 63: zeroed
  };
};

using namespace o2;
using namespace o2::framework;
using RDH = RAWDataHeaderV4V5V6;

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

    // size of the HB frame to be sent
    int frameSize = {0};

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

    // read the next RDH
    mInputFile.read((char*)(&rdh), sizeof(RDH));
    if (mPrint) {
      std::cout << "header_version=" << (int)rdh.version << std::endl;
    }

    // only RDH versions from 4 to 6 are supported
    if (rdh.version < 4 || rdh.version > 6 || rdh.headerSize != 64) {
      return;
    }

    // get the frame size from the RDH offsetToNext field
    frameSize = rdh.offsetToNext;
    if (mPrint) {
      std::cout << "frameSize=" << frameSize << std::endl;
    }

    // allocate the output buffer
    char* buf = (char*)malloc(frameSize);

    // copy the RDH into the output buffer
    memcpy(buf, &rdh, rdh.headerSize);

    // read the frame payload into the output buffer
    mInputFile.read(buf + rdh.headerSize, frameSize - rdh.headerSize);

    // stop if data cannot be read completely
    if (mInputFile.fail()) {
      if (mPrint) {
        std::cout << "end of file reached" << std::endl;
      }
      free(buf);
      pc.services().get<ControlService>().endOfStream();
      return; // probably reached eof
    }

    // create the output message
    auto freefct = [](void* data, void* /*hint*/) { free(data); };
    pc.outputs().adoptChunk(Output{"ROUT", "RAWDATA"}, buf, frameSize, freefct, nullptr);
  }

 private:
  std::ifstream mInputFile{}; ///< input file
  int mFrameMax;              ///< number of frames to process
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
