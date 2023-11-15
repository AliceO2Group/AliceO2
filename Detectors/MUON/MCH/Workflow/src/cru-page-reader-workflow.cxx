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
#include <queue>
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

#include "DPLUtils/DPLRawParser.h"
#include "Headers/RAWDataHeader.h"
#include "DetectorsRaw/RDHUtils.h"
#include "DetectorsRaw/HBFUtils.h"
#include "CommonDataFormat/TFIDInfo.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::raw;

namespace o2
{
namespace mch
{
namespace raw
{

using RDH = o2::header::RDHAny;

static const int NFEEID = 64;
static const int NLINKS = 16;

struct TimeFrame {
  char* buf{nullptr};
  size_t tfSize{0};
  size_t totalSize{0};
  size_t payloadSize{0};
  uint32_t firstOrbit{0xFFFFFFFF};
  std::vector<std::pair<size_t, size_t>> hbframes;

  void computePayloadSize()
  {
    payloadSize = 0;
    if (buf == nullptr) {
      return;
    }

    size_t offset = 0;
    while (offset < totalSize) {
      char* ptr = buf + offset;
      RDH* rdh = (RDH*)ptr;

      auto rdhVersion = o2::raw::RDHUtils::getVersion(rdh);
      auto rdhHeaderSize = o2::raw::RDHUtils::getHeaderSize(rdh);
      auto memorySize = o2::raw::RDHUtils::getMemorySize(rdh);
      auto pageSize = o2::raw::RDHUtils::getOffsetToNext(rdh);

      payloadSize += memorySize - rdhHeaderSize;

      offset += pageSize;
    }
  }

  void print()
  {
    if (buf == nullptr) {
      return;
    }

    int nPrinted = 0;

    printf("\n//////////////////////\n");
    size_t offset = 0;
    size_t nStop = 0;
    while (offset < totalSize) {
      char* ptr = buf + offset;
      RDH* rdh = (RDH*)ptr;

      auto stopBit = o2::raw::RDHUtils::getStop(rdh);
      auto pageSize = o2::raw::RDHUtils::getOffsetToNext(rdh);
      if (stopBit > 0) {
        nStop += 1;
      }

      offset += pageSize;
    }

    offset = 0;
    bool doPrint = false;
    size_t iStop = 0;
    while (offset < totalSize) {
      char* ptr = buf + offset;
      RDH* rdh = (RDH*)ptr;

      auto rdhVersion = o2::raw::RDHUtils::getVersion(rdh);
      uint16_t cruID = o2::raw::RDHUtils::getCRUID(rdh) & 0x3F;
      uint8_t endPointID = o2::raw::RDHUtils::getEndPointID(rdh);
      uint8_t linkID = o2::raw::RDHUtils::getLinkID(rdh);
      uint16_t feeID = cruID * 2 + endPointID;
      auto stopBit = o2::raw::RDHUtils::getStop(rdh);
      auto triggerType = o2::raw::RDHUtils::getTriggerType(rdh);
      auto pageSize = o2::raw::RDHUtils::getOffsetToNext(rdh);

      if (iStop < 2 || iStop > (nStop - 3)) {

        printf("%6d:  version %X  offset %4d  packet %3d  srcID %d  cruID %2d  dp %d  link %2d  orbit %u  bc %4d  trig 0x%08X  page %d  stop %d",
               (int)0, (int)rdhVersion, (int)pageSize,
               (int)RDHUtils::getPacketCounter(rdh), (int)RDHUtils::getSourceID(rdh),
               (int)cruID, (int)endPointID, (int)linkID,
               (uint32_t)RDHUtils::getHeartBeatOrbit(rdh), (int)RDHUtils::getTriggerBC(rdh),
               (int)triggerType, (int)RDHUtils::getPageCounter(rdh), (int)stopBit);
        if ((triggerType & 0x800) != 0) {
          printf(" <===");
        }
        printf("\n");
      }
      if (stopBit > 0 && iStop == 3) {
        printf("........................\n");
      }

      if (stopBit > 0) {
        iStop += 1;
      }

      offset += pageSize;
    }
    fmt::printf("total size: {}\n", totalSize);
    printf("//////////////////////\n");
  }
};

using TFQueue = std::queue<TimeFrame>;

TFQueue tfQueues[NFEEID][NLINKS];

class FileReaderTask
{
 public:
  //_________________________________________________________________________________________________
  void init(framework::InitContext& ic)
  {
    /// Get the input file and other options from the context
    LOG(info) << "initializing file reader";
    mFrameMax = ic.options().get<int>("nframes");
    mTimeFrameMax = ic.options().get<int>("max-time-frame");
    mPrint = ic.options().get<bool>("print");
    mFullHBF = ic.options().get<bool>("full-hbf");
    mFullTF = ic.options().get<bool>("full-tf");
    mSaveTF = ic.options().get<bool>("save-tf");
    mOverlap = ic.options().get<int>("overlap");

    auto inputFileName = ic.options().get<std::string>("infile");
    mInputFile.open(inputFileName, std::ios::binary);
    if (!mInputFile.is_open()) {
      throw std::invalid_argument("Cannot open input file \"" + inputFileName + "\"");
    }

    auto stop = [this]() {
      /// close the input file
      LOG(info) << "stop file reader";
      this->mInputFile.close();
    };
    ic.services().get<CallbackService>().set<CallbackService::Id::Stop>(stop);

    const auto& hbfu = o2::raw::HBFUtils::Instance();
    if (hbfu.runNumber != 0) {
      mTFIDInfo.runNumber = hbfu.runNumber;
    }
    if (hbfu.orbitFirst != 0) {
      mTFIDInfo.firstTForbit = hbfu.orbitFirst;
    }
    if (hbfu.startTime != 0) {
      mTFIDInfo.creation = hbfu.startTime;
    }
  }

  void printHBF(char* framePtr, size_t frameSize)
  {
    size_t pageStart = 0;
    std::cout << "----\n";
    while (pageStart < frameSize) {
      RDH* rdh = (RDH*)(&(framePtr[pageStart]));
      // check that the RDH version is ok (only RDH versions from 4 to 6 are supported at the moment)
      auto rdhVersion = o2::raw::RDHUtils::getVersion(rdh);
      auto rdhHeaderSize = o2::raw::RDHUtils::getHeaderSize(rdh);
      uint16_t cruID = o2::raw::RDHUtils::getCRUID(rdh) & 0x3F;
      uint8_t endPointID = o2::raw::RDHUtils::getEndPointID(rdh);
      uint8_t linkID = o2::raw::RDHUtils::getLinkID(rdh);
      uint16_t feeID = cruID * 2 + endPointID;
      auto stopBit = o2::raw::RDHUtils::getStop(rdh);
      auto triggerType = o2::raw::RDHUtils::getTriggerType(rdh);
      auto pageSize = o2::raw::RDHUtils::getOffsetToNext(rdh);
      auto pageCounter = RDHUtils::getPageCounter(rdh);

      printf("%6d:  V %X  offset %4d  packet %3d  srcID %d  cruID %2d  dp %d  link %2d  orbit %u  bc %4d  trig 0x%08X  p %d  s %d",
             (int)0, (int)rdhVersion, (int)pageSize,
             (int)RDHUtils::getPacketCounter(rdh), (int)RDHUtils::getSourceID(rdh),
             (int)cruID, (int)endPointID, (int)linkID,
             (uint32_t)RDHUtils::getHeartBeatOrbit(rdh), (int)RDHUtils::getTriggerBC(rdh),
             (int)triggerType, (int)pageCounter, (int)stopBit);
      if ((triggerType & 0x800) != 0) {
        printf(" <===");
      }
      printf("\n");
      pageStart += pageSize;
    }
    std::cout << "----\n";
  }

  bool appendHBF(TimeFrame& tf, char* framePtr, size_t frameSize, bool addHBF)
  {
    // new size of the TimeFrame buffer after appending the HBFrame
    size_t newSize = tf.totalSize + frameSize;
    // increase the size of the memory buffer
    tf.buf = (char*)realloc(tf.buf, newSize);
    if (tf.buf == nullptr) {
      std::cout << "failed to allocate TimeFrame buffer" << std::endl;
      return false;
    }

    // copy the HBFrame into the TimeFrame buffer
    char* bufPtr = tf.buf + tf.totalSize;
    memcpy(bufPtr, framePtr, frameSize);

    if (addHBF) {
      // add the offset and size of the HBFrame to the vector
      tf.hbframes.emplace_back(std::make_pair(tf.totalSize, frameSize));
      // increase the  TimeFrame sizes
      tf.tfSize += frameSize;
    }

    // increase the total buffer sizes
    tf.totalSize += frameSize;

    return true;
  }

  //_________________________________________________________________________________________________
  bool sendTF(framework::ProcessingContext& pc)
  {
    uint32_t orbitMin = 0xFFFFFFFF;
    int maxQueueSize = 0;
    for (int feeId = 0; feeId < NFEEID; feeId++) {
      for (int linkId = 0; linkId < NLINKS; linkId++) {
        TFQueue& tfQueue = tfQueues[feeId][linkId];
        if (tfQueue.empty()) {
          continue;
        }
        if (mPrint) {
          std::cout << fmt::format("FEE ID {}  LINK {}   orbit {}    queue size {}", feeId, linkId, tfQueue.front().firstOrbit, tfQueue.size()) << std::endl;
        }
        if (tfQueue.front().firstOrbit < orbitMin) {
          orbitMin = tfQueue.front().firstOrbit;
        }
        if (tfQueue.size() > maxQueueSize) {
          maxQueueSize = tfQueue.size();
        }
      }
    }

    if (maxQueueSize < 3) {
      return false;
    }

    char* outBuf{nullptr};
    size_t outSize{0};
    for (int feeId = 0; feeId < NFEEID; feeId++) {
      for (int linkId = 0; linkId < NLINKS; linkId++) {
        TFQueue& tfQueue = tfQueues[feeId][linkId];
        TimeFrame& tf = tfQueue.front();
        if (tf.firstOrbit != orbitMin) {
          continue;
        }

        size_t newSize = outSize + tf.totalSize;
        // increase the size of the memory buffer
        outBuf = (char*)realloc(outBuf, newSize);
        if (outBuf == nullptr) {
          std::cout << "failed to allocate output buffer of size " << newSize << " bytes" << std::endl;
          return false;
        }
        if (mPrint) {
          std::cout << fmt::format("Appending FEE ID {}  LINK {}   orbit {} to current TF", feeId, linkId, tf.firstOrbit) << std::endl;
        }
        // copy the SubTimeFrame into the TimeFrame buffer
        char* bufPtr = outBuf + outSize;
        memcpy(bufPtr, tf.buf, tf.totalSize);

        outSize += tf.totalSize;
        tfQueue.pop();
      }
    }

    if (mPrint) {
      std::cout << "Sending TF " << orbitMin << " (previous " << mLastTForbit << "  delta " << (orbitMin - mLastTForbit) << ")" << std::endl
                << std::endl;
    }
    mLastTForbit = orbitMin;
    auto freefct = [](void* data, void* /*hint*/) { free(data); };
    pc.outputs().adoptChunk(Output{"RDT", "RAWDATA"}, outBuf, outSize, freefct, nullptr);

    return true;
  }

  //_________________________________________________________________________________________________
  void appendSTF(framework::ProcessingContext& pc)
  {
    /// send one RDH block via DPL
    RDH rdh;

    static int TFid = 0;

    if (mTimeFrameMax > 0 && TFid == mTimeFrameMax) {
      pc.services().get<ControlService>().endOfStream();
      return;
    }

    while (true) {

      // stop if the required number of frames has been reached
      if (mFrameMax == 0) {
        pc.services().get<ControlService>().endOfStream();
        return;
      }

      if (mPrint && false) {
        printf("mFrameMax: %d\n", mFrameMax);
      }
      if (mFrameMax > 0) {
        mFrameMax -= 1;
      }

      // read the next RDH, stop if no more data is available
      if (mPrint) {
        std::cout << "Reading " << sizeof(RDH) << " for RDH from input file\n";
      }
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
      uint16_t cruID = o2::raw::RDHUtils::getCRUID(rdh) & 0x3F;
      uint8_t endPointID = o2::raw::RDHUtils::getEndPointID(rdh);
      uint8_t linkID = o2::raw::RDHUtils::getLinkID(rdh);
      uint16_t feeID = cruID * 2 + endPointID;
      auto stopBit = o2::raw::RDHUtils::getStop(rdh);
      auto triggerType = o2::raw::RDHUtils::getTriggerType(rdh);
      auto pageSize = o2::raw::RDHUtils::getOffsetToNext(rdh);
      auto pageCounter = RDHUtils::getPageCounter(rdh);
      auto orbit = RDHUtils::getHeartBeatOrbit(rdh);
      int bc = (int)RDHUtils::getTriggerBC(rdh);

      if (mPrint) {
        printf("%6d:  V %X  offset %4d  packet %3d  srcID %d  cruID %2d  dp %d  link %2d  orbit %u  bc %4d  trig 0x%08X  p %d  s %d",
               (int)0, (int)rdhVersion, (int)pageSize,
               (int)RDHUtils::getPacketCounter(rdh), (int)RDHUtils::getSourceID(rdh),
               (int)cruID, (int)endPointID, (int)linkID,
               orbit, bc, (int)triggerType, (int)pageCounter, (int)stopBit);
        if ((triggerType & 0x800) != 0) {
          printf(" <===");
        }
        printf("\n");
      }
      if (rdhVersion < 4 || rdhVersion > 6 || rdhHeaderSize != 64) {
        return;
      }

      TFQueue& tfQueue = tfQueues[feeID][linkID];

      // get the frame size from the RDH offsetToNext field
      if (mPrint && false) {
        std::cout << "pageSize=" << pageSize << std::endl;
      }

      // stop if the frame size is too small
      if (pageSize < rdhHeaderSize) {
        std::cout << mFrameMax << " - pageSize too small: " << pageSize << std::endl;
        pc.services().get<ControlService>().endOfStream();
        return;
      }

      // allocate or extend the output buffer
      mTimeFrameBufs[feeID][linkID] = (char*)realloc(mTimeFrameBufs[feeID][linkID], mTimeFrameSizes[feeID][linkID] + pageSize);
      if (mTimeFrameBufs[feeID][linkID] == nullptr) {
        std::cout << mFrameMax << " - failed to allocate buffer" << std::endl;
        pc.services().get<ControlService>().endOfStream();
        return;
      }

      if (mPrint) {
        std::cout << "Copying RDH into buf " << (int)feeID << "," << (int)linkID << std::endl;
        std::cout << "  frame size: " << mTimeFrameSizes[feeID][linkID] << std::endl;
      }
      // copy the RDH into the output buffer
      memcpy(mTimeFrameBufs[feeID][linkID] + mTimeFrameSizes[feeID][linkID], &rdh, rdhHeaderSize);

      // read the frame payload into the output buffer
      if (mPrint) {
        std::cout << "Reading " << pageSize - rdhHeaderSize << " for payload from input file\n";
        std::cout << "Copying payload into buf " << (int)feeID << "," << (int)linkID << std::endl;
        std::cout << "  frame size: " << mTimeFrameSizes[feeID][linkID] << std::endl;
      }
      mInputFile.read(mTimeFrameBufs[feeID][linkID] + mTimeFrameSizes[feeID][linkID] + rdhHeaderSize, pageSize - rdhHeaderSize);

      // stop if data cannot be read completely
      if (mInputFile.fail()) {
        if (mPrint) {
          std::cout << "end of file reached" << std::endl;
        }
        pc.services().get<ControlService>().endOfStream();
        return; // probably reached eof
      }

      // increment the total buffer size
      mTimeFrameSizes[feeID][linkID] += pageSize;

      if ((triggerType & 0x800) != 0 && /*stopBit == 0 && pageCounter == 0 &&*/ bc == 0) {
        // This is the start of a new TimeFrame, so we need to push a new empty TimeFrame in the queue
        if (mPrint) {
          std::cout << "tfQueue.size(): " << tfQueue.size() << std::endl;
        }
        tfQueue.emplace();
        tfQueue.back().firstOrbit = orbit;
      }

      if (stopBit && tfQueue.size() > 0) {
        // we reached the end of the current HBFrame, we need to append it to the TimeFrame

        if (mPrint) {
          std::cout << "Appending HBF from " << (int)feeID << "," << (int)linkID << " to TF #" << tfQueue.size() << std::endl;
          std::cout << "  frame size: " << mTimeFrameSizes[feeID][linkID] << std::endl;
          printHBF(mTimeFrameBufs[feeID][linkID], mTimeFrameSizes[feeID][linkID]);
        }
        if (!appendHBF(tfQueue.back(), mTimeFrameBufs[feeID][linkID], mTimeFrameSizes[feeID][linkID], true)) {
          std::cout << mFrameMax << " - failed to append HBframe" << std::endl;
          pc.services().get<ControlService>().endOfStream();
          return;
        }

        // free the HBFrame buffer
        free(mTimeFrameBufs[feeID][linkID]);
        mTimeFrameBufs[feeID][linkID] = nullptr;
        mTimeFrameSizes[feeID][linkID] = 0;

        if (sendTF(pc)) {
          break;
        }
      }
    }
  }

  //_________________________________________________________________________________________________
  void run(framework::ProcessingContext& pc)
  {
    setMessageHeader(pc, mTFIDInfo);

    if (mFullTF) {
      appendSTF(pc);
      // pc.services().get<ControlService>().endOfStream();
      return;
    }

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
        pc.outputs().adoptChunk(Output{"RDT", "RAWDATA"}, buf, bufSize, freefct, nullptr);

        // stop the readout loop
        break;
      }
    } // while (true)
  }

  void setMessageHeader(ProcessingContext& pc, const o2::dataformats::TFIDInfo& tfid) const
  {
    auto& timingInfo = pc.services().get<o2::framework::TimingInfo>();
    if (tfid.firstTForbit != -1U) {
      timingInfo.firstTForbit = tfid.firstTForbit;
    }
    if (tfid.tfCounter != -1U) {
      timingInfo.tfCounter = tfid.tfCounter;
    }
    if (tfid.runNumber != -1U) {
      timingInfo.runNumber = tfid.runNumber;
    }
    if (tfid.creation != -1U) {
      timingInfo.creation = tfid.creation;
    }
    // LOGP(info, "TimingInfo set to : firstTForbit {}, tfCounter {}, runNumber {}, creatio {}",  timingInfo.firstTForbit, timingInfo.tfCounter, timingInfo.runNumber, timingInfo.creation);
  }

 private:
  std::ifstream mInputFile{}; ///< input file
  int mFrameMax;              ///< number of frames to process
  int mTimeFrameMax;          ///< number of frames to process
  bool mFullHBF;              ///< send full HeartBeat frames
  bool mFullTF;               ///< send full time frames
  bool mSaveTF;               ///< save individual time frames to file
  int mOverlap;               ///< overlap between contiguous TimeFrames
  int mLastTForbit{0};        ///< first orbit number of last transmitted TimeFrame
  bool mPrint = false;        ///< print debug messages
  o2::dataformats::TFIDInfo mTFIDInfo{}; // struct to modify output headers

  char* mTimeFrameBufs[NFEEID][NLINKS] = {nullptr};
  size_t mTimeFrameSizes[NFEEID][NLINKS] = {0};
};

//_________________________________________________________________________________________________
// clang-format off
o2::framework::DataProcessorSpec getFileReaderSpec(const char* specName)
{
  return DataProcessorSpec{
    specName,
    Inputs{},
    Outputs{OutputSpec{"RDT", "RAWDATA", 0, Lifetime::Sporadic}},
    AlgorithmSpec{adaptFromTask<FileReaderTask>()},
    Options{{"infile", VariantType::String, "", {"input file name"}},
            {"nframes", VariantType::Int, -1, {"number of frames to process"}},
            {"max-time-frame", VariantType::Int, -1, {"number of time frames to process"}},
            {"full-hbf", VariantType::Bool, false, {"send full HeartBeat frames"}},
            {"full-tf", VariantType::Bool, false, {"send full time frames"}},
            {"save-tf", VariantType::Bool, false, {"save individual time frames to file"}},
            {"overlap", VariantType::Int, 0, {"overlap between contiguous TimeFrames"}},
            {"print", VariantType::Bool, false, {"verbose output"}}}};
}
// clang-format on

} // end namespace raw
} // end namespace mch
} // end namespace o2

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  std::vector<ConfigParamSpec> options{{"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings"}}};
  workflowOptions.insert(workflowOptions.end(), options.begin(), options.end());
}

#include "Framework/runDataProcessing.h"

using namespace o2;
using namespace o2::framework;

WorkflowSpec defineDataProcessing(const ConfigContext& cfgc)
{
  o2::conf::ConfigurableParam::updateFromString(cfgc.options().get<std::string>("configKeyValues"));
  WorkflowSpec specs;

  // The producer to generate some data in the workflow
  DataProcessorSpec producer = mch::raw::getFileReaderSpec("mch-cru-page-reader");
  specs.push_back(producer);

  return specs;
}
