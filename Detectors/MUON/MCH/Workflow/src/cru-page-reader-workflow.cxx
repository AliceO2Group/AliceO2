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
#include "Framework/DataProcessorSpec.h"
#include "Framework/runDataProcessing.h"

#include "DPLUtils/DPLRawParser.h"
#include "Headers/RAWDataHeader.h"
#include "DetectorsRaw/RDHUtils.h"

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
    LOG(INFO) << "initializing file reader";
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
      LOG(INFO) << "stop file reader";
      this->mInputFile.close();
    };
    ic.services().get<CallbackService>().set(CallbackService::Id::Stop, stop);
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
  void sendTF(framework::ProcessingContext& pc)
  {
    /// send one RDH block via DPL
    RDH rdh;
    char* buf{nullptr};
    size_t frameSize{0};

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

      if (mPrint) {
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
      buf = (char*)realloc(buf, frameSize + pageSize);
      if (buf == nullptr) {
        std::cout << mFrameMax << " - failed to allocate buffer" << std::endl;
        pc.services().get<ControlService>().endOfStream();
        return;
      }

      // copy the RDH into the output buffer
      memcpy(buf + frameSize, &rdh, rdhHeaderSize);

      // read the frame payload into the output buffer
      if (mPrint) {
        std::cout << "Reading " << pageSize - rdhHeaderSize << " for payload from input file\n";
      }
      mInputFile.read(buf + frameSize + rdhHeaderSize, pageSize - rdhHeaderSize);

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
      frameSize += pageSize;

      if ((triggerType & 0x800) != 0 && stopBit == 0 && pageCounter == 0) {
        // This is the start of a new TimeFrame, so we need to take some actions:
        // - push a new TimeFrame in the queue
        // - append the last N HBFrames of the previous TimeFrame at the beginning of the current one
        // - set the initial total size of the TimeFrame buffer
        char* prevTFptr{nullptr};
        size_t prevTFsize{0};

        if (mPrint) {
          std::cout << "tfQueue.size(): " << tfQueue.size() << std::endl;
        }
        if (!tfQueue.empty()) {
          TimeFrame& prevTF = tfQueue.back();
          size_t nhbf = prevTF.hbframes.size();
          if ((mOverlap > 0) && (nhbf >= mOverlap)) {
            size_t hbfID = nhbf - mOverlap;
            prevTFptr = prevTF.buf + prevTF.hbframes[hbfID].first;
            for (size_t i = hbfID; i < nhbf; i++) {
              prevTFsize += prevTF.hbframes[i].second;
            }
          }
        }

        tfQueue.emplace();

        if (prevTFsize > 0) {
          tfQueue.back().buf = (char*)malloc(prevTFsize);
          if (tfQueue.back().buf == nullptr) {
            std::cout << mFrameMax << " - failed to allocate TimeFrame buffer" << std::endl;
            pc.services().get<ControlService>().endOfStream();
            return;
          }

          memcpy(tfQueue.back().buf, prevTFptr, prevTFsize);
        }

        tfQueue.back().totalSize = prevTFsize;
      }

      if (stopBit && tfQueue.size() > 0) {
        // we reached the end of the current HBFrame, we need to append it to the TimeFrame

        if (mPrint) {
          std::cout << "Appending HBF to TF #" << tfQueue.size() << std::endl;
          printHBF(buf, frameSize);
        }
        if (!appendHBF(tfQueue.back(), buf, frameSize, true)) {
          std::cout << mFrameMax << " - failed to append HBframe" << std::endl;
          pc.services().get<ControlService>().endOfStream();
          return;
        }

        if (tfQueue.size() == 2) {
          // we have two TimeFrames in the queue, we also append mOverlap HBFrames to the first one
          if (tfQueue.back().hbframes.size() <= mOverlap) {
            if (mPrint) {
              std::cout << "Appending HBF to TF #1" << std::endl;
              printHBF(buf, frameSize);
            }
            if (!appendHBF(tfQueue.front(), buf, frameSize, false)) {
              std::cout << mFrameMax << " - failed to append HBframe" << std::endl;
              pc.services().get<ControlService>().endOfStream();
              return;
            }
          }
        }

        // free the HBFrame buffer
        free(buf);
        buf = nullptr;
        frameSize = 0;

        if (tfQueue.size() == 2 && tfQueue.back().hbframes.size() >= mOverlap) {
          // we collected enough HBFrames after the last fully recorded TimeFrame, so we can send it
          tfQueue.front().computePayloadSize();
          if (mPrint) {
            tfQueue.front().print();
            sleep(1);
          }
          if (tfQueue.front().payloadSize > 0) {
            if (mSaveTF && TFid < 100) {
              char fname[500];
              snprintf(fname, 499, "tf-%03d.raw", TFid);
              FILE* fout = fopen(fname, "wb");
              if (fout) {
                fwrite(tfQueue.front().buf, tfQueue.front().totalSize, 1, fout);
                fclose(fout);
              }
            }

            auto freefct = [](void* data, void* /*hint*/) { free(data); };
            pc.outputs().adoptChunk(Output{"RDT", "RAWDATA"}, tfQueue.front().buf, tfQueue.front().totalSize, freefct, nullptr);
            TFid += 1;
          }
          tfQueue.pop();
          break;
        }
      }
    }
  }

  //_________________________________________________________________________________________________
  void run(framework::ProcessingContext& pc)
  {
    if (mFullTF) {
      sendTF(pc);
      //pc.services().get<ControlService>().endOfStream();
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

 private:
  std::ifstream mInputFile{}; ///< input file
  int mFrameMax;              ///< number of frames to process
  int mTimeFrameMax;          ///< number of frames to process
  bool mFullHBF;              ///< send full HeartBeat frames
  bool mFullTF;               ///< send full time frames
  bool mSaveTF;               ///< save individual time frames to file
  int mOverlap;               ///< overlap between contiguous TimeFrames
  bool mPrint = false;        ///< print debug messages
};

//_________________________________________________________________________________________________
// clang-format off
o2::framework::DataProcessorSpec getFileReaderSpec()
{
  return DataProcessorSpec{
    "FileReader",
    Inputs{},
    Outputs{OutputSpec{"RDT", "RAWDATA", 0, Lifetime::Timeframe}},
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
