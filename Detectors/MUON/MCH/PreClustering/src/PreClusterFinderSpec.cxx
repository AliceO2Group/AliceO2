// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file PreClusterFinderSpec.cxx
/// \brief Implementation of a data processor to run the preclusterizer
///
/// \author Philippe Pillot, Subatech

#include "PreClusterFinderSpec.h"

#include <iostream>
#include <fstream>
#include <chrono>

#include <stdexcept>

#include "Framework/CallbackService.h"
#include "Framework/ControlService.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Lifetime.h"
#include "Framework/Output.h"
#include "Framework/Task.h"

#include "MCHBase/DigitBlock.h"
#include "MCHBase/PreClusterBlock.h"
#include "PreClusterFinder.h"

namespace o2
{
namespace mch
{

using namespace std;
using namespace o2::framework;

class PreClusterFinderTask
{
 public:
  //_________________________________________________________________________________________________
  void init(framework::InitContext& ic)
  {
    /// Prepare the preclusterizer
    LOG(INFO) << "initializing preclusterizer";

    // Load the mapping from the binary file
    auto fileName = ic.options().get<std::string>("binmapfile");
    try {
      mPreClusterFinder.init(fileName);
    } catch (exception const& e) {
      throw;
    }

    auto stop = [this]() {
      /// Clear the preclusterizer
      auto tStart = std::chrono::high_resolution_clock::now();
      this->mPreClusterFinder.deinit();
      auto tEnd = std::chrono::high_resolution_clock::now();
      LOG(INFO) << "deinitializing preclusterizer in: "
                << std::chrono::duration<double, std::milli>(tEnd - tStart).count() << " ms\n";
    };
    ic.services().get<CallbackService>().set(CallbackService::Id::Stop, stop);

    mPrint = ic.options().get<bool>("print");
  }

  //_________________________________________________________________________________________________
  void run(framework::ProcessingContext& pc)
  {
    /// read the digits, preclusterize and send the preclusters

    // prepare to receive new data
    mPreClusterFinder.reset();

    // get the input buffer
    auto msgIn = pc.inputs().get<gsl::span<char>>("digits");
    auto bufferPtrIn = msgIn.data();
    auto sizeIn = msgIn.size();

    // get header info and check message consistency
    if (sizeIn < SSizeOfDigitBlock) {
      throw out_of_range("missing DigitBlock");
    }
    auto digitBlock(reinterpret_cast<const DigitBlock*>(bufferPtrIn));
    bufferPtrIn += SSizeOfDigitBlock;
    sizeIn -= SSizeOfDigitBlock;
    if (digitBlock->header.fRecordWidth != SSizeOfDigitStruct) {
      throw length_error("incorrect size of digits. Corrupted message?");
    }
    if (sizeIn != digitBlock->header.fNrecords * SSizeOfDigitStruct) {
      throw length_error("incorrect payload");
    }

    // load the digits to get the fired pads
    auto digits(reinterpret_cast<const DigitStruct*>(bufferPtrIn));
    mPreClusterFinder.loadDigits(digits, digitBlock->header.fNrecords);

    // preclusterize
    int nPreClusters = mPreClusterFinder.run();

    // number of DEs with preclusters and total number of pads used
    int nUsedDigits(0);
    int nDEWithPreClusters = mPreClusterFinder.getNDEWithPreClusters(nUsedDigits);

    // create the output message of the exactly needed buffer size
    auto sizeOut = SSizeOfInt + nDEWithPreClusters * 2 * SSizeOfInt +
                   PreClusterBlock::sizeOfPreClusterBlocks(nDEWithPreClusters, nPreClusters, nUsedDigits);
    auto msgOut = pc.outputs().make<char>(Output{"MCH", "PRECLUSTERS", 0, Lifetime::Timeframe}, sizeOut);
    auto bufferPtrOut = msgOut.data();
    if (msgOut.size() != sizeOut) {
      throw length_error("incorrect message payload");
    }

    // store the number of DE with preclusters
    memcpy(bufferPtrOut, &nDEWithPreClusters, SSizeOfInt);
    bufferPtrOut += SSizeOfInt;
    sizeOut -= SSizeOfInt;

    // store preclusters
    try {
      storePreClusters(bufferPtrOut, sizeOut);
    } catch (exception const& e) {
      throw length_error(std::string("fail to store preclusters: ") + e.what());
    }
  }

 private:
  //_________________________________________________________________________________________________
  void storePreClusters(char* buffer, uint32_t size)
  {
    /// store the preclusters in the given buffer

    const PreClusterFinder::PreCluster* cluster(nullptr);
    const DigitStruct* digit(nullptr);
    uint32_t* bytesUsed(nullptr);
    uint32_t totalBytesUsed(0);

    for (int iDE = 0, nDEs = mPreClusterFinder.getNDEs(); iDE < nDEs; ++iDE) {

      if (!mPreClusterFinder.hasPreClusters(iDE)) {
        continue;
      }

      // store the DE ID
      if (size - totalBytesUsed >= SSizeOfInt) {
        auto deId(reinterpret_cast<int*>(buffer + totalBytesUsed));
        *deId = mPreClusterFinder.getDEId(iDE);
        totalBytesUsed += SSizeOfInt;
      } else {
        throw length_error("cannot store DE ID");
      }

      // prepare to store the size of the PreClusterBlock
      if (size - totalBytesUsed >= SSizeOfInt) {
        bytesUsed = reinterpret_cast<uint32_t*>(buffer + totalBytesUsed);
        totalBytesUsed += SSizeOfInt;
      } else {
        throw length_error("cannot store size of the PreClusterBlock");
      }

      // prepare to store the preclusters of this DE
      if (mPreClusterBlock.reset(buffer + totalBytesUsed, size - totalBytesUsed, true) < 0) {
        throw length_error("cannot reset the cluster block");
      }

      for (int iPlane = 0; iPlane < 2; ++iPlane) {
        for (int iCluster = 0, nClusters = mPreClusterFinder.getNPreClusters(iDE, iPlane);
             iCluster < nClusters; ++iCluster) {

          cluster = mPreClusterFinder.getPreCluster(iDE, iPlane, iCluster);
          if (!cluster->storeMe) {
            continue;
          }

          // add the precluster with its first digit
          digit = mPreClusterFinder.getDigit(iDE, cluster->firstPad);
          if (mPreClusterBlock.startPreCluster(*digit) < 0) {
            throw length_error("cannot store a new precluster");
          }

          // loop over other pads and add corresponding digits
          for (uint16_t iOrderedPad = cluster->firstPad + 1; iOrderedPad <= cluster->lastPad; ++iOrderedPad) {
            digit = mPreClusterFinder.getDigit(iDE, iOrderedPad);
            if (mPreClusterBlock.addDigit(*digit) < 0) {
              throw length_error("cannot store a new digit");
            }
          }
        }
      }

      // store the size of the PreClusterBlock
      *bytesUsed = mPreClusterBlock.getCurrentSize();
      totalBytesUsed += *bytesUsed;

      if (mPrint) {
        LOG(INFO) << "block: " << mPreClusterBlock;
      }
    }

    if (totalBytesUsed != size) {
      throw length_error("incorrect payload");
    }
  }

  static constexpr uint32_t SSizeOfInt = sizeof(int);
  static constexpr uint32_t SSizeOfDigitBlock = sizeof(DigitBlock);
  static constexpr uint32_t SSizeOfDigitStruct = sizeof(DigitStruct);

  bool mPrint = false;                  ///< print preclusters
  PreClusterFinder mPreClusterFinder{}; ///< preclusterizer
  PreClusterBlock mPreClusterBlock{};   ///< preclusters data blocks
};

//_________________________________________________________________________________________________
o2::framework::DataProcessorSpec getPreClusterFinderSpec()
{
  return DataProcessorSpec{
    "PreClusterFinder",
    Inputs{InputSpec{"digits", "MCH", "DIGITS", 0, Lifetime::Timeframe}},
    Outputs{OutputSpec{"MCH", "PRECLUSTERS", 0, Lifetime::Timeframe}},
    AlgorithmSpec{adaptFromTask<PreClusterFinderTask>()},
    Options{{"binmapfile", VariantType::String, "", {"binary mapping file name"}},
            {"print", VariantType::Bool, false, {"print preclusters"}}}};
}

} // end namespace mch
} // end namespace o2
