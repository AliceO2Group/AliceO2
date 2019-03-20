// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "PreClusterFinderDevice.h"

#include <chrono>
#include <memory>
#include <stdexcept>
#include <vector>

#include <FairMQLogger.h>
#include <FairMQMessage.h>
#include <options/FairMQProgOptions.h> // device->fConfig
#include "MCHBase/DigitBlock.h"
#include "MCHBase/PreClusterBlock.h"

namespace o2
{
namespace mch
{

using namespace o2::alice_hlt;

using namespace std;

//_________________________________________________________________________________________________
PreClusterFinderDevice::PreClusterFinderDevice() : FairMQDevice()
{
  /// Constructor

  // register a handler for data arriving on "data-in" channel
  OnData("data-in", &PreClusterFinderDevice::processData);
}

//_________________________________________________________________________________________________
void PreClusterFinderDevice::InitTask()
{
  /// Prepare the preclusterizer

  // Get the binary mapping file from the command line option (via fConfig)
  auto fileName = fConfig->GetValue<std::string>("binmapfile");

  // Load the mapping from the binary file
  try {
    mPreClusterFinder.init(fileName);
  } catch (exception const& e) {
    LOG(ERROR) << "PreClusterFinder::InitTask() failed: " << e.what() << ", going to ERROR state.";
    ChangeState(ERROR_FOUND);
    return;
  }
}

//_________________________________________________________________________________________________
void PreClusterFinderDevice::ResetTask()
{
  /// Clear the preclusterizer

  auto tStart = std::chrono::high_resolution_clock::now();

  mPreClusterFinder.deinit();

  auto tEnd = std::chrono::high_resolution_clock::now();
  LOG(INFO) << "Resetting task in: " << std::chrono::duration<double, std::milli>(tEnd - tStart).count() << " ms\n";
}

//_________________________________________________________________________________________________
bool PreClusterFinderDevice::processData(FairMQMessagePtr& msg, int /*index*/)
{
  /// handler is called whenever a message arrives on "data-in",
  /// with a reference to the message and a sub-channel index (here 0)

  static const AliHLTComponentDataType digitBlockDataType = AliHLTComponentDataTypeInitializer("DIGITS  ", "MUON");

  // prepare to receive new data
  mFormatHandler.clear();
  mPreClusterFinder.reset();
  bool validBlockFound(false);

  if (msg->GetSize() > 0) {

    // convert the message in ALICE HLT format
    if (mFormatHandler.addMessage(reinterpret_cast<AliHLTUInt8_t*>(msg->GetData()), msg->GetSize()) > 0) {

      vector<BlockDescriptor>& inputBlocks = mFormatHandler.getBlockDescriptors();
      for (const auto& block : inputBlocks) {

        // only digit blocks
        if (!MatchExactly(block.fDataType, digitBlockDataType)) {
          LOG(INFO) << "not a MUON digit block: " << block.fDataType.fID;
          continue;
        }

        // load the digits to get the fired pads
        auto digitBlock(reinterpret_cast<const DigitBlock*>(block.fPtr));
        auto digits(reinterpret_cast<const DigitStruct*>(digitBlock + 1));
        mPreClusterFinder.loadDigits(digits, digitBlock->header.fNrecords);

        validBlockFound = true;
      }

    } else {
      LOG(WARN) << "no valid data blocks in message";
    }

  } else {
    LOG(WARN) << "ignoring message with payload of size 0";
  }

  if (validBlockFound) {

    // preclusterize
    int nPreClusters = mPreClusterFinder.run();

    if (nPreClusters > 0) {

      // number of DEs with preclusters and total number of pads used
      int nUsedDigits(0);
      int nDEWithPreClusters = mPreClusterFinder.getNDEWithPreClusters(nUsedDigits);

      // create message of the exactly needed buffer size
      auto size = nDEWithPreClusters * sizeof(AliHLTComponentBlockData) +
                  PreClusterBlock::sizeOfPreClusterBlocks(nDEWithPreClusters, nPreClusters, nUsedDigits);
      FairMQMessagePtr msgOut(NewMessage(size));

      // store preclusters
      try {
        storePreClusters(reinterpret_cast<uint8_t*>(msgOut->GetData()), msgOut->GetSize());
      } catch (exception const& e) {
        LOG(ERROR) << "PreClusterFinder::processData() failed storing preclusters: " << e.what();
        return false;
      }

      // Send out the output message
      if (Send(msgOut, "data-out") < 0) {
        LOG(ERROR) << "problem sending message";
        return false;
      }
    }
  }

  // return true if want to be called again (otherwise go to IDLE state)
  return true;
}

//_________________________________________________________________________________________________
void PreClusterFinderDevice::fillBlockData(AliHLTComponentBlockData& blockData)
{
  /// Fill AliHLTComponentBlockData structure with default values

  blockData.fStructSize = sizeof(blockData);
  blockData.fShmKey.fStructSize = sizeof(blockData.fShmKey);
  blockData.fShmKey.fShmType = gkAliHLTComponentInvalidShmType;
  blockData.fShmKey.fShmID = gkAliHLTComponentInvalidShmID;
  blockData.fOffset = 0;
  blockData.fPtr = nullptr;
  blockData.fSize = 0;
  blockData.fDataType = AliHLTComponentDataTypeInitializer("PRECLUST", "MUON");
  blockData.fSpecification = kAliHLTVoidDataSpec;
}

//_________________________________________________________________________________________________
void PreClusterFinderDevice::storePreClusters(uint8_t* buffer, uint32_t size)
{
  /// store the preclusters in the given buffer

  const PreClusterFinder::PreCluster* cluster(nullptr);
  const DigitStruct* digit(nullptr);
  uint32_t totalBytesUsed(0);

  // loop over DEs
  for (int iDE = 0, nDEs = mPreClusterFinder.getNDEs(); iDE < nDEs; ++iDE) {

    if (!mPreClusterFinder.hasPreClusters(iDE)) {
      continue;
    }

    // create the preclusters data block for this DE
    AliHLTComponentBlockData* blockData(nullptr);
    if (size - totalBytesUsed >= sizeof(AliHLTComponentBlockData)) {
      blockData = reinterpret_cast<AliHLTComponentBlockData*>(buffer + totalBytesUsed);
      fillBlockData(*blockData);
      totalBytesUsed += sizeof(AliHLTComponentBlockData);
    } else {
      LOG(ERROR) << "The buffer is too small to store the data block.";
      throw overflow_error("The buffer is too small to store the data block.");
    }

    if (mPreClusterBlock.reset(buffer + totalBytesUsed, size - totalBytesUsed, true) < 0) {
      throw runtime_error("Cannot reset the cluster block.");
    }

    // loop over planes
    for (int iPlane = 0; iPlane < 2; ++iPlane) {

      // loop over preclusters
      for (int iCluster = 0, nClusters = mPreClusterFinder.getNPreClusters(iDE, iPlane); iCluster < nClusters;
           ++iCluster) {

        cluster = mPreClusterFinder.getPreCluster(iDE, iPlane, iCluster);
        if (!cluster->storeMe) {
          continue;
        }

        // add the precluster with its first digit
        digit = mPreClusterFinder.getDigit(iDE, cluster->firstPad);
        if (mPreClusterBlock.startPreCluster(*digit) < 0) {
          throw runtime_error("Cannot store a new precluster.");
        }

        // loop over other pads and add corresponding digits
        for (uint16_t iOrderedPad = cluster->firstPad + 1; iOrderedPad <= cluster->lastPad; ++iOrderedPad) {
          digit = mPreClusterFinder.getDigit(iDE, iOrderedPad);
          if (mPreClusterBlock.addDigit(*digit) < 0) {
            throw runtime_error("Cannot store a new digit.");
          }
        }
      }
    }

    // complete the data block information
    uint32_t bytesUsed(mPreClusterBlock.getCurrentSize());
    blockData->fSize = bytesUsed;
    blockData->fSpecification = mPreClusterFinder.getDEId(iDE);
    totalBytesUsed += bytesUsed;
  }
}

} // namespace mch
} // namespace o2
