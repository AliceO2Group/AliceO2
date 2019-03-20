// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "DumpDigits.h"

#include <cstdint>
#include <vector>

#include <FairMQLogger.h>

#include "MCHBase/DigitBlock.h"
#include "aliceHLTwrapper/AliHLTDataTypes.h"

namespace o2
{
namespace mch
{

using namespace o2::alice_hlt;

using namespace std;

DumpDigits::DumpDigits() : FairMQDevice()
{
  // register a handler for data arriving on "data-in" channel
  OnData("data-in", &DumpDigits::handleData);
}

bool DumpDigits::handleData(FairMQMessagePtr& msg, int /*index*/)
{
  /// handler is called whenever a message arrives on "data-in",
  /// with a reference to the message and a sub-channel index (here 0)

  // convert the message in ALICE HLT format
  vector<o2::alice_hlt::MessageFormat::BufferDesc_t> blockData;
  blockData.emplace_back(
    o2::alice_hlt::MessageFormat::BufferDesc_t(reinterpret_cast<uint8_t*>(msg->GetData()), msg->GetSize()));
  mFormatHandler.clear();
  mFormatHandler.addMessages(blockData);

  // check the input blocks
  vector<BlockDescriptor>& inputBlocks = mFormatHandler.getBlockDescriptors();
  auto nInputBlocks = inputBlocks.size();
  LOG(INFO) << "\n\n------\nnumber of input blocks = " << nInputBlocks;
  if (blockData.size() > 0 && nInputBlocks == 0 && mFormatHandler.getEvtDataList().size() == 0) {
    LOG(WARN) << "warning: none of " << blockData.size() << " input buffer(s) recognized as valid input" << endl;
  }

  // print the content of the blocks
  for (const auto& block : inputBlocks) {
    LOG(INFO) << "\ndata type/origin = " << block.fDataType.fID << "; specification = " << block.fSpecification;
    auto size(block.fSize);
    LOG(INFO) << "block size = " << size;
    auto digitBlock(reinterpret_cast<const DigitBlock*>(block.fPtr));
    LOG(INFO) << "block: " << *digitBlock;
    auto digit(reinterpret_cast<const DigitStruct*>(reinterpret_cast<const DigitBlock*>(block.fPtr) + 1));
    LOG(INFO) << "data: " << *digit;
  }

  // return true if want to be called again (otherwise go to IDLE state)
  return true;
}

} // namespace mch
} // namespace o2
