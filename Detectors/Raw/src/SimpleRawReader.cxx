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

/// \file SimpleRawReader.cxx
/// \brief Simple reader for non-DPL tests
#include "Framework/Logger.h"
#include "DetectorsRaw/SimpleRawReader.h"
#include "DetectorsRaw/SimpleSTF.h"
#include "DetectorsRaw/HBFUtils.h"

using namespace o2::raw;
using namespace o2::framework;

/// due to the dictionary genation for the macros which include this class, the constructors and d-tor are defined here.
///_________________________________________________________________
SimpleRawReader::SimpleRawReader(const std::string& cfg, bool tfPerMessage, int loop)
  : mLoop(loop), mHBFPerMessage(!tfPerMessage), mCFGName(cfg) {}

///_________________________________________________________________
SimpleRawReader::SimpleRawReader() = default;

///_________________________________________________________________
SimpleRawReader::~SimpleRawReader() = default;

///_________________________________________________________________
/// Init the reader from the config string, create buffer
void SimpleRawReader::init()
{
  if (mReader) {
    return;
  }
  if (mCFGName.empty()) {
    throw std::runtime_error("input is not set");
  }
  mReader = std::make_unique<RawFileReader>(mCFGName); // init from configuration file
  uint32_t errCheck = 0xffffffff;
  errCheck ^= 0x1 << RawFileReader::ErrNoSuperPageForTF; // makes no sense for superpages not interleaved by others
  mReader->setCheckErrors(errCheck);
  mReader->init();
}

///_________________________________________________________________
/// read data of all links for next TF, return number of links with data
bool SimpleRawReader::loadNextTF()
{
  mSTF.reset();

  if (!mReader) {
    init();
  }
  int nLinks = mReader->getNLinks(), nread = 0;
  std::vector<InputRoute> inpRoutes;
  SimpleSTF::PartsRef partsRef;
  SimpleSTF::Messages messages;

  auto tfID = mReader->getNextTFToRead();
  if (tfID >= mReader->getNTimeFrames()) {
    if (mReader->getNTimeFrames() && mLoop--) {
      tfID = 0;
      mReader->setNextTFToRead(tfID);
      mLoopsDone++;
      for (int il = 0; il < nLinks; il++) {
        mReader->getLink(il).nextBlock2Read = 0; // think about more elaborate looping scheme, e.g. incrementing the orbits in RDHs
      }
      LOG(INFO) << "Starting new loop " << mLoopsDone << " from the beginning of data";
    } else {
      printStat();
      mDone = true;
      return false;
    }
  }
  auto nHB = HBFUtils::Instance().getNOrbitsPerTF();
  partsRef.reserve(nLinks);
  inpRoutes.reserve(nLinks);
  messages.reserve(2 * (mHBFPerMessage ? nHB * nLinks : nLinks));
  size_t totSize = 0;
  for (int il = 0; il < nLinks; il++) {
    auto& link = mReader->getLink(il);
    auto tfsz = link.getNextTFSize();
    if (!tfsz) {
      continue;
    }
    InputSpec inps(std::string("inpSpec") + std::to_string(il), link.origin, link.description, link.subspec, Lifetime::Timeframe);
    inpRoutes.emplace_back(InputRoute{inps, size_t(il), std::string("src") + std::to_string(il)});

    o2::header::DataHeader hdrTmpl(link.description, link.origin, link.subspec); // template with 0 size
    hdrTmpl.payloadSerializationMethod = o2::header::gSerializationMethodNone;
    hdrTmpl.splitPayloadParts = mHBFPerMessage ? nHB : 1;
    partsRef.emplace_back(messages.size(), hdrTmpl.splitPayloadParts); // entry and nparts the multipart in the messages
    while (hdrTmpl.splitPayloadIndex < hdrTmpl.splitPayloadParts) {

      hdrTmpl.payloadSize = mHBFPerMessage ? link.getNextHBFSize() : tfsz;
      o2::header::Stack headerStack{hdrTmpl, o2f::DataProcessingHeader{mTFIDaccum}};
      auto* hdMessage = messages.emplace_back(std::make_unique<std::vector<char>>(headerStack.size()))->data(); // header message
      memcpy(hdMessage, headerStack.data(), headerStack.size());

      auto* plMessage = messages.emplace_back(std::make_unique<std::vector<char>>(hdrTmpl.payloadSize))->data(); // payload message
      auto bread = mHBFPerMessage ? link.readNextHBF(plMessage) : link.readNextTF(plMessage);
      if (bread != hdrTmpl.payloadSize) {
        LOG(ERROR) << "Link " << il << " read " << bread << " bytes instead of " << hdrTmpl.payloadSize
                   << " expected in TF=" << mTFIDaccum << " part=" << hdrTmpl.splitPayloadIndex;
        throw std::runtime_error("error in link data reading");
      }
      hdrTmpl.splitPayloadIndex++; // prepare for next
      LOG(DEBUG) << "Loaded " << tfsz << " bytes for " << link.describe();
    }
    totSize += tfsz;
    nread++;
  }
  LOG(INFO) << "Loaded " << totSize << " bytes for " << nread << " non-empty links out of " << nLinks;
  mSTF = std::make_unique<SimpleSTF>(std::move(inpRoutes), std::move(partsRef), std::move(messages));

  return nread > 0;
}

///_________________________________________________________________
/// get current SimpleSTD pointer
SimpleSTF* SimpleRawReader::getSimpleSTF()
{
  return mSTF.get();
}

///_________________________________________________________________
/// get current SimpleSTD InputRecord pointer
o2::framework::InputRecord* SimpleRawReader::getInputRecord()
{
  return mSTF ? &mSTF->record : nullptr;
}

///_________________________________________________________________
/// print current statistics
void SimpleRawReader::printStat() const
{
  LOGF(INFO, "Sent payload of %zu bytes in %zu messages sent for %d TFs in %d links", mSentSize, mSentMessages, mTFIDaccum, getNLinks());
}
