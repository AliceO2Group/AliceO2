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

// Adapthed with minimal changes from Gvozden Nescovic code to read sTFs files created by DataDistribution

#include "TFReaderDD/SubTimeFrameFileReader.h"
#include "DetectorsRaw/RDHUtils.h"
#include "Framework/Logger.h"
#include "Framework/OutputRoute.h"
#include "Framework/DataSpecUtils.h"
#include "Framework/DataProcessingHeader.h"
#include <fairmq/Device.h>
#include <fairmq/Message.h>
#include <fairmq/Parts.h>
#include <mutex>

#if __linux__
#include <sys/mman.h>
#endif

// uncomment this to check breakdown of TF building timing
//#define  _RUN_TIMING_MEASUREMENT_

#ifdef _RUN_TIMING_MEASUREMENT_
#include "TStopwatch.h"
#endif

namespace o2
{
namespace rawdd
{
using DetID = o2::detectors::DetID;
using namespace o2::header;
namespace o2f = o2::framework;

////////////////////////////////////////////////////////////////////////////////
/// SubTimeFrameFileReader
////////////////////////////////////////////////////////////////////////////////

SubTimeFrameFileReader::SubTimeFrameFileReader(const std::string& pFileName, o2::detectors::DetID::mask_t detMask, int verb, bool sup0xccdb, bool repaireHeaders, bool rejectDistSTF)
  : mVerbosity(verb), mSup0xccdb(sup0xccdb), mRepaireHeaders(repaireHeaders), mRejectDistSTF(rejectDistSTF)
{
  mFile.reset(BinFileOp::open(pFileName));
  if (!mFile || !mFile->isGood()) {
    return;
  }

  for (DetID::ID id = DetID::First; id <= DetID::Last; id++) {
    mDetOrigMap[DetID::getDataOrigin(id)] = detMask[id];
  }
}

Stack SubTimeFrameFileReader::getHeaderStack(std::size_t& pOrigsize)
{
  // Expect valid Stack in the file.
  // First Header must be DataHeader. The size is unknown since there are multiple versions.
  // Each header in the stack extends BaseHeader

  // Read first the base header then the rest of the extended header. Keep going until the next flag is set.
  // reset the file pointer to the original incoming position, so the complete Stack can be read in
  bool readNextHeader = true;
  std::size_t bufsz = 0, lStackSize = 0;
  std::byte* lStackMem = nullptr;
  pOrigsize = 0;
  const auto lFilePosStart = mFile->position();
  const int cMaxHeaders = 16; // make sure we don't loop forever
  int lNumHeaders = 0;
  while (readNextHeader && (++lNumHeaders <= cMaxHeaders)) {
    if ((lStackSize + sizeof(BaseHeader)) >= bufsz && !(lStackMem = reinterpret_cast<std::byte*>(mFile->bufferize((bufsz += BinFileOp::KBYTE))))) {
      LOGP(error, "Could not bufferize {} bytes to read the headers stack", bufsz);
      return Stack{};
    }
    const auto& lBaseHdr = *reinterpret_cast<BaseHeader*>(lStackMem + lStackSize);
    lStackSize += lBaseHdr.size();
    readNextHeader = (lBaseHdr.next() != nullptr);
  }
  if (lNumHeaders >= cMaxHeaders) {
    LOGP(error, "Reached max number of headers allowed: {}.", cMaxHeaders);
    return Stack{};
  }
  if (lStackSize < sizeof(BaseHeader)) {
    LOGP(error, "Stack size {} is smaller than BaseHeader size {}", lStackSize, sizeof(BaseHeader));
    return Stack{};
  }

  // This must handle different versions of DataHeader, check if DataHeader needs an upgrade by looking at the version number
  const BaseHeader* lBaseOfDH = BaseHeader::get(lStackMem);
  if (!lBaseOfDH) {
    LOGP(error, "Failed to extract the DataHeader from the buffer, position in file {}", mFile->position());
    return Stack{};
  }

  pOrigsize = lStackSize;
  mFile->set_position(lFilePosStart + lStackSize);
  if (lBaseOfDH->headerVersion < DataHeader::sVersion) {
    DataHeader lNewDh;

    // Write over the new DataHeader. We need to update some of the BaseHeader values.
    assert(sizeof(DataHeader) > lBaseOfDH->size()); // current DataHeader must be larger
    std::memcpy(&lNewDh, (void*)lBaseOfDH->data(), lBaseOfDH->size());

    // make sure to bump the version in the BaseHeader. TODO: Is there a better way?
    lNewDh.headerSize = sizeof(DataHeader);
    lNewDh.headerVersion = DataHeader::sVersion;

    if (lBaseOfDH->headerVersion == 1 || lBaseOfDH->headerVersion == 2) {
      // nothing to do for the upgrade
    } else {
      LOGP(error, "DataHeader v{} read from file is not upgraded to the current version {}",
           lBaseOfDH->headerVersion, DataHeader::sVersion);
      LOGP(error, "Try using a newer version of DataDistribution or file a BUG");
    }

    if (lBaseOfDH->size() == lStackSize) {
      return Stack(lNewDh);
    } else {
      assert(lBaseOfDH->size() < lStackSize);
      return Stack(lNewDh, Stack(lStackMem + lBaseOfDH->size()));
    }
  }

  return Stack(lStackMem);
}

const std::string SubTimeFrameFileReader::describeHeader(const o2::header::DataHeader& hd, bool full) const
{
  std::string res = fmt::format("{}", o2f::DataSpecUtils::describe(o2::framework::OutputSpec{hd.dataOrigin, hd.dataDescription, hd.subSpecification}));
  if (full) {
    res += fmt::format(" part:{}/{} sz:{} TF:{} Orb:{} Run:{}", hd.splitPayloadIndex, hd.splitPayloadParts, hd.payloadSize, hd.tfCounter, hd.firstTForbit, hd.runNumber);
  }
  return res;
}

std::uint32_t sRunNumber = 0;                     // TODO: add id to files metadata
std::uint32_t sFirstTForbit = 0;                  // TODO: add id to files metadata
std::uint64_t sCreationTime = 0;
std::mutex stfMtx;

std::unique_ptr<MessagesPerRoute> SubTimeFrameFileReader::read(fair::mq::Device* device, const std::vector<o2f::OutputRoute>& outputRoutes, const std::string& rawChannel, size_t slice)
{
  std::unique_ptr<MessagesPerRoute> messagesPerRoute = std::make_unique<MessagesPerRoute>();
  auto& msgMap = *messagesPerRoute.get();
  assert(device);
  std::unordered_map<o2::header::DataHeader, std::pair<std::string, bool>> channelsMap;
  auto findOutputChannel = [&outputRoutes, &rawChannel, &channelsMap](const o2::header::DataHeader* h, size_t tslice) -> const std::string& {
    if (!rawChannel.empty()) {
      return rawChannel;
    }
    auto& chFromMap = channelsMap[*h];
    if (chFromMap.first.empty() && !chFromMap.second) { // search for channel which is enountered for the 1st time
      chFromMap.second = true;                          // flag that it was already checked
      for (auto& oroute : outputRoutes) {
        LOG(debug) << "comparing with matcher to route " << oroute.matcher << " TSlice:" << oroute.timeslice;
        if (o2f::DataSpecUtils::match(oroute.matcher, h->dataOrigin, h->dataDescription, h->subSpecification) && ((tslice % oroute.maxTimeslices) == oroute.timeslice)) {
          LOG(debug) << "picking the route:" << o2f::DataSpecUtils::describe(oroute.matcher) << " channel " << oroute.channel;
          chFromMap.first = oroute.channel;
          break;
        }
      }
    }
    return chFromMap.first;
  };

  auto addPart = [&msgMap](fair::mq::MessagePtr hd, fair::mq::MessagePtr pl, const std::string& fairMQChannel) {
    fair::mq::Parts* parts = nullptr;
    parts = msgMap[fairMQChannel].get(); // fair::mq::Parts*
    if (!parts) {
      msgMap[fairMQChannel] = std::make_unique<fair::mq::Parts>();
      parts = msgMap[fairMQChannel].get();
    }
    parts->AddPart(std::move(hd));
    parts->AddPart(std::move(pl));
  };

  // record current position
  const auto lTfStartPosition = mFile->position();

  if (lTfStartPosition == mFile->size() || !mFile || !mFile->isGood() || mFile->eof()) {
    return nullptr;
  }
  auto tfID = slice;
  uint32_t runNumberFallBack = sRunNumber;
  uint32_t firstTForbitFallBack = sFirstTForbit;
  uint64_t creationFallBack = sCreationTime;
  bool negativeOrbitNotified = false, noRunNumberNotified = false, creation0Notified = false;
  std::size_t lMetaHdrStackSize = 0;
  const DataHeader* lStfMetaDataHdr = nullptr;
  SubTimeFrameFileMeta lStfFileMeta;

  auto printStack = [tfID](const o2::header::Stack& st) {
    auto dph = o2::header::get<o2f::DataProcessingHeader*>(st.data());
    auto dh = o2::header::get<o2::header::DataHeader*>(st.data());
    LOGP(info, "TF#{} Header for {}/{}/{} @ tfCounter {} run {} | {} of {} size {}, TForbit {} | DPH: {}/{}/{}", tfID,
         dh->dataOrigin.str, dh->dataDescription.str, dh->subSpecification, dh->tfCounter, dh->runNumber,
         dh->splitPayloadIndex, dh->splitPayloadParts, dh->payloadSize, dh->firstTForbit,
         dph ? dph->startTime : 0, dph ? dph->duration : 0, dph ? dph->creation : 0);
  };

  // Read DataHeader + SubTimeFrameFileMeta
  auto lMetaHdrStack = getHeaderStack(lMetaHdrStackSize);
  if (lMetaHdrStackSize == 0) {
    LOG(error) << "Failed to read the TF file header. The file might be corrupted.";
    mFile.reset(nullptr);
    return nullptr;
  }
  lStfMetaDataHdr = o2::header::DataHeader::Get(lMetaHdrStack.first());
  if (mVerbosity > 0) {
    LOGP(info, "read filemeta, pos = {}, size = {}", mFile->position(), sizeof(SubTimeFrameFileMeta));
  }
  if (!mFile->read_advance(&lStfFileMeta, sizeof(SubTimeFrameFileMeta))) {
    return nullptr;
  }
  if (mVerbosity > 0) {
    LOGP(info, "TFMeta : {}", lStfFileMeta.info());
  }
  if (lStfFileMeta.mWriteTimeMs == 0 && creationFallBack != 0) {
    if (!creation0Notified) {
      creation0Notified = true;
      LOGP(warn, "Creation time 0 for timeSlice:{}, redefine to {}", tfID, creationFallBack);
    }
    lStfFileMeta.mWriteTimeMs = creationFallBack;
  } else {
    sCreationTime = lStfFileMeta.mWriteTimeMs;
  }

  // verify we're actually reading the correct data in
  if (!(SubTimeFrameFileMeta::getDataHeader().dataDescription == lStfMetaDataHdr->dataDescription)) {
    LOGP(warning, "Reading bad data: SubTimeFrame META header");
    mFile.reset(nullptr);
    return nullptr;
  }

  // prepare to read the TF data
  const auto lStfSizeInFile = lStfFileMeta.mStfSizeInFile;
  if (lStfSizeInFile == (sizeof(DataHeader) + sizeof(SubTimeFrameFileMeta))) {
    LOGP(warning, "Reading an empty TF from file. Only meta information present");
    mFile.reset(nullptr);
    return nullptr;
  }

  // check there's enough data in the file
  if ((lTfStartPosition + lStfSizeInFile) > mFile->size()) {
    LOGP(warning, "Not enough data in file for this TF. Required: {}, available: {}", lStfSizeInFile, (mFile->size() - lTfStartPosition));
    mFile.reset(nullptr);
    return nullptr;
  }

  // Index
  std::size_t lStfIndexHdrStackSize = 0;
  const DataHeader* lStfIndexHdr = nullptr;

  // Read DataHeader + SubTimeFrameFileMeta
  auto lStfIndexHdrStack = getHeaderStack(lStfIndexHdrStackSize);
  if (lStfIndexHdrStackSize == 0) {
    mFile.reset(nullptr);
    return nullptr;
  }
  lStfIndexHdr = o2::header::DataHeader::Get(lStfIndexHdrStack.first());
  if (!lStfIndexHdr) {
    LOG(error) << "Failed to read the TF index structure. The file might be corrupted.";
    return nullptr;
  }

  if (!mFile->ignore_nbytes(lStfIndexHdr->payloadSize)) {
    return nullptr;
  }
#ifdef _RUN_TIMING_MEASUREMENT_
  TStopwatch readSW, findChanSW, msgSW, addPartSW;
  findChanSW.Stop();
  msgSW.Stop();
  addPartSW.Stop();
#endif
  // Remaining data size of the TF:
  // total size in file - meta (hdr+struct) - index (hdr + payload)
  const auto lStfDataSize = lStfSizeInFile - (lMetaHdrStackSize + sizeof(SubTimeFrameFileMeta)) - (lStfIndexHdrStackSize + lStfIndexHdr->payloadSize);

  std::int64_t lLeftToRead = lStfDataSize;
  STFHeader stfHeader{tfID, -1u, -1u};
  DataHeader prevHeader;
  // read <hdrStack + data> pairs
  while (lLeftToRead > 0) {
    // allocate and read the Headers
    std::size_t lDataHeaderStackSize = 0;
    Stack lDataHeaderStack = getHeaderStack(lDataHeaderStackSize);
    if (lDataHeaderStackSize == 0) {
      mFile.reset(nullptr);
      return nullptr;
    }
    const DataHeader* lDataHeader = o2::header::DataHeader::Get(lDataHeaderStack.first());
    if (!lDataHeader) {
      LOG(error) << "Failed to read the TF HBF DataHeader structure. The file might be corrupted.";
      mFile.reset(nullptr);
      return nullptr;
    }
    DataHeader locDataHeader(*lDataHeader);

    if (mRepaireHeaders) {
      if (locDataHeader == prevHeader) {
        if (prevHeader.tfCounter == locDataHeader.tfCounter && (prevHeader.splitPayloadIndex + 1) != locDataHeader.splitPayloadIndex) {
          if (mVerbosity > 3) {
            LOGP(warn, "Repairing wrong part index for {} to {}", describeHeader(locDataHeader, true), (prevHeader.splitPayloadIndex + 1) % prevHeader.splitPayloadParts);
          }
          locDataHeader.splitPayloadIndex = (++prevHeader.splitPayloadIndex) % prevHeader.splitPayloadParts;
        }
      } else { // new header
        if (locDataHeader.splitPayloadIndex != 0) {
          if (mVerbosity > 2) {
            LOGP(warn, "Repairing wrong part index for new {} to {}", describeHeader(locDataHeader, true), (prevHeader.splitPayloadIndex + 1) % prevHeader.splitPayloadParts);
          }
          locDataHeader.splitPayloadIndex = 0;
        }
      }
      prevHeader = locDataHeader;
    }
    // sanity check
    if (int(locDataHeader.firstTForbit) == -1) {
      if (!negativeOrbitNotified) {
        LOGP(warn, "Negative orbit for timeSlice:{} tfCounter:{} runNumber:{}, redefine to {}", tfID, locDataHeader.tfCounter, locDataHeader.runNumber, firstTForbitFallBack);
        negativeOrbitNotified = true;
      }
      locDataHeader.firstTForbit = firstTForbitFallBack;
    }
    if (locDataHeader.runNumber == 0) {
      if (!noRunNumberNotified) {
        LOGP(warn, "runNumber is 0 for timeSlice:{} tfCounter:{}, redefine to {}", tfID, locDataHeader.tfCounter, runNumberFallBack);
        noRunNumberNotified = true;
      }
      locDataHeader.runNumber = runNumberFallBack;
    }
    const std::uint64_t lDataSize = locDataHeader.payloadSize;

    if (locDataHeader.dataOrigin == o2::header::gDataOriginFLP && locDataHeader.dataDescription == o2::header::gDataDescriptionDISTSTF && mRejectDistSTF) {
      if (mVerbosity > 0) {
        LOGP(warn, "Ignoring stored {}", describeHeader(locDataHeader));
      }
      if (!mFile->ignore_nbytes(lDataSize)) {
        return nullptr;
      }
      lLeftToRead -= (lDataHeaderStackSize + lDataSize); // update the counter
      continue;
    }
    o2::header::Stack headerStack{locDataHeader, o2f::DataProcessingHeader{tfID, 1, lStfFileMeta.mWriteTimeMs}};
    if (stfHeader.runNumber == -1) {
      stfHeader.id = locDataHeader.tfCounter;
      stfHeader.runNumber = locDataHeader.runNumber;
      stfHeader.firstOrbit = locDataHeader.firstTForbit;
      std::lock_guard<std::mutex> lock(stfMtx);
      sRunNumber = stfHeader.runNumber;
      sFirstTForbit = stfHeader.firstOrbit;
    }
    // do we accept these data?
    auto detOrigStatus = mDetOrigMap.find(locDataHeader.dataOrigin);
    if (detOrigStatus != mDetOrigMap.end() && !detOrigStatus->second) { // this is a detector data and we don't want to read it
      if (!mFile->ignore_nbytes(lDataSize)) {
        return nullptr;
      }
      lLeftToRead -= (lDataHeaderStackSize + lDataSize); // update the counter
      continue;
    }
#ifdef _RUN_TIMING_MEASUREMENT_
    findChanSW.Start(false);
#endif
    const auto& fmqChannel = findOutputChannel(&locDataHeader, tfID);
#ifdef _RUN_TIMING_MEASUREMENT_
    findChanSW.Stop();
#endif
    if (fmqChannel.empty()) { // no output channel
      if (!mFile->ignore_nbytes(lDataSize)) {
        return nullptr;
      }
      lLeftToRead -= (lDataHeaderStackSize + lDataSize); // update the counter
      continue;
      //mFileMap.close();
      //return nullptr;
    }
    // read the data

    auto fmqFactory = device->GetChannel(fmqChannel, 0).Transport();
#ifdef _RUN_TIMING_MEASUREMENT_
    msgSW.Start(false);
#endif
    auto lHdrStackMsg = fmqFactory->CreateMessage(headerStack.size(), fair::mq::Alignment{64});
    auto lDataMsg = fmqFactory->CreateMessage(lDataSize, fair::mq::Alignment{64});
#ifdef _RUN_TIMING_MEASUREMENT_
    msgSW.Stop();
#endif
    memcpy(lHdrStackMsg->GetData(), headerStack.data(), headerStack.size());
    LOGP(debug, "read data, pos = {}, size = {} leftToRead {}", mFile->position(), lDataSize, lLeftToRead);

    if (!read_advance(lDataMsg->GetData(), lDataSize)) {
      return nullptr;
    }
    if (mVerbosity > 0) {
      if (mVerbosity > 1 || locDataHeader.splitPayloadIndex == 0) {
        printStack(headerStack);
        if (o2::raw::RDHUtils::checkRDH(lDataMsg->GetData()) && mVerbosity > 2) {
          o2::raw::RDHUtils::printRDH(lDataMsg->GetData());
        }
      }
    }
#ifdef _RUN_TIMING_MEASUREMENT_
    addPartSW.Start(false);
#endif
    if (mVerbosity > 2) {
      LOGP(info, "addPart {} to {} | HdrSize:{} DataSize:{}", describeHeader(locDataHeader, true), fmqChannel, lHdrStackMsg->GetSize(), lDataMsg->GetSize());
    }
    addPart(std::move(lHdrStackMsg), std::move(lDataMsg), fmqChannel);
#ifdef _RUN_TIMING_MEASUREMENT_
    addPartSW.Stop();
#endif
    // update the counter
    lLeftToRead -= (lDataHeaderStackSize + lDataSize);
  }

  if (lLeftToRead < 0) {
    LOG(error) << "FileRead: Read more data than it is indicated in the META header!";
    return nullptr;
  }
  // add TF acknowledge part
  // in case of empty TF fall-back to previous runNumber and fistTForbit
  if (stfHeader.runNumber == -1u) {
    stfHeader.runNumber = runNumberFallBack;
    stfHeader.firstOrbit = firstTForbitFallBack;
    LOGP(info, "Empty TF#{}, fallback to previous runNumber:{} firstTForbit:{}", tfID, stfHeader.runNumber, stfHeader.firstOrbit);
  }

  unsigned stfSS[2] = {0, 0xccdb};
  for (int iss = 0; iss < (mSup0xccdb ? 1 : 2); iss++) {
    o2::header::DataHeader stfDistDataHeader(o2::header::gDataDescriptionDISTSTF, o2::header::gDataOriginFLP, stfSS[iss], sizeof(STFHeader), 0, 1);
    stfDistDataHeader.payloadSerializationMethod = o2::header::gSerializationMethodNone;
    stfDistDataHeader.firstTForbit = stfHeader.firstOrbit;
    stfDistDataHeader.runNumber = stfHeader.runNumber;
    stfDistDataHeader.tfCounter = stfHeader.id;
    const auto fmqChannel = findOutputChannel(&stfDistDataHeader, tfID);
    if (!fmqChannel.empty()) { // no output channel
      auto fmqFactory = device->GetChannel(fmqChannel, 0).Transport();
      o2::header::Stack headerStackSTF{stfDistDataHeader, o2f::DataProcessingHeader{tfID, 1, lStfFileMeta.mWriteTimeMs}};
      if (mVerbosity > 0) {
        printStack(headerStackSTF);
      }
      auto hdMessageSTF = fmqFactory->CreateMessage(headerStackSTF.size(), fair::mq::Alignment{64});
      auto plMessageSTF = fmqFactory->CreateMessage(stfDistDataHeader.payloadSize, fair::mq::Alignment{64});
      memcpy(hdMessageSTF->GetData(), headerStackSTF.data(), headerStackSTF.size());
      memcpy(plMessageSTF->GetData(), &stfHeader, sizeof(STFHeader));
#ifdef _RUN_TIMING_MEASUREMENT_
      addPartSW.Start(false);
#endif
      if (mVerbosity > 2) {
        LOGP(info, "addPart forced {} to {} | HdrSize:{} DataSize:{}", describeHeader(stfDistDataHeader, true), fmqChannel, hdMessageSTF->GetSize(), plMessageSTF->GetSize());
      }
      addPart(std::move(hdMessageSTF), std::move(plMessageSTF), fmqChannel);
#ifdef _RUN_TIMING_MEASUREMENT_
      addPartSW.Stop();
#endif
    }
  }

#ifdef _RUN_TIMING_MEASUREMENT_
  readSW.Stop();
  LOG(info) << "TF creation time: CPU: " << readSW.CpuTime() << " Wall: " << readSW.RealTime() << " s";
  LOG(info) << "AddPart Timer CPU: " << addPartSW.CpuTime() << " Wall: " << addPartSW.RealTime() << " s";
  LOG(info) << "CreMsg  Timer CPU: " << msgSW.CpuTime() << " Wall: " << msgSW.RealTime() << " s";
  LOG(info) << "FndChan Timer CPU: " << findChanSW.CpuTime() << " Wall: " << findChanSW.RealTime() << " s";
#endif
  return messagesPerRoute;
}

} // namespace rawdd
} // namespace o2
