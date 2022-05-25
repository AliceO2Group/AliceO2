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
#include <fairmq/FairMQDevice.h>
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

SubTimeFrameFileReader::SubTimeFrameFileReader(const std::string& pFileName, o2::detectors::DetID::mask_t detMask)
  : mFileName(pFileName)
{
  mFileMap.open(mFileName);
  if (!mFileMap.is_open()) {
    LOG(error) << "Failed to open TF file for reading (mmap).";
    return;
  }
  mFileSize = mFileMap.size();
  mFileMapOffset = 0;

  for (DetID::ID id = DetID::First; id <= DetID::Last; id++) {
    mDetOrigMap[DetID::getDataOrigin(id)] = detMask[id];
  }

#if __linux__
  madvise((void*)mFileMap.data(), mFileMap.size(), MADV_HUGEPAGE | MADV_SEQUENTIAL | MADV_DONTDUMP);
#endif
}

SubTimeFrameFileReader::~SubTimeFrameFileReader()
{
  if (!mFileMap.is_open()) {
#if __linux__
    madvise((void*)mFileMap.data(), mFileMap.size(), MADV_DONTNEED);
#endif
    mFileMap.close();
  }
}

std::size_t SubTimeFrameFileReader::getHeaderStackSize() // throws ios_base::failure
{
  // Expect valid Stack in the file.
  // First Header must be DataHeader. The size is unknown since there are multiple versions.
  // Each header in the stack extends BaseHeader

  // Read first the base header then the rest of the extended header. Keep going until the next flag is set.
  // reset the file pointer to the original incoming position, so the complete Stack can be read in

  bool readNextHeader = true;
  std::size_t lStackSize = 0;
  DataHeader lBaseHdr; // Use DataHeader  since the BaseHeader has no default contructor.

  const auto lFilePosStart = position();

  const auto cMaxHeaders = 16; /* make sure we don't loop forever */
  auto lNumHeaders = 0;
  while (readNextHeader && (++lNumHeaders <= cMaxHeaders)) {
    // read BaseHeader only!
    const auto lBaseHdrPos = position();
    if (!read_advance(&lBaseHdr, sizeof(BaseHeader))) {
      return 0;
    }

    // go back, and read the whole O2 header (Base+Derived)
    set_position(lBaseHdrPos);
    if (!ignore_nbytes(lBaseHdr.size())) {
      return 0;
    }

    lStackSize += lBaseHdr.size();
    readNextHeader = (lBaseHdr.next() != nullptr);
  }
  // reset the file pointer
  set_position(lFilePosStart);

  if (lNumHeaders >= cMaxHeaders) {
    LOGP(error, "FileReader: Reached max number of headers allowed: {}.", cMaxHeaders);
    return 0;
  }

  return lStackSize;
}

Stack SubTimeFrameFileReader::getHeaderStack(std::size_t& pOrigsize)
{
  const auto lStackSize = getHeaderStackSize();
  pOrigsize = lStackSize;

  if (lStackSize < sizeof(BaseHeader)) {
    // error in the stream
    pOrigsize = 0;
    return Stack{};
  }

  std::byte* lStackMem = reinterpret_cast<std::byte*>(peek());
  if (!ignore_nbytes(lStackSize)) {
    // error in the stream
    pOrigsize = 0;
    return Stack{};
  }

  // This must handle different versions of DataHeader
  // check if DataHeader needs an upgrade by looking at the version number
  const BaseHeader* lBaseOfDH = BaseHeader::get(lStackMem);
  if (!lBaseOfDH) {
    return Stack{};
  }

  if (lBaseOfDH->headerVersion < DataHeader::sVersion) {
    DataHeader lNewDh;

    // Write over the new DataHeader. We need to update some of the BaseHeader values.
    assert(sizeof(DataHeader) > lBaseOfDH->size()); // current DataHeader must be larger
    std::memcpy(&lNewDh, (void*)lBaseOfDH->data(), lBaseOfDH->size());

    // make sure to bump the version in the BaseHeader.
    // TODO: Is there a better way?
    lNewDh.headerSize = sizeof(DataHeader);
    lNewDh.headerVersion = DataHeader::sVersion;

    if (lBaseOfDH->headerVersion == 1 || lBaseOfDH->headerVersion == 2) {
      /* nothing to do for the upgrade */
    } else {
      LOGP(error, "FileReader: DataHeader v{} read from file is not upgraded to the current version {}",
           lBaseOfDH->headerVersion, DataHeader::sVersion);
      LOGP(error, "Try using a newer version of DataDistribution or file a BUG");
    }

    if (lBaseOfDH->size() == lStackSize) {
      return Stack(lNewDh);
    } else {
      assert(lBaseOfDH->size() < lStackSize);

      return Stack(
        lNewDh,
        Stack(lStackMem + lBaseOfDH->size()));
    }
  }

  return Stack(lStackMem);
}

std::uint64_t SubTimeFrameFileReader::sStfId = 0; // TODO: add id to files metadata
std::uint32_t sRunNumber = 0;                     // TODO: add id to files metadata
std::uint32_t sFirstTForbit = 0;                  // TODO: add id to files metadata
std::uint64_t sCreationTime = 0;
std::mutex stfMtx;

std::unique_ptr<MessagesPerRoute> SubTimeFrameFileReader::read(FairMQDevice* device, const std::vector<o2f::OutputRoute>& outputRoutes,
                                                               const std::string& rawChannel, bool sup0xccdb, int verbosity)
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

  auto addPart = [&msgMap](FairMQMessagePtr hd, FairMQMessagePtr pl, const std::string& fairMQChannel) {
    FairMQParts* parts = nullptr;
    parts = msgMap[fairMQChannel].get(); // FairMQParts*
    if (!parts) {
      msgMap[fairMQChannel] = std::make_unique<FairMQParts>();
      parts = msgMap[fairMQChannel].get();
    }
    parts->AddPart(std::move(hd));
    parts->AddPart(std::move(pl));
  };

  // record current position
  const auto lTfStartPosition = position();

  if (lTfStartPosition == size() || !mFileMap.is_open() || eof()) {
    return nullptr;
  }
  auto tfID = sStfId;
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
    mFileMap.close();
    return nullptr;
  }
  lStfMetaDataHdr = o2::header::DataHeader::Get(lMetaHdrStack.first());
  if (!read_advance(&lStfFileMeta, sizeof(SubTimeFrameFileMeta))) {
    return nullptr;
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
    mFileMap.close();
    return nullptr;
  }

  // prepare to read the TF data
  const auto lStfSizeInFile = lStfFileMeta.mStfSizeInFile;
  if (lStfSizeInFile == (sizeof(DataHeader) + sizeof(SubTimeFrameFileMeta))) {
    LOGP(warning, "Reading an empty TF from file. Only meta information present");
    mFileMap.close();
    return nullptr;
  }

  // check there's enough data in the file
  if ((lTfStartPosition + lStfSizeInFile) > this->size()) {
    LOGP(warning, "Not enough data in file for this TF. Required: {}, available: {}", lStfSizeInFile, (this->size() - lTfStartPosition));
    mFileMap.close();
    return nullptr;
  }

  // Index
  std::size_t lStfIndexHdrStackSize = 0;
  const DataHeader* lStfIndexHdr = nullptr;

  // Read DataHeader + SubTimeFrameFileMeta
  auto lStfIndexHdrStack = getHeaderStack(lStfIndexHdrStackSize);
  if (lStfIndexHdrStackSize == 0) {
    mFileMap.close();
    return nullptr;
  }
  lStfIndexHdr = o2::header::DataHeader::Get(lStfIndexHdrStack.first());
  if (!lStfIndexHdr) {
    LOG(error) << "Failed to read the TF index structure. The file might be corrupted.";
    return nullptr;
  }

  if (!ignore_nbytes(lStfIndexHdr->payloadSize)) {
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
  // read <hdrStack + data> pairs
  while (lLeftToRead > 0) {

    // allocate and read the Headers
    std::size_t lDataHeaderStackSize = 0;
    Stack lDataHeaderStack = getHeaderStack(lDataHeaderStackSize);
    if (lDataHeaderStackSize == 0) {
      mFileMap.close();
      return nullptr;
    }
    const DataHeader* lDataHeader = o2::header::DataHeader::Get(lDataHeaderStack.first());
    if (!lDataHeader) {
      LOG(error) << "Failed to read the TF HBF DataHeader structure. The file might be corrupted.";
      mFileMap.close();
      return nullptr;
    }
    DataHeader locDataHeader(*lDataHeader);
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
    o2::header::Stack headerStack{locDataHeader, o2f::DataProcessingHeader{tfID, 1, lStfFileMeta.mWriteTimeMs}};
    if (stfHeader.runNumber == -1) {
      stfHeader.id = locDataHeader.tfCounter;
      stfHeader.runNumber = locDataHeader.runNumber;
      stfHeader.firstOrbit = locDataHeader.firstTForbit;
      std::lock_guard<std::mutex> lock(stfMtx);
      sRunNumber = stfHeader.runNumber;
      sFirstTForbit = stfHeader.firstOrbit;
    }

    const std::uint64_t lDataSize = locDataHeader.payloadSize;
    // do we accept these data?
    auto detOrigStatus = mDetOrigMap.find(locDataHeader.dataOrigin);
    if (detOrigStatus != mDetOrigMap.end() && !detOrigStatus->second) { // this is a detector data and we don't want to read it
      if (!ignore_nbytes(lDataSize)) {
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
      if (!ignore_nbytes(lDataSize)) {
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

    if (!read_advance(lDataMsg->GetData(), lDataSize)) {
      return nullptr;
    }
    if (verbosity > 0) {
      if (verbosity > 1 || locDataHeader.splitPayloadIndex == 0) {
        printStack(headerStack);
        if (o2::raw::RDHUtils::checkRDH(lDataMsg->GetData()) && verbosity > 2) {
          o2::raw::RDHUtils::printRDH(lDataMsg->GetData());
        }
      }
    }
#ifdef _RUN_TIMING_MEASUREMENT_
    addPartSW.Start(false);
#endif
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
  for (int iss = 0; iss < (sup0xccdb ? 1 : 2); iss++) {
    o2::header::DataHeader stfDistDataHeader(o2::header::gDataDescriptionDISTSTF, o2::header::gDataOriginFLP, stfSS[iss], sizeof(STFHeader), 0, 1);
    stfDistDataHeader.payloadSerializationMethod = o2::header::gSerializationMethodNone;
    stfDistDataHeader.firstTForbit = stfHeader.firstOrbit;
    stfDistDataHeader.runNumber = stfHeader.runNumber;
    stfDistDataHeader.tfCounter = stfHeader.id;
    stfHeader.id = tfID;
    const auto fmqChannel = findOutputChannel(&stfDistDataHeader, tfID);
    if (!fmqChannel.empty()) { // no output channel
      auto fmqFactory = device->GetChannel(fmqChannel, 0).Transport();
      o2::header::Stack headerStackSTF{stfDistDataHeader, o2f::DataProcessingHeader{tfID, 1, lStfFileMeta.mWriteTimeMs}};
      if (verbosity > 0) {
        printStack(headerStackSTF);
      }
      auto hdMessageSTF = fmqFactory->CreateMessage(headerStackSTF.size(), fair::mq::Alignment{64});
      auto plMessageSTF = fmqFactory->CreateMessage(stfDistDataHeader.payloadSize, fair::mq::Alignment{64});
      memcpy(hdMessageSTF->GetData(), headerStackSTF.data(), headerStackSTF.size());
      memcpy(plMessageSTF->GetData(), &stfHeader, sizeof(STFHeader));
#ifdef _RUN_TIMING_MEASUREMENT_
      addPartSW.Start(false);
#endif
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
  ++sStfId;
  return messagesPerRoute;
}

} // namespace rawdd
} // namespace o2
