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

/// \file RawReaderCRU.cxx
/// \author Jens Wiechula (Jens.Wiechula@ikf.uni-frankfurt.de)
/// \author Torsten Alt (Torsten.Alt@cern.ch)

#include <fmt/format.h>
#include <filesystem>
#include "TSystem.h"
#include "TObjArray.h"

#include "Headers/DataHeader.h"
#include "TPCReconstruction/RawReaderCRU.h"
#include "TPCBase/Mapper.h"
#include "Framework/Logger.h"
#include "DetectorsRaw/RDHUtils.h"
#include "CommonUtils/VerbosityConfig.h"

#define CHECK_BIT(var, pos) ((var) & (1 << (pos)))

using namespace o2::tpc::rawreader;

std::ostream& operator<<(std::ostream& output, const RDH& rdh);

template <typename DataType>
std::istream& operator>>(std::istream& input, DataType& data);

void printHeader();
void printHorizontal(const RDH& rdh);

/*
// putting this here instead of inside the header trigger unreasonably large compliation times for some reason
RawReaderCRUEventSync::LinkInfo& RawReaderCRUEventSync::getLinkInfo(uint32_t heartbeatOrbit, int cru, uint8_t globalLinkID)
{
  // check if event is already registered. If not create a new one.
  auto& event = createEvent(heartbeatOrbit);
  return event.CRUInfoArray[cru].LinkInformation[globalLinkID];
}
*/

RawReaderCRUEventSync::EventInfo& RawReaderCRUEventSync::createEvent(const uint32_t heartbeatOrbit, DataType dataType)
{
  // TODO: might be that reversing the loop below has the same effect as using mLastEvent
  if (mLastEvent && mLastEvent->hasHearbeatOrbit(heartbeatOrbit)) {
    return *mLastEvent;
  }

  for (auto& ev : mEventInformation) {
    const auto hbMatch = ev.hasHearbeatOrbit(heartbeatOrbit);
    const long hbDiff = long(heartbeatOrbit) - long(ev.HeartbeatOrbits.front());
    if (hbMatch) {
      mLastEvent = &ev;
      return ev;
    } else if ((hbDiff >= 0) && (hbDiff < 128)) {
      ev.HeartbeatOrbits.emplace_back(heartbeatOrbit);
      std::sort(ev.HeartbeatOrbits.begin(), ev.HeartbeatOrbits.end());
      mLastEvent = &ev;
      return ev;
    }
  }
  auto& ev = mEventInformation.emplace_back(heartbeatOrbit);
  mLastEvent = &ev;
  return ev;
}

void RawReaderCRUEventSync::analyse(RAWDataType rawDataType)
{
  // expected number of packets in one HBorbit
  const size_t numberOfPackets = ExpectedNumberOfPacketsPerHBFrame;

  for (int iEvent = mEventInformation.size() - 1; iEvent >= 0; --iEvent) {
    auto& event = mEventInformation[iEvent];
    event.IsComplete = true;
    size_t totalPayloadSize = 0;
    for (size_t iCRU = 0; iCRU < event.CRUInfoArray.size(); ++iCRU) {
      const auto& cruInfo = event.CRUInfoArray[iCRU];
      if (!cruInfo.isPresent()) {
        if (mCRUSeen[iCRU] >= 0) {
          LOGP(info, "CRU {} missing in event {}", iCRU, iEvent);
          event.IsComplete = false;
          break;
        }

        continue;
      } else {
        totalPayloadSize += cruInfo.totalPayloadSize();
      }

      if (!cruInfo.isComplete(rawDataType)) {
        LOGP(info, "CRU info is incomplete");
        event.IsComplete = false;
        break;
      }
    }

    // remove empty events
    // can be problems in the filtering in readout
    // typically these are empty HB frame with HB start and HB stop packets only
    if (totalPayloadSize == 0) {
      O2INFO("Removing empty event with HB Orbit %u", event.HeartbeatOrbits[0]);
      mEventInformation.erase(mEventInformation.begin() + iEvent);
    }
  }
}

void RawReaderCRUEventSync::setLinksSeen(const CRU cru, const std::bitset<MaxNumberOfLinks>& links)
{
  for (auto& ev : mEventInformation) {
    auto& cruInfo = ev.CRUInfoArray[cru];
    for (int ilink = 0; ilink < cruInfo.LinkInformation.size(); ++ilink) {
      auto& linkInfo = cruInfo.LinkInformation[ilink];
      linkInfo.WasSeen = links[ilink];
    }
  }
}

void RawReaderCRUEventSync::streamTo(std::ostream& output) const
{
  const std::string redBG("\033[41m");
  const std::string red("\033[31m");
  const std::string green("\033[32m");
  const std::string bold("\033[1m");
  const std::string clear("\033[0m");

  std::cout << "CRU information";
  for (size_t iCRU = 0; iCRU < mCRUSeen.size(); ++iCRU) {
    const auto readerNumber = mCRUSeen[iCRU];
    if (readerNumber >= 0) {
      std::cout << fmt::format("CRU {:2} found in reader {}\n", iCRU, readerNumber);
    }
  }

  std::cout << "Detailed event information\n";
  // event loop
  for (int i = 0; i < mEventInformation.size(); ++i) {
    const auto& event = mEventInformation[i];
    const bool isComplete = event.IsComplete;
    if (!isComplete) {
      std::cout << redBG;
    } else {
      std::cout << green;
    }
    std::cout << "Event " << i << "                                \n"
              << clear << "    heartbeatOrbits: ";
    for (const auto& orbit : event.HeartbeatOrbits) {
      std::cout << orbit << " ";
    }
    std::cout << "\n"
              << "    firstOrbit: " << event.getFirstOrbit() << "\n"
              << "    Is complete: " << isComplete << "\n";

    // cru loop
    for (size_t iCRU = 0; iCRU < event.CRUInfoArray.size(); ++iCRU) {
      const auto& cruInfo = event.CRUInfoArray[iCRU];
      if (!cruInfo.isPresent()) {
        continue;
      }
      std::cout << "        ";
      if (!cruInfo.isComplete()) {
        std::cout << bold + red;
      }
      std::cout << "CRU " << iCRU << clear << "\n";
      const auto& cruLinks = cruInfo.LinkInformation;

      // link loop
      for (size_t iLink = 0; iLink < cruLinks.size(); ++iLink) {
        const auto& linkInfo = event.CRUInfoArray[iCRU].LinkInformation[iLink];
        if (!linkInfo.IsPresent) {
          continue;
        }
        std::cout << "        ";
        if (!linkInfo.isComplete()) {
          std::cout << red;
        }
        std::cout << "Link " << iLink << clear << "\n";
        if (!linkInfo.HBEndSeen) {
          std::cout << red;
        }
        std::cout << "            HBEndSeen: " << linkInfo.HBEndSeen << clear << "\n";
        if (linkInfo.PacketPositions.size() != ExpectedNumberOfPacketsPerHBFrame) {
          std::cout << red;
        }
        std::cout << "            Number of Packets: " << linkInfo.PacketPositions.size() << " (" << ExpectedNumberOfPacketsPerHBFrame << ")" << clear << "\n";
        std::cout << "            Payload size : " << linkInfo.PayloadSize << " (" << linkInfo.PayloadSize / 16 << " GBT frames)"
                  << "\n";
        std::cout << "            Packets: ";
        for (const auto& packet : linkInfo.PacketPositions) {
          std::cout << packet << " ";
        }
        std::cout << "\n";
      }
      std::cout << "\n\n";
    }
  }
}

//==============================================================================
int RawReaderCRU::scanFile()
{
  if (mFileIsScanned) {
    return 0;
  }

  // std::vector<PacketDescriptor> mPacketDescriptorMap;
  // const uint64_t RDH_HEADERWORD0 = 0x1ea04003;
  // const uint64_t RDH_HEADERWORD0 = 0x00004003;
  const uint64_t RDH_HEADERWORD0 = 0x00004000; // + RDHUtils::getVersion<o2::header::RAWDataHeader>();

  auto& file = getFileHandle();

  LOGP(info, "scanning file {}", mInputFileName);
  // get length of file in bytes
  file.seekg(0, file.end);
  mFileSize = file.tellg();
  file.seekg(0, file.beg);

  const bool isTFfile = (mInputFileName.rfind(".tf") == mInputFileName.size() - 3);

  // the file is supposed to contain N x 8kB packets. So the number of packets
  // can be determined by the file-size. Ideally, this is not required but the
  // information is derived directly from the header size and payload size.
  // *** to be adapted to header info ***
  const uint32_t numPackets = mFileSize / (8 * 1024);

  // read in the RDH, then jump to the next RDH position
  RDH rdh;
  o2::header::DataHeader dh;
  uint32_t currentPacket = 0;
  uint32_t lastHeartbeatOrbit = 0;

  if (isTFfile) {
    // skip the StfBuilder meta data information
    file >> dh;
    file.seekg(dh.payloadSize, std::ios::cur);
    file >> dh;
    file.seekg(dh.payloadSize, std::ios::cur);
  }

  size_t currentPos = file.tellg();
  size_t dhPayloadSize{};
  size_t dhPayloadSizeSeen{};

  while ((currentPos < mFileSize) && !file.eof()) {
    // ===| in case of TF data file read data header |===
    if (isTFfile && (!dhPayloadSize || (dhPayloadSizeSeen == dhPayloadSize))) {
      file >> dh;
      dhPayloadSize = dh.payloadSize;
      if (dh.dataOrigin != o2::header::gDataOriginTPC) {
        file.seekg(dhPayloadSize, file.cur);
        currentPos = file.tellg();
        dhPayloadSize = 0;
        continue;
      }
      dhPayloadSizeSeen = 0;
      currentPos = file.tellg();
    }

    // ===| read in the RawDataHeader at the current position |=================
    file >> rdh;

    const size_t packetSize = RDHUtils::getOffsetToNext(rdh);
    const size_t offset = packetSize - RDHUtils::getHeaderSize(rdh);
    const auto memorySize = RDHUtils::getMemorySize(rdh);
    const auto payloadSize = memorySize - RDHUtils::getHeaderSize(rdh);
    dhPayloadSizeSeen += packetSize;

    // ===| check for truncated file |==========================================
    const size_t curPos = file.tellg();
    if ((curPos + offset) > mFileSize) {
      LOGP(error, "File truncated at {}, offset {} would exceed file size of {}", curPos, offset, mFileSize);
      break;
    }

    // ===| skip IDC data |=====================================================
    const auto detField = o2::raw::RDHUtils::getDetectorField(rdh);
    if (((detField != 0xdeadbeef) && (detField > 2)) || (payloadSize == 0)) {
      file.seekg(offset, file.cur);
      ++currentPacket;
      currentPos = file.tellg();
      continue;
    }

    // ===| try to detect data type if not already set |========================
    //
    // for now we assume only HB scaling and triggered mode
    //
    // in case of triggered data we assume that that the for pageCnt == 1 we have
    //   triggerType == 0x10 in the firt packet
    //
    if (mManager) {
      if (mManager->mDetectDataType) {
        const uint64_t triggerTypeForTriggeredData = 0x10;
        const uint64_t triggerType = RDHUtils::getTriggerType(rdh);
        const uint64_t pageCnt = RDHUtils::getPageCounter(rdh);
        const uint64_t linkID = RDHUtils::getLinkID(rdh);

        // if (pageCnt == 0) {
        if ((linkID == 15) || (detField == 0x1) || (detField == 0x2)) {
          mManager->mRawDataType = RAWDataType::LinkZS;
          LOGP(info, "Detected LinkZS data");
          mManager->mDetectDataType = false;
        }
        //}

        if (pageCnt == 1) {
          if (triggerType == triggerTypeForTriggeredData) {
            mManager->mDataType = DataType::Triggered;
            O2INFO("Detected triggered data");
          } else {
            mManager->mDataType = DataType::HBScaling;
            O2INFO("Detected HB scaling");
          }
          mManager->mDetectDataType = false;
        }
      }
    }

    // ===| get relavant data information |=====================================
    auto feeId = RDHUtils::getFEEID(rdh);
    // treat old RDH where feeId was not set properly
    if (feeId == 4844) {
      const rdh_utils::FEEIDType cru = RDHUtils::getCRUID(rdh);
      const rdh_utils::FEEIDType link = RDHUtils::getLinkID(rdh);
      const rdh_utils::FEEIDType endPoint = RDHUtils::getEndPointID(rdh);
      feeId = rdh_utils::getFEEID(cru, endPoint, link);

      RDHUtils::setFEEID(rdh, feeId);
    }
    const auto heartbeatOrbit = RDHUtils::getHeartBeatOrbit(rdh);
    const auto heartbeatOrbitEvent = isTFfile ? dh.firstTForbit : RDHUtils::getHeartBeatOrbit(rdh);
    const auto endPoint = rdh_utils::getEndPoint(feeId);
    auto linkID = rdh_utils::getLink(feeId);
    if (linkID == 21 && detField == 0x02) {
      linkID = 0;
    }
    const auto globalLinkID = linkID + endPoint * 12;

    // ===| check if cru should be forced |=====================================
    if (!mForceCRU) {
      mCRU = rdh_utils::getCRU(feeId);
      // mCRU = RDHUtils::getCRUID(rdh); // work-around for MW2 data
    } else {
      // overwrite cru id in rdh for further processing
      RDHUtils::setCRUID(rdh, mCRU);
    }

    // ===| find evnet info or create a new one |===============================
    RawReaderCRUEventSync::LinkInfo* linkInfo = nullptr;
    if (mManager) {
      // in case of triggered mode, we use the first heartbeat orbit as event identifier
      if ((lastHeartbeatOrbit == 0) || (heartbeatOrbitEvent != lastHeartbeatOrbit)) {
        mManager->mEventSync.createEvent(heartbeatOrbitEvent, mManager->getDataType());
        lastHeartbeatOrbit = heartbeatOrbitEvent;
      }
      linkInfo = &mManager->mEventSync.getLinkInfo(rdh, mManager->getDataType());
      mManager->mEventSync.setCRUSeen(mCRU, mReaderNumber);
    }

    // ===| set up packet descriptor map for GBT frames |=======================
    //
    // * check Header for Header ID
    // * create the packet descriptor
    // * set the mLinkPresent flag
    //
    if ((rdh.word0 & 0x0000FFF0) == RDH_HEADERWORD0) {
      // non 0 stop bit means data with payload
      if (payloadSize) {
        mPacketDescriptorMaps[globalLinkID].emplace_back(currentPos, mCRU, linkID, endPoint, memorySize, packetSize, heartbeatOrbit);
        mLinkPresent[globalLinkID] = true;
        mPacketsPerLink[globalLinkID]++;
        if (linkInfo) {
          linkInfo->PacketPositions.emplace_back(mPacketsPerLink[globalLinkID] - 1);
          linkInfo->IsPresent = true;
          linkInfo->PayloadSize += payloadSize;
        }
      }
      if (RDHUtils::getStop(rdh) == 1) {
        // stop bit 1 means we hit the HB end frame without payload.
        // This marks the end of an "event" in HB scaling mode.
        if (linkInfo) {
          linkInfo->HBEndSeen = true;
        }
      }
    } else {
      O2ERROR("Found header word %x and required header word %x don't match, at %zu, stopping file scan", rdh.word0, RDH_HEADERWORD0, currentPos);
      break;
    }

    // debug output
    if (mVerbosity && CHECK_BIT(mDebugLevel, DebugLevel::RDHDump)) {
      printHorizontal(rdh);
      if (RDHUtils::getStop(rdh)) {
        std::cout << "\n";
        printHeader();
      }
    }

    file.seekg(offset, file.cur);
    ++currentPacket;
    currentPos = file.tellg();
  }

  // close the File
  file.close();

  // go through events and set the status if links were seen
  if (mManager) {
    // in case of triggered mode, we use the first heartbeat orbit as event identifier
    mManager->mEventSync.setLinksSeen(mCRU, mLinkPresent);
  }

  if (mVerbosity) {
    // show the mLinkPresent map
    std::cout << "Links present" << std::endl;
    for (int i = 0; i < MaxNumberOfLinks; i++) {
      mLinkPresent[i] == true ? std::cout << "1 " : std::cout << "0 ";
    };
    std::cout << '\n';

    std::cout << std::dec
              << "File Name         : " << mInputFileName << "\n"
              << "File size [bytes] : " << mFileSize << "\n"
              << "Packets           : " << numPackets << "\n"
              << "\n";

    if (mVerbosity > 1) {
      // ===| display packet statistics |===
      for (int i = 0; i < MaxNumberOfLinks; i++) {
        if (mLinkPresent[i]) {
          std::cout << "Packets for link " << i << ": " << mPacketsPerLink[i] << "\n";
          //
          // ===| display the packet descriptor map |===
          for (const auto& pd : mPacketDescriptorMaps[i]) {
            std::cout << pd << "\n";
          }
        }
      }
      std::cout << "\n";
    }
  }

  mFileIsScanned = true;

  return 0;
}

void RawReaderCRU::findSyncPositions()
{
  auto& file = getFileHandle();

  // loop over the MaxNumberOfLinks potential links in the data
  // only if data from the link is present and selected
  // for decoding it will be decoded.
  for (int link = 0; link < MaxNumberOfLinks; link++) {
    // all links have been selected
    if (!checkLinkPresent(link)) {
      continue;
    }

    if (mVerbosity) {
      std::cout << "Finding sync pattern for link " << link << std::endl;
      std::cout << "Num packets : " << mPacketsPerLink[link] << std::endl;
    }

    GBTFrame gFrame;
    uint32_t packetID{0};

    // loop over the packets for each link and process them
    for (auto packet : mPacketDescriptorMaps[link]) {
      gFrame.setPacketNumber(packetID);

      file.seekg(packet.getPayloadOffset(), file.beg);

      // read in the data frame by frame, extract the 5-bit halfwords for
      // the two data streams and store them in the corresponding half-word
      // vectors
      for (int frames = 0; frames < packet.getPayloadSize() / 16; frames++) {
        file >> gFrame;
        // extract the half words from the 4 32-bit words
        gFrame.getFrameHalfWords();
        gFrame.updateSyncCheck(mSyncPositions[link]);
      };

      packetID++;

      // TODO: In future there might be more then one sync in the stream
      //       this should be takein into account
      if (syncFoundForLink(link)) {
        if (mVerbosity && CHECK_BIT(mDebugLevel, DebugLevel::SyncPositions)) {
          std::cout << "Sync positions for link " << link << '\n';
          const auto& syncs = mSyncPositions[link];
          for (int i = 0; i < syncs.size(); ++i) {
            std::cout << i << " : " << syncs[i];
          }
          std::cout << '\n';
        }
        break;
      }
    }
  }
}

int RawReaderCRU::processPacket(GBTFrame& gFrame, uint32_t startPos, uint32_t size, ADCRawData& rawData)
{
  // open the data file
  auto& file = getFileHandle();

  // jump to the start position of the packet
  file.seekg(startPos, file.beg);

  // read in the data frame by frame, extract the 5-bit halfwords for
  // the two data streams and store them in the corresponding half-word
  // vectors
  for (int frames = 0; frames < size / 16; frames++) {
    file >> gFrame;

    // extract the half words from the 4 32-bit words
    gFrame.getFrameHalfWords();

    // debug output
    if (mVerbosity && CHECK_BIT(mDebugLevel, DebugLevel::GBTFrames)) {
      std::cout << gFrame;
    }

    gFrame.getAdcValues(rawData);
    gFrame.updateSyncCheck(mVerbosity && CHECK_BIT(mDebugLevel, DebugLevel::SyncPositions));
    if (!(rawData.getNumTimebins() % 16) && (rawData.getNumTimebins() >= mNumTimeBins * 16)) {
      return 1;
    }
  };
  return 0;
}

int RawReaderCRU::processMemory(const std::vector<std::byte>& data, ADCRawData& rawData)
{
  GBTFrame gFrame;

  const bool dumpSyncPositoins = CHECK_BIT(mDebugLevel, DebugLevel::SyncPositions);

  // 16 bytes is the size of a GBT frame
  for (int iFrame = 0; iFrame < data.size() / 16; ++iFrame) {
    gFrame.setFrameNumber(iFrame);
    gFrame.setPacketNumber(iFrame / 508);

    // in readFromMemory a simple memcopy to the internal data structure is done
    // I tried using the memory block directly, storing in an internal data member
    // reinterpret_cast<const uint32_t*>(data.data() + iFrame * 16), so it could be accessed the
    // same way as the mData array.
    // however, this was ~5% slower in execution time. I suspect due to cache misses
    gFrame.readFromMemory(gsl::span<const std::byte>(data.data() + iFrame * 16, 16));

    // extract the half words from the 4 32-bit words
    gFrame.getFrameHalfWords();

    // debug output
    if (mVerbosity && CHECK_BIT(mDebugLevel, DebugLevel::GBTFrames)) {
      std::cout << gFrame;
    }

    gFrame.getAdcValues(rawData);
    gFrame.updateSyncCheck(mVerbosity && dumpSyncPositoins);
    if (!(rawData.getNumTimebins() % 16) && (rawData.getNumTimebins() >= mNumTimeBins * 16)) {
      break;
    }
  };

  if (mDumpTextFiles && dumpSyncPositoins) {
    const auto fileName = mOutputFilePrefix + "/LinkPositions.txt";
    std::ofstream file(fileName, std::ofstream::app);
    auto& syncPositions = gFrame.getSyncArray();

    for (int s = 0; s < 5; ++s) {
      auto& syncPos = syncPositions[s];
      if (syncPos.synched()) {
        file << mEventNumber << "\t"
             << mCRU.number() << "\t"
             << mLink << "\t"
             << s << "\t"
             << syncPos.getPacketNumber() << "\t"
             << syncPos.getFrameNumber() << "\t"
             << syncPos.getHalfWordPosition() << "\n";
      }
    }
  }
  return 0;
}

size_t RawReaderCRU::getNumberOfEvents() const
{
  return mManager ? mManager->mEventSync.getNumberOfEvents(mCRU) : 0;
}

void RawReaderCRU::fillADCdataMap(const ADCRawData& rawData)
{
  // TODO: Ugly copy below in runADCDataCallback. Modification in here should be also refected there
  const auto& mapper = Mapper::instance();

  // cru and link must be set correctly before
  const CRU cru(mCRU);
  const int fecLinkOffsetCRU = (mapper.getPartitionInfo(cru.partition()).getNumberOfFECs() + 1) / 2;
  const int fecInPartition = (mLink % 12) + (mLink > 11) * fecLinkOffsetCRU;
  const int regionIter = mCRU % 2;

  const int sampaMapping[10] = {0, 0, 1, 1, 2, 3, 3, 4, 4, 2};
  const int channelOffset[10] = {0, 16, 0, 16, 0, 0, 16, 0, 16, 16};

  for (int istreamm = 0; istreamm < 5; ++istreamm) {
    const int partitionStream = istreamm + regionIter * 5;
    const int sampa = sampaMapping[partitionStream];

    const auto& dataVector = rawData.getDataVector(istreamm);

    // loop over all data. Each stream has 16 ADC values for each sampa channel times nTimeBins
    for (int idata = 0; idata < dataVector.size(); ++idata) {
      const int ichannel = idata % 16;
      const int sampaChannel = ichannel + channelOffset[partitionStream];
      const auto& padPos = mapper.padPosRegion(cru.region(), fecInPartition, sampa, sampaChannel);
      mADCdata[padPos].emplace_back(dataVector[idata]);
    }
  }
}

void RawReaderCRU::runADCDataCallback(const ADCRawData& rawData)
{
  // TODO: Ugly copy below in runADCDataCallback. Modification in here should be also refected there
  const auto& mapper = Mapper::instance();

  // cru and link must be set correctly before
  const CRU cru(mCRU);
  const int fecLinkOffsetCRU = (mapper.getPartitionInfo(cru.partition()).getNumberOfFECs() + 1) / 2;
  const int fecInPartition = (mLink % 12) + (mLink > 11) * fecLinkOffsetCRU;
  const int regionIter = mCRU % 2;

  const int sampaMapping[10] = {0, 0, 1, 1, 2, 3, 3, 4, 4, 2};
  const int channelOffset[10] = {0, 16, 0, 16, 0, 0, 16, 0, 16, 16};

  for (int istreamm = 0; istreamm < 5; ++istreamm) {
    const int partitionStream = istreamm + regionIter * 5;
    const int sampa = sampaMapping[partitionStream];

    const auto& dataVector = rawData.getDataVector(istreamm);
    if (dataVector.size() < 16) {
      continue;
    }

    // loop over all data. Each stream has 16 ADC values for each sampa channel times nTimeBins
    for (int ichannel = 0; ichannel < 16; ++ichannel) {
      const int sampaChannel = ichannel + channelOffset[partitionStream];
      const auto& padPos = mapper.padROCPos(cru, fecInPartition, sampa, sampaChannel);
      // printf("Fill: %d %d %d %d / %d %d %d\n", int(mCRU), int(cru.roc()), ichannel, sampaChannel, int(padPos.getROC()), int(padPos.getRow()), int(padPos.getPad()));
      mManager->mADCDataCallback(padPos, cru, gsl::span<const uint32_t>(dataVector.data() + ichannel, dataVector.size() - ichannel));
    }
  }
}

int RawReaderCRU::processDataFile()
{
  GBTFrame gFrame;
  // gFrame.setSyncPositions(mSyncPositions[mLink]);

  ADCRawData rawData;

  if (mVerbosity) {
    std::cout << "Processing data for link " << mLink << std::endl;
    std::cout << "Num packets : " << mPacketsPerLink[mLink] << std::endl;
  }

  // ===| mapping to be updated |===============================================
  // CRU cru; // assuming each decoder only hast once CRU
  const int link = mLink;

  const auto& linkInfoArray = mManager->mEventSync.getLinkInfoArrayForEvent(mEventNumber, mCRU);

  // loop over the packets for each link and process them
  // for (const auto& packet : mPacketDescriptorMaps[link]) {
  for (auto packetNumber : linkInfoArray[link].PacketPositions) {
    const auto& packet = mPacketDescriptorMaps[link][packetNumber];

    // cru = packet.getCRUID();
    //  std::cout << "Packet : " << packetID << std::endl;
    gFrame.setPacketNumber(packetNumber);
    int retCode = processPacket(gFrame, packet.getPayloadOffset(), packet.getPayloadSize(), rawData);
    if (retCode) {
      break;
    }

    // if (rawData.getNumTimebins() >= mNumTimeBins * 16)
    // break;
  };

  // ===| fill ADC data to the output structure |===
  if (mFillADCdataMap) {
    fillADCdataMap(rawData);
  }
  if (mManager && mManager->mADCDataCallback) {
    runADCDataCallback(rawData);
  }

  // std::cout << "Output Data" << std::endl;

  //// display the data
  if (mDumpTextFiles) {
    std::ofstream file;
    std::string fileName;
    for (int s = 0; s < 5; s++) {
      if (mStream == 0x0 or ((mStream >> s) & 0x1) == 0x1) {
        if (gFrame.syncFound(s) == false) {
          std::cout << "No sync found" << std::endl;
        }
        // debug output
        rawData.setOutputStream(s);
        rawData.setNumTimebins(mNumTimeBins);
        if (mVerbosity && CHECK_BIT(mDebugLevel, DebugLevel::ADCValues)) {
          std::cout << rawData << std::endl;
        };
        // write the data to file
        // fileName = "ADC_" + std::to_string(mLink) + "_" + std::to_string(s);
        fileName = mOutputFilePrefix + "_ADC_" + std::to_string(mLink) + "_" + std::to_string(s) + ".txt";
        file.open(fileName, std::ofstream::out);
        file << rawData;
        file.close();
      }
    }
  }
  return 0;
}

void RawReaderCRU::processDataMemory()
{

  if (mVerbosity) {
    std::cout << "Processing data for link " << mLink << std::endl;
    std::cout << "Num packets : " << mPacketsPerLink[mLink] << std::endl;
  }

  size_t dataSize = 4000 * 16;
  // if (mDataType == DataType::HBScaling) {
  // dataSize =
  // } else if (mDataType == DataType::Triggered) {
  //// in triggered mode 4000 GBT frames are read out
  //// 16 is the size of a GBT frame in byte
  // dataSize = 4000 * 16;
  // }

  std::vector<std::byte> data;
  data.reserve(dataSize);
  collectGBTData(data);

  ADCRawData rawData;
  processMemory(data, rawData);

  // ===| fill ADC data to the output structure |===
  if (mFillADCdataMap) {
    fillADCdataMap(rawData);
  }
  if (mManager->mADCDataCallback) {
    runADCDataCallback(rawData);
  }
}

void RawReaderCRU::collectGBTData(std::vector<std::byte>& data)
{
  const auto& linkInfoArray = mManager->mEventSync.getLinkInfoArrayForEvent(mEventNumber, mCRU);
  auto& file = getFileHandle();

  size_t presentDataPosition = 0;

  // loop over the packets for each link and process them
  // for (const auto& packet : mPacketDescriptorMaps[link]) {
  for (auto packetNumber : linkInfoArray[mLink].PacketPositions) {
    const auto& packet = mPacketDescriptorMaps[mLink][packetNumber];

    const auto payloadStart = packet.getPayloadOffset();
    const auto payloadSize = size_t(packet.getPayloadSize());
    data.insert(data.end(), payloadSize, (std::byte)0);
    // jump to the start position of the packet
    file.seekg(payloadStart, std::ios::beg);

    // read data
    file.read(((char*)data.data()) + presentDataPosition, payloadSize);

    presentDataPosition += payloadSize;
  }
}

void RawReaderCRU::processLinkZS()
{
  const auto& eventInfo = mManager->mEventSync.getEventInfo(mEventNumber);
  const auto& linkInfoArray = eventInfo.CRUInfoArray[mCRU].LinkInformation;
  const auto firstOrbitInEvent = eventInfo.getFirstOrbit();

  auto& file = getFileHandle();

  char buffer[8192];

  // loop over the packets for each link and process them
  for (const auto packetNumber : linkInfoArray[mLink].PacketPositions) {
    const auto& packet = mPacketDescriptorMaps[mLink][packetNumber];
    const size_t payloadOffset = packet.getPayloadOffset();
    const size_t payloadSize = packet.getPayloadSize();
    if ((payloadOffset + payloadSize) > mFileSize) {
      LOGP(error, "File truncated at {}, size {} would exceed file size of {}", payloadOffset, payloadSize, mFileSize);
      break;
    }
    file.seekg(payloadOffset, file.beg);
    file.read(buffer, payloadSize);
    const uint32_t syncOffsetReference = 144;                                                                                                                                              // <<< TODO: fix value as max offset over all links
    o2::tpc::raw_processing_helpers::processZSdata(buffer, payloadSize, packet.getFEEID(), packet.getHeartBeatOrbit(), firstOrbitInEvent, syncOffsetReference, mManager->mLinkZSCallback); // last parameter should be true for MW2 data
  }
}

void RawReaderCRU::processLinks(const uint32_t linkMask)
{
  if (!mManager) {
    LOGP(error, "cannont run without manager");
    return;
  }

  try {
    // read the input data from file, create the packet descriptor map
    // and the mLinkPresent map.
    scanFile();

    // check if selected event is valid
    if (mEventNumber >= mManager->mEventSync.getNumberOfEvents()) {
      O2ERROR("Selected event number %u is larger than the events in file %lu", mEventNumber, mManager->mEventSync.getNumberOfEvents());
      return;
    }

    // loop over the MaxNumberOfLinks potential links in the data
    // only if data from the link is present and selected
    // for decoding it will be decoded.
    for (int lnk = 0; lnk < MaxNumberOfLinks; lnk++) {
      // all links have been selected
      if (((linkMask == 0) || ((linkMask >> lnk) & 1)) && checkLinkPresent(lnk) == true) {
        // set the active link variable and process the data
        if (mDebugLevel) {
          fmt::print("Processing link {}\n", lnk);
        }
        setLink(lnk);
        if (mManager->mRawDataType == RAWDataType::GBT) {
          // processDataFile();
          processDataMemory();
        } else {
          processLinkZS();
        }
      }
    }

  } catch (const RawReaderCRU::Error& e) {
    std::cout << e.what() << std::endl;
    exit(10);
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << std::endl;
    exit(100);
  } catch (...) {
    std::cerr << "ERROR: Unknown error" << std::endl;
    exit(1000);
  }
}

void RawReaderCRU::processFile(const std::string_view inputFile, uint32_t timeBins, uint32_t linkMask, uint32_t stream, uint32_t debugLevel, uint32_t verbosity, const std::string_view outputFilePrefix)
{
  // Instantiate the RawReaderCRU
  RawReaderCRUManager cruManager;
  cruManager.setDebugLevel(debugLevel);
  // RawReaderCRU& rawReaderCRU = cruManager.createReader(inputFile, timeBins, 0, stream, debugLevel, verbosity, outputFilePrefix);
  cruManager.setupReaders(inputFile, timeBins, debugLevel, verbosity, outputFilePrefix);
  for (auto& reader : cruManager.getReaders()) {
    reader->mDumpTextFiles = true;
    reader->mFillADCdataMap = false;
  }
  cruManager.init();

  if (CHECK_BIT(debugLevel, DebugLevel::SyncPositions)) {
    std::string outPrefix = outputFilePrefix.length() ? outputFilePrefix.data() : "./";
    const auto fileName = outPrefix + "/LinkPositions.txt";
    std::ofstream file(fileName, std::ofstream::out);
    file << "EventNumber/I"
         << ":"
         << "CRU"
         << ":"
         << "Link"
         << ":"
         << "Stream"
         << ":"
         << "PacketNumber"
         << ":"
         << "FrameNumber"
         << ":"
         << "HalfWordPosition"
         << "\n";
    file.close();
  }

  for (int ievent = 0; ievent < cruManager.getNumberOfEvents(); ++ievent) {
    fmt::print("=============| event {: 5d} |===============\n", ievent);

    cruManager.processEvent(ievent);
  }
}

void RawReaderCRU::copyEvents(const std::vector<uint32_t>& eventNumbers, std::string outputDirectory, std::ios_base::openmode mode)
{
  // assemble output file name
  std::string outputFileName(gSystem->BaseName(mInputFileName.data()));
  if (outputDirectory.empty()) {
    outputFileName.insert(0, "filtered.");
    outputDirectory = gSystem->DirName(mInputFileName.data());
  }
  outputFileName.insert(0, "/");
  outputFileName.insert(0, outputDirectory);

  std::ofstream outputFile(outputFileName, std::ios_base::binary | mode);

  // open the input file
  auto& file = getFileHandle();

  // data buffer. Maximum size is 8k
  char buffer[8192];

  // loop over events
  for (const auto eventNumber : eventNumbers) {

    const auto& linkInfoArray = mManager->mEventSync.getLinkInfoArrayForEvent(eventNumber, mCRU);

    for (int iLink = 0; iLink < MaxNumberOfLinks; ++iLink) {
      const auto& linkInfo = linkInfoArray[iLink];
      if (!linkInfo.IsPresent) {
        continue;
      }
      for (auto packetNumber : linkInfo.PacketPositions) {
        const auto& packet = mPacketDescriptorMaps[iLink][packetNumber];
        file.seekg(packet.getHeaderOffset(), file.beg);
        file.read(buffer, packet.getPacketSize());
        outputFile.write(buffer, packet.getPacketSize());
      }
    }
  }
}

void RawReaderCRU::writeGBTDataPerLink(std::string_view outputDirectory, int maxEvents)
{
  // open the input file
  auto& file = getFileHandle();

  // data buffer. Maximum size is 8k
  char buffer[8192];

  // loop over events
  for (int eventNumber = 0; eventNumber < getNumberOfEvents(); ++eventNumber) {
    if ((maxEvents > -1) && (eventNumber >= maxEvents)) {
      break;
    }

    const auto& linkInfoArray = mManager->mEventSync.getLinkInfoArrayForEvent(eventNumber, mCRU);

    for (int iLink = 0; iLink < MaxNumberOfLinks; ++iLink) {
      const auto& linkInfo = linkInfoArray[iLink];
      if (!linkInfo.IsPresent) {
        continue;
      }
      printf("Event %4d, Link %2d\n", eventNumber, iLink);

      const int ep = iLink >= 12;
      const int link = iLink - (ep)*12;
      auto outputFileName = fmt::format("{}/CRU_{:02}_EP_{}_Link_{:02}", outputDirectory.data(), (int)mCRU, ep, link);
      std::ofstream outputFile(outputFileName, std::ios_base::binary | std::ios_base::app);

      for (auto packetNumber : linkInfo.PacketPositions) {
        const auto& packet = mPacketDescriptorMaps[iLink][packetNumber];
        file.seekg(packet.getPayloadOffset(), file.beg);
        file.read(buffer, packet.getPayloadSize());
        outputFile.write(buffer, packet.getPayloadSize());
      }
    }
  }
}
//==============================================================================
//===| stream overloads for helper classes |====================================
//

void ADCRawData::streamTo(std::ostream& output) const
{
  const auto numTimeBins = std::min(getNumTimebins(), mNumTimeBins);
  for (int i = 0; i < numTimeBins * 16; i++) {
    if (i % 16 == 0) {
      output << std::setw(4) << std::to_string(i / 16) << " : ";
    }
    output << std::setw(4) << mADCRaw[(mOutputStream)][i];
    output << (((i + 1) % 16 == 0) ? "\n" : " ");
  };
};

void GBTFrame::streamFrom(std::istream& input)
{
  mFilePos = input.tellg();
  mFrameNum++;
  for (int i = 0; i < 4; i++) {
    input.read(reinterpret_cast<char*>(&(mData[i])), sizeof(mData[i]));
  };
}

// std::ostream& operator<<(std::ostream& output, const RawReaderCRU::GBTFrame& frame)
void GBTFrame::streamTo(std::ostream& output) const
{
  const auto offset = mPrevHWpos ^ 4;
  output << std::dec << "\033[94m"
         << std::setfill('0') << std::setw(8) << mPacketNum << " "
         << std::setfill('0') << std::setw(8) << mFilePos << " "
         << std::setfill('0') << std::setw(8) << mFrameNum << " : "
         << std::hex
         << std::setfill('0') << std::setw(8) << mData[3] << "."
         << std::setfill('0') << std::setw(8) << mData[2] << "."
         << std::setfill('0') << std::setw(8) << mData[1] << "."
         << std::setfill('0') << std::setw(8) << mData[0] << " : "
         << "\033[0m";
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 4; j++) {
      output << std::hex << std::setw(4) << mFrameHalfWords[i][j + offset] << " ";
    }
    output << "| ";
  };
  output << std::endl;
}

// friend std::ostream& operator<<(std::ostream& output, const PacketDescriptor& PD)
void RawReaderCRU::PacketDescriptor::streamTo(std::ostream& output) const
{
  output << "===| Packet Descriptor |================================="
         << "\n";
  output << "CRU ID         :  " << std::dec << getCRUID() << "\n";
  output << "Link ID        :  " << getLinkID() << "\n";
  output << "Global Link ID :  " << getGlobalLinkID() << "\n";
  output << "Header Offset  :  " << getHeaderOffset() << "\n";
  output << "Payload Offset :  " << getPayloadOffset() << "\n";
  output << "Payload Size   :  " << getPayloadSize() << "\n";
  output << "\n";
}

std::ostream& operator<<(std::ostream& output, const RDH& rdh)
{
  output << "word0            : 0x" << std::setfill('0') << std::setw(16) << std::hex << rdh.word0 << "\n"
         << std::dec;
  output << "  version        : " << RDHUtils::getVersion(rdh) << "\n";
  output << "  headerSize     : " << RDHUtils::getHeaderSize(rdh) << "\n";
  if (RDHUtils::getVersion(rdh) == 4) {
    output << "  blockLength    : " << RDHUtils::getBlockLength(rdh) << "\n";
  }
  output << "  feeId          : " << RDHUtils::getFEEID(rdh) << "\n";
  output << "  priority       : " << RDHUtils::getPriorityBit(rdh) << "\n";
  output << "\n";

  output << "word1            : 0x" << std::setfill('0') << std::setw(16) << std::hex << rdh.word1 << "\n"
         << std::dec;
  output << "  Offset to next : " << int(RDHUtils::getOffsetToNext(rdh)) << "\n";
  output << "  Memory size    : " << int(RDHUtils::getMemorySize(rdh)) << "\n";
  output << "  LinkID         : " << int(RDHUtils::getLinkID(rdh)) << "\n";
  output << "  Global LinkID  : " << int(RDHUtils::getLinkID(rdh)) + (((rdh.word1 >> 32) >> 28) * 12) << "\n";
  output << "  CRUid          : " << RDHUtils::getCRUID(rdh) << "\n";
  output << "  Packet Counter : " << RDHUtils::getPacketCounter(rdh) << "\n";
  output << "  DataWrapper-ID : " << (((rdh.word1 >> 32) >> 28) & 0x01) << "\n";
  output << "\n";

  output << "word2            : 0x" << std::setfill('0') << std::setw(16) << std::hex << rdh.word2 << "\n"
         << std::dec;
  output << "  triggerOrbit   : " << RDHUtils::getTriggerOrbit(rdh) << "\n";
  output << "  heartbeatOrbit : " << RDHUtils::getHeartBeatOrbit(rdh) << "\n";
  output << "\n";

  output << "word3            : 0x" << std::setfill('0') << std::setw(16) << std::hex << rdh.word3 << "\n"
         << std::dec;
  output << "\n";

  output << "word4            : 0x" << std::setfill('0') << std::setw(16) << std::hex << rdh.word4 << "\n"
         << std::dec;
  output << "  triggerBC      : " << RDHUtils::getTriggerBC(rdh) << "\n";
  output << "  heartbeatBC    : " << RDHUtils::getHeartBeatBC(rdh) << "\n";
  output << "  triggerType    : " << RDHUtils::getTriggerType(rdh) << "\n";
  output << "\n";

  output << "word5            : 0x" << std::setfill('0') << std::setw(16) << std::hex << rdh.word5 << "\n"
         << std::dec;
  output << "\n";

  output << "word6            : 0x" << std::setfill('0') << std::setw(16) << std::hex << rdh.word6 << "\n"
         << std::dec;
  output << "  detectorField  : " << RDHUtils::getDetectorField(rdh) << "\n";
  output << "  par            : " << RDHUtils::getDetectorPAR(rdh) << "\n";
  output << "  stop           : " << RDHUtils::getStop(rdh) << "\n";
  output << "  pageCnt        : " << RDHUtils::getPageCounter(rdh) << "\n";
  output << "\n";

  output << "word7            : 0x" << std::setfill('0') << std::setw(16) << std::hex << rdh.word7 << "\n"
         << std::dec;
  output << "\n";

  return output;
}

void printHeader()
{
  fmt::print("{:>5} {:>4} {:>4} {:>4} {:>6} {:>4} {:>3} {:>4} {:>10} {:>5} {:>1}\n",
             "PkC", "pCnt", "trg", "fId", "OffN", "Mem", "CRU", "GLID", "HBOrbit", "HBBC", "s");
}

void printHorizontal(const RDH& rdh)
{
  const int globalLinkID = int(RDHUtils::getLinkID(rdh)) + (((rdh.word1 >> 32) >> 28) * 12);

  fmt::print("{:>5} {:>4} {:>4} {:>4} {:>6} {:>4} {:>3} {:>4} {:>10} {:>5} {:>1}\n",
             (uint64_t)RDHUtils::getPacketCounter(rdh),
             (uint64_t)RDHUtils::getPageCounter(rdh),
             (uint64_t)RDHUtils::getTriggerType(rdh),
             (uint64_t)RDHUtils::getFEEID(rdh),
             (uint64_t)RDHUtils::getOffsetToNext(rdh),
             (uint64_t)RDHUtils::getMemorySize(rdh),
             (uint64_t)RDHUtils::getCRUID(rdh),
             (uint64_t)globalLinkID,
             (uint64_t)RDHUtils::getHeartBeatOrbit(rdh),
             (uint64_t)RDHUtils::getHeartBeatBC(rdh),
             (uint64_t)RDHUtils::getStop(rdh));
}

template <typename DataType>
std::istream& operator>>(std::istream& input, DataType& data)
{
  const int dataTypeSize = sizeof(data);
  auto charPtr = reinterpret_cast<char*>(&data);
  input.read(charPtr, dataTypeSize);
  return input;
}

//==============================================================================
void RawReaderCRUManager::init()
{
  if (mIsInitialized) {
    return;
  }

  for (auto& reader : mRawReadersCRU) {
    reader->scanFile();
  }

  mEventSync.sortEvents();
  mEventSync.analyse(mRawDataType);

  O2INFO("Event information:");
  O2INFO("    Number of all events:      %lu", getNumberOfEvents());
  O2INFO("    Number of complete events: %lu", getNumberOfCompleteEvents());

  if (CHECK_BIT(mDebugLevel, DebugLevel::EventInfo)) {
    std::cout << mEventSync;
  }

  mIsInitialized = true;
}

void RawReaderCRUManager::setupReaders(const std::string_view inputFileNames,
                                       uint32_t numTimeBins,
                                       uint32_t debugLevel,
                                       uint32_t verbosity,
                                       const std::string_view outputFilePrefix)
{
  reset();
  const TString files = gSystem->GetFromPipe(TString::Format("ls %s", inputFileNames.data()));
  std::unique_ptr<TObjArray> arr(files.Tokenize("\n"));
  setDebugLevel(debugLevel);

  for (auto file : *arr) {
    // fix the number of time bins
    auto& reader = createReader(file->GetName(), numTimeBins);
    reader.setReaderNumber(mRawReadersCRU.size() - 1);
    reader.setVerbosity(verbosity);
    reader.setDebugLevel(debugLevel);
    reader.setOutputFilePrefix(outputFilePrefix);
    O2INFO("Adding file: %s\n", file->GetName());
  }
}

void RawReaderCRUManager::copyEvents(const std::vector<uint32_t> eventNumbers, std::string_view outputDirectory, std::ios_base::openmode mode)
{
  // make sure events have been built
  init();
  const auto& cruSeen = mEventSync.getCRUSeen();

  for (size_t iCRU = 0; iCRU < cruSeen.size(); ++iCRU) {
    const auto readerNumber = cruSeen[iCRU];
    if (readerNumber >= 0) {
      auto& reader = mRawReadersCRU[readerNumber];
      reader->forceCRU(iCRU);
      reader->copyEvents(eventNumbers, outputDirectory.data(), mode);
    }
  }
}

void RawReaderCRUManager::copyEvents(const std::string_view inputFileNames, const std::vector<uint32_t> eventNumbers, std::string_view outputDirectory, std::ios_base::openmode mode)
{
  RawReaderCRUManager manager;
  manager.setupReaders(inputFileNames);
  manager.copyEvents(eventNumbers, outputDirectory, mode);
}

void RawReaderCRUManager::writeGBTDataPerLink(std::string_view outputDirectory, int maxEvents)
{
  init();
  const auto& cruSeen = mEventSync.getCRUSeen();

  for (size_t iCRU = 0; iCRU < cruSeen.size(); ++iCRU) {
    const auto readerNumber = cruSeen[iCRU];
    if (readerNumber >= 0) {
      auto& reader = mRawReadersCRU[readerNumber];
      reader->forceCRU(iCRU);
      reader->writeGBTDataPerLink(outputDirectory, maxEvents);
    }
  }
}

void RawReaderCRUManager::writeGBTDataPerLink(const std::string_view inputFileNames, std::string_view outputDirectory, int maxEvents)
{
  if (!std::filesystem::exists(outputDirectory)) {
    if (!std::filesystem::create_directories(outputDirectory)) {
      LOG(fatal) << "could not create output directory " << outputDirectory;
    } else {
      LOG(info) << "created output directory " << outputDirectory;
    }
  }

  RawReaderCRUManager manager;
  manager.setupReaders(inputFileNames);
  manager.writeGBTDataPerLink(outputDirectory, maxEvents);
}

void RawReaderCRUManager::processEvent(uint32_t eventNumber, EndReaderCallback endReader)
{
  const auto& cruSeen = mEventSync.getCRUSeen();

  for (size_t iCRU = 0; iCRU < cruSeen.size(); ++iCRU) {
    const auto readerNumber = cruSeen[iCRU];
    if (readerNumber >= 0) {
      auto& reader = mRawReadersCRU[readerNumber];
      if (reader->getFillADCdataMap()) {
        LOGF(warning, "Filling of ADC data map not supported in RawReaderCRUManager::processEvent, it is disabled now. use ADCDataCallback");
        reader->setFillADCdataMap(false);
      }
      reader->setEventNumber(eventNumber);
      reader->forceCRU(iCRU);
      reader->processLinks();
      if (endReader) {
        endReader();
      }
    }
  }
}
