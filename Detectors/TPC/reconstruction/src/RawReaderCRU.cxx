// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file RawReaderCRU.cxx
/// \author Jens Wiechula (Jens.Wiechula@ikf.uni-frankfurt.de)
/// \author Torsten Alt (Torsten.Alt@cern.ch)

#include <fmt/format.h>

#include "TSystem.h"
#include "TObjArray.h"

#include "TPCReconstruction/RawReaderCRU.h"
#include "TPCBase/Mapper.h"
#include "Framework/Logger.h"

#define CHECK_BIT(var, pos) ((var) & (1 << (pos)))
using RDH = o2::header::RAWDataHeader;
//using namespace o2::tpc;
using namespace o2::tpc::rawreader;

std::ostream& operator<<(std::ostream& output, const RDH& rdh);
std::istream& operator>>(std::istream& input, RDH& rdh);
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

RawReaderCRUEventSync::EventInfo& RawReaderCRUEventSync::createEvent(const RDH& rdh, DataType dataType)
{
  const auto heartbeatOrbit = rdh.heartbeatOrbit;

  for (auto& ev : mEventInformation) {
    const auto hbMatch = ev.hasHearbeatOrbit(heartbeatOrbit);
    if (hbMatch) {
      return ev;
    } else if (ev.HeartbeatOrbits.back() == heartbeatOrbit - 1) {
      ev.HeartbeatOrbits.emplace_back(heartbeatOrbit);
      return ev;
    }
  }
  return mEventInformation.emplace_back(heartbeatOrbit);
}

void RawReaderCRUEventSync::analyse()
{
  //expected number of packets in one HBorbit
  const size_t numberOfPackets = ExpectedNumberOfPacketsPerHBFrame;

  for (int iEvent = mEventInformation.size() - 1; iEvent >= 0; --iEvent) {
    auto& event = mEventInformation[iEvent];
    event.IsComplete = true;
    size_t totalPayloadSize = 0;
    for (size_t iCRU = 0; iCRU < event.CRUInfoArray.size(); ++iCRU) {
      const auto& cruInfo = event.CRUInfoArray[iCRU];
      if (!cruInfo.isPresent()) {
        if (mCRUSeen[iCRU]) {
          event.IsComplete = false;
          break;
        }

        continue;
      } else {
        totalPayloadSize += cruInfo.totalPayloadSize();
      }

      if (!cruInfo.isComplete()) {
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
  if (mFileIsScanned)
    return 0;

  // std::vector<PacketDescriptor> mPacketDescriptorMap;
  //const uint64_t RDH_HEADERWORD0 = 0x1ea04003;
  //const uint64_t RDH_HEADERWORD0 = 0x00004003;
  const uint64_t RDH_HEADERWORD0 = 0x00004004;

  std::ifstream file;
  file.open(mInputFileName, std::ifstream::binary);
  if (!file.good())
    throw std::runtime_error("Unable to open or access file " + mInputFileName);

  // get length of file in bytes
  file.seekg(0, file.end);
  mFileSize = file.tellg();
  file.seekg(0, file.beg);

  // the file is supposed to contain N x 8kB packets. So the number of packets
  // can be determined by the file-size. Ideally, this is not required but the
  // information is derived directly from the header size and payload size.
  // *** to be adapted to header info ***
  const uint32_t numPackets = mFileSize / (8 * 1024);

  // read in the RDH, then jump to the next RDH position
  RDH rdh;
  uint32_t currentPacket = 0;

  while (currentPacket < numPackets) {
    const uint32_t currentPos = file.tellg();

    // ===| read in the RawDataHeader at the current position |=================
    file >> rdh;

    const size_t packetSize = rdh.offsetToNext;
    const size_t offset = packetSize - rdh.headerSize;

    // ===| try to detect data type if not already set |========================
    //
    // for now we assume only HB scaling and triggered mode
    //
    // in case of triggered data we assume that that the for pageCnt == 1 we have
    //   triggerType == 0x10 in the firt packet
    //
    if (mManager) {
      if (mManager->mDataType == DataType::TryToDetect) {
        const uint64_t triggerTypeForTriggeredData = 0x10;
        const uint64_t triggerType = rdh.triggerType;
        const uint64_t pageCnt = rdh.pageCnt;

        if (pageCnt == 1) {
          if (triggerType == triggerTypeForTriggeredData) {
            mManager->mDataType = DataType::Triggered;
            O2INFO("Detected triggered data");
          } else {
            mManager->mDataType = DataType::HBScaling;
            O2INFO("Detected HB scaling");
          }
        }
      }
    }

    // ===| get relavant data information |=====================================
    //const auto heartbeatOrbit = rdh.heartbeatOrbit;
    const auto dataWrapperID = rdh.endPointID;
    const auto linkID = rdh.linkID;
    const auto globalLinkID = linkID + dataWrapperID * 12;
    //const auto blockLength = rdh.blockLength;
    const auto memorySize = rdh.memorySize;
    const auto payloadSize = rdh.memorySize - rdh.headerSize;

    // ===| check if cru should be forced |=====================================
    if (!mForceCRU) {
      mCRU = rdh.cruID;
    } else {
      //overwrite cru id in rdh for further processing
      rdh.cruID = mCRU;
    }

    // ===| find evnet info or create a new one |===============================
    RawReaderCRUEventSync::LinkInfo* linkInfo = nullptr;
    if (mManager) {
      // in case of triggered mode, we use the first heartbeat orbit as event identifier
      linkInfo = &mManager->mEventSync.getLinkInfo(rdh, mManager->getDataType());
      mManager->mEventSync.setCRUSeen(mCRU);
    }
    //std::cout << "block length: " << blockLength << '\n';

    // ===| set up packet descriptor map for GBT frames |=======================
    //
    // * check Header for Header ID
    // * create the packet descriptor
    // * set the mLinkPresent flag
    //
    if ((rdh.word0 & 0x0000FFFF) == RDH_HEADERWORD0) {
      // non 0 stop bit means data with payload
      if (rdh.stop == 0) {
        mPacketDescriptorMaps[globalLinkID].emplace_back(currentPos, mCRU, linkID, dataWrapperID, memorySize, packetSize);
        mLinkPresent[globalLinkID] = true;
        mPacketsPerLink[globalLinkID]++;
        if (linkInfo) {
          linkInfo->PacketPositions.emplace_back(mPacketsPerLink[globalLinkID] - 1);
          linkInfo->IsPresent = true;
          linkInfo->PayloadSize += payloadSize;
        }
      } else if (rdh.stop == 1) {
        // stop bit 1 means we hit the HB end frame without payload.
        // This marks the end of an "event" in HB scaling mode.
        if (linkInfo) {
          linkInfo->HBEndSeen = true;
        }
      } else {
        O2ERROR("Unknown stop code: %lu", rdh.stop);
      }
      //std::cout << dataWrapperID << "." << linkID << " (" << globalLinkID << ")\n";
    } else {
      O2ERROR("Found header word %x and required header word %x don't match", rdh.word0, RDH_HEADERWORD0);
    };

    // debug output
    if (CHECK_BIT(mDebugLevel, DebugLevel::RDHDump)) {
      //std::cout << "Packet " << std::setw(5) << currentPacket << " - Link " << int(linkID) << "\n";
      //std::cout << rdh;
      printHorizontal(rdh);
      if (rdh.stop) {
        std::cout << "\n";
        printHeader();
      }
    };
    // std::cout << "Position after read : " << std::dec << file.tellg() << std::endl;
    file.seekg(offset, file.cur);
    ++currentPacket;
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
  std::ifstream file;
  file.open(mInputFileName, std::ifstream::binary);
  if (!file.good())
    throw std::runtime_error("Unable to open or access file " + mInputFileName);

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
        if (CHECK_BIT(mDebugLevel, DebugLevel::SyncPositions)) {
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
  std::ifstream file;
  file.open(mInputFileName, std::ifstream::binary);
  if (!file.good())
    throw std::runtime_error("Unable to open or access file " + mInputFileName);

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
    if (CHECK_BIT(mDebugLevel, DebugLevel::GBTFrames)) {
      std::cout << gFrame;
    }

    gFrame.getAdcValues(rawData);
    gFrame.updateSyncCheck(CHECK_BIT(mDebugLevel, DebugLevel::SyncPositions));
    if (!(rawData.getNumTimebins() % 16) && (rawData.getNumTimebins() >= mNumTimeBins * 16)) {
      return 1;
    }
  };
  return 0;
}

int RawReaderCRU::processMemory(const std::vector<o2::byte>& data, ADCRawData& rawData)
{
  GBTFrame gFrame;

  // 16 bytes is the size of a GBT frame
  for (int iFrame = 0; iFrame < data.size() / 16; ++iFrame) {
    gFrame.setFrameNumber(iFrame);
    gFrame.setPacketNumber(iFrame / 508);

    // in readFromMemory a simple memcopy to the internal data structure is done
    // I tried using the memory block directly, storing in an internal data member
    // reinterpret_cast<const uint32_t*>(data.data() + iFrame * 16), so it could be accessed the
    // same way as the mData array.
    // however, this was ~5% slower in execution time. I suspect due to cache misses
    gFrame.readFromMemory(gsl::span<const o2::byte>(data.data() + iFrame * 16, 16));

    // extract the half words from the 4 32-bit words
    gFrame.getFrameHalfWords();

    // debug output
    if (CHECK_BIT(mDebugLevel, DebugLevel::GBTFrames)) {
      std::cout << gFrame;
    }

    gFrame.getAdcValues(rawData);
    gFrame.updateSyncCheck(CHECK_BIT(mDebugLevel, DebugLevel::SyncPositions));
    if (!(rawData.getNumTimebins() % 16) && (rawData.getNumTimebins() >= mNumTimeBins * 16)) {
      return 1;
    }
  };
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
      //printf("Fill: %d %d %d %d / %d %d %d\n", int(mCRU), int(cru.roc()), ichannel, sampaChannel, int(padPos.getROC()), int(padPos.getRow()), int(padPos.getPad()));
      mManager->mADCDataCallback(padPos, cru, gsl::span<const uint32_t>(dataVector.data() + ichannel, dataVector.size() - ichannel));
    }
  }
}

int RawReaderCRU::processDataFile()
{
  GBTFrame gFrame;
  //gFrame.setSyncPositions(mSyncPositions[mLink]);

  ADCRawData rawData;

  if (mVerbosity) {
    std::cout << "Processing data for link " << mLink << std::endl;
    std::cout << "Num packets : " << mPacketsPerLink[mLink] << std::endl;
  }

  // ===| mapping to be updated |===============================================
  //CRU cru; // assuming each decoder only hast once CRU
  const int link = mLink;

  const auto& linkInfoArray = mManager->mEventSync.getLinkInfoArrayForEvent(mEventNumber, mCRU);

  // loop over the packets for each link and process them
  //for (const auto& packet : mPacketDescriptorMaps[link]) {
  for (auto packetNumber : linkInfoArray[link].PacketPositions) {
    const auto& packet = mPacketDescriptorMaps[link][packetNumber];

    //cru = packet.getCRUID();
    // std::cout << "Packet : " << packetID << std::endl;
    gFrame.setPacketNumber(packetNumber);
    int retCode = processPacket(gFrame, packet.getPayloadOffset(), packet.getPayloadSize(), rawData);
    if (retCode) {
      break;
    }

    //if (rawData.getNumTimebins() >= mNumTimeBins * 16)
    //break;
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
        if (gFrame.mSyncFound(s) == false) {
          std::cout << "No sync found" << std::endl;
        }
        // debug output
        rawData.setOutputStream(s);
        rawData.setNumTimebins(mNumTimeBins);
        if (CHECK_BIT(mDebugLevel, DebugLevel::ADCValues)) {
          std::cout << rawData << std::endl;
        };
        // write the data to file
        //fileName = "ADC_" + std::to_string(mLink) + "_" + std::to_string(s);
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
  //if (mDataType == DataType::HBScaling) {
  //dataSize =
  //} else if (mDataType == DataType::Triggered) {
  //// in triggered mode 4000 GBT frames are read out
  //// 16 is the size of a GBT frame in byte
  //dataSize = 4000 * 16;
  //}

  std::vector<o2::byte> data(dataSize);
  collectGBTData(data);

  ADCRawData rawData;
  processMemory(data, rawData);

  // ===| fill ADC data to the output structure |===
  if (mFillADCdataMap) {
    fillADCdataMap(rawData);
  }
  if (mManager && mManager->mADCDataCallback) {
    runADCDataCallback(rawData);
  }
}

void RawReaderCRU::collectGBTData(std::vector<o2::byte>& data)
{
  const int link = mLink;

  const auto& mapper = Mapper::instance();

  const auto& linkInfoArray = mManager->mEventSync.getLinkInfoArrayForEvent(mEventNumber, mCRU);
  std::ifstream file;
  file.open(mInputFileName, std::ios::binary);
  if (!file.good())
    throw std::runtime_error("Unable to open or access file " + mInputFileName);

  size_t presentDataPosition = 0;

  // loop over the packets for each link and process them
  //for (const auto& packet : mPacketDescriptorMaps[link]) {
  for (auto packetNumber : linkInfoArray[link].PacketPositions) {
    const auto& packet = mPacketDescriptorMaps[link][packetNumber];

    const auto payloadStart = packet.getPayloadOffset();
    const auto payloadSize = std::min(size_t(packet.getPayloadSize()), data.size() - presentDataPosition);
    // jump to the start position of the packet
    file.seekg(payloadStart, std::ios::beg);

    // read data
    file.read(((char*)data.data()) + presentDataPosition, payloadSize);

    presentDataPosition += payloadSize;
  };
}

void RawReaderCRU::processLinks(const uint32_t linkMask)
{
  try {
    // read the input data from file, create the packet descriptor map
    // and the mLinkPresent map.
    scanFile();

    // check if selected event is valid
    if (mManager && mEventNumber >= mManager->mEventSync.getNumberOfEvents()) {
      O2ERROR("Selected event number %u is larger then the events in the file %lu", mEventNumber, mManager->mEventSync.getNumberOfEvents());
      return;
    }

    // loop over the MaxNumberOfLinks potential links in the data
    // only if data from the link is present and selected
    // for decoding it will be decoded.
    for (int lnk = 0; lnk < MaxNumberOfLinks; lnk++) {
      // all links have been selected
      if (linkMask == 0 && checkLinkPresent(lnk) == true) {
        // set the active link variable and process the data
        if (mDebugLevel) {
          fmt::print("Processing link {}\n", lnk);
        }
        setLink(lnk);
        //processDataFile();
        processDataMemory();
      } else if (((linkMask >> lnk) & 0x1) == 0x1 && checkLinkPresent(lnk) == true) {
        // set the active link variable and process the data
        if (mDebugLevel) {
          fmt::print("Processing link {}\n", lnk);
        }
        setLink(lnk);
        //processDataFile();
        processDataMemory();
      };
    };

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
  RawReaderCRU& rawReaderCRU = cruManager.createReader(inputFile, timeBins, 0, stream, debugLevel, verbosity, outputFilePrefix);
  cruManager.init();
  rawReaderCRU.mDumpTextFiles = true;
  rawReaderCRU.mFillADCdataMap = false;
  for (int ievent = 0; ievent < rawReaderCRU.getNumberOfEvents(); ++ievent) {
    fmt::print("=============| event {: 5d} |===============\n", ievent);
    rawReaderCRU.setEventNumber(ievent);
    rawReaderCRU.processLinks(linkMask);
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
  std::ifstream file;
  file.open(mInputFileName, std::ifstream::binary);
  if (!file.good()) {
    throw std::runtime_error("Unable to open or access file " + mInputFileName);
  }

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
//==============================================================================
//===| stream overloads for helper classes |====================================
//

void ADCRawData::streamTo(std::ostream& output) const
{
  const auto numTimeBins = std::min(getNumTimebins(), mNumTimeBins);
  for (int i = 0; i < numTimeBins * 16; i++) {
    if (i % 16 == 0)
      output << std::setw(4) << std::to_string(i / 16) << " : ";
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

//std::ostream& operator<<(std::ostream& output, const RawReaderCRU::GBTFrame& frame)
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

//friend std::ostream& operator<<(std::ostream& output, const PacketDescriptor& PD)
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
  output << "  version        : " << rdh.version << "\n";
  output << "  headerSize     : " << rdh.headerSize << "\n";
  output << "  blockLength    : " << rdh.blockLength << "\n";
  output << "  feeId          : " << rdh.feeId << "\n";
  output << "  priority       : " << rdh.priority << "\n";
  output << "  zero0          : " << rdh.zero0 << "\n";
  output << "\n";

  output << "word1            : 0x" << std::setfill('0') << std::setw(16) << std::hex << rdh.word1 << "\n"
         << std::dec;
  output << "  Offset to next : " << int(rdh.offsetToNext) << "\n";
  output << "  Memory size    : " << int(rdh.memorySize) << "\n";
  output << "  LinkID         : " << int(rdh.linkID) << "\n";
  output << "  Global LinkID  : " << int(rdh.linkID) + (((rdh.word1 >> 32) >> 28) * 12) << "\n";
  output << "  CRUid          : " << rdh.cruID << "\n";
  output << "  Packet Counter : " << rdh.packetCounter << "\n";
  output << "  DataWrapper-ID : " << (((rdh.word1 >> 32) >> 28) & 0x01) << "\n";
  output << "\n";

  output << "word2            : 0x" << std::setfill('0') << std::setw(16) << std::hex << rdh.word2 << "\n"
         << std::dec;
  output << "  triggerOrbit   : " << rdh.triggerOrbit << "\n";
  output << "  heartbeatOrbit : " << rdh.heartbeatOrbit << "\n";
  output << "\n";

  output << "word3            : 0x" << std::setfill('0') << std::setw(16) << std::hex << rdh.word3 << "\n"
         << std::dec;
  output << "\n";

  output << "word4            : 0x" << std::setfill('0') << std::setw(16) << std::hex << rdh.word4 << "\n"
         << std::dec;
  output << "  triggerBC      : " << rdh.triggerBC << "\n";
  output << "  zero41         : " << rdh.zero41 << "\n";
  output << "  heartbeatBC    : " << rdh.heartbeatBC << "\n";
  output << "  zero42         : " << rdh.zero42 << "\n";
  output << "  triggerType    : " << rdh.triggerType << "\n";
  output << "\n";

  output << "word5            : 0x" << std::setfill('0') << std::setw(16) << std::hex << rdh.word5 << "\n"
         << std::dec;
  output << "\n";

  output << "word6            : 0x" << std::setfill('0') << std::setw(16) << std::hex << rdh.word6 << "\n"
         << std::dec;
  output << "  detectorField  : " << rdh.detectorField << "\n";
  output << "  par            : " << rdh.par << "\n";
  output << "  stop           : " << rdh.stop << "\n";
  output << "  pageCnt        : " << rdh.pageCnt << "\n";
  output << "  zero6          : " << rdh.zero6 << "\n";
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
  const int globalLinkID = int(rdh.linkID) + (((rdh.word1 >> 32) >> 28) * 12);

  fmt::print("{:>5} {:>4} {:>4} {:>4} {:>6} {:>4} {:>3} {:>4} {:>10} {:>5} {:>1}\n",
             (uint64_t)rdh.packetCounter,
             (uint64_t)rdh.pageCnt,
             (uint64_t)rdh.triggerType,
             (uint64_t)rdh.feeId,
             (uint64_t)rdh.offsetToNext,
             (uint64_t)rdh.memorySize,
             (uint64_t)rdh.cruID,
             (uint64_t)globalLinkID,
             (uint64_t)rdh.heartbeatOrbit,
             (uint64_t)rdh.heartbeatBC,
             (uint64_t)rdh.stop);
}

std::istream& operator>>(std::istream& input, RDH& rdh)
{
  const int headerSize = sizeof(rdh);
  auto charPtr = reinterpret_cast<char*>(&rdh);
  input.read(charPtr, headerSize);
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
  mEventSync.analyse();

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
    reader.setVerbosity(verbosity);
    reader.setDebugLevel(debugLevel);
    O2INFO("Adding file: %s\n", file->GetName());
  }
}

void RawReaderCRUManager::copyEvents(const std::vector<uint32_t> eventNumbers, std::string_view outputDirectory, std::ios_base::openmode mode)
{
  // make sure events have been built
  init();
  for (auto& rawReader : mRawReadersCRU) {
    rawReader->copyEvents(eventNumbers, outputDirectory.data(), mode);
  }
}

void RawReaderCRUManager::copyEvents(const std::string_view inputFileNames, const std::vector<uint32_t> eventNumbers, std::string_view outputDirectory, std::ios_base::openmode mode)
{
  RawReaderCRUManager manager;
  manager.setupReaders(inputFileNames);
  manager.copyEvents(eventNumbers, outputDirectory, mode);
}
