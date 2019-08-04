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

#include "TPCReconstruction/RawReaderCRU.h"
#include "TPCBase/Mapper.h"

#define CHECK_BIT(var, pos) ((var) & (1 << (pos)))
using RDH = o2::header::RAWDataHeader;
using namespace o2::tpc;

std::ostream& operator<<(std::ostream& output, const RDH& rdh);
std::istream& operator>>(std::istream& input, RDH& rdh);

int RawReaderCRU::scanFile()
{
  if (mFileIsScanned)
    return 0;

  // std::vector<PacketDescriptor> mPacketDescriptorMap;
  //const uint64_t RDH_HEADERWORD0 = 0x1ea04003;
  const uint64_t RDH_HEADERWORD0 = 0x00004003;

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

    // read in the RawDataHeader at the current position
    file >> rdh;
    //std::cout << std::hex << rdh.word0 << " " << (rdh.word0 & 0xFFFFFFFF) << "\n";
    // get the link ID from header sub word 3
    // Bit 28 - 28 : DataWrapper-ID
    // Bit 27 - 16 : CRU-ID
    // Bit 15 - 08 : Packet Counter
    // Bit 07 - 00 : Link ID
    //auto getLinkID = [rdh]() {
    //const auto subword3 = rdh.word1 >> 32;
    //return (subword3 & 0xFF);
    //};

    //auto getCRUID = [rdh]() {
    //const auto subword3 = rdh.word1 >> 32;
    //return (subword3 >> 16) & 0x0FFF;
    //};

    auto getDataWrapperID = [rdh]() {
      auto subword3 = rdh.word1 >> 32;
      return (subword3 >> 28) & 0x01;
    };

    //const auto linkID = ((rdh.word1 >> 32) & 0xff);
    const auto dataWrapperID = getDataWrapperID();
    const auto linkID = rdh.linkID;
    const auto globalLinkID = linkID + dataWrapperID * 12;
    if (!mForceCRU) {
      mCRU = rdh.cruID;
    }

    const auto blockLength = rdh.blockLength;
    //std::cout << "block length: " << blockLength << '\n';

    // check Header for Header ID and create the packet descriptor and set the mLinkPresent flag
    if ((rdh.word0 & 0x0000FFFF) == RDH_HEADERWORD0) {
      mPacketDescriptorMaps[globalLinkID].emplace_back(currentPos, mCRU, linkID, dataWrapperID, blockLength);
      mLinkPresent[globalLinkID] = true;
      mPacketsPerLink[globalLinkID]++;
      //std::cout << dataWrapperID << "." << linkID << " (" << globalLinkID << ")\n";
    };

    // debug output
    if (CHECK_BIT(mDebugLevel, 0)) {
      std::cout << "Packet " << std::setw(5) << currentPacket << " - Link " << linkID << "\n";
      std::cout << rdh;
      std::cout << "\n";
    };
    // std::cout << "Position after read : " << std::dec << file.tellg() << std::endl;
    file.seekg(8128, file.cur);
    ++currentPacket;
  };

  // close the File
  file.close();

  if (mVerbosity) {
    // show the mLinkPresent map
    std::cout << "Links present" << std::endl;
    for (int i = 0; i < mNumberLinks; i++) {
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
      for (int i = 0; i < mNumberLinks; i++) {
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

  // loop over the mNumberLinks potential links in the data
  // only if data from the link is present and selected
  // for decoding it will be decoded.
  for (int link = 0; link < mNumberLinks; link++) {
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
      gFrame.setPacketID(packetID);

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
        if (CHECK_BIT(mDebugLevel, 0)) {
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
    // backup the halfword of the frame before calculating the
    // new halfwords. The previous half words might be needed
    // to decode the ADC values.
    gFrame.storePrevFrame();
    // extract the half words from the 4 32-bit words
    gFrame.getFrameHalfWords();

    // debug output
    if (CHECK_BIT(mDebugLevel, 1)) {
      std::cout << gFrame;
    }

    gFrame.getAdcValues(rawData);
    gFrame.updateSyncCheck(CHECK_BIT(mDebugLevel, 0));
    if (!(rawData.getNumTimebins() % 16) && (rawData.getNumTimebins() >= mNumTimeBins * 16))
      return 1;
  };
  return 0;
}

int RawReaderCRU::processData()
{
  GBTFrame gFrame;
  //gFrame.setSyncPositions(mSyncPositions[mLink]);

  uint32_t packetID = 0;
  ADCRawData rawData;

  if (mVerbosity) {
    std::cout << "Processing data for link " << mLink << std::endl;
    std::cout << "Num packets : " << mPacketsPerLink[mLink] << std::endl;
  }

  // ===| mapping to be updated |===============================================
  CRU cru; // assuming each decoder only hast once CRU
  const int link = mLink;

  const auto& mapper = Mapper::instance();

  // loop over the packets for each link and process them
  for (auto packet : mPacketDescriptorMaps[link]) {
    cru = packet.getCRUID();
    // std::cout << "Packet : " << packetID << std::endl;
    gFrame.setPacketID(packetID);
    int retCode = processPacket(gFrame, packet.getPayloadOffset(), packet.getPayloadSize(), rawData);
    if (retCode)
      break;
    packetID++;

    //if (rawData.getNumTimebins() >= mNumTimeBins * 16)
    //break;
  };

  // ===| fill ADC data to the output structure |===
  if (mFillADCdataMap) {
    const int fecLinkOffsetCRU = (mapper.getPartitionInfo(CRU(cru).partition()).getNumberOfFECs() + 1) / 2;
    const int fecInPartition = (mLink % 12) + (mLink > 11) * fecLinkOffsetCRU;
    const int regionIter = cru % 2;

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

  // std::cout << "Output Data" << std::endl;

  //// display the data
  if (mDumpTextFiles) {
    std::ofstream file;
    std::string fileName;
    for (int s = 0; s < 5; s++) {
      if (mStream == 0x0 or ((mStream >> s) & 0x1) == 0x1) {
        if (gFrame.mSyncFound(s) == false)
          std::cout << "No sync found" << std::endl;
        // debug output
        rawData.setOutputStream(s);
        rawData.setNumTimebins(mNumTimeBins);
        if (CHECK_BIT(mDebugLevel, 2)) {
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

void RawReaderCRU::processLinks(const uint32_t linkMask)
{
  try {
    // read the input data from file, create the packet descriptor map
    // and the mLinkPresent map.
    scanFile();

    // loop over the mNumberLinks potential links in the data
    // only if data from the link is present and selected
    // for decoding it will be decoded.
    for (int lnk = 0; lnk < mNumberLinks; lnk++) {
      // all links have been selected
      if (linkMask == 0 && checkLinkPresent(lnk) == true) {
        // set the active link variable and process the data
        setLink(lnk);
        processData();
      } else if (((linkMask >> lnk) & 0x1) == 0x1 && checkLinkPresent(lnk) == true) {
        // set the active link variable and process the data
        setLink(lnk);
        processData();
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
  RawReaderCRU rawReaderCRU(inputFile, timeBins, 0, stream, debugLevel, verbosity, outputFilePrefix);
  rawReaderCRU.mDumpTextFiles = true;
  rawReaderCRU.mFillADCdataMap = false;
  rawReaderCRU.processLinks(linkMask);
}

//==============================================================================
//===| stream overloads for helper classes |====================================
//

void RawReaderCRU::ADCRawData::streamTo(std::ostream& output) const
{
  const auto numTimeBins = std::min(getNumTimebins(), mNumTimeBins);
  for (int i = 0; i < numTimeBins * 16; i++) {
    if (i % 16 == 0)
      output << std::setw(4) << std::to_string(i / 16) << " : ";
    output << std::setw(4) << mADCRaw[(mOutputStream)][i];
    output << (((i + 1) % 16 == 0) ? "\n" : " ");
  };
};

void RawReaderCRU::GBTFrame::streamFrom(std::istream& input)
{
  mFilePos = input.tellg();
  mFrameNum++;
  for (int i = 0; i < 4; i++) {
    input.read(reinterpret_cast<char*>(&(mData[i])), sizeof(mData[i]));
  };
}

//std::ostream& operator<<(std::ostream& output, const RawReaderCRU::GBTFrame& frame)
void RawReaderCRU::GBTFrame::streamTo(std::ostream& output) const
{
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
    for (int j = 0; j < 4; j++)
      output << std::hex << std::setw(4) << mFrameHalfWords[i][j] << " ";
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

std::istream& operator>>(std::istream& input, RDH& rdh)
{
  const int headerSize = sizeof(rdh);
  const int wordSize = sizeof(rdh.word0);
  const int halfWordSize = wordSize / 2;
  const int numberOfHalfWords = headerSize / halfWordSize; // number of 32 bit words
  auto charPtr = reinterpret_cast<char*>(&rdh);

  for (int iHalfWord = 0; iHalfWord < numberOfHalfWords; ++iHalfWord) {
    input.read(charPtr + iHalfWord * halfWordSize, halfWordSize); // bits  0-31
  }

  return input;
}

//std::istream& operator>>(std::istream& input, RDH& rdh)
//{
//const int wordSize = sizeof(rdh.word0);
//const int halfWordSize = wordSize / 2;
//decltype(rdh.word0)* wordPtr = nullptr;

//for (int i = 0; i < 8; ++i) {
//switch (i) {
//case 0:
//wordPtr = &rdh.word0;
//break;
//case 1:
//wordPtr = &rdh.word1;
//break;
//case 2:
//wordPtr = &rdh.word2;
//break;
//case 3:
//wordPtr = &rdh.word3;
//break;
//case 4:
//wordPtr = &rdh.word4;
//break;
//case 5:
//wordPtr = &rdh.word5;
//break;
//case 6:
//wordPtr = &rdh.word6;
//break;
//case 7:
//wordPtr = &rdh.word7;
//break;
//}
//auto charPtr = reinterpret_cast<char*>(wordPtr);

//input.read(charPtr, halfWordSize);                // bits  0-31
//input.read(charPtr + halfWordSize, halfWordSize); // bits 32-63
//}

//return input;
//}
