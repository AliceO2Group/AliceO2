// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_TPC_RAWREADERCRU_H_
#define O2_TPC_RAWREADERCRU_H_

/// \file RawReaderCRU.h
/// \author Jens Wiechula (Jens.Wiechula@ikf.uni-frankfurt.de)
/// \author Torsten Alt (Torsten.Alt@cern.ch)

// read a file into memory
#include <iostream> // std::cout
#include <fstream>  // std::std::ifstream
#include <cstdint>
#include <iomanip>
#include <vector>
#include <map>
#include <array>
#include <bitset>
#include <cmath>
#include <string_view>
#include <algorithm>

#include "TPCBase/CRU.h"
#include "Headers/RAWDataHeader.h"
#include "TPCBase/PadPos.h"

//#include "git_info.hpp"
namespace o2
{
namespace tpc
{
/// \class RawReaderCRUSync
/// \brief Synchronize the events over multiple CRUs
/// An event structure to keep track of packets inside the readers which belong to the
/// same event. An event is identified by all packets with the same heart beat counter
/// \author Jens Wiechula (Jens.Wiechula@ikf.uni-frankfurt.de)
class RawReaderCRUEventSync
{
 public:
  static constexpr size_t ExpectedNumberOfPacketsPerHBFrame{8}; ///< expected number of packets in one HB frame per link

  // ---------------------------------------------------------------------------
  /// \struct LinkInfo
  /// \brief helper to store link information in an event
  struct LinkInfo {
    size_t getNumberOfPackets() const { return PacketPositions.size(); }
    bool HBEndSeen{false};
    bool IsPresent{false};
    bool isComplete() const { return HBEndSeen && (PacketPositions.size() == ExpectedNumberOfPacketsPerHBFrame); }

    std::vector<size_t> PacketPositions{}; ///< all packet positions of this link in an event
  };
  //using LinkInfoArray_t = std::array<LinkInfo, 24>;
  using LinkInfoArray_t = std::vector<LinkInfo>;

  // ---------------------------------------------------------------------------
  /// \struct CRUInfo
  /// \brief summary information of on CRU
  struct CRUInfo {
    CRUInfo() : LinkInformation(24) {}

    bool isPresent() const
    {
      for (const auto& link : LinkInformation) {
        if (link.IsPresent) {
          return true;
        }
      }
      return false;
    }

    bool isComplete() const
    {
      for (const auto& link : LinkInformation) {
        if (link.IsPresent && !link.isComplete()) {
          return false;
        }
      }
      return true;
    }

    LinkInfoArray_t LinkInformation;
  };
  //using CRUInfoArray_t = std::array<CRUInfo, CRU::MaxCRU>;
  using CRUInfoArray_t = std::vector<CRUInfo>;

  // ---------------------------------------------------------------------------
  /// \struct EventInfo
  /// \brief helper to store event information
  struct EventInfo {
    EventInfo() : CRUInfoArray(CRU::MaxCRU) {}
    EventInfo(uint32_t heartbeatOrbit) : HeartbeatOrbit{heartbeatOrbit}, CRUInfoArray(CRU::MaxCRU) {}
    //EventInfo() {}
    //EventInfo(uint32_t heartbeatOrbit) : HeartbeatOrbit{heartbeatOrbit} {}
    EventInfo(const EventInfo&) = default;

    bool operator<(const EventInfo& other) const { return HeartbeatOrbit < other.HeartbeatOrbit; }

    uint32_t HeartbeatOrbit{0};

    CRUInfoArray_t CRUInfoArray; ///< Link information for each cru
    bool IsComplete{false};      ///< if event is complete
  };

  // ---------------------------------------------------------------------------
  using EventInfoVector = std::vector<EventInfo>;

  /// get link information for a specific event and cru
  LinkInfo& getLinkInfo(uint32_t heartbeatOrbit, int cru, uint8_t globalLinkID)
  {
    // check if event is already registered. If not create a new one.
    auto& event = createEvent(heartbeatOrbit);
    return event.CRUInfoArray[cru].LinkInformation[globalLinkID];
  }

  /// get array with all link informaiton for a specific event number and cru
  const LinkInfoArray_t& getLinkInfoArrayForEvent(size_t eventNumber, int cru) const { return mEventInformation[eventNumber].CRUInfoArray[cru].LinkInformation; }

  /// get number of all events
  size_t getNumberOfEvents() const { return mEventInformation.size(); }

  /// get number of complete events
  size_t getNumberOfCompleteEvents() const
  {
    size_t nComplete = 0;
    for (const auto& event : mEventInformation) {
      nComplete += event.IsComplete;
    }
    return nComplete;
  }

  /// get number of events for a certain CRU
  /// \todo extract correct number
  size_t getNumberOfEvents(CRU /*cru*/) const { return mEventInformation.size(); }

  /// check if eent is complete
  bool isEventComplete(size_t eventNumber) const
  {
    if (eventNumber >= mEventInformation.size()) {
      return false;
    }
    return mEventInformation[eventNumber].IsComplete;
  }

  /// sort events ascending in heartbeatOrbit number
  void sortEvents() { std::sort(mEventInformation.begin(), mEventInformation.end()); }

  /// create a new event or return the one with the given HB orbit
  EventInfo& createEvent(uint32_t heartbeatOrbit)
  {
    for (auto& ev : mEventInformation) {
      if (ev.HeartbeatOrbit == heartbeatOrbit) {
        return ev;
      }
    }
    return mEventInformation.emplace_back(heartbeatOrbit);
  }

  /// analyse events and mark complete events
  void analyse();

  const EventInfoVector& getEventInfoVector() const { return mEventInformation; }

  /// write data to ostream
  void streamTo(std::ostream& output) const;

  /// overloading output stream operator
  friend std::ostream& operator<<(std::ostream& output, const RawReaderCRUEventSync& eventSync)
  {
    eventSync.streamTo(output);
    return output;
  }

 private:
  EventInfoVector mEventInformation{}; ///< event information
  LinkInfo mLinkInfo{};

  ClassDefNV(RawReaderCRUEventSync, 0); // event synchronisation for raw reader instances
};

// =============================================================================
// =============================================================================
// =============================================================================
/// \class RawReaderCRU
/// \brief Reader for RAW TPC data
/// \author Jens Wiechula (Jens.Wiechula@ikf.uni-frankfurt.de)
/// \author Torsten Alt (Torsten.Alt@cern.ch)
class RawReaderCRU
{
 public:
  class ADCRawData;
  class GBTFrame;
  class PacketDescriptor;

  using RDH = o2::header::RAWDataHeader;
  static constexpr int MaxNumberOfLinks = 24; ///< maximum number of links

  /// constructor
  /// \param
  RawReaderCRU(const std::string_view inputFileName,
               uint32_t numTimeBins = 0,
               uint32_t link = 0,
               uint32_t stream = 0,
               uint32_t debugLevel = 0,
               uint32_t verbosity = 0,
               const std::string_view outputFilePrefix = "")
    : mDebugLevel(debugLevel),
      mVerbosity(verbosity),
      mNumTimeBins(numTimeBins),
      mLink(link),
      mStream(stream),
      mCRU(),
      mFileSize(-1),
      mPacketsPerLink(),
      mLinkPresent(),
      mPacketDescriptorMaps(),
      mInputFileName(inputFileName),
      mOutputFilePrefix(outputFilePrefix)
  {
    if (mOutputFilePrefix.empty()) {
      mOutputFilePrefix = mInputFileName.substr(0, mInputFileName.rfind('.'));
    }
  }

  /**
   * Exception class for decoder error
   */
  class Error : public std::runtime_error
  {
   public:
    Error(const std::string& what_arg) : std::runtime_error("Decoding error:" + what_arg) {}
  };

  /// scan file for RDH and packet positions
  ///
  /// scanFile scans the input file and checks the data for the RawDataHeader (RDH). It assumes
  /// that the data is made of 8kB packages for the moment, so it doesn't take the information
  /// in the header into account. It also checks the link ID in the RDH link ID field and fills
  /// the mLinkPresent variable to indicate which links are present in the data file.
  int scanFile();

  /// set the present (global) link to be processed
  /// \param link present link
  ///
  /// Global link means the CRU link from 0-23, where 12-23
  void setLink(uint32_t link) { mLink = link; }

  /// set verbosity level
  void setVerbosity(uint32_t verbosity = 1) { mVerbosity = verbosity; }

  /// set debug level
  void setDebugLevel(uint32_t debugLevel = 1) { mDebugLevel = debugLevel; }

  /// set event number
  void setEventNumber(uint32_t eventNumber = 0) { mEventNumber = eventNumber; }

  /// get event number
  uint32_t getEventNumber() const { return mEventNumber; }

  /// get number of events
  size_t getNumberOfEvents() const { return mEventSync ? mEventSync->getNumberOfEvents(mCRU) : 0; }

  /// status bits of present links
  bool checkLinkPresent(uint32_t link) { return mLinkPresent[link]; }

  /// process all data for the selected link
  int processData();

  /// process single packet
  int processPacket(GBTFrame& gFrame, uint32_t startPos, uint32_t size, ADCRawData& rawData);

  /// process links
  void processLinks(const uint32_t linkMask = 0);

  /// find sync positions for all links
  void findSyncPositions();

  /// check if sync positions were found for a specific link
  /// \param link link to check
  bool syncFoundForLink(const int link) const
  {
    const auto& arr = mSyncPositions[link];
    return arr[0].synched() && arr[1].synched() && arr[2].synched() && arr[3].synched() && arr[4].synched();
  }

  /// test function to process the data
  /// \param inputFile input file name
  /// \param linkMask mask of links
  static void
    processFile(const std::string_view inputFile, uint32_t timeBins = 0, uint32_t linkMask = 0, uint32_t stream = 0, uint32_t debugLevel = 0, uint32_t verbosity = 0, const std::string_view outputFilePrefix = "");

  /// return the ADC data map
  const std::map<PadPos, std::vector<uint16_t>>& getADCMap() const { return mADCdata; }

  /// clear the ADC data map
  void clearMap() { mADCdata.clear(); }

  /// return the CRU
  const CRU& getCRU() const { return mCRU; }

  /// for the CRU id
  void forceCRU(int cru)
  {
    mCRU = cru;
    mForceCRU = true;
  }

  /// set the event sync
  void setEventSync(RawReaderCRUEventSync* eventSync) { mEventSync = eventSync; }

  //===========================================================================
  //===| Nested helper classes |===============================================
  //

 public:
  /// \class SyncPosition
  /// \brief helper encoding of the sync position
  class SyncPosition
  {
   public:
    /// default constructor
    SyncPosition() = default;

    /// set sync position for data decoding
    /// \param pN packet number
    /// \param fN frame number
    /// \param fP file position
    /// \param hP half word position
    void setPos(uint32_t pN, uint32_t fN, uint32_t fP, uint32_t hP)
    {
      mSyncFound = true;
      mPacketNum = pN;
      mFrameNum = fN;
      mFilePos = fP;
      mHalfWordPos = hP;
    };

    /// return half word position
    uint32_t getHalfWordPosition() const { return mHalfWordPos; };

    /// return if sync was found
    bool synched() const { return mSyncFound; };

    /// overloading output stream operator to output the GBT Frame and the halfwords
    friend std::ostream& operator<<(std::ostream& output, const SyncPosition& sp)
    {
      output << "SYNC found at" << std::dec
             << "; Filepos : " << sp.mFilePos
             << "; Packet : " << sp.mPacketNum
             << "; Frame : " << sp.mFrameNum
             << "; Halfword : " << sp.mHalfWordPos
             << '\n';

      return output;
    };

   private:
    bool mSyncFound{false};   ///< if sync pattern was found
    uint32_t mPacketNum{0};   ///< packet number
    uint32_t mFrameNum{0};    ///< frame number
    uint32_t mFilePos{0};     ///< file position
    uint32_t mHalfWordPos{0}; ///< half word position
  };

  using SyncArray = std::array<SyncPosition, 5>;

  // ===========================================================================
  /// \class ADCRawData
  /// \brief helper to store the ADC raw data
  class ADCRawData
  {
   public:
    using DataVector = std::vector<uint32_t>;
    /// add a stream
    void add(int stream, uint32_t v0, uint32_t v1)
    {
      mADCRaw[stream].emplace_back(v0);
      mADCRaw[stream].emplace_back(v1);
    };

    /// select the stream for which data should be processed
    void setOutputStream(uint32_t outStream) { mOutputStream = outStream; };

    /// set the number of time bins for the selected stream. If the number of timebins
    /// set exceeds the maximum timebins in the stream, the maximum will be set as the
    /// limit.
    void setNumTimebins(uint32_t numTB)
    {
      if (numTB >= mADCRaw[mOutputStream].size())
        mNumTimeBins = mADCRaw[mOutputStream].size();
      else
        mNumTimeBins = numTB;
    };

    /// number of time bin for selected stream
    uint32_t getNumTimebins() const { return mADCRaw[mOutputStream].size(); };

    /// write data to ostream
    void streamTo(std::ostream& output) const;

    /// overloading output stream operator
    friend std::ostream& operator<<(std::ostream& output, const ADCRawData& rawData)
    {
      rawData.streamTo(output);
      return output;
    }

    /// get the data vector for a specific stream
    const DataVector& getDataVector(int stream) { return mADCRaw[stream]; }

   private:
    uint32_t mOutputStream{0};           // variable to set the output stream for the << operator
    uint32_t mNumTimeBins{0};            // variable to set the number of timebins for the << operator
    std::array<DataVector, 5> mADCRaw{}; // array of 5 vectors to hold the raw ADC data for each stream
  };
  friend std::ostream& operator<<(std::ostream& output, const RawReaderCRU::ADCRawData& rawData);

  // ===========================================================================
  /// \class GBTFrame
  /// \brief helper to encapsulate a GBTFrame
  class GBTFrame
  {
   public:
    /// default constructor
    GBTFrame() = default;

    /// set the sync positions
    void setSyncPositions(const SyncArray& syncArray) { mSyncPos = syncArray; }

    /// set syncronisation found for stream
    /// \param stream stream
    bool mSyncFound(int stream) { return mSyncPos[stream].synched(); };

    /// get the syncronisation array
    const SyncArray& getSyncArray() const { return mSyncPos; }

    /// update the sync check
    void updateSyncCheck(SyncArray& syncArray);

    /// update the sync check
    void updateSyncCheck(bool verbose = false);

    /// extract the 4 5b halfwords for the 5 data streams from one GBT frame
    void getFrameHalfWords();

    /// store the half words of the current frame in the previous frame data structure. Both
    /// frame information is needed to reconstruct the ADC stream since it can spread across
    /// 2 frames, depending on the position of the SYNC pattern.
    void storePrevFrame();

    /// decode ADC values
    void getAdcValues(ADCRawData& rawData);

    /// read from istream
    void streamFrom(std::istream& input);

    /// write data to ostream
    void streamTo(std::ostream& output) const;

    /// overloading input stream operator to read in the 4 32b words into the
    /// GBTFrame array mData
    friend std::istream& operator>>(std::istream& input, GBTFrame& frame)
    {
      frame.streamFrom(input);
      return input;
    }

    /// overloading output stream operator to output the GBT Frame and the halfwords
    friend std::ostream& operator<<(std::ostream& output, const GBTFrame& frame)
    {
      frame.streamTo(output);
      return output;
    }

    /// set packet number
    void setPacketNumber(uint32_t packetNumber) { mPacketNum = packetNumber; }

   private:
    std::array<uint32_t, 4> mData{};      ///< data to decode
    SyncArray mSyncPos{};                 ///< sync position of the streams
    uint32_t mFrameHalfWords[5][4]{};     ///< fixed size 2D array to contain the 4 halfwords for the 5 data streams of a link
    uint32_t mPrevFrameHalfWords[5][4]{}; ///< previous half word, required for decoding
    uint32_t mSyncCheckRegister[5]{};     ///< array of registers to check the SYNC pattern for the 5 data streams
    uint32_t mFilePos{0};                 ///< position in the raw data file (for back-tracing)
    uint32_t mFrameNum{0};                ///< current GBT frame number
    uint32_t mPacketNum{0};               ///< number of present 8k packet

    /// Bit-shift operations helper function operating on the 4 32-bit words of them
    /// GBT frame. Source bit "s" is shifted to target position "t"
    uint32_t bit(int s, int t) const;
  };

  // ===========================================================================
  /// \class PacketDescriptor
  /// \brief helper class to store packet positions inside the file
  ///
  /// The link id is the link in the CRU stream. The CRU has 2x12 links (A and B).
  /// The TPC uses a maximum of 2x10 links. The layout of the links is as follows,
  /// looking from the back side of the sector, so the FEE side.
  ///
  /// ROC Region Partition                          Links
  ///  O    9        4
  ///  O    8        4
  ///  --------------------------------------------------------------
  ///  O    7        3
  ///  O    6        3
  ///  --------------------------------------------------------------
  ///  O    5        2
  ///  O    4        2
  ///  ==============================================================
  ///  I    3        1
  ///  I    2        1
  ///  --------------------------------------------------------------
  ///  I    1        0 B6(18) | B5(17) | B4(16) | B3(15) | B2(14) | B1(13) | B0(12) | A7(7) | A6(6) | A5(5) | A4(4) | A3(3) | A2(2) | A1(1) | A0(0)
  ///  I    0        0 B6(18) | B5(17) | B4(16) | B3(15) | B2(14) | B1(13) | B0(12) | A7(7) | A6(6) | A5(5) | A4(4) | A3(3) | A2(2) | A1(1) | A0(0)
  ///
  ///
  ///
  ///
  ///
  class PacketDescriptor
  {
   public:
    PacketDescriptor(uint32_t headOff, uint32_t cruID, uint32_t linkID, uint32_t dataWrapperID, uint32_t payLoadSize = 7840) : mHeaderOffset(headOff),
                                                                                                                               mFEEID(cruID + (linkID << 9) + (dataWrapperID << 13)),
                                                                                                                               mPayloadSize(payLoadSize)
    {
    }

    constexpr uint32_t getHeaderSize() const
    {
      return sizeof(RDH);
    }
    uint32_t getHeaderOffset() const { return mHeaderOffset; }
    uint32_t getPayloadOffset() const { return mHeaderOffset + getHeaderSize(); }
    uint32_t getPayloadSize() const { return mPayloadSize; }

    uint16_t getCRUID() const { return mFEEID & 0x01FF; }
    uint16_t getLinkID() const { return (mFEEID >> 9) & 0x0F; }
    uint16_t getGlobalLinkID() const { return ((mFEEID >> 9) & 0x0F) + getDataWrapperID() * 12; }
    uint16_t getDataWrapperID() const { return (mFEEID >> 13) & 0x01; }

    /// write data to ostream
    void streamTo(std::ostream& output) const;

    friend std::ostream& operator<<(std::ostream& output, const PacketDescriptor& packetDescriptor)
    {
      packetDescriptor.streamTo(output);
      return output;
    }

   private:
    uint32_t mHeaderOffset; ///< header offset
    uint16_t mPayloadSize;  ///< payload size
    uint16_t mFEEID;        ///< link ID -- BIT 0-8: CRUid -- BIT 9-12: LinkID -- BIT 13: DataWrapperID -- BIT 14,15: unused
  };

  // ===========================================================================
  // ===| data members |========================================================
  //

 private:
  using PacketDescriptorMap = std::vector<PacketDescriptor>;
  uint32_t mDebugLevel;                                                    ///< debug level
  uint32_t mVerbosity;                                                     ///< verbosity
  uint32_t mNumTimeBins;                                                   ///< number of time bins to process
  uint32_t mLink;                                                          ///< present link being processed
  uint32_t mStream;                                                        ///< present stream being processed
  uint32_t mEventNumber = 0;                                               ///< current event number to process
  CRU mCRU;                                                                ///< CRU
  int mFileSize;                                                           ///< size of the input file
  bool mDumpTextFiles = false;                                             ///< dump debugging text files
  bool mFillADCdataMap = true;                                             ///< fill the ADC data map
  bool mForceCRU = false;                                                  ///< force CRU: overwrite value from RDH
  bool mFileIsScanned = false;                                             ///< if file was already scanned
  std::array<uint32_t, MaxNumberOfLinks> mPacketsPerLink;                  ///< array to keep track of the number of packets per link
  std::bitset<MaxNumberOfLinks> mLinkPresent;                              ///< info if link is present in data; information retrieved from scanning the RDH headers
  std::array<PacketDescriptorMap, MaxNumberOfLinks> mPacketDescriptorMaps; ///< array to hold vectors thhe packet descriptors
  std::string mInputFileName;                                              ///< input file name
  std::string mOutputFilePrefix;                                           ///< input file name
  std::array<SyncArray, MaxNumberOfLinks> mSyncPositions{};                ///< sync positions for each link
  // not so nice but simples way to store the ADC data
  std::map<PadPos, std::vector<uint16_t>> mADCdata; ///< decoded ADC data
  RawReaderCRUEventSync* mEventSync{nullptr};       ///< event synchronization information

  ClassDefNV(RawReaderCRU, 0); // raw reader class

}; // class RawReaderCRU

// ===| inline definitions |====================================================
inline uint32_t RawReaderCRU::GBTFrame::bit(int s, int t) const
{
  // std::cout << std::dec << s << " ";
  return (s < 32 ? ((mData[0] & (1 << s)) >> s) << t : (s < 64 ? ((mData[1] & (1 << (s - 32))) >> (s - 32)) << t : (s < 96 ? ((mData[2] & (1 << (s - 64))) >> (s - 64)) << t : (((mData[3] & (1 << (s - 96))) >> (s - 96)) << t))));
};

inline void RawReaderCRU::GBTFrame::updateSyncCheck(SyncArray& syncArray)
{
  for (int s = 0; s < 5; s++)
    for (int h = 0; h < 4; h++) {
      // shift in a 1 if the halfword is 0x15
      if (mFrameHalfWords[s][h] == 0x15)
        mSyncCheckRegister[s] = (mSyncCheckRegister[s] << 1) | 1;
      // shift in a 0 if the halfword is 0xA
      else if (mFrameHalfWords[s][h] == 0xA)
        mSyncCheckRegister[s] = (mSyncCheckRegister[s] << 1);
      // otherwise reset the register to 0
      else
        mSyncCheckRegister[s] = 0;
      // std::cout << " SReg : " << s << " : " << std::hex << mSyncCheckRegister[s] << std::endl;
      // check the register content for the SYNC pattern
      // Patterh is : 1100.1100.1100.1100.1111.0000.1111.0000 = 0xCCCCF0F0
      // alternative Pattern for first corrupted SYNC word : 0100.1100.1100.1100.1111.0000.1111.0000 = 4CCC.F0F0
      if (mSyncCheckRegister[s] == 0xCCCCF0F0 or mSyncCheckRegister[s] == 0x4CCCF0F0 or mSyncCheckRegister[s] == 0x0CCCF0F0) {
        syncArray[s].setPos(mPacketNum, mFrameNum, mFilePos, h);
      };
    };
}

inline void RawReaderCRU::GBTFrame::updateSyncCheck(bool verbose)
{
  for (int s = 0; s < 5; s++)
    for (int h = 0; h < 4; h++) {
      // shift in a 1 if the halfword is 0x15
      if (mFrameHalfWords[s][h] == 0x15)
        mSyncCheckRegister[s] = (mSyncCheckRegister[s] << 1) | 1;
      // shift in a 0 if the halfword is 0xA
      else if (mFrameHalfWords[s][h] == 0xA)
        mSyncCheckRegister[s] = (mSyncCheckRegister[s] << 1);
      // otherwise reset the register to 0
      else
        mSyncCheckRegister[s] = 0;
      // std::cout << " SReg : " << s << " : " << std::hex << mSyncCheckRegister[s] << std::endl;
      // check the register content for the SYNC pattern
      // Patterh is : 1100.1100.1100.1100.1111.0000.1111.0000 = 0xCCCCF0F0
      // alternative Pattern for first corrupted SYNC word : 0100.1100.1100.1100.1111.0000.1111.0000 = 4CCC.F0F0
      if (mSyncCheckRegister[s] == 0xCCCCF0F0 or mSyncCheckRegister[s] == 0x4CCCF0F0 or mSyncCheckRegister[s] == 0x0CCCF0F0) {
        mSyncPos[s].setPos(mPacketNum, mFrameNum, mFilePos, h);
        if (verbose) {
          std::cout << s << " : " << mSyncPos[s];
        }
      };
    };
}

/// extract the 4 5b halfwords for the 5 data streams from one GBT frame
inline void RawReaderCRU::GBTFrame::getFrameHalfWords()
{
  uint32_t P[5][4] = {{19, 18, 17, 16}, {39, 38, 37, 36}, {63, 62, 61, 60}, {83, 82, 81, 80}, {107, 106, 105, 104}};
  uint32_t res = 0;
  // i = Stream, j = Halfword
  for (int i = 0; i < 5; i++)
    for (int j = 0; j < 4; j++) {
      res = bit(P[i][j], 4) | bit(P[i][j] - 4, 3) | bit(P[i][j] - 8, 2) | bit(P[i][j] - 12, 1) | bit(P[i][j] - 16, 0);
      mFrameHalfWords[i][j] = res;
      // std::cout << " S : " << i << " H : " << j << " : " << std::hex << res << std::endl;
    };
}

/// store the half words of the current frame in the previous frame data structure. Both
/// frame information is needed to reconstruct the ADC stream since it can spread across
/// 2 frames, depending on the position of the SYNC pattern.
inline void RawReaderCRU::GBTFrame::storePrevFrame()
{
  for (int s = 0; s < 5; s++)
    for (int h = 0; h < 4; h++)
      mPrevFrameHalfWords[s][h] = mFrameHalfWords[s][h];
}

/// decode ADC values
inline void RawReaderCRU::GBTFrame::getAdcValues(ADCRawData& rawData)
{
  uint32_t pos;
  uint32_t v0;
  uint32_t v1;
  // loop over all the 5 data streams
  for (int s = 0; s < 5; s++) {
    // assemble the 2 ADC words from 4 half-words. Which halfwords are
    // used depends on the position of the SYNC pattern.
    pos = mSyncPos[s].getHalfWordPosition();
    if (pos == 0) {
      v0 = (mPrevFrameHalfWords[s][2] << 5) | mPrevFrameHalfWords[s][1];
      v1 = (mFrameHalfWords[s][0] << 5) | mPrevFrameHalfWords[s][3];
    } else if (pos == 1) {
      v0 = (mPrevFrameHalfWords[s][3] << 5) | mPrevFrameHalfWords[s][2];
      v1 = (mFrameHalfWords[s][1] << 5) | mFrameHalfWords[s][0];
    } else if (pos == 2) {
      v0 = (mFrameHalfWords[s][0] << 5) | mPrevFrameHalfWords[s][3];
      v1 = (mFrameHalfWords[s][2] << 5) | mFrameHalfWords[s][1];
    } else if (pos == 3) {
      v0 = (mFrameHalfWords[s][1] << 5) | mFrameHalfWords[s][0];
      v1 = (mFrameHalfWords[s][3] << 5) | mFrameHalfWords[s][2];
    };
    // add the two ADC values for the current processed stream
    if (mSyncPos[s].synched() == true) {
      // std::cout << "Adding ADC" << std::endl;
      rawData.add(s, v0, v1);
    };
  };
  // std::cout << std::endl;
}

// =============================================================================
// =============================================================================
// =============================================================================
/// \class RawReaderArrayCRU
/// \brief Array of raw readers RawReaderCRU
/// This class holds an array of raw readers of type RawReaderCRU.
/// It creates an event structure to keep track of packets inside the readers which belong to the
/// same event. An event is identified by all packets with the same heart beat counter
/// \author Jens Wiechula (Jens.Wiechula@ikf.uni-frankfurt.de)
class RawReaderCRUManager
{
 public:
  /// constructor
  RawReaderCRUManager() = default;

  /// create a new raw reader
  RawReaderCRU& createReader(std::string_view fileName, uint32_t numTimeBins)
  {
    mRawReadersCRU.emplace_back(std::make_unique<RawReaderCRU>(fileName, numTimeBins));
    return *mRawReadersCRU.back().get();
  }

  /// initialize all readers
  void init();

  /// return vector of readers
  auto& getReaders() { return mRawReadersCRU; }

  /// return vector of readers
  const auto& getReaders() const { return mRawReadersCRU; }

  /// return number of configured raw readers
  auto getNumberOfReaders() const { return mRawReadersCRU.size(); }

  /// return last reader
  RawReaderCRU& getLastReader() { return *mRawReadersCRU.back().get(); }

  /// return last reader
  const RawReaderCRU& getLastReader() const { return *mRawReadersCRU.back().get(); }

  /// set event number to all sub-readers
  void setEventNumber(uint32_t eventNumber)
  {
    for (auto& reader : mRawReadersCRU) {
      reader->setEventNumber(eventNumber);
    }
  }

  /// get number of all events
  size_t getNumberOfEvents() const { return mEventSync.getNumberOfEvents(); }

  /// get number of complete events
  size_t getNumberOfCompleteEvents() const { return mEventSync.getNumberOfCompleteEvents(); }

  /// check if event is complete
  bool isEventComplete(size_t eventNumber) const { return mEventSync.isEventComplete(eventNumber); }

  /// set debug level
  void setDebugLevel(uint32_t debugLevel) { mDebugLevel = debugLevel; }

 private:
  std::vector<std::unique_ptr<RawReaderCRU>> mRawReadersCRU{}; ///< cru type raw readers
  RawReaderCRUEventSync mEventSync{};                          ///< event synchronisation
  uint32_t mDebugLevel{0};
  bool mIsInitialized{false}; ///< if init was called already

  ClassDefNV(RawReaderCRUManager, 0); // Manager class for CRU raw readers
};

} // namespace tpc
} // namespace o2
#endif
