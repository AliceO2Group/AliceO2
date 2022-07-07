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
#include <functional>
#include <gsl/span>
#include "DetectorsRaw/RDHUtils.h"

#include "TPCBase/CRU.h"
#include "Headers/RAWDataHeader.h"
#include "Headers/RDHAny.h"
#include "TPCBase/PadPos.h"
#include "TPCBase/PadROCPos.h"
#include "TPCBase/RDHUtils.h"
#include "TPCReconstruction/RawProcessingHelpers.h"

//#include "git_info.hpp"
namespace o2
{
namespace tpc
{
namespace rawreader
{

using RDHUtils = o2::raw::RDHUtils;
using RDH = o2::header::RDHAny;

/// debug level bits
enum DebugLevel : uint8_t {
  SyncPositions = 0, ///< dump sync positions
  GBTFrames = 1,     ///< dump all GBT frames
  ADCValues = 2,     ///< dump extracted ADC values
  RDHDump = 3,       ///< dump full RDH
  EventInfo = 5      ///< dump event synchronisation information
};

/// data type
enum class DataType : uint8_t {
  Continuous = 1, ///< continuous data taking
  HBScaling = 2,  ///< heart beat sclaing mode
  Triggered = 3,  ///< triggered data
};

/// file type
enum class ReaderType : uint8_t {
  FLP = 0, ///< single files from FLP as 8k pages with RAWDataHeader
  EPN = 1, ///< STF builder data merged on EPN
};

/// data type
enum class RAWDataType : uint8_t {
  GBT = 0,    ///< GBT encoded raw data
  LinkZS = 1, ///< Link based zero suppression
};

static constexpr int MaxNumberOfLinks = 24; ///< maximum number of links

// ===========================================================================
/// \class ADCRawData
/// \brief helper to store the ADC raw data
class ADCRawData
{
 public:
  using DataVector = std::vector<uint32_t>;

  /// default ctor. Resesrve 520 time bins per stream
  ADCRawData()
  {
    for (auto& data : mADCRaw) {
      data.reserve(520 * 16);
    }
  }

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
    if (numTB >= mADCRaw[mOutputStream].size()) {
      mNumTimeBins = mADCRaw[mOutputStream].size();
    } else {
      mNumTimeBins = numTB;
    }
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
  const DataVector& getDataVector(int stream) const { return mADCRaw[stream]; }

  /// overloading output stream operator to output ADCRawData
  friend std::ostream& operator<<(std::ostream& output, const ADCRawData& rawData);

  /// reset
  void reset()
  {
    mOutputStream = 0;
    mNumTimeBins = 0;
    for (auto& data : mADCRaw) {
      data.clear();
    }
  }

  bool hasData() const
  {
    for (auto& data : mADCRaw) {
      if (data.size()) {
        return true;
      }
    }
    return false;
  }

 private:
  uint32_t mOutputStream{0};           // variable to set the output stream for the << operator
  uint32_t mNumTimeBins{0};            // variable to set the number of timebins for the << operator
  std::array<DataVector, 5> mADCRaw{}; // array of 5 vectors to hold the raw ADC data for each stream
};                                     // class ADCRawData

//==============================================================================
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

  uint32_t getFrameNumber() const { return mFrameNum; }

  uint32_t getPacketNumber() const { return mPacketNum; }

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
};                          // class SyncPosition

using SyncArray = std::array<SyncPosition, 5>;

// ===========================================================================
/// \class GBTFrame
/// \brief helper to encapsulate a GBTFrame
class GBTFrame
{
 public:
  using adc_t = uint32_t;

  /// default constructor
  GBTFrame() = default;

  /// set the sync positions
  void setSyncPositions(const SyncArray& syncArray) { mSyncPos = syncArray; }

  /// set syncronisation found for stream
  /// \param stream stream
  bool syncFound(int stream) { return mSyncPos[stream].synched(); };

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

  /// read from memory
  void readFromMemory(gsl::span<const std::byte> data);

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

  /// set packet number
  void setFrameNumber(uint32_t frameNumber) { mFrameNum = frameNumber; }

 private:
  std::array<adc_t, 4> mData{};     ///< data to decode
  SyncArray mSyncPos{};             ///< sync position of the streams
  adc_t mFrameHalfWords[5][8]{};    ///< fixed size 2D array to contain the 4 half words + 4 previous half words for the 5 data streams of a link
  uint32_t mPrevHWpos{0};           ///< indicating the position of the previous HW in mFrameHalfWords, either 0 or 4
  uint32_t mSyncCheckRegister[5]{}; ///< array of registers to check the SYNC pattern for the 5 data streams
  uint32_t mFilePos{0};             ///< position in the raw data file (for back-tracing)
  uint32_t mFrameNum{0};            ///< current GBT frame number
  uint32_t mPacketNum{0};           ///< number of present 8k packet

  /// Bit-shift operations helper function operating on the 4 32-bit words of them
  /// GBT frame. Source bit "s" is shifted to target position "t"
  template <typename T>
  T bit(T s, T t) const;

  // /// get value of specific bit
  // static constexpr uint32_t getBit(uint32_t value, uint32_t bit)
  //{
  // return (value & (1 << bit)) >> bit;
  //}

  /// shift bit from one position to another
  template <typename T>
  static constexpr T shiftBit(T value, T from, T to)
  {
    return (value & (1 << from)) >> from << to;
  }
}; // class GBTFrame
class RawReaderCRUManager;

/// \class RawReaderCRUSync
/// \brief Synchronize the events over multiple CRUs
/// An event structure to keep track of packets inside the readers which belong to the
/// same event. An event is identified by all packets with the same heart beat counter
/// \author Jens Wiechula (Jens.Wiechula@ikf.uni-frankfurt.de)
class RawReaderCRUEventSync
{
 public:
  static constexpr size_t ExpectedNumberOfPacketsPerHBFrame{8}; ///< expected number of packets in one HB frame per link for HB scaling mode
  static constexpr size_t ExpectedPayloadSize{64000};           ///< expected payload size in case of triggered mode (4000 GBT Frames of 16byte)

  /// default ctor
  RawReaderCRUEventSync() { reset(); }

  // ---------------------------------------------------------------------------
  /// \struct LinkInfo
  /// \brief helper to store link information in an event
  struct LinkInfo {
    size_t PayloadSize{};                  ///< total payload size of the link in the present event
    bool HBEndSeen{false};                 ///< if HB end frame was seen
    bool IsPresent{false};                 ///< if the link is present in the current event
    bool WasSeen{false};                   ///< if the link was seen in any event
    std::vector<size_t> PacketPositions{}; ///< all packet positions of this link in an event

    /// number of packets in the link
    size_t getNumberOfPackets() const { return PacketPositions.size(); }

    /// if link data is complete
    bool isComplete() const { return HBEndSeen && (((PacketPositions.size() == ExpectedNumberOfPacketsPerHBFrame)) || (PayloadSize == ExpectedPayloadSize)); }

    /// if packet is the first one
    bool isFirstPacket() const { return (PacketPositions.size() == 0); }
  };
  // using LinkInfoArray_t = std::array<LinkInfo, 24>;
  using LinkInfoArray_t = std::vector<LinkInfo>;

  // ---------------------------------------------------------------------------
  /// \struct CRUInfo
  /// \brief summary information of on CRU
  struct CRUInfo {
    CRUInfo() : LinkInformation(MaxNumberOfLinks) {}

    bool isPresent() const
    {
      for (const auto& link : LinkInformation) {
        if (link.IsPresent) {
          return true;
        }
      }
      return false;
    }

    bool isComplete(RAWDataType rawDataType = RAWDataType::GBT) const
    {
      for (const auto& link : LinkInformation) {
        if (link.WasSeen && !link.IsPresent) {
          return false;
        }
        if (rawDataType == RAWDataType::GBT) {
          if (!link.isComplete()) {
            return false;
          }
        }
        if (rawDataType == RAWDataType::LinkZS) {
          if (link.IsPresent && !link.HBEndSeen) {
            return false;
          }
        }
      }
      return true;
    }

    size_t totalPayloadSize() const
    {
      size_t totalPayloadSize = 0;
      for (const auto& link : LinkInformation) {
        if (link.IsPresent) {
          totalPayloadSize += link.PayloadSize;
        }
      }
      return totalPayloadSize;
    }

    LinkInfoArray_t LinkInformation;
  };
  // using CRUInfoArray_t = std::array<CRUInfo, CRU::MaxCRU>;
  using CRUInfoArray_t = std::vector<CRUInfo>;

  // ---------------------------------------------------------------------------
  /// \struct EventInfo
  /// \brief helper to store event information
  struct EventInfo {
    EventInfo() : CRUInfoArray(CRU::MaxCRU) {}
    EventInfo(uint32_t heartbeatOrbit) : CRUInfoArray(CRU::MaxCRU) { HeartbeatOrbits.emplace_back(heartbeatOrbit); }
    EventInfo(const EventInfo&) = default;

    bool operator<(const EventInfo& other) const { return HeartbeatOrbits.back() < other.HeartbeatOrbits[0]; }

    /// check if heartbeatOrbit contributes to the event
    bool hasHearbeatOrbit(uint32_t heartbeatOrbit) const { return std::find(HeartbeatOrbits.begin(), HeartbeatOrbits.end(), heartbeatOrbit) != HeartbeatOrbits.end(); }

    uint32_t getFirstOrbit() const { return HeartbeatOrbits.size() ? *std::min_element(HeartbeatOrbits.begin(), HeartbeatOrbits.end()) : 0; }

    std::vector<uint32_t> HeartbeatOrbits{}; ///< vector of heartbeat orbits contributing to the event
    CRUInfoArray_t CRUInfoArray;             ///< Link information for each cru
    bool IsComplete{false};                  ///< if event is complete
  };

  // ---------------------------------------------------------------------------
  using EventInfoVector = std::vector<EventInfo>;

  /// get link information for a specific event and cru
  LinkInfo& getLinkInfo(const RDH& rdh, DataType dataType)
  {
    if (!mLastEvent) {
      const auto heartbeatOrbit = RDHUtils::getHeartBeatOrbit(rdh);
      createEvent(heartbeatOrbit, dataType);
    }

    const auto detField = o2::raw::RDHUtils::getDetectorField(rdh);
    const auto feeId = RDHUtils::getFEEID(rdh);
    const auto endPoint = rdh_utils::getEndPoint(feeId);
    auto link = rdh_utils::getLink(feeId);
    if (link == 21 && detField == 0x02) {
      link = 0;
    }
    const auto globalLink = link + endPoint * 12;
    const auto cru = rdh_utils::getCRU(feeId);

    return mLastEvent->CRUInfoArray[cru].LinkInformation[globalLink];
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
  EventInfo& createEvent(const uint32_t heartbeatOrbit, DataType dataType);

  /// analyse events and mark complete events
  void analyse(RAWDataType rawDataType = RAWDataType::GBT);

  /// all event information
  const EventInfoVector& getEventInfoVector() const { return mEventInformation; }

  /// information of specific event
  const EventInfo& getEventInfo(size_t eventNumber) const { return mEventInformation[eventNumber]; }

  /// write data to ostream
  void streamTo(std::ostream& output) const;

  /// set links that were seen for a CRU
  void setLinksSeen(const CRU cru, const std::bitset<MaxNumberOfLinks>& links);

  /// set a cru as seen
  void setCRUSeen(const CRU cru, const uint16_t reader = 0) { mCRUSeen[cru] = reader; }

  /// return CRU seen information
  const auto& getCRUSeen() const { return mCRUSeen; }

  /// get the reader associated to the CRU
  uint32_t getReaderNumber(uint32_t cru) { return mCRUSeen[cru]; }

  /// reset all information
  void reset()
  {
    mEventInformation.clear();
    mCRUSeen.fill(-1);
  }

  /// overloading output stream operator
  friend std::ostream& operator<<(std::ostream& output, const RawReaderCRUEventSync& eventSync)
  {
    eventSync.streamTo(output);
    return output;
  }

 private:
  EventInfoVector mEventInformation{};         ///< event information
  EventInfo* mLastEvent{nullptr};              ///< Last event that was created
  std::array<int16_t, CRU::MaxCRU> mCRUSeen{}; ///< if cru was seen, number is for which decoder

  ClassDefNV(RawReaderCRUEventSync, 0); // event synchronisation for raw reader instances
};                                      // class RawReaderCRUEventSync

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
  class PacketDescriptor;
  using PacketDescriptorMap = std::vector<PacketDescriptor>;
  using PacketDescriptorMapArray = std::array<PacketDescriptorMap, MaxNumberOfLinks>;

  /// constructor
  /// \param
  RawReaderCRU(const std::string_view inputFileName,
               uint32_t numTimeBins = 0,
               uint32_t link = 0,
               uint32_t stream = 0,
               uint32_t debugLevel = 0,
               uint32_t verbosity = 0,
               const std::string_view outputFilePrefix = "",
               uint32_t readerNumber = 0)
    : mDebugLevel(debugLevel),
      mVerbosity(verbosity),
      mNumTimeBins(numTimeBins),
      mLink(link),
      mStream(stream),
      mReaderNumber(readerNumber),
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

  /// set the reader number in the manager
  void setReaderNumber(uint32_t readerNumber) { mReaderNumber = readerNumber; }

  /// set filling of ADC data map
  void setFillADCdataMap(bool fill) { mFillADCdataMap = fill; }

  /// get filling of ADC data map
  bool getFillADCdataMap() const { return mFillADCdataMap; }

  /// get event number
  uint32_t getEventNumber() const { return mEventNumber; }

  /// get number of events
  size_t getNumberOfEvents() const;

  /// status bits of present links
  bool checkLinkPresent(uint32_t link) { return mLinkPresent[link]; }

  /// process all data for the selected link reading single 8k packet from file
  int processDataFile();

  /// process data in case of Link based zero suppression
  void processLinkZS();

  /// Collect data to memory and process data
  void processDataMemory();

  /// process single packet
  int processPacket(GBTFrame& gFrame, uint32_t startPos, uint32_t size, ADCRawData& rawData);

  /// Process data from memory for a single link
  /// The data must be collected before, merged over 8k packets
  int processMemory(const std::vector<std::byte>& data, ADCRawData& rawData);

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
  void setManager(RawReaderCRUManager* manager) { mManager = manager; }

  /// get manager
  RawReaderCRUManager* getManager() { return mManager; }

  /// copy single events to another file
  void copyEvents(const std::vector<uint32_t>& eventNumbers, std::string outputDirectory, std::ios_base::openmode mode = std::ios_base::openmode(0));

  /// write GBT data into separate files per link
  void writeGBTDataPerLink(std::string_view outputDirectory, int maxEvents = -1);

  /// run a data filling callback function
  void runADCDataCallback(const ADCRawData& rawData);

  /// set output file prefix
  void setOutputFilePrefix(std::string_view prefix) { mOutputFilePrefix = prefix; }

  /// output file prefix
  const std::string& getOutputFilePrefix() const { return mOutputFilePrefix; }

  /// get packet descriptor map array
  const PacketDescriptorMapArray& getPacketDescriptorMaps() const { return mPacketDescriptorMaps; }

  /// file handling
  std::ifstream& getFileHandle()
  {
    if (!mFileHandle.is_open()) {
      mFileHandle.open(mInputFileName, std::ios::binary);
      if (!mFileHandle.good()) {
        throw std::runtime_error("Unable to open or access file " + mInputFileName);
      }
    }

    return mFileHandle;
  }

  //===========================================================================
  //===| Nested helper classes |===============================================
  //

 public:
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
    using FEEIDType = rdh_utils::FEEIDType;

    PacketDescriptor(size_t headOff, uint32_t cru, uint32_t link, uint32_t endPoint,
                     uint16_t memorySize = 7840, uint16_t packetSize = 8192,
                     uint32_t heartbeatOrbit = 0) : mHeaderOffset(headOff),
                                                    mFEEID(rdh_utils::getFEEID(cru, endPoint, link)),
                                                    mMemorySize(memorySize),
                                                    mPacketSize(packetSize),
                                                    mHeartbeatOrbit(heartbeatOrbit)
    {
    }

    constexpr uint32_t getHeaderSize() const
    {
      return sizeof(RDH);
    }

    uint32_t getHeartBeatOrbit() const { return mHeartbeatOrbit; }
    size_t getHeaderOffset() const { return mHeaderOffset; }
    size_t getPayloadOffset() const { return mHeaderOffset + getHeaderSize(); }
    uint32_t getPayloadSize() const { return mMemorySize - getHeaderSize(); }
    uint32_t getPacketSize() const { return mPacketSize; }

    FEEIDType getFEEID() const { return mFEEID; }
    FEEIDType getCRUID() const { return rdh_utils::getCRU(mFEEID); }
    FEEIDType getLinkID() const { return rdh_utils::getLink(mFEEID); }
    FEEIDType getGlobalLinkID() const { return getLinkID() + getEndPoint() * 12; }
    FEEIDType getEndPoint() const { return rdh_utils::getEndPoint(mFEEID); }

    /// write data to ostream
    void streamTo(std::ostream& output) const;

    friend std::ostream& operator<<(std::ostream& output, const PacketDescriptor& packetDescriptor)
    {
      packetDescriptor.streamTo(output);
      return output;
    }

   private:
    size_t mHeaderOffset;     ///< header offset
    uint32_t mHeartbeatOrbit; ///< heartbeatOrbit
    uint16_t mMemorySize;     ///< payload size
    uint16_t mPacketSize;     ///< packet size
    FEEIDType mFEEID;         ///< FEEid as defined in TPCBase/RDHUtils.h
  };

  // ===========================================================================
  // ===| data members |========================================================
  //

 private:
  uint32_t mDebugLevel;                                     ///< debug level
  uint32_t mVerbosity;                                      ///< verbosity
  uint32_t mNumTimeBins;                                    ///< number of time bins to process
  uint32_t mLink;                                           ///< present link being processed
  uint32_t mStream;                                         ///< present stream being processed
  uint32_t mEventNumber = 0;                                ///< current event number to process
  uint32_t mReaderNumber = 0;                               ///< raw reader number in manager
  CRU mCRU;                                                 ///< CRU
  size_t mFileSize;                                         ///< size of the input file
  bool mDumpTextFiles = false;                              ///< dump debugging text files
  bool mFillADCdataMap = true;                              ///< fill the ADC data map
  bool mForceCRU = false;                                   ///< force CRU: overwrite value from RDH
  bool mFileIsScanned = false;                              ///< if file was already scanned
  std::array<uint32_t, MaxNumberOfLinks> mPacketsPerLink;   ///< array to keep track of the number of packets per link
  std::bitset<MaxNumberOfLinks> mLinkPresent;               ///< info if link is present in data; information retrieved from scanning the RDH headers
  PacketDescriptorMapArray mPacketDescriptorMaps;           ///< array to hold vectors thhe packet descriptors
  std::string mInputFileName;                               ///< input file name
  std::string mOutputFilePrefix;                            ///< input file name
  std::array<SyncArray, MaxNumberOfLinks> mSyncPositions{}; ///< sync positions for each link
  // not so nice but simplest way to store the ADC data
  std::map<PadPos, std::vector<uint16_t>> mADCdata; ///< decoded ADC data
  RawReaderCRUManager* mManager{nullptr};           ///< event synchronization information

  std::ifstream mFileHandle; ///< file handle for input file

  /// collect raw GBT data
  void collectGBTData(std::vector<std::byte>& data);

  /// fill adc data to output map
  void fillADCdataMap(const ADCRawData& rawData);

  ClassDefNV(RawReaderCRU, 0); // raw reader class

}; // class RawReaderCRU

// ===| inline definitions |====================================================
template <typename T>
inline T GBTFrame::bit(T s, T t) const
{
  const T dataWord = s >> 5;
  return shiftBit(mData[dataWord], s, t);
};

inline void GBTFrame::updateSyncCheck(SyncArray& syncArray)
{
  const auto offset = mPrevHWpos ^ 4;

  for (int s = 0; s < 5; s++) {
    for (int h = 0; h < 4; h++) {
      const auto hPos = h + offset; // set position of last filled HW
      // shift in a 1 if the halfword is 0x15
      if (mFrameHalfWords[s][hPos] == 0x15) {
        mSyncCheckRegister[s] = (mSyncCheckRegister[s] << 1) | 1;
        // shift in a 0 if the halfword is 0xA
      } else if (mFrameHalfWords[s][hPos] == 0xA) {
        mSyncCheckRegister[s] = (mSyncCheckRegister[s] << 1);
        // otherwise reset the register to 0
      } else {
        mSyncCheckRegister[s] = 0;
      }
      // std::cout << " SReg : " << s << " : " << std::hex << mSyncCheckRegister[s] << std::endl;
      // check the register content for the SYNC pattern
      // Patterh is : 1100.1100.1100.1100.1111.0000.1111.0000 = 0xCCCCF0F0
      // alternative Pattern for first corrupted SYNC word : 0100.1100.1100.1100.1111.0000.1111.0000 = 4CCC.F0F0
      if (mSyncCheckRegister[s] == 0xCCCCF0F0 or mSyncCheckRegister[s] == 0x4CCCF0F0 or mSyncCheckRegister[s] == 0x0CCCF0F0) {
        syncArray[s].setPos(mPacketNum, mFrameNum, mFilePos, h);
      };
    }
  };
}

inline void GBTFrame::updateSyncCheck(bool verbose)
{
  const auto offset = mPrevHWpos ^ 4;

  for (int s = 0; s < 5; s++) {
    for (int h = 0; h < 4; h++) {
      const auto hPos = h + offset; // set position of last filled HW
      // shift in a 1 if the halfword is 0x15
      if (mFrameHalfWords[s][hPos] == 0x15) {
        mSyncCheckRegister[s] = (mSyncCheckRegister[s] << 1) | 1;
        // shift in a 0 if the halfword is 0xA
      } else if (mFrameHalfWords[s][hPos] == 0xA) {
        mSyncCheckRegister[s] = (mSyncCheckRegister[s] << 1);
        // otherwise reset the register to 0
      } else {
        mSyncCheckRegister[s] = 0;
      }
      // std::cout << " SReg : " << s << " : " << std::hex << mSyncCheckRegister[s] << std::endl;
      // check the register content for the SYNC pattern
      // Patterh is : 1100.1100.1100.1100.1111.0000.1111.0000 = 0xCCCCF0F0
      // alternative Pattern for first corrupted SYNC word : 0100.1100.1100.1100.1111.0000.1111.0000 = 4CCC.F0F0
      if (mSyncCheckRegister[s] == 0xCCCCF0F0 or mSyncCheckRegister[s] == 0x4CCCF0F0 or mSyncCheckRegister[s] == 0x0CCCF0F0) {
        mSyncPos[s].setPos(mPacketNum, mFrameNum, mFilePos, h);
        if (verbose) {
          std::cout << s << " : " << mSyncPos[s];
        }
      }
    }
  }
}

/// extract the 4 5b halfwords for the 5 data streams from one GBT frame
/// the 4 5b halfwords of the previous frame are stored in the same structure
/// the position of the previous frame is indicated by mPrevHWpos
inline void GBTFrame::getFrameHalfWords()
{
  constexpr adc_t P[5][4] = {{19, 18, 17, 16}, {39, 38, 37, 36}, {63, 62, 61, 60}, {83, 82, 81, 80}, {107, 106, 105, 104}};
  // i = Stream, j = Halfword
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 4; j++) {
      mFrameHalfWords[i][j + mPrevHWpos] = bit(P[i][j], adc_t(4)) |
                                           bit(adc_t(P[i][j] - 4), adc_t(3)) |
                                           bit(adc_t(P[i][j] - 8), adc_t(2)) |
                                           bit(adc_t(P[i][j] - 12), adc_t(1)) |
                                           bit(adc_t(P[i][j] - 16), adc_t(0));
      // std::cout << " S : " << i << " H : " << j << " : " << std::hex << res << std::endl;
    };
  }
  mPrevHWpos ^= 4; // toggle position of previous HW position
}

/// decode ADC values
inline void GBTFrame::getAdcValues(ADCRawData& rawData)
{
  // loop over all the 5 data streams
  for (int s = 0; s < 5; s++) {
    if (!mSyncPos[s].synched()) {
      continue;
    }

    // assemble the 2 ADC words from 4 half-words. Which halfwords are
    // used depends on the position of the SYNC pattern.
    const uint32_t pos = mSyncPos[s].getHalfWordPosition();
    const uint32_t v0 = (mFrameHalfWords[s][(pos + 2 + mPrevHWpos) & 7] << 5) | mFrameHalfWords[s][(pos + 1 + mPrevHWpos) & 7];
    const uint32_t v1 = (mFrameHalfWords[s][(pos + 4 + mPrevHWpos) & 7] << 5) | mFrameHalfWords[s][(pos + 3 + mPrevHWpos) & 7];

    // add the two ADC values for the current processed stream
    rawData.add(s, v0, v1);
  };
  // std::cout << std::endl;
}

inline void GBTFrame::readFromMemory(gsl::span<const std::byte> data)
{
  assert(sizeof(mData) == data.size_bytes());
  memcpy(mData.data(), data.data(), data.size_bytes());
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
  using ADCDataCallback = std::function<Int_t(const PadROCPos&, const CRU&, const gsl::span<const uint32_t>)>;
  using LinkZSCallback = o2::tpc::raw_processing_helpers::ADCCallback;
  using EndReaderCallback = std::function<void()>;

  /// constructor
  RawReaderCRUManager() = default;

  /// create a new raw reader
  RawReaderCRU& createReader(const std::string_view inputFileName,
                             uint32_t numTimeBins = 0,
                             uint32_t link = 0,
                             uint32_t stream = 0,
                             uint32_t debugLevel = 0,
                             uint32_t verbosity = 0,
                             const std::string_view outputFilePrefix = "")
  // RawReaderCRU& createReader(std::string_view fileName, uint32_t numTimeBins)
  {
    mRawReadersCRU.emplace_back(std::make_unique<RawReaderCRU>(inputFileName, numTimeBins, 0, stream, debugLevel, verbosity, outputFilePrefix, mRawReadersCRU.size()));
    mRawReadersCRU.back()->setManager(this);
    return *mRawReadersCRU.back().get();
  }

  void setupReaders(const std::string_view inputFileNames,
                    uint32_t numTimeBins = 1000,
                    uint32_t debugLevel = 0,
                    uint32_t verbosity = 0,
                    const std::string_view outputFilePrefix = "");

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

  /// reset readers
  void resetReaders() { mRawReadersCRU.clear(); }

  /// reset event synchronisation info
  void resetEventSync() { mEventSync.reset(); }

  /// reset all
  void reset()
  {
    resetReaders();
    resetEventSync();
    mIsInitialized = false;
  }

  /// set event number to all sub-readers
  void setEventNumber(uint32_t eventNumber)
  {
    for (auto& reader : mRawReadersCRU) {
      reader->setEventNumber(eventNumber);
    }
  }

  /// get the event sync
  const RawReaderCRUEventSync& getEventSync() const { return mEventSync; }

  /// get number of all events
  size_t getNumberOfEvents() const { return mEventSync.getNumberOfEvents(); }

  /// get number of complete events
  size_t getNumberOfCompleteEvents() const { return mEventSync.getNumberOfCompleteEvents(); }

  /// check if event is complete
  bool isEventComplete(size_t eventNumber) const { return mEventSync.isEventComplete(eventNumber); }

  /// set debug level
  void setDebugLevel(uint32_t debugLevel) { mDebugLevel = debugLevel; }

  /// set data type
  void setDataType(DataType dataType, RAWDataType rawType)
  {
    mDataType = dataType;
    mRawDataType = rawType;
    mDetectDataType = false;
  }

  /// get data type
  DataType getDataType() const { return mDataType; }

  /// copy single events to another file
  void copyEvents(const std::vector<uint32_t> eventNumbers, std::string_view outputDirectory, std::ios_base::openmode mode = std::ios_base::openmode(0));

  /// copy single events from raw input files to another file
  static void copyEvents(const std::string_view inputFileNames, const std::vector<uint32_t> eventNumbers, std::string_view outputDirectory, std::ios_base::openmode mode = std::ios_base::openmode(0));

  /// copy single events from raw input files to another file
  void writeGBTDataPerLink(std::string_view outputDirectory, int maxEvents = -1);

  /// copy single events from raw input files to another file
  static void writeGBTDataPerLink(const std::string_view inputFileNames, std::string_view outputDirectory, int maxEvents = -1);

  /// set a callback function
  void setADCDataCallback(ADCDataCallback function) { mADCDataCallback = function; }

  /// set a callback function for decoded LinkZS data
  void setLinkZSCallback(LinkZSCallback function) { mLinkZSCallback = function; }

  /// get LinkZSCallback
  LinkZSCallback getLinkZSCallback() { return mLinkZSCallback; }

  /// process event calling mADCDataCallback to process values
  void processEvent(uint32_t eventNumber, EndReaderCallback endReader = nullptr);

 private:
  std::vector<std::unique_ptr<RawReaderCRU>> mRawReadersCRU{}; ///< cru type raw readers
  RawReaderCRUEventSync mEventSync{};                          ///< event synchronisation
  uint32_t mDebugLevel{0};                                     ///< debug level
  DataType mDataType{DataType::Continuous};                    ///< data type
  RAWDataType mRawDataType{RAWDataType::GBT};                  ///< raw data type
  bool mDetectDataType{true};                                  ///< try to detect data types
  bool mIsInitialized{false};                                  ///< if init was called already
  ADCDataCallback mADCDataCallback{nullptr};                   ///< callback function for filling the ADC data
  LinkZSCallback mLinkZSCallback{nullptr};                     ///< callback for decoded linkZS data

  friend class RawReaderCRU;

  ClassDefNV(RawReaderCRUManager, 0); // Manager class for CRU raw readers
};

} // namespace rawreader
} // namespace tpc
} // namespace o2
#endif
