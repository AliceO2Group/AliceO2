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

///
/// \file    DataDecoder.h
/// \author  Andrea Ferrero
///
/// \brief Definition of the decoder for the MCH data
///

#ifndef O2_MCH_DATADECODER_H_
#define O2_MCH_DATADECODER_H_

#include <gsl/span>
#include <unordered_set>
#include <unordered_map>
#include <fstream>

#include "Headers/RDHAny.h"
#include "DataFormatsMCH/Digit.h"
#include "MCHBase/DecoderError.h"
#include "MCHBase/HeartBeatPacket.h"
#include "MCHRawDecoder/OrbitInfo.h"
#include "MCHRawDecoder/PageDecoder.h"

namespace o2
{
namespace mch
{
namespace raw
{

using RdhHandler = std::function<void(o2::header::RDHAny*)>;

// custom hash for OrbitInfo objects
struct OrbitInfoHash {
  std::size_t operator()(const OrbitInfo& info) const noexcept
  {
    return std::hash<uint64_t>{}(info.get());
  }
};

void dumpOrbits(const std::unordered_set<OrbitInfo, OrbitInfoHash>& mOrbits);

//_________________________________________________________________
//
// Data decoder
//_________________________________________________________________
class DataDecoder
{
 public:
  static constexpr int32_t tfTimeMax{0x7FFFFFFF};
  static constexpr int32_t tfTimeInvalid{-tfTimeMax};

  enum class TimeRecoMode : uint8_t {
    HBPackets = 0,
    BCReset = 1
  };

  /// Structure storing the raw SAMPA information
  struct SampaInfo {
    union {
      uint32_t id = 0;
      struct {
        uint32_t chip : 1;
        uint32_t ds : 6;
        uint32_t solar : 16;
        uint32_t unused : 9;
      };
    };

    union {
      // default value
      uint64_t time = 0x0000000000000000;
      struct {                       ///
        uint32_t sampaTime : 10;     /// bit 0 to 9: sampa time
        uint32_t bunchCrossing : 20; /// bit 10 to 29: bunch crossing counter
        uint32_t reserved : 2;       /// bit 30 to 31: reserved
        uint32_t orbit;              /// bit 32 to 63: orbit
      };                             ///
    };
    uint32_t getBXTime() const
    {
      return (bunchCrossing + (sampaTime * 4));
    }
    int32_t tfTime;
    bool timeValid() const
    {
      return (tfTime != tfTimeInvalid);
    }

    bool operator==(const SampaInfo&) const;
    bool operator<(const SampaInfo& rhs) const
    {
      if (id < rhs.id) {
        return true;
      } else if (time < rhs.time) {
        return true;
      }
      return false;
    }
  };

  /// Structure holding the value of the BC counter from the last decoded Heartbeat packet
  /// from a given SAMPA chip, as well as the value of the first orbit in the corresponding TimeFrame
  /// The structure also keeps track of the previous orbit/bc pair, in order to perform consistency checks
  struct TimeFrameStartRecord {
    TimeFrameStartRecord() = default;

    /// store the new orbit/bc pair, and copy the existing one in the "*Prev" data members
    bool update(uint32_t orbit, uint32_t bunchCrossing, bool verbose = false);
    bool check(int32_t orbit, uint32_t bc, int32_t orbitRef, uint32_t bcRef, bool verbose = false);

    int64_t mOrbit{-1};
    int64_t mBunchCrossing{-1};

    int64_t mOrbitPrev{-1};
    int64_t mBunchCrossingPrev{-1};

    bool mValid{true};
  };

  /// Structure used internally to store the information of the decoded digits.
  /// In addition to the standard Digit structure, it also keeps the raw SAMPA information
  struct RawDigit {
    o2::mch::Digit digit;
    SampaInfo info;
    auto getDetID() const { return digit.getDetID(); }
    auto getPadID() const { return digit.getPadID(); }
    uint32_t getADC() const { return digit.getADC(); }
    auto getTime() const { return info.tfTime; }
    bool timeValid() const { return info.timeValid(); }
    auto getOrbit() const { return info.orbit; }
    auto getBunchCrossing() const { return info.bunchCrossing; }
    auto getSampaTime() const { return info.sampaTime; }
    auto getBXTime() const { return info.getBXTime(); }

    bool operator==(const RawDigit&) const;
  };

  using RawDigitVector = std::vector<RawDigit>;

  DataDecoder(SampaChannelHandler channelHandler, RdhHandler rdhHandler,
              std::string mapCRUfile, std::string mapFECfile,
              bool ds2manu, bool verbose, bool useDummyElecMap, TimeRecoMode timeRecoMode = TimeRecoMode::HBPackets);

  void reset();

  /// Store the value of the first orbit in the TimeFrame to be processed
  /// Must be called before processing the TmeFrame buffer
  void setFirstOrbitInTF(uint32_t orbit);

  /** Decode one TimeFrame buffer and fill the vector of digits.
   *  @return true if decoding went ok, or false otherwise.
   *  if false is returned, the decoding of the (rest of the) TF should be
   *  abandonned simply.
   */
  bool decodeBuffer(gsl::span<const std::byte> buf);

  /// For a given SAMPA chip, update the information about the BC counter value at the beginning of the TimeFrame
  void updateTimeFrameStartRecord(uint64_t chipId, uint32_t mFirstOrbitInTF, uint32_t bcTF);
  /// Convert a Solar/Ds/Chip triplet into an unique chip index
  static uint64_t getChipId(uint32_t solar, uint32_t ds, uint32_t chip);
  /// Helper function for computing the digit time relative to the beginning of the TimeFrame
  static int32_t getDigitTimeHBPackets(uint32_t orbitTF, uint32_t bcTF, uint32_t orbitDigit, uint32_t bcDigit);
  /// Compute the time of all the digits that have been decoded in the current TimeFrame
  void computeDigitsTimeHBPackets();
  int32_t getDigitTimeBCRst(uint32_t orbitTF, uint32_t bcTF, uint32_t orbitDigit, uint32_t bcDigit);
  /// Compute the time of all the digits that have been decoded in the current TimeFrame
  void computeDigitsTimeBCRst();
  void computeDigitsTime();
  void checkDigitsTime();

  /// Get the vector of digits that have been decoded in the current TimeFrame
  const RawDigitVector& getDigits() const { return mDigits; }
  /// Get the list of orbits that have been found in the current TimeFrame for each CRU link
  const std::unordered_set<OrbitInfo, OrbitInfoHash>& getOrbits() const { return mOrbits; }
  /// Get the list of decoding errors that have been found in the current TimeFrame
  const std::vector<o2::mch::DecoderError>& getErrors() const { return mErrors; }
  /// Get the list of heart-beat packets that have been found in the current TimeFrame
  const std::vector<o2::mch::HeartBeatPacket>& getHBPackets() const { return mHBPackets; }
  /// Initialize the digits from an external vector. To be only used for unit tests.
  void setDigits(const RawDigitVector& digits) { mDigits = digits; }

  /// send all messages from our error map to the infologger
  void logErrorMap(int tfcount) const;

 private:
  void initElec2DetMapper(std::string filename);
  void initFee2SolarMapper(std::string filename);
  void init();
  void decodePage(gsl::span<const std::byte> page);
  void dumpDigits();
  bool getPadMapping(const DsElecId& dsElecId, DualSampaChannelId channel, int& deId, int& dsIddet, int& padId);
  bool addDigit(const DsElecId& dsElecId, DualSampaChannelId channel, const o2::mch::raw::SampaCluster& sc);
  bool getTimeFrameStartRecord(const RawDigit& digit, uint32_t& orbit, uint32_t& bc);
  bool getMergerChannelId(const DsElecId& dsElecId, DualSampaChannelId channel, uint32_t& chId, uint32_t& dsId);
  uint64_t getMergerChannelBitmask(DualSampaChannelId channel);
  void updateMergerRecord(uint32_t mergerChannelId, uint32_t mergerBoardId, uint64_t mergerChannelBitmask, uint32_t digitId);
  bool mergeDigits(uint32_t mergerChannelId, uint32_t mergerBoardId, uint64_t mergerChannelBitmask, o2::mch::raw::SampaCluster& sc);

  // structure that stores the index of the last decoded digit for a given readout channel,
  // as well as the time stamp of the last ADC sample of the digit
  struct MergerChannelRecord {
    MergerChannelRecord() = default;
    uint32_t digitId{0xFFFF};
    uint32_t bcEnd{0xFFFF};
  };

  static constexpr uint32_t sMaxSolarId = 200 * 8 - 1;
  static constexpr uint32_t sReadoutBoardsNum = (sMaxSolarId + 1) * 40;
  static constexpr uint32_t sReadoutChipsNum = sReadoutBoardsNum * 2;
  static constexpr uint32_t sReadoutChannelsNum = sReadoutChipsNum * 32;
  // table storing the last recorded TF time stamp in SAMPA BC counter units
  std::vector<TimeFrameStartRecord> mTimeFrameStartRecords;

  TimeRecoMode mTimeRecoMode{TimeRecoMode::HBPackets}; ///< method used to reconstruct the digits time

  // table storing the digits merging information for each readout channel in the MCH system
  std::vector<MergerChannelRecord> mMergerRecords; ///< merger records for all MCH readout channels
  std::vector<uint64_t> mMergerRecordsReady;       ///< merger status flags, one bit for one DS channel

  Elec2DetMapper mElec2Det{nullptr};       ///< front-end electronics mapping
  FeeLink2SolarMapper mFee2Solar{nullptr}; ///< CRU electronics mapping
  std::string mMapFECfile;                 ///< optional text file with custom front-end electronics mapping
  std::string mMapCRUfile;                 ///< optional text file with custom CRU mapping

  o2::mch::raw::PageDecoder mDecoder; ///< CRU page decoder

  RawDigitVector mDigits;                               ///< vector of decoded digits
  std::unordered_set<OrbitInfo, OrbitInfoHash> mOrbits; ///< list of orbits in the processed buffer
  std::vector<o2::mch::DecoderError> mErrors;           ///< list of decoding errors in the processed buffer
  std::vector<o2::mch::HeartBeatPacket> mHBPackets;     ///< list of heart-beat packets in the processed buffer

  uint32_t mOrbitsInTF{128};    ///< number of orbits in one time frame
  uint32_t mBcInOrbit;          ///< number of bunch crossings in one orbit
  uint32_t mFirstOrbitInTF;     ///< first orbit in the processed time-frame
  uint32_t mSampaTimeOffset{0}; ///< SAMPA BC counter value to be subtracted from the HBPacket BC at the TF start

  SampaChannelHandler mChannelHandler;                  ///< optional user function to be called for each decoded SAMPA hit
  std::function<void(o2::header::RDHAny*)> mRdhHandler; ///< optional user function to be called for each RDH

  bool mDebug{false};
  int mErrorCount{0};
  bool mDs2manu{false};
  uint32_t mOrbit{0};
  bool mUseDummyElecMap{false};
  std::map<std::string, uint64_t> mErrorMap; // counts for error messages
};

bool operator<(const DataDecoder::RawDigit& d1, const DataDecoder::RawDigit& d2);

std::ostream& operator<<(std::ostream& os, const DataDecoder::RawDigit& d);

std::string asString(const DataDecoder::RawDigit& d);

} // namespace raw
} // namespace mch
} // end namespace o2
#endif // O2_MCH_DATADECODER_H_
