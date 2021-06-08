// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

#include "Headers/RDHAny.h"
#include "DataFormatsMCH/Digit.h"
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
  };

  struct SampaTimeFrameStart {
    SampaTimeFrameStart(uint32_t orbit, uint32_t bunchCrossing) : mOrbit(orbit), mBunchCrossing(bunchCrossing) {}

    uint32_t mOrbit{0};
    int32_t mBunchCrossing{0};
  };

  using SampaTimeFrameStarts = std::unordered_map<uint32_t, std::optional<SampaTimeFrameStart>>;

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

  DataDecoder(SampaChannelHandler channelHandler, RdhHandler rdhHandler, std::string mapCRUfile, std::string mapFECfile, bool ds2manu, bool verbose);

  void reset();
  void setFirstTForbit(uint32_t orbit) { mFirstTForbit = orbit; }
  void decodeBuffer(gsl::span<const std::byte> buf);

  static int32_t digitsTimeDiff(uint32_t orbit1, uint32_t bc1, uint32_t orbit2, uint32_t bc2);
  static void computeDigitsTime_(RawDigitVector& digits, SampaTimeFrameStarts& sampaTimeFrameStarts, bool debug);
  void computeDigitsTime()
  {
    computeDigitsTime_(mDigits, mSampaTimeFrameStarts, mDebug);
  }

  const RawDigitVector& getDigits() const { return mDigits; }
  const std::unordered_set<OrbitInfo, OrbitInfoHash>& getOrbits() const { return mOrbits; }

 private:
  void initElec2DetMapper(std::string filename);
  void initFee2SolarMapper(std::string filename);
  void init();
  void decodePage(gsl::span<const std::byte> page);
  void dumpDigits();

  Elec2DetMapper mElec2Det{nullptr};       ///< front-end electronics mapping
  FeeLink2SolarMapper mFee2Solar{nullptr}; ///< CRU electronics mapping
  std::string mMapFECfile;                 ///< optional text file with custom front-end electronics mapping
  std::string mMapCRUfile;                 ///< optional text file with custom CRU mapping

  o2::mch::raw::PageDecoder mDecoder; ///< CRU page decoder

  RawDigitVector mDigits; ///< vector of decoded digits

  std::unordered_set<OrbitInfo, OrbitInfoHash> mOrbits; ///< list of orbits in the processed buffer
  SampaTimeFrameStarts mSampaTimeFrameStarts;           ///< time stamps of the TimeFrames in the processed buffer

  SampaChannelHandler mChannelHandler;                  ///< optional user function to be called for each decoded SAMPA hit
  std::function<void(o2::header::RDHAny*)> mRdhHandler; ///< optional user function to be called for each RDH

  bool mDebug{false};
  bool mDs2manu{false};
  uint32_t mFirstTForbit{0};
  uint32_t mOrbit{0};
  uint32_t mOrbitId{0};
};

bool operator<(const DataDecoder::RawDigit& d1, const DataDecoder::RawDigit& d2);

std::ostream& operator<<(std::ostream& os, const DataDecoder::RawDigit& d);

} // namespace raw
} // namespace mch
} // end namespace o2
#endif //O2_MCH_DATADECODER_H_
