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

#include <gsl/span>
#include <unordered_set>
#include <unordered_map>

#include "Headers/RDHAny.h"
#include "MCHBase/Digit.h"
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
    uint32_t getBXTime()
    {
      return (bunchCrossing + (sampaTime * 4));
    }
  };

  struct SampaTimeFrameStart {
    SampaTimeFrameStart(uint32_t orbit, uint32_t bunchCrossing) : mOrbit(orbit), mBunchCrossing(bunchCrossing) {}

    uint32_t mOrbit{0};
    int32_t mBunchCrossing{0};
  };

  using SampaTimeFrameStarts = std::unordered_map<uint32_t, std::optional<SampaTimeFrameStart>>;

  DataDecoder(SampaChannelHandler channelHandler, RdhHandler rdhHandler, std::string mapCRUfile, std::string mapFECfile, bool ds2manu, bool verbose);

  void reset();
  void decodeBuffer(gsl::span<const std::byte> buf);

  static int32_t digitsTimeDiff(uint32_t orbit1, uint32_t bc1, uint32_t orbit2, uint32_t bc2);
  static void computeDigitsTime_(std::vector<o2::mch::Digit>& digits, std::vector<SampaInfo>& sampaInfo, SampaTimeFrameStarts& sampaTimeFrameStarts, bool debug);
  void computeDigitsTime()
  {
    computeDigitsTime_(mOutputDigits, mSampaInfos, mSampaTimeFrameStarts, mDebug);
  }

  const std::vector<o2::mch::Digit>& getOutputDigits() const { return mOutputDigits; }
  const std::unordered_set<OrbitInfo, OrbitInfoHash>& getOrbits() const { return mOrbits; }

 private:
  void initElec2DetMapper(std::string filename);
  void initFee2SolarMapper(std::string filename);
  void init();
  void decodePage(gsl::span<const std::byte> page);
  void dumpDigits(bool bending);

  Elec2DetMapper mElec2Det{nullptr};       ///< front-end electronics mapping
  FeeLink2SolarMapper mFee2Solar{nullptr}; ///< CRU electronics mapping
  std::string mMapFECfile;                 ///< optional text file with custom front-end electronics mapping
  std::string mMapCRUfile;                 ///< optional text file with custom CRU mapping

  o2::mch::raw::PageDecoder mDecoder; ///< CRU page decoder

  std::vector<o2::mch::Digit> mOutputDigits; ///< vector of decoded digits
  std::vector<SampaInfo> mSampaInfos;        ///< vector of auxiliary SampaInfo objects

  std::unordered_set<OrbitInfo, OrbitInfoHash> mOrbits; ///< list of orbits in the processed buffer
  SampaTimeFrameStarts mSampaTimeFrameStarts;           ///< time stamps of the TimeFrames in the processed buffer

  SampaChannelHandler mChannelHandler;                  ///< optional user function to be called for each decoded SAMPA hit
  std::function<void(o2::header::RDHAny*)> mRdhHandler; ///< optional user function to be called for each RDH

  bool mDebug{false};
  bool mDs2manu{false};
  uint32_t mOrbit{0};
};

} // namespace raw
} // namespace mch
} // end namespace o2
