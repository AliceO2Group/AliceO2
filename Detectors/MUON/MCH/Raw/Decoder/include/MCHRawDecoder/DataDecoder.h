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

//_________________________________________________________________
//
// Data decoder
//_________________________________________________________________
class DataDecoder
{
 public:
  DataDecoder(SampaChannelHandler channelHandler, RdhHandler rdhHandler, std::string mapCRUfile, std::string mapFECfile, bool ds2manu, bool verbose);

  void reset();
  void decodeBuffer(gsl::span<const std::byte> page);

  const std::vector<o2::mch::Digit>& getOutputDigits() const { return mOutputDigits; }
  const std::unordered_set<OrbitInfo, OrbitInfoHash>& getOrbits() const { return mOrbits; }

 private:
  void initElec2DetMapper(std::string filename);
  void initFee2SolarMapper(std::string filename);
  void init();

  Elec2DetMapper mElec2Det{nullptr};
  FeeLink2SolarMapper mFee2Solar{nullptr};
  o2::mch::raw::PageDecoder mDecoder;
  size_t mNrdhs{0};
  std::vector<o2::mch::Digit> mOutputDigits;
  std::unordered_set<OrbitInfo, OrbitInfoHash> mOrbits; ///< list of orbits in the processed buffer

  SampaChannelHandler mChannelHandler;
  std::function<void(o2::header::RDHAny*)> mRdhHandler;

  std::string mMapCRUfile;
  std::string mMapFECfile;

  bool mDebug{false};
  bool mDs2manu{false};
};

} // namespace raw
} // namespace mch
} // end namespace o2
