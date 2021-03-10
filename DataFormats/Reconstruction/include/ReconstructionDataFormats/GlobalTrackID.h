// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file  GlobalTrackID.h
/// \brief Global index for barrel track: provides provenance (detectors combination), index in respective array and some number of bits
/// \author ruben.shahoyan@cern.ch

#ifndef O2_GLOBAL_TRACK_ID
#define O2_GLOBAL_TRACK_ID

#include "CommonDataFormat/AbstractRef.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include <iosfwd>
#include <string>
#include <array>
#include <string_view>
#include <bitset>

namespace o2
{
namespace dataformats
{

class GlobalTrackID : public AbstractRef<25, 5, 2>
{
 public:
  using DetID = o2::detectors::DetID;

  enum Source : uint8_t { // provenance of the
    ITS,                  // standalone detectors
    TPC,
    TRD,
    TOF,
    PHS, // FIXME Not sure PHS ... FDD should be kept here, at the moment
    CPV, // they are here for completeness
    EMC,
    HMP,
    MFT,
    MCH,
    MID,
    ZDC,
    FT0,
    FV0,
    FDD,
    ITSTPC, // 2-detector tracks
    TPCTOF,
    TPCTRD,
    ITSTPCTRD, // 3-detector tracks
    ITSTPCTOF,
    TPCTRDTOF,
    ITSTPCTRDTOF, // full barrel track
    //
    NSources
  };

  using AbstractRef<25, 5, 2>::AbstractRef;

  static const std::array<DetID::mask_t, NSources> SourceDetectorsMasks; // RS cannot be made constexpr since operator| is not constexpr
  static constexpr std::string_view NONE{"none"};                        ///< keywork for no sources
  static constexpr std::string_view ALL{"all"};                          ///< keywork for all sources
  typedef std::bitset<NSources> mask_t;

  static constexpr std::array<mask_t, NSources> sMasks = ///< detectot masks
    {math_utils::bit2Mask(ITS), math_utils::bit2Mask(TPC), math_utils::bit2Mask(TRD), math_utils::bit2Mask(TOF), math_utils::bit2Mask(PHS),
     math_utils::bit2Mask(CPV), math_utils::bit2Mask(EMC), math_utils::bit2Mask(HMP), math_utils::bit2Mask(MFT), math_utils::bit2Mask(MCH),
     math_utils::bit2Mask(MID), math_utils::bit2Mask(ZDC), math_utils::bit2Mask(FT0), math_utils::bit2Mask(FV0), math_utils::bit2Mask(FDD),
     math_utils::bit2Mask(ITSTPC), math_utils::bit2Mask(TPCTOF), math_utils::bit2Mask(TPCTRD), math_utils::bit2Mask(ITSTPCTRD),
     math_utils::bit2Mask(ITSTPCTOF), math_utils::bit2Mask(TPCTRDTOF), math_utils::bit2Mask(ITSTPCTRDTOF)};

  // methods for detector level manipulations
  static const auto getSourceDetectorsMask(int i) { return SourceDetectorsMasks[i]; }
  static bool includesDet(DetID id, GlobalTrackID::mask_t srcm);
  auto getSourceDetectorsMask() const { return getSourceDetectorsMask(getSource()); }
  bool includesDet(DetID id) const { return (getSourceDetectorsMask() & DetID::getMask(id)).any(); }

  // methods for source level manipulations
  static auto getSourceName(int s) { return DetID::getNames(getSourceDetectorsMask(s), '-'); }
  constexpr mask_t getSourceMask(int s) const { return sMasks[s]; }
  mask_t getSourceMask() const { return getSourceMask(getSource()); }
  auto getSourceName() const { return getSourceName(getSource()); }
  static mask_t getSourcesMask(const std::string_view srcList);
  static bool includesSource(int s, mask_t srcm) { return srcm[s]; }

  operator int() const { return int(getIndex()); }

  std::string asString() const;
  void print() const;

  ClassDefNV(GlobalTrackID, 2);
};

std::ostream& operator<<(std::ostream& os, const o2::dataformats::GlobalTrackID& v);

} // namespace dataformats
} // namespace o2

namespace std
{
// defining std::hash for GlobalTrackIndex to be used with std containers
template <>
struct hash<o2::dataformats::GlobalTrackID> {
 public:
  size_t operator()(const o2::dataformats::GlobalTrackID& id) const
  {
    return id.getRawWOFlags();
  }
};
} // namespace std

#endif
