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

  static const std::array<DetID::mask_t, NSources> DetectorMasks; // RS cannot be made constexpr since operator| is not constexpr
  using AbstractRef<25, 5, 2>::AbstractRef;

  static const auto getSourceMask(int i) { return DetectorMasks[i]; }
  static auto getSourceName(int i) { return DetID::getNames(getSourceMask(i)); }

  auto getSourceName() const { return getSourceName(getSource()); }
  auto getSourceMask() const { return getSourceMask(getSource()); }
  bool includesDet(DetID id) const { return (getSourceMask() & DetID::getMask(id)).any(); }

  operator int() const { return int(getIndex()); }

  std::string asString() const;
  void print() const;

  ClassDefNV(GlobalTrackID, 2);
};

std::ostream& operator<<(std::ostream& os, const o2::dataformats::GlobalTrackID& v);

} // namespace dataformats
} // namespace o2

#endif
