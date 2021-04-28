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

#include "GPUCommonBitSet.h"
#include "CommonDataFormat/AbstractRef.h"
#include "DetectorsCommonDataFormats/DetID.h"
#ifndef GPUCA_GPUCODE
#include <iosfwd>
#include <string>
#include <array>
#include <string_view>
#include <bitset>
#endif // GPUCA_GPUCODE_DEVICE

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
  typedef o2::gpu::gpustd::bitset<NSources> mask_t;

#ifndef GPUCA_GPUCODE
  static constexpr std::string_view NONE{"none"};                        ///< keywork for no sources
  static constexpr std::string_view ALL{"all"};                          ///< keywork for all sources
#endif

  // methods for detector level manipulations
  GPUd() static constexpr DetID::mask_t getSourceDetectorsMask(int i);
  GPUd() static bool includesDet(DetID id, GlobalTrackID::mask_t srcm);
  GPUdi() auto getSourceDetectorsMask() const { return getSourceDetectorsMask(getSource()); }
  GPUdi() bool includesDet(DetID id) const { return (getSourceDetectorsMask() & DetID::getMask(id)).any(); }

  // methods for source level manipulations
#ifndef GPUCA_GPUCODE
  static auto getSourceName(int s) { return DetID::getNames(getSourceDetectorsMask(s), '-'); }
  static mask_t getSourcesMask(const std::string_view srcList);
  static std::string getSourcesNames(mask_t srcm);
  auto getSourceName() const { return getSourceName(getSource()); }
#endif // GPUCA_GPUCODE
  GPUd() static constexpr mask_t getSourceMask(int s);
  GPUdi() mask_t getSourceMask() const { return getSourceMask(getSource()); }
  GPUdi() static bool includesSource(int s, mask_t srcm) { return srcm[s]; }
  GPUdi() operator int() const { return int(getIndex()); }

#ifndef GPUCA_GPUCODE
  std::string asString() const;
  void print() const;
#endif // GPUCA_GPUCODE

  ClassDefNV(GlobalTrackID, 3);
};

#ifndef GPUCA_GPUCODE
std::ostream& operator<<(std::ostream& os, const o2::dataformats::GlobalTrackID& v);
#endif // GPUCA_GPUCODE

namespace globaltrackid_internal
{
// static constexpr array class members not possible on the GPU, thus we use this trick.
using DetID = o2::detectors::DetID;
GPUconstexpr() DetID::mask_t SourceDetectorsMasks[GlobalTrackID::NSources] = {
  DetID::getMask(DetID::ITS),
  DetID::getMask(DetID::TPC),
  DetID::getMask(DetID::TRD),
  DetID::getMask(DetID::TOF),
  DetID::getMask(DetID::PHS),
  DetID::getMask(DetID::CPV),
  DetID::getMask(DetID::EMC),
  DetID::getMask(DetID::HMP),
  DetID::getMask(DetID::MFT),
  DetID::getMask(DetID::MCH),
  DetID::getMask(DetID::MID),
  DetID::getMask(DetID::ZDC),
  DetID::getMask(DetID::FT0),
  DetID::getMask(DetID::FV0),
  DetID::getMask(DetID::FDD),
  //
  DetID::getMask(DetID::ITS) | DetID::getMask(DetID::TPC),
  DetID::getMask(DetID::TPC) | DetID::getMask(DetID::TOF),
  DetID::getMask(DetID::TPC) | DetID::getMask(DetID::TRD),
  DetID::getMask(DetID::ITS) | DetID::getMask(DetID::TPC) | DetID::getMask(DetID::TRD),
  DetID::getMask(DetID::ITS) | DetID::getMask(DetID::TPC) | DetID::getMask(DetID::TOF),
  DetID::getMask(DetID::TPC) | DetID::getMask(DetID::TRD) | DetID::getMask(DetID::TOF),
  DetID::getMask(DetID::ITS) | DetID::getMask(DetID::TPC) | DetID::getMask(DetID::TRD) | DetID::getMask(DetID::TOF)};

GPUconstexpr() GlobalTrackID::mask_t sMasks[GlobalTrackID::NSources] = ///< detectot masks
  {math_utils::bit2Mask(GlobalTrackID::ITS), math_utils::bit2Mask(GlobalTrackID::TPC), math_utils::bit2Mask(GlobalTrackID::TRD), math_utils::bit2Mask(GlobalTrackID::TOF), math_utils::bit2Mask(GlobalTrackID::PHS),
   math_utils::bit2Mask(GlobalTrackID::CPV), math_utils::bit2Mask(GlobalTrackID::EMC), math_utils::bit2Mask(GlobalTrackID::HMP), math_utils::bit2Mask(GlobalTrackID::MFT), math_utils::bit2Mask(GlobalTrackID::MCH),
   math_utils::bit2Mask(GlobalTrackID::MID), math_utils::bit2Mask(GlobalTrackID::ZDC), math_utils::bit2Mask(GlobalTrackID::FT0), math_utils::bit2Mask(GlobalTrackID::FV0), math_utils::bit2Mask(GlobalTrackID::FDD),
   math_utils::bit2Mask(GlobalTrackID::ITSTPC), math_utils::bit2Mask(GlobalTrackID::TPCTOF), math_utils::bit2Mask(GlobalTrackID::TPCTRD), math_utils::bit2Mask(GlobalTrackID::ITSTPCTRD),
   math_utils::bit2Mask(GlobalTrackID::ITSTPCTOF), math_utils::bit2Mask(GlobalTrackID::TPCTRDTOF), math_utils::bit2Mask(GlobalTrackID::ITSTPCTRDTOF)};
} // namespace globaltrackid_internal

GPUdi() constexpr GlobalTrackID::DetID::mask_t GlobalTrackID::getSourceDetectorsMask(int i) { return globaltrackid_internal::SourceDetectorsMasks[i]; }
GPUdi() constexpr GlobalTrackID::mask_t GlobalTrackID::getSourceMask(int s) { return globaltrackid_internal::sMasks[s]; }

GPUdi() bool GlobalTrackID::includesDet(DetID id, GlobalTrackID::mask_t srcm)
{
  for (int i = 0; i < NSources; i++) {
    if (includesSource(i, srcm) && getSourceDetectorsMask(i) == id.getMask()) {
      return true;
    }
  }
  return false;
}

} // namespace dataformats
} // namespace o2

#ifndef GPUCA_GPUCODE
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
#endif // GPUCA_GPUCODE

#endif
