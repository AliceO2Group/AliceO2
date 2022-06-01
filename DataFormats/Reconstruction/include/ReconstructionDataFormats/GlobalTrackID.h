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
    MFTMCH,
    ITSTPCTRD, // 3-detector tracks
    ITSTPCTOF,
    TPCTRDTOF,
    MFTMCHMID,
    ITSTPCTRDTOF, // full barrel track
    ITSAB,        // ITS AfterBurner tracklets
    CTP,
    //
    MCHMID, // Temporary ordering
    //
    NSources
  };

  using AbstractRef<25, 5, 2>::AbstractRef;
  static_assert(NSources <= 32, "bitset<32> insufficient");
  typedef o2::gpu::gpustd::bitset<32> mask_t;

#ifndef GPUCA_GPUCODE
  static constexpr std::string_view NONE{"none"}; ///< keywork for no sources
  static constexpr std::string_view ALL{"all"};   ///< keywork for all sources
#endif
  static constexpr mask_t MASK_ALL = (1u << NSources) - 1;
  static constexpr mask_t MASK_NONE = 0;

  // methods for detector level manipulations
  GPUd() static constexpr DetID::mask_t getSourceDetectorsMask(int i);
  GPUd() static constexpr DetID::mask_t getSourcesDetectorsMask(GlobalTrackID::mask_t srcm);
  GPUd() static bool includesDet(DetID id, GlobalTrackID::mask_t srcm);
  GPUdi() auto getSourceDetectorsMask() const { return getSourceDetectorsMask(getSource()); }
  GPUdi() bool includesDet(DetID id) const { return (getSourceDetectorsMask() & DetID::getMask(id)).any(); }

  // methods for source level manipulations
#ifndef GPUCA_GPUCODE
  static auto getSourceName(int s)
  {
    return DetID::getNames(getSourceDetectorsMask(s), '-');
  }
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
  DetID::mask_t(DetID::getMask(DetID::ITS)),
  DetID::mask_t(DetID::getMask(DetID::TPC)),
  DetID::mask_t(DetID::getMask(DetID::TRD)),
  DetID::mask_t(DetID::getMask(DetID::TOF)),
  DetID::mask_t(DetID::getMask(DetID::PHS)),
  DetID::mask_t(DetID::getMask(DetID::CPV)),
  DetID::mask_t(DetID::getMask(DetID::EMC)),
  DetID::mask_t(DetID::getMask(DetID::HMP)),
  DetID::mask_t(DetID::getMask(DetID::MFT)),
  DetID::mask_t(DetID::getMask(DetID::MCH)),
  DetID::mask_t(DetID::getMask(DetID::MID)),
  DetID::mask_t(DetID::getMask(DetID::ZDC)),
  DetID::mask_t(DetID::getMask(DetID::FT0)),
  DetID::mask_t(DetID::getMask(DetID::FV0)),
  DetID::mask_t(DetID::getMask(DetID::FDD)),
  //
  DetID::mask_t(DetID::getMask(DetID::ITS) | DetID::getMask(DetID::TPC)),
  DetID::mask_t(DetID::getMask(DetID::TPC) | DetID::getMask(DetID::TOF)),
  DetID::mask_t(DetID::getMask(DetID::TPC) | DetID::getMask(DetID::TRD)),
  DetID::mask_t(DetID::getMask(DetID::MFT) | DetID::getMask(DetID::MCH)),
  DetID::mask_t(DetID::getMask(DetID::ITS) | DetID::getMask(DetID::TPC) | DetID::getMask(DetID::TRD)),
  DetID::mask_t(DetID::getMask(DetID::ITS) | DetID::getMask(DetID::TPC) | DetID::getMask(DetID::TOF)),
  DetID::mask_t(DetID::getMask(DetID::TPC) | DetID::getMask(DetID::TRD) | DetID::getMask(DetID::TOF)),
  DetID::mask_t(DetID::getMask(DetID::MFT) | DetID::getMask(DetID::MCH) | DetID::getMask(DetID::MID)),
  DetID::mask_t(DetID::getMask(DetID::ITS) | DetID::getMask(DetID::TPC) | DetID::getMask(DetID::TRD) | DetID::getMask(DetID::TOF)),
  DetID::mask_t(DetID::getMask(DetID::ITS)),
  DetID::mask_t(DetID::getMask(DetID::CTP)),
  DetID::mask_t(DetID::getMask(DetID::MCH) | DetID::getMask(DetID::MID)) // Temporary ordering
};

GPUconstexpr() GlobalTrackID::mask_t sMasks[GlobalTrackID::NSources] = ///< detector masks
  {
    GlobalTrackID::mask_t(math_utils::bit2Mask(GlobalTrackID::ITS)), GlobalTrackID::mask_t(math_utils::bit2Mask(GlobalTrackID::TPC)), GlobalTrackID::mask_t(math_utils::bit2Mask(GlobalTrackID::TRD)), GlobalTrackID::mask_t(math_utils::bit2Mask(GlobalTrackID::TOF)), GlobalTrackID::mask_t(math_utils::bit2Mask(GlobalTrackID::PHS)),
    GlobalTrackID::mask_t(math_utils::bit2Mask(GlobalTrackID::CPV)), GlobalTrackID::mask_t(math_utils::bit2Mask(GlobalTrackID::EMC)), GlobalTrackID::mask_t(math_utils::bit2Mask(GlobalTrackID::HMP)), GlobalTrackID::mask_t(math_utils::bit2Mask(GlobalTrackID::MFT)), GlobalTrackID::mask_t(math_utils::bit2Mask(GlobalTrackID::MCH)),
    GlobalTrackID::mask_t(math_utils::bit2Mask(GlobalTrackID::MID)), GlobalTrackID::mask_t(math_utils::bit2Mask(GlobalTrackID::ZDC)), GlobalTrackID::mask_t(math_utils::bit2Mask(GlobalTrackID::FT0)), GlobalTrackID::mask_t(math_utils::bit2Mask(GlobalTrackID::FV0)), GlobalTrackID::mask_t(math_utils::bit2Mask(GlobalTrackID::FDD)),
    GlobalTrackID::mask_t(math_utils::bit2Mask(GlobalTrackID::ITSTPC)), GlobalTrackID::mask_t(math_utils::bit2Mask(GlobalTrackID::TPCTOF)), GlobalTrackID::mask_t(math_utils::bit2Mask(GlobalTrackID::TPCTRD)), GlobalTrackID::mask_t(math_utils::bit2Mask(GlobalTrackID::MFTMCH)), GlobalTrackID::mask_t(math_utils::bit2Mask(GlobalTrackID::ITSTPCTRD)),
    GlobalTrackID::mask_t(math_utils::bit2Mask(GlobalTrackID::ITSTPCTOF)), GlobalTrackID::mask_t(math_utils::bit2Mask(GlobalTrackID::TPCTRDTOF)), GlobalTrackID::mask_t(math_utils::bit2Mask(GlobalTrackID::ITSTPCTRDTOF)), GlobalTrackID::mask_t(math_utils::bit2Mask(GlobalTrackID::MFTMCHMID)),
    GlobalTrackID::mask_t(math_utils::bit2Mask(GlobalTrackID::ITSAB)),
    GlobalTrackID::mask_t(math_utils::bit2Mask(GlobalTrackID::CTP)),
    GlobalTrackID::mask_t(math_utils::bit2Mask(GlobalTrackID::MCHMID)) // Temporary ordering
};
} // namespace globaltrackid_internal

GPUdi() constexpr GlobalTrackID::DetID::mask_t GlobalTrackID::getSourceDetectorsMask(int i) { return globaltrackid_internal::SourceDetectorsMasks[i]; }
GPUdi() constexpr GlobalTrackID::mask_t GlobalTrackID::getSourceMask(int s) { return globaltrackid_internal::sMasks[s]; }

GPUdi() bool GlobalTrackID::includesDet(DetID id, GlobalTrackID::mask_t srcm)
{
  for (int i = 0; i < NSources; i++) {
    if (includesSource(i, srcm) && (getSourceDetectorsMask(i) & id.getMask()).any()) {
      return true;
    }
  }
  return false;
}

GPUd() constexpr GlobalTrackID::DetID::mask_t GlobalTrackID::getSourcesDetectorsMask(GlobalTrackID::mask_t srcm)
{
  GlobalTrackID::DetID::mask_t mdet;
  for (int i = 0; i < NSources; i++) {
    if (srcm[i]) {
      mdet |= getSourceDetectorsMask(i);
    }
  }
  return mdet;
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
