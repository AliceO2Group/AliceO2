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

/// @brief ALICE detectors ID's, names, masks
///
/// @author Ruben Shahoyan, ruben.shahoyan@cern.ch

/*!
  Example of class usage:
  using namespace o2::base;
  DetID det[3] = {DetID(DetID::ITS), DetID(DetID::TPC), DetID(DetID::TRD)};
  DetID::mask_t mskTot;
  for (int i=0;i<3;i++) {
    printf("detID: %2d %10s 0x%lx\n",det[i].getID(),det[i].getName(),det[i].getMask().to_ulong());
    mskTot |= det[i].getMask();
  }
  printf("joint mask: 0x%lx\n",mskTot.to_ulong());
 */

#ifndef O2_BASE_DETID_
#define O2_BASE_DETID_

#include "GPUCommonRtypes.h"
#include "GPUCommonBitSet.h"
#include "MathUtils/Utils.h"
#include "DetectorsCommonDataFormats/UpgradesStatus.h"
#ifndef GPUCA_GPUCODE_DEVICE
#include "Headers/DataHeader.h"
#include <array>
#include <bitset>
#include <cassert>
#include <cstdint>
#include <string_view>
#include <string>
#include <type_traits>
#endif

namespace o2
{
namespace header
{
}
namespace detectors
{

namespace o2h = o2::header;

/// Static class with identifiers, bitmasks and names for ALICE detectors
class DetID
{
 public:
  /// Detector identifiers: continuous, starting from 0
  typedef int ID;

  static constexpr ID ITS = 0;
  static constexpr ID TPC = 1;
  static constexpr ID TRD = 2;
  static constexpr ID TOF = 3;
  static constexpr ID PHS = 4;
  static constexpr ID CPV = 5;
  static constexpr ID EMC = 6;
  static constexpr ID HMP = 7;
  static constexpr ID MFT = 8;
  static constexpr ID MCH = 9;
  static constexpr ID MID = 10;
  static constexpr ID ZDC = 11;
  static constexpr ID FT0 = 12;
  static constexpr ID FV0 = 13;
  static constexpr ID FDD = 14;
  static constexpr ID TST = 15;
  static constexpr ID CTP = 16;
#ifdef ENABLE_UPGRADES
  static constexpr ID IT3 = 17;
  static constexpr ID TRK = 18;
  static constexpr ID FT3 = 19;
  static constexpr ID FCT = 20;
  static constexpr ID Last = FCT;
#else
  static constexpr ID Last = CTP; ///< if extra detectors added, update this !!!
#endif
  static constexpr ID First = ITS;

  static constexpr int nDetectors = Last + 1; ///< number of defined detectors
  typedef o2::gpu::gpustd::bitset<32> mask_t;
  static_assert(nDetectors <= 32, "bitset<32> insufficient");

  static constexpr mask_t FullMask = (0x1u << nDetectors) - 1;

#ifndef GPUCA_GPUCODE_DEVICE
  static constexpr std::string_view NONE{"none"}; ///< keywork for no-detector
  static constexpr std::string_view ALL{"all"};   ///< keywork for all detectors
#endif                                            // GPUCA_GPUCODE_DEVICE

  constexpr GPUdi() DetID(ID id) : mID(id)
  {
  }

#ifndef GPUCA_GPUCODE_DEVICE
  constexpr DetID(const char* name) : mID(nameToID(name, First))
  {
    // construct from the name
    assert(mID < nDetectors);
  }
#endif // GPUCA_GPUCODE_DEVICE

  GPUdDefault() DetID(const DetID& src) = default;
  GPUdDefault() DetID& operator=(const DetID& src) = default;
  // we need default c-tor only for root persistency, code must use c-tor with argument
  DetID() : mID(First) {}

  /// get detector id
  GPUdi() ID getID() const { return mID; }
  /// get detector mask
  GPUdi() mask_t getMask() const { return getMask(mID); }
#ifndef GPUCA_GPUCODE_DEVICE
  /// get detector origin
  GPUdi() o2h::DataOrigin getDataOrigin() const { return getDataOrigin(mID); }
  /// get detector name
  const char* getName() const { return getName(mID); }
#endif // GPUCA_GPUCODE_DEVICE
  /// conversion operator to int
  GPUdi() operator int() const { return static_cast<int>(mID); }

  //  ---------------- general static methods -----------------
  /// get number of defined detectors
  GPUdi() static constexpr int getNDetectors() { return nDetectors; }
  // detector ID to mask conversion
  GPUd() static constexpr mask_t getMask(ID id);

#ifndef GPUCA_GPUCODE_DEVICE
  /// names of defined detectors
  static constexpr const char* getName(ID id) { return sDetNames[id]; }
  // detector ID to DataOrigin conversions
  static constexpr o2h::DataOrigin getDataOrigin(ID id) { return sOrigins[id]; }

  // detector masks from any non-alpha-num delimiter-separated list (empty if NONE is supplied)
  static mask_t getMask(const std::string_view detList);

  static std::string getNames(mask_t mask, char delimiter = ',');

  inline static constexpr int nameToID(char const* name, int id = First)
  {
    return id > Last ? -1 : sameStr(name, sDetNames[id]) ? id
                                                         : nameToID(name, id + 1);
  }

#endif // GPUCA_GPUCODE_DEVICE

  static bool upgradesEnabled()
  {
#ifdef ENABLE_UPGRADES
    return true;
#else
    return false;
#endif
  }

 private:
  // are 2 strings equal ? (trick from Giulio)
  GPUdi() static constexpr bool sameStr(char const* x, char const* y)
  {
    return !*x && !*y ? true : /* default */ (*x == *y && sameStr(x + 1, y + 1));
  }

  ID mID = First; ///< detector ID

#ifndef GPUCA_GPUCODE_DEVICE
  // detector names, will be defined in DataSources
  static constexpr const char* sDetNames[nDetectors + 1] = ///< defined detector names
#ifdef ENABLE_UPGRADES
    {"ITS", "TPC", "TRD", "TOF", "PHS", "CPV", "EMC", "HMP", "MFT", "MCH", "MID", "ZDC", "FT0", "FV0", "FDD", "TST", "CTP", "IT3", "TRK", "FT3", "FCT", nullptr};
#else
    {"ITS", "TPC", "TRD", "TOF", "PHS", "CPV", "EMC", "HMP", "MFT", "MCH", "MID", "ZDC", "FT0", "FV0", "FDD", "TST", "CTP", nullptr};
#endif

  static constexpr std::array<o2h::DataOrigin, nDetectors>
    sOrigins = ///< detector data origins
    {o2h::gDataOriginITS, o2h::gDataOriginTPC, o2h::gDataOriginTRD, o2h::gDataOriginTOF, o2h::gDataOriginPHS,
     o2h::gDataOriginCPV, o2h::gDataOriginEMC, o2h::gDataOriginHMP, o2h::gDataOriginMFT, o2h::gDataOriginMCH,
     o2h::gDataOriginMID, o2h::gDataOriginZDC, o2h::gDataOriginFT0, o2h::gDataOriginFV0, o2h::gDataOriginFDD,
     o2h::gDataOriginTST, o2h::gDataOriginCTP
#ifdef ENABLE_UPGRADES
     ,
     o2h::gDataOriginIT3, o2h::gDataOriginTRK, o2h::gDataOriginFT3, o2h::gDataOriginFCT
#endif
  };
#endif // GPUCA_GPUCODE_DEVICE

  ClassDefNV(DetID, 4);
};

namespace detid_internal
{
// static constexpr array class members not possible on the GPU, thus we use this trick.
GPUconstexpr() DetID::mask_t sMasks[DetID::nDetectors] = ///< detectot masks
  {DetID::mask_t(math_utils::bit2Mask(DetID::ITS)), DetID::mask_t(math_utils::bit2Mask(DetID::TPC)), DetID::mask_t(math_utils::bit2Mask(DetID::TRD)), DetID::mask_t(math_utils::bit2Mask(DetID::TOF)), DetID::mask_t(math_utils::bit2Mask(DetID::PHS)),
   DetID::mask_t(math_utils::bit2Mask(DetID::CPV)), DetID::mask_t(math_utils::bit2Mask(DetID::EMC)), DetID::mask_t(math_utils::bit2Mask(DetID::HMP)), DetID::mask_t(math_utils::bit2Mask(DetID::MFT)), DetID::mask_t(math_utils::bit2Mask(DetID::MCH)),
   DetID::mask_t(math_utils::bit2Mask(DetID::MID)), DetID::mask_t(math_utils::bit2Mask(DetID::ZDC)), DetID::mask_t(math_utils::bit2Mask(DetID::FT0)), DetID::mask_t(math_utils::bit2Mask(DetID::FV0)), DetID::mask_t(math_utils::bit2Mask(DetID::FDD)),
   DetID::mask_t(math_utils::bit2Mask(DetID::TST)), DetID::mask_t(math_utils::bit2Mask(DetID::CTP))
#ifdef ENABLE_UPGRADES
                                                      ,
   DetID::mask_t(math_utils::bit2Mask(DetID::IT3)), DetID::mask_t(math_utils::bit2Mask(DetID::TRK)), DetID::mask_t(math_utils::bit2Mask(DetID::FT3)), DetID::mask_t(math_utils::bit2Mask(DetID::FCT))
#endif
};
} // namespace detid_internal

GPUdi() constexpr DetID::mask_t DetID::getMask(ID id) { return detid_internal::sMasks[id]; }

} // namespace detectors
} // namespace o2

#endif
