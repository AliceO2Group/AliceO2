// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

#include <Rtypes.h>
#include <array>
#include <bitset>
#include <cassert>
#include <cstdint>
#include <string_view>
#include <string>
#include <type_traits>
#include "MathUtils/Utils.h"
#include "Headers/DataHeader.h"

namespace o2
{
namespace detectors
{

namespace o2h = o2::header;

/// Static class with identifiers, bitmasks and names for ALICE detectors
class DetID
{
 public:
  /// Detector identifiers: continuous, starting from 0
  typedef std::int32_t ID;

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
  static constexpr ID ACO = 15;
#ifdef ENABLE_UPGRADES
  static constexpr ID IT3 = 16;
  static constexpr ID TRK = 17;
  static constexpr ID Last = TRK;
#else
  static constexpr ID Last = ACO; ///< if extra detectors added, update this !!!
#endif
  static constexpr ID First = ITS;

  static constexpr int nDetectors = Last + 1; ///< number of defined detectors

  static constexpr std::string_view NONE{"none"}; ///< keywork for no-detector
  static constexpr std::string_view ALL{"all"};   ///< keywork for all detectors

  typedef std::bitset<nDetectors> mask_t;

  DetID(ID id) : mID(id) {}
  DetID(const char* name);
  DetID(const DetID& src) = default;
  DetID& operator=(const DetID& src) = default;

  /// get derector id
  ID getID() const { return mID; }
  /// get detector mask
  mask_t getMask() const { return getMask(mID); }
  /// get detector mask
  o2h::DataOrigin getDataOrigin() const { return getDataOrigin(mID); }
  /// get detector name
  const char* getName() const { return getName(mID); }
  /// conversion operator to int
  operator int() const { return static_cast<int>(mID); }

  //  ---------------- general static methods -----------------
  /// get number of defined detectors
  static constexpr int getNDetectors() { return nDetectors; }
  /// names of defined detectors
  static constexpr const char* getName(ID id) { return sDetNames[id]; }
  // detector ID to mask conversion
  static constexpr mask_t getMask(ID id) { return sMasks[id]; }
  // detector ID to DataOrigin conversions
  static constexpr o2h::DataOrigin getDataOrigin(ID id) { return sOrigins[id]; }

  // detector masks from any non-alpha-num delimiter-separated list (empty if NONE is supplied)
  static mask_t getMask(const std::string_view detList);

  static std::string getNames(mask_t mask, char delimiter = ',');

  // we need default c-tor only for root persistency, code must use c-tor with argument
  DetID() : mID(First) {}

 private:
  // are 2 strings equal ? (trick from Giulio)
  inline static constexpr bool sameStr(char const* x, char const* y)
  {
    return !*x && !*y ? true : /* default */ (*x == *y && sameStr(x + 1, y + 1));
  }

  inline static constexpr int nameToID(char const* name, int id)
  {
    return id > Last ? id : sameStr(name, sDetNames[id]) ? id : nameToID(name, id + 1);
  }

  ID mID = First; ///< detector ID

  static constexpr const char* sDetNames[nDetectors + 1] = ///< defined detector names
#ifdef ENABLE_UPGRADES
    {"ITS", "TPC", "TRD", "TOF", "PHS", "CPV", "EMC", "HMP", "MFT", "MCH", "MID", "ZDC", "FT0", "FV0", "FDD", "ACO", "IT3", "TRK", nullptr};
#else
    {"ITS", "TPC", "TRD", "TOF", "PHS", "CPV", "EMC", "HMP", "MFT", "MCH", "MID", "ZDC", "FT0", "FV0", "FDD", "ACO", nullptr};
#endif
  // detector names, will be defined in DataSources
  static constexpr std::array<mask_t, nDetectors> sMasks = ///< detectot masks
    {math_utils::bit2Mask(ITS), math_utils::bit2Mask(TPC), math_utils::bit2Mask(TRD), math_utils::bit2Mask(TOF), math_utils::bit2Mask(PHS),
     math_utils::bit2Mask(CPV), math_utils::bit2Mask(EMC), math_utils::bit2Mask(HMP), math_utils::bit2Mask(MFT), math_utils::bit2Mask(MCH),
     math_utils::bit2Mask(MID), math_utils::bit2Mask(ZDC), math_utils::bit2Mask(FT0), math_utils::bit2Mask(FV0), math_utils::bit2Mask(FDD),
     math_utils::bit2Mask(ACO)
#ifdef ENABLE_UPGRADES
       ,
     math_utils::bit2Mask(IT3), math_utils::bit2Mask(TRK)
#endif
  };

  static constexpr std::array<o2h::DataOrigin, nDetectors>
    sOrigins = ///< detector data origins
    {o2h::gDataOriginITS, o2h::gDataOriginTPC, o2h::gDataOriginTRD, o2h::gDataOriginTOF, o2h::gDataOriginPHS,
     o2h::gDataOriginCPV, o2h::gDataOriginEMC, o2h::gDataOriginHMP, o2h::gDataOriginMFT, o2h::gDataOriginMCH,
     o2h::gDataOriginMID, o2h::gDataOriginZDC, o2h::gDataOriginFT0, o2h::gDataOriginFV0, o2h::gDataOriginFDD, o2h::gDataOriginACO
#ifdef ENABLE_UPGRADES
     ,
     o2h::gDataOriginIT3, o2h::gDataOriginTRK
#endif
  };

  ClassDefNV(DetID, 1);
};

} // namespace detectors
} // namespace o2

#endif
