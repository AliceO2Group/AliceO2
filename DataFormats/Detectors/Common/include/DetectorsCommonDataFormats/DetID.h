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
#include <type_traits>
#include "MathUtils/Utils.h"

namespace o2
{
namespace detectors
{
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
  static constexpr ID First = ITS;
  static constexpr ID Last = ACO; ///< if extra detectors added, update this !!!

  static constexpr int nDetectors = Last + 1; ///< number of defined detectors

  typedef std::bitset<nDetectors> mask_t;

  DetID(ID id) : mID(id) {}
  DetID(const char* name);
  DetID(const DetID& src) = default;
  DetID& operator=(const DetID& src) = default;

  /// get derector id
  ID getID() const { return mID; }
  /// get detector mask
  mask_t getMask() const { return getMask(mID); }
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

    {"ITS", "TPC", "TRD", "TOF", "PHS", "CPV", "EMC", "HMP", "MFT", "MCH", "MID", "ZDC", "FT0", "FV0", "FDD", "ACO", nullptr};

  // detector names, will be defined in DataSources
  static constexpr std::array<mask_t, nDetectors> sMasks = ///< detectot masks
    {utils::bit2Mask(ITS), utils::bit2Mask(TPC), utils::bit2Mask(TRD), utils::bit2Mask(TOF), utils::bit2Mask(PHS),
     utils::bit2Mask(CPV), utils::bit2Mask(EMC), utils::bit2Mask(HMP), utils::bit2Mask(MFT), utils::bit2Mask(MCH),
     utils::bit2Mask(MID), utils::bit2Mask(ZDC), utils::bit2Mask(FT0), utils::bit2Mask(FV0), utils::bit2Mask(FDD), utils::bit2Mask(ACO)};

  ClassDefNV(DetID, 1);
};
} // namespace detectors
} // namespace o2
#endif
