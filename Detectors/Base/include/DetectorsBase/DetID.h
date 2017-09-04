// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @copyright
/// © Copyright 2014 Copyright Holders of the ALICE O2 collaboration.
/// See https://aliceinfo.cern.ch/AliceO2 for details on the Copyright holders.
/// This software is distributed under the terms of the
/// GNU General Public License version 3 (GPL Version 3).
///
/// License text in a separate file.
///
/// In applying this license, CERN does not waive the privileges and immunities
/// granted to it by virtue of its status as an Intergovernmental Organization
/// or submit itself to any jurisdiction.

/// @brief ALICE detectors ID's, names, masks
///
/// @author Ruben Shahoyan, ruben.shahoyan@cern.ch

/*!
  Example of class usage:
  using namespace o2::base;
  DetID det[3] = {DetID(DetID::ITS), DetID(DetID::TPC), DetID(DetID::TRD)};
  int mskTot = 0;
  for (int i=0;i<3;i++) {
    printf("detID: %2d %10s 0x%x\n",det[i].getID(),det[i].getName(),det[i].getMask());
    mskTot |= det[i].getMask();
  }
  printf("joint mask: 0x%x\n",mskTot);
 */

#ifndef O2_BASE_DETID_
#define O2_BASE_DETID_

#include <cstdint>
#include <array>
#include <type_traits>
#include <Rtypes.h>

namespace o2
{
namespace Base
{
/// generic template to convert enum to underlying int type
template <typename E>
constexpr typename std::underlying_type<E>::type toInt(const E e)
{
  return static_cast<typename std::underlying_type<E>::type>(e);
}

constexpr std::int32_t IDtoMask(int id) { return 0x1 << id; }
 
/// Static class with identifiers, bitmasks and names for ALICE detectors
class DetID
{
 public:
  /// Detector identifiers: continuous, starting from 0
  enum ID : std::int32_t {
    ITS = 0,
    TPC,
    TRD,
    TOF,
    PHS,
    CPV,
    EMC,
    HMP,
    MFT,
    MCH,
    MID,
    ZDC,
    FIT,
    First = ITS,
    Last = FIT
  };

  DetID(ID id);

  /// get derector id
  ID getID() const { return mID; }

  /// get detector mask
  std::int32_t getMask() const { return getMask(mID); }

  /// get detector name
  const char* getName() const { return getName(mID); }

  /// conversion operator to int
  operator int() const { return static_cast<int>(mID); }

  //  ---------------- general static methods -----------------
  /// get number of defined detectors
  static constexpr int getNDetectors() { return nDetectors; }

  /// names of defined detectors
  static const char* getName(ID id) { return sDetNames[toInt(id)]; }

  // detector ID to mask conversion
  static std::int32_t getMask(ID id) { return sMasks[toInt(id)]; }

  // we need default c-tor only for root persistency, code must use c-tor with argument
  DetID() : mID(ID::First) {}

 private:
  ID mID; ///< detector ID

  /// number of defined detectors
  static constexpr int nDetectors = toInt(ID::Last) + 1;

  static constexpr std::array<const char[4], nDetectors> sDetNames =      ///< defined detector names
    {"ITS", "TPC", "TRD", "TOF", "PHS", "CPV", "EMC", "HMP", "MFT", "MCH", "MID", "ZDC", "FIT"};

  // detector names, will be defined in DataSources
  static constexpr std::array<std::int32_t,nDetectors> sMasks =  ///< detectot masks for bitvectors
    { IDtoMask(toInt(ITS)), IDtoMask(toInt(TPC)), IDtoMask(toInt(TRD)),
      IDtoMask(toInt(TOF)), IDtoMask(toInt(PHS)), IDtoMask(toInt(CPV)),
      IDtoMask(toInt(EMC)), IDtoMask(toInt(HMP)), IDtoMask(toInt(MFT)),
      IDtoMask(toInt(MCH)), IDtoMask(toInt(MID)), IDtoMask(toInt(ZDC)),
      IDtoMask(toInt(FIT)) };

  ClassDefNV(DetID, 1);
};
}
}

#endif
