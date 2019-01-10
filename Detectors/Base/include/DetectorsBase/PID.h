// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file PID.h
/// \brief particle ids, masses, names class definition
/// \author ruben.shahoyan@cern.ch

#ifndef ALICEO2_BASE_PID_H_
#define ALICEO2_BASE_PID_H_

#include <Rtypes.h>
#include "CommonConstants/PhysicsConstants.h"

namespace o2
{
namespace Base
{
class PID
{
 public:
  // particle identifiers, continuos starting from 0
  typedef std::int32_t ID;
  
  static constexpr ID Electron = 0;
  static constexpr ID Muon = 1;
  static constexpr ID Pion = 2;
  static constexpr ID Kaon = 3;
  static constexpr ID Proton = 4;
  static constexpr ID Deuteron = 5;
  static constexpr ID Triton = 6;
  static constexpr ID Helium3 = 7;
  static constexpr ID Alpha  = 8;
  
  static constexpr ID First = Electron;
  static constexpr ID Last = Alpha; ///< if extra IDs added, update this !!!
  static constexpr int NIDs = Last + 1; ///< number of defined IDs

  PID() = default;
  PID(ID id) : mID(id) {}
  PID(const char* name);
  PID(const PID& src) = default;
  PID& operator=(const PID& src) = default;

  ID getID() const { return mID; }

  float getMass() const { return sMasses[mID]; }
  const char* getName() const { return sNames[mID]; }
  
  static constexpr const char* getName(ID id) { return sNames[id]; }
  static constexpr float getMass(ID id) { return sMasses[id]; }
  
  
 private:
  ID mID = Pion;

  // are 2 strings equal ? (trick from Giulio)
  inline static constexpr bool sameStr(char const* x, char const* y)
  {
    return !*x && !*y ? true : /* default */ (*x == *y && sameStr(x + 1, y + 1));
  }

  inline static constexpr int nameToID(char const* name, int id)
  {
    return id > Last ? id : sameStr(name, sNames[id]) ? id : nameToID(name, id + 1);
  }

  static constexpr const char* sNames[NIDs + 1] = ///< defined particle names
    { "Electron", "Muon", "Pion", "Kaon", "Proton", "Deuteron", "Triton", "Alpa", nullptr };

  static constexpr const float sMasses[NIDs] = ///< defined particle masses
    { o2::constants::physics::MassElectron, o2::constants::physics::MassMuon,
      o2::constants::physics::MassPionCharged, o2::constants::physics::MassKaonCharged,
      o2::constants::physics::MassProton, o2::constants::physics::MassDeuteron,
      o2::constants::physics::MassTriton, o2::constants::physics::MassHelium3,
      o2::constants::physics::MassAlpha };

  
  
  ClassDefNV(PID, 1);
};
} // namespace track
} // namespace o2

#endif
