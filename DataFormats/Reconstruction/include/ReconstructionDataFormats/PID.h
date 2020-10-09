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

#ifndef ALICEO2_track_PID_H_
#define ALICEO2_track_PID_H_

#include "GPUCommonRtypes.h"
#include "CommonConstants/PhysicsConstants.h"

namespace o2
{
namespace track
{
class PID
{
 public:
  // particle identifiers, continuos starting from 0
  typedef uint8_t ID;

  static constexpr ID Electron = 0;
  static constexpr ID Muon = 1;
  static constexpr ID Pion = 2;
  static constexpr ID Kaon = 3;
  static constexpr ID Proton = 4;
  static constexpr ID Deuteron = 5;
  static constexpr ID Triton = 6;
  static constexpr ID Helium3 = 7;
  static constexpr ID Alpha = 8;

  static constexpr ID First = Electron;
  static constexpr ID Last = Alpha;     ///< if extra IDs added, update this !!!
  static constexpr ID NIDs = Last + 1;  ///< number of defined IDs

  // PID for derived particles
  static constexpr ID PI0 = 9;
  static constexpr ID Photon = 10;
  static constexpr ID K0 = 11;
  static constexpr ID Lambda = 12;
  static constexpr ID HyperTriton = 13;
  static constexpr ID FirstExt = PI0;
  static constexpr ID LastExt = HyperTriton;
  static constexpr ID NIDsTot = LastExt + 1; ///< total number of defined IDs

  PID() = default;
  PID(ID id) : mID(id) {}
  PID(const char* name);
  PID(const PID& src) = default;
  PID& operator=(const PID& src) = default;

  ID getID() const { return mID; }
  operator ID() const { return getID(); }

  float getMass() const { return getMass(mID); }
  float getMass2Z() const { return getMass2Z(mID); }
  int getCharge() const { return getCharge(mID); }
  const char* getName() const { return getName(mID); }

  static constexpr const char* getName(ID id) { return sNames[id]; }
  static constexpr float getMass(ID id) { return sMasses[id]; }
  static constexpr float getMass2Z(ID id) { return sMasses2Z[id]; }
  static constexpr int getCharge(ID id) { return sCharges[id]; }

 private:
  ID mID = Pion;

  // are 2 strings equal ? (trick from Giulio)
  inline static constexpr bool sameStr(char const* x, char const* y)
  {
    return !*x && !*y ? true : /* default */ (*x == *y && sameStr(x + 1, y + 1));
  }

  inline static constexpr ID nameToID(char const* name, ID id)
  {
    return id > LastExt ? id : sameStr(name, sNames[id]) ? id : nameToID(name, id + 1);
  }

  static constexpr const char* sNames[NIDsTot + 1] = ///< defined particle names
    {"Electron", "Muon", "Pion", "Kaon", "Proton", "Deuteron", "Triton", "He3", "Alpha",
     "Pion0", "Photon", "K0", "Lambda", "HyperTriton",
     nullptr};

  static constexpr const float sMasses[NIDsTot] = ///< defined particle masses
    {o2::constants::physics::MassElectron, o2::constants::physics::MassMuon,
     o2::constants::physics::MassPionCharged, o2::constants::physics::MassKaonCharged,
     o2::constants::physics::MassProton, o2::constants::physics::MassDeuteron,
     o2::constants::physics::MassTriton, o2::constants::physics::MassHelium3,
     o2::constants::physics::MassAlpha,
     o2::constants::physics::MassPionNeutral, o2::constants::physics::MassPhoton,
     o2::constants::physics::MassKaonNeutral, o2::constants::physics::MassLambda,
     o2::constants::physics::MassHyperTriton};

  static constexpr const float sMasses2Z[NIDsTot] = ///< defined particle masses / Z
    {o2::constants::physics::MassElectron, o2::constants::physics::MassMuon,
     o2::constants::physics::MassPionCharged, o2::constants::physics::MassKaonCharged,
     o2::constants::physics::MassProton, o2::constants::physics::MassDeuteron,
     o2::constants::physics::MassTriton, o2::constants::physics::MassHelium3 / 2.,
     o2::constants::physics::MassAlpha / 2.,
     0, 0, 0, 0, o2::constants::physics::MassHyperTriton};

  static constexpr const int sCharges[NIDsTot] = ///< defined particle charges
    {1, 1, 1, 1, 1, 1, 1, 2, 2,
     0, 0, 0, 0, 1};

  ClassDefNV(PID, 2);
};
} // namespace track
} // namespace o2

#endif
