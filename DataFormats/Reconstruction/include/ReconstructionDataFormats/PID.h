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

/// \file PID.h
/// \brief particle ids, masses, names class definition
/// \author ruben.shahoyan@cern.ch

#ifndef ALICEO2_track_PID_H_
#define ALICEO2_track_PID_H_

#include "GPUCommonDef.h"
#include "GPUCommonRtypes.h"
#include "CommonConstants/PhysicsConstants.h"

namespace o2
{
namespace track
{
namespace o2cp = o2::constants::physics;

namespace pid_constants // GPUs currently cannot have static constexpr array members
{
typedef uint8_t ID;
static constexpr ID NIDsTot = 17;
GPUconstexpr() const char* sNames[NIDsTot + 1] = ///< defined particle names
  {"Electron", "Muon", "Pion", "Kaon", "Proton", "Deuteron", "Triton", "He3", "Alpha",
   "Pion0", "Photon", "K0", "Lambda", "HyperTriton", "Hyperhydrog4", "XiMinus", "OmegaMinus", nullptr};

GPUconstexpr() const float sMasses[NIDsTot] = ///< defined particle masses
  {o2cp::MassElectron, o2cp::MassMuon, o2cp::MassPionCharged, o2cp::MassKaonCharged,
   o2cp::MassProton, o2cp::MassDeuteron, o2cp::MassTriton, o2cp::MassHelium3,
   o2cp::MassAlpha, o2cp::MassPionNeutral, o2cp::MassPhoton,
   o2cp::MassKaonNeutral, o2cp::MassLambda, o2cp::MassHyperTriton, o2cp::MassHyperhydrog4, o2cp::MassXiMinus, o2cp::MassOmegaMinus};

GPUconstexpr() const float sMasses2[NIDsTot] = ///< defined particle masses^2
  {o2cp::MassElectron * o2cp::MassElectron,
   o2cp::MassMuon* o2cp::MassMuon,
   o2cp::MassPionCharged* o2cp::MassPionCharged,
   o2cp::MassKaonCharged* o2cp::MassKaonCharged,
   o2cp::MassProton* o2cp::MassProton,
   o2cp::MassDeuteron* o2cp::MassDeuteron,
   o2cp::MassTriton* o2cp::MassTriton,
   o2cp::MassHelium3* o2cp::MassHelium3,
   o2cp::MassAlpha* o2cp::MassAlpha,
   o2cp::MassPionNeutral* o2cp::MassPionNeutral,
   o2cp::MassPhoton* o2cp::MassPhoton,
   o2cp::MassKaonNeutral* o2cp::MassKaonNeutral,
   o2cp::MassLambda* o2cp::MassLambda,
   o2cp::MassHyperTriton* o2cp::MassHyperTriton,
   o2cp::MassHyperhydrog4* o2cp::MassHyperhydrog4,
   o2cp::MassXiMinus* o2cp::MassXiMinus,
   o2cp::MassOmegaMinus* o2cp::MassOmegaMinus};

GPUconstexpr() const float sMasses2Z[NIDsTot] = ///< defined particle masses / Z
  {o2cp::MassElectron, o2cp::MassMuon,
   o2cp::MassPionCharged, o2cp::MassKaonCharged,
   o2cp::MassProton, o2cp::MassDeuteron,
   o2cp::MassTriton, o2cp::MassHelium3 / 2.,
   o2cp::MassAlpha / 2.,
   0, 0, 0, 0, o2cp::MassHyperTriton, o2cp::MassHyperhydrog4,
   o2cp::MassXiMinus, o2cp::MassOmegaMinus};

GPUconstexpr() const int sCharges[NIDsTot] = ///< defined particle charges
  {1, 1, 1, 1, 1, 1, 1, 2, 2,
   0, 0, 0, 0, 1, 1,
   1, 1};
} // namespace pid_constants

class PID
{
 public:
  // particle identifiers, continuos starting from 0
  typedef pid_constants::ID ID;

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
  static constexpr ID Hyperhydrog4 = 14;
  static constexpr ID XiMinus = 15;
  static constexpr ID OmegaMinus = 16;
  static constexpr ID FirstExt = PI0;
  static constexpr ID LastExt = OmegaMinus;
  static constexpr ID NIDsTot = pid_constants::NIDsTot; ///< total number of defined IDs
  static_assert(NIDsTot == LastExt + 1, "Incorrect NIDsTot, please update!");

  GPUdDefault() PID() = default;
  GPUd() PID(ID id) : mID(id) {}
  GPUd() PID(const char* name);
  GPUdDefault() PID(const PID& src) = default;
  GPUhdDefault() PID& operator=(const PID& src) = default;

  GPUd() ID getID() const { return mID; }
  GPUd() operator ID() const { return getID(); }

  GPUd() float getMass() const { return getMass(mID); }
  GPUd() float getMass2() const { return getMass2(mID); }
  GPUd() float getMass2Z() const { return getMass2Z(mID); }
  GPUd() int getCharge() const { return getCharge(mID); }

  GPUd() static float getMass(ID id) { return pid_constants::sMasses[id]; }
  GPUd() static float getMass2(ID id) { return pid_constants::sMasses2[id]; }
  GPUd() static float getMass2Z(ID id) { return pid_constants::sMasses2Z[id]; }
  GPUd() static int getCharge(ID id) { return pid_constants::sCharges[id]; }
#ifndef GPUCA_GPUCODE_DEVICE
  GPUd() const char* getName() const
  {
    return getName(mID);
  }
  GPUd() static const char* getName(ID id) { return pid_constants::sNames[id]; }
#endif

 private:
  ID mID = Pion;

  // are 2 strings equal ? (trick from Giulio)
  GPUdi() static constexpr bool sameStr(char const* x, char const* y)
  {
    return !*x && !*y ? true : /* default */ (*x == *y && sameStr(x + 1, y + 1));
  }

#ifndef GPUCA_GPUCODE_DEVICE
  GPUdi() static constexpr ID nameToID(char const* name, ID id)
  {
    return id > LastExt ? id : sameStr(name, pid_constants::sNames[id]) ? id : nameToID(name, id + 1);
  }
#endif

  ClassDefNV(PID, 2);
};
} // namespace track
} // namespace o2

#endif
