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

#ifndef ALICEO2_PASSIVE_BASEPARAM_H_
#define ALICEO2_PASSIVE_BASEPARAM_H_

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"

namespace o2
{
namespace passive
{

// **
// ** Parameters for Passive base configuration
// **

enum MagnetType : int {
  AluminiumStabilizer = 0,  // Using Aluminium stabilizer for the magnet
  CopperStabilizer = 1,     // Using Copper stabilizer for the magnet
  WindingPack = 2,          // Using Winding Pack for the magnet
  SuperconductingMagnet = 3 // Using Superconducting magnet (NbTi+Cu+Al) for the magnet
};

enum MagnetAbsorberLayout : int {
  StandardRadius = 0, // Using standard radius for the magnet and absorber
  ReducedRadius = 1,  // Using reduced radius for the magnet and absorber
  SteppedAbsorber = 2 // Using stepped absorber for the magnet and absorber
};

struct Alice3PassiveBaseParam : public o2::conf::ConfigurableParamHelper<Alice3PassiveBaseParam> {
  // Geometry Builder parameters

  MagnetType mMagType = MagnetType::AluminiumStabilizer;                                // Magnet type: as in MagnetType enum
  MagnetAbsorberLayout mMagLayout = o2::passive::MagnetAbsorberLayout::SteppedAbsorber; // Magnet and absorber layout: as in MagnetAbsorberLayout enum

  O2ParamDef(Alice3PassiveBaseParam, "Alice3PassiveBase");
};

} // namespace passive
} // end namespace o2

#endif // ALICEO2_PASSIVE_BASEPARAM_H_
