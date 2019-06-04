// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \author S+Wenzel - May 2019

#ifndef ALICEO2_EVENTGEN_GUNPARAM_H_
#define ALICEO2_EVENTGEN_GUNPARAM_H_

#include "SimConfig/ConfigurableParam.h"
#include "SimConfig/ConfigurableParamHelper.h"

namespace o2
{
namespace eventgen
{

/**
 ** A parameter class/struct holding values for
 ** the box/gun generator. Allows for quick runtime configurability
 ** of particle type, energy, direction, etc.
 **/
struct BoxGunParam : public o2::conf::ConfigurableParamHelper<BoxGunParam> {
  int pdg = 211;                     // which particle (default pion); could make this an enum
  int number = 10;                   // how many particles
  double eta[2] = { -1, 1 };         // eta range
  double prange[2] = { 0.1, 5 };     // energy range min, max in GeV
  double phirange[2] = { 0., 360. }; // phi range
  bool debug = false;                // whether to print out produced particles
  O2ParamDef(BoxGunParam, "BoxGun");
};

} // end namespace eventgen
} // end namespace o2

#endif // ALICEO2_EVENTGEN_INTERACTIONDIAMONDPARAM_H_
