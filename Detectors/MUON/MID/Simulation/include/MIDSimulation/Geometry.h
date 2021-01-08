// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MIDSimulation/Geometry.h
/// \brief  Implementation of the trigger-stations geometry
/// \author Florian Damas <florian.damas@cern.ch>
/// \date   19 june 2018

#ifndef O2_MID_SIMULATION_GEOMETRY_H
#define O2_MID_SIMULATION_GEOMETRY_H

class TGeoVolume;
class TGeoManager;

#include <vector>
#include <string>
#include "MIDBase/GeometryTransformer.h"

namespace o2
{
namespace mid
{

/// create MID geometry and attach it to existing topVolume
void createGeometry(TGeoVolume& topVolume);

/// get a list of MID sensitive volumes
std::vector<TGeoVolume*> getSensitiveVolumes();

} // namespace mid
} // namespace o2

#endif // O2_MID_SIMULATION_GEOMETRY_H
