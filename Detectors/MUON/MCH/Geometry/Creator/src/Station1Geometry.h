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

/// \file   Station1Geometry.h
/// \brief  Implementation of the station 1 geometry
/// \author Florian Damas <florian.damas@cern.ch>
/// \date   16 mai 2018

#ifndef O2_MCH_GEOMETRY_CREATOR_STATION1GEOMETRY_H
#define O2_MCH_GEOMETRY_CREATOR_STATION1GEOMETRY_H

#include <vector>

class TGeoVolume;

namespace o2::mch
{

void createStation1Geometry(TGeoVolume& topVolume);

std::vector<TGeoVolume*> getStation1SensitiveVolumes();

} // namespace o2::mch
#endif
