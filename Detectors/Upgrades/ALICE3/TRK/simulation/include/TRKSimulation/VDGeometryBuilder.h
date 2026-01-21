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

#ifndef O2_TRK_VDGEOMETRYBUILDER_H
#define O2_TRK_VDGEOMETRYBUILDER_H

class TGeoVolume;

#include <string>
#include <vector>

namespace o2::trk
{

// Build full VD for each design.
// Each function builds one local petal assembly (walls + layers + disks)
// and then places/rotates the petal once into the mother volume.

void createIRISGeometryFullCyl(TGeoVolume* motherVolume);          // Full-cylinder IRIS geometry (no petals, no gaps, no side walls)
void createIRISGeometryFullCylwithDisks(TGeoVolume* motherVolume); // Full-cylinder IRIS geometry (no petals, no gaps, no side walls) incl. disks
void createIRIS4Geometry(TGeoVolume* motherVolume);                // 4 petals, cylindrical L0
void createIRIS4aGeometry(TGeoVolume* motherVolume);               // 3 petals, cylindrical L0
void createIRIS5Geometry(TGeoVolume* motherVolume);                // 4 petals, rectangular L0

void createSinglePetalDebug(TGeoVolume* motherVolume, int petalID = 0, int nPetals = 4, bool rectangularL0 = false);

} // namespace o2::trk

#endif // O2_TRK_VDGEOMETRYBUILDER_H