// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   Geometry.h
/// @brief  Interface for MCH geometry creation

#ifndef O2_MCH_GEOMETRY_CREATOR_GEOMETRY_H
#define O2_MCH_GEOMETRY_CREATOR_GEOMETRY_H

#include <vector>
#include <iostream>

class TGeoVolume;
class TGeoManager;

namespace o2::mch::geo
{

/** createGeometry creates MCH geometry.
 *
 * Geometry comprises volumes, materials and alignable volumes.
 *
 * Note that the geometry of stations 1 and 2 is attached to volume YOUT1
 * if it exist, or to topVolume otherwise.
 * 
 * Geometry for stations 3, 4 and 5 are always attached to topVolume.
 *
 * @param topVolume the volume the MCH geometry is attached topVolume
 */
void createGeometry(TGeoManager& geom, TGeoVolume& topVolume);

/** get a list of MCH sensitive volumes.
 * @returns a vector of all the MCH volumes that participate in the 
 * particle tracking (in the transport sense).
 */
std::vector<TGeoVolume*> getSensitiveVolumes();

/** Add alignable MCH volumes to the global geometry.
 *
 * Creates entries for alignable volumes associating symbolic volume
 * names with corresponding volume paths.
 *
 * @warning It also closes the geometry if it is not yet closed.
 *
 * @param geoManager the (global) TGeoManager instance, which thus
 * must exist before calling this function.
 *
 */
void addAlignableVolumes(TGeoManager& geom);

} // namespace o2::mch::geo

#endif
