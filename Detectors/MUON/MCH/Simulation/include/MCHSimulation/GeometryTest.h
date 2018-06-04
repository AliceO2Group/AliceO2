// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_MCH_SIMULATION_GEOMETRYTEST_H
#define O2_MCH_SIMULATION_GEOMETRYTEST_H

#include <iostream>

class TH2;

namespace o2
{
namespace mch
{
namespace test
{

/// creates MCH geometry from scratch (i.e. from a null TGeoManager)
/// usefull for tests or drawing for instance.
void createStandaloneGeometry();

/// tree like textual dump of the geometry nodes
void showGeometryAsTextTree(const char* fromPath = "", int maxdepth = 2, std::ostream& out = std::cout);

/// basic drawing of the geometry
void drawGeometry();

/// set the volume and daughter visibility for all volumes with a name matching the regexp pattern
void setVolumeVisibility(const char* pattern, bool visible, bool visibleDaughters);

/// set the volume line and fill for all volumes with a name matching the regexp pattern
void setVolumeColor(const char* pattern, int lineColor, int fillColor);
inline void setVolumeColor(const char* pattern, int color)
{
  setVolumeColor(pattern, color, color);
}

/// get a radlen radiograph of a given detection element within box with the given granularity
TH2* getRadio(int detElemId, float xmin, float ymin, float xmax, float ymax, float xstep, float ystep, float thickness = 5 /* cm */);

class Dummy {}; // to force Root produce a dictionary for namespace test (seems it is doing it fully if there are only functions in the namespace)

} // namespace test
} // namespace mch
} // namespace o2

#endif