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

#ifndef O2_MCH_GEOMETRY_TEST_HELPERS_H
#define O2_MCH_GEOMETRY_TEST_HELPERS_H

#include <iostream>

class TH2;

namespace o2::mch::test
{

/// creates MCH geometry from scratch (i.e. from a null TGeoManager)
/// usefull for tests or drawing for instance.
void createStandaloneGeometry();

/// tree like textual dump of the geometry nodes
void showGeometryAsTextTree(const char* fromPath = "", int maxdepth = 2, std::ostream& out = std::cout);

/// basic drawing of the geometry
void drawGeometry();

/// generates zero misalignments for MCH geometry
void zeroMisAlignGeometry(const std::string& ccdbHost = "http://localhost:8080", const std::string& fileName = "MCHMisAlignment.root");

/// generates misalignments for MCH geometry
void misAlignGeometry();

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

class Dummy
{
  // to force Root produce a dictionary for namespace test (seems it is doing it fully if there are only functions in the namespace)
};
} // namespace o2::mch::test

#endif
