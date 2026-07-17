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

#ifndef O2_RICH_BASEPARAM_H
#define O2_RICH_BASEPARAM_H

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"

namespace o2
{
namespace rich
{
struct RICHBaseParam : public o2::conf::ConfigurableParamHelper<RICHBaseParam> {
  double zBaseSize = 18.6;        // cm (18.4 in v3)
  double rMax = 131.0;            // cm (117.0 in v3)
  double rMin = 104.0;            // cm (90.0 in v3)
  double radiatorThickness = 2.0; // cm
  double zRichLength = 700.0;     // cm
  int nRings = 11;               // (25 in v3)
  int nTiles = 44;               // (36 in v3)
  bool oddGeom = true;           // (false in v3)
    
  // The active and passive silicon thicknesses must sum to detectorThickness.
  double siliconeLayerThickness = 0.010;  // cm: 0.1 mm resin layer in front
  double detectorThickness = 0.1; // cm
  double activeSiliconThickness = 0.01;  // cm: 0.1 mm sensitive silicon
  //double passiveSiliconThickness = 0.09f; // cm: (detectorThickness  - activeSiliconThickness)
  
  //cylindrical aerogel layout
  bool useCylindricalAerogel = true;
  double cylindricalAerogelEtaRef = 0.85;
    
  // Enable geometry with rectangular modules
  bool useRectangularModules = true;
    
  // Barrel photosensor active area.
  double sipmActiveSizeZ = 18.0;    // cm
  double sipmActiveSizeRPhi = 17.0; // cm

  // Gas refractive index (then scaled with chromaticity)
  double nGasEffective = 1.0006;

  // Aerogel refractive index (then scaled with chromaticity)
  double nAerogelEffective = 1.03;

  // Parameters for geometry with quadrants
  bool flagUseQuadrants = false;
  // Opening between adjacent vessel quadrants, measured as a chord at shieldRMin.
  double vesselPhiGap = 1.0; // cm
  // Thickness of each lateral insulating wall at a quadrant boundary.
  double vesselThicknessShieldingLateral = 1.0; // cm
  // Rectangular size could be smaller with quadrants (< 17 cm depending on wall thickness)
  double quadrantModuleSizeRPhi = 16.5; // cm

  // Readout stack behind each SiPM plane, thicknesses along the local outward normal.
  double pcb1Thickness = 0.4;         // cm
  double coolingPlateThickness = 0.4; // cm
  double pcb2Thickness = 0.4;         // cm
  double pcb3Thickness = 0.4;         // cm
  // Surface-to-surface gaps between consecutive layers.
  double gapSiPMToPCB1 = 0.10;         // cm
  double gapPCB1ToCoolingPlate = 0.10; // cm
  double gapCoolingPlateToPCB2 = 0.10; // cm
  double gapPCB2ToPCB3 = 0.10;         // cm
    
  // Minimum edge-to-edge clearances used to avoid exact contacts between adjacent modules.
  double moduleClearanceZ = 0.02;    // cm
  double moduleClearanceRPhi = 0.02; // cm

  // Shielding:
  //  Radial boundaries of the complete cylindrical enclosure.
  double shieldRMin = 100.0;
  double shieldRMax = 136.0;
  // Radial thickness of the inner insulating wall.
  double innerWallThickness = 2.0;
  // Radial thickness of the outer insulating wall.
  double outerWallThickness = 2.0;
  // Full longitudinal length of the cylindrical side walls.
  double shieldLengthZ = 220.0;
  // Thickness of each insulating end cap along Z.
  double endCapThicknessZ = 2.0;

  // FWD and BWD RICH (legacy)
  bool enableFWDRich = false;
  bool enableBWDRich = false;
  double rFWDMin = 13.7413;
  double rFWDMax = 103.947;
  // Aerogel:
  double zAerogelMin = 375.;
  double zAerogelMax = 377.;
  // Argon:
  double zArgonMin = 377.;
  double zArgonMax = 407.;
  // Detector:
  double zSiliconMin = 407.;
  double zSiliconMax = 407.2;

  O2ParamDef(RICHBaseParam, "RICHBase");
};

} // namespace rich
} // end namespace o2

#endif