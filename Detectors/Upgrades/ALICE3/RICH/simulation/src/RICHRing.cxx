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

#include "RICHSimulation/RICHRing.h"
#include "RICHBase/GeometryTGeo.h"
#include "RICHBase/RICHBaseParam.h"
#include "Framework/Logger.h"

#include <TGeoManager.h>
#include <TGeoTube.h>
#include <TGeoVolume.h>
#include <TGeoArb8.h>

#include <cmath>
#include <limits>

namespace o2
{
namespace rich
{

namespace // quadrant operations
{

double quadrantDeltaPhiEquation(double x, int nTilesPhi, double rMin, double totalBoundaryWidth)
{
  const double argument = totalBoundaryWidth * TMath::Cos(x / 2.0) / (2.0 * rMin);
  if (TMath::Abs(argument) >= 1.0) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  const double rhs = 2.0 * TMath::Pi() / static_cast<double>(nTilesPhi) - (8.0 / static_cast<double>(nTilesPhi)) * TMath::ASin(argument);
  return rhs - x;
}

double solveQuadrantDeltaPhi(int nTilesPhi, double rMin, double totalBoundaryWidth)
{
  double lower = 0.0;
  double upper = 1.1 * 2.0 * TMath::Pi() / static_cast<double>(nTilesPhi);
  double fLower = quadrantDeltaPhiEquation(lower, nTilesPhi, rMin, totalBoundaryWidth);
  double fUpper = quadrantDeltaPhiEquation(upper, nTilesPhi, rMin, totalBoundaryWidth);
  if (!std::isfinite(fLower) || !std::isfinite(fUpper) || fLower * fUpper > 0.0) {
    return -1.0;
  }
  constexpr double tolerance = 1.0e-12;
  constexpr int maxIterations = 200;
  for (int iteration = 0; iteration < maxIterations; iteration++) {
    const double middle = 0.5 * (lower + upper);
    const double fMiddle = quadrantDeltaPhiEquation(middle, nTilesPhi, rMin, totalBoundaryWidth);
    if (!std::isfinite(fMiddle)) {
      return -1.0;
    }
    if (TMath::Abs(fMiddle) < tolerance || 0.5 * (upper - lower) < tolerance) {
      return middle;
    }
    if (fLower * fMiddle < 0.0) {
      upper = middle;
      fUpper = fMiddle;
    } else {
      lower = middle;
      fLower = fMiddle;
    }
  }
  return 0.5 * (lower + upper);
}

double quadrantModulePhi(int moduleIndex, int nTilesPhi, double deltaPhi, double extraPhi)
{
  const int modulesPerQuadrant = nTilesPhi / 4;
  const int quadrant = moduleIndex / modulesPerQuadrant;
  return extraPhi + static_cast<double>(moduleIndex) * deltaPhi - TMath::Pi() / 4.0 + deltaPhi / 2.0 + 2.0 * static_cast<double>(quadrant) * extraPhi;
}

} // namespace

Ring::Ring(int rPosId,
           int nTilesPhi,
           double rMin,
           double rMax,
           double radThick,
           double radYmin,
           double radYmax,
           double radZ,
           double photThick,
           double photYmin,
           double photYmax,
           double photZ,
           double radRad0,
           double photR0,
           double aerDetDistance,
           double thetaB,
           const std::string motherName)
  : mNTiles{nTilesPhi}, mPosId{rPosId}, mRadThickness{radThick}
{
  TGeoManager* geoManager = gGeoManager;
  TGeoVolume* motherVolume = geoManager->GetVolume(motherName.c_str());

  if (!motherVolume) {
    LOGP(fatal,
         "RICH: mother volume {} not found while creating ring {}",
         motherName,
         rPosId);
  }

  const auto& richPars = RICHBaseParam::Instance();

  const bool useCylindricalAerogel = richPars.useCylindricalAerogel;

  TGeoMedium* medAerogel = gGeoManager->GetMedium("RCH_AEROGEL$");
  if (!medAerogel) {
    LOGP(fatal, "RICH: Aerogel medium not found");
  }

  TGeoMedium* medSi = gGeoManager->GetMedium("RCH_SILICON$");
  if (!medSi) {
    LOGP(fatal, "RICH: Silicon medium not found");
  }

  TGeoMedium* medCO2 = gGeoManager->GetMedium("RCH_CO2$");
  if (!medCO2) {
    LOGP(fatal, "RICH: CO2 medium not found");
  }

  TGeoMedium* medFR4 = gGeoManager->GetMedium("RCH_FR4$");
  if (!medFR4) {
    LOGP(fatal, "RICH: FR4 medium not found");
  }

  TGeoMedium* medAr = gGeoManager->GetMedium("RCH_ARGON$");
  if (!medAr) {
    LOGP(fatal, "RICH: Argon medium not found");
  }

  TGeoMedium* medAl = gGeoManager->GetMedium("RCH_ALUMINUM$");
  if (!medAl) {
    LOGP(fatal, "RICH: Aluminum medium not found");
  }

  TGeoMedium* medSiAbsorber = gGeoManager->GetMedium("RCH_SILICON_ABSORBER$");
  if (!medSiAbsorber) {
    LOGP(fatal, "RICH: Passive silicon absorber medium not found");
  }

  TGeoMedium* medSilicone = gGeoManager->GetMedium("RCH_SILICONE$");
  if (!medSilicone) {
    LOGP(fatal, "RICH: Silicone medium not found");
  }

  TGeoMedium* medHTCC = gGeoManager->GetMedium("RCH_HTCC$");
  if (!medHTCC) {
    LOGP(fatal, "RICH: HTCC medium not found");
  }

  std::vector<TGeoArb8*> radiatorTiles(nTilesPhi), photoFrames(nTilesPhi), photoTiles(nTilesPhi), gasSectors(nTilesPhi);
  LOGP(info, "Creating ring: id: {} with {} tiles. ", rPosId, nTilesPhi);
  LOGP(info, "Rmin: {} Rmax: {} RadThick: {} RadYmin: {} RadYmax: {} RadZ: {} PhotThick: {} PhotYmin: {} PhotYmax: {} PhotZ: {}, zTransRad: {}, zTransPhot: {}, ThetaB: {}",
       rMin, rMax, radThick, radYmin, radYmax, radZ, photThick, photYmin, photYmax, photZ, radRad0, photR0, thetaB);

  // Use different phi depending on use of quadrants or not
  const bool flagUseQuadrants = richPars.flagUseQuadrants;
  if (flagUseQuadrants && (nTilesPhi <= 0 || nTilesPhi % 4 != 0)) {
    LOGP(fatal, "RICH quadrant geometry requires nTilesPhi to be positive and divisible by four; received {}", nTilesPhi);
  }
  const double regularDeltaPhi = 2.0 * TMath::Pi() / static_cast<double>(nTilesPhi);
  double moduleDeltaPhi = regularDeltaPhi;
  double quadrantExtraPhi = 0.0;

  if (flagUseQuadrants) {
    const double totalBoundaryWidth = 2.0 * richPars.vesselThicknessShieldingLateral + richPars.vesselPhiGap;
    if (totalBoundaryWidth >= 2.0 * rMin) {
      LOGP(fatal, "RICH quadrant boundary width {} cm is incompatible with rMin={} cm", totalBoundaryWidth, rMin);
    }
    moduleDeltaPhi = solveQuadrantDeltaPhi(nTilesPhi, rMin, totalBoundaryWidth);

    quadrantExtraPhi = TMath::ASin(totalBoundaryWidth / (2.0 * rMin));

    if (!(moduleDeltaPhi > 0.0)) {
      LOGP(fatal, "RICH ring {} could not solve the quadrant angular pitch", rPosId);
    }
  }

  auto modulePhiRad = [&](int moduleIndex) {
    if (!flagUseQuadrants) {
      // Original placement exactly.
      return static_cast<double>(moduleIndex) * regularDeltaPhi;
    }
    return quadrantModulePhi(moduleIndex, nTilesPhi, moduleDeltaPhi, quadrantExtraPhi);
  };

  const double thetaBDeg = thetaB * 180.0 / TMath::Pi();

  const double sipmActiveSizeZ = richPars.sipmActiveSizeZ;
  // const double sipmActiveSizeRPhi = richPars.sipmActiveSizeRPhi;
  //  Select width depending on having quadrants or not (and wall thickness)
  const double sipmActiveSizeRPhi = flagUseQuadrants ? richPars.quadrantModuleSizeRPhi : richPars.sipmActiveSizeRPhi;

  const double pcb1Thickness = richPars.pcb1Thickness;
  const double coolingPlateThickness = richPars.coolingPlateThickness;
  const double pcb2Thickness = richPars.pcb2Thickness;
  const double pcb3Thickness = richPars.pcb3Thickness;

  const double gapSiPMToPCB1 = richPars.gapSiPMToPCB1;
  const double gapPCB1ToCoolingPlate = richPars.gapPCB1ToCoolingPlate;
  const double gapCoolingPlateToPCB2 = richPars.gapCoolingPlateToPCB2;
  const double gapPCB2ToPCB3 = richPars.gapPCB2ToPCB3;

  const bool oddGeom = richPars.oddGeom;
  const bool useRectangularModules = richPars.useRectangularModules;

  const int nRings = richPars.nRings;

  const double moduleClearanceZ = richPars.moduleClearanceZ;
  const double moduleClearanceRPhi = richPars.moduleClearanceRPhi;

  const double siliconeLayerThickness = richPars.siliconeLayerThickness;
  const double activeSiliconThickness = richPars.activeSiliconThickness;
  const double passiveSiliconThickness = photThick - activeSiliconThickness;

  const double siliconFrontSurfaceOffset = -photThick / 2.0;
  const double siliconeCenterOffset = siliconFrontSurfaceOffset - siliconeLayerThickness / 2.0;
  const double activeSiliconCenterOffset = siliconFrontSurfaceOffset + activeSiliconThickness / 2.0;
  const double passiveSiliconCenterOffset = siliconFrontSurfaceOffset + activeSiliconThickness + passiveSiliconThickness / 2.0;

  if (siliconeLayerThickness <= 0.0) {
    LOGP(fatal, "RICH: siliconeLayerThickness must be positive");
  }

  if (activeSiliconThickness <= 0.0 || activeSiliconThickness >= photThick) {
    LOGP(fatal, "RICH: activeSiliconThickness={} cm must be larger than zero and smaller than detectorThickness={} cm", activeSiliconThickness, photThick);
  }

  if (passiveSiliconThickness <= 0.0) {
    LOGP(fatal, "RICH: passive silicon thickness must be positive");
  }

  if (moduleClearanceZ < 0.0 || moduleClearanceRPhi < 0.0) {
    LOGP(fatal, "RICH: module clearances cannot be negative");
  }

  if (photThick <= 0.0 || sipmActiveSizeZ <= 0.0 || sipmActiveSizeRPhi <= 0.0 || pcb1Thickness <= 0.0 || coolingPlateThickness <= 0.0 || pcb2Thickness <= 0.0 || pcb3Thickness <= 0.0) {
    LOGP(fatal, "RICH: SiPM and readout-stack dimensions must be positive");
  }

  if (gapSiPMToPCB1 < 0.0 || gapPCB1ToCoolingPlate < 0.0 || gapCoolingPlateToPCB2 < 0.0 || gapPCB2ToPCB3 < 0.0) {
    LOGP(fatal, "RICH: readout-stack gaps cannot be negative");
  }

  const double minimumFrameSizeRPhi = photYmin < photYmax ? photYmin : photYmax;

  if (sipmActiveSizeZ > photZ || sipmActiveSizeRPhi > minimumFrameSizeRPhi) {
    LOGP(fatal,
         "RICH: rectangular module {} x {} cm2 does not fit inside the trapezoidal sector {} x [{}, {}] cm2 for ring {}. "
         "For quadrant mode reduce: quadrantModuleSizeRPhi.",
         sipmActiveSizeZ, sipmActiveSizeRPhi, photZ, photYmin, photYmax, rPosId);
  }

  // Number of actual aerogel rows.
  const int nAerogelRows = oddGeom ? nRings : nRings - 1;

  // Convert the projective-ring ID into a contiguous aerogel-row index.
  // Example for nRings=11 and even geometry:
  //   projective IDs:  0 1 2 3 4 [5 skipped] 6 7 8 9 10
  //   aerogel index:   0 1 2 3 4             5 6 7 8  9
  int aerogelRowIndex = rPosId;
  if (!oddGeom && rPosId > nRings / 2) {
    --aerogelRowIndex;
  }

  const double cylindricalAerogelCenterZ = -0.5 * static_cast<double>(nAerogelRows) * radZ + 0.5 * radZ + static_cast<double>(aerogelRowIndex) * radZ;

  int radTileCount{0}, photTileCount{0}; // argSectorsCount{0};

  if (flagUseQuadrants) {
    LOGP(info, "RICH ring {} quadrant placement: deltaPhi={} deg, boundary half-gap={} deg", rPosId, moduleDeltaPhi * 180.0 / TMath::Pi(), quadrantExtraPhi * 180.0 / TMath::Pi());
  }

  // Radiator tiles
  for (auto& radiatorTile : radiatorTiles) {
    // Local Z is the thin (radial) dimension, looking outward from the IP
    // (previously this was local X, while for running with ACTS we need local Z).
    // The placement rotation below is adjusted by +90 deg about Y
    // to keep the tile in the same physical position.
    if (useCylindricalAerogel) {
      // Including gab between adjacent aerogel tiles
      const double cylindricalTileSizeZ = radZ - moduleClearanceZ;
      const double cylindricalTileYmin = radYmin - moduleClearanceRPhi;
      const double cylindricalTileYmax = radYmax - moduleClearanceRPhi;
      if (cylindricalTileSizeZ <= 0.0 || cylindricalTileYmin <= 0.0 || cylindricalTileYmax <= 0.0) {
        LOGP(fatal, "RICH: cylindrical-aerogel clearances are larger than the tile dimensions for ring {}", rPosId);
      }
      radiatorTile = new TGeoArb8(radThick / 2);
      radiatorTile->SetVertex(0, cylindricalTileSizeZ / 2, -cylindricalTileYmin / 2);
      radiatorTile->SetVertex(1, -cylindricalTileSizeZ / 2, -cylindricalTileYmax / 2);
      radiatorTile->SetVertex(2, -cylindricalTileSizeZ / 2, cylindricalTileYmax / 2);
      radiatorTile->SetVertex(3, cylindricalTileSizeZ / 2, cylindricalTileYmin / 2);
      radiatorTile->SetVertex(4, cylindricalTileSizeZ / 2, -cylindricalTileYmin / 2);
      radiatorTile->SetVertex(5, -cylindricalTileSizeZ / 2, -cylindricalTileYmax / 2);
      radiatorTile->SetVertex(6, -cylindricalTileSizeZ / 2, cylindricalTileYmax / 2);
      radiatorTile->SetVertex(7, cylindricalTileSizeZ / 2, cylindricalTileYmin / 2);
    } else {
      // Original non-cylindrical tile definition.
      radiatorTile = new TGeoArb8(radThick / 2);
      radiatorTile->SetVertex(0, radZ / 2, -radYmin / 2);
      radiatorTile->SetVertex(1, -radZ / 2, -radYmax / 2);
      radiatorTile->SetVertex(2, -radZ / 2, radYmax / 2);
      radiatorTile->SetVertex(3, radZ / 2, radYmin / 2);
      radiatorTile->SetVertex(4, radZ / 2, -radYmin / 2);
      radiatorTile->SetVertex(5, -radZ / 2, -radYmax / 2);
      radiatorTile->SetVertex(6, -radZ / 2, radYmax / 2);
      radiatorTile->SetVertex(7, radZ / 2, radYmin / 2);
    }

    TGeoVolume* radiatorTileVol = new TGeoVolume(Form("radTile_%d_%d", rPosId, radTileCount), radiatorTile, medAerogel);
    radiatorTileVol->SetLineColor(kBlue - 9);
    radiatorTileVol->SetLineWidth(1);

    // const double phiDeg = static_cast<double>(radTileCount) * deltaPhiDeg;
    // const double phiRad = static_cast<double>(radTileCount) * 2.0 * TMath::Pi() / static_cast<double>(nTilesPhi);

    const double phiRad = modulePhiRad(radTileCount);
    const double phiDeg = phiRad * 180.0 / TMath::Pi();

    auto* rotRadiator = new TGeoRotation(Form("radTileRotation_%d_%d", radTileCount, rPosId));

    if (useCylindricalAerogel) {
      // The TGeoArb8 local Z axis is the thin direction.
      // RotateY(90 degrees) maps that thin local Z direction onto the global radial direction at phi=0.
      // There is no thetaB tilt because the cylindrical aerogel tiles are parallel to the beam axis.
      rotRadiator->RotateY(90.0);
    } else {
      // Original projective rotation.
      rotRadiator->RotateY(90.0 - thetaBDeg);
    }

    // Rotate the radial tile around the beam axis to its phi sector.
    rotRadiator->RotateZ(phiDeg);

    const double radiatorCenterZ = useCylindricalAerogel ? cylindricalAerogelCenterZ : radRad0 * TMath::Tan(thetaB);

    auto* rotTransRadiator = new TGeoCombiTrans(radRad0 * TMath::Cos(phiRad), radRad0 * TMath::Sin(phiRad), radiatorCenterZ, rotRadiator);

    motherVolume->AddNode(radiatorTileVol, 1, rotTransRadiator);
    radTileCount++;
  }

  // Photosensor tiles: legacy trapezoidal modules and rectangular modules
  if (!useRectangularModules) {
    for (auto& photoTile : photoTiles) {
      const double phiRad = modulePhiRad(photTileCount);
      const double phiDeg = phiRad * 180.0 / TMath::Pi();
      // Local Z is the thin (radial) dimension, looking outward from the IP
      photoTile = new TGeoArb8(photThick / 2);
      photoTile->SetVertex(0, photZ / 2, -photYmin / 2);
      photoTile->SetVertex(1, -photZ / 2, -photYmax / 2);
      photoTile->SetVertex(2, -photZ / 2, photYmax / 2);
      photoTile->SetVertex(3, photZ / 2, photYmin / 2);
      photoTile->SetVertex(4, photZ / 2, -photYmin / 2);
      photoTile->SetVertex(5, -photZ / 2, -photYmax / 2);
      photoTile->SetVertex(6, -photZ / 2, photYmax / 2);
      photoTile->SetVertex(7, photZ / 2, photYmin / 2);

      TGeoVolume* photoTileVol = new TGeoVolume(Form("%s_%d_%d", GeometryTGeo::getRICHSensorPattern(), rPosId, photTileCount), photoTile, medSi);
      photoTileVol->SetLineColor(kOrange + 2);
      photoTileVol->SetLineWidth(1);

      auto* rotPhoto = new TGeoRotation(Form("photoTileRotation_%d_%d", photTileCount, rPosId));
      rotPhoto->RotateY(90.0 - thetaBDeg); // +90 compensates the X->Z swap of the tile's local axes
      // rotPhoto->RotateZ(photTileCount * deltaPhiDeg);
      rotPhoto->RotateZ(phiDeg);
      // auto* rotTransPhoto = new TGeoCombiTrans(photR0 * TMath::Cos(photTileCount * TMath::Pi() / (nTilesPhi / 2)),
      //                                         photR0 * TMath::Sin(photTileCount * TMath::Pi() / (nTilesPhi / 2)),
      //                                         photR0 * TMath::Tan(thetaB),
      //                                         rotPhoto);
      auto* rotTransPhoto = new TGeoCombiTrans(photR0 * TMath::Cos(phiRad), photR0 * TMath::Sin(phiRad), photR0 * TMath::Tan(thetaB), rotPhoto);

      motherVolume->AddNode(photoTileVol, 1, rotTransPhoto);
      photTileCount++;
    }
  } else // <-- New gemetry with rectangular modules
  {
    // Photosensor tiles and readout stack
    for (auto& photoTile : photoTiles) {
      // const double phiDeg = static_cast<double>(photTileCount) * deltaPhiDeg;
      // const double phiRad = static_cast<double>(photTileCount) * 2.0 * TMath::Pi() / static_cast<double>(nTilesPhi);
      const double phiRad = modulePhiRad(photTileCount);
      const double phiDeg = phiRad * 180.0 / TMath::Pi();

      const double photoCenterR = photR0;
      const double photoCenterZ = photR0 * TMath::Tan(thetaB);

      // Unit vector normal to the projective plane, pointing away from the IP. Positive offset places layer behind the SiPM.
      const double normalRadial = TMath::Cos(thetaB);
      const double normalZ = TMath::Sin(thetaB);

      auto makeProjectiveRotation = [&](const char* prefix) {
        auto* rotation = new TGeoRotation(Form("%sRotation_%d_%d", prefix, photTileCount, rPosId));
        rotation->RotateY(90.0 - thetaBDeg); // same orientation as the original photosensor
        rotation->RotateZ(phiDeg);
        return rotation;
      };

      const double frameSizeZ = photZ - moduleClearanceZ;
      const double frameYmin = photYmin - moduleClearanceRPhi;
      const double frameYmax = photYmax - moduleClearanceRPhi;
      // Footprint of the frames with the configured clearances for overlaps
      auto makeFrameFootprint = [&](double thickness) {
        auto* shape = new TGeoArb8(thickness / 2.0);
        shape->SetVertex(0, frameSizeZ / 2.0, -frameYmin / 2.0);
        shape->SetVertex(1, -frameSizeZ / 2.0, -frameYmax / 2.0);
        shape->SetVertex(2, -frameSizeZ / 2.0, frameYmax / 2.0);
        shape->SetVertex(3, frameSizeZ / 2.0, frameYmin / 2.0);
        shape->SetVertex(4, frameSizeZ / 2.0, -frameYmin / 2.0);
        shape->SetVertex(5, -frameSizeZ / 2.0, -frameYmax / 2.0);
        shape->SetVertex(6, -frameSizeZ / 2.0, frameYmax / 2.0);
        shape->SetVertex(7, frameSizeZ / 2.0, frameYmin / 2.0);
        return shape;
      };

      auto makeRectangularFootprint = [&](double thickness) {
        auto* shape = new TGeoArb8(thickness / 2.0);
        shape->SetVertex(0, sipmActiveSizeZ / 2.0, -sipmActiveSizeRPhi / 2.0);
        shape->SetVertex(1, -sipmActiveSizeZ / 2.0, -sipmActiveSizeRPhi / 2.0);
        shape->SetVertex(2, -sipmActiveSizeZ / 2.0, sipmActiveSizeRPhi / 2.0);
        shape->SetVertex(3, sipmActiveSizeZ / 2.0, sipmActiveSizeRPhi / 2.0);
        shape->SetVertex(4, sipmActiveSizeZ / 2.0, -sipmActiveSizeRPhi / 2.0);
        shape->SetVertex(5, -sipmActiveSizeZ / 2.0, -sipmActiveSizeRPhi / 2.0);
        shape->SetVertex(6, -sipmActiveSizeZ / 2.0, sipmActiveSizeRPhi / 2.0);
        shape->SetVertex(7, sipmActiveSizeZ / 2.0, sipmActiveSizeRPhi / 2.0);
        return shape;
      };

      auto addReadoutLayer = [&](const char* prefix,
                                 double thickness,
                                 double centerOffset,
                                 TGeoMedium* medium,
                                 Color_t lineColor,
                                 bool useRectangularFootprint) {
        auto* shape = useRectangularFootprint ? makeRectangularFootprint(thickness) : makeFrameFootprint(thickness);
        auto* volume = new TGeoVolume(Form("%s_%d_%d", prefix, rPosId, photTileCount), shape, medium);
        volume->SetLineColor(lineColor);
        volume->SetLineWidth(1);
        const double layerCenterR = photoCenterR + centerOffset * normalRadial;
        const double layerCenterZ = photoCenterZ + centerOffset * normalZ;
        auto* transform = new TGeoCombiTrans(layerCenterR * TMath::Cos(phiRad), layerCenterR * TMath::Sin(phiRad), layerCenterZ, makeProjectiveRotation(prefix));
        motherVolume->AddNode(volume, 1, transform);
      };

      // ------------------------------------------------------------
      // Optional trapezoidal frame
      // ------------------------------------------------------------
      // This is exactly the old photosensor envelope. It is created for
      // reference, but deliberately not added to the geometry.
      photoFrames[photTileCount] = makeFrameFootprint(photThick);
      auto* photoFrameVol = new TGeoVolume(Form("photoFrame_%d_%d", rPosId, photTileCount), photoFrames[photTileCount], medSi);
      photoFrameVol->SetLineColor(kGray + 2);
      photoFrameVol->SetLineWidth(1);
      // Uncomment only when the mechanical frame material/solid geometry should be included.
      // This would be a solid trapezoid and would overlap the sensitive silicon: need for opening
      // motherVolume->AddNode(photoFrameVol, 1, new TGeoCombiTrans(photoCenterR * TMath::Cos(phiRad), photoCenterR * TMath::Sin(phiRad), photoCenterZ, makeProjectiveRotation("photoFrame")));

      // ------------------------------------------------------------
      // True sensitive silicon: centered 17 x 18 cm2 rectangle
      // ------------------------------------------------------------
      // Local X corresponds to the in-plane Z direction after placement.
      // Local Y corresponds to the in-plane r-phi direction.
      // Local Z is the 1 mm thickness direction.
      /*photoTile = new TGeoArb8(photThick / 2.0);
      photoTile->SetVertex(0, sipmActiveSizeZ / 2.0, -sipmActiveSizeRPhi / 2.0);
      photoTile->SetVertex(1, -sipmActiveSizeZ / 2.0, -sipmActiveSizeRPhi / 2.0);
      photoTile->SetVertex(2, -sipmActiveSizeZ / 2.0, sipmActiveSizeRPhi / 2.0);
      photoTile->SetVertex(3, sipmActiveSizeZ / 2.0, sipmActiveSizeRPhi / 2.0);
      photoTile->SetVertex(4, sipmActiveSizeZ / 2.0, -sipmActiveSizeRPhi / 2.0);
      photoTile->SetVertex(5, -sipmActiveSizeZ / 2.0, -sipmActiveSizeRPhi / 2.0);
      photoTile->SetVertex(6, -sipmActiveSizeZ / 2.0, sipmActiveSizeRPhi / 2.0);
      photoTile->SetVertex(7, sipmActiveSizeZ / 2.0, sipmActiveSizeRPhi / 2.0);
      auto* photoTileVol = new TGeoVolume(Form("%s_%d_%d", GeometryTGeo::getRICHSensorPattern(), rPosId, photTileCount), photoTile, medSi);
      photoTileVol->SetLineColor(kRed);
      photoTileVol->SetLineWidth(1);
      auto* rotTransPhoto = new TGeoCombiTrans(photoCenterR * TMath::Cos(phiRad), photoCenterR * TMath::Sin(phiRad), photoCenterZ, makeProjectiveRotation("photoTile"));
      motherVolume->AddNode(photoTileVol, 1, rotTransPhoto);*/

      // Silicone resin in front of SiPMs
      addReadoutLayer("siliconeLayer", siliconeLayerThickness, siliconeCenterOffset, medSilicone, kOrange + 2, useRectangularModules);

      // Active sensitive silicon.
      photoTile = makeRectangularFootprint(activeSiliconThickness);
      auto* photoTileVol = new TGeoVolume(Form("%s_%d_%d", GeometryTGeo::getRICHSensorPattern(), rPosId, photTileCount), photoTile, medSi);
      const double activeSiliconCenterR = photoCenterR + activeSiliconCenterOffset * normalRadial;
      const double activeSiliconCenterZ = photoCenterZ + activeSiliconCenterOffset * normalZ;
      auto* rotTransPhoto = new TGeoCombiTrans(activeSiliconCenterR * TMath::Cos(phiRad), activeSiliconCenterR * TMath::Sin(phiRad), activeSiliconCenterZ, makeProjectiveRotation("photoTile"));
      motherVolume->AddNode(photoTileVol, 1, rotTransPhoto);

      // Passive silicon absorber.
      addReadoutLayer("siliconAbsorber", passiveSiliconThickness, passiveSiliconCenterOffset, medSiAbsorber, kBlue + 1, true);

      // ------------------------------------------------------------
      // Stack behind the SiPM
      // ------------------------------------------------------------
      // Every gap is surface-to-surface. centerOffset is measured from
      // the SiPM center along the outward local normal.
      double outerSurfaceOffset = photThick / 2.0;

      outerSurfaceOffset += gapSiPMToPCB1;
      const double pcb1CenterOffset = outerSurfaceOffset + pcb1Thickness / 2.0;
      addReadoutLayer("pcb1", pcb1Thickness, pcb1CenterOffset, medFR4, kGreen + 1, useRectangularModules);
      outerSurfaceOffset += pcb1Thickness;

      outerSurfaceOffset += gapPCB1ToCoolingPlate;
      const double coolingPlateCenterOffset = outerSurfaceOffset + coolingPlateThickness / 2.0;
      addReadoutLayer("coolingPlate", coolingPlateThickness, coolingPlateCenterOffset, medHTCC, kRed, useRectangularModules);
      outerSurfaceOffset += coolingPlateThickness;

      outerSurfaceOffset += gapCoolingPlateToPCB2;
      const double pcb2CenterOffset = outerSurfaceOffset + pcb2Thickness / 2.0;
      addReadoutLayer("pcb2", pcb2Thickness, pcb2CenterOffset, medFR4, kGreen + 2, useRectangularModules);
      outerSurfaceOffset += pcb2Thickness;

      outerSurfaceOffset += gapPCB2ToPCB3;
      const double pcb3CenterOffset = outerSurfaceOffset + pcb3Thickness / 2.0;
      addReadoutLayer("pcb3", pcb3Thickness, pcb3CenterOffset, medFR4, kGreen + 3, useRectangularModules);

      photTileCount++;
    }
  }

  // Gas sectors (argon) - legacy code, not used in the current geometry, but kept for reference
  /*
  for (auto& gasSector : gasSectors) {
    double separation{(aerDetDistance - radThick - photThick)};
    auto* radiator = radiatorTiles[argSectorsCount];
    auto* photosensor = photoTiles[argSectorsCount];
    gasSector = new TGeoArb8(separation / 2);

    gasSector->SetVertex(0, -photZ / 2, -photYmin / 2);
    gasSector->SetVertex(1, -photZ / 2, photYmin / 2);
    gasSector->SetVertex(2, photZ / 2, photYmax / 2);
    gasSector->SetVertex(3, photZ / 2, -photYmax / 2);
    gasSector->SetVertex(4, -radZ / 2, -radYmin / 2);
    gasSector->SetVertex(5, -radZ / 2, radYmin / 2);
    gasSector->SetVertex(6, radZ / 2, radYmax / 2);
    gasSector->SetVertex(7, radZ / 2, -radYmax / 2);

    TGeoVolume* gasSectorVol = new TGeoVolume(Form("gasSector_%d_%d", rPosId, argSectorsCount), gasSector, medCO2);
    gasSectorVol->SetVisibility(kTRUE);
    gasSectorVol->SetLineColor(kOrange - 8);
    gasSectorVol->SetLineWidth(1);
    auto* rotGas = new TGeoRotation(Form("gasSectorRotation_%d_%d", argSectorsCount, rPosId));
    rotGas->RotateY(-90 - thetaBDeg);
    //rotGas->RotateZ(argSectorsCount * deltaPhiDeg);
    //auto* rotTransGas = new TGeoCombiTrans((radRad0 + TMath::Cos(thetaB) * (separation + radThick) / 2) * TMath::Cos(argSectorsCount * TMath::Pi() / (nTilesPhi / 2)),
    //                                        (radRad0 + TMath::Cos(thetaB) * (separation + radThick) / 2) * TMath::Sin(argSectorsCount * TMath::Pi() / (nTilesPhi / 2)),
    //                                        radRad0 * TMath::Tan(thetaB) + TMath::Sin(thetaB) * (separation + radThick) / 2,
    //                                         rotGas);
    const double gasPhiRad = modulePhiRad(argSectorsCount);
    rotGas->RotateZ(gasPhiRad * 180.0 / TMath::Pi());
    auto* rotTransGas = new TGeoCombiTrans((radRad0 + TMath::Cos(thetaB) * (separation + radThick) / 2.0) * TMath::Cos(gasPhiRad),
                                           (radRad0 + TMath::Cos(thetaB) * (separation + radThick) / 2.0) * TMath::Sin(gasPhiRad),
                                            radRad0 * TMath::Tan(thetaB) + TMath::Sin(thetaB) * (separation + radThick) / 2.0,
                                            rotGas);
    motherVolume->AddNode(gasSectorVol, 1, rotTransGas);
    argSectorsCount++;
  }
  */
}

FWDRich::FWDRich(std::string name,
                 double rMin,
                 double rMax,
                 double zAerogelMin,
                 double dZAerogel,
                 double zArgonMin,
                 double dZArgon,
                 double zSiliconMin,
                 double dZSilicon) : mName{name},
                                     mRmin{rMin},
                                     mRmax{rMax},
                                     mZAerogelMin{zAerogelMin},
                                     mDZAerogel{dZAerogel},
                                     mZArgonMin{zArgonMin},
                                     mDZArgon{dZArgon},
                                     mZSiliconMin{zSiliconMin},
                                     mDZSilicon{dZSilicon}
{
}

BWDRich::BWDRich(std::string name,
                 double rMin,
                 double rMax,
                 double zAerogelMin,
                 double dZAerogel,
                 double zArgonMin,
                 double dZArgon,
                 double zSiliconMin,
                 double dZSilicon) : mName{name},
                                     mRmin{rMin},
                                     mRmax{rMax},
                                     mZAerogelMin{zAerogelMin},
                                     mDZAerogel{dZAerogel},
                                     mZArgonMin{zArgonMin},
                                     mDZArgon{dZArgon},
                                     mZSiliconMin{zSiliconMin},
                                     mDZSilicon{dZSilicon}
{
}

void FWDRich::createFWDRich(TGeoVolume* motherVolume)
{
  TGeoMedium* medAerogel = gGeoManager->GetMedium("RCH_AEROGEL$");
  if (!medAerogel) {
    LOGP(fatal, "RICH: Aerogel medium not found");
  }
  TGeoMedium* medSi = gGeoManager->GetMedium("RCH_SILICON$");
  if (!medSi) {
    LOGP(fatal, "RICH: Silicon medium not found");
  }
  TGeoMedium* medAr = gGeoManager->GetMedium("RCH_ARGON$");
  if (!medAr) {
    LOGP(fatal, "RICH: Argon medium not found");
  }

  // Create the aerogel volume
  TGeoTube* aerogel = new TGeoTube(mRmin, mRmax, mDZAerogel / 2);
  TGeoVolume* aerogelVol = new TGeoVolume(mName.c_str(), aerogel, medAerogel);
  aerogelVol->SetLineColor(kOrange - 8);

  TGeoTranslation* transAerogel = new TGeoTranslation(0, 0, mZAerogelMin + mDZAerogel / 2);
  motherVolume->AddNode(aerogelVol, 1, transAerogel);

  // Create the argon volume
  TGeoTube* argon = new TGeoTube(mRmin, mRmax, mDZArgon / 2);
  TGeoVolume* argonVol = new TGeoVolume(mName.c_str(), argon, medAr);
  argonVol->SetLineColor(kOrange - 9);

  TGeoTranslation* transArgon = new TGeoTranslation(0, 0, mZArgonMin + mDZArgon / 2);
  motherVolume->AddNode(argonVol, 1, transArgon);

  // Create the silicon volume
  TGeoTube* silicon = new TGeoTube(mRmin, mRmax, mDZSilicon / 2);
  TGeoVolume* siliconVol = new TGeoVolume(mName.c_str(), silicon, medSi);
  siliconVol->SetLineColor(kOrange - 8);

  TGeoTranslation* transSilicon = new TGeoTranslation(0, 0, mZSiliconMin + mDZSilicon / 2);
  motherVolume->AddNode(siliconVol, 1, transSilicon);
}

void BWDRich::createBWDRich(TGeoVolume* motherVolume)
{
  TGeoMedium* medAerogel = gGeoManager->GetMedium("RCH_AEROGEL$");
  if (!medAerogel) {
    LOGP(fatal, "RICH: Aerogel medium not found");
  }
  TGeoMedium* medSi = gGeoManager->GetMedium("RCH_SILICON$");
  if (!medSi) {
    LOGP(fatal, "RICH: Silicon medium not found");
  }
  TGeoMedium* medAr = gGeoManager->GetMedium("RCH_ARGON$");
  if (!medAr) {
    LOGP(fatal, "RICH: Argon medium not found");
  }

  // Create the aerogel volume
  TGeoTube* aerogel = new TGeoTube(mRmin, mRmax, mDZAerogel / 2);
  TGeoVolume* aerogelVol = new TGeoVolume(mName.c_str(), aerogel, medAerogel);
  aerogelVol->SetLineColor(kOrange - 8);

  TGeoTranslation* transAerogel = new TGeoTranslation(0, 0, -mZAerogelMin - mDZAerogel / 2);
  motherVolume->AddNode(aerogelVol, 1, transAerogel);

  // Create the argon volume
  TGeoTube* argon = new TGeoTube(mRmin, mRmax, mDZArgon / 2);
  TGeoVolume* argonVol = new TGeoVolume(mName.c_str(), argon, medAr);
  argonVol->SetLineColor(kOrange - 8);

  TGeoTranslation* transArgon = new TGeoTranslation(0, 0, -mZArgonMin - mDZArgon / 2);
  motherVolume->AddNode(argonVol, 1, transArgon);

  // Create the silicon volume
  TGeoTube* silicon = new TGeoTube(mRmin, mRmax, mDZSilicon / 2);
  TGeoVolume* siliconVol = new TGeoVolume(mName.c_str(), silicon, medSi);
  siliconVol->SetLineColor(kOrange - 8);

  TGeoTranslation* transSilicon = new TGeoTranslation(0, 0, -mZSiliconMin - mDZSilicon / 2);
  motherVolume->AddNode(siliconVol, 1, transSilicon);
}

} // namespace rich
} // namespace o2