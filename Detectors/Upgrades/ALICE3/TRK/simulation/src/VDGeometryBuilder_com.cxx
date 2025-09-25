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

#include "TRKSimulation/VDGeometryBuilder.h"

#include <TGeoMatrix.h>
#include <TGeoVolume.h>

#include "Framework/Logger.h"
#include "TRKBase/GeometryTGeo.h"
#include "TRKSimulation/VDLayer.h"

namespace o2::trk
{

// ---------- Shared constants (specs) ----------
static constexpr float kX2X0 = 100.e-4f; // all-silicon
static constexpr float kLenZ_cm = 50.0;  // cylindrical/rectangular layer length

// Radii
static constexpr float rL0_cm = 0.5; // 5 mm
static constexpr float rL1_cm = 1.2; // 12 mm
static constexpr float rL2_cm = 2.5; // 25 mm

// Rectangular L0 width (IRIS5) in cm
static constexpr float kL0RectWidth_cm = 0.83; // 8.3 mm

// Disk geometry
static constexpr float diskRin_cm = 0.5;  // 5 mm
static constexpr float diskRout_cm = 2.5; // 25 mm
static const float diskZ_cm[6] = {-34.0, -30.0, -26.0, 26.0, 30.0, 34.0};

// Petal wall parameters (cm)
static constexpr float kPetalZ_cm = 68.0;          // full petal length
static constexpr float kWallThick_cm = 0.015;      // 0.15 mm
static constexpr float kInnerWallRadius_cm = 0.48; // outer radius of wall arc (example)
static constexpr float kOuterWallRadius_cm = 3.0;  // outer radius of wall arc (example)
static constexpr float kEps_cm = 0.0001;           // tiny clearance

// ---------- Helpers ----------
static TGeoCombiTrans makePetalRotation(float phiDeg)
{
  auto* rot = new TGeoRotation();
  rot->RotateZ(phiDeg);
  return TGeoCombiTrans(0.0, 0.0, 0.0, rot);
}

// Convert a linear gap at radius R into an angular gap (deg)
inline float degFromArc(float arc, float radius)
{
  // arc and radius in the SAME units (cm or mm); result in degrees
  return (arc / radius) * TMath::RadToDeg();
}

/**
 * Compute silicon segment φ-span (degrees) inside one petal,
 * when you know the number of petals and the linear gap at a given radius.
 *
 * All of: gap and radius must be in the SAME units (cm or mm).
 * If you use cm everywhere (ROOT default), pass gap_cm and radius_cm.
 */
inline float phiSpanFromGap(int nPetals, float gap, float radius)
{
  if (nPetals <= 0 || radius <= 0.0)
    return 0.0;
  const float petalPhiDeg = 360.0 / nPetals;
  const float gapDeg = degFromArc(gap, radius);
  const float phiDeg = petalPhiDeg - gapDeg;
  return (phiDeg > 0.0) ? phiDeg : 0.0;
}

/**
 * Compute silicon segment φ-span (degrees) from a known arc length at a given radius.
 * arcLen and radius must be in the SAME units (cm or mm).
 */
inline float phiSpanFromArc(float arcLen, float radius)
{
  if (arcLen <= 0.0 || radius <= 0.0)
    return 0.0;
  return degFromArc(arcLen, radius);
}

static void buildDisksSinglePetal(TGeoVolume* motherVolume, int petalID, int nPetals)
{
  const float phiDisk_deg = phiSpanFromGap(nPetals, 2 * kWallThick_cm, diskRin_cm);
  const float phiHalfDisk = phiDisk_deg / 2.0;

  for (int i = 0; i < 6; ++i) {
    VDDiskLayer disk(
      /*layerNumber */ i,
      /*layerName   */ std::string(GeometryTGeo::getTRKPetalDiskPattern()) + std::to_string(i),
      /*x2x0        */ kX2X0,
      /*rMin        */ diskRin_cm,
      /*rMax        */ diskRout_cm,
      /*phiSpanDeg  */ phiDisk_deg,
      /*zPos        */ diskZ_cm[i]);

    // Place each disk with local Z translation only (no rotation).
    TGeoTranslation tz(0.0, 0.0, disk.getZPosition());
    disk.createLayer(petalAsm, &tz);
  }
}

static void buildCylLayersForPetal(TGeoVolume* mother, int petalIdx, int nPetals, bool rectangularL0)
{
  const double petalPhiSpan = 360.0 / double(nPetals);
  const double rotPhi = petalPhiSpan * (petalIdx + 0.5);
  auto combi = makePetalRotation(rotPhi);

  // ----- L0 -----
  if (rectangularL0) {
    VDRectangularLayer L0(
      /*layerNumber */ 0,
      /*layerName   */ std::string(GeometryTGeo::getTRKLayerPattern()) + "VD_L0",
      /*x2x0        */ kX2X0,
      /*width       */ float(kL0RectWidth_cm),
      /*lengthZ     */ float(kLenZ_cm),
      /*lengthSensZ */ float(kLenZ_cm) // no Z-segmentation now
    );
    L0.createLayer(mother, &combi);
  } else {
    VDCylindricalLayer L0(
      /*layerNumber */ 0,
      /*layerName   */ std::string(GeometryTGeo::getTRKLayerPattern()) + "VD_L0",
      /*x2x0        */ kX2X0,
      /*radius      */ float(rL0_cm),
      /*phiSpanDeg  */ float(petalPhiSpan),
      /*lengthZ     */ float(kLenZ_cm),
      /*lengthSensZ */ float(kLenZ_cm));
    L0.createLayer(mother, &combi);
  }

  // ----- L1 -----
  VDCylindricalLayer L1(
    1, std::string(GeometryTGeo::getTRKLayerPattern()) + "VD_L1", kX2X0,
    float(rL1_cm), float(petalPhiSpan), float(kLenZ_cm), float(kLenZ_cm));
  L1.createLayer(mother, &combi);

  // ----- L2 -----
  VDCylindricalLayer L2(
    2, std::string(GeometryTGeo::getTRKLayerPattern()) + "VD_L2", kX2X0,
    float(rL2_cm), float(petalPhiSpan), float(kLenZ_cm), float(kLenZ_cm));
  L2.createLayer(mother, &combi);

  // ---------- φ-segmentation hooks (future) ----------
  // Spec notes on arc length and gaps are φ-related:
  //  * L0 arc length 6.247 mm with 1.63 mm gaps
  //  * L1/L2 gaps 1.2 mm
  // To implement: compute per-sensor Δφ = arcLen_mm / radius_mm (in radians → degrees),
  // and Δφ_gap = gap_mm / radius_mm. Then tile sensors around the petal span using
  // additional rotations of sensor volumes. Current code builds one continuous segment per petal.
}

// ---------- Public entry points ----------
void createIRIS4Geometry(TGeoVolume* motherVolume)
{
  if (!motherVolume) {
    LOGP(error, "createIRIS4Geometry: motherVolume is null");
    return;
  }
  const int nPetals = 4;
  for (int p = 0; p < nPetals; ++p) {
    buildCylLayersForPetal(motherVolume, p, nPetals, /*rectangularL0*/ false);
    buildDisksForPetal(motherVolume, p, nPetals);
  }
}

void createIRIS5Geometry(TGeoVolume* motherVolume)
{
  if (!motherVolume) {
    LOGP(error, "createIRIS5Geometry: motherVolume is null");
    return;
  }
  const int nPetals = 4;
  for (int p = 0; p < nPetals; ++p) {
    buildCylLayersForPetal(motherVolume, p, nPetals, /*rectangularL0*/ true);
    buildDisk
