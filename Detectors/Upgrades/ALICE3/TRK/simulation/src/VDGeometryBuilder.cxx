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

#include <TGeoVolume.h>
#include <TGeoMatrix.h>
#include <TGeoTube.h>
#include <TGeoBBox.h>
#include <TMath.h>
#include <TGeoCompositeShape.h>
#include <TString.h>
#include <DetectorsBase/MaterialManager.h>

#include "TGeoManager.h"

#include "Framework/Logger.h"
#include "TRKBase/GeometryTGeo.h"
#include "TRKSimulation/VDLayer.h"
#include "TRKSimulation/VDSensorRegistry.h"

namespace o2::trk
{

static std::vector<VDSensorDesc> gVDSensors; // stays in this TU only
std::vector<VDSensorDesc>& vdSensorRegistry() { return gVDSensors; }

void clearVDSensorRegistry() { gVDSensors.clear(); }

void registerSensor(const std::string& volName, int petal, VDSensorDesc::Region region, VDSensorDesc::Type type, int idx)
{
  gVDSensors.push_back({volName, petal, region, type, idx});
}

static inline std::string makeSensorName(const std::string& layerName, int layerNumber)
{
  return Form("%s_%s%d", layerName.c_str(), o2::trk::GeometryTGeo::getTRKSensorPattern(), layerNumber);
}

namespace
{

// Config: which volumes count as SOLIDS to subtract from the vacuum volume
inline bool isSolidToCut(const TGeoVolume* v)
{
  const char* nm = v->GetName();
  const char* med = v->GetMedium() ? v->GetMedium()->GetName() : "";
  // silicon sensors (barrel + disks)
  if (med && strcmp(med, "TRK_SILICON$") == 0) {
    return true;
  }
  // walls, sidewalls, cold-plate, service rings (names from your builders)
  if (TString(nm).BeginsWith("VD_InnerWallArc")) {
    return true;
  }
  if (TString(nm).BeginsWith("VD_OuterWallArc")) {
    return true;
  }
  if (TString(nm).BeginsWith("VD_SideWall")) {
    return true;
  }
  if (TString(nm).BeginsWith("VD_InnerWallCyl")) {
    return true;
  }
  if (TString(nm).BeginsWith("VD_OuterWallCyl")) {
    return true;
  }
  if (TString(nm).Contains("_Coldplate")) {
    return true;
  }
  if (TString(nm).BeginsWith("IRIS_Service_Neg")) {
    return true;
  }
  if (TString(nm).BeginsWith("IRIS_Service_Pos_InVac")) {
    return true;
  }
  return false;
}

// Ensure every leaf shape has a stable, informative name
inline const char* ensureShapeName(TGeoVolume* v)
{
  auto* sh = v->GetShape();
  TString nm = sh->GetName();
  if (nm.IsNull() || nm.BeginsWith("TGeo")) {
    TString wanted = TString(v->GetName()) + "_sh";
    // avoid collisions
    int k = 0;
    TString cand = wanted;
    auto* shapes = gGeoManager ? gGeoManager->GetListOfShapes() : nullptr;
    while (shapes && shapes->FindObject(cand)) {
      cand = Form("%s_%d", wanted.Data(), ++k);
    }
    sh->SetName(cand);
    if (shapes && !shapes->FindObject(cand)) {
      shapes->Add(sh);
    }
  }
  return sh->GetName();
}

// Recorder state for the petal-local composite
static TString gPetalSolidsFormula;
static int gLocalTrIdx = 0;

// add "ShapeName:IRIS_LOC_TR_k" to the petal-local formula (no outer rotation)
inline void appendLocalTerm(const char* shapeName, const TGeoHMatrix& H)
{
  auto* ct = new TGeoCombiTrans(H);
  ct->SetName(Form("IRIS_LOC_TR_%d", gLocalTrIdx++));
  ct->RegisterYourself();
  if (!gPetalSolidsFormula.IsNull()) {
    gPetalSolidsFormula += "+";
  }
  gPetalSolidsFormula += TString::Format("%s:%s", shapeName, ct->GetName());
}

// DFS: compose LOCAL transforms only (identity prefix), to capture the petal contents
void traversePetalLocal(TGeoVolume* vol, const TGeoHMatrix& prefix)
{
  auto* nodes = vol->GetNodes();
  if (!nodes) {
    return;
  }
  for (int i = 0; i < nodes->GetEntriesFast(); ++i) {
    auto* node = (TGeoNode*)nodes->At(i);
    auto* childV = node->GetVolume();
    TGeoHMatrix H(prefix);
    if (auto* m = node->GetMatrix()) {
      H.Multiply(m);
    }

    if (isSolidToCut(childV)) {
      const char* shapeName = ensureShapeName(childV);
      appendLocalTerm(shapeName, H);
    }
    traversePetalLocal(childV, H);
  }
}

// Build (once) a petal-local composite containing ONLY solids (walls, silicon, coldplate, services, disks)
inline void buildPetalSolidsComposite(TGeoVolume* petalAsm)
{
  // If it already exists, skip
  if (gGeoManager && gGeoManager->GetListOfShapes() && gGeoManager->GetListOfShapes()->FindObject("IRIS_PETAL_SOLIDSsh")) {
    return;
  }

  gPetalSolidsFormula.Clear();
  gLocalTrIdx = 0;

  TGeoHMatrix I; // identity
  traversePetalLocal(petalAsm, I);

  if (gPetalSolidsFormula.IsNull()) {
    LOGP(error, "IRIS_PETAL_SOLIDSsh formula is empty; did not find solids in petal.");
    return;
  }

  LOGP(info, "IRIS_PETAL_SOLIDSsh formula: {}", gPetalSolidsFormula.Data());
  new TGeoCompositeShape("IRIS_PETAL_SOLIDSsh", gPetalSolidsFormula.Data());
}

// Build the global cutout by rotating the petal-local composite n times with (p+0.5) phase
inline void buildIrisCutoutFromPetalSolid(int nPetals)
{
  auto* shps = gGeoManager->GetListOfShapes();
  auto* base = shps ? dynamic_cast<TGeoShape*>(shps->FindObject("IRIS_PETAL_SOLIDSsh")) : nullptr;
  if (!base) {
    LOGP(error, "IRIS cutout: shape 'IRIS_PETAL_SOLIDSsh' not found.");
    return;
  }

  // IMPORTANT: for nPetals==1, a composite expression like "A:tr" is invalid.
  // Just clone the petal solids shape as the global cutout.
  if (nPetals == 1) {
    // Remove any previous shape with same name if it exists (optional but keeps things clean)
    if (shps->FindObject("IRIS_CUTOUTsh")) {
      // ROOT shape lists are owned by gGeoManager; removing is not always necessary.
      // Keeping it simple: just create a unique name if it already exists.
      LOGP(warning, "IRIS cutout: 'IRIS_CUTOUTsh' already exists; overwriting by clone name reuse may be unsafe.");
    }

    auto* cut = dynamic_cast<TGeoShape*>(base->Clone("IRIS_CUTOUTsh"));
    if (!cut) {
      LOGP(error, "IRIS cutout: failed to clone 'IRIS_PETAL_SOLIDSsh' to 'IRIS_CUTOUTsh'.");
      return;
    }

    LOGP(info, "IRIS_CUTOUTsh created as clone of IRIS_PETAL_SOLIDSsh (nPetals=1).");
    return;
  }

  // nPetals > 1: build union of rotated copies
  TString cutFormula;
  for (int p = 0; p < nPetals; ++p) {
    const double phi = (360.0 / nPetals) * (p + 0.5);
    auto* R = new TGeoRotation();
    R->RotateZ(phi);
    auto* RT = new TGeoCombiTrans(0, 0, 0, R);
    RT->SetName(Form("IRIS_PETAL_ROT_%d", p));
    RT->RegisterYourself();

    if (p) {
      cutFormula += "+";
    }
    cutFormula += Form("IRIS_PETAL_SOLIDSsh:%s", RT->GetName());
  }

  LOGP(info, "IRIS_CUTOUTsh formula: {}", cutFormula.Data());
  auto* cut = new TGeoCompositeShape("IRIS_CUTOUTsh", cutFormula.Data());
  (void)cut;

  // Stronger sanity: ensure it parsed into a boolean node
  auto* cutCheck = dynamic_cast<TGeoCompositeShape*>(shps->FindObject("IRIS_CUTOUTsh"));
  if (!cutCheck || !cutCheck->GetBoolNode()) {
    LOGP(error, "IRIS cutout sanity: IRIS_CUTOUTsh exists but parsing failed (no BoolNode).");
  } else {
    LOGP(info, "IRIS cutout sanity: OK ({} petals).", nPetals);
  }
}

} // namespace

// =================== Specs & constants (ROOT units: cm) ===================
static constexpr double kX2X0 = 0.001f;   // 0.1% X0 per layer
static constexpr double kLenZ_cm = 50.0f; // L0/L1/L2 Z length

// Radii (cm)
static constexpr double rL0_cm = 0.5f; // 5 mm
static constexpr double rL1_cm = 1.2f; // 12 mm
static constexpr double rL2_cm = 2.5f; // 25 mm

// IRIS5 rectangular L0 width (cm)
static constexpr double kL0RectHeight_cm = 0.5f; // 5.0 mm
static constexpr double kL0RectWidth_cm = 0.83f; // 8.3 mm

// Disks radii (cm)
static constexpr double diskRin_cm = 0.5f;  // 5 mm
static constexpr double diskRout_cm = 2.5f; // 25 mm
static const double diskZ_cm[6] = {-34.0f, -30.0f, -26.0f, 26.0f, 30.0f, 34.0f};

// Petal walls specifications (cm)
static constexpr double kPetalZ_cm = 70.0f;          // full wall height
static constexpr double kWallThick_cm = 0.015f;      // 0.15 mm
static constexpr double kInnerWallRadius_cm = 0.48f; // 4.8 mm (ALWAYS cylindrical)
static constexpr double kOuterWallRadius_cm = 3.0f;  // 30 mm (can be changed)
static constexpr double kEps_cm = 1.e-4f;

// Coldplate specs (cm)
static constexpr double kColdplateRadius_cm = 2.6f;     // 26 mm (outer radius)
static constexpr double kColdplateThickness_cm = 0.15f; // 1.5 mm
static constexpr double kColdplateZ_cm = 50.0f;         // full length

// ========== φ-span helpers (gap/arc → degrees) ==========
namespace
{

// Convert a linear gap at radius R into an angular gap (deg)
inline double degFromArc(double arc, double radius)
{
  // arc and radius in the SAME units (cm or mm); result in degrees
  return (radius > 0.f) ? (arc / radius) * TMath::RadToDeg() : 0.f;
}

/**
 * Compute silicon segment φ-span (degrees) inside one petal,
 * when you know the number of petals and the linear gap at a given radius.
 *
 * All of: gap and radius must be in the SAME units (cm or mm).
 * If you use cm everywhere (ROOT default), pass gap_cm and radius_cm.
 */
inline double phiSpanFromGap(int nPetals, double gap, double radius)
{
  if (nPetals <= 0 || radius <= 0.f) {
    return 0.f;
  }
  const double petalPhiDeg = 360.f / nPetals;
  const double phi = petalPhiDeg - degFromArc(gap, radius);
  return phi > 0.f ? phi : 0.f;
}

/**
 * Compute silicon segment φ-span (degrees) from a known arc length at a given radius.
 * arcLen and radius must be in the SAME units (cm or mm).
 */
inline double phiSpanFromArc(double arcLen, double radius)
{
  return (arcLen > 0.f && radius > 0.f) ? degFromArc(arcLen, radius) : 0.f;
}

inline TGeoCombiTrans rotZ(double phiDeg)
{
  auto* r = new TGeoRotation();
  r->RotateZ(static_cast<Double_t>(phiDeg));
  return TGeoCombiTrans(0., 0., 0., r);
}
} // namespace

// ============ Petal sub-builders (LOCAL coords only, no rotation) =========

// Walls: inner cylindrical arc at r=4.8 mm (always), outer arc wall, and two side plates.
static void addPetalWalls(TGeoVolume* petalAsm,
                          int nPetals,
                          double outerRadius_cm = kOuterWallRadius_cm,
                          bool withSideWalls = true,
                          bool fullCylindricalRadialWalls = false)
{
  if (!petalAsm) {
    LOGP(error, "addPetalWalls: petalAsm is null");
    return;
  }

  auto& matmgr = o2::base::MaterialManager::Instance();
  const TGeoMedium* med = matmgr.getTGeoMedium("ALICE3_TRKSERVICES_ALUMINIUM5083");

  if (!med) {
    LOGP(warning, "Petal walls: ALICE3_TRKSERVICES_ALUMINIUM5083$ not found, walls not created.");
    return;
  }

  const double halfZ = 0.5 * kPetalZ_cm;

  // In full-cylinder radial-wall mode we ignore nPetals for the radial walls.
  const double halfPhi = fullCylindricalRadialWalls ? 180.0 : 0.5 * (360.0 / static_cast<double>(nPetals));

  // ---- Inner radial wall ----
  if (fullCylindricalRadialWalls) {
    auto* s = new TGeoTube(static_cast<Double_t>(kInnerWallRadius_cm),
                           static_cast<Double_t>(kInnerWallRadius_cm + kWallThick_cm),
                           static_cast<Double_t>(halfZ));
    auto* v = new TGeoVolume("VD_InnerWallCyl", s, med);
    v->SetLineColor(kGray + 2);
    v->SetTransparency(70);
    petalAsm->AddNode(v, 1);
  } else {
    auto* s = new TGeoTubeSeg(static_cast<Double_t>(kInnerWallRadius_cm),
                              static_cast<Double_t>(kInnerWallRadius_cm + kWallThick_cm),
                              static_cast<Double_t>(halfZ),
                              static_cast<Double_t>(-halfPhi),
                              static_cast<Double_t>(+halfPhi));
    auto* v = new TGeoVolume("VD_InnerWallArc", s, med);
    v->SetLineColor(kGray + 2);
    v->SetTransparency(70);
    petalAsm->AddNode(v, 1);
  }

  // ---- Outer radial wall ----
  if (fullCylindricalRadialWalls) {
    auto* s = new TGeoTube(static_cast<Double_t>(outerRadius_cm),
                           static_cast<Double_t>(outerRadius_cm + kWallThick_cm),
                           static_cast<Double_t>(halfZ));
    auto* v = new TGeoVolume("VD_OuterWallCyl", s, med);
    v->SetLineColor(kGray + 2);
    v->SetTransparency(70);
    petalAsm->AddNode(v, 1);
  } else {
    auto* s = new TGeoTubeSeg(static_cast<Double_t>(outerRadius_cm),
                              static_cast<Double_t>(outerRadius_cm + kWallThick_cm),
                              static_cast<Double_t>(halfZ),
                              static_cast<Double_t>(-halfPhi),
                              static_cast<Double_t>(+halfPhi));
    auto* v = new TGeoVolume("VD_OuterWallArc", s, med);
    v->SetLineColor(kGray + 2);
    v->SetTransparency(70);
    petalAsm->AddNode(v, 1);
  }

  // ---- Side plates (skip in "single petal full cylinders" mode) ----
  if (!withSideWalls) {
    return;
  }

  // ---- Side walls (boxes) at ±halfPhi ----
  const double radialLen = (outerRadius_cm - (kInnerWallRadius_cm + kWallThick_cm));
  auto* sideS = new TGeoBBox(static_cast<Double_t>(0.5f * radialLen),
                             static_cast<Double_t>(0.5f * kWallThick_cm),
                             static_cast<Double_t>(halfZ));
  auto* sideV = new TGeoVolume("VD_SideWall", sideS, med);
  sideV->SetLineColor(kGray + 2);
  sideV->SetTransparency(70);

  for (int sgn : {-1, +1}) {
    const double phi = sgn * halfPhi;
    const double rMid = kInnerWallRadius_cm + kWallThick_cm + 0.5f * radialLen;
    const double rad = static_cast<double>(TMath::DegToRad());
    const double x = rMid * std::cos(phi * rad);
    const double y = rMid * std::sin(phi * rad);
    auto* rot = new TGeoRotation();
    rot->RotateZ(static_cast<Double_t>(phi));
    auto* tr = new TGeoCombiTrans(static_cast<Double_t>(x),
                                  static_cast<Double_t>(y),
                                  0.0, rot);
    petalAsm->AddNode(sideV, (sgn < 0 ? 1 : 2), tr);
  }
}

// Build inner layers (L0..L2). L0 may be rectangular (IRIS5) or cylindrical.
// φ-spans derive from spec gaps/arc; all local placement (no rotation).
static void addBarrelLayers(TGeoVolume* petalAsm, int nPetals, int petalID, bool rectangularL0, bool fullCylinders)
{
  if (!petalAsm) {
    LOGP(error, "addBarrelLayers: petalAsm is null");
    return;
  }

  // Per spec (mm → cm)
  constexpr double gapL0_cm = 0.163f;  // 1.63 mm
  constexpr double gapL1L2_cm = 0.12f; // 1.2 mm
  constexpr double arcL0_cm = 0.6247f; // 6.247 mm

  // φ spans
  const double phiL0_deg = fullCylinders ? 360.0 : phiSpanFromGap(nPetals, gapL0_cm, rL0_cm);
  const double phiL1_deg = fullCylinders ? 360.0 : phiSpanFromGap(nPetals, gapL1L2_cm, rL1_cm);
  const double phiL2_deg = fullCylinders ? 360.0 : phiSpanFromGap(nPetals, gapL1L2_cm, rL2_cm);

  const std::string nameL0 =
    std::string(o2::trk::GeometryTGeo::getTRKPetalPattern()) + std::to_string(petalID) + "_" +
    std::string(o2::trk::GeometryTGeo::getTRKPetalLayerPattern()) + "0";

  if (!fullCylinders && rectangularL0) {
    VDRectangularLayer L0(0,
                          nameL0,
                          kX2X0, kL0RectWidth_cm, kLenZ_cm, kLenZ_cm);

    // Correct translation: move to radius + half width along x
    double x = kL0RectHeight_cm + L0.getChipThickness() / 2.;
    LOGP(info, "Placing rectangular L0 at r={:.3f} cm (half-width={:.3f} cm)", x, 0.5f * kL0RectWidth_cm);
    double y = 0.0;
    double z = 0.0;

    // Correct rotation: rotate 90 degrees around z so long side is horizontal
    auto* rot = new TGeoRotation();
    rot->RotateZ(90.0);

    auto* tr = new TGeoCombiTrans(x, y, z, rot);
    L0.createLayer(petalAsm, tr);
    registerSensor(makeSensorName(nameL0, 0), petalID, VDSensorDesc::Region::Barrel, VDSensorDesc::Type::Plane, /*idx*/ 0);
  } else {
    VDCylindricalLayer L0(0,
                          nameL0,
                          kX2X0, rL0_cm, phiL0_deg, kLenZ_cm, kLenZ_cm);
    L0.createLayer(petalAsm, nullptr);
    registerSensor(makeSensorName(nameL0, 0), petalID, VDSensorDesc::Region::Barrel, VDSensorDesc::Type::Curved, /*idx*/ 0);
  }

  const std::string nameL1 =
    std::string(o2::trk::GeometryTGeo::getTRKPetalPattern()) + std::to_string(petalID) + "_" +
    std::string(o2::trk::GeometryTGeo::getTRKPetalLayerPattern()) + "1";

  VDCylindricalLayer L1(1,
                        nameL1,
                        kX2X0, rL1_cm, phiL1_deg, kLenZ_cm, kLenZ_cm);
  L1.createLayer(petalAsm, nullptr);
  registerSensor(makeSensorName(nameL1, 1), petalID, VDSensorDesc::Region::Barrel, VDSensorDesc::Type::Curved, /*idx*/ 1);

  const std::string nameL2 =
    std::string(o2::trk::GeometryTGeo::getTRKPetalPattern()) + std::to_string(petalID) + "_" +
    std::string(o2::trk::GeometryTGeo::getTRKPetalLayerPattern()) + "2";

  VDCylindricalLayer L2(2,
                        nameL2,
                        kX2X0, rL2_cm, phiL2_deg, kLenZ_cm, kLenZ_cm);
  L2.createLayer(petalAsm, nullptr);
  registerSensor(makeSensorName(nameL2, 2), petalID, VDSensorDesc::Region::Barrel, VDSensorDesc::Type::Curved, /*idx*/ 2);
}

// Build cold plate (cylindrical) in local coordinates, and add it to the petal assembly.
static void addColdPlate(TGeoVolume* petalAsm, int nPetals, int petalId, bool fullCylinders = false)
{
  if (!petalAsm) {
    LOGP(error, "addColdPlate: petalAsm is null");
    return;
  }

  // Resolve medium: prefer provided medium, otherwise try to fetch from geo manager
  const TGeoMedium* med = gGeoManager->GetMedium("ALICE3_TRKSERVICES_CERAMIC");
  if (!med) {
    LOGP(error, "addColdPlate: can't find the medium.");
  }

  // Angular span for one petal (deg)
  constexpr double gapL1L2_cm = 0.12f; // 1.2 mm

  // φ spans
  const double phiSpanColdplate_deg =
    fullCylinders ? 360.0 : phiSpanFromGap(nPetals, gapL1L2_cm, rL2_cm); // L2 gap-defined in normal mode
  const double halfPhiDeg = 0.5 * phiSpanColdplate_deg;
  const double startPhi = -halfPhiDeg;
  const double endPhi = +halfPhiDeg;

  // Build tube segment: inner radius, outer radius = inner + thickness, half-length Z
  auto* shape = new TGeoTubeSeg(static_cast<Double_t>(kColdplateRadius_cm),
                                static_cast<Double_t>(kColdplateRadius_cm + kColdplateThickness_cm),
                                static_cast<Double_t>(0.5 * kColdplateZ_cm),
                                static_cast<Double_t>(startPhi),
                                static_cast<Double_t>(endPhi));

  TString volName = TString::Format("Petal%d_Coldplate", petalId);
  auto* coldVol = new TGeoVolume(volName, shape, med);
  coldVol->SetLineColor(kAzure - 3);
  coldVol->SetTransparency(10);

  // Place in local petal coordinates (no extra transform); keep object alive by allocating shape/volume on heap.
  petalAsm->AddNode(coldVol, 1);

  LOGP(info, "Adding cold plate {} r={:.3f} cm t={:.3f} cm Lz={:.3f} cm φ=[{:.3f}, {:.3f}]",
       volName.Data(), kColdplateRadius_cm, kColdplateThickness_cm, kColdplateZ_cm, startPhi, endPhi);
}

// Add IRIS service module(s) as aluminum annular cylinders placed outside the petals.
// The two modules are placed at z = ±(36 + halfLength).
static void addIRISServiceModules(TGeoVolume* petalAsm, int nPetals)
{
  if (!petalAsm) {
    LOGP(error, "addIRISServiceModules: petalAsm is null");
    return;
  }

  auto* matAl = new TGeoMaterial("ALUMINUM", 26.9815, 13, 2.70);
  const TGeoMedium* med = new TGeoMedium("ALUMINUM", 4, matAl);

  if (!med) {
    LOGP(error, "addIRISServiceModules: ALUMINUM medium not found.");
    return;
  }

  constexpr double radius = 3.2;      // cm (inner radius)
  constexpr double thickness = 0.133; // cm (radial thickness)
  constexpr double halfLength = 19.5; // cm (half-length along Z)
  const double rIn = radius;
  const double rOut = radius + thickness;

  // Petal angular span. If you have an exact half-φ from your walls, use it here.
  const double halfPhi_deg = 0.5 * (360.0 / double(nPetals));

  // Create shape once and reuse
  auto* segSh = new TGeoTubeSeg(
    "IRIS_SERVICE_SEGsh",
    rIn, rOut,
    halfLength,
    -halfPhi_deg, halfPhi_deg);

  // Positive Z module
  TString namePos = "IRIS_Service_Pos";
  auto* volPos = new TGeoVolume(namePos, segSh, med);
  volPos->SetLineColor(kRed + 2);
  volPos->SetTransparency(50);

  // Negative Z module: reuse same shape object, give different name
  TString nameNeg = "IRIS_Service_Neg";
  auto* volNeg = new TGeoVolume(nameNeg, segSh, med);
  volNeg->SetLineColor(kRed + 2);
  volNeg->SetTransparency(50);

  // Translations (heap-allocated so ROOT keeps them)
  const double zpos = 36.0 + halfLength;
  auto* transPos = new TGeoTranslation(0.0, 0.0, static_cast<Double_t>(zpos));
  auto* transNeg = new TGeoTranslation(0.0, 0.0, static_cast<Double_t>(-zpos));

  // Add to mother volume
  petalAsm->AddNode(volPos, 1, transPos);
  petalAsm->AddNode(volNeg, 2, transNeg);

  LOGP(info, "Added IRIS service modules at z = ±{} cm, r=[{}, {}] cm", zpos, rIn, rOut);
}

//     Only the A-side "inside vacuum" piece participates in the cutout.
static void addIRISServiceModulesSegmented(TGeoVolume* petalAsm, int nPetals)
{
  if (!petalAsm) {
    LOGP(error, "addIRISServiceModulesSegmented: petalAsm is null");
    return;
  }

  // --- Service geometry (same as your previous values)
  constexpr double rIn = 3.2;         // cm
  constexpr double thickness = 0.133; // cm
  constexpr double rOut = rIn + thickness;
  constexpr double halfLen = 19.5;      // cm
  constexpr double z0 = 36.0 + halfLen; // 55.5 cm center of +Z service
  const double zMinA = z0 - halfLen;    // 36.0 cm
  const double zMaxA = z0 + halfLen;    // 75.0 cm

  // --- Vacuum vessel window around z∈[-L/2, +L/2] with wall thickness on +Z side
  //     Keep these in sync with TRKServices::createVacuumCompositeShape()
  constexpr double vacuumVesselLength = 76.0;             // cm
  constexpr double vacuumVesselThickness = 0.08;          // cm (0.8 mm)
  const double halfVess = 0.5 * vacuumVesselLength;       // 38.0 cm
  const double gapStart = halfVess;                       // 38.00
  const double gapEnd = halfVess + vacuumVesselThickness; // 38.08

  // --- Petal φ-span (segment)
  const double halfPhi = 0.5 * (360.0 / double(nPetals));

  auto* matAl = new TGeoMaterial("ALUMINUM", 26.9815, 13, 2.70);
  const TGeoMedium* med = new TGeoMedium("ALUMINUM", 4, matAl);

  if (!med) {
    LOGP(error, "addIRISServiceModules: ALUMINUM medium not found.");
    return;
  }

  // =========================
  // C-side (negative Z) whole
  // =========================
  {
    auto* sh = new TGeoTubeSeg(rIn, rOut, halfLen, -halfPhi, +halfPhi);
    auto* vN = new TGeoVolume("IRIS_Service_Neg", sh, med);
    vN->SetLineColor(kRed + 2);
    vN->SetTransparency(55);
    petalAsm->AddNode(vN, 1, new TGeoTranslation(0., 0., -(z0)));
  }

  // =====================================
  // A-side (positive Z): split with a gap
  // =====================================
  // Piece 1 (INSIDE vacuum): z ∈ [zMinA, min(zMaxA, gapStart)]  → goes into cutout
  const double L_inVac = std::max(0.0, std::min(zMaxA, gapStart) - zMinA); // expected ~2.0 cm
  if (L_inVac > 0) {
    const double dz = 0.5 * L_inVac;
    const double zc = zMinA + dz; // center of lower slice, ≈ 37.0 cm
    auto* sh = new TGeoTubeSeg(rIn, rOut, dz, -halfPhi, halfPhi);
    sh->SetName("IRIS_SERVICE_POS_INVACsh");
    auto* vP = new TGeoVolume("IRIS_Service_Pos_InVac", sh, med);
    vP->SetLineColor(kRed + 2);
    vP->SetTransparency(55);
    petalAsm->AddNode(vP, 1, new TGeoTranslation(0., 0., zc));
    LOGP(info, "IRIS A-side (InVac): z=[{:.3f},{:.3f}] cm, len={:.3f} cm",
         zc - dz, zc + dz, 2 * dz);
  } else {
    LOGP(warning, "IRIS A-side (InVac): no overlap with vacuum (L_inVac<=0)");
  }

  // Gap (no material): (gapStart, gapEnd) = (38.00, 38.08)

  // Piece 2 (OUT of vacuum): z ∈ [max(zMinA, gapEnd), zMaxA]  → NOT in cutout
  const double L_outVac = std::max(0.0, zMaxA - std::max(zMinA, gapEnd)); // expected ~36.92 cm
  if (L_outVac > 0) {
    const double dz = 0.5 * L_outVac;
    const double zc = std::max(zMinA, gapEnd) + dz; // center of upper slice
    auto* sh = new TGeoTubeSeg(rIn, rOut, dz, -halfPhi, +halfPhi);
    sh->SetName("IRIS_SERVICE_POS_OUTVACsh");
    auto* vP = new TGeoVolume("IRIS_Service_Pos_OutVac", sh, med);
    vP->SetLineColor(kRed + 1);
    vP->SetTransparency(70);
    petalAsm->AddNode(vP, 2, new TGeoTranslation(0., 0., +zc));
    LOGP(info, "IRIS A-side (OutVac): z=[{:.3f},{:.3f}] cm, len={:.3f} cm",
         zc - dz, zc + dz, 2 * dz);
  } else {
    LOGP(warning, "IRIS A-side (OutVac): no upper piece (L_outVac<=0)");
  }
}

// Build disks in local coords: each disk gets only a local Z translation.
// φ span from gap at rOut.
static void addDisks(TGeoVolume* petalAsm, int nPetals, int petalID, bool fullCylinders)
{

  if (!petalAsm) {
    LOGP(error, "addDisks: petalAsm is null");
    return;
  }

  const double phiDisk_deg = fullCylinders ? 360.0 : phiSpanFromGap(nPetals, 2 * kWallThick_cm, diskRin_cm);

  for (int i = 0; i < 6; ++i) {
    const std::string nameD =
      std::string(o2::trk::GeometryTGeo::getTRKPetalPattern()) + std::to_string(petalID) + "_" +
      std::string(o2::trk::GeometryTGeo::getTRKPetalDiskPattern()) + std::to_string(i);

    VDDiskLayer disk(i,
                     nameD,
                     kX2X0, diskRin_cm, diskRout_cm, phiDisk_deg, diskZ_cm[i]);

    // Local Z placement only
    auto* tr = new TGeoTranslation(0.0, 0.0, static_cast<Double_t>(disk.getZPosition()));
    disk.createLayer(petalAsm, tr);
    registerSensor(makeSensorName(nameD, i), petalID, VDSensorDesc::Region::Disk, VDSensorDesc::Type::Plane, /*idx*/ i);
  }
}

// Add Z end-cap walls to "close" the petal/cylinder volume at zMin and zMax.
// Implemented as thin rings (TGeoTube) with thickness 'capThick_cm' in Z,
// spanning radii [rIn_cm, rOut_cm].
static void addPetalEndCaps(TGeoVolume* petalAsm,
                            int petalId,
                            double rIn_cm,
                            double rOut_cm,
                            double zMin_cm,
                            double zMax_cm,
                            double capThick_cm)
{
  if (!petalAsm) {
    LOGP(error, "addPetalEndCaps: petalAsm is null");
    return;
  }

  auto& matmgr = o2::base::MaterialManager::Instance();
  const TGeoMedium* med =
    matmgr.getTGeoMedium("ALICE3_TRKSERVICES_ALUMINIUM5083");

  if (!med) {
    LOGP(warning,
         "addPetalEndCaps: ALICE3_TRKSERVICES_ALUMINIUM5083 not found, caps not created.");
    return;
  }

  const double halfT = 0.5 * capThick_cm;

  auto* sh = new TGeoTube(static_cast<Double_t>(rIn_cm),
                          static_cast<Double_t>(rOut_cm),
                          static_cast<Double_t>(halfT));

  TString vname = Form("Petal%d_ZCap", petalId);
  auto* v = new TGeoVolume(vname, sh, med);
  v->SetLineColor(kGray + 2);
  v->SetTransparency(70);

  auto* trMin = new TGeoTranslation(0.0, 0.0,
                                    static_cast<Double_t>(zMin_cm + halfT));
  auto* trMax = new TGeoTranslation(0.0, 0.0,
                                    static_cast<Double_t>(zMax_cm - halfT));

  petalAsm->AddNode(v, 1, trMin);
  petalAsm->AddNode(v, 2, trMax);
}

// Build one complete petal assembly (walls + L0..L2 + disks) in LOCAL coords.
static TGeoVolume* buildPetalAssembly(int nPetals,
                                      int petalID,
                                      bool rectangularL0,
                                      bool fullCylinders,
                                      bool withSideWalls)
{
  auto* petalAsm = new TGeoVolumeAssembly(Form("PETAL_%d", petalID));

  // In the special mode: no side walls, but keep radial walls as FULL cylinders.
  addPetalWalls(petalAsm, nPetals, kOuterWallRadius_cm,
                /*withSideWalls=*/withSideWalls,
                /*fullCylindricalRadialWalls=*/fullCylinders);

  addBarrelLayers(petalAsm, nPetals, petalID, rectangularL0, fullCylinders);
  addDisks(petalAsm, nPetals, petalID, fullCylinders);

  addColdPlate(petalAsm, nPetals, petalID, /*fullCylinders=*/false);
  addIRISServiceModulesSegmented(petalAsm, nPetals);

  return petalAsm;
}

static TGeoVolume* buildFullCylAssembly(int petalID, bool withDisks)
{
  // IMPORTANT: keep naming consistent with createIRIS4/5 (PETAL_%d)
  auto* petalAsm = new TGeoVolumeAssembly(Form("PETAL_%d", petalID));

  // Radial walls only: full 360° cylinders, no side plates
  addPetalWalls(petalAsm,
                /*nPetals=*/1,
                /*outerRadius_cm=*/kOuterWallRadius_cm,
                /*withSideWalls=*/false,
                /*fullCylindricalRadialWalls=*/true);

  // --- Z end-cap walls to close the petal in Z ---
  {
    const double zMin = -0.5 * kLenZ_cm;
    const double zMax = +0.5 * kLenZ_cm;
    const double rIn = kInnerWallRadius_cm;
    const double rOut = kOuterWallRadius_cm + kWallThick_cm;

    addPetalEndCaps(petalAsm,
                    petalID,
                    rIn,
                    rOut,
                    zMin,
                    zMax,
                    kWallThick_cm);
  }

  // Full 360° barrel cylinders
  addBarrelLayers(petalAsm,
                  /*nPetals=*/1,
                  /*petalID=*/petalID,
                  /*rectangularL0=*/false,
                  /*fullCylinders=*/true);

  addColdPlate(petalAsm, 1, petalID, /*fullCylinders=*/true);
  addIRISServiceModulesSegmented(petalAsm, /*nPetals=*/1);

  // Optionally add full 360° disks
  if (withDisks) {
    addDisks(petalAsm,
             /*nPetals=*/1,
             /*petalID=*/petalID,
             /*fullCylinders=*/true);
  }

  return petalAsm;
}

// =================== Public entry points ===================

void createIRIS4Geometry(TGeoVolume* motherVolume)
{
  if (!motherVolume) {
    LOGP(error, "createIRIS4Geometry: motherVolume is null");
    return;
  }

  clearVDSensorRegistry();

  constexpr int nPetals = 4;
  for (int p = 0; p < nPetals; ++p) {
    auto* petal = buildPetalAssembly(nPetals, p, /*rectangularL0*/ false,
                                     /*fullCylinders=*/false,
                                     /*withSideWalls=*/true);
    // Build the petal-local solids composite once from the FIRST petal
    if (p == 0) {
      buildPetalSolidsComposite(petal); // <-- captures only SOLIDS in local coords
    }
    const double phiDeg = (360.0 / double(nPetals)) * (double(p) + 0.5);
    auto* R = new TGeoRotation();
    R->RotateZ(phiDeg);
    auto* T = new TGeoCombiTrans(0, 0, 0, R);
    motherVolume->AddNode(petal, p + 1, T);
  }
  buildIrisCutoutFromPetalSolid(nPetals);
}

void createIRIS5Geometry(TGeoVolume* motherVolume)
{
  if (!motherVolume) {
    LOGP(error, "createIRIS5Geometry: motherVolume is null");
    return;
  }

  clearVDSensorRegistry();

  constexpr int nPetals = 4;
  for (int p = 0; p < nPetals; ++p) {
    auto* petal = buildPetalAssembly(nPetals, p, /*rectangularL0*/ true,
                                     /*fullCylinders=*/false,
                                     /*withSideWalls=*/true);
    // Build the petal-local solids composite once from the FIRST petal
    if (p == 0) {
      buildPetalSolidsComposite(petal); // <-- captures only SOLIDS in local coords
    }
    const double phiDeg = (360.0 / double(nPetals)) * (double(p) + 0.5);
    auto* R = new TGeoRotation();
    R->RotateZ(phiDeg);
    auto* T = new TGeoCombiTrans(0, 0, 0, R);
    motherVolume->AddNode(petal, p + 1, T);
  }
  buildIrisCutoutFromPetalSolid(nPetals);
}

void createIRIS4aGeometry(TGeoVolume* motherVolume)
{
  if (!motherVolume) {
    LOGP(error, "createIRIS4aGeometry: motherVolume is null");
    return;
  }

  clearVDSensorRegistry();

  constexpr int nPetals = 3;
  for (int p = 0; p < nPetals; ++p) {
    auto* petal = buildPetalAssembly(nPetals, p, /*rectangularL0*/ false,
                                     /*fullCylinders=*/false,
                                     /*withSideWalls=*/true);
    // Build the petal-local solids composite once from the FIRST petal
    if (p == 0) {
      buildPetalSolidsComposite(petal); // <-- captures only SOLIDS in local coords
    }
    const double phiDeg = (360.0 / double(nPetals)) * (double(p) + 0.5);
    auto* R = new TGeoRotation();
    R->RotateZ(phiDeg);
    auto* T = new TGeoCombiTrans(0, 0, 0, R);
    motherVolume->AddNode(petal, p + 1, T);
  }
  buildIrisCutoutFromPetalSolid(nPetals);
}

void createIRISGeometryFullCyl(TGeoVolume* motherVolume)
{
  if (!motherVolume) {
    LOGP(error, "createIRISGeometryFullCyl: motherVolume is null");
    return;
  }

  clearVDSensorRegistry();

  constexpr int nPetals = 1;
  constexpr int petalID = 0;

  auto* petal = buildFullCylAssembly(petalID, /*withDisks=*/false);
  motherVolume->AddNode(petal, 1, nullptr);

  buildPetalSolidsComposite(petal);
  buildIrisCutoutFromPetalSolid(nPetals);
}

void createIRISGeometryFullCylwithDisks(TGeoVolume* motherVolume)
{
  if (!motherVolume) {
    LOGP(error, "createIRISGeometryFullCylDisks: motherVolume is null");
    return;
  }

  clearVDSensorRegistry();

  constexpr int nPetals = 1;
  constexpr int petalID = 0;

  auto* petal = buildFullCylAssembly(petalID, /*withDisks=*/true);
  motherVolume->AddNode(petal, 1, nullptr);

  // Same cutout pipeline as createIRIS4/5:
  buildPetalSolidsComposite(petal);
  buildIrisCutoutFromPetalSolid(nPetals);
}

void createSinglePetalDebug(TGeoVolume* motherVolume, int petalID, int nPetals, bool rectangularL0)
{
  auto* petal = buildPetalAssembly(nPetals, petalID, rectangularL0, false, true);

  // Optionally rotate the petal for display
  const double phiDeg = (360.f / static_cast<double>(nPetals)) * (static_cast<double>(petalID) + 0.5f);
  auto* R = new TGeoCombiTrans(0, 0, 0, new TGeoRotation("", phiDeg, 0, 0));
  motherVolume->AddNode(petal, 1, R);

  LOGP(info, "Debug: Added Petal{} to {}", petalID, motherVolume->GetName());
}

} // namespace o2::trk
