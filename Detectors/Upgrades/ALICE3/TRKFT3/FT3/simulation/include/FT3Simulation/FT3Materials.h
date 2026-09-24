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


/// \file FT3Materials.h
/// \brief Materials of the FT3 detector, and access to the media made from them

#ifndef FT3MATERIALS_H
#define FT3MATERIALS_H

#include <array>
#include <unordered_map>
#include <TColor.h>

class TGeoMedium;

namespace o2::ft3
{
namespace Materials
{
// The name FT3 registers itself under with the MaterialManager. getMedium()
// looks media up under this key, so it has to be the same name the Detector
// hands to DetImpl<Detector>.
constexpr const char* moduleName = "FT3";

/*
 * Materials of the FT3 detector.
 *
 * Everything the simulation needs to know about a material lives in the
 * materials map below, keyed by its FT3-local ID: composition, density,
 * transport parameters and the colour its volumes are drawn in.
 * Detector::createMaterials() registers the whole map with the MaterialManager,
 * and FT3Module/FT3Layer reach the media through getMedium() below, so no ID,
 * density or colour is ever written out by hand.
 */
enum class MaterialID : unsigned {
  Air = 1,
  Silicon,
  Copper,
  Kapton,
  CarbonFiber,
  Epoxy,
  Aluminum,
  Foam,
  Water
};

// Transport parameters of a medium, in the order expected by Detector::Medium()
struct TrackingParams {
  float tmaxfd; // maximum field-induced angular deviation per step, degrees
  float stemax; // maximum step length, cm
  float deemax; // maximum fractional energy loss per step
  float epsil;  // tracking precision, cm
  float stmin;  // minimum step length, cm
};

constexpr TrackingParams sensitiveTracking = {0.1f, 0.0075f, 0.1f, 1.0e-4f, 0.0f};
constexpr TrackingParams passiveTracking = {0.1f, 1.0f, 0.1f, 1.0e-4f, 0.0f};

// Maximum number of elements any of the mixtures below is built from
constexpr unsigned maxMaterialComponents = 4;
using ComponentArray = std::array<float, maxMaterialComponents>;

struct MaterialProperties {
  const char* name;
  int colour;         // ROOT colour every volume made of this material is drawn in
  float density;     // g/cm3
  // Radiation and nuclear interaction length, cm. Only single elements carry
  // them: Mixture() derives both from the composition and takes no such
  // arguments. A non-positive value lets the transport engine compute it.
  float radl;
  float absl;
  int nComponents;   // 0: single element; > 0: mixture by weight; < 0: mixture by atom count
  ComponentArray a;  // mass numbers; only a[0] is used for a single element
  ComponentArray z;  // atomic numbers; only z[0] is used for a single element
  ComponentArray w;  // weight fractions or atom counts; unused for a single element
  TrackingParams tracking;
};

/*
 * Silicon, copper and carbon fibre are shared with TRK and are kept numerically
 * identical to its SILICON$, COPPER$ and CARBONFIBER$ (TRK Detector::createMaterials()),
 * down to the radiation lengths. Kapton, epoxy and aluminium have no TRK
 * counterpart: TRK models the flex as the effective FPC$ mixture instead.
 */
inline const std::unordered_map<MaterialID, MaterialProperties> materials = {
  // Air volumes get their colour set individually where they are built
  {MaterialID::Air, {"Air", kWhite, 1.20479e-3f, 0.0f, 0.0f, 4, {12.0107f, 14.0067f, 15.9994f, 39.948f}, {6.0f, 7.0f, 8.0f, 18.0f}, {0.000124f, 0.755267f, 0.231781f, 0.012827f}, passiveTracking}},
  {MaterialID::Silicon, {"Silicon", kGreen, 2.33f, 9.36f, 999.0f, 0, {28.086f}, {14.0f}, {}, sensitiveTracking}},
  // Copper planes of the end-of-stave cards: X0 = 1.436 cm
  {MaterialID::Copper, {"Copper", kOrange, 8.96f, 1.436f, 999.0f, 0, {63.546f}, {29.0f}, {}, passiveTracking}},
  // Kapton: C22 H10 N2 O5, by weight fraction. Also the cooling pipe material.
  {MaterialID::Kapton, {"Kapton", kYellow, 1.346f, 0.0f, 0.0f, 4, {12.0107f, 1.00794f, 14.0067f, 15.999f}, {6.0f, 1.0f, 7.0f, 8.0f}, {0.5641f, 0.2564f, 0.0513f, 0.1282f}, passiveTracking}},
  // Carbon fibre: density tuned so X0 ~ 27 cm, as in TRK
  // TODO: Check with Rene the exact type of carbon fiber
  {MaterialID::CarbonFiber, {"CarbonFiber", kGray + 1, 1.45f, 27.0f, 999.0f, 0, {12.0107f}, {6.0f}, {}, passiveTracking}},
  // Epoxy: C18 H19 O3, by atom count (negative nComponents)
  {MaterialID::Epoxy, {"Epoxy", kBlue, 2.186f, 0.0f, 0.0f, -3, {12.0107f, 1.00794f, 15.999f}, {6.0f, 1.0f, 8.0f}, {18.0f, 19.0f, 3.0f}, passiveTracking}},
  // No TRK counterpart; X0 and lambda are left to the transport engine
  {MaterialID::Aluminum, {"Aluminum", kBlack, 2.7f, 0.0f, 0.0f, 0, {26.98f}, {13.0f}, {}, passiveTracking}},
  // Carbon foam core of the disk separation layer
  {MaterialID::Foam, {"Foam", kBlack, 0.17f, 0.0f, 0.0f, 0, {12.0107f}, {6.0f}, {}, passiveTracking}},
  // Coolant inside the kapton pipes
  {MaterialID::Water, {"Water", kBlue, 1.064f, 0.0f, 0.0f, 0, {18.01528f}, {8.0f}, {}, passiveTracking}}};
// The inactive rim of a sensor is made of silicon as well, but is drawn
// separately so that it can be told apart from the active area.
const int SiInactiveColor = kRed;
} // namespace Materials

/// Retrieve one of the media registered by Detector::createMaterials().
///
/// A free function rather than a member of Detector: the media live in the
/// MaterialManager singleton keyed by Materials::moduleName, not in the
/// detector object, so the lookup needs no Detector instance.
TGeoMedium* getMedium(Materials::MaterialID id);
} // namespace o2::ft3

#endif // FT3MATERIALS_H
