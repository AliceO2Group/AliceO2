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

/// \file FT3ModuleConstants.h
/// \brief Definition of various constants for tiling the modules of sensors

#ifndef FT3MODULECONSTANTS_H
#define FT3MODULECONSTANTS_H

#include <array>
#include <unordered_map>
#include <vector>
#include <map>
#include <TColor.h>
#include <TMath.h>

namespace o2::ft3::ModuleConstants
{
/* CURRENT STATUS:
 * 25x29mm sensors, 2mm inactive on one side
 * Most granular layout is 2x1 sensors, where the one on the right has the inactive region
 * on the right, and the one on the left has the inactive region on the left.
 * When stacking 2x1 modules, there is a 0.2mm gap between them. By default, we assume this
 * gap to be ABOVE the most recently placed module.
 *
 * |<- 25mm ->||<- 25mm ->|
 * _______________________
 * ------------------------  0.2mm gap above
 * | |        ||        | |
 * | |        ||        | |
 * | |        ||        | |
 * | |        ||        | |  29mm sensor height
 * | |        ||        | |
 * | |        ||        | |
 * ------------------------
 *            ^
 *            |
 *   0.2mm gap in the middle
 */
// First set all layout constants for the rest of the function
const double single_sensor_width = 2.5;
const double single_sensor_height = 2.9;
const double inactive_width = 0.2;
const double sensor2x1_gap = 0.02;     // both between L&R sensors in 2x1, and between two 2x1s
const double stackGap = sensor2x1_gap; // gap between 2xN module stacks

const double active_width = single_sensor_width - inactive_width;
const double active_height = single_sensor_height;

const double sensor2x1_width = 2 * single_sensor_width;
const double sensor2x1_active_width = 2 * active_width;
const double sensor2x1_height = single_sensor_height;
const std::vector<unsigned> kSensorsPerStack = {4, 2, 1};
inline const double getStackHeight(unsigned nSensorsPerStack)
{
  return nSensorsPerStack * sensor2x1_height +
         (nSensorsPerStack - 1) * sensor2x1_gap;
}

// small helper function to get 1-indexed stave ID, counting from the middle outwards,
// with negative IDs on the left and positive IDs on the right
inline const int staveIdxToID(int staveIdx, unsigned nStavesPerDisc)
{
  unsigned nStavesOneSide = nStavesPerDisc / 2;
  bool isRight = staveIdx >= nStavesOneSide;
  return staveIdx - nStavesOneSide + isRight;
}

// material properties
const double siliconThickness = 0.01;
const double copperThickness = 0.006;
const double kaptonThickness = 0.03;
const double epoxyThickness = 0.0012;

const double effectiveCarbonThickness_Stave = 0.02; // foam + shell
const double staveOpeningAngle = 60 * TMath::DegToRad();
const double sinTheta = TMath::Sin(staveOpeningAngle / 2);
const double alpha = TMath::Pi() / 2 - staveOpeningAngle / 2; // bottom angles
const double staveSensorGap = 0.1;                            // 2mm padding on each side when sensor is glued
const double staveTriangleHeight = (sensor2x1_width + 2 * staveSensorGap) / 2.0 / tan(staveOpeningAngle / 2.0);
/*
 * Now describe the offset of every other stave in z to avoid overlaps
 * ______      ______
 * \    /______\    / | <-- z_offsetStave
 *  \  / \    / \  /
 *   \/   \  /   \/
 *         \/
 */
// If midpoint spacing becomes non constant, this becomes a function
// TODO: add some tolerance to avoid overlaps?
inline const double z_offsetStave(double x_midpoint_spacing)
{
  return staveTriangleHeight *
         (2 - x_midpoint_spacing / (sensor2x1_width / 2 + staveSensorGap));
}

/*
 * Materials of the FT3 module.
 *
 * Everything the simulation needs to know about a material lives in the
 * materials map below, keyed by its FT3-local ID: composition, density,
 * transport parameters and the colour its volumes are drawn in.
 * Detector::createMaterials() registers the whole map with the MaterialManager,
 * and FT3Module/FT3Layer look the media up again by MaterialID, so no ID,
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
  {MaterialID::Air, {"AIR$", kWhite, 1.20479e-3f, 0.0f, 0.0f, 4, {12.0107f, 14.0067f, 15.9994f, 39.948f}, {6.0f, 7.0f, 8.0f, 18.0f}, {0.000124f, 0.755267f, 0.231781f, 0.012827f}, passiveTracking}},
  {MaterialID::Silicon, {"SILICON$", kGreen, 2.33f, 9.36f, 999.0f, 0, {28.086f}, {14.0f}, {}, sensitiveTracking}},
  // Copper planes of the end-of-stave cards: X0 = 1.436 cm
  {MaterialID::Copper, {"COPPER$", kOrange, 8.96f, 1.436f, 999.0f, 0, {63.546f}, {29.0f}, {}, passiveTracking}},
  // Kapton: C22 H10 N2 O5, by weight fraction. Also the cooling pipe material.
  {MaterialID::Kapton, {"KAPTON$", kYellow, 1.346f, 0.0f, 0.0f, 4, {12.0107f, 1.00794f, 14.0067f, 15.999f}, {6.0f, 1.0f, 7.0f, 8.0f}, {0.5641f, 0.2564f, 0.0513f, 0.1282f}, passiveTracking}},
  // Carbon fibre: density tuned so X0 ~ 27 cm, as in TRK
  // TODO: Check with Rene the exact type of carbon fiber
  {MaterialID::CarbonFiber, {"CARBONFIBER$", kGray + 1, 1.45f, 27.0f, 999.0f, 0, {12.0107f}, {6.0f}, {}, passiveTracking}},
  // Epoxy: C18 H19 O3, by atom count (negative nComponents)
  {MaterialID::Epoxy, {"EPOXY$", kBlue, 2.186f, 0.0f, 0.0f, -3, {12.0107f, 1.00794f, 15.999f}, {6.0f, 1.0f, 8.0f}, {18.0f, 19.0f, 3.0f}, passiveTracking}},
  // No TRK counterpart; X0 and lambda are left to the transport engine
  {MaterialID::Aluminum, {"ALUMINUM$", kBlack, 2.7f, 0.0f, 0.0f, 0, {26.98f}, {13.0f}, {}, passiveTracking}},
  // Carbon foam core of the disk separation layer
  {MaterialID::Foam, {"FOAM$", kBlack, 0.17f, 0.0f, 0.0f, 0, {12.0107f}, {6.0f}, {}, passiveTracking}},
  // Coolant inside the kapton pipes
  {MaterialID::Water, {"WATER$", kBlue, 1.064f, 0.0f, 0.0f, 0, {18.01528f}, {8.0f}, {}, passiveTracking}}};
// The inactive rim of a sensor is made of silicon as well, but is drawn
// separately so that it can be told apart from the active area.
const int SiInactiveColor = kRed;

// Struct for stave position configuration (varies between ML/OT)
struct StaveConfig {
  const unsigned isML; // whether this config is for ML or OT
  /*
   * Constants for staves are written for both positive
   * and negative x even though they are just mirrored now,
   * because there might be design changes in the future
   * that require a non-mirrored layout, making it easier to
   * change here if so required, even though it looks uglier now.
   *
   * The second element in the mapping pair is whether the stave
   * with a certain ID should be mirrored around the x-axis.
   */
  // map from Stave ID (1-indexed from other documents) to midpoint
  // Do NOT add any zero midpoints, this is taken off separately
  const std::map<int, std::pair<double, bool>>& staveID_to_y_midpoint;
  // lengths of staves, their midpoint, and their face
  const std::vector<double>& y_lengths;
  const std::vector<double>& x_midpoints;
  const double x_midpoint_spacing;
  // whether staves can be placed outside of nominal radii
  const double maxToleranceInner;
  const double maxToleranceOuter;
  // which side of the disc do we place the stave?
  // kSegmentedStave: staggering staves in z (see z_offsetStave)
  // accessed via stave index, NOT stave ID
  const std::vector<bool>& staveOnFront;
};

namespace OT_StavePositions
{
const std::map<int, std::pair<double, bool>> staveID_to_y_midpoint = {
  {-2, {39.0, true}},
  {-1, {41.4, true}},
  {1, {41.4, true}},
  {2, {39.0, true}}};
const std::vector<double> y_lengths = {
  52.8, 66.0, 79.2, 92.4, 99.0, 105.6, 118.8, 118.8,
  128.7, 132.0, 132.0, 138.6, 138.6, 56.1, 52.8,
  52.8, 56.1, 138.6, 138.6, 132.0, 132.0, 128.7,
  118.8, 118.8, 105.6, 99.0, 92.4, 79.2, 66.0, 52.8};
const std::vector<double> x_midpoints = {
  -65.25, -60.75, -56.25, -51.75, -47.25, -42.75, -38.25,       // L
  -33.75, -29.25, -24.75, -20.25, -15.75, -11.25, -6.75, -2.25, // L
  2.25, 6.75, 11.25, 15.75, 20.25, 24.75, 29.25, 33.75,         // R
  38.25, 42.75, 47.25, 51.75, 56.25, 60.75, 65.25               // R
};
const double x_midpoint_spacing = 4.5; // assume constant for now
const double maxToleranceInner = 0.;   // default not allowed inwards
const double maxToleranceOuter = 3.4;  // leave 1mm for layer air encapsulation
const std::vector<bool> staveOnFront =
  {
    1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, // L
    0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0  // R
};
} // namespace OT_StavePositions

namespace ML_StavePositions
{
// Use prelim numbers for now, these will change! TODO
const std::map<int, std::pair<double, bool>> staveID_to_y_midpoint = {
  {-3, {19.1, true}},
  {-2, {21.8, true}},
  {-1, {22.5, true}},
  {1, {22.5, true}},
  {2, {21.8, true}},
  {3, {19.1, true}}};
const std::vector<double> y_lengths = {
  30.5, 44.5, 53.6, 60.0, 64.6, 29.5, 25.8, 25.0,
  25.0, 25.8, 29.5, 64.6, 60.0, 53.6, 44.5, 30.5};
const std::vector<double> x_midpoints = {
  -33.75, -29.25, -24.75, -20.25, -15.75, -11.25, -6.75, -2.25, // L
  2.25, 6.75, 11.25, 15.75, 20.25, 24.75, 29.25, 33.75          // R
};
const double x_midpoint_spacing = 4.5;
const double maxToleranceInner = 0.;  // default not allowed inwards
const double maxToleranceOuter = 3.4; // leave 1mm for layer air encapsulation
const std::vector<bool> staveOnFront =
  {
    1, 0, 1, 0, 1, 0, 1, 0, // L
    1, 0, 1, 0, 1, 0, 1, 0  // R
};
} // namespace ML_StavePositions

// Get stave configuration based on tracker type
inline StaveConfig getStaveConfig(bool isInnerDisk)
{
  if (isInnerDisk) {
    return StaveConfig{
      true, // isML
      ML_StavePositions::staveID_to_y_midpoint,
      ML_StavePositions::y_lengths,
      ML_StavePositions::x_midpoints,
      ML_StavePositions::x_midpoint_spacing,
      ML_StavePositions::maxToleranceInner,
      ML_StavePositions::maxToleranceOuter,
      ML_StavePositions::staveOnFront};
  } else {
    return StaveConfig{
      false, // isML
      OT_StavePositions::staveID_to_y_midpoint,
      OT_StavePositions::y_lengths,
      OT_StavePositions::x_midpoints,
      OT_StavePositions::x_midpoint_spacing,
      OT_StavePositions::maxToleranceInner,
      OT_StavePositions::maxToleranceOuter,
      OT_StavePositions::staveOnFront};
  }
}

} // namespace o2::ft3::ModuleConstants

#endif // FT3MODULECONSTANTS_H