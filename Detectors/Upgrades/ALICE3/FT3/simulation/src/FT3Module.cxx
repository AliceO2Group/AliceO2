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

/// \file FT3Module.cxx
/// \brief Implementation of the FT3Module class

#include "FT3Simulation/FT3Module.h"
#include "FT3Base/FT3BaseParam.h"
#include <TGeoManager.h>
#include <TGeoMaterial.h>
#include <TGeoMedium.h>
#include <TGeoBBox.h>
#include <TGeoXtru.h>
#include <TGeoMatrix.h>
#include <TGeoCompositeShape.h>
#include <Framework/Logger.h>
#include <cmath>
#include <iostream>
#include <vector>
#include <set>
#include <algorithm>
#include <utility>

TGeoMaterial* FT3Module::siliconMat = nullptr;
TGeoMedium* FT3Module::siliconMed = nullptr;

TGeoMaterial* FT3Module::copperMat = nullptr;
TGeoMedium* FT3Module::copperMed = nullptr;

TGeoMaterial* FT3Module::kaptonMat = nullptr;
TGeoMedium* FT3Module::kaptonMed = nullptr;

TGeoMaterial* FT3Module::epoxyMat = nullptr;
TGeoMedium* FT3Module::epoxyMed = nullptr;

TGeoMaterial* FT3Module::AluminumMat = nullptr;
TGeoMedium* FT3Module::AluminumMed = nullptr;

TGeoMaterial* FT3Module::carbonFiberMat = nullptr;
TGeoMedium* FT3Module::carbonFiberMed = nullptr;

void FT3Module::initialize_materials()
{
  LOG(debug) << "FT3Module: initialize_materials";
  if (siliconMat) {
    return;
  }

  TGeoManager* geoManager = gGeoManager;

  auto* itsH = new TGeoElement("FT3_H", "Hydrogen", 1, 1.00794);
  auto* itsC = new TGeoElement("FT3_C", "Carbon", 6, 12.0107);
  auto* itsO = new TGeoElement("FT3_O", "Oxygen", 8, 15.994);

  siliconMat = new TGeoMaterial("FT3_Silicon", 28.0855, 14, 2.33);
  siliconMed = new TGeoMedium("FT3_Silicon", 1, siliconMat);

  copperMat = new TGeoMaterial("FT3_Copper", 63.546, 29, 8.96);
  copperMed = new TGeoMedium("FT3_Copper", 2, copperMat);

  kaptonMat = new TGeoMaterial("FT3_Kapton", 13.84, 6.88, 1.346);
  kaptonMed = new TGeoMedium("FT3_Kapton", 3, kaptonMat);

  // TODO: Check with Rene the exact type of carbon fiber
  carbonFiberMat = new TGeoMaterial("FT3_Carbon", 12.0107, 6, 1.8);
  carbonFiberMed = new TGeoMedium("FT3_Carbon", 6, carbonFiberMat);

  // Epoxy: C18 H19 O3
  auto* itsEpoxy = new TGeoMixture("FT3_Epoxy", 3);
  itsEpoxy->AddElement(itsC, 18);
  itsEpoxy->AddElement(itsH, 19);
  itsEpoxy->AddElement(itsO, 3);
  itsEpoxy->SetDensity(2.186);

  epoxyMed = new TGeoMedium("FT3_Epoxy", 4, itsEpoxy);
  epoxyMat = epoxyMed->GetMaterial();

  AluminumMat = new TGeoMaterial("Aluminum", 26.98, 13, 2.7);
  AluminumMed = new TGeoMedium("Aluminum", 5, AluminumMat);
  LOG(debug) << "FT3Module: done initialize_materials";
}

double calculate_y_circle(double x, double radius)
{
  return (x * x < radius * radius) ? std::sqrt(radius * radius - x * x) : 0;
}

std::pair<double, double> calculate_y_range(
  double x_left, double x_right, double Rin, double Rout)
{
  double max_y_abs;
  double min_y_abs;
  /*
   * Have 5 cases:
   * (1) Stave wholly on the left of inner radius
   * (2) Stave wholly on the left, but within inner radius
   * (3) Stave crosses the middle x=0
   * (4) Stave wholly on the right, but within inner radius
   * (5) Stave wholly on the right of inner radius
   */
  if (x_right < -Rin) {
    // Stave is completely on the left of inner radius
    min_y_abs = 0;
    max_y_abs = calculate_y_circle(x_left, Rout);
  } else if (x_left < -Constants::sensor2x1_width) {
    // Stave is completely on the left, but within inner radius
    min_y_abs = calculate_y_circle(x_right, Rin);
    max_y_abs = calculate_y_circle(x_left, Rout);
  } else if (x_left < 0) {
    // Stave crosses the middle x=0
    min_y_abs = Rin;
    // x_right should be > 0, but might have FLP issues, so do abs nonetheless
    max_y_abs = calculate_y_circle(std::max(std::abs(x_left), std::abs(x_right)), Rout);
  } else if (x_left < Rin) {
    // Stave is completely on the right, but within inner radius
    min_y_abs = calculate_y_circle(x_left, Rin);
    max_y_abs = calculate_y_circle(x_right, Rout);
  } else {
    // Stave is completely on the right of inner radius
    min_y_abs = 0.;
    max_y_abs = calculate_y_circle(x_right, Rout);
  }
  return {min_y_abs, max_y_abs};
}

/*
 * This function is a helper function which will pad out the stave with sensors
 * until there is no more space available.
 *
 * Arguments:
 * y_positions: a pair of vectors, where each vector contains pairs of
 *              y position and stack height for the positive and negative y positions respectively.
 *              This argument will be appended with the new sensor positions and stack heights.
 * Rout: the outer radius of the layer
 * Rin: the inner radius of the layer
 * x_left: the x position of the left edge of the sensor to be placed
 * kSensorStack: the number of sensors to be stacked on top of each other
 * tolerance: the tolerance to be subtracted from the maximum y position to avoid
 *            placing sensors too close to the edge. If this is negative, it effectively
 *            means that you can place sensors beyond the nominal disc edge
 * y_start: the y positions to start placing sensors,
 *          for positive and negative y respectively
 */
void FT3Module::fill_stave(PosNegPositionTypes& y_positions, double Rin, double Rout,
                           double x_left, unsigned kSensorStack, PositionRangeType y_ranges,
                           std::pair<double, double>& absAllowedYRange)
{
  // start with upper half of the stave, then mirror to the bottom half
  // add the height of kSensorStack sensors + the gaps in between them
  double sensorStackHeight = Constants::getStackHeight(kSensorStack);
  double sensorAbsStackYShift = sensorStackHeight + Constants::stackGap;

  // in case a big tolerance is given, cut on the given range instead
  double max_sensor_y_abs = std::min(absAllowedYRange.second, y_ranges.first.second);

  double y_top; // top half of the xy grid, y>0
  // either start at given value (adjusted for tolerance), or at last placed sensors
  if (!y_positions.first.empty()) { // sensors already placed
    double previousStackHeight = Constants::getStackHeight(y_positions.first.back().second);
    y_top = y_positions.first.back().first + previousStackHeight + Constants::stackGap;
  } else if (absAllowedYRange.first > 0) {
    // there is a minimum inner value --> start at the max of the two
    y_top = std::max(absAllowedYRange.first, y_ranges.first.first);
  } else {
    // No inner minimum value, start at given value
    y_top = y_ranges.first.first;
  }
  // fill positive y sensor positions
  while ((y_top + sensorStackHeight) <= max_sensor_y_abs) {
    y_positions.first.emplace_back(y_top, kSensorStack);
    y_top += sensorAbsStackYShift;
  }

  // now we do the same for the negative y positions
  // they do not have to be exactly mirrored, hence done separately
  double y_bottom;
  if (!y_positions.second.empty()) {
    // subtract instead to move further down
    double previousStackHeight = Constants::getStackHeight(y_positions.second.back().second);
    y_bottom = y_positions.second.back().first - previousStackHeight - Constants::stackGap;
  } else if (absAllowedYRange.first > 0) {
    // there is a minimum inner value --> start at the min of the two
    y_bottom = std::min(-absAllowedYRange.first, y_ranges.second.first);
  } else {
    // No inner minimum value, start at given value
    y_bottom = y_ranges.second.first;
  }
  // fill in the sensors on negative y
  while ((y_bottom - sensorStackHeight) >= -max_sensor_y_abs) {
    y_positions.second.emplace_back(y_bottom, kSensorStack);
    y_bottom -= sensorAbsStackYShift;
  }
}

/*
 * Create the vertices of the triangles that make up the stave cross section
 *
 * Each array of 3 corresponds to x or z values of the 3 triangle vertices,
 * and the outer array corresponds to which triangle:
 *
 * [x_outer, z_outer, x_inner, z_inner], each of which has three values
 */
std::array<std::array<double, 3>, 4> buildStaveTriangle(int direction)
{
  // Set some constants for readability
  double d = Constants::effectiveCarbonThickness_Stave;
  double H = Constants::staveTriangleHeight;
  /*
   * Inner and outer vertices of the stave cross section triangle
   * all vertices are at y_mid, we simply extend the triangle into y dir.
   * We work in the local coordinate system of the stave, but still
   * call the coordinates x and z for readability.
   *
   * 1. Get all local coordinates of the two triangle vertices
   * 2. Extrude a volume from the subtracted triangle cross section area
   * 3. Rotate the volume around the x-axis since it is by default in xy,
   *    and extruded in z. Rotate by -90 for xz -> xy, otherwise xz -> x(-y)
   * 4. Translate the volume to the given position (arguments)
   *
   */
  std::array<double, 3> xv_inner, xv_outer, zv_inner, zv_outer;
  // calculate the coordinates of the triangle vertices
  // Top/bottom vertex (apex)
  xv_outer[0] = 0;
  zv_outer[0] = (direction == 1) ? -H
                                 : H;
  ;
  // right
  xv_outer[1] = Constants::sensor2x1_width / 2 + Constants::staveSensorGap;
  zv_outer[1] = 0;
  // left
  xv_outer[2] = -xv_outer[1];
  zv_outer[2] = 0;

  // now get inner vertices, shifted inwards by effective carbon thickness
  xv_inner[0] = xv_outer[0];
  double z_shift_inner = d / Constants::sinTheta;
  zv_inner[0] = (direction == 1) ? zv_outer[0] + z_shift_inner
                                 : zv_outer[0] - z_shift_inner;
  // face vertices, first right
  zv_inner[1] = (direction == 1) ? zv_outer[1] - d
                                 : zv_outer[1] + d;
  double x_shift_abs = d / TMath::Tan(Constants::alpha / 2);
  xv_inner[1] = xv_outer[1] - x_shift_abs;
  // left
  zv_inner[2] = zv_inner[1];
  xv_inner[2] = -xv_inner[1];

  return {xv_outer, zv_outer, xv_inner, zv_inner};
}

/*
 * This function creates a carbon fibre volume for the stave,
 * onto which the sensor and its support will be glued.
 */
void FT3Module::addStaveVolume(
  TGeoVolume* motherVolume, std::string volumeName, int direction,
  unsigned* volume_count, double staveLength,
  std::array<std::array<double, 3>, 4> staveTriangles,
  std::pair<double, double>& absAllowedYRange,
  double x_mid, double y_mid, double z_stave_shift_forward)
{
  // The allowed y range is assumed to be non-negative.
  if (absAllowedYRange.first < 0 || absAllowedYRange.second < 0 ||
      absAllowedYRange.first >= absAllowedYRange.second) {
    LOG(error) << "Invalid allowed y range in addStaveVolume(): ("
               << absAllowedYRange.first << ", " << absAllowedYRange.second
               << "). Both values must be non-negative and the first "
               << "value must be less than the second value.";
    return;
  }
  // Set the lower and upper y values of the stave:
  double y_lower = y_mid - staveLength / 2;
  double y_upper = y_mid + staveLength / 2;
  bool splitStave = false;
  if (y_lower > 0) { // This stave is fully above x-axis
    y_lower = std::max(y_lower, absAllowedYRange.first);
    y_upper = std::min(y_upper, absAllowedYRange.second);
  } else if (y_upper < 0) { // stave entirely below x-axis
    y_lower = std::max(y_lower, -absAllowedYRange.second);
    y_upper = std::min(y_upper, -absAllowedYRange.first);
  } else { // Full range stave that goes across x-axis
    // Here we might have to cut the stave up into two pieces
    if (absAllowedYRange.first > 0) {
      // There is a minimum inner value --> Split stave
      splitStave = true;
      y_lower = absAllowedYRange.first;
    } else {
      // regular stave, use full length, but don't forget outer cut
      y_lower = std::max(y_lower, -absAllowedYRange.second);
    }
    y_upper = std::min(y_upper, absAllowedYRange.second);
  }
  double staveLengthToUse = y_upper - y_lower;
  /*
   * create the extruded volumes from z=0 (later y=0 after rotation) to stave length
   * and not from midpoint - staveLength/2 to midpoint + staveLength/2, translate later
   *
   * Note also that we first need to check if the length is allowed given the inner
   * and outer radius of the layer.
   */
  TGeoXtru* staveFull = new TGeoXtru(2);
  staveFull->SetName((volumeName + "_Xtru_outer").c_str());
  staveFull->DefinePolygon(3, staveTriangles[0].data(), staveTriangles[1].data());
  staveFull->DefineSection(0, 0);
  staveFull->DefineSection(1, staveLengthToUse);

  TGeoXtru* staveInner = new TGeoXtru(2);
  staveInner->SetName((volumeName + "_Xtru_inner").c_str());
  staveInner->DefinePolygon(3, staveTriangles[2].data(), staveTriangles[3].data());
  staveInner->DefineSection(0, 0);
  staveInner->DefineSection(1, staveLengthToUse);

  TGeoCompositeShape* staveShape = new TGeoCompositeShape(
    (volumeName + "_shape").c_str(),
    Form("%s - %s", staveFull->GetName(), staveInner->GetName()));
  TGeoVolume* staveVolume = new TGeoVolume(
    (volumeName).c_str(),
    staveShape,
    carbonFiberMed);
  staveVolume->SetLineColor(Constants::carbonFiberColor);
  staveVolume->SetFillColorAlpha(Constants::carbonFiberColor, 0.4);

  TGeoRotation* rot = new TGeoRotation();
  rot->RotateX(-90); // lift from xy plane into xz plane
  /*
   * After rotations the face of the stave lies in the xy-plane,
   * facing downwards for direction == 1 and upwards for direction == 0.
   * We still need to shift it in z to get the right staggered layout.
   * This means moving the staves that must be shifted in the opposite
   * direction they are facing: up for direction 1, and down for direction 0.
   *
   * Unlike a regular node placement, we have to put the stave at its
   * starting point in y, not the midpoint. Hence, if we have the mirror,
   * the starting point is the upper y value, since that is the bottom
   * of the mirrored stave -- by the outer radius
   */
  double z_shift = (direction == 1) ? z_stave_shift_forward : -z_stave_shift_forward;
  TGeoCombiTrans* combiTrans =
    new TGeoCombiTrans(x_mid, y_lower, z_shift, rot);
  motherVolume->AddNode(staveVolume,
                        *volume_count,
                        combiTrans);
  (*volume_count)++;

  // if the stave needs to be split, reuse the same volume on opposite side
  if (splitStave) {
    TGeoCombiTrans* combiTransSplit =
      new TGeoCombiTrans(x_mid, -y_upper, z_shift, rot);
    motherVolume->AddNode(staveVolume,
                          *volume_count,
                          combiTransSplit);
    (*volume_count)++;
  }
}

/*
 * Generic helper function that adds a box at the given position with
 * the given dimensions to the given mother volume, with the given color and name.
 */

void FT3Module::addDetectorVolume(
  TGeoVolume* motherVolume, std::string volumeName, int color,
  unsigned* volume_count, double x_mid, double y_mid, double z_mid,
  double x_half_length, double y_half_length, double z_half_length)
{
  TGeoManager* geoManager = gGeoManager;
  TGeoVolume* volume = geoManager->MakeBox(volumeName.c_str(), siliconMed, x_half_length,
                                           y_half_length, z_half_length);
  volume->SetLineColor(color);
  volume->SetFillColorAlpha(color, 0.4);
  motherVolume->AddNode(
    volume,
    *volume_count,
    new TGeoTranslation( // midpoint of box to add
      x_mid,
      y_mid,
      z_mid) // TGeoTranslation
  );         // addNode
  (*volume_count)++;
}

/*
 * This function adds a glue volume between two element layers,
 * immediately for a whole 2x1 layout, under both the active and inactive region.
 */
void FT3Module::add2x1GlueVolume(
  TGeoVolume* motherVolume, int layerNumber, int direction, unsigned stave_idx,
  unsigned* volume_count, double x_mid, double y_mid, double z_mid,
  std::string element_glued_to)
{
  std::string glue_name = "FT3glue_" + element_glued_to + "_" + std::to_string(layerNumber) + "_" + std::to_string(direction) + "_" + std::to_string(stave_idx) + "_" + std::to_string(*volume_count);
  addDetectorVolume(
    motherVolume, glue_name, Constants::glueColor, volume_count,
    x_mid, y_mid, z_mid,
    Constants::sensor2x1_width / 2, Constants::sensor2x1_height / 2, Constants::epoxyThickness / 2);
}

/*
 * This function adds a copper volume onto which the silicon sensor is glued.
 * As with the glue, this is a whole 2x1 layout volume.
 */
void FT3Module::add2x1CopperVolume(
  TGeoVolume* motherVolume, int layerNumber, int direction, unsigned stave_idx,
  unsigned* volume_count, double x_mid, double y_mid, double z_mid)
{
  std::string copper_name = "FT3Copper_" + std::to_string(layerNumber) + "_" + std::to_string(direction) + "_" + std::to_string(stave_idx) + "_" + std::to_string(*volume_count);
  addDetectorVolume(
    motherVolume, copper_name, Constants::CuColor, volume_count,
    x_mid, y_mid, z_mid,
    Constants::sensor2x1_width / 2, Constants::sensor2x1_height / 2, Constants::copperThickness / 2);
}

/*
 * This function adds a kapton volume behind the copper, which represents the ???
 * As with copper and glue, this is a whole 2x1 layout volume.
 */
void FT3Module::add2x1KaptonVolume(
  TGeoVolume* motherVolume, int layerNumber, int direction, unsigned stave_idx,
  unsigned* volume_count, double x_mid, double y_mid, double z_mid)
{
  std::string kapton_name = "FT3Kapton_" + std::to_string(layerNumber) + "_" + std::to_string(direction) + "_" + std::to_string(stave_idx) + "_" + std::to_string(*volume_count);
  addDetectorVolume(
    motherVolume, kapton_name, Constants::kaptonColor, volume_count,
    x_mid, y_mid, z_mid,
    Constants::sensor2x1_width / 2, Constants::sensor2x1_height / 2, Constants::kaptonThickness / 2);
}

/*
 * This function adds a single sensor (currently 2.5x3.2mm) to the given mother volume
 * at the given (x,y,z) position of the module.
 *
 * Because the sensor has an inactive region of 0.2mm on one side, we also add a
 * separate volume for the inactive region, which will be either on the left or
 * or right dependent on the if the sensor is on the left or right in a 2x1 layout.
 * See FT3Module.h for more details on the layout.
 *
 * Arguments:
 * motherVolume: the volume to which the sensor volume will be added
 * layerNumber: the layer number of the sensor, used for naming
 * direction: the direction of the sensor (forward or backward eta), used for naming
 * x_mid: the x position of the center of the sensor volume
 * y_mid: the y position of the center of the sensor volume
 * z_mid: the z position of the center of the sensor volume
 * isLeft: whether the sensor is on the left or right in the 2x1 layout
 */
void FT3Module::addSingleSensorVolume(
  TGeoVolume* motherVolume, int layerNumber, int direction, unsigned stave_idx,
  unsigned* volume_count, double active_x_mid, double y_mid, double z_mid,
  bool isLeft)
{
  TGeoVolume* sensor;
  TGeoManager* geoManager = gGeoManager;
  // ACTIVE AREA
  std::string sensor_name = "FT3Sensor_" + std::to_string(layerNumber) + "_" + std::to_string(direction) + "_" + std::to_string(stave_idx) + "_" + std::to_string(*volume_count);
  sensor = geoManager->MakeBox(sensor_name.c_str(), siliconMed, Constants::active_width / 2,
                               Constants::single_sensor_height / 2, Constants::siliconThickness / 2);
  sensor->SetLineColor(Constants::SiColor);
  sensor->SetFillColorAlpha(Constants::SiColor, 0.4);
  motherVolume->AddNode(
    sensor,
    *volume_count,
    new TGeoTranslation( // midpoint of box to add
      active_x_mid,
      y_mid,
      z_mid) // TGeoTranslation
  );         // addNode
  (*volume_count)++;
  // INACTIVE STRIP ON LEFT OR RIGHT
  double inactive_x_mid = isLeft ? (active_x_mid - Constants::active_width / 2 - Constants::inactive_width / 2)
                                 : (active_x_mid + Constants::active_width / 2 + Constants::inactive_width / 2);
  std::string sensor_inactive_name =
    "FT3Sensor_Inactive_" + std::to_string(layerNumber) + "_" + std::to_string(direction) + "_" + std::to_string(stave_idx) + "_" + std::to_string(*volume_count);
  sensor = geoManager->MakeBox(sensor_inactive_name.c_str(), siliconMed, Constants::inactive_width / 2,
                               Constants::single_sensor_height / 2, Constants::siliconThickness / 2);
  sensor->SetLineColor(Constants::SiInactiveColor);
  sensor->SetFillColorAlpha(Constants::SiInactiveColor, 0.4);
  motherVolume->AddNode(
    sensor,
    *volume_count,
    new TGeoTranslation( // midpoint of box to add
      inactive_x_mid,
      y_mid,
      z_mid) // TGeoTranslation
  );         // addNode
  (*volume_count)++;
}

void FT3Module::create_layout_staveGeo(double mZ, int layerNumber, int direction,
                                       double Rin, double Rout, double z_offset_local,
                                       const Constants::StaveConfig& staveConfig,
                                       TGeoVolume* motherVolume)
{
  LOG(debug) << "FT3Module: create_layout_staveGeo - Layer "
             << layerNumber << ", Direction " << direction;

  FT3Module::initialize_materials();
  auto& ft3Params = o2::ft3::FT3BaseParam::Instance();

  // First let's define some constants used throughout
  /*
   * we build the volume from the outside in, starting with the silicon,
   * then glue & materials towards the stave. Depending on direction,
   * the distance from the center will be mirrored.
   *
   * | SILICON SENSOR | GLUE | COPPER | KAPTON | GLUE | CARBON STAVE |
   * ----------------------------------------------------------------> z
   *
   * Naturally, this will be mirrored for layers in the backwards direction,
   * such that the face of the sensors always face the interaction region.
   *
   * Currently, we stipulate that the default stave face is at local z=0,
   * that is then shifted by the half air thickness encapsulating the layer
   * to avoid overlaps with the air and services. All offsets are
   * calculated for backward direction (since that is a positive shift),
   * and then flipped for forward. At that point, the innermost/frontmost
   * stave face is at the edge of the air volume, so we shift it back a little
   * to make space for the sensor materials and a slight margin.
   */
  double totalSensorMaterialThickness =
    Constants::epoxyThickness + Constants::kaptonThickness + Constants::copperThickness +
    Constants::epoxyThickness + Constants::siliconThickness;
  double z_offset_to_carbon_face = z_offset_local - totalSensorMaterialThickness - 0.1;
  double z_offset_to_glue_Ka =
    z_offset_to_carbon_face + Constants::epoxyThickness / 2;
  double z_offset_to_kapton =
    z_offset_to_carbon_face + Constants::epoxyThickness +
    Constants::kaptonThickness / 2;
  double z_offset_to_copper =
    z_offset_to_carbon_face + Constants::epoxyThickness +
    Constants::kaptonThickness + Constants::copperThickness / 2;
  double z_offset_to_glue_Si =
    z_offset_to_carbon_face + Constants::epoxyThickness + Constants::kaptonThickness +
    Constants::copperThickness + Constants::epoxyThickness / 2;
  double z_offset_to_silicon =
    z_offset_to_carbon_face + Constants::epoxyThickness +
    Constants::kaptonThickness + Constants::copperThickness +
    Constants::epoxyThickness + Constants::siliconThickness / 2;

  // initialise all y_positions, vector over all staves/columns
  std::vector<PosNegPositionTypes> y_positionsPosNeg;
  unsigned volume_count = 0; // give each subvolume a unique ID
  // stave triangle cross sections are the same for every stave (direction based)
  std::array<std::array<double, 3>, 4> staveTriangles = buildStaveTriangle(direction);
  // Create the stave volumes and fill the y positions where to put sensors on the stave
  for (unsigned i_stave = 0; i_stave < staveConfig.x_midpoints.size(); i_stave++) {
    y_positionsPosNeg.emplace_back(PosNegPositionTypes{PositionTypes{}, PositionTypes{}});
    const int staveID = Constants::staveIdxToID(i_stave, staveConfig.x_midpoints.size());

    double y_midpoint = 0.;
    bool mirrorStaveAroundX = false;
    // default positive and negative starting points has a gap around x-axis for symmetry
    double stave_half_length = staveConfig.y_lengths[i_stave] / 2;
    PositionRangeType y_ranges;
    if (ft3Params.placeSensorInMiddleOfStave) {
      /*
       * We want a sensor to cross over the x-axis for coverage at y=0
       * N.B. not necessarily exactly mirrored, only if stack gap is the same
       * as the gap between sensors in a stack.
       */
      y_ranges = {{-Constants::sensor2x1_height / 2,
                   stave_half_length},
                  {-Constants::sensor2x1_height / 2 - Constants::stackGap,
                   -stave_half_length}};
    } else {
      /*
       * Otherwise have a gap around y=0, so sensors are not placed there.
       * This means the stave is perfectly mirrored around the x-axis.
       */
      y_ranges = {{Constants::stackGap / 2, stave_half_length},
                  {-Constants::stackGap / 2, -stave_half_length}};
    }
    auto y_midpoint_it = staveConfig.staveID_to_y_midpoint.find(staveID);
    if (y_midpoint_it != staveConfig.staveID_to_y_midpoint.end()) {
      // there is a defined midpoint for this stave, use this for starting points
      y_midpoint = y_midpoint_it->second.first; // avoid double map lookup
      mirrorStaveAroundX = y_midpoint_it->second.second;
      y_ranges.first = {y_midpoint - stave_half_length, y_midpoint + stave_half_length};
      y_ranges.second = {-y_midpoint + stave_half_length, -y_midpoint - stave_half_length};
    }

    // Define tolerances for cutting staves and placing sensors
    double tolerance_inner = -1000; // large negative number to allow given numbers
    double tolerance_outer = -1000;
    // cut staves on nominal inner radius if specified
    if (ft3Params.cutStavesOnNominalRadius_inner) {
      tolerance_inner = 0.;
    }
    if (ft3Params.cutStavesOnNominalRadius_outer) {
      tolerance_outer = 0.;
    }

    /*
     * There are three cases in which we want to mirror the stave around the x-axis,
     * which correspond to the stave not going fully from + to - Rout in y.
     *
     * (1) The inner tolerance is 0 (or positive)
     *    a) AND either x_left or x_right lies within the inner radius
     * (2) The inner tolerance is large (allow stave placement as wished)
     *    a) AND the given stave midpoint is above the inner radius
     */
    double x_left = staveConfig.x_midpoints[i_stave] - Constants::sensor2x1_width / 2;
    double x_right = x_left + Constants::sensor2x1_width;
    std::pair<double, double> absAllowedYRange =
      calculate_y_range(x_left, x_right, Rin, Rout);

    /*
     * Shift allowed range by tolerance. Note that both values in the range must
     * be non-negative, and if the inner is not, then set it to 0. This just means
     * that there is no lower limit. The upper limit must however be larger than 0,
     * if it is not, then skip this stave and give a warning.
     */
    absAllowedYRange.first += tolerance_inner;
    absAllowedYRange.second -= tolerance_outer;

    if (absAllowedYRange.first < 0) {
      absAllowedYRange.first = 0;
    }
    if (absAllowedYRange.second <= 0) {
      LOG(warning) << "For stave " << i_stave << " in layer " << layerNumber
                   << " with direction " << direction << ": no space to place sensors after applying tolerances, skipping stave.";
      continue;
    }

    // Get whether the stave is shifted backward or not before creating
    double z_stave_shift_abs = staveConfig.staveOnFront[i_stave] ? 0 : Constants::z_offsetStave(staveConfig.x_midpoint_spacing);
    double z_stave_shift_forward = // move staves more inward to fit in layer volume
      -z_offset_to_carbon_face + z_stave_shift_abs;
    std::string stave_volume_name =
      "Stave_" + std::to_string(i_stave) + "_" + std::to_string(layerNumber) +
      "_" + std::to_string(direction);
    addStaveVolume(
      motherVolume, stave_volume_name, direction, &volume_count,
      staveConfig.y_lengths[i_stave], staveTriangles, absAllowedYRange,
      staveConfig.x_midpoints[i_stave], y_midpoint, z_stave_shift_forward);
    // Now create the mirrored stave
    if (mirrorStaveAroundX) {
      addStaveVolume(
        motherVolume, stave_volume_name + "_mirrored", direction, &volume_count,
        staveConfig.y_lengths[i_stave], staveTriangles, absAllowedYRange,
        staveConfig.x_midpoints[i_stave], -y_midpoint, z_stave_shift_forward);
    }

    // now add the sensor positions on the stave
    for (unsigned i_kSens = 0; i_kSens < Constants::kSensorsPerStack.size(); i_kSens++) {
      fill_stave(y_positionsPosNeg.back(), Rin, Rout, x_left,
                 Constants::kSensorsPerStack[i_kSens], y_ranges,
                 absAllowedYRange);
    }
  }

  // Create volumes for the sensors and the support materials on top of the stave
  for (unsigned i_stave = 0; i_stave < staveConfig.x_midpoints.size(); i_stave++) {
    double x_mid = staveConfig.x_midpoints[i_stave];
    int staveID = Constants::staveIdxToID(i_stave, staveConfig.x_midpoints.size());
    /*
     * Declare an offset multiplier for the z offsets, used for distinguishing
     * sensors facing either forward or backward.
     *
     * In the stave layout, all sensors face inward, and isFront
     * refers to whether a stave is shifted backwards or not. Thus,
     * we decide the offset multiplier only with direction, to
     * keep the face facing inwards.
     */
    bool isFront;
    if (direction == 1) { // direction = 1 is forward
      isFront = staveConfig.staveOnFront[i_stave];
    } else {
      isFront = !(staveConfig.staveOnFront[i_stave]);
    }
    int z_offset_multiplier = (direction == 1) ? -1 : 1;

    // Get whether the stave is shifted for staggering or not
    double z_stave_shift = 0;
    if (!staveConfig.staveOnFront[i_stave]) {
      // in forward direction, shifting backwards means +z shift
      z_stave_shift = (direction == 1) ? Constants::z_offsetStave(staveConfig.x_midpoint_spacing)
                                       : -Constants::z_offsetStave(staveConfig.x_midpoint_spacing);
    }

    for (int y_sign = -1; y_sign < 2; y_sign += 2) {
      // place sensors at positive and negative y
      const auto& positions = (y_sign == 1) ? y_positionsPosNeg[i_stave].first
                                            : y_positionsPosNeg[i_stave].second;
      // define starting midpoint: y = y_start +- distance to middle of sensor
      for (unsigned i_y_pos = 0; i_y_pos < positions.size(); i_y_pos++) {
        double y_mid = positions[i_y_pos].first + y_sign * Constants::sensor2x1_height / 2;
        for (unsigned i_sens = 0; i_sens < positions[i_y_pos].second; i_sens++) {
          TGeoVolume* sensor;
          // ------------ (1) Silicon sensor ------------
          // left single sensor of the 2x1
          double z_mid = z_offset_to_silicon * z_offset_multiplier + z_stave_shift;
          addSingleSensorVolume(
            motherVolume, layerNumber, direction, i_stave, &volume_count,
            x_mid - Constants::active_width / 2, y_mid, z_mid, true);
          // right single sensor of the 2x1
          addSingleSensorVolume(
            motherVolume, layerNumber, direction, i_stave, &volume_count,
            x_mid + Constants::active_width / 2, y_mid, z_mid, false);
          // ------------ (2) Epoxy glue layer between silicon and copper (FPC) ------------
          z_mid = z_offset_to_glue_Si * z_offset_multiplier + z_stave_shift;
          add2x1GlueVolume(
            motherVolume, layerNumber, direction, i_stave, &volume_count,
            x_mid, y_mid, z_mid, "SiCu");
          // ------------ (3) Copper layer (FPC) ------------
          z_mid = z_offset_to_copper * z_offset_multiplier + z_stave_shift;
          add2x1CopperVolume(
            motherVolume, layerNumber, direction, i_stave, &volume_count,
            x_mid, y_mid, z_mid);
          // ------------ (4) Kapton layer (FPC) ------------
          z_mid = z_offset_to_kapton * z_offset_multiplier + z_stave_shift;
          add2x1KaptonVolume(
            motherVolume, layerNumber, direction, i_stave, &volume_count,
            x_mid, y_mid, z_mid);
          // ------------ (5) Epoxy glue layer between stave and Kapton ------------
          z_mid = z_offset_to_glue_Ka * z_offset_multiplier + z_stave_shift;
          add2x1GlueVolume(
            motherVolume, layerNumber, direction, i_stave, &volume_count,
            x_mid, y_mid, z_mid, "CarbonKapton");
          // increment to next sensor: (height + gap of one sensor)
          y_mid += y_sign * (Constants::sensor2x1_height + Constants::sensor2x1_gap);
        } // sensors in stack
      } // for y_sign (writing of positive or negative y positions)
    } // i_y_pos
  } // i_stave
}

void FT3Module::create_layout(double mZ, int layerNumber, int direction, double Rin, double Rout, double overlap, const std::string& face, const std::string& layout_type, TGeoVolume* motherVolume)
{

  LOG(debug) << "FT3Module: create_layout - Layer " << layerNumber << ", Direction " << direction << ", Face " << face;
  TGeoManager* geoManager = gGeoManager;

  FT3Module::initialize_materials();

  // double sensor_width = 2.5;
  // double sensor_height = 9.6;
  // double active_width = 2.3;
  // double active_height = 9.6;

  double sensor_width = 5.0;
  double sensor_height = 9.6;
  double inactive_width = 0.2; // per side
  double active_width = 4.6;
  double active_height = 9.6;

  double silicon_thickness = 0.01;
  double copper_thickness = 0.006;
  double kapton_thickness = 0.03;
  double epoxy_thickness = 0.0012;

  double carbonFiberThickness = 0.01;

  double foamSpacingThickness = 1.0;

  int dist_offset = 0;

  double x_offset;
  double y_offset;

  double z_offset = (face == "front") ? -foamSpacingThickness / 2.0 - carbonFiberThickness : foamSpacingThickness / 2.0 + carbonFiberThickness;

  // offset correction
  if (sensor_height == 3.2 && sensor_width == 2.5) {
    x_offset = 0.8;
    y_offset = 1.5;
  } else if (sensor_height == 19.2 && sensor_width == 5) {
    x_offset = 0.7;
    y_offset = 9;
  } else {
    x_offset = sensor_width / 2;
    y_offset = sensor_height / 2;
  }

  double x_condition_min = 0;
  double x_condition_max = 0;
  double offset_Rin_lower = 0;
  double offset_Rin_upper = 0;
  bool adjust_bottom_y_pos = false;
  bool adjust_bottom_y_neg = false;
  double x_adjust_bottom_y_pos = 0;
  double bottom_y_pos_value = 0;
  double bottom_y_neg_value = 0;

  double Rin_offset = (sensor_height == 19.2) ? 1 : 0;
  double Rout_offset = (sensor_height == 19.2) ? 1 : 0;

  if (Rin == 7 && sensor_height == 9.6 && sensor_width == 5) {
    x_condition_min = -Rin - 2;
    x_condition_max = Rin;
    dist_offset = 2;
    adjust_bottom_y_pos = true;
    adjust_bottom_y_neg = true;
    x_adjust_bottom_y_pos = 3.5;
    bottom_y_pos_value = 3.5;
    bottom_y_neg_value = -3.5;
  } else if (Rin == 5 && sensor_height == 9.6 && sensor_width == 5) {
    x_condition_min = -Rin - 6;
    x_condition_max = Rin;
    adjust_bottom_y_pos = true;
    adjust_bottom_y_neg = true;
    x_adjust_bottom_y_pos = 3.5;
    bottom_y_pos_value = 3.5;
    bottom_y_neg_value = -3.5;
  } else if ((Rin == 5 || Rin == 7) && sensor_height == 19.2) {
    x_condition_min = -Rin - 3;
    x_condition_max = Rin - 0.2;
    dist_offset = 2;
    adjust_bottom_y_pos = false;
    adjust_bottom_y_neg = false;
  } else if (Rin == 5 && sensor_height == 3.2) {
    x_condition_min = -(Rin + 2.6);
    x_condition_max = Rin + 1.5;
    adjust_bottom_y_pos = true;
    adjust_bottom_y_neg = true;
    x_adjust_bottom_y_pos = 3.5;
    bottom_y_pos_value = 3.5;
    bottom_y_neg_value = -3.5;
  } else if (Rin == 7 && sensor_height == 3.2) {
    x_condition_min = -Rin - 1;
    x_condition_max = Rin - 0.2;
    adjust_bottom_y_pos = true;
    adjust_bottom_y_neg = true;
    x_adjust_bottom_y_pos = 3.5;
    bottom_y_pos_value = 3.5;
    bottom_y_neg_value = -3.5;
  } else if (Rin == 5 && sensor_height == 9.6 && sensor_width == 2.5) {
    x_condition_min = -(Rin + 2.6);
    x_condition_max = Rin;
    adjust_bottom_y_pos = true;
    adjust_bottom_y_neg = true;
    x_adjust_bottom_y_pos = 3.5;
    bottom_y_pos_value = 3.5;
    bottom_y_neg_value = -3.5;
  } else if (Rin == 7 && sensor_height == 9.6 && sensor_width == 2.5) {
    x_condition_min = -Rin - 2.6;
    x_condition_max = Rin + 1;
    dist_offset = 2;
    adjust_bottom_y_pos = true;
    adjust_bottom_y_neg = true;
    x_adjust_bottom_y_pos = 5.5;
    bottom_y_pos_value = 3.5;
    bottom_y_neg_value = -3.5;
  } else if (Rin == 10 && sensor_height == 9.6 && sensor_width == 5.0) {
    x_condition_min = -Rin - 4;
    x_condition_max = Rin;
    dist_offset = 2;
    adjust_bottom_y_pos = false;
    adjust_bottom_y_neg = false;
    x_adjust_bottom_y_pos = 3.5;
    bottom_y_pos_value = 3.5;
    bottom_y_neg_value = -3.5;
  } else if (Rin == 20 && sensor_height == 9.6 && sensor_width == 5.0) {
    x_condition_min = -Rin - 4;
    x_condition_max = Rin;
    dist_offset = 2;
    adjust_bottom_y_pos = false;
    adjust_bottom_y_neg = false;
    x_adjust_bottom_y_pos = 3.5;
    bottom_y_pos_value = 3.5;
    bottom_y_neg_value = -3.5;
  } else {
    LOG(warning) << "Different config - to determine offsets needed for " << "Rin = " << Rin << " ; sensor_height = " << sensor_height << " ; sensor_width = " << sensor_width << " layer " << layerNumber;
    x_condition_min = -Rin - sensor_width;
    x_condition_max = Rin;
    adjust_bottom_y_pos = false;
    adjust_bottom_y_neg = false;
  }

  offset_Rin_lower = Rin - Rin_offset;
  offset_Rin_upper = Rout + Rout_offset;

  std::set<std::pair<double, double>> placed_sensors;
  int sensor_count = 0;

  int placementCounter = 0;
  bool justSkipped = false;

  std::vector<double> X_positions;
  std::vector<int> justSkipped1;

  if (sensor_width == 2.5) {
    // logic for placement - x positions with complete overlap
    if (face == "front") {
      X_positions = {-63.4, -60.9, -54.2, -51.7, -45.0, -42.5, -35.8, -33.3, -26.6, -24.1, -17.4, -14.9,
                     -8.2, -5.7, 1.0, 3.5, 10.2, 12.7, 19.4, 21.9, 28.6, 31.1, 37.8, 40.3, 47.0, 49.5,
                     56.2, 58.7, 65.4};
      justSkipped1 = {1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1};
    } else if (face == "back") {
      X_positions = {-65.5, -58.8, -56.3, -49.6, -47.1, -40.4, -37.9, -31.2, -28.7, -22.0, -19.5, -12.8,
                     -10.3, -3.6, -1.1, 5.6, 8.1, 14.8, 17.3, 24.0, 26.5, 33.2, 35.7, 42.4, 44.9,
                     51.6, 54.1, 60.8, 63.3};
      justSkipped1 = {0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0};
    }
  } else {
    if (Rin == 10 || Rin == 20) { // v3 paving, rough attempt
      float overlap = 0.3;
      // NB: these are left edges
      float X_start = -2.0 - 13.5 * (sensor_width - overlap);
      float X_start_pos = 2.0 - 0.5 * (sensor_width - overlap);
      if (face == "back") {
        X_start += (sensor_width - overlap);
        X_start_pos += (sensor_width - overlap);
      }
      while (X_start < -2) {
        X_positions.push_back(X_start);
        justSkipped1.push_back(1);
        X_start += 2 * (sensor_width - overlap);
      }
      while (X_start_pos < Rout + x_offset - sensor_width) {
        X_positions.push_back(X_start_pos);
        justSkipped1.push_back(1);
        X_start_pos += 2 * (sensor_width - overlap);
      }
    } else {
      // filling for sensors with 2x width, each row skipped
      if (face == "front") {
        X_positions = {-63.4, -54.2, -45, -35.8, -26.6, -17.4, -8.2, 1., 10.2, 19.4, 28.6, 37.8, 47., 56.2, 65.4};
        justSkipped1 = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
      } else if (face == "back") {
        X_positions = {-58.8, -49.6, -40.4, -31.2, -22, -12.8, -3.6, 5.6, 14.8, 24, 33.2, 42.4, 51.6, 60.8};
        justSkipped1 = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
      }
    }
  }

  if (layout_type == "rectangular") {

    double x_start = -Rout;
    double x_end = Rout;

    std::vector<double> x_positions;
    for (double x = x_start; x <= x_end; x += sensor_width) {
      x_positions.push_back(x);
    }

    int rowCounter = 0;
    const int rowsToAlternate = 2;

    for (size_t i = 0; i < X_positions.size(); ++i) {

      double x = X_positions[i];
      bool justSkippedValue = justSkipped1[i];

      std::vector<double> y_positions_positive;
      std::vector<double> y_positions_negative;

      for (double y = -Rout - Rin_offset; y <= Rout + Rin_offset; y += sensor_height) {
        std::vector<std::pair<double, double>> corners = {
          {x, y},
          {x + sensor_width, y},
          {x, y + sensor_height},
          {x + sensor_width, y + sensor_height}};

        bool within_bounds = std::all_of(corners.begin(), corners.end(), [&](const std::pair<double, double>& corner) {
          double cx = corner.first;
          double cy = corner.second;
          return (offset_Rin_lower <= std::sqrt(cx * cx + cy * cy) && std::sqrt(cx * cx + cy * cy) <= offset_Rin_upper);
        });

        if (within_bounds) {
          if (y >= 0) {
            y_positions_positive.push_back(y);
          } else {
            y_positions_negative.push_back(y);
          }
        }
      }

      // adjust y positions near inner circle for positive y
      if (x_condition_min <= x && x <= x_condition_max && !y_positions_positive.empty()) {
        double first_y_pos = y_positions_positive.front();
        double last_y_pos = y_positions_positive.back() - sensor_height;
        double top_y_pos = std::min(calculate_y_circle(x, Rout), calculate_y_circle(x + sensor_width, Rout));
        double bottom_y_pos = std::max(calculate_y_circle(x, Rin), calculate_y_circle(x + sensor_width, Rin));
        double top_distance_pos = top_y_pos - last_y_pos;

        if (adjust_bottom_y_pos && x > x_adjust_bottom_y_pos) {
          bottom_y_pos = bottom_y_pos_value;
        }

        double bottom_distance_pos = first_y_pos - bottom_y_pos;

        if (std::abs(top_distance_pos + bottom_distance_pos) >= sensor_height) {
          for (auto& y : y_positions_positive) {
            y -= bottom_distance_pos - 0.2;
          }
          y_positions_positive.push_back(y_positions_positive.back() + sensor_height);
        }
      }

      // adjust y positions near inner circle for negative y
      if (x_condition_min <= x && x <= x_condition_max && !y_positions_negative.empty()) {
        double first_y_neg = y_positions_negative.front();
        double last_y_neg = y_positions_negative.back() + sensor_height;
        double top_y_neg = -std::min(calculate_y_circle(x, Rout), calculate_y_circle(x + sensor_width, Rout));
        double bottom_y_neg = -std::max(calculate_y_circle(x, Rin), calculate_y_circle(x + sensor_width, Rin));
        double top_distance_neg = -(top_y_neg - first_y_neg);

        if (adjust_bottom_y_neg && x > x_adjust_bottom_y_pos) {
          bottom_y_neg = bottom_y_neg_value;
        }

        double bottom_distance_neg = -(last_y_neg - bottom_y_neg);

        top_distance_neg = std::abs(top_distance_neg);
        bottom_distance_neg = std::abs(bottom_distance_neg);
        std::sort(y_positions_negative.begin(), y_positions_negative.end());

        if (std::abs(top_distance_neg + bottom_distance_neg) >= sensor_height) {
          if (sensor_height == 19.2) {
            for (auto& y : y_positions_negative) {
              y -= bottom_distance_neg;
            }
          } else {
            for (auto& y : y_positions_negative) {
              y += bottom_distance_neg - 0.2;
            }
          }
          y_positions_negative.push_back(y_positions_negative.front() - sensor_height);
        }
      }

      // adjust positions for the rest of the disk
      if ((x < x_condition_min || x > x_condition_max) && !y_positions_negative.empty() && !y_positions_positive.empty()) {
        double first_y_neg = y_positions_negative.front();
        double last_y_pos = y_positions_positive.back() + sensor_height;
        double top_y_pos = std::min(calculate_y_circle(x, Rout), calculate_y_circle(x + sensor_width, Rout));
        double bottom_y_pos = -top_y_pos;

        double top_distance_pos = std::abs(top_y_pos - last_y_pos);
        double bottom_distance_pos = std::abs(first_y_neg - bottom_y_pos);

        if (top_distance_pos + bottom_distance_pos >= sensor_height) {
          for (auto& y : y_positions_positive) {
            y += top_distance_pos - 0.2;
          }
          for (auto& y : y_positions_negative) {
            y += top_distance_pos - 0.2;
          }
          double new_y = y_positions_negative.front() - sensor_height;

          if (static_cast<int>(new_y) > static_cast<int>(bottom_y_pos)) {
            y_positions_negative.push_back(new_y);
          }
        }

        // Make symmetric adjustments
        std::sort(y_positions_negative.begin(), y_positions_negative.end());
        std::sort(y_positions_positive.begin(), y_positions_positive.end());

        double first_y_pos = y_positions_negative.front();

        last_y_pos = y_positions_positive.back() + sensor_height;

        top_y_pos = std::min(calculate_y_circle(x, Rout), calculate_y_circle(x + sensor_width, Rout));
        bottom_y_pos = -top_y_pos;
        top_distance_pos = std::abs(top_y_pos - last_y_pos);
        bottom_distance_pos = std::abs(first_y_pos - bottom_y_pos);

        double Lb = (bottom_distance_pos + top_distance_pos) / 2;

        if (top_distance_pos < Lb) {
          double shift = Lb - top_distance_pos;
          for (auto& y : y_positions_negative) {
            y -= shift;
          }
          for (auto& y : y_positions_positive) {
            y -= shift;
          }
        } else if (top_distance_pos > Lb) {
          double shift = top_distance_pos - Lb;
          for (auto& y : y_positions_negative) {
            y += shift;
          }
          for (auto& y : y_positions_positive) {
            y += shift;
          }
        }
      }

      std::vector<double> y_positions = y_positions_positive;
      y_positions.insert(y_positions.end(), y_positions_negative.begin(), y_positions_negative.end());

      for (double y : y_positions) {

        int SiColor;
        double R_material_threshold = 0;

        if (placed_sensors.find({x, y}) == placed_sensors.end()) {
          placed_sensors.insert({x, y});
          TGeoVolume* sensor;

          double inactive_width = (sensor_width - active_width) / 2;
          double left_inactive_x_shift;
          double right_inactive_x_shift;
          double active_x_shift_sensor;

          if (face == "front") {

            double active_x_shift, inactive_x_shift;

            if (justSkippedValue) {
              active_x_shift = x + inactive_width / 2;
              active_x_shift_sensor = active_x_shift + inactive_width;

              inactive_x_shift = x - active_width / 2 + inactive_width / 2;
            } else {
              active_x_shift = x - inactive_width / 2;
              active_x_shift_sensor = active_x_shift - inactive_width;

              inactive_x_shift = x + active_width / 2 - inactive_width / 2;
            }

            double inactive_x_shift_left, inactive_x_shift_right;

            if (sensor_width == 5.0) {

              inactive_x_shift_left = x - sensor_width / 2 + inactive_width;
              inactive_x_shift_right = x + sensor_width / 2;
            }

            std::vector<std::pair<double, double>> corners_shifted = {
              {x, y},
              {x + sensor_width, y},
              {x, y + sensor_height},
              {x + sensor_width, y + sensor_height}};

            bool within_bounds = true;
            for (const auto& corner : corners_shifted) {
              double cx = corner.first;
              double cy = corner.second;
              double dist = std::sqrt(cx * cx + cy * cy);

              if (Rin > dist || dist >= Rout) {
                within_bounds = false;
                break;
              }
            }

            if (within_bounds) {

              double r_squared = (x + x_offset) * (x + x_offset) + (y + y_offset) * (y + y_offset);

              if (r_squared < R_material_threshold * R_material_threshold) {
                silicon_thickness = 0.005;
                copper_thickness = 0.00475;
                kapton_thickness = 0.03;
                epoxy_thickness = 0.0012;

                SiColor = kOrange;
              } else {
                silicon_thickness = 0.01;
                copper_thickness = 0.006;
                kapton_thickness = 0.03;
                epoxy_thickness = 0.0012;

                SiColor = kGreen;
              }

              if (sensor_width == 2.5) {
                // silicon
                std::string sensor_name = "FT3Sensor_front_" + std::to_string(layerNumber) + "_" + std::to_string(direction) + "_" + std::to_string(sensor_count);
                sensor = geoManager->MakeBox(sensor_name.c_str(), siliconMed, active_width / 2, active_height / 2, silicon_thickness / 2);
                sensor->SetLineColor(SiColor);
                sensor->SetFillColorAlpha(SiColor, 0.4);
                motherVolume->AddNode(sensor, sensor_count++, new TGeoTranslation(active_x_shift_sensor + x_offset, y + y_offset, mZ + z_offset - epoxy_thickness - kapton_thickness - copper_thickness - epoxy_thickness - silicon_thickness / 2));

                std::string inactive_name = "FT3inactive_front_" + std::to_string(layerNumber) + "_" + std::to_string(direction) + "_" + std::to_string(sensor_count);
                sensor = geoManager->MakeBox(inactive_name.c_str(), siliconMed, (sensor_width - active_width) / 2, sensor_height / 2, silicon_thickness / 2);
                sensor->SetLineColor(kRed);
                sensor->SetFillColorAlpha(kRed, 1.0);
                motherVolume->AddNode(sensor, sensor_count++, new TGeoTranslation(x_offset + inactive_x_shift, y + y_offset, mZ + z_offset - epoxy_thickness - kapton_thickness - copper_thickness - epoxy_thickness - silicon_thickness / 2));

              } else {

                std::string sensor_name = "FT3Sensor_front_" + std::to_string(layerNumber) + "_" + std::to_string(direction) + "_" + std::to_string(sensor_count);
                sensor = geoManager->MakeBox(sensor_name.c_str(), siliconMed, active_width / 2, sensor_height / 2, silicon_thickness / 2);
                sensor->SetLineColor(SiColor);
                sensor->SetFillColorAlpha(SiColor, 0.4);
                motherVolume->AddNode(sensor, sensor_count++, new TGeoTranslation(x_offset + x + inactive_width / 2, y + y_offset, mZ + z_offset - epoxy_thickness - kapton_thickness - copper_thickness - epoxy_thickness - silicon_thickness / 2));

                std::string inactive_name_left = "FT3inactive_left_front_" + std::to_string(layerNumber) + "_" + std::to_string(direction) + "_" + std::to_string(sensor_count);
                sensor = geoManager->MakeBox(inactive_name_left.c_str(), siliconMed, inactive_width / 2, sensor_height / 2, silicon_thickness / 2);
                sensor->SetLineColor(kRed);
                sensor->SetFillColorAlpha(kRed, 1.0);
                motherVolume->AddNode(sensor, sensor_count++, new TGeoTranslation(x_offset + inactive_x_shift_left, y + y_offset, mZ + z_offset - epoxy_thickness - kapton_thickness - copper_thickness - epoxy_thickness - silicon_thickness / 2));

                std::string inactive_name_right = "FT3inactive_right_front_" + std::to_string(layerNumber) + "_" + std::to_string(direction) + "_" + std::to_string(sensor_count);
                sensor = geoManager->MakeBox(inactive_name_right.c_str(), siliconMed, inactive_width / 2, sensor_height / 2, silicon_thickness / 2);
                sensor->SetLineColor(kRed);
                sensor->SetFillColorAlpha(kRed, 1.0);
                motherVolume->AddNode(sensor, sensor_count++, new TGeoTranslation(x_offset + inactive_x_shift_right, y + y_offset, mZ + z_offset - epoxy_thickness - kapton_thickness - copper_thickness - epoxy_thickness - silicon_thickness / 2));
              }

              // silicon-to-FPC epoxy glue
              std::string glue_up_name = "FT3glue_up_front_" + std::to_string(layerNumber) + "_" + std::to_string(direction) + "_" + std::to_string(sensor_count);
              sensor = geoManager->MakeBox(glue_up_name.c_str(), epoxyMed, sensor_width / 2, sensor_height / 2, epoxy_thickness / 2);
              sensor->SetLineColor(kBlue);
              sensor->SetFillColorAlpha(kBlue, 1.0);
              motherVolume->AddNode(sensor, sensor_count++, new TGeoTranslation(x_offset + active_x_shift, y + y_offset, mZ + z_offset - epoxy_thickness - kapton_thickness - copper_thickness - epoxy_thickness / 2));

              if (r_squared < R_material_threshold * R_material_threshold) {
                std::string alu_name = "FT3aluminum_front_" + std::to_string(layerNumber) + "_" + std::to_string(direction) + "_" + std::to_string(sensor_count);
                sensor = geoManager->MakeBox(alu_name.c_str(), AluminumMed, sensor_width / 2, sensor_height / 2, copper_thickness / 2);
                sensor->SetLineColor(kBlack);
                sensor->SetFillColorAlpha(kBlack, 0.4);
                motherVolume->AddNode(sensor, sensor_count++, new TGeoTranslation(active_x_shift + x_offset, y + y_offset, mZ + z_offset - epoxy_thickness - kapton_thickness - copper_thickness / 2));

              } else {
                std::string copper_name = "FT3copper_front_" + std::to_string(layerNumber) + "_" + std::to_string(direction) + "_" + std::to_string(sensor_count);
                sensor = geoManager->MakeBox(copper_name.c_str(), copperMed, sensor_width / 2, sensor_height / 2, copper_thickness / 2);
                sensor->SetLineColor(kBlack);
                sensor->SetFillColorAlpha(kBlack, 0.4);
                motherVolume->AddNode(sensor, sensor_count++, new TGeoTranslation(active_x_shift + x_offset, y + y_offset, mZ + z_offset - epoxy_thickness - kapton_thickness - copper_thickness / 2));
              }

              // kapton
              std::string fpc_name = "FT3fpc_front_" + std::to_string(layerNumber) + "_" + std::to_string(direction) + "_" + std::to_string(sensor_count);
              sensor = geoManager->MakeBox(fpc_name.c_str(), kaptonMed, sensor_width / 2, sensor_height / 2, kapton_thickness / 2);
              sensor->SetLineColor(kGreen);
              sensor->SetFillColorAlpha(kGreen, 0.4);
              motherVolume->AddNode(sensor, sensor_count++, new TGeoTranslation(active_x_shift + x_offset, y + y_offset, mZ + z_offset - epoxy_thickness - kapton_thickness / 2));

              // FPC-to-support epoxy glue
              std::string glue_down_name = "FT3glue_down_front_" + std::to_string(layerNumber) + "_" + std::to_string(direction) + "_" + std::to_string(sensor_count);
              sensor = geoManager->MakeBox(glue_down_name.c_str(), epoxyMed, sensor_width / 2, sensor_height / 2, epoxy_thickness / 2);
              sensor->SetLineColor(kBlue);
              sensor->SetFillColorAlpha(kBlue, 1.0);
              motherVolume->AddNode(sensor, sensor_count++, new TGeoTranslation(x_offset + active_x_shift, y + y_offset, mZ + z_offset - epoxy_thickness / 2));
            }
          } else {
            double x_shifted = x;
            double inactive_x_shift, active_x_shift;
            double active_x_shift_sensor;

            if (justSkippedValue) {
              active_x_shift = x + inactive_width / 2;
              active_x_shift_sensor = active_x_shift + inactive_width;

              inactive_x_shift = x - active_width / 2 + inactive_width / 2;
            } else {
              active_x_shift = x - inactive_width / 2;
              active_x_shift_sensor = active_x_shift - inactive_width;

              inactive_x_shift = x + active_width / 2 - inactive_width / 2;
            }

            double inactive_x_shift_left, inactive_x_shift_right;

            if (sensor_width == 5.0) {

              inactive_x_shift_left = x - sensor_width / 2 + inactive_width;
              inactive_x_shift_right = x + sensor_width / 2;
            }

            std::vector<std::pair<double, double>> corners_shifted = {
              {x_shifted, y},
              {x_shifted + sensor_width, y},
              {x_shifted, y + sensor_height},
              {x_shifted + sensor_width, y + sensor_height}};

            bool within_bounds = true;
            for (const auto& corner : corners_shifted) {
              double cx = corner.first;
              double cy = corner.second;
              double dist = std::sqrt(cx * cx + cy * cy);

              if (Rin > dist + dist_offset || dist >= Rout) {
                within_bounds = false;
                break;
              }
            }

            if (within_bounds) {

              double r_squared = (x + x_offset) * (x + x_offset) + (y + y_offset) * (y + y_offset);

              if (r_squared < R_material_threshold * R_material_threshold) {
                silicon_thickness = 0.005;
                copper_thickness = 0.00475; // thinner -> + replaced by alu
                kapton_thickness = 0.03;
                epoxy_thickness = 0.0006;

                SiColor = kOrange;
              } else {
                silicon_thickness = 0.01;
                copper_thickness = 0.006;
                kapton_thickness = 0.03;
                epoxy_thickness = 0.0012;

                SiColor = kGreen;
              }

              // FPC-to-support epoxy glue
              std::string glue_down_name = "FT3glue_down_back_" + std::to_string(layerNumber) + "_" + std::to_string(direction) + "_" + std::to_string(sensor_count);
              sensor = geoManager->MakeBox(glue_down_name.c_str(), epoxyMed, sensor_width / 2, sensor_height / 2, epoxy_thickness / 2);
              sensor->SetLineColor(kBlue);
              sensor->SetFillColorAlpha(kBlue, 1.0);
              motherVolume->AddNode(sensor, sensor_count++, new TGeoTranslation(x_offset + active_x_shift, y + y_offset, mZ + z_offset + epoxy_thickness / 2));

              // Kapton
              std::string fpc_name = "FT3fpc_back_" + std::to_string(layerNumber) + "_" + std::to_string(direction) + "_" + std::to_string(sensor_count);
              sensor = geoManager->MakeBox(fpc_name.c_str(), kaptonMed, sensor_width / 2, sensor_height / 2, kapton_thickness / 2);
              sensor->SetLineColor(kGreen);
              sensor->SetFillColorAlpha(kGreen, 0.4);
              motherVolume->AddNode(sensor, sensor_count++, new TGeoTranslation(active_x_shift + x_offset, y + y_offset, mZ + z_offset + epoxy_thickness + kapton_thickness / 2));

              if (r_squared < R_material_threshold * R_material_threshold) {
                // replace copper with alu
                std::string alu_name = "FT3aluminum_back_" + std::to_string(layerNumber) + "_" + std::to_string(direction) + "_" + std::to_string(sensor_count);
                sensor = geoManager->MakeBox(alu_name.c_str(), AluminumMed, sensor_width / 2, sensor_height / 2, copper_thickness / 2);
                sensor->SetLineColor(kBlack);
                sensor->SetFillColorAlpha(kBlack, 0.4);
                motherVolume->AddNode(sensor, sensor_count++, new TGeoTranslation(active_x_shift + x_offset, y + y_offset, mZ + z_offset + epoxy_thickness + kapton_thickness + copper_thickness / 2));

              } else {
                std::string copper_name = "FT3copper_back_" + std::to_string(layerNumber) + "_" + std::to_string(direction) + "_" + std::to_string(sensor_count);
                sensor = geoManager->MakeBox(copper_name.c_str(), copperMed, sensor_width / 2, sensor_height / 2, copper_thickness / 2);
                sensor->SetLineColor(kBlack);
                sensor->SetFillColorAlpha(kBlack, 0.4);
                motherVolume->AddNode(sensor, sensor_count++, new TGeoTranslation(active_x_shift + x_offset, y + y_offset, mZ + z_offset + epoxy_thickness + kapton_thickness + copper_thickness / 2));
              }

              // silicon-to-FPC epoxy glue
              std::string glue_up_name = "FT3glue_up_back_" + std::to_string(layerNumber) + "_" + std::to_string(direction) + "_" + std::to_string(sensor_count);
              sensor = geoManager->MakeBox(glue_up_name.c_str(), epoxyMed, sensor_width / 2, sensor_height / 2, epoxy_thickness / 2);
              sensor->SetLineColor(kBlue);
              sensor->SetFillColorAlpha(kBlue, 1.0);
              motherVolume->AddNode(sensor, sensor_count++, new TGeoTranslation(x_offset + active_x_shift, y + y_offset, mZ + z_offset + epoxy_thickness + kapton_thickness + copper_thickness + epoxy_thickness / 2));

              if (sensor_width == 2.5) {

                std::string sensor_name = "FT3Sensor_back_" + std::to_string(layerNumber) + "_" + std::to_string(direction) + "_" + std::to_string(sensor_count);
                sensor = geoManager->MakeBox(sensor_name.c_str(), siliconMed, active_width / 2, active_height / 2, silicon_thickness / 2);
                sensor->SetLineColor(SiColor);
                sensor->SetFillColorAlpha(SiColor, 0.4);
                motherVolume->AddNode(sensor, sensor_count++, new TGeoTranslation(active_x_shift_sensor + x_offset, y + y_offset, mZ + z_offset + epoxy_thickness + kapton_thickness + copper_thickness + epoxy_thickness + silicon_thickness / 2));

                std::string inactive_name = "FT3inactive_back_" + std::to_string(layerNumber) + "_" + std::to_string(direction) + "_" + std::to_string(sensor_count);
                sensor = geoManager->MakeBox(inactive_name.c_str(), siliconMed, (sensor_width - active_width) / 2, sensor_height / 2, silicon_thickness / 2);
                sensor->SetLineColor(kRed);
                sensor->SetFillColorAlpha(kRed, 1.0);
                motherVolume->AddNode(sensor, sensor_count++, new TGeoTranslation(x_offset + inactive_x_shift, y + y_offset, mZ + z_offset + epoxy_thickness + kapton_thickness + copper_thickness + epoxy_thickness + silicon_thickness / 2));

              } else {
                // active (4.6 cm centered)
                std::string sensor_name = "FT3Sensor_back_" + std::to_string(layerNumber) + "_" + std::to_string(direction) + "_" + std::to_string(sensor_count);
                sensor = geoManager->MakeBox(sensor_name.c_str(), siliconMed, active_width / 2, sensor_height / 2, silicon_thickness / 2);
                sensor->SetLineColor(SiColor);
                sensor->SetFillColorAlpha(SiColor, 0.4);
                motherVolume->AddNode(sensor, sensor_count++, new TGeoTranslation(x_offset + x_shifted + inactive_width / 2, y + y_offset, mZ + z_offset + epoxy_thickness + kapton_thickness + copper_thickness + epoxy_thickness + silicon_thickness / 2));

                // left inactive strip
                std::string inactive_name_left = "FT3inactive_left_back_" + std::to_string(layerNumber) + "_" + std::to_string(direction) + "_" + std::to_string(sensor_count);
                sensor = geoManager->MakeBox(inactive_name_left.c_str(), siliconMed, inactive_width / 2, sensor_height / 2, silicon_thickness / 2);
                sensor->SetLineColor(kRed);
                sensor->SetFillColorAlpha(kRed, 1.0);
                motherVolume->AddNode(sensor, sensor_count++, new TGeoTranslation(x_offset + inactive_x_shift_left, y + y_offset, mZ + z_offset + epoxy_thickness + kapton_thickness + copper_thickness + epoxy_thickness + silicon_thickness / 2));

                // right inactive strip
                std::string inactive_name_right = "FT3inactive_right_back_" + std::to_string(layerNumber) + "_" + std::to_string(direction) + "_" + std::to_string(sensor_count);
                sensor = geoManager->MakeBox(inactive_name_right.c_str(), siliconMed, inactive_width / 2, sensor_height / 2, silicon_thickness / 2);
                sensor->SetLineColor(kRed);
                sensor->SetFillColorAlpha(kRed, 1.0);
                motherVolume->AddNode(sensor, sensor_count++, new TGeoTranslation(x_offset + inactive_x_shift_right, y + y_offset, mZ + z_offset + epoxy_thickness + kapton_thickness + copper_thickness + epoxy_thickness + silicon_thickness / 2));
              }
            }
          }
        }
      }

      rowCounter++;
    }
  }
  LOG(debug) << "FT3Module: done create_layout";
}

void FT3Module::createModule(double mZ, int layerNumber, int direction, double Rin, double Rout, double overlap, const std::string& face, const std::string& layout_type, TGeoVolume* motherVolume)
{

  LOG(debug) << "FT3Module: createModule - Layer " << layerNumber << ", Direction " << direction << ", Face " << face;
  create_layout(mZ, layerNumber, direction, Rin, Rout, overlap, face, layout_type, motherVolume);
  LOG(debug) << "FT3Module: done createModule";
}

void FT3Module::createModule_staveGeo(double mZ, int layerNumber, int direction,
                                      double Rin, double Rout, double z_offset_local,
                                      const Constants::StaveConfig& staveConfig,
                                      TGeoVolume* motherVolume)
{
  LOG(debug) << "FT3Module: createModule_staveGeo - Layer " << layerNumber
             << " at z=" << mZ << ", Direction " << direction;
  create_layout_staveGeo(mZ, layerNumber, direction, Rin, Rout,
                         z_offset_local, staveConfig, motherVolume);
  LOG(debug) << "FT3Module: done createModule_staveGeo";
}
