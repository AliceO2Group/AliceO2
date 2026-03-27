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
#include <TGeoMatrix.h>
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
void FT3Module::fill_stave(PosNegPositionTypes& y_positions, double Rout,
                           double x_left, unsigned kSensorStack, double tolerance,
                           std::pair<double, double> y_start={0, 0})
{
  // start with upper half of the stave, then mirror to the bottom half
  double x_right = x_left + Constants::sensor2x1_width;
  double y_top = y_start.first;
  // either start at given start position, or at the top of the last placed sensors
  if (!y_positions.first.empty()) {
    y_top = y_positions.first.back().first
          + Constants::sensor2x1_height * y_positions.first.back().second
          + Constants::sensor2x1_gap;
  }
  // add the height of kSensorStack sensors + the gaps in between them
  double sensorTileHeight = Constants::sensor2x1_height * kSensorStack
                          + Constants::sensor2x1_gap * (kSensorStack - 1);

  double max_y_abs;
  // y_max(x_left) > y_max(x_right) means that the top of the sensor
  // will hit the outer radius at x_right first
  if (x_left > -Constants::single_sensor_width) {
    // tolerance already in maximum y position
    max_y_abs = calculate_y_circle(x_right, Rout) - tolerance;
  } else {
    max_y_abs = calculate_y_circle(x_left, Rout) - tolerance;
  }
  unsigned n_sensors_placed = y_positions.first.size() + y_positions.second.size();
  LOG(info) << "\tFT3Module: Filling stave at x = " << (x_left + Constants::sensor2x1_width / 2)
            << " with sensors of height " << sensorTileHeight
            << ". Starting positive y position: " << y_top
            << ", maximum positive y position: " << max_y_abs
            << ", with initially " << n_sensors_placed << " sensors already placed.";

  while ( (y_top + sensorTileHeight) <= max_y_abs ) {
    y_positions.first.emplace_back(y_top, kSensorStack);
    LOG(info) << "\t\t\tFT3Module: Placed sensor at y = " << y_top;
    y_top += sensorTileHeight + Constants::sensor2x1_gap;
  }

  // now we do the same for the negative y positions
  // they do not have to be exactly mirrored, hence done separately
  double y_bottom = y_start.second;
  if (!y_positions.second.empty()) {
    // subtract instead to move further down
    y_bottom = y_positions.second.back().first
             - Constants::sensor2x1_height * y_positions.second.back().second
             - Constants::sensor2x1_gap;
  }

  LOG(info) << "\tFT3Module: Starting negative y position: " << y_bottom
            << ", minimum negative y position: " << -max_y_abs;
  while ( (y_bottom - sensorTileHeight) >= -max_y_abs ) {
    y_positions.second.emplace_back(y_bottom, kSensorStack);
    LOG(info) << "\t\t\tFT3Module: Placed sensor at y = " << y_bottom;
    y_bottom -= (sensorTileHeight + Constants::sensor2x1_gap);
  }
  unsigned sensors_placed_after = y_positions.first.size() + y_positions.second.size();
  LOG(info) << "\tFT3Module: Done filling stave. Now have "
            << sensors_placed_after << " sensors in total.";
}

/*
 * Generic helper function that adds a box at the given position with
 * the given dimensions to the given mother volume, with the given color and name.
 */

void FT3Module::addDetectorVolume(
  TGeoVolume* motherVolume, std::string volumeName, int color, unsigned* sensor_count,
  double x_mid, double y_mid, double z_mid,
  double x_half_length, double y_half_length, double z_half_length)
{
  TGeoManager* geoManager = gGeoManager;
  TGeoVolume* volume = geoManager->MakeBox(volumeName.c_str(), siliconMed, x_half_length,
                                           y_half_length, z_half_length);
  volume->SetLineColor(color);
  volume->SetFillColorAlpha(color, 0.4);
  motherVolume->AddNode(
    volume,
    1,
    new TGeoTranslation(  // midpoint of box to add
      x_mid,
      y_mid,
      z_mid
    )  // TGeoTranslation
  );  // addNode
}

/*
 * This function adds a glue volume between two element layers,
 * immediately for a whole 2x1 layout, under both the active and inactive region.
 */
void FT3Module::add2x1GlueVolume(
  TGeoVolume* motherVolume, int layerNumber, int direction, unsigned* sensor_count,
  std::string side_str, double x_mid, double y_mid, double z_mid,
  std::string element_glued_to)
{
  std::string glue_name = "FT3glue_" + element_glued_to + "_" + side_str + "_"
                        + std::to_string(layerNumber) + "_" + std::to_string(direction)
                        + "_" + std::to_string(*sensor_count);
  addDetectorVolume(
    motherVolume, glue_name, Constants::glueColor, sensor_count,
    x_mid, y_mid, z_mid,
    Constants::sensor2x1_width / 2, Constants::sensor2x1_height / 2, Constants::epoxyThickness / 2
  );
}

/*
 * This function adds a copper volume onto which the silicon sensor is glued.
 * As with the glue, this is a whole 2x1 layout volume.
 */
void FT3Module::add2x1CopperVolume(
  TGeoVolume* motherVolume, int layerNumber, int direction, unsigned* sensor_count,
  std::string side_str, double x_mid, double y_mid, double z_mid)
{
  std::string copper_name = "FT3Copper_" + side_str + "_" + std::to_string(layerNumber) + "_"
                          + std::to_string(direction) + "_" + std::to_string(*sensor_count);
  addDetectorVolume(
    motherVolume, copper_name, Constants::CuColor, sensor_count,
    x_mid, y_mid, z_mid,
    Constants::sensor2x1_width / 2, Constants::sensor2x1_height / 2, Constants::copperThickness / 2
  );
}

/*
 * This function adds a kapton volume behind the copper, which represents the ???
 * As with copper and glue, this is a whole 2x1 layout volume.
 */
void FT3Module::add2x1KaptonVolume(
  TGeoVolume* motherVolume, int layerNumber, int direction, unsigned* sensor_count,
  std::string side_str, double x_mid, double y_mid, double z_mid)
{
  std::string kapton_name = "FT3Kapton_" + side_str + "_" + std::to_string(layerNumber) + "_"
                          + std::to_string(direction) + "_" + std::to_string(*sensor_count);
  addDetectorVolume(
    motherVolume, kapton_name, Constants::kaptonColor, sensor_count,
    x_mid, y_mid, z_mid,
    Constants::sensor2x1_width / 2, Constants::sensor2x1_height / 2, Constants::kaptonThickness / 2
  );
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
 * side_str: string indicating whether the sensor is on the front or back
 * isLeft: whether the sensor is on the left or right in the 2x1 layout
 */
void FT3Module::addSingleSensorVolume(
  TGeoVolume* motherVolume, int layerNumber, int direction, unsigned* sensor_count,
  double active_x_mid, double y_mid, double z_mid, std::string side_str, bool isLeft)
{
  TGeoVolume* sensor;
  TGeoManager* geoManager = gGeoManager;
  // ACTIVE AREA
  std::string sensor_name = "FT3Sensor_" + side_str + "_" + std::to_string(layerNumber) + "_"
                          + std::to_string(direction) + "_" + std::to_string(*sensor_count);
  sensor = geoManager->MakeBox(sensor_name.c_str(), siliconMed, Constants::active_width / 2,
                                Constants::single_sensor_height / 2, Constants::siliconThickness / 2);
  sensor->SetLineColor(Constants::SiColor);
  sensor->SetFillColorAlpha(Constants::SiColor, 0.4);
  motherVolume->AddNode(
    sensor,
    *sensor_count++,
    new TGeoTranslation(  // midpoint of box to add
      active_x_mid,
      y_mid,
      z_mid
    )  // TGeoTranslation
  );  // addNode
  // INACTIVE STRIP ON LEFT OR RIGHT
  double inactive_x_mid = isLeft ? (active_x_mid - Constants::active_width / 2 - Constants::inactive_width / 2)
                                 : (active_x_mid + Constants::active_width / 2 + Constants::inactive_width / 2);
  std::string sensor_inactive_left_name =
    "FT3Sensor_InactiveLeft_" + side_str + "_" + std::to_string(layerNumber) + "_"
    + std::to_string(direction) + "_" + std::to_string(*sensor_count);
  sensor = geoManager->MakeBox(sensor_inactive_left_name.c_str(), siliconMed, Constants::inactive_width / 2,
                                Constants::single_sensor_height / 2, Constants::siliconThickness / 2);
  sensor->SetLineColor(Constants::SiInactiveColor);
  sensor->SetFillColorAlpha(Constants::SiInactiveColor, 0.4);
  motherVolume->AddNode(
    sensor,
    *sensor_count++,
    new TGeoTranslation(  // midpoint of box to add
      inactive_x_mid,
      y_mid,
      z_mid
    )  // TGeoTranslation
  );  // addNode
}

void FT3Module::create_layout_scopingV3(double mZ, int layerNumber, int direction,
                                        double Rin, double Rout, double overlap,
                                        TGeoVolume* motherVolume)
{
  LOG(info) << "FT3Module: create_layout_scopingV3 - Layer "
            << layerNumber << ", Direction " << direction;

  FT3Module::initialize_materials();
  auto& ft3Params = o2::ft3::FT3BaseParam::Instance();

  // initialise all y_positions, vector over all staves
  std::vector<PosNegPositionTypes> y_positionsPosNeg;
  // Fill all staves
  for (unsigned i_stave = 0; i_stave < Constants::x_midpoints.size(); i_stave++) {
    y_positionsPosNeg.emplace_back(PosNegPositionTypes{PositionTypes{}, PositionTypes{}});

    double y_midpoint = 0.;
    // default positive and negative starting points has a gap around x-axis
    std::pair<double, double> y_start{0., Constants::sensor2x1_gap};
    const int staveID = Constants::staveIdxToID(i_stave);
    auto y_midpoint_it = Constants::staveID_to_y_midpoint.find(staveID);
    if ( y_midpoint_it != Constants::staveID_to_y_midpoint.end() ) {
      // there is a defined midpoint for this stave, use this for starting points
      y_midpoint = y_midpoint_it->second;  // avoid double map lookup
      double y_start_pos = y_midpoint - Constants::y_lengths[i_stave] / 2;
    }

    double x_left = Constants::x_midpoints[i_stave] - Constants::sensor2x1_width / 2;
    double x_right = x_left + Constants::sensor2x1_width;
    double tolerance = -Constants::sensor_stack_height;  // allow one sensor placement beyond
    // cut staves on nominal inner radius if specified
    if (ft3Params.cutStavesOnNominalRadius) {
      double min_y_at_x;
      if (x_left * x_right < 0) {
        // stave crosses y-axis, so we start at y=Rin
        min_y_at_x = Rin;
      } else if (x_left > 0) {
        // stave is on the right side, so minimum y is at x_left
        min_y_at_x = calculate_y_circle(x_left, Rin);
      } else {
        // stave is on the left side, so minimum y is at x_right
        min_y_at_x = calculate_y_circle(x_right, Rin);
      }
      y_start = {min_y_at_x, -min_y_at_x};
      tolerance = 0.; // no tolerance in case of cutting at nominal radius
    }
    // fill_stave(y_positionsPosNeg, Rout, x_left, Constants::kSensorsPerStack, -3);
    LOG(info) << "FT3Module: Filling Stave " << staveID << " (x = "
              << Constants::x_midpoints[i_stave] << ") with sensors. Starting y positions: "
              << y_start.first << " (positive), " << y_start.second << " (negative).";
    fill_stave(y_positionsPosNeg.back(), Rout, x_left, Constants::kSensorsPerStack,
               tolerance, y_start);
  }

  unsigned sensor_count = 0;
  for (unsigned i_stave = 0; i_stave < Constants::x_midpoints.size(); i_stave++) {
    double x_mid = Constants::x_midpoints[i_stave];
    LOG(info) << "FT3Module: Adding sensor volumes for Stave " << Constants::staveIdxToID(i_stave)
              << " (x = " << x_mid << ") with " << y_positionsPosNeg[i_stave].first.size() << " positive and "
              << y_positionsPosNeg[i_stave].second.size() << " negative sensor positions.";
    for (unsigned i_y_pos = 0; i_y_pos < y_positionsPosNeg[i_stave].first.size(); i_y_pos++) {
      for (unsigned i_y_sign = 0; i_y_sign < 2; i_y_sign++) {
        // TODO: Make this loop over all sensors in a stack, don't just assume one sensor per stack
        TGeoVolume* sensor;
        // place sensors at positive and negative y
        const auto& positions = (i_y_sign == 0) ? y_positionsPosNeg[i_stave].first 
                                                : y_positionsPosNeg[i_stave].second;
        double y_mid = positions[i_y_pos].first + Constants::sensor2x1_height / 2;

        // get which side we are on: if backward discs we mirror from front so it's the same
        // layout from the frame of the particle, regardless which direction
        bool isFront;
        if (!direction)  // direction = 0 is forward
          isFront = Constants::staveOnFront[i_stave];
        else
          isFront = !(Constants::staveOnFront[i_stave]);
        /* 
        * we build the volume from the outside in, starting with the silicon,
        * then glue & materials towards the stave. Depending on whether it's front or back,
        * the distance from the center will be mirrored so that we get the following:
        * 
        * Front (ordered in z, assuming the forward direction is to the right):
        * | SILICON SENSOR | GLUE | COPPER | KAPTON | GLUE | STAVE | SUPPORT STRUCTURE |
        * 
        * Back (ordered in z, assuming the forward direction is to the right):
        * | SUPPORT STRUCTURE | STAVE | GLUE | KAPTON | COPPER | GLUE | SILICON SENSOR |
        * 
        * Note that we do not place stave and support structure material here, that is
        * assumed to have been placed by the Layer creation.
        */
        double z_offset_centre_to_stave = Constants::foamSpacingThickness / 2.0 + Constants::carbonFiberThickness;
        double z_offset_stave_to_silicon = Constants::epoxyThickness + Constants::kaptonThickness + Constants::copperThickness
                                        + Constants::epoxyThickness + Constants::siliconThickness / 2;
        double z_offset_stave_to_glue_Si = Constants::epoxyThickness + Constants::kaptonThickness + Constants::copperThickness
                                        + Constants::epoxyThickness / 2;
        double z_offset_stave_to_copper = Constants::epoxyThickness + Constants::kaptonThickness + Constants::copperThickness / 2;
        double z_offset_stave_to_kapton = Constants::epoxyThickness + Constants::kaptonThickness / 2;
        double z_offset_stave_to_glue_Cu = Constants::epoxyThickness / 2;

        // for the front, we have to subtract the z offsets since we are going in
        // negative z direction, while it's opposite for the back
        int z_offset_multiplier = isFront ? -1 : 1;
        std::string side_str = isFront ? "front" : "back";
        // ------------ (1) Silicon sensor ------------
        // left single sensor of the 2x1
        double z_mid = (z_offset_centre_to_stave + z_offset_stave_to_silicon) * z_offset_multiplier;
        addSingleSensorVolume(
          motherVolume, layerNumber, direction, &sensor_count,
          x_mid - Constants::active_width / 2, y_mid, z_mid, side_str, true
        );
        // right single sensor of the 2x1
        addSingleSensorVolume(
          motherVolume, layerNumber, direction, &sensor_count,
          x_mid + Constants::active_width / 2, y_mid, z_mid, side_str, false
        );

        // ------------ (2) Epoxy glue layer between silicon and copper (FPC) ------------
        z_mid = (z_offset_centre_to_stave + z_offset_stave_to_glue_Si) * z_offset_multiplier;
        add2x1GlueVolume(
          motherVolume, layerNumber, direction, &sensor_count,
          side_str, x_mid, y_mid, z_mid, "SiCu"
        );
        // ------------ (3) Copper layer (FPC) ------------
        z_mid = (z_offset_centre_to_stave + z_offset_stave_to_copper) * z_offset_multiplier;
        add2x1CopperVolume(
          motherVolume, layerNumber, direction, &sensor_count,
          side_str, x_mid, y_mid, z_mid
        );
        // ------------ (4) Kapton layer (FPC) ------------
        z_mid = (z_offset_centre_to_stave + z_offset_stave_to_kapton) * z_offset_multiplier;
        add2x1KaptonVolume(
          motherVolume, layerNumber, direction, &sensor_count,
          side_str, x_mid, y_mid, z_mid
        );
        // ------------ (5) Epoxy glue layer between stave and FPC copper ------------
        z_mid = (z_offset_centre_to_stave + z_offset_stave_to_glue_Cu) * z_offset_multiplier;
        add2x1GlueVolume(
          motherVolume, layerNumber, direction, &sensor_count,
          side_str, x_mid, y_mid, z_mid, "StaveKapton"
        );
      }  // for i_y_sign (writing of positive or negative y positions)
    }  // i_y_pos
  }  // i_stave
  

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

void FT3Module::createModule_scopingV3(double mZ, int layerNumber, int direction,
                                       double Rin, double Rout, double overlap,
                                       TGeoVolume* motherVolume) {
  LOG(debug) << "FT3Module: createModule_scopingV3 - Layer " << layerNumber
             << ", Direction " << direction;
  create_layout_scopingV3(mZ, layerNumber, direction, Rin, Rout, overlap, motherVolume);
  LOG(debug) << "FT3Module: done createModule_scopingV3";
}
