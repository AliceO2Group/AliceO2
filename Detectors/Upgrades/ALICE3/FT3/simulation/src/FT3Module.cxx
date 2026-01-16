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

  double foamSpacingThickness = 0.5;

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

  if (Rin == 7 && sensor_height == 9.6 && sensor_width == 5) {
    x_condition_min = -Rin - 2;
    x_condition_max = Rin;
    adjust_bottom_y_pos = true;
    adjust_bottom_y_neg = true;
    x_adjust_bottom_y_pos = 3.5;
    bottom_y_pos_value = 3.5;
    bottom_y_neg_value = -3.5;

    dist_offset = 2;

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
  } else {
    std::cout << "Different config - to determine offsets needed." << std::endl;
    x_condition_min = -Rin;
    x_condition_max = Rin;
    adjust_bottom_y_pos = false;
    adjust_bottom_y_neg = false;
  }

  double Rin_offset = (sensor_height == 19.2) ? 1 : 0;
  double Rout_offset = (sensor_height == 19.2) ? 1 : 0;

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
    // filling for sensors with 2x width, each row skipped
    if (face == "front") {
      X_positions = {-63.4, -54.2, -45, -35.8, -26.6, -17.4, -8.2, 1., 10.2, 19.4, 28.6, 37.8, 47., 56.2, 65.4};
      justSkipped1 = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    } else if (face == "back") {
      X_positions = {-58.8, -49.6, -40.4, -31.2, -22, -12.8, -3.6, 5.6, 14.8, 24, 33.2, 42.4, 51.6, 60.8};
      justSkipped1 = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
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
                std::string sensor_name = "FT3sensor_front_" + std::to_string(layerNumber) + "_" + std::to_string(direction) + "_" + std::to_string(sensor_count);
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

                std::string sensor_name = "FT3sensor_front_" + std::to_string(layerNumber) + "_" + std::to_string(direction) + "_" + std::to_string(sensor_count);
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

                std::string sensor_name = "FT3sensor_back_" + std::to_string(layerNumber) + "_" + std::to_string(direction) + "_" + std::to_string(sensor_count);
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
                std::string sensor_name = "FT3sensor_back_" + std::to_string(layerNumber) + "_" + std::to_string(direction) + "_" + std::to_string(sensor_count);
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
