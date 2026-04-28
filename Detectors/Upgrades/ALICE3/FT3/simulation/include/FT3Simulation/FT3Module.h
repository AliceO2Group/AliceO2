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

/// \file FT3Module.h
/// \brief Definition of the FT3Module class

#ifndef FT3MODULE_H
#define FT3MODULE_H

#include <TGeoVolume.h>
#include <string>
#include <vector>

#include "FT3Simulation/FT3ModuleConstants.h"

// define types for y positions, second element is the stack height
using PositionType = std::pair<double, unsigned>;
using PositionTypes = std::vector<PositionType>;
using PosNegPositionTypes = std::pair<PositionTypes, PositionTypes>;
// define type of the y position range: First pair is (min, max) for positive y
using PositionRangeType = std::pair<std::pair<double, double>, std::pair<double, double>>;
namespace Constants = o2::ft3::ModuleConstants;

class FT3Module
{

 public:
  static void initialize_materials();
  static TGeoMaterial* siliconMat;
  static TGeoMedium* siliconMed;
  static TGeoMaterial* copperMat;
  static TGeoMedium* copperMed;
  static TGeoMaterial* kaptonMat;
  static TGeoMedium* kaptonMed;
  static TGeoMaterial* epoxyMat;
  static TGeoMedium* epoxyMed;
  static TGeoMaterial* AluminumMat;
  static TGeoMedium* AluminumMed;
  static TGeoMaterial* carbonFiberMat;
  static TGeoMedium* carbonFiberMed;

  const char* mDetName;

  static void createModule(
    double mZ, int layerNumber, int direction, double Rin,
    double Rout, double overlap, const std::string& face,
    const std::string& layout_type, TGeoVolume* motherVolume);

  void createModule_staveGeo(
    double mZ, int layerNumber, int direction, double Rin,
    double Rout, double z_offset_local, const Constants::StaveConfig& staveConfig,
    TGeoVolume* motherVolume);

 private:
  static void create_layout(
    double mZ, int layerNumber, int direction, double Rin,
    double Rout, double overlap, const std::string& face,
    const std::string& layout_type, TGeoVolume* motherVolume);

  void create_layout_staveGeo(
    double mZ, int layerNumber, int direction, double Rin,
    double Rout, double z_offset_local, const Constants::StaveConfig& staveConfig,
    TGeoVolume* motherVolume);

  // Helper functions
  void fill_stave(PosNegPositionTypes& y_positions, double Rin, double Rout,
                  double x_left, unsigned kSensorStack, PositionRangeType y_range,
                  std::pair<double, double>& absAllowedYRange);
  void addStaveVolume(
    TGeoVolume* motherVolume, std::string volumeName, int direction,
    unsigned* volume_count, double staveLength,
    std::array<std::array<double, 3>, 4> staveTriangles,
    std::pair<double, double>& absAllowedYRange,
    double x_mid, double y_mid, double z_stave_shift_forward);
  void addDetectorVolume(
    TGeoVolume* motherVolume, std::string volumeName, int color, unsigned* volume_count,
    double x_mid, double y_mid, double z_mid,
    double x_half_length, double y_half_length, double z_half_length);

  void add2x1GlueVolume(
    TGeoVolume* motherVolume, int layerNumber, int direction, unsigned stave_idx,
    unsigned* volume_count, double x_mid, double y_mid, double z_mid,
    std::string element_glued_to);

  void add2x1CopperVolume(
    TGeoVolume* motherVolume, int layerNumber, int direction, unsigned stave_idx,
    unsigned* volume_count, double x_mid, double y_mid, double z_mid);

  void add2x1KaptonVolume(
    TGeoVolume* motherVolume, int layerNumber, int direction, unsigned stave_idx,
    unsigned* volume_count, double x_mid, double y_mid, double z_mid);

  void addSingleSensorVolume(
    TGeoVolume* motherVolume, int layerNumber, int direction, unsigned stave_idx,
    unsigned* volume_count, double active_x_mid, double y_mid, double z_mid, bool isLeft);
};

#endif // FT3MODULE_H
