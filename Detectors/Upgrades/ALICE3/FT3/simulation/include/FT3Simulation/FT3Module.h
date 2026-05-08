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

  const char* mDetName;

  static void createModule(double mZ, int layerNumber, int direction, double Rin, double Rout, double overlap, const std::string& face, const std::string& layout_type, TGeoVolume* motherVolume);

 private:
  static void create_layout(double mZ, int layerNumber, int direction, double Rin, double Rout, double overlap, const std::string& face, const std::string& layout_type, TGeoVolume* motherVolume);
};

#endif // FT3MODULE_H
