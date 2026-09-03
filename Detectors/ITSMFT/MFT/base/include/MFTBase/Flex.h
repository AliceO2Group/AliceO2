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

/// \file Flex.h
/// \brief Flex (Printed Cabled Board) class for ALICE MFT upgrade
/// \author Franck Manso <franck.manso@cern.ch>

#ifndef ALICEO2_MFT_FLEX_H_
#define ALICEO2_MFT_FLEX_H_

#include "Rtypes.h"

#include <string>

class TGeoVolume;
class TGeoVolumeAssembly;

namespace o2
{
namespace mft
{

class Flex
{

 public:
  Flex();
  Flex(LadderSegmentation* ladder);
  ~Flex();
  TGeoVolumeAssembly* makeFlex(Int_t nbsensors, Double_t length);
  void makeElectricComponents(TGeoVolumeAssembly* flex, Int_t nbsensors, Double_t length, Double_t zvarnish);

  /// Name of the flex shared by every ladder carrying nbsensors sensors.
  ///
  /// One flex is built per sensor-count class and placed on all the ladders of that class,
  /// so the name is keyed by the sensor count and no longer by half/disk/ladder. The ladder
  /// a given placement belongs to is still read from the node path, for example
  /// /cave_1/barrel_1/MFT_0/MFT_H_0_0/MFT_D_0_0_0/MFT_L_0_0_5_5/flex_3_1
  static std::string composeFlexName(Int_t nbsensors);

  /// Name of one layer inside that flex: "lineslayer", "alulayer", "kaptonlayer" or
  /// "varnishlayer". The varnish is placed twice, iflag 0 in front of the cold plate and
  /// iflag 1 outside; the other layers take no iflag.
  static std::string composeFlexLayerName(const char* layer, Int_t nbsensors, Int_t iflag = -1);

  /// The flex volume of that class in the current geometry, or nullptr if it has none.
  /// Resolved by name, so it also answers on a geometry read back from a file. To go from
  /// a ladder to its flex, take the sensor count from
  /// GeometryTGeo::getNumberOfSensorsPerLadder(half, disk, ladder) and pass it here.
  static TGeoVolumeAssembly* getFlexVolume(Int_t nbsensors);

 private:
  TGeoVolume* makeLines(Int_t nbsensors, Double_t length, Double_t width, Double_t thickness);
  TGeoVolume* makeAGNDandDGND(Int_t nbsensors, Double_t length, Double_t width, Double_t thickness);
  TGeoVolume* makeKapton(Int_t nbsensors, Double_t length, Double_t width, Double_t thickness);
  TGeoVolume* makeVarnish(Int_t nbsensors, Double_t length, Double_t width, Double_t thickness, Int_t iflag);
  TGeoVolumeAssembly* makeElectricComponent(Double_t dx, Double_t dy, Double_t dz, Int_t iflag);

  Double_t* mFlexOrigin;
  LadderSegmentation* mLadderSeg;

  ClassDefNV(Flex, 1);
};
} // namespace mft
} // namespace o2

#endif
