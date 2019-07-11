// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Flex.h
/// \brief Flex (Printed Cabled Board) class for ALICE MFT upgrade
/// \author Franck Manso <franck.manso@cern.ch>

#ifndef ALICEO2_MFT_FLEX_H_
#define ALICEO2_MFT_FLEX_H_

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

 private:
  TGeoVolume* makeLines(Int_t nbsensors, Double_t length, Double_t width, Double_t thickness);
  TGeoVolume* makeAGNDandDGND(Double_t length, Double_t width, Double_t thickness);
  TGeoVolume* makeKapton(Double_t length, Double_t width, Double_t thickness);
  TGeoVolume* makeVarnish(Double_t length, Double_t width, Double_t thickness, Int_t iflag);
  TGeoVolumeAssembly* makeElectricComponent(Double_t dx, Double_t dy, Double_t dz, Int_t iflag);

  Double_t* mFlexOrigin;
  LadderSegmentation* mLadderSeg;

  ClassDef(Flex, 1)
};
} // namespace mft
} // namespace o2

#endif
