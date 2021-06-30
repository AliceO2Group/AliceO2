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

/// \file HalfCone.h
/// \brief Class building geometry of one half of one MFT half-cone
/// \author sbest@pucp.pe, eric.endress@gmx.de, franck.manso@clermont.in2p3.fr
/// \date 15/12/2016

#ifndef ALICEO2_MFT_HALFCONE_H_
#define ALICEO2_MFT_HALFCONE_H_

class TGeoVolumeAssembly;

namespace o2
{
namespace mft
{

class HalfCone
{

 public:
  HalfCone();
  void makeAirVentilation(TGeoVolumeAssembly* HalfConeVolume, Int_t half, Int_t signe);
  void makeMotherBoards(TGeoVolumeAssembly* HalfConeVolume, Int_t half, Int_t signe, Double_t tyMB0, Double_t tyMB0_3, Double_t tzMB0);
  void makeFlexCables(TGeoVolumeAssembly* HalfConeVolume, Int_t half, Int_t signe);
  void makeReadoutCables(TGeoVolumeAssembly* HalfConeVolume, Int_t half, Int_t signe);
  void makePowerCables(TGeoVolumeAssembly* HalfConeVolume, Int_t half, Int_t signe);

  ~HalfCone();

  TGeoVolumeAssembly* createHalfCone(Int_t half);

 protected:
  TGeoVolumeAssembly* mHalfCone;

 private:
  ClassDef(HalfCone, 1);
};
} // namespace mft
} // namespace o2

#endif
