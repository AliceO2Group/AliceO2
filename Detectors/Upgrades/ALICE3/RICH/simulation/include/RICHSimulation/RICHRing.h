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
//
// Design and equations: Nicola Nicassio nicola.nicassio@cern.ch

#ifndef ALICEO2_RICH_RING_H
#define ALICEO2_RICH_RING_H

#include <TGeoManager.h>
#include <Rtypes.h>
#include <TMath.h>

namespace o2
{
namespace rich
{
class Ring
{
 public:
  Ring() = default;
  // Angle M_i: the angle formed by the normal line to both tile planes (radiator and photosensitive surface) passing by the center of the ring
  // Angle T_i: the angle formed by the line passing by the farest border of the tile and the center of the ring
  // Z_r: length of the radiator in Z
  // Z_p: length of the photosensitive surface in Z
  // DeltaRSurf_i: radial dinstance between two surfaces of tiles
  // R_ph: radius of the photosensitive surface (from the center)
  // z_ph: z position of the photosensitive surface (from the center)
  Ring(int rPosId,
       int nTilesPhi,
       float rMin,
       float rMax,
       float radThick,
       float radYmin,
       float radYmax,
       float radZ,
       float photThick,
       float photYmin,
       float photYmax,
       float photZ,
       float radRad0,
       float photRad0,
       float aerDetDistance,
       float thetaB,
       const std::string motherName = "RICHV");
  ~Ring() = default;

  auto getDeltaPhiPos() const { return TMath::TwoPi() / mNTiles; }
  void createRing(TGeoVolume* motherVolume);

 private:
  int mPosId;           // id of the ring
  int mNTiles;          // number of modules
  float mRRad;          // max distance for radiators
  float mRPhot;         // max distance for photosensitive surfaces
  float mRadThickness;  // thickness of the radiator
  float mPhotThickness; // thickness of the photosensitive surface

  ClassDef(Ring, 0);
};

} // namespace rich
} // namespace o2
#endif // ALICEO2_RICH_RING_H