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
       double rMin,
       double rMax,
       double radThick,
       double radYmin,
       double radYmax,
       double radZ,
       double photThick,
       double photYmin,
       double photYmax,
       double photZ,
       double radRad0,
       double photRad0,
       double aerDetDistance,
       double thetaB,
       const std::string motherName = "RICHV");
  ~Ring() = default;

  auto getDeltaPhiPos() const { return TMath::TwoPi() / mNTiles; }
  void createRing(TGeoVolume* motherVolume);
  int getPosId() const { return mPosId; }
  int getNTiles() const { return mNTiles; }

 private:
  int mPosId;           // id of the ring
  int mNTiles;          // number of modules
  double mRRad;         // max distance for radiators
  double mRPhot;        // max distance for photosensitive surfaces
  double mRadThickness; // thickness of the radiator
  double mPhotThickness; // thickness of the photosensitive surface

  ClassDef(Ring, 0);
};

// Definitions for fwd and bwd RICH are put here
class FWDRich
{
 public:
  FWDRich() = default;
  FWDRich(std::string name,
          double rMin,
          double rMax,
          double zAerogelMin,
          double dZAerogel,
          double zArgonMin,
          double dZArgon,
          double zSiliconMin,
          double dZSilicon);
  void createFWDRich(TGeoVolume* motherVolume);

 protected:
  std::string mName;
  double mRmin;
  double mRmax;

  // Aerogel:
  double mZAerogelMin;
  double mDZAerogel;

  // Argon:
  double mZArgonMin;
  double mDZArgon;

  // Silicon:
  double mZSiliconMin;
  double mDZSilicon;

  ClassDef(FWDRich, 0);
};

class BWDRich
{
 public:
  BWDRich() = default;
  BWDRich(std::string name,
          double rMin,
          double rMax,
          double zAerogelMin,
          double dZAerogel,
          double zArgonMin,
          double dZArgon,
          double zSiliconMin,
          double dZSilicon);
  void createBWDRich(TGeoVolume* motherVolume);

 protected:
  std::string mName;
  double mRmin;
  double mRmax;

  // Aerogel:
  double mZAerogelMin;
  double mDZAerogel;

  // Argon:
  double mZArgonMin;
  double mDZArgon;

  // Silicon:
  double mZSiliconMin;
  double mDZSilicon;

  ClassDef(BWDRich, 0);
};

} // namespace rich
} // namespace o2
#endif // ALICEO2_RICH_RING_H