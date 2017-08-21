// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "TOFBase/Geo.h"
#include "TMath.h"

ClassImp(o2::tof::Geo);

using namespace o2::tof;

constexpr Float_t Geo::ANGLES[NPLATES][NMAXNSTRIP];
constexpr Float_t Geo::HEIGHTS[NPLATES][NMAXNSTRIP];
constexpr Float_t Geo::DISTANCES[NPLATES][NMAXNSTRIP];
constexpr Bool_t Geo::FEAWITHMASKS[18];
constexpr Float_t Geo::ROOF2PARAMETERS[3];

void Geo::GetDetID(Float_t* pos, Int_t* det)
{
  //
  // Returns Detector Indices (iSect,iPlate,iStrip,iPadX,iPadZ)
  // space point coor (x,y,z) (cm)

  Float_t posLocal[3];
  for (Int_t ii = 0; ii < 3; ii++)
    posLocal[ii] = pos[ii];


  det[0] = GetSector(posLocal);
  
  FromGlobalToPlate(posLocal,det[0]);

  det[1] = GetPlate(posLocal);

  det[2] = FromPlateToStrip(posLocal,det[1]);

  det[3] = GetPadZ(posLocal);
  det[4] = GetPadX(posLocal);
}
void Geo::FromGlobalToPlate(Float_t* pos,Int_t  isector){

  if (isector == -1) {
    Error("FromGlobalToPlate","Sector Index not valid (-1)");
    return;
  }

  // ALICE reference frame -> B071/B074/B075 = BTO1/2/3 reference frame
  Double_t angles[6] = { 90., 90. + (isector + 0.5) * PHISEC, 0., 0., 90., (isector + 0.5) * PHISEC };
  Rotation(pos, angles);

  Float_t step[3] = { 0., 0., static_cast<Float_t>((RMAX + RMIN) * 0.5) };
  Translation(pos, step);

  // B071/B074/B075 = BTO1/2/3 reference frame -> FTOA = FLTA reference frame
  angles[0] = 90.;
  angles[1] = 0.;
  angles[2] = 0.;
  angles[3] = 0.;
  angles[4] = 90.;
  angles[5] = 270.;

  Rotation(pos, angles);
}

Int_t Geo::FromPlateToStrip(Float_t* pos,Int_t  iplate){
  if (iplate == -1) {
    return -1;
  }

  Int_t nstrips = 0;
  switch (iplate) {
    case 0:
    case 4:
      nstrips = NSTRIPC;
      break;
    case 1:
    case 3:
      nstrips = NSTRIPB;
      break;
    case 2:
      nstrips = NSTRIPA;
      break;
  }

  constexpr Float_t HGLFY = HFILIY + 2 * HGLASSY; // heigth of GLASS+FISHLINE  Layer
  constexpr Float_t HSTRIPY = 2. * HHONY + 2. * HPCBY + 4. * HRGLY + 2. * HGLFY + HCPCBY; // 3.11

  Double_t angles[6];
  Float_t step[3];

  // FTOA/B/C = FLTA/B/C reference frame -> FSTR reference frame
  for (Int_t istrip = 0; istrip < nstrips; istrip++) {
    Float_t posLoc2[3] = { pos[0], pos[1], pos[2] };

    step[0] = 0.;
    step[1] = GetHeights(iplate, istrip);
    step[2] = -GetDistances(iplate, istrip);
    Translation(posLoc2, step);

    if (GetAngles(iplate, istrip) > 0.) {
      angles[0] = 90.;
      angles[1] = 0.;
      angles[2] = 90. + GetAngles(iplate, istrip);
      angles[3] = 90.;
      angles[4] = GetAngles(iplate, istrip);
      angles[5] = 90.;
    } else if (GetAngles(iplate, istrip) == 0.) {
      angles[0] = 90.;
      angles[1] = 0.;
      angles[2] = 90.;
      angles[3] = 90.;
      angles[4] = 0;
      angles[5] = 0.;
    } else if (GetAngles(iplate, istrip) < 0.) {
      angles[0] = 90.;
      angles[1] = 0.;
      angles[2] = 90. + GetAngles(iplate, istrip);
      angles[3] = 90.;
      angles[4] = -GetAngles(iplate, istrip);
      angles[5] = 270.;
    }
    Rotation(posLoc2, angles);

    if ((TMath::Abs(pos[0]) <= STRIPLENGTH * 0.5) && (TMath::Abs(pos[1]) <= HSTRIPY * 0.5) &&
        (TMath::Abs(pos[2]) <= WCPCBZ * 0.5)) {

      step[0] = -0.5 * NPADX * XPAD;
      step[1] = 0.;
      step[2] = -0.5 * NPADZ * ZPAD;
      Translation(posLoc2, step);

      for (Int_t jj = 0; jj < 3; jj++)
        pos[jj] = posLoc2[jj];

      return istrip;
    }
  }
}

Int_t Geo::GetSector(const Float_t* pos)
{
  //
  // Returns the Sector index
  //

  Int_t iSect = -1;

  Float_t x = pos[0];
  Float_t y = pos[1];
  Float_t z = pos[2];

  Float_t rho2 = x * x + y * y;

  if (!((z >= -ZLENA * 0.5 && z <= ZLENA * 0.5) && (rho2 >= (RMIN2) && rho2 <= (RMAX2)))) {
    // AliError("Detector Index could not be determined");
    return iSect;
  }

  Float_t phi = TMath::Pi() + TMath::ATan2(-y, -x);

  iSect = (Int_t)(phi * TMath::RadToDeg() / PHISEC);

  return iSect;
}

Int_t Geo::GetPlate(const Float_t* pos)
{
  //
  // Returns the Plate index
  //

  Int_t iPlate = -1;

  Float_t yLocal = pos[1];
  Float_t zLocal = pos[2];

  Float_t deltaRhoLoc = (RMAX - RMIN) * 0.5 - MODULEWALLTHICKNESS + yLocal;
  Float_t deltaZetaLoc = TMath::Abs(zLocal);

  Float_t deltaRHOmax = 0.;

  if (TMath::Abs(zLocal) >= EXTERINTERMODBORDER1 && TMath::Abs(zLocal) <= EXTERINTERMODBORDER2) {
    deltaRhoLoc -= LENGTHEXINMODBORDER;
    deltaZetaLoc = EXTERINTERMODBORDER2 - deltaZetaLoc;
    deltaRHOmax = (RMAX - RMIN) * 0.5 - MODULEWALLTHICKNESS - 2. * LENGTHEXINMODBORDER; // old 5.35, new 4.8

    if (deltaRhoLoc > deltaZetaLoc * deltaRHOmax / (INTERCENTRMODBORDER2 - INTERCENTRMODBORDER1)) {
      if (zLocal < 0)
        iPlate = 0;
      else
        iPlate = 4;
    } else {
      if (zLocal < 0)
        iPlate = 1;
      else
        iPlate = 3;
    }
  } else if (TMath::Abs(zLocal) >= INTERCENTRMODBORDER1 && TMath::Abs(zLocal) <= INTERCENTRMODBORDER2) {
    deltaRhoLoc -= LENGTHINCEMODBORDERD;
    deltaZetaLoc = deltaZetaLoc - INTERCENTRMODBORDER1;
    deltaRHOmax = (RMAX - RMIN) * 0.5 - MODULEWALLTHICKNESS - 2. * LENGTHINCEMODBORDERD; // old 0.39, new 0.2

    if (deltaRhoLoc > deltaZetaLoc * deltaRHOmax / (INTERCENTRMODBORDER2 - INTERCENTRMODBORDER1))
      iPlate = 2;
    else {
      if (zLocal < 0)
        iPlate = 1;
      else
        iPlate = 3;
    }
  }

  if (zLocal > -ZLENA * 0.5 && zLocal < -EXTERINTERMODBORDER2)
    iPlate = 0;
  else if (zLocal > -EXTERINTERMODBORDER1 && zLocal < -INTERCENTRMODBORDER2)
    iPlate = 1;
  else if (zLocal > -INTERCENTRMODBORDER1 && zLocal < INTERCENTRMODBORDER1)
    iPlate = 2;
  else if (zLocal > INTERCENTRMODBORDER2 && zLocal < EXTERINTERMODBORDER1)
    iPlate = 3;
  else if (zLocal > EXTERINTERMODBORDER2 && zLocal < ZLENA * 0.5)
    iPlate = 4;

  return iPlate;
}

Int_t Geo::GetPadZ(const Float_t* pos)
{
  //
  // Returns the Pad index along Z
  //

  Int_t iPadZ = (Int_t)(pos[2] / ZPAD);
  if (iPadZ == NPADZ)
    iPadZ--;
  else if (iPadZ > NPADZ)
    iPadZ = -1;

  return iPadZ;
}
//_____________________________________________________________________________
Int_t Geo::GetPadX(const Float_t* pos)
{
  //
  // Returns the Pad index along X
  //

  Int_t iPadX = (Int_t)(pos[0] / XPAD);
  if (iPadX == NPADX)
    iPadX--;
  else if (iPadX > NPADX)
    iPadX = -1;

  return iPadX;
}

void Geo::Translation(Float_t* xyz, Float_t translationVector[3])
{
  //
  // Return the vector xyz translated by translationVector vector
  //

  Int_t ii = 0;

  for (ii = 0; ii < 3; ii++)
    xyz[ii] -= translationVector[ii];

  return;
}

void Geo::Rotation(Float_t* xyz, Double_t rotationAngles[6])
{
  //
  // Return the vector xyz rotated according to the rotationAngles angles
  //

  Int_t ii = 0;
  /*
    TRotMatrix *matrix = new TRotMatrix("matrix","matrix", angles[0], angles[1],
    angles[2], angles[3],
    angles[4], angles[5]);
  */

  for (ii = 0; ii < 6; ii++)
    rotationAngles[ii] *= TMath::DegToRad();

  Float_t xyzDummy[3] = { 0., 0., 0. };

  for (ii = 0; ii < 3; ii++) {
    xyzDummy[ii] = xyz[0] * TMath::Sin(rotationAngles[2 * ii]) * TMath::Cos(rotationAngles[2 * ii + 1]) +
                   xyz[1] * TMath::Sin(rotationAngles[2 * ii]) * TMath::Sin(rotationAngles[2 * ii + 1]) +
                   xyz[2] * TMath::Cos(rotationAngles[2 * ii]);
  }

  for (ii = 0; ii < 3; ii++)
    xyz[ii] = xyzDummy[ii];

  return;
}

void Geo::InverseRotation(Float_t* xyz, Double_t rotationAngles[6])
{
  //
  // Rotates the vector xyz acordint to the rotationAngles
  //

  Int_t ii = 0;

  for (ii = 0; ii < 6; ii++)
    rotationAngles[ii] *= TMath::DegToRad();

  Float_t xyzDummy[3] = { 0., 0., 0. };

  xyzDummy[0] = xyz[0] * TMath::Sin(rotationAngles[0]) * TMath::Cos(rotationAngles[1]) +
                xyz[1] * TMath::Sin(rotationAngles[2]) * TMath::Cos(rotationAngles[3]) +
                xyz[2] * TMath::Sin(rotationAngles[4]) * TMath::Cos(rotationAngles[5]);

  xyzDummy[1] = xyz[0] * TMath::Sin(rotationAngles[0]) * TMath::Sin(rotationAngles[1]) +
                xyz[1] * TMath::Sin(rotationAngles[2]) * TMath::Sin(rotationAngles[3]) +
                xyz[2] * TMath::Sin(rotationAngles[4]) * TMath::Sin(rotationAngles[5]);

  xyzDummy[2] = xyz[0] * TMath::Cos(rotationAngles[0]) + xyz[1] * TMath::Cos(rotationAngles[2]) +
                xyz[2] * TMath::Cos(rotationAngles[4]);

  for (ii = 0; ii < 3; ii++)
    xyz[ii] = xyzDummy[ii];

  return;
}
