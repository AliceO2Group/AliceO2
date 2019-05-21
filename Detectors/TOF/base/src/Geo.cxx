// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "TOFBase/Geo.h"
#include "TGeoManager.h"
#include "TMath.h"
#include "FairLogger.h"
#include "DetectorsBase/GeometryManager.h"

ClassImp(o2::tof::Geo);

using namespace o2::tof;

constexpr Float_t Geo::ANGLES[NPLATES][NMAXNSTRIP];
constexpr Float_t Geo::HEIGHTS[NPLATES][NMAXNSTRIP];
constexpr Float_t Geo::DISTANCES[NPLATES][NMAXNSTRIP];
constexpr Bool_t Geo::FEAWITHMASKS[NSECTORS];
constexpr Float_t Geo::ROOF2PARAMETERS[3];

Bool_t Geo::mToBeIntit = kTRUE;
Float_t Geo::mRotationMatrixSector[NSECTORS + 1][3][3];
Float_t Geo::mRotationMatrixPlateStrip[NPLATES][NMAXNSTRIP][3][3];
Float_t Geo::mPadPosition[NSECTORS][NPLATES][NMAXNSTRIP][NPADZ][NPADX][3];

void Geo::Init()
{
  LOG(INFO) << "tof::Geo: Initialization of TOF rotation parameters";

  if (!gGeoManager) {
    LOG(WARNING) << " no TGeo! Loading it";
    o2::base::GeometryManager::loadGeometry();
  }

  int det[5];
  Char_t path[200];
  for (Int_t isector = 0; isector < NSECTORS; isector++) {
    det[0] = isector;
    for (Int_t iplate = 0; iplate < NPLATES; iplate++) {
      det[1] = iplate;
      for (Int_t istrip = 0; istrip < NMAXNSTRIP; istrip++) {
	det[2] = istrip;
	if (getAngles(iplate, istrip) > 0.) {
	  for (Int_t ipadz = 0; ipadz < NPADZ; ipadz++) {
	    det[3] = ipadz;
	    for (Int_t ipadx = 0; ipadx < NPADX; ipadx++) {
	      det[4] = ipadx;
	      getVolumePath(det, path);
	      gGeoManager->cd(path);
	      TGeoHMatrix global;
	      global = *gGeoManager->GetCurrentMatrix();
	      const Double_t* tr = global.GetTranslation();
	      mPadPosition[isector][iplate][istrip][ipadz][ipadz][0] = tr[0];
	      mPadPosition[isector][iplate][istrip][ipadz][ipadz][1] = tr[1];
	      mPadPosition[isector][iplate][istrip][ipadz][ipadz][2] = tr[2];
	    }
	  }
	}
      }
    }
  }

  Double_t rotationAngles[6] =
    { 90., 90. /*+ (isector + 0.5) * PHISEC*/, 0., 0., 90., 0 /* + (isector + 0.5) * PHISEC*/ };
  for (Int_t ii = 0; ii < 6; ii++)
    rotationAngles[ii] *= TMath::DegToRad();

  for (Int_t isector = 0; isector < NSECTORS; isector++) {
    rotationAngles[5] = ((isector + 0.5) * PHISEC) * TMath::DegToRad();
    rotationAngles[1] = 90. * TMath::DegToRad() + rotationAngles[5];

    for (Int_t ii = 0; ii < 3; ii++) {
      mRotationMatrixSector[isector][ii][0] =
        TMath::Sin(rotationAngles[2 * ii]) * TMath::Cos(rotationAngles[2 * ii + 1]);
      mRotationMatrixSector[isector][ii][1] =
        TMath::Sin(rotationAngles[2 * ii]) * TMath::Sin(rotationAngles[2 * ii + 1]);
      mRotationMatrixSector[isector][ii][2] = TMath::Cos(rotationAngles[2 * ii]);
    }
  }

  rotationAngles[0] = 90. * TMath::DegToRad();
  rotationAngles[1] = 0.;
  rotationAngles[2] = 0.;
  rotationAngles[3] = 0.;
  rotationAngles[4] = 90. * TMath::DegToRad();
  rotationAngles[5] = 270. * TMath::DegToRad();
  for (Int_t ii = 0; ii < 3; ii++) {
    mRotationMatrixSector[NSECTORS][ii][0] =
      TMath::Sin(rotationAngles[2 * ii]) * TMath::Cos(rotationAngles[2 * ii + 1]);
    mRotationMatrixSector[NSECTORS][ii][1] =
      TMath::Sin(rotationAngles[2 * ii]) * TMath::Sin(rotationAngles[2 * ii + 1]);
    mRotationMatrixSector[NSECTORS][ii][2] = TMath::Cos(rotationAngles[2 * ii]);
  }

  for (Int_t iplate = 0; iplate < NPLATES; iplate++) {
    for (Int_t istrip = 0; istrip < NMAXNSTRIP; istrip++) {
      if (getAngles(iplate, istrip) > 0.) {
        rotationAngles[0] = 90. * TMath::DegToRad();
        rotationAngles[1] = 0.;
        rotationAngles[3] = 90. * TMath::DegToRad();
        rotationAngles[4] = getAngles(iplate, istrip) * TMath::DegToRad();
        rotationAngles[5] = 90. * TMath::DegToRad();
        rotationAngles[2] = 90. * TMath::DegToRad() + rotationAngles[4];
      } else if (getAngles(iplate, istrip) == 0.) {
        rotationAngles[0] = 90. * TMath::DegToRad();
        rotationAngles[1] = 0.;
        rotationAngles[2] = 90. * TMath::DegToRad();
        rotationAngles[3] = 90. * TMath::DegToRad();
        rotationAngles[4] = 0;
        rotationAngles[5] = 0.;
      } else if (getAngles(iplate, istrip) < 0.) {
        rotationAngles[0] = 90. * TMath::DegToRad();
        rotationAngles[1] = 0.;
        rotationAngles[3] = 90. * TMath::DegToRad();
        rotationAngles[4] = -getAngles(iplate, istrip) * TMath::DegToRad();
        rotationAngles[5] = 270. * TMath::DegToRad();
        rotationAngles[2] = 90. * TMath::DegToRad() - rotationAngles[4];
      }

      for (Int_t ii = 0; ii < 3; ii++) {
        mRotationMatrixPlateStrip[iplate][istrip][ii][0] =
          TMath::Sin(rotationAngles[2 * ii]) * TMath::Cos(rotationAngles[2 * ii + 1]);
        mRotationMatrixPlateStrip[iplate][istrip][ii][1] =
          TMath::Sin(rotationAngles[2 * ii]) * TMath::Sin(rotationAngles[2 * ii + 1]);
        mRotationMatrixPlateStrip[iplate][istrip][ii][2] = TMath::Cos(rotationAngles[2 * ii]);
      }
    }
  }

  mToBeIntit = kFALSE;
}

void Geo::getVolumePath(const Int_t* ind, Char_t* path)
{
  //--------------------------------------------------------------------
  // This function returns the volume path of a given pad
  //--------------------------------------------------------------------
  Int_t sector = ind[0];

  const Int_t kSize = 100;

  Char_t string1[kSize];
  Char_t string2[kSize];
  Char_t string3[kSize];

  Int_t icopy = -1;
  icopy = sector;

  snprintf(string1, kSize, "/cave_1/B077_1/BSEGMO%i_1/BTOF%i_1", icopy, icopy);

  Bool_t fgHoles = kTRUE;

  Int_t iplate = ind[1];
  Int_t istrip = ind[2];
  if (iplate == 0)
    icopy = istrip;
  if (iplate == 1)
    icopy = istrip + NSTRIPC;
  if (iplate == 2)
    icopy = istrip + NSTRIPC + NSTRIPB;
  if (iplate == 3)
    icopy = istrip + NSTRIPC + NSTRIPB + NSTRIPA;
  if (iplate == 4)
    icopy = istrip + NSTRIPC + 2 * NSTRIPB + NSTRIPA;
  icopy++;
  snprintf(string2, kSize, "FTOA_0/FLTA_0/FSTR_%i", icopy);
  if (fgHoles && (sector == 13 || sector == 14 || sector == 15)) {
    if (iplate < 2)
      snprintf(string2, kSize, "FTOB_0/FLTB_0/FSTR_%i", icopy);
    if (iplate > 2)
      snprintf(string2, kSize, "FTOC_0/FLTC_0/FSTR_%i", icopy);
  }

  Int_t padz = ind[3] + 1;
  Int_t padx = ind[4] + 1;
  snprintf(string3, kSize, "FPCB_1/FSEN_1/FSEZ_%i/FPAD_%i", padz, padx);
  snprintf(path, 2 * kSize, "%s/%s/%s", string1, string2, string3);
}

void Geo::getPos(Int_t* det, Float_t* pos)
{
  //
  // Returns space point coor (x,y,z) (cm)  for Detector
  // Indices  (iSect,iPlate,iStrip,iPadZ,iPadX)
  //
  if (mToBeIntit)
    Init();

  pos[0] =mPadPosition[det[0]][det[1]][det[2]][det[3]][det[4]][0];
  pos[1] =mPadPosition[det[0]][det[1]][det[2]][det[3]][det[4]][1];
  pos[2] =mPadPosition[det[0]][det[1]][det[2]][det[3]][det[4]][2];
}

void Geo::getDetID(Float_t* pos, Int_t* det)
{
  //
  // Returns Detector Indices (iSect,iPlate,iStrip,iPadZ,iPadX)
  // space point coor (x,y,z) (cm)

  if (mToBeIntit)
    Init();

  Float_t posLocal[3];
  for (Int_t ii = 0; ii < 3; ii++)
    posLocal[ii] = pos[ii];

  det[0] = getSector(posLocal);

  fromGlobalToSector(posLocal, det[0]);

  det[1] = getPlate(posLocal);

  det[2] = fromPlateToStrip(posLocal, det[1]);

  det[3] = getPadZ(posLocal);
  det[4] = getPadX(posLocal);
}

void Geo::getVolumeIndices(Int_t index, Int_t *detId)
{
  //
  // Retrieve volume indices from the calibration channel index 
  //
  Int_t npadxstrip = NPADX*NPADZ;

  detId[0] = index/npadxstrip/NSTRIPXSECTOR;

  Int_t dummyStripPerModule = 
    ( index - ( NSTRIPXSECTOR*npadxstrip*detId[0]) ) / npadxstrip;
  if (dummyStripPerModule<NSTRIPC) {
    detId[1] = 0;
    detId[2] = dummyStripPerModule;
  }
  else if (dummyStripPerModule>=NSTRIPC && dummyStripPerModule<NSTRIPC+NSTRIPB) {
    detId[1] = 1;
    detId[2] = dummyStripPerModule-NSTRIPC;
  }
  else if (dummyStripPerModule>=NSTRIPC+NSTRIPB && dummyStripPerModule<NSTRIPC+NSTRIPB+NSTRIPA) {
    detId[1] = 2;
    detId[2] = dummyStripPerModule-NSTRIPC-NSTRIPB;
  }
  else if (dummyStripPerModule>=NSTRIPC+NSTRIPB+NSTRIPA && dummyStripPerModule<NSTRIPC+NSTRIPB+NSTRIPA+NSTRIPB) {
    detId[1] = 3;
    detId[2] = dummyStripPerModule-NSTRIPC-NSTRIPB-NSTRIPA;
  }
  else if (dummyStripPerModule>=NSTRIPC+NSTRIPB+NSTRIPA+NSTRIPB && dummyStripPerModule<NSTRIPXSECTOR) {
    detId[1] = 4;
    detId[2] = dummyStripPerModule-NSTRIPC-NSTRIPB-NSTRIPA-NSTRIPB;
  }

  Int_t padPerStrip = ( index - ( NSTRIPXSECTOR*npadxstrip*detId[0]) ) - dummyStripPerModule*npadxstrip;

  detId[3] = padPerStrip / NPADX; // padZ
  detId[4] = padPerStrip - detId[3]*NPADX; // padX

}

Int_t Geo::getIndex(const Int_t * detId)
{
  //Retrieve calibration channel index 
  Int_t isector = detId[0];
  if (isector >= NSECTORS){
    printf("Wrong sector number in TOF (%d) !\n",isector);
    return -1;
  }
  Int_t iplate = detId[1];
  if (iplate >= NPLATES){
    printf("Wrong plate number in TOF (%d) !\n",iplate);
    return -1;
  }
  Int_t istrip = detId[2];
  Int_t stripOffset = getStripNumberPerSM(iplate,istrip);
  if (stripOffset==-1) {
    printf("Wrong strip number per SM in TOF (%d) !\n",stripOffset);
    return -1;
  }

  Int_t ipadz = detId[3];
  Int_t ipadx = detId[4];

  Int_t idet = ((2*(NSTRIPC+NSTRIPB)+NSTRIPA)*NPADZ*NPADX)*isector +
               (stripOffset*NPADZ*NPADX)+
               (NPADX)*ipadz+
                ipadx;
  return idet;
}

Int_t Geo::getStripNumberPerSM(Int_t iplate, Int_t istrip)
{
  //
  // Get the serial number of the TOF strip number istrip [0,14/18],
  //   in the module number iplate [0,4].
  // This number will range in [0,90].
  //

  Int_t index = -1;

  Bool_t check = (
                  (iplate<0 || iplate>=NPLATES)
                  ||
                  (
                   (iplate==2 && (istrip<0 || istrip>=NSTRIPA))
                   ||
                   (iplate!=2 && (istrip<0 || istrip>=NSTRIPC))
                   )
                  );

  if (iplate<0 || iplate>=NPLATES)
    LOG(ERROR) << "getStripNumberPerSM : " << "Wrong plate number in TOF (" << iplate << ")!\n";

  if (
      (iplate==2 && (istrip<0 || istrip>=NSTRIPA))
      ||
      (iplate!=2 && (istrip<0 || istrip>=NSTRIPC))
      )
    LOG(ERROR) << "getStripNumberPerSM : " << " Wrong strip number in TOF (strip=" << istrip << " in the plate= " << iplate << ")!\n";

  Int_t stripOffset = 0;
  switch (iplate) {
  case 0:
    stripOffset = 0;
    break;
  case 1:
    stripOffset = NSTRIPC;
    break;
  case 2:
    stripOffset = NSTRIPC+NSTRIPB;
    break;
  case 3:
    stripOffset = NSTRIPC+NSTRIPB+NSTRIPA;
    break;
  case 4:
    stripOffset = NSTRIPC+NSTRIPB+NSTRIPA+NSTRIPB;
    break;
  };

  if (!check) index = stripOffset + istrip;

  return index;

}

void Geo::fromGlobalToSector(Float_t* pos, Int_t isector)
{
  if (isector == -1) {
    //LOG(ERROR) << "Sector Index not valid (-1)\n";
    return;
  }

  // ALICE reference frame -> B071/B074/B075 = BTO1/2/3 reference frame
  rotateToSector(pos, isector);

  Float_t step[3] = { 0., 0., static_cast<Float_t>((RMAX + RMIN) * 0.5) };
  translate(pos, step);

  // B071/B074/B075 = BTO1/2/3 reference frame -> FTOA = FLTA reference frame
  rotateToSector(pos, NSECTORS);
}

Int_t Geo::fromPlateToStrip(Float_t* pos, Int_t iplate)
{
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

  Float_t step[3];

  // FTOA/B/C = FLTA/B/C reference frame -> FSTR reference frame
  for (Int_t istrip = 0; istrip < nstrips; istrip++) {
    Float_t posLoc2[3] = { pos[0], pos[1], pos[2] };

    step[0] = 0.;
    step[1] = getHeights(iplate, istrip);
    step[2] = -getDistances(iplate, istrip);
    translate(posLoc2, step);
    rotateToStrip(posLoc2, iplate, istrip);
    if ((TMath::Abs(posLoc2[0]) <= STRIPLENGTH * 0.5) && (TMath::Abs(posLoc2[1]) <= HSTRIPY * 0.5) &&
        (TMath::Abs(posLoc2[2]) <= WCPCBZ * 0.5)) {
      step[0] = -0.5 * NPADX * XPAD;
      step[1] = 0.;
      step[2] = -0.5 * NPADZ * ZPAD;
      translate(posLoc2, step);

      for (Int_t jj = 0; jj < 3; jj++)
        pos[jj] = posLoc2[jj];

      return istrip;
    }
  }
  return -1;
}

Int_t Geo::getSector(const Float_t* pos)
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

void Geo::getPadDxDyDz(const Float_t * pos,Int_t * det, Float_t * DeltaPos) 
{  
  //  
  // Returns the x coordinate in the Pad reference frame  
  //  
  if (mToBeIntit) 
    Init();     
  
  for (Int_t ii = 0; ii < 3; ii++)  
    DeltaPos[ii] = pos[ii]; 
  
  det[0] = getSector(DeltaPos);  
  fromGlobalToSector(DeltaPos, det[0]);  
  det[1] = getPlate(DeltaPos);  
  det[2] = fromPlateToStrip(DeltaPos, det[1]); 
  det[3] = getPadZ(DeltaPos);  
  det[4] = getPadX(DeltaPos);  
  // translate to the pad center 
  
  Float_t step[3]; 
  
  step[0] = (det[4]+0.5)*XPAD; 
  
  step[1] = 0.; 
 
  step[2] = (det[3]+0.5)*ZPAD;
  translate(DeltaPos,step); 
} 

Int_t Geo::getPlate(const Float_t* pos)
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

Int_t Geo::getPadZ(const Float_t* pos)
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
Int_t Geo::getPadX(const Float_t* pos)
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

void Geo::translate(Float_t* xyz, Float_t translationVector[3])
{
  //
  // Return the vector xyz translated by translationVector vector
  //

  for (Int_t ii = 0; ii < 3; ii++)
    xyz[ii] -= translationVector[ii];

  return;
}

void Geo::rotateToSector(Float_t* xyz, Int_t isector)
{
  if (mToBeIntit)
    Init();

  Float_t xyzDummy[3] = { 0., 0., 0. };

  for (Int_t ii = 0; ii < 3; ii++) {
    xyzDummy[ii] = xyz[0] * mRotationMatrixSector[isector][ii][0] + xyz[1] * mRotationMatrixSector[isector][ii][1] +
                   xyz[2] * mRotationMatrixSector[isector][ii][2];
  }

  for (Int_t ii = 0; ii < 3; ii++)
    xyz[ii] = xyzDummy[ii];

  return;
}

void Geo::rotateToStrip(Float_t* xyz, Int_t iplate, Int_t istrip)
{
  Float_t xyzDummy[3] = { 0., 0., 0. };

  for (Int_t ii = 0; ii < 3; ii++) {
    xyzDummy[ii] = xyz[0] * mRotationMatrixPlateStrip[iplate][istrip][ii][0] +
                   xyz[1] * mRotationMatrixPlateStrip[iplate][istrip][ii][1] +
                   xyz[2] * mRotationMatrixPlateStrip[iplate][istrip][ii][2];
  }

  for (Int_t ii = 0; ii < 3; ii++)
    xyz[ii] = xyzDummy[ii];

  return;
}

void Geo::rotate(Float_t* xyz, Double_t rotationAngles[6])
{
  //
  // Return the vector xyz rotated according to the rotationAngles angles
  //

  /*
    TRotMatrix *matrix = new TRotMatrix("matrix","matrix", angles[0], angles[1],
    angles[2], angles[3],
    angles[4], angles[5]);
  */

  for (Int_t ii = 0; ii < 6; ii++)
    rotationAngles[ii] *= TMath::DegToRad();

  Float_t xyzDummy[3] = { 0., 0., 0. };

  for (Int_t ii = 0; ii < 3; ii++) {
    xyzDummy[ii] = xyz[0] * TMath::Sin(rotationAngles[2 * ii]) * TMath::Cos(rotationAngles[2 * ii + 1]) +
                   xyz[1] * TMath::Sin(rotationAngles[2 * ii]) * TMath::Sin(rotationAngles[2 * ii + 1]) +
                   xyz[2] * TMath::Cos(rotationAngles[2 * ii]);
  }

  for (Int_t ii = 0; ii < 3; ii++)
    xyz[ii] = xyzDummy[ii];

  return;
}

void Geo::antiRotate(Float_t* xyz, Double_t rotationAngles[6])
{
  //
  // Rotates the vector xyz acordint to the rotationAngles
  //

  for (Int_t ii = 0; ii < 6; ii++)
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

  for (Int_t ii = 0; ii < 3; ii++)
    xyz[ii] = xyzDummy[ii];

  return;
}

Int_t Geo::getIndexFromEquipment(Int_t icrate, Int_t islot, Int_t ichain, Int_t itdc)
{
  return 0; // to be implemented
}

// cable length map
Float_t Geo::CABLELENGTH[Geo::kNCrate][10][Geo::kNChain][Geo::kNTdc / 3] = { // Cable Lengths
  {
    //crate 0
    { { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 } },                                         //slot 3, chain
    { { 175.0, 175.0, 175.0, 187.5, 187.5 }, { 200.0, 200.0, 212.5, 212.5, 212.5 } }, //slot 4, chain
    { { 212.5, 237.5, 225.0, 237.5, 237.5 }, { 250.0, 262.5, 262.5, 262.5, 262.5 } }, //slot 5, chain
    { { 262.5, 262.5, 275.0, 275.0, 275.0 }, { 287.5, 287.5, 300.0, 300.0, 300.0 } }, //slot 6, chain
    { { 312.5, 300.0, 312.5, 312.5, 325.0 }, { 337.5, 325.0, 337.5, 337.5, 350.0 } }, //slot 7, chain
    { { 350.0, 362.5, 350.0, 362.5, 362.5 }, { 387.5, 387.5, 400.0, 400.0, 412.5 } }, //slot 8, chain
    { { 387.5, 387.5, 387.5, 387.5, 400.0 }, { 412.5, 412.5, 425.0, 412.5, 425.0 } }, //slot 9, chain
    { { 425.0, 425.0, 425.0, 425.0, 425.0 }, { 462.5, 450.0, 450.0, 462.5, 450.0 } }, //slot 10, chain
    { { 462.5, 475.0, 475.0, 475.0, 475.0 }, { 487.5, 487.5, 487.5, 500.0, 500.0 } }, //slot 11, chain
    { { 500.0, 512.5, 512.5, 512.5, 512.5 }, { 525.0, 537.5, 537.5, 550.0, 550.0 } }  //slot 12, chain
  },
  {
    //crate 1
    { { 162.5, 0, 0, 0, 0 }, { 175.0, 0, 0, 0, 0 } },                                 //slot 3, chain
    { { 187.5, 200.0, 200.0, 187.5, 187.5 }, { 212.5, 225.0, 225.0, 212.5, 212.5 } }, //slot 4, chain
    { { 250.0, 250.0, 237.5, 237.5, 237.5 }, { 275.0, 275.0, 275.0, 262.5, 262.5 } }, //slot 5, chain
    { { 287.5, 287.5, 287.5, 275.0, 275.0 }, { 312.5, 312.5, 312.5, 300.0, 300.0 } }, //slot 6, chain
    { { 337.5, 337.5, 325.0, 325.0, 325.0 }, { 362.5, 362.5, 350.0, 350.0, 350.0 } }, //slot 7, chain
    { { 375.0, 375.0, 375.0, 375.0, 375.0 }, { 412.5, 412.5, 400.0, 400.0, 400.0 } }, //slot 8, chain
    { { 412.5, 412.5, 412.5, 412.5, 412.5 }, { 437.5, 437.5, 437.5, 437.5, 437.5 } }, //slot 9, chain
    { { 450.0, 450.0, 437.5, 437.5, 437.5 }, { 475.0, 475.0, 462.5, 450.0, 462.5 } }, //slot 10, chain
    { { 487.5, 487.5, 487.5, 475.0, 475.0 }, { 512.5, 512.5, 512.5, 500.0, 500.0 } }, //slot 11, chain
    { { 525.0, 525.0, 512.5, 512.5, 512.5 }, { 550.0, 550.0, 550.0, 537.5, 537.5 } }  //slot 12, chain
  },
  {
    //crate 2
    { { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 } },                                         //slot 3, chain
    { { 175.0, 175.0, 175.0, 187.5, 187.5 }, { 200.0, 200.0, 212.5, 212.5, 212.5 } }, //slot 4, chain
    { { 212.5, 237.5, 225.0, 237.5, 237.5 }, { 250.0, 262.5, 262.5, 262.5, 262.5 } }, //slot 5, chain
    { { 262.5, 262.5, 275.0, 275.0, 275.0 }, { 287.5, 287.5, 300.0, 300.0, 300.0 } }, //slot 6, chain
    { { 312.5, 300.0, 312.5, 312.5, 325.0 }, { 337.5, 325.0, 337.5, 337.5, 350.0 } }, //slot 7, chain
    { { 350.0, 362.5, 350.0, 362.5, 362.5 }, { 387.5, 387.5, 400.0, 400.0, 412.5 } }, //slot 8, chain
    { { 387.5, 387.5, 387.5, 387.5, 400.0 }, { 412.5, 412.5, 425.0, 412.5, 425.0 } }, //slot 9, chain
    { { 425.0, 425.0, 425.0, 425.0, 425.0 }, { 462.5, 450.0, 450.0, 462.5, 450.0 } }, //slot 10, chain
    { { 462.5, 475.0, 475.0, 475.0, 475.0 }, { 487.5, 487.5, 487.5, 500.0, 500.0 } }, //slot 11, chain
    { { 500.0, 512.5, 512.5, 512.5, 512.5 }, { 525.0, 537.5, 537.5, 550.0, 550.0 } }  //slot 12, chain
  },
  {
    //crate 3
    { { 162.5, 0, 0, 0, 0 }, { 175.0, 0, 0, 0, 0 } },                                 //slot 3, chain
    { { 187.5, 200.0, 200.0, 187.5, 187.5 }, { 212.5, 225.0, 225.0, 212.5, 212.5 } }, //slot 4, chain
    { { 250.0, 250.0, 237.5, 237.5, 237.5 }, { 275.0, 275.0, 275.0, 262.5, 262.5 } }, //slot 5, chain
    { { 287.5, 287.5, 287.5, 275.0, 275.0 }, { 312.5, 312.5, 312.5, 300.0, 300.0 } }, //slot 6, chain
    { { 337.5, 337.5, 325.0, 325.0, 325.0 }, { 362.5, 362.5, 350.0, 350.0, 350.0 } }, //slot 7, chain
    { { 375.0, 375.0, 375.0, 375.0, 375.0 }, { 412.5, 412.5, 400.0, 400.0, 400.0 } }, //slot 8, chain
    { { 412.5, 412.5, 412.5, 412.5, 412.5 }, { 437.5, 437.5, 437.5, 437.5, 437.5 } }, //slot 9, chain
    { { 450.0, 450.0, 437.5, 437.5, 437.5 }, { 475.0, 475.0, 462.5, 450.0, 462.5 } }, //slot 10, chain
    { { 487.5, 487.5, 487.5, 475.0, 475.0 }, { 512.5, 512.5, 512.5, 500.0, 500.0 } }, //slot 11, chain
    { { 525.0, 525.0, 512.5, 512.5, 512.5 }, { 550.0, 550.0, 550.0, 537.5, 537.5 } }  //slot 12, chain
  },
  {
    //crate 4
    { { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 } },                                         //slot 3, chain
    { { 175.0, 175.0, 175.0, 187.5, 187.5 }, { 200.0, 200.0, 212.5, 212.5, 212.5 } }, //slot 4, chain
    { { 212.5, 237.5, 225.0, 237.5, 237.5 }, { 250.0, 262.5, 262.5, 262.5, 262.5 } }, //slot 5, chain
    { { 262.5, 262.5, 275.0, 275.0, 275.0 }, { 287.5, 287.5, 300.0, 300.0, 300.0 } }, //slot 6, chain
    { { 312.5, 300.0, 312.5, 312.5, 325.0 }, { 337.5, 325.0, 337.5, 337.5, 350.0 } }, //slot 7, chain
    { { 350.0, 362.5, 350.0, 362.5, 362.5 }, { 387.5, 387.5, 400.0, 400.0, 400.0 } }, //slot 8, chain
    { { 387.5, 387.5, 387.5, 400.0, 400.0 }, { 412.5, 412.5, 412.5, 425.0, 425.0 } }, //slot 9, chain
    { { 425.0, 425.0, 425.0, 437.5, 437.5 }, { 462.5, 450.0, 450.0, 462.5, 450.0 } }, //slot 10, chain
    { { 475.0, 475.0, 487.5, 487.5, 487.5 }, { 487.5, 487.5, 500.0, 500.0, 500.0 } }, //slot 11, chain
    { { 500.0, 512.5, 512.5, 512.5, 512.5 }, { 525.0, 537.5, 537.5, 537.5, 550.0 } }  //slot 12, chain
  },
  {
    //crate 5
    { { 162.5, 0, 0, 0, 0 }, { 175.0, 0, 0, 0, 0 } },                                 //slot 3, chain
    { { 187.5, 200.0, 200.0, 187.5, 187.5 }, { 212.5, 225.0, 225.0, 212.5, 212.5 } }, //slot 4, chain
    { { 250.0, 250.0, 237.5, 237.5, 237.5 }, { 275.0, 275.0, 275.0, 262.5, 262.5 } }, //slot 5, chain
    { { 287.5, 287.5, 287.5, 275.0, 275.0 }, { 312.5, 312.5, 312.5, 300.0, 300.0 } }, //slot 6, chain
    { { 337.5, 337.5, 325.0, 325.0, 325.0 }, { 362.5, 362.5, 350.0, 350.0, 350.0 } }, //slot 7, chain
    { { 375.0, 375.0, 375.0, 375.0, 375.0 }, { 412.5, 412.5, 412.5, 412.5, 400.0 } }, //slot 8, chain
    { { 425.0, 412.5, 412.5, 412.5, 412.5 }, { 437.5, 437.5, 437.5, 437.5, 425.0 } }, //slot 9, chain
    { { 450.0, 450.0, 450.0, 437.5, 437.5 }, { 475.0, 475.0, 475.0, 462.5, 462.5 } }, //slot 10, chain
    { { 487.5, 487.5, 487.5, 487.5, 487.5 }, { 500.0, 512.5, 512.5, 512.5, 487.5 } }, //slot 11, chain
    { { 525.0, 525.0, 512.5, 525.0, 512.5 }, { 550.0, 550.0, 537.5, 537.5, 537.5 } }  //slot 12, chain
  },
  {
    //crate 6
    { { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 } },                                         //slot 3, chain
    { { 175.0, 175.0, 175.0, 187.5, 187.5 }, { 200.0, 200.0, 212.5, 212.5, 212.5 } }, //slot 4, chain
    { { 212.5, 237.5, 225.0, 237.5, 237.5 }, { 250.0, 262.5, 262.5, 262.5, 262.5 } }, //slot 5, chain
    { { 262.5, 262.5, 275.0, 275.0, 275.0 }, { 287.5, 287.5, 300.0, 300.0, 300.0 } }, //slot 6, chain
    { { 312.5, 300.0, 312.5, 312.5, 325.0 }, { 337.5, 325.0, 337.5, 337.5, 350.0 } }, //slot 7, chain
    { { 350.0, 362.5, 350.0, 362.5, 362.5 }, { 387.5, 387.5, 400.0, 400.0, 400.0 } }, //slot 8, chain
    { { 387.5, 387.5, 387.5, 400.0, 400.0 }, { 412.5, 412.5, 412.5, 425.0, 425.0 } }, //slot 9, chain
    { { 425.0, 425.0, 425.0, 437.5, 437.5 }, { 462.5, 450.0, 450.0, 462.5, 450.0 } }, //slot 10, chain
    { { 475.0, 475.0, 487.5, 487.5, 487.5 }, { 487.5, 487.5, 500.0, 500.0, 500.0 } }, //slot 11, chain
    { { 500.0, 512.5, 512.5, 512.5, 512.5 }, { 525.0, 537.5, 537.5, 537.5, 550.0 } }  //slot 12, chain
  },
  {
    //crate 7
    { { 162.5, 0, 0, 0, 0 }, { 175.0, 0, 0, 0, 0 } },                                 //slot 3, chain
    { { 187.5, 200.0, 200.0, 187.5, 187.5 }, { 212.5, 225.0, 225.0, 212.5, 212.5 } }, //slot 4, chain
    { { 250.0, 250.0, 237.5, 237.5, 237.5 }, { 275.0, 275.0, 275.0, 262.5, 262.5 } }, //slot 5, chain
    { { 287.5, 287.5, 287.5, 275.0, 275.0 }, { 312.5, 312.5, 312.5, 300.0, 300.0 } }, //slot 6, chain
    { { 337.5, 337.5, 325.0, 325.0, 325.0 }, { 362.5, 362.5, 350.0, 350.0, 350.0 } }, //slot 7, chain
    { { 375.0, 375.0, 375.0, 375.0, 375.0 }, { 412.5, 412.5, 412.5, 412.5, 400.0 } }, //slot 8, chain
    { { 425.0, 412.5, 412.5, 412.5, 412.5 }, { 437.5, 437.5, 437.5, 437.5, 425.0 } }, //slot 9, chain
    { { 450.0, 450.0, 450.0, 437.5, 437.5 }, { 475.0, 475.0, 475.0, 462.5, 462.5 } }, //slot 10, chain
    { { 487.5, 487.5, 487.5, 487.5, 487.5 }, { 500.0, 512.5, 512.5, 512.5, 487.5 } }, //slot 11, chain
    { { 525.0, 525.0, 512.5, 525.0, 512.5 }, { 550.0, 550.0, 537.5, 537.5, 537.5 } }  //slot 12, chain
  },
  {
    //crate 8
    { { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 } },                                         //slot 3, chain
    { { 175.0, 175.0, 175.0, 187.5, 187.5 }, { 200.0, 200.0, 212.5, 212.5, 212.5 } }, //slot 4, chain
    { { 212.5, 237.5, 225.0, 237.5, 237.5 }, { 250.0, 262.5, 262.5, 262.5, 262.5 } }, //slot 5, chain
    { { 262.5, 262.5, 275.0, 275.0, 275.0 }, { 287.5, 287.5, 300.0, 300.0, 300.0 } }, //slot 6, chain
    { { 312.5, 300.0, 312.5, 312.5, 325.0 }, { 337.5, 325.0, 337.5, 337.5, 350.0 } }, //slot 7, chain
    { { 350.0, 362.5, 350.0, 362.5, 362.5 }, { 387.5, 387.5, 400.0, 400.0, 400.0 } }, //slot 8, chain
    { { 387.5, 387.5, 400.0, 400.0, 400.0 }, { 412.5, 412.5, 425.0, 425.0, 425.0 } }, //slot 9, chain
    { { 425.0, 425.0, 437.5, 437.5, 437.5 }, { 462.5, 450.0, 450.0, 462.5, 450.0 } }, //slot 10, chain
    { { 475.0, 475.0, 487.5, 487.5, 487.5 }, { 487.5, 487.5, 487.5, 500.0, 500.0 } }, //slot 11, chain
    { { 500.0, 500.0, 512.5, 512.5, 512.5 }, { 525.0, 537.5, 537.5, 537.5, 550.0 } }  //slot 12, chain
  },
  {
    //crate 9
    { { 162.5, 0, 0, 0, 0 }, { 175.0, 0, 0, 0, 0 } },                                 //slot 3, chain
    { { 187.5, 200.0, 200.0, 187.5, 187.5 }, { 212.5, 225.0, 225.0, 212.5, 212.5 } }, //slot 4, chain
    { { 250.0, 250.0, 237.5, 237.5, 237.5 }, { 275.0, 275.0, 275.0, 262.5, 262.5 } }, //slot 5, chain
    { { 287.5, 287.5, 287.5, 275.0, 275.0 }, { 312.5, 312.5, 312.5, 300.0, 300.0 } }, //slot 6, chain
    { { 337.5, 337.5, 325.0, 325.0, 325.0 }, { 362.5, 362.5, 350.0, 350.0, 350.0 } }, //slot 7, chain
    { { 375.0, 375.0, 375.0, 375.0, 375.0 }, { 412.5, 412.5, 412.5, 412.5, 400.0 } }, //slot 8, chain
    { { 425.0, 412.5, 412.5, 412.5, 412.5 }, { 437.5, 437.5, 437.5, 425.0, 425.0 } }, //slot 9, chain
    { { 450.0, 450.0, 450.0, 437.5, 437.5 }, { 475.0, 475.0, 475.0, 462.5, 462.5 } }, //slot 10, chain
    { { 487.5, 487.5, 487.5, 487.5, 487.5 }, { 512.5, 512.5, 512.5, 512.5, 500.0 } }, //slot 11, chain
    { { 525.0, 525.0, 525.0, 512.5, 512.5 }, { 550.0, 550.0, 537.5, 537.5, 537.5 } }  //slot 12, chain
  },
  {
    //crate 10
    { { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 } },                                         //slot 3, chain
    { { 175.0, 175.0, 175.0, 187.5, 187.5 }, { 200.0, 200.0, 212.5, 212.5, 212.5 } }, //slot 4, chain
    { { 212.5, 237.5, 225.0, 237.5, 237.5 }, { 250.0, 262.5, 262.5, 262.5, 262.5 } }, //slot 5, chain
    { { 262.5, 262.5, 275.0, 275.0, 275.0 }, { 287.5, 287.5, 300.0, 300.0, 300.0 } }, //slot 6, chain
    { { 312.5, 300.0, 312.5, 312.5, 325.0 }, { 337.5, 325.0, 337.5, 337.5, 350.0 } }, //slot 7, chain
    { { 350.0, 362.5, 350.0, 362.5, 362.5 }, { 387.5, 387.5, 400.0, 400.0, 400.0 } }, //slot 8, chain
    { { 387.5, 387.5, 400.0, 400.0, 400.0 }, { 412.5, 412.5, 425.0, 425.0, 425.0 } }, //slot 9, chain
    { { 425.0, 425.0, 437.5, 437.5, 437.5 }, { 462.5, 450.0, 450.0, 462.5, 450.0 } }, //slot 10, chain
    { { 475.0, 475.0, 487.5, 487.5, 487.5 }, { 487.5, 487.5, 487.5, 500.0, 500.0 } }, //slot 11, chain
    { { 500.0, 500.0, 512.5, 512.5, 512.5 }, { 525.0, 537.5, 537.5, 537.5, 550.0 } }  //slot 12, chain
  },
  {
    //crate 11
    { { 162.5, 0, 0, 0, 0 }, { 175.0, 0, 0, 0, 0 } },                                 //slot 3, chain
    { { 187.5, 200.0, 200.0, 187.5, 187.5 }, { 212.5, 225.0, 225.0, 212.5, 212.5 } }, //slot 4, chain
    { { 250.0, 250.0, 237.5, 237.5, 237.5 }, { 275.0, 275.0, 275.0, 262.5, 262.5 } }, //slot 5, chain
    { { 287.5, 287.5, 287.5, 275.0, 275.0 }, { 312.5, 312.5, 312.5, 300.0, 300.0 } }, //slot 6, chain
    { { 337.5, 337.5, 325.0, 325.0, 325.0 }, { 362.5, 362.5, 350.0, 350.0, 350.0 } }, //slot 7, chain
    { { 375.0, 375.0, 375.0, 375.0, 375.0 }, { 412.5, 412.5, 412.5, 412.5, 400.0 } }, //slot 8, chain
    { { 425.0, 412.5, 412.5, 412.5, 412.5 }, { 437.5, 437.5, 437.5, 425.0, 425.0 } }, //slot 9, chain
    { { 450.0, 450.0, 450.0, 437.5, 437.5 }, { 475.0, 475.0, 475.0, 462.5, 462.5 } }, //slot 10, chain
    { { 487.5, 487.5, 487.5, 487.5, 487.5 }, { 512.5, 512.5, 512.5, 512.5, 500.0 } }, //slot 11, chain
    { { 525.0, 525.0, 525.0, 512.5, 512.5 }, { 550.0, 550.0, 537.5, 537.5, 537.5 } }  //slot 12, chain
  },
  {
    //crate 12
    { { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 } },                                         //slot 3, chain
    { { 175.0, 175.0, 175.0, 187.5, 187.5 }, { 200.0, 200.0, 212.5, 212.5, 212.5 } }, //slot 4, chain
    { { 212.5, 237.5, 225.0, 237.5, 237.5 }, { 250.0, 262.5, 262.5, 262.5, 262.5 } }, //slot 5, chain
    { { 262.5, 262.5, 275.0, 275.0, 275.0 }, { 287.5, 287.5, 300.0, 300.0, 300.0 } }, //slot 6, chain
    { { 312.5, 300.0, 312.5, 312.5, 325.0 }, { 337.5, 325.0, 337.5, 337.5, 350.0 } }, //slot 7, chain
    { { 350.0, 362.5, 350.0, 362.5, 362.5 }, { 387.5, 387.5, 400.0, 400.0, 400.0 } }, //slot 8, chain
    { { 387.5, 387.5, 387.5, 400.0, 400.0 }, { 412.5, 412.5, 412.5, 425.0, 425.0 } }, //slot 9, chain
    { { 425.0, 425.0, 425.0, 437.5, 437.5 }, { 462.5, 450.0, 450.0, 462.5, 450.0 } }, //slot 10, chain
    { { 475.0, 487.5, 487.5, 487.5, 487.5 }, { 487.5, 487.5, 500.0, 500.0, 500.0 } }, //slot 11, chain
    { { 500.0, 512.5, 512.5, 512.5, 512.5 }, { 525.0, 537.5, 537.5, 537.5, 550.0 } }  //slot 12, chain
  },
  {
    //crate 13
    { { 162.5, 0, 0, 0, 0 }, { 175.0, 0, 0, 0, 0 } },                                 //slot 3, chain
    { { 187.5, 200.0, 200.0, 187.5, 187.5 }, { 212.5, 225.0, 225.0, 212.5, 212.5 } }, //slot 4, chain
    { { 250.0, 250.0, 237.5, 237.5, 237.5 }, { 275.0, 275.0, 275.0, 262.5, 262.5 } }, //slot 5, chain
    { { 287.5, 287.5, 287.5, 275.0, 275.0 }, { 362.5, 362.5, 350.0, 350.0, 350.0 } }, //slot 6, chain
    { { 337.5, 337.5, 325.0, 325.0, 325.0 }, { 362.5, 362.5, 350.0, 350.0, 350.0 } }, //slot 7, chain
    { { 375.0, 375.0, 375.0, 375.0, 375.0 }, { 412.5, 412.5, 412.5, 412.5, 400.0 } }, //slot 8, chain
    { { 425.0, 412.5, 412.5, 412.5, 412.5 }, { 437.5, 437.5, 437.5, 437.5, 425.0 } }, //slot 9, chain
    { { 450.0, 450.0, 450.0, 437.5, 437.5 }, { 475.0, 475.0, 475.0, 462.5, 462.5 } }, //slot 10, chain
    { { 487.5, 487.5, 487.5, 487.5, 487.5 }, { 512.5, 512.5, 512.5, 500.0, 500.0 } }, //slot 11, chain
    { { 525.0, 525.0, 525.0, 525.0, 512.5 }, { 550.0, 550.0, 537.5, 537.5, 537.5 } }  //slot 12, chain
  },
  {
    //crate 14
    { { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 } },                                         //slot 3, chain
    { { 175.0, 175.0, 175.0, 187.5, 187.5 }, { 200.0, 200.0, 212.5, 212.5, 212.5 } }, //slot 4, chain
    { { 212.5, 237.5, 225.0, 237.5, 237.5 }, { 250.0, 262.5, 262.5, 262.5, 262.5 } }, //slot 5, chain
    { { 262.5, 262.5, 275.0, 275.0, 275.0 }, { 287.5, 287.5, 300.0, 300.0, 300.0 } }, //slot 6, chain
    { { 312.5, 300.0, 312.5, 312.5, 325.0 }, { 337.5, 325.0, 337.5, 337.5, 350.0 } }, //slot 7, chain
    { { 350.0, 362.5, 350.0, 362.5, 362.5 }, { 387.5, 387.5, 400.0, 400.0, 400.0 } }, //slot 8, chain
    { { 387.5, 387.5, 387.5, 400.0, 400.0 }, { 412.5, 412.5, 412.5, 425.0, 425.0 } }, //slot 9, chain
    { { 425.0, 425.0, 425.0, 437.5, 437.5 }, { 462.5, 450.0, 450.0, 462.5, 450.0 } }, //slot 10, chain
    { { 475.0, 487.5, 487.5, 487.5, 487.5 }, { 487.5, 487.5, 500.0, 500.0, 500.0 } }, //slot 11, chain
    { { 500.0, 512.5, 512.5, 512.5, 512.5 }, { 525.0, 537.5, 537.5, 537.5, 550.0 } }  //slot 12, chain
  },
  {
    //crate 15
    { { 162.5, 0, 0, 0, 0 }, { 175.0, 0, 0, 0, 0 } },                                 //slot 3, chain
    { { 187.5, 200.0, 200.0, 187.5, 187.5 }, { 212.5, 225.0, 225.0, 212.5, 212.5 } }, //slot 4, chain
    { { 250.0, 250.0, 237.5, 237.5, 237.5 }, { 275.0, 275.0, 275.0, 262.5, 262.5 } }, //slot 5, chain
    { { 287.5, 287.5, 287.5, 275.0, 275.0 }, { 362.5, 362.5, 350.0, 350.0, 350.0 } }, //slot 6, chain
    { { 337.5, 337.5, 325.0, 325.0, 325.0 }, { 362.5, 362.5, 350.0, 350.0, 350.0 } }, //slot 7, chain
    { { 375.0, 375.0, 375.0, 375.0, 375.0 }, { 412.5, 412.5, 412.5, 412.5, 400.0 } }, //slot 8, chain
    { { 425.0, 412.5, 412.5, 412.5, 412.5 }, { 437.5, 437.5, 437.5, 437.5, 425.0 } }, //slot 9, chain
    { { 450.0, 450.0, 450.0, 437.5, 437.5 }, { 475.0, 475.0, 475.0, 462.5, 462.5 } }, //slot 10, chain
    { { 487.5, 487.5, 487.5, 487.5, 487.5 }, { 512.5, 512.5, 512.5, 500.0, 500.0 } }, //slot 11, chain
    { { 525.0, 525.0, 525.0, 525.0, 512.5 }, { 550.0, 550.0, 537.5, 537.5, 537.5 } }  //slot 12, chain
  },
  {
    //crate 16
    { { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 } },                                         //slot 3, chain
    { { 175.0, 175.0, 175.0, 187.5, 187.5 }, { 200.0, 200.0, 212.5, 212.5, 212.5 } }, //slot 4, chain
    { { 212.5, 237.5, 225.0, 237.5, 237.5 }, { 250.0, 262.5, 262.5, 262.5, 262.5 } }, //slot 5, chain
    { { 262.5, 262.5, 275.0, 275.0, 275.0 }, { 287.5, 287.5, 300.0, 300.0, 300.0 } }, //slot 6, chain
    { { 312.5, 300.0, 312.5, 312.5, 325.0 }, { 337.5, 325.0, 337.5, 337.5, 350.0 } }, //slot 7, chain
    { { 350.0, 362.5, 350.0, 362.5, 362.5 }, { 387.5, 387.5, 400.0, 400.0, 400.0 } }, //slot 8, chain
    { { 387.5, 387.5, 400.0, 400.0, 400.0 }, { 412.5, 412.5, 425.0, 425.0, 425.0 } }, //slot 9, chain
    { { 425.0, 425.0, 437.5, 437.5, 437.5 }, { 462.5, 450.0, 450.0, 462.5, 450.0 } }, //slot 10, chain
    { { 475.0, 475.0, 475.0, 487.5, 487.5 }, { 487.5, 487.5, 487.5, 500.0, 500.0 } }, //slot 11, chain
    { { 500.0, 500.0, 512.5, 512.5, 512.5 }, { 525.0, 537.5, 537.5, 537.5, 550.0 } }  //slot 12, chain
  },
  {
    //crate 17
    { { 162.5, 0, 0, 0, 0 }, { 175.0, 0, 0, 0, 0 } },                                 //slot 3, chain
    { { 187.5, 200.0, 200.0, 187.5, 187.5 }, { 212.5, 225.0, 225.0, 212.5, 212.5 } }, //slot 4, chain
    { { 250.0, 250.0, 237.5, 237.5, 237.5 }, { 275.0, 275.0, 275.0, 262.5, 262.5 } }, //slot 5, chain
    { { 287.5, 287.5, 287.5, 275.0, 275.0 }, { 312.5, 312.5, 312.5, 300.0, 300.0 } }, //slot 6, chain
    { { 337.5, 337.5, 325.0, 325.0, 325.0 }, { 362.5, 362.5, 350.0, 350.0, 350.0 } }, //slot 7, chain
    { { 375.0, 375.0, 375.0, 375.0, 375.0 }, { 412.5, 412.5, 412.5, 412.5, 400.0 } }, //slot 8, chain
    { { 425.0, 412.5, 412.5, 412.5, 412.5 }, { 437.5, 437.5, 437.5, 425.0, 425.0 } }, //slot 9, chain
    { { 450.0, 450.0, 450.0, 437.5, 437.5 }, { 475.0, 475.0, 475.0, 462.5, 462.5 } }, //slot 10, chain
    { { 487.5, 487.5, 487.5, 487.5, 487.5 }, { 512.5, 512.5, 512.5, 512.5, 500.0 } }, //slot 11, chain
    { { 525.0, 525.0, 525.0, 512.5, 512.5 }, { 550.0, 550.0, 537.5, 537.5, 537.5 } }  //slot 12, chain
  },
  {
    //crate 18
    { { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 } },                                         //slot 3, chain
    { { 175.0, 175.0, 175.0, 187.5, 187.5 }, { 200.0, 200.0, 212.5, 212.5, 212.5 } }, //slot 4, chain
    { { 212.5, 237.5, 225.0, 237.5, 237.5 }, { 250.0, 262.5, 262.5, 262.5, 262.5 } }, //slot 5, chain
    { { 262.5, 262.5, 275.0, 275.0, 275.0 }, { 287.5, 287.5, 300.0, 300.0, 300.0 } }, //slot 6, chain
    { { 312.5, 300.0, 312.5, 312.5, 325.0 }, { 337.5, 325.0, 337.5, 337.5, 350.0 } }, //slot 7, chain
    { { 350.0, 362.5, 350.0, 362.5, 362.5 }, { 387.5, 387.5, 400.0, 400.0, 400.0 } }, //slot 8, chain
    { { 387.5, 387.5, 400.0, 400.0, 400.0 }, { 412.5, 412.5, 425.0, 425.0, 425.0 } }, //slot 9, chain
    { { 425.0, 425.0, 437.5, 437.5, 437.5 }, { 462.5, 450.0, 450.0, 462.5, 450.0 } }, //slot 10, chain
    { { 475.0, 475.0, 475.0, 487.5, 487.5 }, { 487.5, 487.5, 487.5, 500.0, 500.0 } }, //slot 11, chain
    { { 500.0, 500.0, 512.5, 512.5, 512.5 }, { 525.0, 537.5, 537.5, 537.5, 550.0 } }  //slot 12, chain
  },
  {
    //crate 19
    { { 162.5, 0, 0, 0, 0 }, { 175.0, 0, 0, 0, 0 } },                                 //slot 3, chain
    { { 187.5, 200.0, 200.0, 187.5, 187.5 }, { 212.5, 225.0, 225.0, 212.5, 212.5 } }, //slot 4, chain
    { { 250.0, 250.0, 237.5, 237.5, 237.5 }, { 275.0, 275.0, 275.0, 262.5, 262.5 } }, //slot 5, chain
    { { 287.5, 287.5, 287.5, 275.0, 275.0 }, { 312.5, 312.5, 312.5, 300.0, 300.0 } }, //slot 6, chain
    { { 337.5, 337.5, 325.0, 325.0, 325.0 }, { 362.5, 362.5, 350.0, 350.0, 350.0 } }, //slot 7, chain
    { { 375.0, 375.0, 375.0, 375.0, 375.0 }, { 412.5, 412.5, 412.5, 412.5, 400.0 } }, //slot 8, chain
    { { 425.0, 412.5, 412.5, 412.5, 412.5 }, { 437.5, 437.5, 437.5, 425.0, 425.0 } }, //slot 9, chain
    { { 450.0, 450.0, 450.0, 437.5, 437.5 }, { 475.0, 475.0, 475.0, 462.5, 462.5 } }, //slot 10, chain
    { { 487.5, 487.5, 487.5, 487.5, 487.5 }, { 512.5, 512.5, 512.5, 512.5, 500.0 } }, //slot 11, chain
    { { 525.0, 525.0, 525.0, 512.5, 512.5 }, { 550.0, 550.0, 537.5, 537.5, 537.5 } }  //slot 12, chain
  },
  {
    //crate 20
    { { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 } },                                         //slot 3, chain
    { { 175.0, 175.0, 175.0, 187.5, 187.5 }, { 200.0, 200.0, 212.5, 212.5, 212.5 } }, //slot 4, chain
    { { 212.5, 237.5, 225.0, 237.5, 237.5 }, { 250.0, 262.5, 262.5, 262.5, 262.5 } }, //slot 5, chain
    { { 262.5, 262.5, 275.0, 275.0, 275.0 }, { 287.5, 287.5, 300.0, 300.0, 300.0 } }, //slot 6, chain
    { { 312.5, 300.0, 312.5, 312.5, 325.0 }, { 337.5, 325.0, 337.5, 337.5, 350.0 } }, //slot 7, chain
    { { 350.0, 362.5, 350.0, 362.5, 362.5 }, { 387.5, 387.5, 400.0, 400.0, 400.0 } }, //slot 8, chain
    { { 387.5, 387.5, 387.5, 400.0, 400.0 }, { 412.5, 412.5, 425.0, 425.0, 425.0 } }, //slot 9, chain
    { { 425.0, 425.0, 437.5, 437.5, 437.5 }, { 462.5, 450.0, 450.0, 462.5, 450.0 } }, //slot 10, chain
    { { 475.0, 475.0, 487.5, 487.5, 487.5 }, { 487.5, 487.5, 487.5, 500.0, 500.0 } }, //slot 11, chain
    { { 500.0, 500.0, 512.5, 512.5, 512.5 }, { 525.0, 537.5, 537.5, 537.5, 550.0 } }  //slot 12, chain
  },
  {
    //crate 21
    { { 162.5, 0, 0, 0, 0 }, { 175.0, 0, 0, 0, 0 } },                                 //slot 3, chain
    { { 187.5, 200.0, 200.0, 187.5, 187.5 }, { 212.5, 225.0, 225.0, 212.5, 212.5 } }, //slot 4, chain
    { { 250.0, 250.0, 237.5, 237.5, 237.5 }, { 275.0, 275.0, 275.0, 262.5, 262.5 } }, //slot 5, chain
    { { 287.5, 287.5, 287.5, 275.0, 275.0 }, { 312.5, 312.5, 312.5, 300.0, 300.0 } }, //slot 6, chain
    { { 337.5, 337.5, 325.0, 325.0, 325.0 }, { 362.5, 362.5, 350.0, 350.0, 350.0 } }, //slot 7, chain
    { { 375.0, 375.0, 375.0, 375.0, 375.0 }, { 412.5, 412.5, 412.5, 400.0, 400.0 } }, //slot 8, chain
    { { 425.0, 425.0, 412.5, 412.5, 412.5 }, { 437.5, 437.5, 437.5, 437.5, 425.0 } }, //slot 9, chain
    { { 450.0, 450.0, 450.0, 437.5, 437.5 }, { 475.0, 475.0, 475.0, 462.5, 462.5 } }, //slot 10, chain
    { { 487.5, 487.5, 487.5, 487.5, 487.5 }, { 512.5, 512.5, 512.5, 512.5, 500.0 } }, //slot 11, chain
    { { 525.0, 525.0, 525.0, 512.5, 512.5 }, { 550.0, 550.0, 537.5, 537.5, 537.5 } }  //slot 12, chain
  },
  {
    //crate 22
    { { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 } },                                         //slot 3, chain
    { { 175.0, 175.0, 175.0, 187.5, 187.5 }, { 200.0, 200.0, 212.5, 212.5, 212.5 } }, //slot 4, chain
    { { 212.5, 237.5, 225.0, 237.5, 237.5 }, { 250.0, 262.5, 262.5, 262.5, 262.5 } }, //slot 5, chain
    { { 262.5, 262.5, 275.0, 275.0, 275.0 }, { 287.5, 287.5, 300.0, 300.0, 300.0 } }, //slot 6, chain
    { { 312.5, 300.0, 312.5, 312.5, 325.0 }, { 337.5, 325.0, 337.5, 337.5, 350.0 } }, //slot 7, chain
    { { 350.0, 362.5, 350.0, 362.5, 362.5 }, { 387.5, 387.5, 400.0, 400.0, 400.0 } }, //slot 8, chain
    { { 387.5, 387.5, 387.5, 400.0, 400.0 }, { 412.5, 412.5, 425.0, 425.0, 425.0 } }, //slot 9, chain
    { { 425.0, 425.0, 437.5, 437.5, 437.5 }, { 462.5, 450.0, 450.0, 462.5, 450.0 } }, //slot 10, chain
    { { 475.0, 475.0, 487.5, 487.5, 487.5 }, { 487.5, 487.5, 487.5, 500.0, 500.0 } }, //slot 11, chain
    { { 500.0, 500.0, 512.5, 512.5, 512.5 }, { 525.0, 537.5, 537.5, 537.5, 550.0 } }  //slot 12, chain
  },
  {
    //crate 23
    { { 162.5, 0, 0, 0, 0 }, { 175.0, 0, 0, 0, 0 } },                                 //slot 3, chain
    { { 187.5, 200.0, 200.0, 187.5, 187.5 }, { 212.5, 225.0, 225.0, 212.5, 212.5 } }, //slot 4, chain
    { { 250.0, 250.0, 237.5, 237.5, 237.5 }, { 275.0, 275.0, 275.0, 262.5, 262.5 } }, //slot 5, chain
    { { 287.5, 287.5, 287.5, 275.0, 275.0 }, { 312.5, 312.5, 312.5, 300.0, 300.0 } }, //slot 6, chain
    { { 337.5, 337.5, 325.0, 325.0, 325.0 }, { 362.5, 362.5, 350.0, 350.0, 350.0 } }, //slot 7, chain
    { { 375.0, 375.0, 375.0, 375.0, 375.0 }, { 412.5, 412.5, 412.5, 400.0, 400.0 } }, //slot 8, chain
    { { 425.0, 425.0, 412.5, 412.5, 412.5 }, { 437.5, 437.5, 437.5, 437.5, 425.0 } }, //slot 9, chain
    { { 450.0, 450.0, 450.0, 437.5, 437.5 }, { 475.0, 475.0, 475.0, 462.5, 462.5 } }, //slot 10, chain
    { { 487.5, 487.5, 487.5, 487.5, 487.5 }, { 512.5, 512.5, 512.5, 512.5, 500.0 } }, //slot 11, chain
    { { 525.0, 525.0, 525.0, 512.5, 512.5 }, { 550.0, 550.0, 537.5, 537.5, 537.5 } }  //slot 12, chain
  },
  {
    //crate 24
    { { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 } },                                         //slot 3, chain
    { { 175.0, 175.0, 175.0, 187.5, 187.5 }, { 200.0, 200.0, 212.5, 212.5, 212.5 } }, //slot 4, chain
    { { 212.5, 237.5, 225.0, 237.5, 237.5 }, { 250.0, 262.5, 262.5, 262.5, 262.5 } }, //slot 5, chain
    { { 262.5, 262.5, 275.0, 275.0, 275.0 }, { 287.5, 287.5, 300.0, 300.0, 300.0 } }, //slot 6, chain
    { { 312.5, 300.0, 312.5, 312.5, 325.0 }, { 337.5, 325.0, 337.5, 337.5, 350.0 } }, //slot 7, chain
    { { 350.0, 362.5, 350.0, 362.5, 362.5 }, { 387.5, 387.5, 400.0, 400.0, 400.0 } }, //slot 8, chain
    { { 387.5, 387.5, 400.0, 400.0, 400.0 }, { 412.5, 425.0, 425.0, 425.0, 425.0 } }, //slot 9, chain
    { { 425.0, 437.5, 437.5, 437.5, 437.5 }, { 462.5, 450.0, 450.0, 462.5, 450.0 } }, //slot 10, chain
    { { 475.0, 475.0, 487.5, 487.5, 487.5 }, { 487.5, 487.5, 487.5, 500.0, 500.0 } }, //slot 11, chain
    { { 500.0, 500.0, 512.5, 512.5, 512.5 }, { 525.0, 537.5, 537.5, 537.5, 550.0 } }  //slot 12, chain
  },
  {
    //crate 25
    { { 162.5, 0, 0, 0, 0 }, { 175.0, 0, 0, 0, 0 } },                                 //slot 3, chain
    { { 187.5, 200.0, 200.0, 187.5, 187.5 }, { 212.5, 225.0, 225.0, 212.5, 212.5 } }, //slot 4, chain
    { { 250.0, 250.0, 237.5, 237.5, 237.5 }, { 275.0, 275.0, 275.0, 262.5, 262.5 } }, //slot 5, chain
    { { 287.5, 287.5, 287.5, 275.0, 275.0 }, { 312.5, 312.5, 312.5, 300.0, 300.0 } }, //slot 6, chain
    { { 337.5, 337.5, 325.0, 325.0, 325.0 }, { 362.5, 362.5, 350.0, 350.0, 350.0 } }, //slot 7, chain
    { { 375.0, 375.0, 375.0, 375.0, 375.0 }, { 412.5, 412.5, 412.5, 412.5, 400.0 } }, //slot 8, chain
    { { 425.0, 425.0, 412.5, 412.5, 412.5 }, { 437.5, 437.5, 437.5, 437.5, 425.0 } }, //slot 9, chain
    { { 450.0, 450.0, 450.0, 437.5, 437.5 }, { 475.0, 475.0, 475.0, 462.5, 462.5 } }, //slot 10, chain
    { { 487.5, 487.5, 487.5, 487.5, 487.5 }, { 512.5, 512.5, 512.5, 512.5, 500.0 } }, //slot 11, chain
    { { 525.0, 525.0, 525.0, 512.5, 512.5 }, { 550.0, 550.0, 537.5, 537.5, 537.5 } }  //slot 12, chain
  },
  {
    //crate 26
    { { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 } },                                         //slot 3, chain
    { { 175.0, 175.0, 175.0, 187.5, 187.5 }, { 200.0, 200.0, 212.5, 212.5, 212.5 } }, //slot 4, chain
    { { 212.5, 237.5, 225.0, 237.5, 237.5 }, { 250.0, 262.5, 262.5, 262.5, 262.5 } }, //slot 5, chain
    { { 262.5, 262.5, 275.0, 275.0, 275.0 }, { 287.5, 287.5, 300.0, 300.0, 300.0 } }, //slot 6, chain
    { { 312.5, 300.0, 312.5, 312.5, 325.0 }, { 337.5, 325.0, 337.5, 337.5, 350.0 } }, //slot 7, chain
    { { 350.0, 362.5, 350.0, 362.5, 362.5 }, { 387.5, 387.5, 400.0, 400.0, 400.0 } }, //slot 8, chain
    { { 387.5, 387.5, 400.0, 400.0, 400.0 }, { 412.5, 425.0, 425.0, 425.0, 425.0 } }, //slot 9, chain
    { { 425.0, 437.5, 437.5, 437.5, 437.5 }, { 462.5, 450.0, 450.0, 462.5, 450.0 } }, //slot 10, chain
    { { 475.0, 475.0, 487.5, 487.5, 487.5 }, { 487.5, 487.5, 487.5, 500.0, 500.0 } }, //slot 11, chain
    { { 500.0, 500.0, 512.5, 512.5, 512.5 }, { 525.0, 537.5, 537.5, 537.5, 550.0 } }  //slot 12, chain
  },
  {
    //crate 27
    { { 162.5, 0, 0, 0, 0 }, { 175.0, 0, 0, 0, 0 } },                                 //slot 3, chain
    { { 187.5, 200.0, 200.0, 187.5, 187.5 }, { 212.5, 225.0, 225.0, 212.5, 212.5 } }, //slot 4, chain
    { { 250.0, 250.0, 237.5, 237.5, 237.5 }, { 275.0, 275.0, 275.0, 262.5, 262.5 } }, //slot 5, chain
    { { 287.5, 287.5, 287.5, 275.0, 275.0 }, { 312.5, 312.5, 312.5, 300.0, 300.0 } }, //slot 6, chain
    { { 337.5, 337.5, 325.0, 325.0, 325.0 }, { 362.5, 362.5, 350.0, 350.0, 350.0 } }, //slot 7, chain
    { { 375.0, 375.0, 375.0, 375.0, 375.0 }, { 412.5, 412.5, 412.5, 412.5, 400.0 } }, //slot 8, chain
    { { 425.0, 425.0, 412.5, 412.5, 412.5 }, { 437.5, 437.5, 437.5, 437.5, 425.0 } }, //slot 9, chain
    { { 450.0, 450.0, 450.0, 437.5, 437.5 }, { 475.0, 475.0, 475.0, 462.5, 462.5 } }, //slot 10, chain
    { { 487.5, 487.5, 487.5, 487.5, 487.5 }, { 512.5, 512.5, 512.5, 512.5, 500.0 } }, //slot 11, chain
    { { 525.0, 525.0, 525.0, 512.5, 512.5 }, { 550.0, 550.0, 537.5, 537.5, 537.5 } }  //slot 12, chain
  },
  {
    //crate 28
    { { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 } },                                         //slot 3, chain
    { { 175.0, 175.0, 175.0, 187.5, 187.5 }, { 200.0, 200.0, 212.5, 212.5, 212.5 } }, //slot 4, chain
    { { 212.5, 237.5, 225.0, 237.5, 237.5 }, { 250.0, 262.5, 262.5, 262.5, 262.5 } }, //slot 5, chain
    { { 262.5, 262.5, 275.0, 275.0, 275.0 }, { 287.5, 287.5, 300.0, 300.0, 300.0 } }, //slot 6, chain
    { { 312.5, 300.0, 312.5, 312.5, 325.0 }, { 337.5, 325.0, 337.5, 337.5, 350.0 } }, //slot 7, chain
    { { 350.0, 362.5, 350.0, 362.5, 362.5 }, { 387.5, 387.5, 400.0, 400.0, 400.0 } }, //slot 8, chain
    { { 387.5, 387.5, 387.5, 400.0, 400.0 }, { 412.5, 412.5, 412.5, 425.0, 425.0 } }, //slot 9, chain
    { { 425.0, 425.0, 425.0, 437.5, 437.5 }, { 462.5, 450.0, 450.0, 462.5, 450.0 } }, //slot 10,chain
    { { 475.0, 475.0, 487.5, 487.5, 487.5 }, { 487.5, 487.5, 500.0, 500.0, 500.0 } }, //slot 11,chain
    { { 500.0, 512.5, 512.5, 512.5, 512.5 }, { 525.0, 537.5, 537.5, 537.5, 550.0 } }  //slot 12,chain
  },
  {
    //crate 29
    { { 162.5, 0, 0, 0, 0 }, { 175.0, 0, 0, 0, 0 } },                                 //slot 3, chain
    { { 187.5, 200.0, 200.0, 187.5, 187.5 }, { 212.5, 225.0, 225.0, 212.5, 212.5 } }, //slot 4, chain
    { { 250.0, 250.0, 237.5, 237.5, 237.5 }, { 275.0, 275.0, 275.0, 262.5, 262.5 } }, //slot 5, chain
    { { 287.5, 287.5, 287.5, 275.0, 275.0 }, { 312.5, 312.5, 312.5, 300.0, 300.0 } }, //slot 6, chain
    { { 337.5, 337.5, 325.0, 325.0, 325.0 }, { 362.5, 362.5, 350.0, 350.0, 350.0 } }, //slot 7, chain
    { { 375.0, 375.0, 375.0, 375.0, 375.0 }, { 412.5, 412.5, 412.5, 412.5, 400.0 } }, //slot 8, chain
    { { 425.0, 412.5, 412.5, 412.5, 412.5 }, { 437.5, 437.5, 437.5, 437.5, 425.0 } }, //slot 9, chain
    { { 450.0, 450.0, 450.0, 437.5, 437.5 }, { 475.0, 475.0, 475.0, 462.5, 462.5 } }, //slot 10, chain
    { { 487.5, 487.5, 487.5, 487.5, 487.5 }, { 512.5, 512.5, 512.5, 512.5, 500.0 } }, //slot 11, chain
    { { 525.0, 525.0, 512.5, 525.0, 525.0 }, { 550.0, 550.0, 537.5, 537.5, 537.5 } }  //slot 12, chain
  },
  {
    //crate 30
    { { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 } },                                         //slot 3, chain
    { { 175.0, 175.0, 175.0, 187.5, 187.5 }, { 200.0, 200.0, 212.5, 212.5, 212.5 } }, //slot 4, chain
    { { 212.5, 237.5, 225.0, 237.5, 237.5 }, { 250.0, 262.5, 262.5, 262.5, 262.5 } }, //slot 5, chain
    { { 262.5, 262.5, 275.0, 275.0, 275.0 }, { 287.5, 287.5, 300.0, 300.0, 300.0 } }, //slot 6, chain
    { { 312.5, 300.0, 312.5, 312.5, 325.0 }, { 337.5, 325.0, 337.5, 337.5, 350.0 } }, //slot 7, chain
    { { 350.0, 362.5, 350.0, 362.5, 362.5 }, { 387.5, 387.5, 400.0, 400.0, 400.0 } }, //slot 8, chain
    { { 387.5, 387.5, 387.5, 400.0, 400.0 }, { 412.5, 412.5, 412.5, 425.0, 425.0 } }, //slot 9, chain
    { { 425.0, 425.0, 425.0, 437.5, 437.5 }, { 462.5, 450.0, 450.0, 462.5, 450.0 } }, //slot 10, chain
    { { 475.0, 475.0, 487.5, 487.5, 487.5 }, { 487.5, 487.5, 500.0, 500.0, 500.0 } }, //slot 11, chain
    { { 500.0, 512.5, 512.5, 512.5, 512.5 }, { 525.0, 537.5, 537.5, 537.5, 550.0 } }  //slot 12, chain
  },
  {
    //crate 31
    { { 162.5, 0, 0, 0, 0 }, { 175.0, 0, 0, 0, 0 } },                                 //slot 3, chain
    { { 187.5, 200.0, 200.0, 187.5, 187.5 }, { 212.5, 225.0, 225.0, 212.5, 212.5 } }, //slot 4, chain
    { { 250.0, 250.0, 237.5, 237.5, 237.5 }, { 275.0, 275.0, 275.0, 262.5, 262.5 } }, //slot 5, chain
    { { 287.5, 287.5, 287.5, 275.0, 275.0 }, { 312.5, 312.5, 312.5, 300.0, 300.0 } }, //slot 6, chain
    { { 337.5, 337.5, 325.0, 325.0, 325.0 }, { 362.5, 362.5, 350.0, 350.0, 350.0 } }, //slot 7, chain
    { { 375.0, 375.0, 375.0, 375.0, 375.0 }, { 412.5, 412.5, 412.5, 412.5, 400.0 } }, //slot 8, chain
    { { 425.0, 412.5, 412.5, 412.5, 412.5 }, { 437.5, 437.5, 437.5, 437.5, 425.0 } }, //slot 9, chain
    { { 450.0, 450.0, 450.0, 437.5, 437.5 }, { 475.0, 475.0, 475.0, 462.5, 462.5 } }, //slot 10, chain
    { { 487.5, 487.5, 487.5, 487.5, 487.5 }, { 512.5, 512.5, 512.5, 512.5, 500.0 } }, //slot 11, chain
    { { 525.0, 525.0, 512.5, 525.0, 525.0 }, { 550.0, 550.0, 537.5, 537.5, 537.5 } }  //slot 12, chain
  },
  {
    //crate 32
    { { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 } },                                         //slot 3, chain
    { { 175.0, 175.0, 175.0, 187.5, 187.5 }, { 200.0, 200.0, 212.5, 212.5, 212.5 } }, //slot 4, chain
    { { 212.5, 237.5, 225.0, 237.5, 237.5 }, { 250.0, 262.5, 262.5, 262.5, 262.5 } }, //slot 5, chain
    { { 262.5, 262.5, 275.0, 275.0, 275.0 }, { 287.5, 287.5, 300.0, 300.0, 300.0 } }, //slot 6, chain
    { { 312.5, 300.0, 312.5, 312.5, 325.0 }, { 337.5, 325.0, 337.5, 337.5, 350.0 } }, //slot 7, chain
    { { 350.0, 362.5, 350.0, 362.5, 362.5 }, { 387.5, 387.5, 387.5, 387.5, 400.0 } }, //slot 8, chain
    { { 387.5, 387.5, 387.5, 387.5, 400.0 }, { 412.5, 412.5, 425.0, 412.5, 425.0 } }, //slot 9, chain
    { { 425.0, 425.0, 425.0, 425.0, 425.0 }, { 462.5, 450.0, 450.0, 462.5, 450.0 } }, //slot 10, chain
    { { 462.5, 462.5, 462.5, 475.0, 475.0 }, { 487.5, 487.5, 487.5, 500.0, 500.0 } }, //slot 11, chain
    { { 500.0, 512.5, 512.5, 512.5, 512.5 }, { 525.0, 537.5, 537.5, 537.5, 550.0 } }  //slot 12, chain
  },
  {
    //crate 33
    { { 162.5, 0, 0, 0, 0 }, { 175.0, 0, 0, 0, 0 } },                                 //slot 3, chain
    { { 187.5, 200.0, 200.0, 187.5, 187.5 }, { 212.5, 225.0, 225.0, 212.5, 212.5 } }, //slot 4, chain
    { { 250.0, 250.0, 237.5, 237.5, 237.5 }, { 275.0, 275.0, 275.0, 262.5, 262.5 } }, //slot 5, chain
    { { 287.5, 287.5, 287.5, 275.0, 275.0 }, { 312.5, 312.5, 312.5, 300.0, 300.0 } }, //slot 6, chain
    { { 337.5, 337.5, 325.0, 325.0, 325.0 }, { 362.5, 362.5, 350.0, 350.0, 350.0 } }, //slot 7, chain
    { { 375.0, 375.0, 375.0, 375.0, 375.0 }, { 400.0, 400.0, 400.0, 400.0, 400.0 } }, //slot 8, chain
    { { 412.5, 412.5, 412.5, 412.5, 412.5 }, { 437.5, 437.5, 437.5, 437.5, 437.5 } }, //slot 9, chain
    { { 437.5, 437.5, 437.5, 437.5, 437.5 }, { 462.5, 462.5, 462.5, 450.0, 462.5 } }, //slot 10, chain
    { { 487.5, 487.5, 487.5, 475.0, 475.0 }, { 512.5, 512.5, 512.5, 500.0, 500.0 } }, //slot 11, chain
    { { 525.5, 512.5, 512.5, 512.5, 512.5 }, { 550.0, 550.0, 537.5, 537.5, 537.5 } }  //slot 12, chain
  },
  {
    //crate 34
    { { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 } },                                         //slot 3, chain
    { { 175.0, 175.0, 175.0, 187.5, 187.5 }, { 200.0, 200.0, 212.5, 212.5, 212.5 } }, //slot 4, chain
    { { 212.5, 237.5, 225.0, 237.5, 237.5 }, { 250.0, 262.5, 262.5, 262.5, 262.5 } }, //slot 5, chain
    { { 262.5, 262.5, 275.0, 275.0, 275.0 }, { 287.5, 287.5, 300.0, 300.0, 300.0 } }, //slot 6, chain
    { { 312.5, 300.0, 312.5, 312.5, 325.0 }, { 337.5, 325.0, 337.5, 337.5, 350.0 } }, //slot 7, chain
    { { 350.0, 362.5, 350.0, 362.5, 362.5 }, { 387.5, 387.5, 387.5, 387.5, 400.0 } }, //slot 8, chain
    { { 387.5, 387.5, 387.5, 387.5, 400.0 }, { 412.5, 412.5, 425.0, 412.5, 425.0 } }, //slot 9, chain
    { { 425.0, 425.0, 425.0, 425.0, 425.0 }, { 462.5, 450.0, 450.0, 462.5, 450.0 } }, //slot 10, chain
    { { 462.5, 462.5, 462.5, 475.0, 475.0 }, { 487.5, 487.5, 487.5, 500.0, 500.0 } }, //slot 11, chain
    { { 500.0, 512.5, 512.5, 512.5, 512.5 }, { 525.0, 537.5, 537.5, 537.5, 550.0 } }  //slot 12, chain
  },
  {
    //crate 35
    { { 162.5, 0, 0, 0, 0 }, { 175.0, 0, 0, 0, 0 } },                                 //slot 3, chain
    { { 187.5, 200.0, 200.0, 187.5, 187.5 }, { 212.5, 225.0, 225.0, 212.5, 212.5 } }, //slot 4, chain
    { { 250.0, 250.0, 237.5, 237.5, 237.5 }, { 275.0, 275.0, 275.0, 262.5, 262.5 } }, //slot 5, chain
    { { 287.5, 287.5, 287.5, 275.0, 275.0 }, { 312.5, 312.5, 312.5, 300.0, 300.0 } }, //slot 6, chain
    { { 337.5, 337.5, 325.0, 325.0, 325.0 }, { 362.5, 362.5, 350.0, 350.0, 350.0 } }, //slot 7, chain
    { { 375.0, 375.0, 375.0, 375.0, 375.0 }, { 400.0, 400.0, 400.0, 400.0, 400.0 } }, //slot 8, chain
    { { 412.5, 412.5, 412.5, 412.5, 412.5 }, { 437.5, 437.5, 437.5, 437.5, 437.5 } }, //slot 9, chain
    { { 437.5, 437.5, 437.5, 437.5, 437.5 }, { 462.5, 462.5, 462.5, 450.0, 462.5 } }, //slot 10, chain
    { { 487.5, 487.5, 487.5, 475.0, 475.0 }, { 512.5, 512.5, 512.5, 500.0, 500.0 } }, //slot 11, chain
    { { 525.5, 512.5, 512.5, 512.5, 512.5 }, { 550.0, 550.0, 537.5, 537.5, 537.5 } }  //slot 12, chain
  },
  {
    //crate 36
    { { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 } },                                         //slot 3, chain
    { { 175.0, 175.0, 175.0, 187.5, 187.5 }, { 200.0, 200.0, 212.5, 212.5, 212.5 } }, //slot 4, chain
    { { 212.5, 237.5, 225.0, 237.5, 237.5 }, { 250.0, 262.5, 262.5, 262.5, 262.5 } }, //slot 5, chain
    { { 262.5, 262.5, 275.0, 275.0, 275.0 }, { 287.5, 287.5, 300.0, 300.0, 300.0 } }, //slot 6, chain
    { { 312.5, 300.0, 312.5, 312.5, 325.0 }, { 337.5, 325.0, 337.5, 337.5, 350.0 } }, //slot 7, chain
    { { 350.0, 362.5, 350.0, 362.5, 362.5 }, { 387.5, 400.0, 400.0, 400.0, 412.5 } }, //slot 8, chain
    { { 387.5, 387.5, 400.0, 400.0, 412.5 }, { 412.5, 412.5, 425.0, 425.0, 425.0 } }, //slot 9, chain
    { { 425.0, 425.0, 425.0, 437.5, 437.5 }, { 475.0, 450.0, 450.0, 462.5, 462.5 } }, //slot 10, chain
    { { 475.0, 475.0, 475.0, 487.5, 487.5 }, { 500.0, 500.0, 500.0, 500.0, 500.0 } }, //slot 11, chain
    { { 512.5, 512.5, 512.5, 512.5, 512.5 }, { 525.0, 537.5, 537.5, 537.5, 550.0 } }  //slot 12, chain
  },
  {
    //crate 37
    { { 162.5, 0, 0, 0, 0 }, { 175.0, 0, 0, 0, 0 } },                                 //slot 3, chain
    { { 187.5, 200.0, 200.0, 187.5, 187.5 }, { 212.5, 225.0, 225.0, 212.5, 212.5 } }, //slot 4, chain
    { { 250.0, 250.0, 237.5, 237.5, 237.5 }, { 275.0, 275.0, 275.0, 262.5, 262.5 } }, //slot 5, chain
    { { 287.5, 287.5, 287.5, 275.0, 275.0 }, { 312.5, 312.5, 312.5, 300.0, 300.0 } }, //slot 6, chain
    { { 337.5, 337.5, 325.0, 325.0, 325.0 }, { 362.5, 362.5, 350.0, 350.0, 350.0 } }, //slot 7, chain
    { { 387.5, 375.0, 375.0, 375.0, 375.0 }, { 412.5, 412.5, 412.5, 400.0, 400.0 } }, //slot 8, chain
    { { 425.0, 412.5, 412.5, 412.5, 412.5 }, { 437.5, 437.5, 437.5, 437.5, 437.5 } }, //slot 9, chain
    { { 450.0, 450.0, 450.0, 450.0, 437.5 }, { 475.0, 475.0, 475.0, 462.5, 462.5 } }, //slot 10, chain
    { { 487.5, 487.5, 487.5, 487.5, 487.5 }, { 512.5, 512.5, 512.5, 512.5, 512.5 } }, //slot 11, chain
    { { 525.0, 525.0, 525.0, 525.0, 512.5 }, { 550.0, 550.0, 537.5, 537.5, 537.5 } }  //slot 12, chain
  },
  {
    //crate 38
    { { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 } },                                         //slot 3, chain
    { { 175.0, 175.0, 175.0, 187.5, 187.5 }, { 200.0, 200.0, 212.5, 212.5, 212.5 } }, //slot 4, chain
    { { 212.5, 237.5, 225.0, 237.5, 237.5 }, { 250.0, 262.5, 262.5, 262.5, 262.5 } }, //slot 5, chain
    { { 262.5, 262.5, 275.0, 275.0, 275.0 }, { 287.5, 287.5, 300.0, 300.0, 300.0 } }, //slot 6, chain
    { { 312.5, 300.0, 312.5, 312.5, 325.0 }, { 337.5, 325.0, 337.5, 337.5, 350.0 } }, //slot 7, chain
    { { 350.0, 362.5, 350.0, 362.5, 362.5 }, { 387.5, 400.0, 400.0, 400.0, 412.5 } }, //slot 8, chain
    { { 387.5, 387.5, 400.0, 400.0, 412.5 }, { 412.5, 412.5, 425.0, 425.0, 425.0 } }, //slot 9, chain
    { { 425.0, 425.0, 425.0, 437.5, 437.5 }, { 475.0, 450.0, 450.0, 462.5, 462.5 } }, //slot 10, chain
    { { 475.0, 475.0, 475.0, 487.5, 487.5 }, { 500.0, 500.0, 500.0, 500.0, 500.0 } }, //slot 11, chain
    { { 512.5, 512.5, 512.5, 512.5, 512.5 }, { 525.0, 537.5, 537.5, 537.5, 550.0 } }  //slot 12, chain
  },
  {
    //crate 39
    { { 162.5, 0, 0, 0, 0 }, { 175.0, 0, 0, 0, 0 } },                                 //slot 3, chain
    { { 187.5, 200.0, 200.0, 187.5, 187.5 }, { 212.5, 225.0, 225.0, 212.5, 212.5 } }, //slot 4, chain
    { { 250.0, 250.0, 237.5, 237.5, 237.5 }, { 275.0, 275.0, 275.0, 262.5, 262.5 } }, //slot 5, chain
    { { 287.5, 287.5, 287.5, 275.0, 275.0 }, { 312.5, 312.5, 312.5, 300.0, 300.0 } }, //slot 6, chain
    { { 337.5, 337.5, 325.0, 325.0, 325.0 }, { 362.5, 362.5, 350.0, 350.0, 350.0 } }, //slot 7, chain
    { { 387.5, 375.0, 375.0, 375.0, 375.0 }, { 412.5, 412.5, 412.5, 400.0, 400.0 } }, //slot 8, chain
    { { 425.0, 412.5, 412.5, 412.5, 412.5 }, { 437.5, 437.5, 437.5, 437.5, 437.5 } }, //slot 9, chain
    { { 450.0, 450.0, 450.0, 450.0, 437.5 }, { 475.0, 475.0, 475.0, 462.5, 462.5 } }, //slot 10, chain
    { { 487.5, 487.5, 487.5, 487.5, 487.5 }, { 512.5, 512.5, 512.5, 512.5, 512.5 } }, //slot 11, chain
    { { 525.0, 525.0, 525.0, 525.0, 512.5 }, { 550.0, 550.0, 537.5, 537.5, 537.5 } }  //slot 12, chain
  },
  {
    //crate 40
    { { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 } },                                         //slot 3, chain
    { { 175.0, 175.0, 175.0, 187.5, 187.5 }, { 200.0, 200.0, 212.5, 212.5, 212.5 } }, //slot 4, chain
    { { 212.5, 237.5, 225.0, 237.5, 237.5 }, { 250.0, 262.5, 262.5, 262.5, 262.5 } }, //slot 5, chain
    { { 262.5, 262.5, 275.0, 275.0, 275.0 }, { 287.5, 287.5, 300.0, 300.0, 300.0 } }, //slot 6, chain
    { { 312.5, 300.0, 312.5, 312.5, 325.0 }, { 337.5, 325.0, 337.5, 337.5, 350.0 } }, //slot 7, chain
    { { 350.0, 362.5, 350.0, 362.5, 362.5 }, { 387.5, 400.0, 400.0, 400.0, 412.5 } }, //slot 8, chain
    { { 387.5, 387.5, 400.0, 400.0, 412.5 }, { 412.5, 412.5, 425.0, 425.0, 425.0 } }, //slot 9, chain
    { { 425.0, 425.0, 425.0, 437.5, 437.5 }, { 475.0, 450.0, 450.0, 462.5, 462.5 } }, //slot 10, chain
    { { 475.0, 487.5, 487.5, 487.5, 487.5 }, { 500.0, 500.0, 500.0, 500.0, 500.0 } }, //slot 11, chain
    { { 500.0, 512.5, 512.5, 512.5, 512.5 }, { 525.0, 537.5, 537.5, 537.5, 550.0 } }  //slot 12, chain
  },
  {
    //crate 41
    { { 162.5, 0, 0, 0, 0 }, { 175.0, 0, 0, 0, 0 } },                                 //slot 3, chain
    { { 187.5, 200.0, 200.0, 187.5, 187.5 }, { 212.5, 225.0, 225.0, 212.5, 212.5 } }, //slot 4, chain
    { { 250.0, 250.0, 237.5, 237.5, 237.5 }, { 275.0, 275.0, 275.0, 262.5, 262.5 } }, //slot 5, chain
    { { 287.5, 287.5, 287.5, 275.0, 275.0 }, { 312.5, 312.5, 312.5, 300.0, 300.0 } }, //slot 6, chain
    { { 337.5, 337.5, 325.0, 325.0, 325.0 }, { 362.5, 326.5, 350.0, 350.0, 350.0 } }, //slot 7, chain
    { { 375.0, 375.0, 375.0, 375.0, 387.5 }, { 412.5, 412.5, 412.5, 400.0, 400.0 } }, //slot 8, chain
    { { 425.0, 412.5, 412.5, 412.5, 412.5 }, { 437.5, 437.5, 437.5, 437.5, 437.5 } }, //slot 9, chain
    { { 450.0, 450.0, 450.0, 450.0, 437.5 }, { 475.0, 475.0, 475.0, 462.5, 462.5 } }, //slot 10, chain
    { { 487.5, 487.5, 487.5, 487.5, 487.5 }, { 512.5, 512.5, 512.5, 512.5, 500.0 } }, //slot 11, chain
    { { 525.0, 525.0, 525.0, 525.0, 512.5 }, { 550.0, 550.0, 537.5, 537.5, 537.5 } }  //slot 12, chain
  },
  {
    //crate 42
    { { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 } },                                         //slot 3, chain
    { { 175.0, 175.0, 175.0, 187.5, 187.5 }, { 200.0, 200.0, 212.5, 212.5, 212.5 } }, //slot 4, chain
    { { 212.5, 237.5, 225.0, 237.5, 237.5 }, { 250.0, 262.5, 262.5, 262.5, 262.5 } }, //slot 5, chain
    { { 262.5, 262.5, 275.0, 275.0, 275.0 }, { 287.5, 287.5, 300.0, 300.0, 300.0 } }, //slot 6, chain
    { { 312.5, 300.0, 312.5, 312.5, 325.0 }, { 337.5, 325.0, 337.5, 337.5, 350.0 } }, //slot 7, chain
    { { 350.0, 362.5, 350.0, 362.5, 362.5 }, { 387.5, 400.0, 400.0, 400.0, 412.5 } }, //slot 8, chain
    { { 387.5, 387.5, 400.0, 400.0, 412.5 }, { 412.5, 412.5, 425.0, 425.0, 425.0 } }, //slot 9, chain
    { { 425.0, 425.0, 425.0, 437.5, 437.5 }, { 475.0, 450.0, 450.0, 462.5, 462.5 } }, //slot 10, chain
    { { 475.0, 487.5, 487.5, 487.5, 487.5 }, { 500.0, 500.0, 500.0, 500.0, 500.0 } }, //slot 11, chain
    { { 500.0, 512.5, 512.5, 512.5, 512.5 }, { 525.0, 537.5, 537.5, 537.5, 550.0 } }  //slot 12, chain
  },
  {
    //crate 43
    { { 162.5, 0, 0, 0, 0 }, { 175.0, 0, 0, 0, 0 } },                                 //slot 3, chain
    { { 187.5, 200.0, 200.0, 187.5, 187.5 }, { 212.5, 225.0, 225.0, 212.5, 212.5 } }, //slot 4, chain
    { { 250.0, 250.0, 237.5, 237.5, 237.5 }, { 275.0, 275.0, 275.0, 262.5, 262.5 } }, //slot 5, chain
    { { 287.5, 287.5, 287.5, 275.0, 275.0 }, { 312.5, 312.5, 312.5, 300.0, 300.0 } }, //slot 6, chain
    { { 337.5, 337.5, 325.0, 325.0, 325.0 }, { 362.5, 326.5, 350.0, 350.0, 350.0 } }, //slot 7, chain
    { { 375.0, 375.0, 375.0, 375.0, 387.5 }, { 412.5, 412.5, 412.5, 400.0, 400.0 } }, //slot 8, chain
    { { 425.0, 412.5, 412.5, 412.5, 412.5 }, { 437.5, 437.5, 437.5, 437.5, 437.5 } }, //slot 9, chain
    { { 450.0, 450.0, 450.0, 450.0, 437.5 }, { 475.0, 475.0, 475.0, 462.5, 462.5 } }, //slot 10, chain
    { { 487.5, 487.5, 487.5, 487.5, 487.5 }, { 512.5, 512.5, 512.5, 512.5, 500.0 } }, //slot 11, chain
    { { 525.0, 525.0, 525.0, 525.0, 512.5 }, { 550.0, 550.0, 537.5, 537.5, 537.5 } }  //slot 12, chain
  },
  {
    //crate 44
    { { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 } },                                         //slot 3, chain
    { { 175.0, 175.0, 175.0, 187.5, 187.5 }, { 200.0, 200.0, 212.5, 212.5, 212.5 } }, //slot 4, chain
    { { 212.5, 237.5, 225.0, 237.5, 237.5 }, { 250.0, 262.5, 262.5, 262.5, 262.5 } }, //slot 5, chain
    { { 262.5, 262.5, 275.0, 275.0, 275.0 }, { 287.5, 287.5, 300.0, 300.0, 300.0 } }, //slot 6, chain
    { { 312.5, 300.0, 312.5, 312.5, 325.0 }, { 337.5, 325.0, 337.5, 337.5, 350.0 } }, //slot 7, chain
    { { 350.0, 362.5, 350.0, 362.5, 362.5 }, { 387.5, 387.5, 400.0, 400.0, 400.0 } }, //slot 8, chain
    { { 387.5, 387.5, 387.5, 400.0, 400.0 }, { 412.5, 412.5, 412.5, 425.0, 425.0 } }, //slot 9, chain
    { { 425.0, 425.0, 425.0, 437.5, 437.5 }, { 462.5, 450.0, 450.0, 462.5, 450.0 } }, //slot 10, chain
    { { 475.0, 487.5, 487.5, 487.5, 487.5 }, { 487.5, 487.5, 500.0, 500.0, 500.0 } }, //slot 11, chain
    { { 500.0, 512.5, 512.5, 512.5, 512.5 }, { 525.0, 537.5, 537.5, 537.5, 550.0 } }  //slot 12, chain
  },
  {
    //crate 45
    { { 162.5, 0, 0, 0, 0 }, { 175.0, 0, 0, 0, 0 } },                                 //slot 3, chain
    { { 187.5, 200.0, 200.0, 187.5, 187.5 }, { 212.5, 225.0, 225.0, 212.5, 212.5 } }, //slot 4, chain
    { { 250.0, 250.0, 237.5, 237.5, 237.5 }, { 275.0, 275.0, 275.0, 262.5, 262.5 } }, //slot 5, chain
    { { 287.5, 287.5, 287.5, 275.0, 275.0 }, { 312.5, 312.5, 312.5, 300.0, 300.0 } }, //slot 6, chain
    { { 337.5, 337.5, 325.0, 325.0, 325.0 }, { 362.5, 362.5, 350.0, 350.0, 350.0 } }, //slot 7, chain
    { { 375.0, 375.0, 375.0, 375.0, 375.0 }, { 412.5, 412.5, 412.5, 412.5, 400.0 } }, //slot 8, chain
    { { 425.0, 412.5, 412.5, 412.5, 412.5 }, { 437.5, 437.5, 437.5, 437.5, 425.0 } }, //slot 9, chain
    { { 450.0, 450.0, 450.0, 437.5, 437.5 }, { 475.0, 475.0, 475.0, 462.5, 462.5 } }, //slot 10, chain
    { { 487.5, 487.5, 487.5, 487.5, 487.5 }, { 512.5, 512.5, 512.5, 512.5, 500.0 } }, //slot 11, chain
    { { 525.0, 525.0, 525.0, 525.0, 512.5 }, { 550.0, 550.0, 537.5, 537.5, 537.5 } }  //slot 12, chain
  },
  {
    //crate 46
    { { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 } },                                         //slot 3, chain
    { { 175.0, 175.0, 175.0, 187.5, 187.5 }, { 200.0, 200.0, 212.5, 212.5, 212.5 } }, //slot 4, chain
    { { 212.5, 237.5, 225.0, 237.5, 237.5 }, { 250.0, 262.5, 262.5, 262.5, 262.5 } }, //slot 5, chain
    { { 262.5, 262.5, 275.0, 275.0, 275.0 }, { 287.5, 287.5, 300.0, 300.0, 300.0 } }, //slot 6, chain
    { { 312.5, 300.0, 312.5, 312.5, 325.0 }, { 337.5, 325.0, 337.5, 337.5, 350.0 } }, //slot 7, chain
    { { 350.0, 362.5, 350.0, 362.5, 362.5 }, { 387.5, 387.5, 400.0, 400.0, 400.0 } }, //slot 8, chain
    { { 387.5, 387.5, 387.5, 400.0, 400.0 }, { 412.5, 412.5, 412.5, 425.0, 425.0 } }, //slot 9, chain
    { { 425.0, 425.0, 425.0, 437.5, 437.5 }, { 462.5, 450.0, 450.0, 462.5, 450.0 } }, //slot 10, chain
    { { 475.0, 487.5, 487.5, 487.5, 487.5 }, { 487.5, 487.5, 500.0, 500.0, 500.0 } }, //slot 11, chain
    { { 500.0, 512.5, 512.5, 512.5, 512.5 }, { 525.0, 537.5, 537.5, 537.5, 550.0 } }  //slot 12, chain
  },
  {
    //crate 47
    { { 162.5, 0, 0, 0, 0 }, { 175.0, 0, 0, 0, 0 } },                                 //slot 3, chain
    { { 187.5, 200.0, 200.0, 187.5, 187.5 }, { 212.5, 225.0, 225.0, 212.5, 212.5 } }, //slot 4, chain
    { { 250.0, 250.0, 237.5, 237.5, 237.5 }, { 275.0, 275.0, 275.0, 262.5, 262.5 } }, //slot 5, chain
    { { 287.5, 287.5, 287.5, 275.0, 275.0 }, { 312.5, 312.5, 312.5, 300.0, 300.0 } }, //slot 6, chain
    { { 337.5, 337.5, 325.0, 325.0, 325.0 }, { 362.5, 362.5, 350.0, 350.0, 350.0 } }, //slot 7, chain
    { { 375.0, 375.0, 375.0, 375.0, 375.0 }, { 412.5, 412.5, 412.5, 412.5, 400.0 } }, //slot 8, chain
    { { 425.0, 412.5, 412.5, 412.5, 412.5 }, { 437.5, 437.5, 437.5, 437.5, 425.0 } }, //slot 9, chain
    { { 450.0, 450.0, 450.0, 437.5, 437.5 }, { 475.0, 475.0, 475.0, 462.5, 462.5 } }, //slot 10, chain
    { { 487.5, 487.5, 487.5, 487.5, 487.5 }, { 512.5, 512.5, 512.5, 512.5, 500.0 } }, //slot 11, chain
    { { 525.0, 525.0, 525.0, 525.0, 512.5 }, { 550.0, 550.0, 537.5, 537.5, 537.5 } }  //slot 12, chain
  },
  {
    //crate 48
    { { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 } },                                         //slot 3, chain
    { { 175.0, 175.0, 175.0, 187.5, 187.5 }, { 200.0, 200.0, 212.5, 212.5, 212.5 } }, //slot 4, chain
    { { 212.5, 237.5, 225.0, 237.5, 237.5 }, { 250.0, 262.5, 262.5, 262.5, 262.5 } }, //slot 5, chain
    { { 262.5, 262.5, 275.0, 275.0, 275.0 }, { 287.5, 287.5, 300.0, 300.0, 300.0 } }, //slot 6, chain
    { { 312.5, 300.0, 312.5, 312.5, 325.0 }, { 337.5, 325.0, 337.5, 337.5, 350.0 } }, //slot 7, chain
    { { 350.0, 362.5, 350.0, 362.5, 362.5 }, { 387.5, 387.5, 400.0, 400.0, 400.0 } }, //slot 8, chain
    { { 387.5, 387.5, 400.0, 400.0, 400.0 }, { 400.0, 400.0, 412.5, 412.5, 412.5 } }, //slot 9, chain
    { { 437.5, 437.5, 437.5, 437.5, 437.5 }, { 462.5, 462.5, 462.5, 462.5, 462.5 } }, //slot 10, chain
    { { 475.0, 475.0, 487.5, 487.5, 487.5 }, { 487.5, 487.5, 487.5, 500.0, 500.0 } }, //slot 11, chain
    { { 500.0, 500.0, 512.5, 512.5, 512.5 }, { 525.0, 537.5, 537.5, 537.5, 550.0 } }  //slot 12, chain
  },
  {
    //crate 49
    { { 162.5, 0, 0, 0, 0 }, { 175.0, 0, 0, 0, 0 } },                                 //slot 3, chain
    { { 187.5, 200.0, 200.0, 187.5, 187.5 }, { 212.5, 225.0, 225.0, 212.5, 212.5 } }, //slot 4, chain
    { { 250.0, 250.0, 237.5, 237.5, 237.5 }, { 275.0, 275.0, 275.0, 262.5, 262.5 } }, //slot 5, chain
    { { 287.5, 287.5, 287.5, 275.0, 275.0 }, { 312.5, 312.5, 312.5, 300.0, 300.0 } }, //slot 6, chain
    { { 337.5, 337.5, 325.0, 325.0, 325.0 }, { 362.5, 362.5, 350.0, 350.0, 350.0 } }, //slot 7, chain
    { { 375.0, 375.0, 375.0, 375.0, 375.0 }, { 412.5, 400.0, 400.0, 400.0, 400.0 } }, //slot 8, chain
    { { 437.5, 425.0, 412.5, 412.5, 412.5 }, { 437.5, 437.5, 437.5, 437.5, 425.0 } }, //slot 9, chain
    { { 450.0, 450.0, 437.5, 437.5, 437.5 }, { 475.0, 475.0, 475.0, 475.0, 462.5 } }, //slot 10, chain
    { { 487.5, 487.5, 487.5, 487.5, 487.5 }, { 512.5, 512.5, 512.5, 512.5, 500.0 } }, //slot 11, chain
    { { 525.0, 525.0, 525.0, 512.5, 512.5 }, { 550.0, 550.0, 537.5, 537.5, 537.5 } }  //slot 12, chain
  },
  {
    //crate 50
    { { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 } },                                         //slot 3, chain
    { { 175.0, 175.0, 175.0, 187.5, 187.5 }, { 200.0, 200.0, 212.5, 212.5, 212.5 } }, //slot 4, chain
    { { 212.5, 237.5, 225.0, 237.5, 237.5 }, { 250.0, 262.5, 262.5, 262.5, 262.5 } }, //slot 5, chain
    { { 262.5, 262.5, 275.0, 275.0, 275.0 }, { 287.5, 287.5, 300.0, 300.0, 300.0 } }, //slot 6, chain
    { { 312.5, 300.0, 312.5, 312.5, 325.0 }, { 337.5, 325.0, 337.5, 337.5, 350.0 } }, //slot 7, chain
    { { 350.0, 362.5, 350.0, 362.5, 362.5 }, { 387.5, 387.5, 400.0, 400.0, 412.5 } }, //slot 8, chain
    { { 387.5, 387.5, 400.0, 400.0, 412.5 }, { 412.5, 412.5, 425.0, 425.0, 425.0 } }, //slot 9, chain
    { { 425.0, 425.0, 437.5, 437.5, 437.5 }, { 462.5, 450.0, 450.0, 450.0, 450.0 } }, //slot 10, chain
    { { 475.0, 475.0, 475.0, 487.5, 487.5 }, { 487.5, 487.5, 487.5, 500.0, 500.0 } }, //slot 11, chain
    { { 500.0, 500.0, 512.5, 512.5, 512.5 }, { 525.0, 537.5, 537.5, 537.5, 550.0 } }  //slot 12, chain
  },
  {
    //crate 51
    { { 162.5, 0, 0, 0, 0 }, { 175.0, 0, 0, 0, 0 } },                                 //slot 3, chain
    { { 187.5, 200.0, 200.0, 187.5, 187.5 }, { 212.5, 225.0, 225.0, 212.5, 212.5 } }, //slot 4, chain
    { { 250.0, 250.0, 237.5, 237.5, 237.5 }, { 275.0, 275.0, 275.0, 262.5, 262.5 } }, //slot 5, chain
    { { 287.5, 287.5, 287.5, 275.0, 275.0 }, { 312.5, 312.5, 312.5, 300.0, 300.0 } }, //slot 6, chain
    { { 337.5, 337.5, 325.0, 325.0, 325.0 }, { 362.5, 362.5, 350.0, 350.0, 350.0 } }, //slot 7, chain
    { { 375.0, 375.0, 375.0, 375.0, 375.0 }, { 412.5, 412.5, 412.5, 412.5, 412.5 } }, //slot 8, chain
    { { 425.0, 412.5, 412.5, 412.5, 412.5 }, { 437.5, 437.5, 425.0, 425.0, 425.0 } }, //slot 9, chain
    { { 450.0, 450.0, 437.5, 437.5, 437.5 }, { 475.0, 475.0, 475.0, 462.5, 462.5 } }, //slot 10, chain
    { { 487.5, 487.5, 487.5, 487.5, 487.5 }, { 512.5, 512.5, 512.5, 512.5, 500.0 } }, //slot 11, chain
    { { 525.0, 525.0, 525.0, 512.5, 512.5 }, { 550.0, 550.0, 537.5, 537.5, 537.5 } }  //slot 12, chain
  },
  {
    //crate 52
    { { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 } },                                         //slot 3, chain
    { { 175.0, 175.0, 175.0, 187.5, 187.5 }, { 200.0, 200.0, 212.5, 212.5, 212.5 } }, //slot 4, chain
    { { 212.5, 237.5, 225.0, 237.5, 237.5 }, { 250.0, 262.5, 262.5, 262.5, 262.5 } }, //slot 5, chain
    { { 262.5, 262.5, 275.0, 275.0, 275.0 }, { 287.5, 287.5, 300.0, 300.0, 300.0 } }, //slot 6, chain
    { { 312.5, 300.0, 312.5, 312.5, 325.0 }, { 337.5, 325.0, 337.5, 337.5, 350.0 } }, //slot 7, chain
    { { 350.0, 362.5, 350.0, 362.5, 362.5 }, { 387.5, 387.5, 400.0, 400.0, 400.0 } }, //slot 8, chain
    { { 387.5, 387.5, 400.0, 400.0, 400.0 }, { 412.5, 412.5, 425.0, 425.0, 425.0 } }, //slot 9, chain
    { { 425.0, 425.0, 437.5, 437.5, 437.5 }, { 462.5, 450.0, 450.0, 462.5, 450.0 } }, //slot 10, chain
    { { 475.0, 475.0, 475.0, 0, 0 }, { 487.5, 487.5, 487.5, 0, 0 } },                 //slot 11, chain
    { { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 } }                                          //slot 12, chain
  },
  {
    //crate 53
    { { 162.5, 0, 0, 0, 0 }, { 175.0, 0, 0, 0, 0 } },                                 //slot 3, chain
    { { 187.5, 200.0, 200.0, 187.5, 187.5 }, { 212.5, 225.0, 225.0, 212.5, 212.5 } }, //slot 4, chain
    { { 250.0, 250.0, 237.5, 237.5, 237.5 }, { 275.0, 275.0, 275.0, 262.5, 262.5 } }, //slot 5, chain
    { { 287.5, 287.5, 287.5, 275.0, 275.0 }, { 312.5, 312.5, 312.5, 300.0, 300.0 } }, //slot 6, chain
    { { 337.5, 337.5, 325.0, 325.0, 325.0 }, { 362.5, 362.5, 350.0, 350.0, 350.0 } }, //slot 7, chain
    { { 375.0, 375.0, 375.0, 375.0, 375.0 }, { 412.5, 412.5, 412.5, 412.5, 400.0 } }, //slot 8, chain
    { { 425.0, 412.5, 412.5, 412.5, 412.5 }, { 437.5, 437.5, 437.5, 425.0, 425.0 } }, //slot 9, chain
    { { 450.0, 450.0, 450.0, 437.5, 437.5 }, { 475.0, 475.0, 475.0, 462.5, 462.5 } }, //slot 10, chain
    { { 0, 0, 0, 487.5, 487.5 }, { 0, 0, 0, 512.5, 500.0 } },                         //slot 11, chain
    { { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 } }                                          //slot 12, chain
  },
  {
    //crate 54
    { { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 } },                                         //slot 3, chain
    { { 175.0, 175.0, 175.0, 187.5, 187.5 }, { 200.0, 200.0, 212.5, 212.5, 212.5 } }, //slot 4, chain
    { { 212.5, 237.5, 225.0, 237.5, 237.5 }, { 250.0, 262.5, 262.5, 262.5, 262.5 } }, //slot 5, chain
    { { 262.5, 262.5, 275.0, 275.0, 275.0 }, { 287.5, 287.5, 300.0, 300.0, 300.0 } }, //slot 6, chain
    { { 312.5, 300.0, 312.5, 312.5, 325.0 }, { 337.5, 325.0, 337.5, 337.5, 350.0 } }, //slot 7, chain
    { { 350.0, 362.5, 350.0, 362.5, 362.5 }, { 387.5, 387.5, 400.0, 400.0, 400.0 } }, //slot 8, chain
    { { 387.5, 387.5, 400.0, 400.0, 400.0 }, { 412.5, 412.5, 425.0, 425.0, 425.0 } }, //slot 9, chain
    { { 425.0, 425.0, 437.5, 437.5, 437.5 }, { 462.5, 450.0, 450.0, 462.5, 450.0 } }, //slot 10, chain
    { { 475.0, 475.0, 475.0, 0, 0 }, { 487.5, 487.5, 487.5, 0, 0 } },                 //slot 11, chain
    { { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 } }                                          //slot 12, chain
  },
  {
    //crate 55
    { { 162.5, 0, 0, 0, 0 }, { 175.0, 0, 0, 0, 0 } },                                 //slot 3, chain
    { { 187.5, 200.0, 200.0, 187.5, 187.5 }, { 212.5, 225.0, 225.0, 212.5, 212.5 } }, //slot 4, chain
    { { 250.0, 250.0, 237.5, 237.5, 237.5 }, { 275.0, 275.0, 275.0, 262.5, 262.5 } }, //slot 5, chain
    { { 287.5, 287.5, 287.5, 275.0, 275.0 }, { 312.5, 312.5, 312.5, 300.0, 300.0 } }, //slot 6, chain
    { { 337.5, 337.5, 325.0, 325.0, 325.0 }, { 362.5, 362.5, 350.0, 350.0, 350.0 } }, //slot 7, chain
    { { 375.0, 375.0, 375.0, 375.0, 375.0 }, { 412.5, 412.5, 412.5, 412.5, 400.0 } }, //slot 8, chain
    { { 425.0, 412.5, 412.5, 412.5, 412.5 }, { 437.5, 437.5, 437.5, 425.0, 425.0 } }, //slot 9, chain
    { { 450.0, 450.0, 450.0, 437.5, 437.5 }, { 475.0, 475.0, 475.0, 462.5, 462.5 } }, //slot 10, chain
    { { 0, 0, 0, 487.5, 487.5 }, { 0, 0, 0, 512.5, 500.0 } },                         //slot 11, chain
    { { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 } }                                          //slot 12, chain
  },
  {
    //crate 56
    { { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 } },                                         //slot 3, chain
    { { 175.0, 175.0, 175.0, 187.5, 187.5 }, { 200.0, 200.0, 212.5, 212.5, 212.5 } }, //slot 4, chain
    { { 212.5, 237.5, 225.0, 237.5, 237.5 }, { 250.0, 262.5, 262.5, 262.5, 262.5 } }, //slot 5, chain
    { { 262.5, 262.5, 275.0, 275.0, 275.0 }, { 287.5, 287.5, 300.0, 300.0, 300.0 } }, //slot 6, chain
    { { 312.5, 300.0, 312.5, 312.5, 325.0 }, { 337.5, 325.0, 337.5, 337.5, 350.0 } }, //slot 7, chain
    { { 350.0, 362.5, 350.0, 362.5, 362.5 }, { 387.5, 387.5, 400.0, 400.0, 400.0 } }, //slot 8, chain
    { { 387.5, 387.5, 400.0, 400.0, 400.0 }, { 412.5, 412.5, 425.0, 425.0, 425.0 } }, //slot 9, chain
    { { 425.0, 425.0, 437.5, 437.5, 437.5 }, { 462.5, 450.0, 450.0, 462.5, 450.0 } }, //slot 10, chain
    { { 462.5, 462.5, 475.0, 0, 0 }, { 487.5, 487.5, 487.5, 0, 0 } },                 //slot 11, chain
    { { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 } }                                          //slot 12, chain
  },
  {
    //crate 57
    { { 162.5, 0, 0, 0, 0 }, { 175.0, 0, 0, 0, 0 } },                                 //slot 3, chain
    { { 187.5, 200.0, 200.0, 187.5, 187.5 }, { 212.5, 225.0, 225.0, 212.5, 212.5 } }, //slot 4, chain
    { { 250.0, 250.0, 237.5, 237.5, 237.5 }, { 275.0, 275.0, 275.0, 262.5, 262.5 } }, //slot 5, chain
    { { 287.5, 287.5, 287.5, 275.0, 275.0 }, { 312.5, 312.5, 312.5, 300.0, 300.0 } }, //slot 6, chain
    { { 337.5, 337.5, 325.0, 325.0, 325.0 }, { 362.5, 362.5, 350.0, 350.0, 350.0 } }, //slot 7, chain
    { { 375.0, 375.0, 375.0, 375.0, 375.0 }, { 412.5, 412.5, 412.5, 412.5, 400.0 } }, //slot 8, chain
    { { 425.0, 412.5, 412.5, 412.5, 412.5 }, { 437.5, 437.5, 437.5, 425.0, 425.0 } }, //slot 9, chain
    { { 450.0, 450.0, 450.0, 437.5, 437.5 }, { 475.0, 462.5, 462.5, 462.5, 462.5 } }, //slot 10, chain
    { { 0, 0, 0, 487.5, 487.5 }, { 0, 0, 0, 512.5, 500.0 } },                         //slot 11, chain
    { { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 } }                                          //slot 12, chain
  },
  {
    //crate 58
    { { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 } },                                         //slot 3, chain
    { { 175.0, 175.0, 175.0, 187.5, 187.5 }, { 200.0, 200.0, 212.5, 212.5, 212.5 } }, //slot 4, chain
    { { 212.5, 237.5, 225.0, 237.5, 237.5 }, { 250.0, 262.5, 262.5, 262.5, 262.5 } }, //slot 5, chain
    { { 262.5, 262.5, 275.0, 275.0, 275.0 }, { 287.5, 287.5, 300.0, 300.0, 300.0 } }, //slot 6, chain
    { { 312.5, 300.0, 312.5, 312.5, 325.0 }, { 337.5, 325.0, 337.5, 337.5, 350.0 } }, //slot 7, chain
    { { 350.0, 362.5, 350.0, 362.5, 362.5 }, { 387.5, 387.5, 400.0, 400.0, 400.0 } }, //slot 8, chain
    { { 387.5, 387.5, 400.0, 400.0, 400.0 }, { 412.5, 412.5, 425.0, 425.0, 425.0 } }, //slot 9, chain
    { { 425.0, 425.0, 437.5, 437.5, 437.5 }, { 462.5, 450.0, 450.0, 462.5, 450.0 } }, //slot 10, chain
    { { 462.5, 462.5, 475.0, 0, 0 }, { 487.5, 487.5, 487.5, 0, 0 } },                 //slot 11, chain
    { { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 } }                                          //slot 12, chain
  },
  {
    //crate 59
    { { 162.5, 0, 0, 0, 0 }, { 175.0, 0, 0, 0, 0 } },                                 //slot 3, chain
    { { 187.5, 200.0, 200.0, 187.5, 187.5 }, { 212.5, 225.0, 225.0, 212.5, 212.5 } }, //slot 4, chain
    { { 250.0, 250.0, 237.5, 237.5, 237.5 }, { 275.0, 275.0, 275.0, 262.5, 262.5 } }, //slot 5, chain
    { { 287.5, 287.5, 287.5, 275.0, 275.0 }, { 312.5, 312.5, 312.5, 300.0, 300.0 } }, //slot 6, chain
    { { 337.5, 337.5, 325.0, 325.0, 325.0 }, { 362.5, 362.5, 350.0, 350.0, 350.0 } }, //slot 7, chain
    { { 375.0, 375.0, 375.0, 375.0, 375.0 }, { 412.5, 412.5, 412.5, 412.5, 400.0 } }, //slot 8, chain
    { { 425.0, 412.5, 412.5, 412.5, 412.5 }, { 437.5, 437.5, 437.5, 425.0, 425.0 } }, //slot 9, chain
    { { 450.0, 450.0, 450.0, 437.5, 437.5 }, { 475.0, 462.5, 462.5, 462.5, 462.5 } }, //slot 10, chain
    { { 0, 0, 0, 487.5, 487.5 }, { 0, 0, 0, 512.5, 500.0 } },                         //slot 11, chain
    { { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 } }                                          //slot 12, chain
  },
  {
    //crate 60
    { { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 } },                                         //slot 3, chain
    { { 175.0, 175.0, 175.0, 187.5, 187.5 }, { 200.0, 200.0, 212.5, 212.5, 212.5 } }, //slot 4, chain
    { { 212.5, 237.5, 225.0, 237.5, 237.5 }, { 250.0, 262.5, 262.5, 262.5, 262.5 } }, //slot 5, chain
    { { 262.5, 262.5, 275.0, 275.0, 275.0 }, { 287.5, 287.5, 300.0, 300.0, 300.0 } }, //slot 6, chain
    { { 312.5, 300.0, 312.5, 312.5, 325.0 }, { 337.5, 325.0, 337.5, 337.5, 350.0 } }, //slot 7, chain
    { { 350.0, 362.5, 350.0, 362.5, 362.5 }, { 387.5, 387.5, 400.0, 400.0, 400.0 } }, //slot 8, chain
    { { 387.5, 387.5, 387.5, 400.0, 400.0 }, { 412.5, 412.5, 425.0, 425.0, 425.0 } }, //slot 9, chain
    { { 425.0, 425.0, 437.5, 437.5, 437.5 }, { 462.5, 450.0, 450.0, 462.5, 450.0 } }, //slot 10, chain
    { { 475.0, 475.0, 487.5, 0, 0 }, { 487.5, 487.5, 487.5, 0, 0 } },                 //slot 11, chain
    { { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 } }                                          //slot 12, chain
  },
  {
    //crate 61
    { { 162.5, 0, 0, 0, 0 }, { 175.0, 0, 0, 0, 0 } },                                 //slot 3, chain
    { { 187.5, 200.0, 200.0, 187.5, 187.5 }, { 212.5, 225.0, 225.0, 212.5, 212.5 } }, //slot 4, chain
    { { 250.0, 250.0, 237.5, 237.5, 237.5 }, { 275.0, 275.0, 275.0, 262.5, 262.5 } }, //slot 5, chain
    { { 287.5, 287.5, 287.5, 275.0, 275.0 }, { 312.5, 312.5, 312.5, 300.0, 300.0 } }, //slot 6, chain
    { { 337.5, 337.5, 325.0, 325.0, 325.0 }, { 362.5, 362.5, 350.0, 350.0, 350.0 } }, //slot 7, chain
    { { 375.0, 375.0, 375.0, 375.0, 375.0 }, { 412.5, 412.5, 412.5, 400.0, 400.0 } }, //slot 8, chain
    { { 425.0, 425.0, 412.5, 412.5, 412.5 }, { 437.5, 437.5, 437.5, 437.5, 425.0 } }, //slot 9, chain
    { { 450.0, 450.0, 450.0, 437.5, 437.5 }, { 475.0, 475.0, 475.0, 462.5, 462.5 } }, //slot 10, chain
    { { 0, 0, 0, 487.5, 487.5 }, { 0, 0, 0, 512.5, 500.0 } },                         //slot 11, chain
    { { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 } }                                          //slot 12, chain
  },
  {
    //crate 62
    { { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 } },                                         //slot 3, chain
    { { 175.0, 175.0, 175.0, 187.5, 187.5 }, { 200.0, 200.0, 212.5, 212.5, 212.5 } }, //slot 4, chain
    { { 212.5, 237.5, 225.0, 237.5, 237.5 }, { 250.0, 262.5, 262.5, 262.5, 262.5 } }, //slot 5, chain
    { { 262.5, 262.5, 275.0, 275.0, 275.0 }, { 287.5, 287.5, 300.0, 300.0, 300.0 } }, //slot 6, chain
    { { 312.5, 300.0, 312.5, 312.5, 325.0 }, { 337.5, 325.0, 337.5, 337.5, 350.0 } }, //slot 7, chain
    { { 350.0, 362.5, 350.0, 362.5, 362.5 }, { 387.5, 387.5, 400.0, 400.0, 400.0 } }, //slot 8, chain
    { { 387.5, 387.5, 387.5, 400.0, 400.0 }, { 412.5, 412.5, 425.0, 425.0, 425.0 } }, //slot 9, chain
    { { 425.0, 425.0, 437.5, 437.5, 437.5 }, { 462.5, 450.0, 450.0, 462.5, 450.0 } }, //slot 10, chain
    { { 475.0, 475.0, 487.5, 0, 0 }, { 487.5, 487.5, 487.5, 0, 0 } },                 //slot 11, chain
    { { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 } }                                          //slot 12, chain
  },
  {
    //crate 63
    { { 162.5, 0, 0, 0, 0 }, { 175.0, 0, 0, 0, 0 } },                                 //slot 3, chain
    { { 187.5, 200.0, 200.0, 187.5, 187.5 }, { 212.5, 225.0, 225.0, 212.5, 212.5 } }, //slot 4, chain
    { { 250.0, 250.0, 237.5, 237.5, 237.5 }, { 275.0, 275.0, 275.0, 262.5, 262.5 } }, //slot 5, chain
    { { 287.5, 287.5, 287.5, 275.0, 275.0 }, { 312.5, 312.5, 312.5, 300.0, 300.0 } }, //slot 6, chain
    { { 337.5, 337.5, 325.0, 325.0, 325.0 }, { 362.5, 362.5, 350.0, 350.0, 350.0 } }, //slot 7, chain
    { { 375.0, 375.0, 375.0, 375.0, 375.0 }, { 412.5, 412.5, 412.5, 400.0, 400.0 } }, //slot 8, chain
    { { 425.0, 425.0, 412.5, 412.5, 412.5 }, { 437.5, 437.5, 437.5, 437.5, 425.0 } }, //slot 9, chain
    { { 450.0, 450.0, 450.0, 437.5, 437.5 }, { 475.0, 475.0, 475.0, 462.5, 462.5 } }, //slot 10, chain
    { { 0, 0, 0, 487.5, 487.5 }, { 0, 0, 0, 512.5, 500.0 } },                         //slot 11, chain
    { { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 } }                                          //slot 12, chain
  },
  {
    //crate 64
    { { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 } },                                         //slot 3, chain
    { { 175.0, 175.0, 175.0, 187.5, 187.5 }, { 200.0, 200.0, 212.5, 212.5, 212.5 } }, //slot 4, chain
    { { 212.5, 237.5, 225.0, 237.5, 237.5 }, { 250.0, 262.5, 262.5, 262.5, 262.5 } }, //slot 5, chain
    { { 262.5, 262.5, 275.0, 275.0, 275.0 }, { 287.5, 287.5, 300.0, 300.0, 300.0 } }, //slot 6, chain
    { { 312.5, 300.0, 312.5, 312.5, 325.0 }, { 337.5, 325.0, 337.5, 337.5, 350.0 } }, //slot 7, chain
    { { 350.0, 362.5, 350.0, 362.5, 362.5 }, { 387.5, 400.0, 400.0, 400.0, 400.0 } }, //slot 8, chain
    { { 387.5, 387.5, 400.0, 400.0, 412.5 }, { 412.5, 412.5, 425.0, 425.0, 425.0 } }, //slot 9, chain
    { { 425.0, 425.0, 425.0, 437.5, 437.5 }, { 475.0, 450.0, 450.0, 462.5, 462.5 } }, //slot 10, chain
    { { 475.0, 475.0, 475.0, 487.5, 487.5 }, { 500.0, 500.0, 500.0, 500.0, 500.0 } }, //slot 11, chain
    { { 500.0, 512.5, 512.5, 512.5, 512.5 }, { 525.0, 537.5, 537.5, 537.5, 550.0 } }  //slot 12, chain
  },
  {
    //crate 65
    { { 162.5, 0, 0, 0, 0 }, { 175.0, 0, 0, 0, 0 } },                                 //slot 3, chain
    { { 187.5, 200.0, 200.0, 187.5, 187.5 }, { 212.5, 225.0, 225.0, 212.5, 212.5 } }, //slot 4, chain
    { { 250.0, 250.0, 237.5, 237.5, 237.5 }, { 275.0, 275.0, 275.0, 262.5, 262.5 } }, //slot 5, chain
    { { 287.5, 287.5, 287.5, 275.0, 275.0 }, { 312.5, 312.5, 312.5, 300.0, 300.0 } }, //slot 6, chain
    { { 337.5, 337.5, 325.0, 325.0, 325.0 }, { 362.5, 362.5, 350.0, 350.0, 350.0 } }, //slot 7, chain
    { { 375.0, 375.0, 375.0, 375.0, 387.5 }, { 412.5, 412.5, 412.5, 400.0, 400.0 } }, //slot 8, chain
    { { 425.0, 412.5, 412.5, 412.5, 412.5 }, { 437.5, 437.5, 437.5, 437.5, 437.5 } }, //slot 9, chain
    { { 450.0, 450.0, 450.0, 450.0, 437.5 }, { 475.0, 475.0, 475.0, 462.5, 462.5 } }, //slot 10, chain
    { { 487.5, 487.5, 487.5, 487.5, 487.5 }, { 512.5, 512.5, 512.5, 512.5, 500.0 } }, //slot 11, chain
    { { 525.0, 525.0, 525.0, 525.0, 512.5 }, { 550.0, 550.0, 537.5, 537.5, 537.5 } }  //slot 12, chain
  },
  {
    //crate 66
    { { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 } },                                         //slot 3, chain
    { { 175.0, 175.0, 175.0, 187.5, 187.5 }, { 200.0, 200.0, 212.5, 212.5, 212.5 } }, //slot 4, chain
    { { 212.5, 237.5, 225.0, 237.5, 237.5 }, { 250.0, 262.5, 262.5, 262.5, 262.5 } }, //slot 5, chain
    { { 262.5, 262.5, 275.0, 275.0, 275.0 }, { 287.5, 287.5, 300.0, 300.0, 300.0 } }, //slot 6, chain
    { { 312.5, 300.0, 312.5, 312.5, 325.0 }, { 337.5, 325.0, 337.5, 337.5, 350.0 } }, //slot 7, chain
    { { 350.0, 362.5, 350.0, 362.5, 362.5 }, { 387.5, 400.0, 400.0, 400.0, 400.0 } }, //slot 8, chain
    { { 387.5, 387.5, 400.0, 400.0, 412.5 }, { 412.5, 412.5, 425.0, 425.0, 425.0 } }, //slot 9, chain
    { { 425.0, 425.0, 425.0, 437.5, 437.5 }, { 475.0, 450.0, 450.0, 462.5, 462.5 } }, //slot 10, chain
    { { 475.0, 475.0, 475.0, 487.5, 487.5 }, { 500.0, 500.0, 500.0, 500.0, 500.0 } }, //slot 11, chain
    { { 500.0, 512.5, 512.5, 512.5, 512.5 }, { 525.0, 537.5, 537.5, 537.5, 550.0 } }  //slot 12, chain
  },
  {
    //crate 67
    { { 162.5, 0, 0, 0, 0 }, { 175.0, 0, 0, 0, 0 } },                                 //slot 3, chain
    { { 187.5, 200.0, 200.0, 187.5, 187.5 }, { 212.5, 225.0, 225.0, 212.5, 212.5 } }, //slot 4, chain
    { { 250.0, 250.0, 237.5, 237.5, 237.5 }, { 275.0, 275.0, 275.0, 262.5, 262.5 } }, //slot 5, chain
    { { 287.5, 287.5, 287.5, 275.0, 275.0 }, { 312.5, 312.5, 312.5, 300.0, 300.0 } }, //slot 6, chain
    { { 337.5, 337.5, 325.0, 325.0, 325.0 }, { 362.5, 362.5, 350.0, 350.0, 350.0 } }, //slot 7, chain
    { { 375.0, 375.0, 375.0, 375.0, 387.5 }, { 412.5, 412.5, 412.5, 400.0, 400.0 } }, //slot 8, chain
    { { 425.0, 412.5, 412.5, 412.5, 412.5 }, { 437.5, 437.5, 437.5, 437.5, 437.5 } }, //slot 9, chain
    { { 450.0, 450.0, 450.0, 450.0, 437.5 }, { 475.0, 475.0, 475.0, 462.5, 462.5 } }, //slot 10, chain
    { { 487.5, 487.5, 487.5, 487.5, 487.5 }, { 512.5, 512.5, 512.5, 512.5, 500.0 } }, //slot 11, chain
    { { 525.0, 525.0, 525.0, 525.0, 512.5 }, { 550.0, 550.0, 537.5, 537.5, 537.5 } }  //slot 12, chain
  },
  {
    //crate 68
    { { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 } },                                         //slot 3, chain
    { { 175.0, 175.0, 175.0, 187.5, 187.5 }, { 200.0, 200.0, 212.5, 212.5, 212.5 } }, //slot 4, chain
    { { 212.5, 237.5, 225.0, 237.5, 237.5 }, { 250.0, 262.5, 262.5, 262.5, 262.5 } }, //slot 5, chain
    { { 262.5, 262.5, 275.0, 275.0, 275.0 }, { 287.5, 287.5, 300.0, 300.0, 300.0 } }, //slot 6, chain
    { { 312.5, 300.0, 312.5, 312.5, 325.0 }, { 337.5, 325.0, 337.5, 337.5, 350.0 } }, //slot 7, chain
    { { 350.0, 362.5, 350.0, 362.5, 362.5 }, { 387.5, 387.5, 400.0, 400.0, 400.0 } }, //slot 8, chain
    { { 387.5, 387.5, 387.5, 400.0, 400.0 }, { 412.5, 412.5, 412.5, 425.0, 425.0 } }, //slot 9, chain
    { { 425.0, 425.0, 425.0, 437.5, 437.5 }, { 462.5, 450.0, 450.0, 462.5, 450.0 } }, //slot 10, chain
    { { 475.0, 475.0, 487.5, 487.5, 487.5 }, { 487.5, 487.5, 500.0, 500.0, 500.0 } }, //slot 11, chain
    { { 500.0, 512.5, 512.5, 512.5, 512.5 }, { 525.0, 537.5, 537.5, 537.5, 550.0 } }  //slot 12, chain
  },
  {
    //crate 69
    { { 162.5, 0, 0, 0, 0 }, { 175.0, 0, 0, 0, 0 } },                                 //slot 3, chain
    { { 187.5, 200.0, 200.0, 187.5, 187.5 }, { 212.5, 225.0, 225.0, 212.5, 212.5 } }, //slot 4, chain
    { { 250.0, 250.0, 237.5, 237.5, 237.5 }, { 275.0, 275.0, 275.0, 262.5, 262.5 } }, //slot 5, chain
    { { 287.5, 287.5, 287.5, 275.0, 275.0 }, { 312.5, 312.5, 312.5, 300.0, 300.0 } }, //slot 6, chain
    { { 337.5, 337.5, 325.0, 325.0, 325.0 }, { 362.5, 362.5, 350.0, 350.0, 350.0 } }, //slot 7, chain
    { { 375.0, 375.0, 375.0, 375.0, 375.0 }, { 412.5, 412.5, 412.5, 412.5, 400.0 } }, //slot 8, chain
    { { 425.0, 412.5, 412.5, 412.5, 412.5 }, { 437.5, 437.5, 437.5, 437.5, 425.0 } }, //slot 9, chain
    { { 450.0, 450.0, 450.0, 437.5, 437.5 }, { 475.0, 475.0, 475.0, 462.5, 462.5 } }, //slot 10, chain
    { { 487.5, 487.5, 487.5, 487.5, 487.5 }, { 512.5, 512.5, 512.5, 512.5, 500.0 } }, //slot 11, chain
    { { 525.0, 525.0, 525.0, 525.0, 512.5 }, { 550.0, 550.0, 537.5, 537.5, 537.5 } }  //slot 12, chain
  },
  {
    //crate 70
    { { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 } },                                         //slot 3, chain
    { { 175.0, 175.0, 175.0, 187.5, 187.5 }, { 200.0, 200.0, 212.5, 212.5, 212.5 } }, //slot 4, chain
    { { 212.5, 237.5, 225.0, 237.5, 237.5 }, { 250.0, 262.5, 262.5, 262.5, 262.5 } }, //slot 5, chain
    { { 262.5, 262.5, 275.0, 275.0, 275.0 }, { 287.5, 287.5, 300.0, 300.0, 300.0 } }, //slot 6, chain
    { { 312.5, 300.0, 312.5, 312.5, 325.0 }, { 337.5, 325.0, 337.5, 337.5, 350.0 } }, //slot 7, chain
    { { 350.0, 362.5, 350.0, 362.5, 362.5 }, { 387.5, 387.5, 400.0, 400.0, 400.0 } }, //slot 8, chain
    { { 387.5, 387.5, 387.5, 400.0, 400.0 }, { 412.5, 412.5, 412.5, 425.0, 425.0 } }, //slot 9, chain
    { { 425.0, 425.0, 425.0, 437.5, 437.5 }, { 462.5, 450.0, 450.0, 462.5, 450.0 } }, //slot 10, chain
    { { 475.0, 475.0, 487.5, 487.5, 487.5 }, { 487.5, 487.5, 500.0, 500.0, 500.0 } }, //slot 11, chain
    { { 500.0, 512.5, 512.5, 512.5, 512.5 }, { 525.0, 537.5, 537.5, 537.5, 550.0 } }  //slot 12, chain
  },
  {
    //crate 71
    { { 162.5, 0, 0, 0, 0 }, { 175.0, 0, 0, 0, 0 } },                                 //slot 3, chain
    { { 187.5, 200.0, 200.0, 187.5, 187.5 }, { 212.5, 225.0, 225.0, 212.5, 212.5 } }, //slot 4, chain
    { { 250.0, 250.0, 237.5, 237.5, 237.5 }, { 275.0, 275.0, 275.0, 262.5, 262.5 } }, //slot 5, chain
    { { 287.5, 287.5, 287.5, 275.0, 275.0 }, { 312.5, 312.5, 312.5, 300.0, 300.0 } }, //slot 6, chain
    { { 337.5, 337.5, 325.0, 325.0, 325.0 }, { 362.5, 362.5, 350.0, 350.0, 350.0 } }, //slot 7, chain
    { { 375.0, 375.0, 375.0, 375.0, 375.0 }, { 412.5, 412.5, 412.5, 412.5, 400.0 } }, //slot 8, chain
    { { 425.0, 412.5, 412.5, 412.5, 412.5 }, { 437.5, 437.5, 437.5, 437.5, 425.0 } }, //slot 9, chain
    { { 450.0, 450.0, 450.0, 437.5, 437.5 }, { 475.0, 475.0, 475.0, 462.5, 462.5 } }, //slot 10, chain
    { { 487.5, 487.5, 487.5, 487.5, 487.5 }, { 512.5, 512.5, 512.5, 512.5, 500.0 } }, //slot 11, chain
    { { 525.0, 525.0, 525.0, 525.0, 512.5 }, { 550.0, 550.0, 537.5, 537.5, 537.5 } }  //slot 12, chain
  }
};
