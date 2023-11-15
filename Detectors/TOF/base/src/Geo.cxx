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

#include "TOFBase/Geo.h"
#include "TGeoManager.h"
#include "TMath.h"
#include "Framework/Logger.h"
#include "DetectorsBase/GeometryManager.h"
#include "CommonUtils/StringUtils.h"
#include <string>
#include "MathUtils/Utils.h"

ClassImp(o2::tof::Geo);

using namespace o2::tof;

constexpr Float_t Geo::ANGLES[NPLATES][NMAXNSTRIP];
constexpr Float_t Geo::HEIGHTS[NPLATES][NMAXNSTRIP];
constexpr Float_t Geo::DISTANCES[NPLATES][NMAXNSTRIP];
constexpr Bool_t Geo::FEAWITHMASKS[NSECTORS];
constexpr Float_t Geo::ROOF2PARAMETERS[3];

Bool_t Geo::mToBeInit = kTRUE;
Bool_t Geo::mToBeInitIndexing = kTRUE;
Float_t Geo::mRotationMatrixSector[NSECTORS + 1][3][3];
Float_t Geo::mRotationMatrixPlateStrip[NSECTORS][NPLATES][NMAXNSTRIP][3][3];
Float_t Geo::mPadPosition[NSECTORS][NPLATES][NMAXNSTRIP][NPADZ][NPADX][3];
Float_t Geo::mGeoDistances[NSECTORS][NPLATES][NMAXNSTRIP];
Float_t Geo::mGeoHeights[NSECTORS][NPLATES][NMAXNSTRIP];
Float_t Geo::mGeoX[NSECTORS][NPLATES][NMAXNSTRIP];
Int_t Geo::mPlate[NSTRIPXSECTOR];
Int_t Geo::mStripInPlate[NSTRIPXSECTOR];
std::array<std::vector<float>, 5> Geo::mDistances[NSECTORS];

void Geo::Init()
{
  if (!mToBeInit) {
    return;
  }
  LOG(info) << "tof::Geo: Initialization of TOF rotation parameters";

  if (!gGeoManager) {
    LOG(fatal) << "geometry is not loaded";
  }

  Double_t rotationAngles[6] =
    {90., 90. /*+ (isector + 0.5) * PHISEC*/, 0., 0., 90., 0 /* + (isector + 0.5) * PHISEC*/};
  for (Int_t ii = 0; ii < 6; ii++) {
    rotationAngles[ii] *= TMath::DegToRad();
  }

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

  double toMyCoord[9];
  toMyCoord[0] = mRotationMatrixSector[NSECTORS][0][0];
  toMyCoord[1] = mRotationMatrixSector[NSECTORS][1][0];
  toMyCoord[2] = mRotationMatrixSector[NSECTORS][2][0];

  toMyCoord[3] = mRotationMatrixSector[NSECTORS][0][1];
  toMyCoord[4] = mRotationMatrixSector[NSECTORS][1][1];
  toMyCoord[5] = mRotationMatrixSector[NSECTORS][2][1];

  toMyCoord[6] = mRotationMatrixSector[NSECTORS][0][2];
  toMyCoord[7] = mRotationMatrixSector[NSECTORS][1][2];
  toMyCoord[8] = mRotationMatrixSector[NSECTORS][2][2];

  TGeoHMatrix myMatCoord;
  myMatCoord.SetRotation(toMyCoord);
  myMatCoord = myMatCoord.Inverse();

  /*
  printf("SECTOR BACK\n");
  printf("%f %f %f\n",mRotationMatrixSector[NSECTORS][0][0],mRotationMatrixSector[NSECTORS][0][1],mRotationMatrixSector[NSECTORS][0][2]);
  printf("%f %f %f\n",mRotationMatrixSector[NSECTORS][1][0],mRotationMatrixSector[NSECTORS][1][1],mRotationMatrixSector[NSECTORS][1][2]);
  printf("%f %f %f\n",mRotationMatrixSector[NSECTORS][2][0],mRotationMatrixSector[NSECTORS][2][1],mRotationMatrixSector[NSECTORS][2][2]);
  printf("\n");
  */

  int det[5];
  for (Int_t isector = 0; isector < NSECTORS; isector++) {
    det[0] = isector;
    auto string1 = fmt::format("/cave_1/barrel_1/B077_1/BSEGMO{:d}_1/BTOF{:d}_1", isector, isector);
    gGeoManager->cd(string1.c_str());
    TGeoHMatrix sectorMat = *gGeoManager->GetCurrentMatrix();
    // put translation to zero -> only rotation are considered for sectors, translation are applied directly for strips
    double trans0[3] = {0., 0., 0.};
    sectorMat.SetTranslation(trans0);
    const Double_t* rot = sectorMat.GetRotationMatrix();

    /*
    printf("SECTOR %d rotations\n",isector);
    printf("%f %f %f\n",mRotationMatrixSector[isector][0][0],mRotationMatrixSector[isector][0][1],mRotationMatrixSector[isector][0][2]);
    printf("%f %f %f\n",mRotationMatrixSector[isector][1][0],mRotationMatrixSector[isector][1][1],mRotationMatrixSector[isector][1][2]);
    printf("%f %f %f\n",mRotationMatrixSector[isector][2][0],mRotationMatrixSector[isector][2][1],mRotationMatrixSector[isector][2][2]);
    printf("vs...\n");
    */

    mRotationMatrixSector[isector][0][0] = rot[0];
    mRotationMatrixSector[isector][1][0] = rot[1];
    mRotationMatrixSector[isector][2][0] = rot[2];

    mRotationMatrixSector[isector][0][1] = rot[3];
    mRotationMatrixSector[isector][1][1] = rot[4];
    mRotationMatrixSector[isector][2][1] = rot[5];

    mRotationMatrixSector[isector][0][2] = rot[6];
    mRotationMatrixSector[isector][1][2] = rot[7];
    mRotationMatrixSector[isector][2][2] = rot[8];

    // let's invert the matric to move from global to sector coordinates
    sectorMat = sectorMat.Inverse();

    /*
    printf("%f %f %f\n",mRotationMatrixSector[isector][0][0],mRotationMatrixSector[isector][0][1],mRotationMatrixSector[isector][0][2]);
    printf("%f %f %f\n",mRotationMatrixSector[isector][1][0],mRotationMatrixSector[isector][1][1],mRotationMatrixSector[isector][1][2]);
    printf("%f %f %f\n",mRotationMatrixSector[isector][2][0],mRotationMatrixSector[isector][2][1],mRotationMatrixSector[isector][2][2]);
    printf("\n");
    */

    for (Int_t iplate = 0; iplate < NPLATES; iplate++) {
      det[1] = iplate;

      if (iplate == 2 && (isector == 13 || isector == 14 || isector == 15)) {
        continue; // PHOS HOLES
      }
      int istripOff = 0;
      if (iplate == 1) {
        istripOff = NSTRIPC;
      } else if (iplate == 2) {
        istripOff = NSTRIPC + NSTRIPB;
      }
      if (iplate == 3) {
        istripOff = NSTRIPC + NSTRIPB + NSTRIPA;
      }
      if (iplate == 4) {
        istripOff = NSTRIPC + 2 * NSTRIPB + NSTRIPA;
      }
      istripOff++;

      for (Int_t istrip = 0; istrip < NSTRIPC; istrip++) { // maximum number of strip is 19 for plate B and C
        auto string2 = fmt::format("FTOA_0/FLTA_0/FSTR_{:d}", istrip + istripOff);
        if (isector == 13 || isector == 14 || isector == 15) {
          if (iplate < 2) {
            string2 = fmt::format("FTOB_0/FLTB_0/FSTR_{:d}", istrip + istripOff);
          }
          if (iplate > 2) {
            string2 = fmt::format("FTOC_0/FLTC_0/FSTR_{:d}", istrip + istripOff);
          }
        }

        det[2] = istrip;
        if (!(iplate == 2 && istrip >= NSTRIPA)) { // the middle plate (A) has only 15 strips

          gGeoManager->cd(o2::utils::Str::concat_string(string1, '/', string2).c_str());
          TGeoHMatrix aliceToStrip = *gGeoManager->GetCurrentMatrix();
          TGeoHMatrix stripMat = sectorMat * aliceToStrip * myMatCoord; // strip in sector coordinate

          // load strip alignment parameters from current geometry
          const Double_t* tr = stripMat.GetTranslation();
          const Double_t* rot = stripMat.GetRotationMatrix();
          mGeoDistances[isector][iplate][istrip] = tr[1];                     // DISTANCES[iplate][istrip];
          mGeoHeights[isector][iplate][istrip] = tr[2] - (RMAX + RMIN) * 0.5; // HEIGHTS[iplate][istrip];
          mGeoX[isector][iplate][istrip] = tr[0];

          /*
          printf("TRANSLATION\n");
          printf("dist=%f, heights=%f, x=%f\n",DISTANCES[iplate][istrip],HEIGHTS[iplate][istrip],0.);
          printf("vs...\n");
          printf("dist=%lf, heights=%lf, x=%lf",tr[1],tr[2]-385.520360,tr[0]);
          printf("\n");
          */

          mRotationMatrixPlateStrip[isector][iplate][istrip][0][0] = rot[0];
          mRotationMatrixPlateStrip[isector][iplate][istrip][1][0] = rot[1];
          mRotationMatrixPlateStrip[isector][iplate][istrip][2][0] = rot[2];

          mRotationMatrixPlateStrip[isector][iplate][istrip][0][1] = rot[3];
          mRotationMatrixPlateStrip[isector][iplate][istrip][1][1] = rot[4];
          mRotationMatrixPlateStrip[isector][iplate][istrip][2][1] = rot[5];

          mRotationMatrixPlateStrip[isector][iplate][istrip][0][2] = rot[6];
          mRotationMatrixPlateStrip[isector][iplate][istrip][1][2] = rot[7];
          mRotationMatrixPlateStrip[isector][iplate][istrip][2][2] = rot[8];

          // load pad positions from current geometry
          for (Int_t ipadz = 0; ipadz < NPADZ; ipadz++) {
            det[3] = ipadz;
            for (Int_t ipadx = 0; ipadx < NPADX; ipadx++) {
              det[4] = ipadx;
              gGeoManager->cd(getVolumePath(det).c_str());
              TGeoHMatrix global;
              global = *gGeoManager->GetCurrentMatrix();
              tr = global.GetTranslation();
              mPadPosition[isector][iplate][istrip][ipadz][ipadx][0] = tr[0];
              mPadPosition[isector][iplate][istrip][ipadz][ipadx][1] = tr[1];
              mPadPosition[isector][iplate][istrip][ipadz][ipadx][2] = tr[2];
            }
          }
        }
      }
    }
  }

  /*
  for (Int_t isector = 0; isector < NSECTORS; isector++) {
    for (Int_t iplate = 0; iplate < NPLATES; iplate++) {
      for (Int_t istrip = 0; istrip < NSTRIPC; istrip++) {
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

  printf("SECTOR %d, PLATE %d, STRIP %d rotations\n",isector,iplate,istrip);
  printf("%f %f %f\n",mRotationMatrixPlateStrip[isector][iplate][istrip][0][0],mRotationMatrixPlateStrip[isector][iplate][istrip][0][1],mRotationMatrixPlateStrip[isector][iplate][istrip][0][2]);
  printf("%f %f %f\n",mRotationMatrixPlateStrip[isector][iplate][istrip][1][0],mRotationMatrixPlateStrip[isector][iplate][istrip][1][1],mRotationMatrixPlateStrip[isector][iplate][istrip][1][2]);
  printf("%f %f %f\n",mRotationMatrixPlateStrip[isector][iplate][istrip][2][0],mRotationMatrixPlateStrip[isector][iplate][istrip][2][1],mRotationMatrixPlateStrip[isector][iplate][istrip][2][2]);
  printf("vs...\n");

  for (Int_t ii = 0; ii < 3; ii++) {
    mRotationMatrixPlateStrip[isector][iplate][istrip][ii][0] =
      TMath::Sin(rotationAngles[2 * ii]) * TMath::Cos(rotationAngles[2 * ii + 1]);
    mRotationMatrixPlateStrip[isector][iplate][istrip][ii][1] =
      TMath::Sin(rotationAngles[2 * ii]) * TMath::Sin(rotationAngles[2 * ii + 1]);
    mRotationMatrixPlateStrip[isector][iplate][istrip][ii][2] = TMath::Cos(rotationAngles[2 * ii]);
  }

  printf("%f %f %f\n",mRotationMatrixPlateStrip[isector][iplate][istrip][0][0],mRotationMatrixPlateStrip[isector][iplate][istrip][0][1],mRotationMatrixPlateStrip[isector][iplate][istrip][0][2]);
  printf("%f %f %f\n",mRotationMatrixPlateStrip[isector][iplate][istrip][1][0],mRotationMatrixPlateStrip[isector][iplate][istrip][1][1],mRotationMatrixPlateStrip[isector][iplate][istrip][1][2]);
  printf("%f %f %f\n",mRotationMatrixPlateStrip[isector][iplate][istrip][2][0],mRotationMatrixPlateStrip[isector][iplate][istrip][2][1],mRotationMatrixPlateStrip[isector][iplate][istrip][2][2]);
  printf("\n");

      }
    }
  }
  */

  InitIndices();
  mToBeInit = kFALSE;
}

void Geo::InitIdeal()
{
  mToBeInit = true;
  mToBeInitIndexing = true;

  if (!mToBeInit) {
    return;
  }
  LOG(info) << "tof::Geo: Initialization of TOF rotation parameters with ideal";

  Double_t rotationAngles[6] =
    {90., 90. /*+ (isector + 0.5) * PHISEC*/, 0., 0., 90., 0 /* + (isector + 0.5) * PHISEC*/};
  for (Int_t ii = 0; ii < 6; ii++) {
    rotationAngles[ii] *= TMath::DegToRad();
  }

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

  for (Int_t isector = 0; isector < NSECTORS; isector++) {
    for (Int_t iplate = 0; iplate < NPLATES; iplate++) {
      for (Int_t istrip = 0; istrip < NSTRIPC; istrip++) {
        mGeoDistances[isector][iplate][istrip] = DISTANCES[iplate][istrip];
        mGeoHeights[isector][iplate][istrip] = HEIGHTS[iplate][istrip];
        mGeoX[isector][iplate][istrip] = 0.0;

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
          mRotationMatrixPlateStrip[isector][iplate][istrip][ii][0] =
            TMath::Sin(rotationAngles[2 * ii]) * TMath::Cos(rotationAngles[2 * ii + 1]);
          mRotationMatrixPlateStrip[isector][iplate][istrip][ii][1] =
            TMath::Sin(rotationAngles[2 * ii]) * TMath::Sin(rotationAngles[2 * ii + 1]);
          mRotationMatrixPlateStrip[isector][iplate][istrip][ii][2] = TMath::Cos(rotationAngles[2 * ii]);
        }
      }
    }
  }

  InitIndices();
  mToBeInit = kFALSE;

  for (Int_t isector = 0; isector < NSECTORS; isector++) {
    for (Int_t iplate = 0; iplate < NPLATES; iplate++) {
      for (Int_t istrip = 0; istrip < NSTRIPC; istrip++) {
        for (int iz = 0; iz < 2; iz++) {
          for (int ix = 0; ix < 48; ix++) {
            int det[] = {isector, iplate, istrip, iz, ix};
            float posPad[3] = {0., 0., 0.};
            getPosInPadCoord(det, posPad);
            antiRotateToStrip(posPad, iplate, istrip, isector);
            antiRotateToSector(posPad, 18);
            antiRotateToSector(posPad, isector);

            mPadPosition[isector][iplate][istrip][iz][ix][0] = -posPad[0];
            mPadPosition[isector][iplate][istrip][iz][ix][1] = -posPad[1];
            mPadPosition[isector][iplate][istrip][iz][ix][2] = -posPad[2];
          }
        }
      }
    }
  }
}

void Geo::InitIndices()
{
  if (!mToBeInitIndexing) {
    return;
  }
  mToBeInitIndexing = kFALSE;
  // initialization of some indices arrays

  for (Int_t istrip = 0; istrip < NSTRIPXSECTOR; ++istrip) {
    if (istrip < NSTRIPC) {
      mPlate[istrip] = 0;
      mStripInPlate[istrip] = istrip;
    } else if (istrip >= NSTRIPC && istrip < NSTRIPC + NSTRIPB) {
      mPlate[istrip] = 1;
      mStripInPlate[istrip] = istrip - NSTRIPC;
    } else if (istrip >= NSTRIPC + NSTRIPB && istrip < NSTRIPC + NSTRIPB + NSTRIPA) {
      mPlate[istrip] = 2;
      mStripInPlate[istrip] = istrip - NSTRIPC - NSTRIPB;
    } else if (istrip >= NSTRIPC + NSTRIPB + NSTRIPA && istrip < NSTRIPC + NSTRIPB + NSTRIPA + NSTRIPB) {
      mPlate[istrip] = 3;
      mStripInPlate[istrip] = istrip - NSTRIPC - NSTRIPB - NSTRIPA;
    } else if (istrip >= NSTRIPC + NSTRIPB + NSTRIPA + NSTRIPB && istrip < NSTRIPXSECTOR) {
      mPlate[istrip] = 4;
      mStripInPlate[istrip] = istrip - NSTRIPC - NSTRIPB - NSTRIPA - NSTRIPB;
    }
  }

  Int_t nstrips = NSTRIPC;
  for (Int_t isector = 0; isector < NSECTORS; isector++) {
    for (int iplate = 0; iplate < 5; ++iplate) {
      if (iplate == 1 || iplate == 3) {
        nstrips = NSTRIPB;
      } else if (iplate == 2) {
        nstrips = NSTRIPA;
      }
      mDistances[isector][iplate].reserve(nstrips);
      for (int i = 0; i < nstrips; ++i) {
        mDistances[isector][iplate].push_back(getGeoDistances(isector, iplate, nstrips - i - 1));
      }
    }
  }
}

std::string Geo::getVolumePath(const Int_t* ind)
{
  //--------------------------------------------------------------------
  // This function returns the volume path of a given pad
  //--------------------------------------------------------------------
  Int_t sector = ind[0];

  Int_t icopy = -1;
  icopy = sector;

  auto string1 = fmt::format("/cave_1/barrel_1/B077_1/BSEGMO{:d}_1/BTOF{:d}_1", icopy, icopy);

  Bool_t fgHoles = kTRUE;

  Int_t iplate = ind[1];
  Int_t istrip = ind[2];
  if (iplate == 0) {
    icopy = istrip;
  }
  if (iplate == 1) {
    icopy = istrip + NSTRIPC;
  }
  if (iplate == 2) {
    icopy = istrip + NSTRIPC + NSTRIPB;
  }
  if (iplate == 3) {
    icopy = istrip + NSTRIPC + NSTRIPB + NSTRIPA;
  }
  if (iplate == 4) {
    icopy = istrip + NSTRIPC + 2 * NSTRIPB + NSTRIPA;
  }
  icopy++;
  auto string2 = fmt::format("FTOA_0/FLTA_0/FSTR_{:d}", icopy);
  if (fgHoles && (sector == 13 || sector == 14 || sector == 15)) {
    if (iplate < 2) {
      string2 = fmt::format("FTOB_0/FLTB_0/FSTR_{:d}", icopy);
    }
    if (iplate > 2) {
      string2 = fmt::format("FTOC_0/FLTC_0/FSTR_{:d}", icopy);
    }
  }

  Int_t padz = ind[3] + 1;
  Int_t padx = ind[4] + 1;
  return o2::utils::Str::concat_string(string1, '/', string2, '/', fmt::format("FPCB_1/FSEN_1/FSEZ_{:d}/FPAD_{:d}", padz, padx));
}

void Geo::getPos(Int_t* det, Float_t* pos)
{
  //
  // Returns space point coor (x,y,z) (cm)  for Detector
  // Indices  (iSect,iPlate,iStrip,iPadZ,iPadX)
  //
  if (mToBeInit) {
    Init();
  }

  //  printf("TOFDBG: %d, %d, %d, %d, %d    ->    %f %f %f\n", det[0], det[1], det[2], det[3], det[4], mPadPosition[det[0]][det[1]][det[2]][det[3]][det[4]][0], mPadPosition[det[0]][det[1]][det[2]][det[3]][det[4]][1], mPadPosition[det[0]][det[1]][det[2]][det[3]][det[4]][2]);
  pos[0] = mPadPosition[det[0]][det[1]][det[2]][det[3]][det[4]][0];
  pos[1] = mPadPosition[det[0]][det[1]][det[2]][det[3]][det[4]][1];
  pos[2] = mPadPosition[det[0]][det[1]][det[2]][det[3]][det[4]][2];
}

void Geo::getDetID(Float_t* pos, Int_t* det)
{
  //
  // Returns Detector Indices (iSect,iPlate,iStrip,iPadZ,iPadX)
  // space point coor (x,y,z) (cm)

  if (mToBeInit) {
    Init();
  }

  Float_t posLocal[3];
  for (Int_t ii = 0; ii < 3; ii++) {
    posLocal[ii] = pos[ii];
  }

  det[0] = getSector(posLocal);
  if (det[0] == -1) {
    return;
  }

  fromGlobalToSector(posLocal, det[0]);

  det[1] = getPlate(posLocal);
  if (det[1] == -1) {
    return;
  }

  det[2] = fromPlateToStrip(posLocal, det[1], det[0]);
  if (det[2] == -1) {
    return;
  }

  det[3] = getPadZ(posLocal);
  det[4] = getPadX(posLocal);
}

void Geo::getVolumeIndices(Int_t index, Int_t* detId)
{
  //
  // Retrieve volume indices from the calibration channel index
  //

  if (mToBeInitIndexing) {
    InitIndices();
  }
  detId[0] = index * NPADS_INV_INT * NSTRIPXSECTOR_INV_INT;

  Int_t dummyStripPerModule = index / NPADS - NSTRIPXSECTOR * detId[0];
  detId[1] = mPlate[dummyStripPerModule];
  detId[2] = mStripInPlate[dummyStripPerModule];
  Int_t padPerStrip = index - (NSTRIPXSECTOR * detId[0] + dummyStripPerModule) * NPADS;

  detId[3] = padPerStrip / NPADX;            // padZ
  detId[4] = padPerStrip - detId[3] * NPADX; // padX
}

Int_t Geo::getIndex(const Int_t* detId)
{
  //Retrieve calibration channel index
  Int_t isector = detId[0];
  if (isector >= NSECTORS) {
    printf("Wrong sector number in TOF (%d) !\n", isector);
    return -1;
  }
  Int_t iplate = detId[1];
  if (iplate >= NPLATES) {
    printf("Wrong plate number in TOF (%d) !\n", iplate);
    return -1;
  }
  Int_t istrip = detId[2];
  Int_t stripOffset = getStripNumberPerSM(iplate, istrip);
  if (stripOffset == -1) {
    printf("Wrong strip number per SM in TOF (%d) !\n", stripOffset);
    return -1;
  }

  Int_t ipadz = detId[3];
  Int_t ipadx = detId[4];

  Int_t idet = ((2 * (NSTRIPC + NSTRIPB) + NSTRIPA) * NPADZ * NPADX) * isector +
               (stripOffset * NPADZ * NPADX) +
               (NPADX)*ipadz +
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

  Bool_t check = ((iplate < 0 || iplate >= NPLATES) ||
                  ((iplate == 2 && (istrip < 0 || istrip >= NSTRIPA)) ||
                   (iplate != 2 && (istrip < 0 || istrip >= NSTRIPC))));

  if (iplate < 0 || iplate >= NPLATES) {
    LOG(error) << "getStripNumberPerSM : "
               << "Wrong plate number in TOF (" << iplate << ")!\n";
  }

  if (
    (iplate == 2 && (istrip < 0 || istrip >= NSTRIPA)) ||
    (iplate != 2 && (istrip < 0 || istrip >= NSTRIPC))) {
    LOG(error) << "getStripNumberPerSM : "
               << " Wrong strip number in TOF (strip=" << istrip << " in the plate= " << iplate << ")!\n";
  }

  Int_t stripOffset = 0;
  switch (iplate) {
    case 0:
      stripOffset = 0;
      break;
    case 1:
      stripOffset = NSTRIPC;
      break;
    case 2:
      stripOffset = NSTRIPC + NSTRIPB;
      break;
    case 3:
      stripOffset = NSTRIPC + NSTRIPB + NSTRIPA;
      break;
    case 4:
      stripOffset = NSTRIPC + NSTRIPB + NSTRIPA + NSTRIPB;
      break;
  };

  if (!check) {
    index = stripOffset + istrip;
  }

  return index;
}

void Geo::getPosInSectorCoord(const Int_t* detId, float* pos)
{
  Init();
  fromGlobalToSector(pos, detId[0]);

  float swap = pos[0];
  pos[0] = pos[1];
  pos[1] = swap;
  pos[2] = -pos[2];
}

void Geo::getPosInStripCoord(const Int_t* detId, float* pos)
{
  Init();
  fromGlobalToSector(pos, detId[0]);

  float step[3];
  step[0] = getGeoX(detId[0], detId[1], detId[2]);
  step[1] = getGeoHeights(detId[0], detId[1], detId[2]);
  step[2] = -getGeoDistances(detId[0], detId[1], detId[2]);
  translate(pos[0], pos[1], pos[2], step);
  rotateToStrip(pos, detId[1], detId[2], detId[0]);
}

void Geo::getPosInPadCoord(const Int_t* detId, float* pos)
{
  Init();
  fromGlobalToSector(pos, detId[0]);

  float step[3];
  step[0] = getGeoX(detId[0], detId[1], detId[2]);
  step[1] = getGeoHeights(detId[0], detId[1], detId[2]);
  step[2] = -getGeoDistances(detId[0], detId[1], detId[2]);
  translate(pos[0], pos[1], pos[2], step);
  rotateToStrip(pos, detId[1], detId[2], detId[0]);

  pos[0] -= (detId[4] + 0.5) * XPAD - XHALFSTRIP;
  pos[2] -= (detId[3] - 0.5) * ZPAD;
}

void Geo::getPosInSectorCoord(int ch, float* pos)
{
  int det[5];
  getVolumeIndices(ch, det);
  getPosInSectorCoord(det, pos);
}

void Geo::getPosInStripCoord(int ch, float* pos)
{
  int det[5];
  getVolumeIndices(ch, det);
  getPosInStripCoord(det, pos);
}

void Geo::getPosInPadCoord(int ch, float* pos)
{
  int det[5];
  getVolumeIndices(ch, det);
  getPosInPadCoord(det, pos);
}

void Geo::fromGlobalToSector(Float_t* pos, Int_t isector)
{
  if (isector == -1) {
    //LOG(error) << "Sector Index not valid (-1)\n";
    return;
  }

  // ALICE reference frame -> B071/B074/B075 = BTO1/2/3 reference frame
  rotateToSector(pos, isector);

  Float_t step[3] = {0., 0., static_cast<Float_t>((RMAX + RMIN) * 0.5)};
  translate(pos, step);

  // B071/B074/B075 = BTO1/2/3 reference frame -> FTOA = FLTA reference frame
  rotateToSector(pos, NSECTORS);
}

Int_t Geo::fromPlateToStrip(Float_t* pos, Int_t iplate, Int_t isector)
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

  constexpr Float_t HGLFY = HFILIY + 2 * HGLASSY;                                         // heigth of GLASS+FISHLINE  Layer
  constexpr Float_t HSTRIPY = 2. * HHONY + 2. * HPCBY + 4. * HRGLY + 2. * HGLFY + HCPCBY; // 3.11

  Float_t step[3];

  // FTOA/B/C = FLTA/B/C reference frame -> FSTR reference frame

  // we restrict the search of the strip to a more reasonable range, considering that the
  // DISTANCES are never larger than 10 between consecutive strips
  auto ilower = std::lower_bound(mDistances[isector][iplate].begin(), mDistances[isector][iplate].end(), -pos[2]);
  int stripFound = mDistances[isector][iplate].size() - 1 - std::distance(mDistances[isector][iplate].begin(), ilower);
  int firstStripToCheck = stripFound;
  int lastStripToCheck = stripFound;
  if (stripFound != 0) {
    while (std::abs(pos[2] + getGeoDistances(isector, iplate, firstStripToCheck - 1)) < 10) {
      --firstStripToCheck;
    }
  }
  if (stripFound != nstrips - 1) {
    while (std::abs(pos[2] + getGeoDistances(isector, iplate, lastStripToCheck + 1)) < 10) {
      ++lastStripToCheck;
    }
  }

  for (Int_t istrip = firstStripToCheck; istrip <= lastStripToCheck; ++istrip) {
    Float_t posLoc2[3] = {pos[0], pos[1], pos[2]};

    step[0] = getGeoX(isector, iplate, istrip);
    step[1] = getGeoHeights(isector, iplate, istrip);
    step[2] = -getGeoDistances(isector, iplate, istrip);
    translate(posLoc2[0], posLoc2[1], posLoc2[2], step);

    if (fabs(posLoc2[1]) > 10) {
      continue;
    }
    if (fabs(posLoc2[2]) > 10) {
      continue;
    }

    float distanceSquared = posLoc2[1] * posLoc2[1] + posLoc2[2] * posLoc2[2];

    if (distanceSquared > 45) {
      continue;
    }

    rotateToStrip(posLoc2, iplate, istrip, isector);

    if ((TMath::Abs(posLoc2[0]) <= STRIPLENGTH * 0.5) && (TMath::Abs(posLoc2[1]) <= HSTRIPY * 0.5) &&
        (TMath::Abs(posLoc2[2]) <= WCPCBZ * 0.5)) {
      step[0] = -0.5 * NPADX * XPAD;
      step[1] = 0.;
      step[2] = -0.5 * NPADZ * ZPAD;
      //translate(posLoc2, step);
      translate(posLoc2[0], posLoc2[1], posLoc2[2], step);

      for (Int_t jj = 0; jj < 3; ++jj) {
        pos[jj] = posLoc2[jj];
      }

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

  //Float_t x = pos[0];
  //Float_t y = pos[1];
  //Float_t z = pos[2];

  Float_t rho2 = pos[0] * pos[0] + pos[1] * pos[1];

  if (!((pos[2] >= -ZLENA * 0.5 && pos[2] <= ZLENA * 0.5) && (rho2 >= (RMIN2) && rho2 <= (RMAX2)))) {
    // AliError("Detector Index could not be determined");
    return iSect;
  }

  Float_t phi = TMath::Pi() + o2::math_utils::fastATan2(-pos[1], -pos[0]);

  iSect = (Int_t)(phi * TMath::RadToDeg() * PHISECINV);

  return iSect;
}

void Geo::getPadDxDyDz(const Float_t* pos, Int_t* det, Float_t* DeltaPos, int sector)
{
  //
  // Returns the x coordinate in the Pad reference frame
  //
  if (mToBeInit) {
    Init();
  }

  for (Int_t ii = 0; ii < 3; ii++) {
    DeltaPos[ii] = pos[ii];
  }

  det[0] = sector;
  if (det[0] == -1) {
    det[0] = getSector(DeltaPos);

    if (det[0] == -1) {
      return;
    }
  }

  fromGlobalToSector(DeltaPos, det[0]);
  det[1] = getPlate(DeltaPos);
  if (det[1] == -1) {
    return;
  }

  det[2] = fromPlateToStrip(DeltaPos, det[1], det[0]);
  if (det[2] == -1) {
    return;
  }

  det[3] = getPadZ(DeltaPos);
  det[4] = getPadX(DeltaPos);
  // translate to the pad center

  Float_t step[3];

  step[0] = (det[4] + 0.5) * XPAD;

  step[1] = 0.;

  step[2] = (det[3] + 0.5) * ZPAD;
  translate(DeltaPos, step);
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
      if (zLocal < 0) {
        iPlate = 0;
      } else {
        iPlate = 4;
      }
    } else {
      if (zLocal < 0) {
        iPlate = 1;
      } else {
        iPlate = 3;
      }
    }
  } else if (TMath::Abs(zLocal) >= INTERCENTRMODBORDER1 && TMath::Abs(zLocal) <= INTERCENTRMODBORDER2) {
    deltaRhoLoc -= LENGTHINCEMODBORDERD;
    deltaZetaLoc = deltaZetaLoc - INTERCENTRMODBORDER1;
    deltaRHOmax = (RMAX - RMIN) * 0.5 - MODULEWALLTHICKNESS - 2. * LENGTHINCEMODBORDERD; // old 0.39, new 0.2

    if (deltaRhoLoc > deltaZetaLoc * deltaRHOmax / (INTERCENTRMODBORDER2 - INTERCENTRMODBORDER1)) {
      iPlate = 2;
    } else {
      if (zLocal < 0) {
        iPlate = 1;
      } else {
        iPlate = 3;
      }
    }
  }

  if (zLocal > -ZLENA * 0.5 && zLocal < -EXTERINTERMODBORDER2) {
    iPlate = 0;
  } else if (zLocal > -EXTERINTERMODBORDER1 && zLocal < -INTERCENTRMODBORDER2) {
    iPlate = 1;
  } else if (zLocal > -INTERCENTRMODBORDER1 && zLocal < INTERCENTRMODBORDER1) {
    iPlate = 2;
  } else if (zLocal > INTERCENTRMODBORDER2 && zLocal < EXTERINTERMODBORDER1) {
    iPlate = 3;
  } else if (zLocal > EXTERINTERMODBORDER2 && zLocal < ZLENA * 0.5) {
    iPlate = 4;
  }

  return iPlate;
}

Int_t Geo::getPadZ(const Float_t* pos)
{
  //
  // Returns the Pad index along Z
  //

  Int_t iPadZ = (Int_t)(pos[2] / ZPAD);
  if (iPadZ == NPADZ) {
    iPadZ--;
  } else if (iPadZ > NPADZ) {
    iPadZ = -1;
  }

  return iPadZ;
}
//_____________________________________________________________________________
Int_t Geo::getPadX(const Float_t* pos)
{
  //
  // Returns the Pad index along X
  //

  Int_t iPadX = (Int_t)(pos[0] / XPAD);
  if (iPadX == NPADX) {
    iPadX--;
  } else if (iPadX > NPADX) {
    iPadX = -1;
  }

  return iPadX;
}

void Geo::translate(Float_t* xyz, Float_t translationVector[3])
{
  //
  // Return the vector xyz translated by translationVector vector
  //

  for (Int_t ii = 0; ii < 3; ii++) {
    xyz[ii] -= translationVector[ii];
  }

  return;
}
void Geo::translate(Float_t& x, Float_t& y, Float_t& z, Float_t translationVector[3])
{
  //
  // Return the vector xyz translated by translationVector vector
  //

  x -= translationVector[0];
  y -= translationVector[1];
  z -= translationVector[2];

  /*
  for (Int_t ii = 0; ii < 3; ii++) {
    xyz[ii] -= translationVector[ii];
  }
  */
  return;
}
void Geo::antiRotateToSector(Float_t* xyz, Int_t isector)
{
  if (mToBeInit) {
    Init();
  }

  Float_t xyzDummy[3] = {0., 0., 0.};

  for (Int_t ii = 0; ii < 3; ii++) {
    xyzDummy[ii] = xyz[0] * mRotationMatrixSector[isector][0][ii] + xyz[1] * mRotationMatrixSector[isector][1][ii] +
                   xyz[2] * mRotationMatrixSector[isector][2][ii];
  }

  for (Int_t ii = 0; ii < 3; ii++) {
    xyz[ii] = xyzDummy[ii];
  }

  return;
}

void Geo::rotateToSector(Float_t* xyz, Int_t isector)
{
  if (mToBeInit) {
    Init();
  }

  Float_t xyzDummy[3] = {0., 0., 0.};

  for (Int_t ii = 0; ii < 3; ii++) {
    xyzDummy[ii] = xyz[0] * mRotationMatrixSector[isector][ii][0] + xyz[1] * mRotationMatrixSector[isector][ii][1] +
                   xyz[2] * mRotationMatrixSector[isector][ii][2];
  }

  for (Int_t ii = 0; ii < 3; ii++) {
    xyz[ii] = xyzDummy[ii];
  }

  return;
}

void Geo::antiRotateToStrip(Float_t* xyz, Int_t iplate, Int_t istrip, Int_t isector)
{
  Float_t xyzDummy[3] = {0., 0., 0.};

  for (Int_t ii = 0; ii < 3; ii++) {
    xyzDummy[ii] = xyz[0] * mRotationMatrixPlateStrip[isector][iplate][istrip][0][ii] +
                   xyz[1] * mRotationMatrixPlateStrip[isector][iplate][istrip][1][ii] +
                   xyz[2] * mRotationMatrixPlateStrip[isector][iplate][istrip][2][ii];
  }

  for (Int_t ii = 0; ii < 3; ii++) {
    xyz[ii] = xyzDummy[ii];
  }

  return;
}

void Geo::rotateToStrip(Float_t* xyz, Int_t iplate, Int_t istrip, Int_t isector)
{
  Float_t xyzDummy[3] = {0., 0., 0.};

  for (Int_t ii = 0; ii < 3; ii++) {
    xyzDummy[ii] = xyz[0] * mRotationMatrixPlateStrip[isector][iplate][istrip][ii][0] +
                   xyz[1] * mRotationMatrixPlateStrip[isector][iplate][istrip][ii][1] +
                   xyz[2] * mRotationMatrixPlateStrip[isector][iplate][istrip][ii][2];
  }

  for (Int_t ii = 0; ii < 3; ii++) {
    xyz[ii] = xyzDummy[ii];
  }

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

  for (Int_t ii = 0; ii < 6; ii++) {
    rotationAngles[ii] *= TMath::DegToRad();
  }

  Float_t xyzDummy[3] = {0., 0., 0.};

  for (Int_t ii = 0; ii < 3; ii++) {
    xyzDummy[ii] = xyz[0] * TMath::Sin(rotationAngles[2 * ii]) * TMath::Cos(rotationAngles[2 * ii + 1]) +
                   xyz[1] * TMath::Sin(rotationAngles[2 * ii]) * TMath::Sin(rotationAngles[2 * ii + 1]) +
                   xyz[2] * TMath::Cos(rotationAngles[2 * ii]);
  }

  for (Int_t ii = 0; ii < 3; ii++) {
    xyz[ii] = xyzDummy[ii];
  }

  return;
}

void Geo::antiRotate(Float_t* xyz, Double_t rotationAngles[6])
{
  //
  // Rotates the vector xyz acordint to the rotationAngles
  //

  for (Int_t ii = 0; ii < 6; ii++) {
    rotationAngles[ii] *= TMath::DegToRad();
  }

  Float_t xyzDummy[3] = {0., 0., 0.};

  xyzDummy[0] = xyz[0] * TMath::Sin(rotationAngles[0]) * TMath::Cos(rotationAngles[1]) +
                xyz[1] * TMath::Sin(rotationAngles[2]) * TMath::Cos(rotationAngles[3]) +
                xyz[2] * TMath::Sin(rotationAngles[4]) * TMath::Cos(rotationAngles[5]);

  xyzDummy[1] = xyz[0] * TMath::Sin(rotationAngles[0]) * TMath::Sin(rotationAngles[1]) +
                xyz[1] * TMath::Sin(rotationAngles[2]) * TMath::Sin(rotationAngles[3]) +
                xyz[2] * TMath::Sin(rotationAngles[4]) * TMath::Sin(rotationAngles[5]);

  xyzDummy[2] = xyz[0] * TMath::Cos(rotationAngles[0]) + xyz[1] * TMath::Cos(rotationAngles[2]) +
                xyz[2] * TMath::Cos(rotationAngles[4]);

  for (Int_t ii = 0; ii < 3; ii++) {
    xyz[ii] = xyzDummy[ii];
  }

  return;
}

Int_t Geo::getIndexFromEquipment(Int_t icrate, Int_t islot, Int_t ichain, Int_t itdc)
{
  return 0; // to be implemented
}

void Geo::getStripAndModule(Int_t iStripPerSM, Int_t& iplate, Int_t& istrip)
{
  //
  // Convert the serial number of the TOF strip number iStripPerSM [0,90]
  // in module number iplate [0,4] and strip number istrip [0,14/18].
  // Copied from AliRoot TOF::AliTOFGeometry
  //

  if (iStripPerSM < 0 || iStripPerSM >= NSTRIPC + NSTRIPB + NSTRIPA + NSTRIPB + NSTRIPC) {
    iplate = -1;
    istrip = -1;
  } else if (iStripPerSM < NSTRIPC) {
    iplate = 0;
    istrip = iStripPerSM;
  } else if (iStripPerSM >= NSTRIPC && iStripPerSM < NSTRIPC + NSTRIPB) {
    iplate = 1;
    istrip = iStripPerSM - NSTRIPC;
  } else if (iStripPerSM >= NSTRIPC + NSTRIPB && iStripPerSM < NSTRIPC + NSTRIPB + NSTRIPA) {
    iplate = 2;
    istrip = iStripPerSM - NSTRIPC - NSTRIPB;
  } else if (iStripPerSM >= NSTRIPC + NSTRIPB + NSTRIPA && iStripPerSM < NSTRIPC + NSTRIPB + NSTRIPA + NSTRIPB) {
    iplate = 3;
    istrip = iStripPerSM - NSTRIPC - NSTRIPB - NSTRIPA;
  } else if (iStripPerSM >= NSTRIPC + NSTRIPB + NSTRIPA + NSTRIPB && iStripPerSM < NSTRIPC + NSTRIPB + NSTRIPA + NSTRIPB + NSTRIPC) {
    iplate = 4;
    istrip = iStripPerSM - NSTRIPC - NSTRIPB - NSTRIPA - NSTRIPB;
  }
}
