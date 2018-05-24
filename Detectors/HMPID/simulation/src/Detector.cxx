// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "../../simulation/include/HMPIDSimulation/Detector.h"

#include "TGeoManager.h"
#include "TGeoShapeAssembly.h"
#include "TGeoNode.h"
#include "TGeoBBox.h"
#include "TGeoXtru.h"
#include "TGeoCompositeShape.h"
#include "TVirtualMC.h"
#include "TF1.h"
#include "TF2.h"
//#include "TGeoHMatrix.h"
#include "DetectorsBase/MaterialManager.h"

namespace o2
{
namespace hmpid
{

Detector::Detector(Bool_t active) : o2::Base::DetImpl<Detector>("HMP", active), mHits(new std::vector<HitType>) {}

bool Detector::ProcessHits(FairVolume* v)
{
  // later on return true if there was a hit!
  return false;
}

void Detector::Register() { FairRootManager::Instance()->RegisterAny(addNameTo("Hit").data(), mHits, true); }

void Detector::Reset() {}

void Detector::defineSensitiveVolumes()
{
  // define sensitive volumes here
}

void Detector::Initialize()
{
  // register the sensitive volumes with FairRoot
  defineSensitiveVolumes();
  o2::Base::Detector::Initialize();
}

//*********************************************************************************************************

void Detector::IdealPosition(Int_t iCh, TGeoHMatrix* pMatrix) // ideal position of given chamber
{
  // Construct ideal position matrix for a given chamber
  // Arguments: iCh- chamber ID; pMatrix- pointer to precreated unity matrix where to store the results
  //   Returns: none
  const double kAngHor = 19.5;            //  horizontal angle between chambers  19.5 grad
  const double kAngVer = 20;              //  vertical angle between chambers    20   grad
  const double kAngCom = 30;              //  common HMPID rotation with respect to x axis  30   grad
  const double kTrans[3] = { 490, 0, 0 }; //  center of the chamber is on window-gap surface
  pMatrix->RotateY(90);                   //  rotate around y since initial position is in XY plane -> now in YZ plane
  pMatrix->SetTranslation(kTrans);        //  now plane in YZ is shifted along x
  switch (iCh) {
    case 0:
      pMatrix->RotateY(kAngHor);
      pMatrix->RotateZ(-kAngVer);
      break; // right and down
    case 1:
      pMatrix->RotateZ(-kAngVer);
      break; // down
    case 2:
      pMatrix->RotateY(kAngHor);
      break; // right
    case 3:
      break; // no rotation
    case 4:
      pMatrix->RotateY(-kAngHor);
      break; // left
    case 5:
      pMatrix->RotateZ(kAngVer);
      break; // up
    case 6:
      pMatrix->RotateY(-kAngHor);
      pMatrix->RotateZ(kAngVer);
      break; // left and up
  }
  pMatrix->RotateZ(kAngCom); // apply common rotation  in XY plane
}

void Detector::IdealPositionCradle(Int_t iCh, TGeoHMatrix* pMatrix) // ideal position of given one module of the cradle
{
  // Construct ideal position matrix for a given module cradle
  // Arguments: iCh- chamber ID; pMatrix- pointer to precreated unity matrix where to store the results
  //   Returns: none
  const double kAngHor = 19.5;                   //  horizontal angle between chambers  19.5 grad
  const double kAngVer = 20;                     //  vertical angle between chambers    20   grad
  const double kAngCom = 30;                     //  common HMPID rotation with respect to x axis  30   grad
  const double kTrans[3] = { 423. + 29, 0, 67 }; //  z-center of the cradle module
  pMatrix->RotateY(90);            //  rotate around y since initial position is in XY plane -> now in YZ plane
  pMatrix->SetTranslation(kTrans); //  now plane in YZ is shifted along x
  switch (iCh) {
    case 0:
      pMatrix->RotateY(kAngHor);
      pMatrix->RotateZ(-kAngVer);
      break; // right and down
    case 1:
      pMatrix->RotateZ(-kAngVer);
      break; // down
    case 2:
      pMatrix->RotateY(kAngHor);
      break; // right
    case 3:
      break; // no rotation
    case 4:
      pMatrix->RotateY(-kAngHor);
      break; // left
    case 5:
      pMatrix->RotateZ(kAngVer);
      break; // up
    case 6:
      pMatrix->RotateY(-kAngHor);
      pMatrix->RotateZ(kAngVer);
      break; // left and up
  }
  pMatrix->RotateZ(kAngCom); // apply common rotation  in XY plane
}

void Detector::createMaterials()
{
  // implement materials here
  // Definition of available HMPID materials
  // Arguments: none
  //   Returns: none

  // clm update material definition later on from Antonello
  // Ported to O2 (24.4.2018) -- taken from AliRoot AliHMPIDv3

  // data from PDG booklet 2002     density [gr/cm^3] rad len [cm] abs len [cm]
  float aAir[4] = { 12, 14, 16, 36 }, zAir[4] = { 6, 7, 8, 18 }, wAir[4] = { 0.000124, 0.755267, 0.231781, 0.012827 },
        dAir = 0.00120479;
  int nAir = 4; // mixture 0.9999999
  float aC6F14[2] = { 12.01, 18.99 }, zC6F14[2] = { 6, 9 }, wC6F14[2] = { 6, 14 }, dC6F14 = 1.68;
  int nC6F14 = -2;
  float aSiO2[2] = { 28.09, 15.99 }, zSiO2[2] = { 14, 8 }, wSiO2[2] = { 1, 2 }, dSiO2 = 2.64;
  int nSiO2 = -2;
  float aCH4[2] = { 12.01, 1.01 }, zCH4[2] = { 6, 1 }, wCH4[2] = { 1, 4 }, dCH4 = 7.17e-4;
  int nCH4 = -2;

  float aRoha = 12.01, zRoha = 6, dRoha = 0.10, radRoha = 18.80,
        absRoha = 86.3 / dRoha; // special material- quasi quartz
  float aCu = 63.55, zCu = 29, dCu = 8.96, radCu = 1.43, absCu = 134.9 / dCu;
  float aW = 183.84, zW = 74, dW = 19.30, radW = 0.35, absW = 185.0 / dW;
  float aAl = 26.98, zAl = 13, dAl = 2.70, radAl = 8.90, absAl = 106.4 / dAl;
  float aAr = 39.94, zAr = 18, dAr = 1.396e-3, radAr = 14.0, absAr = 117.2 / dAr;

  int matId = 0;            // tmp material id number
  int unsens = 0, sens = 1; // sensitive or unsensitive medium
  int itgfld;
  float maxfld;
  o2::Base::Detector::initFieldTrackingParams(itgfld, maxfld);

  float tmaxfd = -10.0; // max deflection angle due to magnetic field in one step
  float deemax = -0.2;  // max fractional energy loss in one step
  float stemax = -0.1;  // max step allowed [cm]
  float epsil = 0.001;  // abs tracking precision [cm]
  float stmin = -0.001; // min step size [cm] in continius process transport, negative value: choose it automatically

  // PCB copmposed mainly by G10 (Si,C,H,O) -> CsI is negligible (<500nm thick)
  // So what is called CsI has the optical properties of CsI, but the composition of G-10 (for delta elec, etc
  // production...)

  float aG10[4] = { 28.09, 12.01, 1.01, 16.00 };
  float zG10[4] = { 14., 6., 1., 8. };
  float wG10[4] = { 0.129060, 0.515016, 0.061873, 0.294050 };
  float dG10 = 1.7;
  int nG10 = 4;

  Mixture(++matId, "Air", aAir, zAir, dAir, nAir, wAir);
  Medium(kAir, "Air", matId, unsens, itgfld, maxfld, tmaxfd, stemax, deemax, epsil, stmin);

  Mixture(++matId, "C6F14", aC6F14, zC6F14, dC6F14, nC6F14, wC6F14);
  Medium(kC6F14, "C6F14", matId, unsens, itgfld, maxfld, tmaxfd, stemax, deemax, epsil, stmin);

  Mixture(++matId, "SiO2", aSiO2, zSiO2, dSiO2, nSiO2, wSiO2);
  Medium(kSiO2, "SiO2", matId, unsens, itgfld, maxfld, tmaxfd, stemax, deemax, epsil, stmin);

  Mixture(++matId, "CH4", aCH4, zCH4, dCH4, nCH4, wCH4);
  Medium(kCH4, "CH4", matId, unsens, itgfld, maxfld, tmaxfd, stemax, deemax, epsil, stmin);

  Mixture(++matId, "CsI+PCB", aG10, zG10, dG10, nG10, wG10);
  Medium(kCsI, "CsI", matId, sens, itgfld, maxfld, tmaxfd, stemax, deemax, epsil, stmin); // sensitive

  Mixture(++matId, "Neo", aSiO2, zSiO2, dSiO2, nSiO2, wSiO2);
  Medium(kNeo, "Neo", matId, unsens, itgfld, maxfld, tmaxfd, stemax, deemax, epsil, stmin); // clm neoceram

  Material(++matId, "Roha", aRoha, zRoha, dRoha, radRoha, absRoha);
  Medium(kRoha, "Roha", matId, unsens, itgfld, maxfld, tmaxfd, stemax, deemax, epsil, stmin); // Roha->honeycomb

  Material(++matId, "Cu", aCu, zCu, dCu, radCu, absCu);
  Medium(kCu, "Cu", matId, unsens, itgfld, maxfld, tmaxfd, stemax, deemax, epsil, stmin);

  Material(++matId, "W", aW, zW, dW, radW, absW);
  Medium(kW, "W", matId, unsens, itgfld, maxfld, tmaxfd, stemax, deemax, epsil, stmin);

  Material(++matId, "Al", aAl, zAl, dAl, radAl, absAl);
  Medium(kAl, "Al", matId, unsens, itgfld, maxfld, tmaxfd, stemax, deemax, epsil, stmin);

  Material(++matId, "Ar", aAr, zAr, dAr, radAr, absAr);
  Medium(kAr, "Ar", matId, unsens, itgfld, maxfld, tmaxfd, stemax, deemax, epsil, stmin);
}

//**************************************************************************************************

TGeoVolume* Detector::createChamber(int number)
{
  // Single module geometry building

  double cm = 1, mm = 0.1 * cm, um = 0.001 * mm; // default is cm

  TGeoVolume* hmp = new TGeoVolumeAssembly(Form("Hmp%i", number));

  auto& matmgr = o2::Base::MaterialManager::Instance();

  TGeoMedium* al = matmgr.getTGeoMedium("HMP_Al");
  TGeoMedium* ch4 = matmgr.getTGeoMedium("HMP_CH4");
  TGeoMedium* roha = matmgr.getTGeoMedium("HMP_Roha");
  TGeoMedium* neoc = matmgr.getTGeoMedium("HMP_Neo");
  TGeoMedium* c6f14 = matmgr.getTGeoMedium("HMP_C6F14");
  TGeoMedium* sio2 = matmgr.getTGeoMedium("HMP_SiO2");
  TGeoMedium* cu = matmgr.getTGeoMedium("HMP_Cu");
  TGeoMedium* w = matmgr.getTGeoMedium("HMP_W");
  TGeoMedium* csi = matmgr.getTGeoMedium("HMP_CsI");
  TGeoMedium* ar = matmgr.getTGeoMedium("HMP_Ar");

  TGeoRotation* rot = new TGeoRotation("HwireRot");
  rot->RotateY(90); // rotate wires around Y to be along X (initially along Z)
  TGeoVolume* sbo = gGeoManager->MakeBox("Hsbo", ch4, 1419 * mm / 2, 1378.00 * mm / 2, 50.5 * mm / 2); // 2072P1
  TGeoVolume* cov = gGeoManager->MakeBox("Hcov", al, 1419 * mm / 2, 1378.00 * mm / 2, 0.5 * mm / 2);
  TGeoVolume* hon = gGeoManager->MakeBox("Hhon", roha, 1359 * mm / 2, 1318.00 * mm / 2, 49.5 * mm / 2);
  TGeoVolume* rad = gGeoManager->MakeBox("Hrad", c6f14, 1330 * mm / 2, 413.00 * mm / 2, 24.0 * mm / 2); // 2011P1
  TGeoVolume* neo = gGeoManager->MakeBox("Hneo", neoc, 1330 * mm / 2, 413.00 * mm / 2, 4.0 * mm / 2);
  TGeoVolume* win = gGeoManager->MakeBox("Hwin", sio2, 1330 * mm / 2, 413.00 * mm / 2, 5.0 * mm / 2);
  TGeoVolume* si1 = gGeoManager->MakeBox("Hsi1", sio2, 1330 * mm / 2, 5.00 * mm / 2, 15.0 * mm / 2);
  TGeoVolume* si2 = gGeoManager->MakeBox("Hsi2", neoc, 10 * mm / 2, 403.00 * mm / 2, 15.0 * mm / 2);
  TGeoVolume* spa = gGeoManager->MakeTube("Hspa", sio2, 0 * mm, 5.00 * mm, 15.0 * mm / 2);
  TGeoVolume* fr4 = gGeoManager->MakeBox("Hfr4", ch4, 1407 * mm / 2, 1366.00 * mm / 2, 15.0 * mm / 2); // 2043P1
  TGeoVolume* f4a = gGeoManager->MakeBox("Hf4a", al, 1407 * mm / 2, 1366.00 * mm / 2, 10.0 * mm / 2);
  TGeoVolume* f4i = gGeoManager->MakeBox("Hf4i", ch4, 1323 * mm / 2, 1296.00 * mm / 2, 10.0 * mm / 2);
  TGeoVolume* col = gGeoManager->MakeTube("Hcol", cu, 0 * mm, 100.00 * um, 1323.0 * mm / 2);
  TGeoVolume* sec = gGeoManager->MakeBox("Hsec", ch4, 648 * mm / 2, 411.00 * mm / 2,
                                         6.2 * mm / 2); // sec=gap 2099P1 (6.2 = 4.45 + 0.05 (1/2 diameter wire)+1.7)

  double cellx = 8.04 * mm, celly = 8.4 * mm;
  int nPadX = 80, nPadY = 48;
  TGeoVolume* gap = gGeoManager->MakeBox("Hgap", ch4, cellx * nPadX / 2, celly * nPadY / 2,
                                         6.2 * mm / 2);  // x=8.04*80 y=8.4*48 z=pad+pad-ano+marign 2006p1
  TGeoVolume* row = gap->Divide("Hrow", 2, nPadY, 0, 0); // along Y->48 rows
  TGeoVolume* cel = row->Divide(Form("Hcel%i", number), 1, nPadX, 0, 0); // along X->80 cells
  TGeoVolume* cat = gGeoManager->MakeTube("Hcat", cu, 0.00 * mm, 50.00 * um, cellx / 2);
  TGeoVolume* ano = gGeoManager->MakeTube("Hano", w, 0.00 * mm, 20.00 * um, cellx / 2);
  TGeoVolume* pad = gGeoManager->MakeBox(Form("Hpad%i", number), csi, 7.54 * mm / 2, 7.90 * mm / 2,
                                         1.7 * mm / 2); // 2006P1 PCB material...
  TGeoVolume* fr1 = gGeoManager->MakeBox("Hfr1", al, 1463 * mm / 2, 1422.00 * mm / 2,
                                         58.3 * mm / 2); // 2040P1 and pad plane is excluded (62 - 2 - 17)
  TGeoVolume* fr1up = gGeoManager->MakeBox("Hfr1up", ch4, (1426.00 - 37.00) * mm / 2, (1385.00 - 37.00) * mm / 2,
                                           20.0 * mm / 2); // 2040P1

  TGeoVolume* fr1upcard = gGeoManager->MakeBox("Hfr1upcard", ch4, 662. * mm / 2., 425. * mm / 2.,
                                               19.0 * mm / 2); // needed to set the gassiplex

  TGeoVolume* fr1perUpBig = gGeoManager->MakeBox("Hfr1perUpBig", ch4, 1389 * mm / 2, 35 * mm / 2, 10 * mm / 2);
  TGeoVolume* fr1perUpSma =
    gGeoManager->MakeBox("Hfr1perUpSma", ch4, 35 * mm / 2, (1385 - 37 - 2 * 35) * mm / 2, 10 * mm / 2);
  TGeoVolume* fr1perDowBig = gGeoManager->MakeBox("Hfr1perDowBig", ch4, 1389 * mm / 2, 46 * mm / 2, 2.3 * mm / 2);
  TGeoVolume* fr1perDowSma =
    gGeoManager->MakeBox("Hfr1perDowSma", ch4, 46 * mm / 2, (1385 - 37 - 2 * 46) * mm / 2, 2.3 * mm / 2);

  TGeoVolume* ppf = gGeoManager->MakeBox("Hppf", al, 648 * mm / 2, 411.00 * mm / 2, 38.3 * mm / 2); // 2001P2
  TGeoVolume* lar = gGeoManager->MakeBox("Hlar", ar, 181 * mm / 2, 89.25 * mm / 2, 38.3 * mm / 2);  // 2001P2
  TGeoVolume* smo = gGeoManager->MakeBox("Hsmo", ar, 114 * mm / 2, 89.25 * mm / 2, 38.3 * mm / 2);  // 2001P2

  TGeoVolume* cufoil = gGeoManager->MakeBox("Hcufoil", csi, 662. * mm / 2., 425. * mm / 2.,
                                            1. * mm / 2.); // PCB foil at the back of the ppf with holes for GASSIPLEX
  TGeoVolume* rect = gGeoManager->MakeBox("Hrect", ch4, 48 * mm / 2, 19 * mm / 2., 1 * mm / 2.);

  TGeoVolume* fr3 = gGeoManager->MakeBox("Hfr3", al, 1463 * mm / 2, 1422 * mm / 2, 34 * mm / 2);          // 2041P1
  TGeoVolume* fr3up = gGeoManager->MakeBox("Hfr3up", ch4, 1323 * mm / 2, 1282 * mm / 2, 20 * mm / 2);     // 2041P1
  TGeoVolume* fr3down = gGeoManager->MakeBox("Hfr3down", ch4, 1437 * mm / 2, 1370 * mm / 2, 14 * mm / 2); // 2041P1

  TGeoVolume* proxgap1 = gGeoManager->MakeBox("Hproxgap1", ch4, 1407 * mm / 2, 1366.00 * mm / 2,
                                              (9. - 7.5) * mm / 2.); // methane volume between quartz and fr4
  TGeoVolume* proxgap2 = gGeoManager->MakeBox("Hproxgap2", ch4, 1407 * mm / 2, 1366.00 * mm / 2,
                                              (81.7 - 6.2 - 34. - 9. - 7.5) * mm / 2.); // methane volume between fr4
                                                                                        // and Hgap(tot height(81.7) -
                                                                                        // Hsec (6.2) - proxygap2 (34)
                                                                                        // -upper bound of fr4 (9+7.5))

  // ^ Y   z=         z=-12mm      z=98.25mm               ALIC->7xHmp (virtual)-->1xHsbo (virtual) --->2xHcov (real)
  // 2072P1
  // |  ____________________________________                                    |                   |-->1xHhon (real)
  // 2072P1
  // | |   ______     ____          ______  |                                   |
  //   |  |      |   |    |   *    |      | |                                   |->3xHrad (virtual) --->1xHneo (real)
  //   2011P1
  //   |  |50.5mm|   |24mm|   *    |45.5mm| |                                   |                   |-->1xHwin (real)
  //   2011P1
  //   |  |      |   |    |   *    |      | |                                   |                   |-->2xHsi1 (real)
  //   2011P1
  //   |  |      |   |____|   *    |______| |                                   |                   |-->2xHsi2 (real)
  //   2011P1
  //   |  |      |    ____    *     ______  |                                   |                   |->30xHspa (real)
  //   2011P1
  //   |  |      |   |    |   *    |      | |                                   |
  //   |  |      |   |    |   *    |      | |                                   |->1xHfr4 (vitual) --->1xHf4a
  //   (real)---->1xHf4i(virtual) 2043P1
  //   |  |  sb  |   | rad|   *    |      | |                                   |                  |-->322xHcol (real)
  //   2043P1
  //   |  |      |   |____|   *    |______| |                                   |
  //   |  |      |    ____    *     ______  |                                   |->1xHfr1 (real) --> 6xHppf(real)
  //   ---->8xHlar (virtual) 2001P1
  //   |  |      |   |    |   *    |      | |                                   | |--->8xHsmo (virtual) 2001P1
  //   |  |      |   |    |   *    |      | |                                   |
  //   |  |      |   |    |   *    |      | |                                   |-> 6xHgap (virtual) --->48xHrow
  //   (virtual) -->80xHcel (virtual) -->4xHcat (real) from p84 TDR
  //   |  |______|   |____|   *    |______| | |-->2xHano (real) from p84 TDR
  //   |____________________________________| |-->1xHpad (real) from p84 TDR
  //                                                       --->Z
  hmp->AddNode(sbo, 1, new TGeoTranslation(0 * mm, 0 * mm, -73.75 * mm)); // p.84 TDR
  sbo->AddNode(hon, 1, new TGeoTranslation(0 * mm, 0 * mm, 0 * mm));      // 2072P1
  sbo->AddNode(cov, 1, new TGeoTranslation(0 * mm, 0 * mm, +25 * mm));
  sbo->AddNode(cov, 2, new TGeoTranslation(0 * mm, 0 * mm, -25 * mm));
  hmp->AddNode(rad, 2, new TGeoTranslation(0 * mm, +434 * mm, -12.00 * mm));
  hmp->AddNode(rad, 1, new TGeoTranslation(0 * mm, 0 * mm, -12.00 * mm));
  hmp->AddNode(rad, 0, new TGeoTranslation(0 * mm, -434 * mm, -12.00 * mm));
  rad->AddNode(neo, 1, new TGeoTranslation(0 * mm, 0 * mm, -10.0 * mm));
  rad->AddNode(win, 1, new TGeoTranslation(0 * mm, 0 * mm, 9.5 * mm));
  rad->AddNode(si1, 1, new TGeoTranslation(0 * mm, -204 * mm, -0.5 * mm));
  rad->AddNode(si1, 2, new TGeoTranslation(0 * mm, +204 * mm, -0.5 * mm));
  rad->AddNode(si2, 1, new TGeoTranslation(-660 * mm, 0 * mm, -0.5 * mm));
  rad->AddNode(si2, 2, new TGeoTranslation(+660 * mm, 0 * mm, -0.5 * mm));
  for (Int_t i = 0; i < 3; i++)
    for (Int_t j = 0; j < 10; j++)
      rad->AddNode(spa, 10 * i + j,
                   new TGeoTranslation(-1330 * mm / 2 + 116 * mm + j * 122 * mm, (i - 1) * 105 * mm, -0.5 * mm));
  hmp->AddNode(fr4, 1, new TGeoTranslation(0 * mm, 0 * mm, 9.00 * mm)); // p.84 TDR
  for (int i = 1; i <= 322; i++)
    fr4->AddNode(col, i, new TGeoCombiTrans(0 * mm, -1296 / 2 * mm + i * 4 * mm, -5 * mm, rot)); // F4 2043P1
  fr4->AddNode(f4a, 1, new TGeoTranslation(0 * mm, 0 * mm, 2.5 * mm));
  f4a->AddNode(f4i, 1, new TGeoTranslation(0 * mm, 0 * mm, 0 * mm));
  hmp->AddNode(sec, 4, new TGeoTranslation(-335 * mm, +433 * mm, 78.6 * mm));
  hmp->AddNode(sec, 5, new TGeoTranslation(+335 * mm, +433 * mm, 78.6 * mm));
  hmp->AddNode(sec, 2, new TGeoTranslation(-335 * mm, 0 * mm, 78.6 * mm));
  hmp->AddNode(sec, 3, new TGeoTranslation(+335 * mm, 0 * mm, 78.6 * mm));
  hmp->AddNode(sec, 0, new TGeoTranslation(-335 * mm, -433 * mm, 78.6 * mm));
  hmp->AddNode(sec, 1, new TGeoTranslation(+335 * mm, -433 * mm, 78.6 * mm));
  sec->AddNode(gap, 1, new TGeoTranslation(0, 0, 0. * mm));
  cel->AddNode(cat, 1, new TGeoCombiTrans(0, 3.15 * mm, -2.70 * mm, rot)); // 4 cathode wires
  cel->AddNode(ano, 1, new TGeoCombiTrans(0, 2.00 * mm, -0.29 * mm, rot)); // 2 anod wires
  cel->AddNode(cat, 2, new TGeoCombiTrans(0, 1.05 * mm, -2.70 * mm, rot));
  cel->AddNode(cat, 3, new TGeoCombiTrans(0, -1.05 * mm, -2.70 * mm, rot));
  cel->AddNode(ano, 2, new TGeoCombiTrans(0, -2.00 * mm, -0.29 * mm, rot));
  cel->AddNode(cat, 4, new TGeoCombiTrans(0, -3.15 * mm, -2.70 * mm, rot));
  cel->AddNode(pad, 1, new TGeoTranslation(0, 0.00 * mm, 2.25 * mm)); // 1 pad

  hmp->AddNode(fr1, 1, new TGeoTranslation(0., 0., (80. + 1.7) * mm + 58.3 * mm / 2.));
  fr1->AddNode(fr1up, 1, new TGeoTranslation(0., 0., (58.3 * mm - 20.00 * mm) / 2.));

  fr1->AddNode(fr1perUpBig, 0,
               new TGeoTranslation(0., (1385 - 37 - 35) * mm / 2., (58.3 * mm - 20.00 * 2 * mm - 10.0 * mm) / 2.));
  fr1->AddNode(fr1perUpSma, 0,
               new TGeoTranslation((1426 - 37 - 35) * mm / 2., 0., (58.3 * mm - 20.00 * 2 * mm - 10.0 * mm) / 2.));
  fr1->AddNode(fr1perUpBig, 1,
               new TGeoTranslation(0., -(1385 - 37 - 35) * mm / 2., (58.3 * mm - 20.00 * 2 * mm - 10.0 * mm) / 2.));
  fr1->AddNode(fr1perUpSma, 1,
               new TGeoTranslation(-(1426 - 37 - 35) * mm / 2., 0., (58.3 * mm - 20.00 * 2 * mm - 10.0 * mm) / 2.));

  fr1->AddNode(fr1perDowBig, 0, new TGeoTranslation(0., (1385 - 37) * mm / 2., (-58.3 * mm + 2.3 * mm) / 2.));
  fr1->AddNode(fr1perDowSma, 0, new TGeoTranslation((1426 - 37) * mm / 2., 0., (-58.3 * mm + 2.3 * mm) / 2.));
  fr1->AddNode(fr1perDowBig, 1, new TGeoTranslation(0., -(1385 - 37) * mm / 2., (-58.3 * mm + 2.3 * mm) / 2.));
  fr1->AddNode(fr1perDowSma, 1, new TGeoTranslation(-(1426 - 37) * mm / 2., 0., (-58.3 * mm + 2.3 * mm) / 2.));

  fr1->AddNode(ppf, 4, new TGeoTranslation(-335 * mm, 433 * mm, (-58.3 + 38.3) * mm / 2.));
  fr1->AddNode(ppf, 5, new TGeoTranslation(335 * mm, 433 * mm, (-58.3 + 38.3) * mm / 2.));
  fr1->AddNode(ppf, 2, new TGeoTranslation(-335 * mm, 0., (-58.3 + 38.3) * mm / 2.));
  fr1->AddNode(ppf, 3, new TGeoTranslation(335 * mm, 0., (-58.3 + 38.3) * mm / 2.));
  fr1->AddNode(ppf, 0, new TGeoTranslation(-335 * mm, -433 * mm, (-58.3 + 38.3) * mm / 2.));
  fr1->AddNode(ppf, 1, new TGeoTranslation(335 * mm, -433 * mm, (-58.3 + 38.3) * mm / 2.));

  Double_t offsetx = 16. * mm, offsety = 34. * mm / 2., interdistx = 48 * mm + offsetx + 0.6666 * mm,
           interdisty = 19. * mm + 2. * offsety;

  // gassiplex implementation
  // it is in 3 different volumes: Hrec (in Hcufoil)+Hext

  TGeoVolume* gassipl2 = gGeoManager->MakeBox("Hgassipl2", csi, 32. * mm / 2, 3. * mm / 2., 1. * mm / 2.); // in Hrect
  TGeoVolume* gassipl3 =
    gGeoManager->MakeBox("Hgassipl3", csi, 60. * mm / 2, 3. * mm / 2., 19. * mm / 2.); // in Hfr1upcard
  TGeoVolume* gassipl4 = gGeoManager->MakeBox(
    "Hgassipl4", csi, 60. * mm / 2, 3. * mm / 2.,
    91. * mm / 2.); // in Hext (the big rectangle of the card is 110 mm long, 62 mm wide and 1.5 mm high)
  TGeoVolume* busext = gGeoManager->MakeTubs("Hbusext", csi, 29 * mm, 30 * mm, 40 * mm / 2., 0., 180); // in Hext
  TGeoVolume* ext = new TGeoVolumeAssembly("Hext");

  rect->AddNode(gassipl2, 1, new TGeoTranslation(0., 0., 0));

  for (Int_t hor = 0; hor < 10; hor++) {
    for (Int_t vert = 0; vert < 8; vert++) {
      cufoil->AddNode(rect, hor + vert * 10,
                      new TGeoTranslation(offsetx + 48. * mm / 2 + hor * interdistx - 662. * mm / 2,
                                          offsety + 19. * mm / 2 + vert * interdisty - 425. * mm / 2., 0.));
      fr1upcard->AddNode(gassipl3, hor + vert * 10,
                         new TGeoTranslation(offsetx + 48. * mm / 2 + hor * interdistx - 662. * mm / 2,
                                             offsety + 19. * mm / 2 + vert * interdisty - 425. * mm / 2., 0.));
      ext->AddNode(gassipl4, hor + vert * 10,
                   new TGeoTranslation(offsetx + 48. * mm / 2 + hor * interdistx - 662. * mm / 2,
                                       offsety + 19. * mm / 2 + vert * interdisty - 425. * mm / 2., 0));
      ext->AddNode(busext, hor + vert * 10,
                   new TGeoTranslation(offsetx + 48. * mm / 2 + hor * interdistx - 662. * mm / 2,
                                       offsety + 19. * mm / 2 + vert * interdisty - 425. * mm / 2 + 3. * mm / 2., 0));
    }
  }

  fr1up->AddNode(cufoil, 4, new TGeoTranslation(-335 * mm, 433 * mm, -20.0 * mm / 2 + 1. * mm / 2));
  fr1up->AddNode(cufoil, 5, new TGeoTranslation(335 * mm, 433 * mm, -20.0 * mm / 2 + 1. * mm / 2));
  fr1up->AddNode(cufoil, 2, new TGeoTranslation(-335 * mm, 0, -20.0 * mm / 2 + 1. * mm / 2));
  fr1up->AddNode(cufoil, 3, new TGeoTranslation(335 * mm, 0, -20.0 * mm / 2 + 1. * mm / 2));
  fr1up->AddNode(cufoil, 0, new TGeoTranslation(-335 * mm, -433 * mm, -20.0 * mm / 2 + 1. * mm / 2));
  fr1up->AddNode(cufoil, 1, new TGeoTranslation(335 * mm, -433 * mm, -20.0 * mm / 2 + 1. * mm / 2));

  fr1up->AddNode(fr1upcard, 4, new TGeoTranslation(-335 * mm, 433 * mm, 1. * mm / 2.));
  fr1up->AddNode(fr1upcard, 5, new TGeoTranslation(335 * mm, 433 * mm, 1. * mm / 2.));
  fr1up->AddNode(fr1upcard, 2, new TGeoTranslation(-335 * mm, 0, 1. * mm / 2.));
  fr1up->AddNode(fr1upcard, 3, new TGeoTranslation(335 * mm, 0, 1. * mm / 2.));
  fr1up->AddNode(fr1upcard, 0, new TGeoTranslation(-335 * mm, -433 * mm, 1. * mm / 2));
  fr1up->AddNode(fr1upcard, 1, new TGeoTranslation(335 * mm, -433 * mm, 1. * mm / 2.));

  hmp->AddNode(ext, 4, new TGeoTranslation(-335 * mm, +433 * mm, (80. + 1.7) * mm + 58.3 * mm + 91 * mm / 2.));
  hmp->AddNode(ext, 5, new TGeoTranslation(+335 * mm, +433 * mm, (80. + 1.7) * mm + 58.3 * mm + 91 * mm / 2.));
  hmp->AddNode(ext, 2, new TGeoTranslation(-335 * mm, 0 * mm, (80. + 1.7) * mm + 58.3 * mm + 91 * mm / 2.));
  hmp->AddNode(ext, 3, new TGeoTranslation(+335 * mm, 0 * mm, (80. + 1.7) * mm + 58.3 * mm + 91 * mm / 2.));
  hmp->AddNode(ext, 0, new TGeoTranslation(-335 * mm, -433 * mm, (80. + 1.7) * mm + 58.3 * mm + 91 * mm / 2.));
  hmp->AddNode(ext, 1, new TGeoTranslation(+335 * mm, -433 * mm, (80. + 1.7) * mm + 58.3 * mm + 91 * mm / 2.));

  hmp->AddNode(proxgap1, 0, new TGeoTranslation(0., 0., (9. - 7.5) * mm / 2.)); // due to the TGeoVolumeAssembly
                                                                                // definition the ch4 volume must be
                                                                                // inserted around the collecting wires
  hmp->AddNode(proxgap2, 0,
               new TGeoTranslation(0., 0., (9 + 7.5 + 34) * mm +
                                             (81.7 - 6.2 - 34. - 9. - 7.5) * mm /
                                               2.)); // tot height(81.7) - Hsec - proxygap2 - top edge fr4 at(9+7.5) mm

  // ^ Y  single cell                                                5.5mm CH4 = 1*mm CsI + 4.45*mm CsI x cath +0.05*mm
  // safety margin
  // |      ______________________________
  // |     |                              |          ^                            ||
  //       |                              | 1.05mm                                ||
  // 2.2*mm| xxxxxxxxxxxxxxxxxxxxxxxxxxxx |--              50um  x                || cat shift  x=0mm , y= 3.15mm ,
  // z=-2.70mm
  //       |                              |                                       ||
  //       |                              |                                       ||
  // __    |  ..........................  | 2.1mm                    20un .       ||  ano shift x=0mm , y= 2.00mm ,
  // z=-0.29mm
  //       |                              |                                       ||
  //       |                              |                                       ||
  //       | xxxxxxxxxxxxxxxxxxxxxxxxxxxx |--                    x                ||  cat shift x=0mm , y= 1.05mm ,
  //       z=-2.70mm
  //       |                              |                                       ||
  //       |                              |         8.4mm                         ||
  // 4*mm  |                              | 2.1mm                                 ||  pad shift x=0mm , y= 0.00mm ,
  // z=2.25*mm
  //       |                              |                                       ||
  //       |                              |                                       ||
  //       | xxxxxxxxxxxxxxxxxxxxxxxxxxxx |--                    x                ||  cat shift x=0mm , y=-1.05mm ,
  //       z=-2.70mm
  //       |                              |                                       ||
  //       |                              |                                       ||
  // __    |  ..........................  | 2.1mm                         . 2.04mm||  ano shift x=0mm , y=-2.00mm ,
  // z=-0.29mm
  //       |                              |                                       ||
  //       |                              |                                       ||
  //       | xxxxxxxxxxxxxxxxxxxxxxxxxxxx |--                    x    4.45mm      ||  cat shift x=0mm , y=-3.15mm ,
  //       z=-2.70mm
  // 2.2*mm|                              |                                       ||
  //       |                              | 1.05mm                                ||
  //       |______________________________|          v                            ||
  //       <             8 mm             >
  //                                   ----->X                                 ----->Z

  ppf->AddNode(lar, 0, new TGeoTranslation(-224.5 * mm, -151.875 * mm, 0. * mm));
  ppf->AddNode(lar, 1, new TGeoTranslation(-224.5 * mm, -50.625 * mm, 0. * mm));
  ppf->AddNode(lar, 2, new TGeoTranslation(-224.5 * mm, +50.625 * mm, 0. * mm));
  ppf->AddNode(lar, 3, new TGeoTranslation(-224.5 * mm, +151.875 * mm, 0. * mm));
  ppf->AddNode(lar, 4, new TGeoTranslation(+224.5 * mm, -151.875 * mm, 0. * mm));
  ppf->AddNode(lar, 5, new TGeoTranslation(+224.5 * mm, -50.625 * mm, 0. * mm));
  ppf->AddNode(lar, 6, new TGeoTranslation(+224.5 * mm, +50.625 * mm, 0. * mm));
  ppf->AddNode(lar, 7, new TGeoTranslation(+224.5 * mm, +151.875 * mm, 0. * mm));
  ppf->AddNode(smo, 0, new TGeoTranslation(-65.0 * mm, -151.875 * mm, 0. * mm));
  ppf->AddNode(smo, 1, new TGeoTranslation(-65.0 * mm, -50.625 * mm, 0. * mm));
  ppf->AddNode(smo, 2, new TGeoTranslation(-65.0 * mm, +50.625 * mm, 0. * mm));
  ppf->AddNode(smo, 3, new TGeoTranslation(-65.0 * mm, +151.875 * mm, 0. * mm));
  ppf->AddNode(smo, 4, new TGeoTranslation(+65.0 * mm, -151.875 * mm, 0. * mm));
  ppf->AddNode(smo, 5, new TGeoTranslation(+65.0 * mm, -50.625 * mm, 0. * mm));
  ppf->AddNode(smo, 6, new TGeoTranslation(+65.0 * mm, +50.625 * mm, 0. * mm));
  ppf->AddNode(smo, 7, new TGeoTranslation(+65.0 * mm, +151.875 * mm, 0. * mm));

  // hmp->AddNode(fr3,1,new TGeoTranslation(0.,0.,(81.7-29.)*mm-34.*mm/2));
  hmp->AddNode(fr3, 1, new TGeoTranslation(0., 0., (9. + 7.5) * mm + 34. * mm / 2));
  fr3->AddNode(fr3up, 1, new TGeoTranslation(0., 0., 7 * mm));
  fr3->AddNode(fr3down, 1, new TGeoTranslation(0., 0., -10 * mm));

  return hmp;
}

//*************************************************************************************************************

TGeoVolume* Detector::CreateCradle()
{

  // Method that builds the Cradle geometry
  // according to the base topology created
  // in CradleBaseVolume(...)

  Double_t mm = 0.1;

  Double_t params[10] = { 0.5, 10., 24., -1, 5.2, 1.5, 3.5, 8.5, 3.8, 0. };
  TGeoMedium* med = gGeoManager->GetMedium("HMP_Al");
  TGeoVolume* cradle = new TGeoVolumeAssembly("Hcradle");

  // Double_t baselong[7]={6037*mm-2*60*mm, 6037*mm-2*60*mm,60*mm,0.,100*mm,10*mm,10*mm};//2CRE2112P3
  Double_t baselong[7] = {
    6037 * mm - 2 * 100 * mm, 6037 * mm - 2 * 100 * mm, 60 * mm, 0., 100 * mm, 10 * mm, 10 * mm
  }; // 2CRE2112P3
  TGeoVolume* lbase = CradleBaseVolume(med, baselong, "cradleLbase");
  lbase->SetLineColor(kGray);

  Double_t baseshort[7] = {
    1288. * mm + 2 * 100 * mm, 1288. * mm + 2 * 100 * mm, 60 * mm, 0., 100 * mm, 10 * mm, 10 * mm
  }; // 2CRE2112P3
  TGeoVolume* sbase = CradleBaseVolume(med, baseshort, "cradleSbase");
  sbase->SetLineColor(kGray);

  // one side

  Double_t height = 30. * mm; // 30 = 2*(1488/2-729) (2CRE2112P3)
  Double_t tubeheight = 50. * mm;
  Double_t heightred = 5. * mm;
  Double_t zred = 5. * mm;
  Double_t oneshift = tubeheight / TMath::Tan(TMath::DegToRad() * 20.) + (1458. - 35) * mm / 2 - (1607 - 35) * mm / 2;
  Double_t linclined[7] = {
    1458. * mm - params[6] - 0.5, 1607. * mm - params[6] - 0.5, tubeheight, oneshift, height, heightred, zred
  }; // 3.5 is for not correct measurements in 2CRE2112P3<=> 597!=inclined*sin(20)
  TGeoVolume* inclin = CradleBaseVolume(med, linclined, "inclinedbar");
  inclin->SetLineColor(kGray);
  Double_t lhorizontal[7] = {
    1641.36 * mm + params[7], 1659. * mm + params[7], tubeheight, 0, height, heightred, zred
  };
  TGeoVolume* horiz = CradleBaseVolume(med, lhorizontal, "horizontalbar");
  horiz->SetLineColor(kGray);

  // inner bars, they are named as the numbering in 2CRE2112P3
  Double_t fourshift = tubeheight / TMath::Tan(TMath::DegToRad() * 55.);
  Double_t lfour[7] = { 592 * mm, 592 * mm, tubeheight, fourshift, height, heightred, zred };
  TGeoVolume* four = CradleBaseVolume(med, lfour, "bar4");
  four->SetLineColor(kGray);

  Double_t fiveshift = tubeheight / TMath::Tan(TMath::DegToRad() * 75);
  Double_t lfive[7] = { 500. * mm, 500. * mm, tubeheight, fiveshift, height, heightred, zred };
  TGeoVolume* five = CradleBaseVolume(med, lfive, "bar5");
  five->SetLineColor(kGray);

  Double_t sixshift = tubeheight / TMath::Tan(TMath::DegToRad() * 55) + 459 * mm / 2 - 480 * mm / 2;
  Double_t lsix[7] = { 456 * mm, 477 * mm, tubeheight, sixshift, height, heightred, zred };
  TGeoVolume* six = CradleBaseVolume(med, lsix, "bar6");
  six->SetLineColor(kGray);

  Double_t sevenshift = tubeheight / TMath::Tan(TMath::DegToRad() * 50) + 472 * mm / 2 - 429. * mm / 2;
  Double_t lseven[7] = { 429 * mm, 472 * mm, tubeheight, sevenshift, height, heightred, zred };
  TGeoVolume* seven = CradleBaseVolume(med, lseven, "bar7");
  seven->SetLineColor(kGray);

  Double_t eightshift = tubeheight / TMath::Tan(TMath::DegToRad() * 30) + 244. * mm / 2 - 200. * mm / 2 - 3;
  Double_t leight[7] = { 200. * mm, 244. * mm, tubeheight, eightshift, height, heightred, zred };
  TGeoVolume* eight = CradleBaseVolume(med, leight, "bar8");
  eight->SetLineColor(kGray);

  Double_t nineshift = -tubeheight / TMath::Tan(TMath::DegToRad() * 71) + 83. * mm / 2 - 66. * mm / 2;
  Double_t lnine[7] = { 59.5 * mm, 76.5 * mm, tubeheight, nineshift, height, heightred, zred };
  TGeoVolume* nine = CradleBaseVolume(med, lnine, "bar9");
  nine->SetLineColor(kGray);

  Double_t tenshift = (-tubeheight / TMath::Tan(TMath::DegToRad() * 60) - 221. * mm / 2 + 195. * mm / 2);
  Double_t lten[7] = { 195. * mm, 221. * mm, tubeheight, tenshift, height, heightred, zred };
  TGeoVolume* ten = CradleBaseVolume(med, lten, "bar10");
  ten->SetLineColor(kGray);

  Double_t elevenshift = (-tubeheight / TMath::Tan(TMath::DegToRad() * 70) - 338. * mm / 2 + 315. * mm / 2);
  Double_t leleven[7] = { 308. * mm, 331. * mm, tubeheight, elevenshift, height, heightred, zred };
  TGeoVolume* eleven = CradleBaseVolume(med, leleven, "bar11");
  eleven->SetLineColor(kGray);

  Double_t twelveshift = (-tubeheight / TMath::Tan(TMath::DegToRad() * 60) - 538. * mm / 2 + 508. * mm / 2);
  Double_t ltwelve[7] = { 507. * mm, 537. * mm, tubeheight, twelveshift, height, heightred, zred };
  TGeoVolume* twelve = CradleBaseVolume(med, ltwelve, "bar12");
  twelve->SetLineColor(kGray);

  Double_t thirteenshift = tubeheight / TMath::Tan(TMath::DegToRad() * 43);
  Double_t lthirteen[7] = { 708. * mm, 708. * mm, tubeheight, thirteenshift, height, heightred, zred };
  TGeoVolume* thirteen = CradleBaseVolume(med, lthirteen, "bar13");
  thirteen->SetLineColor(kGray);

  // vertical rectangles
  TGeoVolume* vbox = new TGeoVolumeAssembly("Hvbox");
  vbox->SetLineColor(kViolet);
  Double_t width = 50. * mm;

  TGeoVolume* vboxlast = new TGeoVolumeAssembly("Hvboxlast"); // vertical structure on the short base
  vboxlast->SetLineColor(kViolet);

  Double_t barheight = 100. * mm;
  Double_t lAfourteen[7] = { 1488. * mm, 1488. * mm, barheight, 0, width, heightred, zred };
  TGeoVolume* afourteen = CradleBaseVolume(med, lAfourteen, "bar14top");
  afourteen->SetLineColor(kGray);

  Double_t lBfourteen[7] = { 387 * mm, 387. * mm, barheight, 0, width, heightred, zred };
  TGeoVolume* bfourteen = CradleBaseVolume(med, lBfourteen, "bar14vert");
  bfourteen->SetLineColor(kGray);

  Double_t lCfourteen[7] = { 1288. * mm, 1288. * mm, barheight, 0, width, heightred, zred };
  TGeoVolume* cfourteen = CradleBaseVolume(med, lCfourteen, "bar14bot");
  cfourteen->SetLineColor(kGray);

  Double_t oblshift = 50. * mm / TMath::Tan(TMath::DegToRad() * 35);
  Double_t lDfourteen[7] = { 603. * mm, 603. * mm, 50. * mm, oblshift, width, heightred, zred };
  TGeoVolume* dfourteen = CradleBaseVolume(med, lDfourteen, "bar14incl");
  dfourteen->SetLineColor(kGray);

  Double_t lDfourteenlast[7] = { 667. * mm, 667. * mm, 50. * mm, oblshift, width, heightred, zred };
  TGeoVolume* dfourteenlast = CradleBaseVolume(med, lDfourteenlast, "bar14incllast");
  dfourteenlast->SetLineColor(kGray);

  vbox->AddNode(afourteen, 1, new TGeoTranslation(0., 487. * mm / 2 - 100. * mm / 2, 0.));
  TGeoRotation* vinrot = new TGeoRotation("vertbar");
  vinrot->RotateZ(90);
  vbox->AddNode(bfourteen, 1, new TGeoCombiTrans(1488 * mm / 2 - 100. * mm / 2, -100. * mm / 2, 0., vinrot));
  vbox->AddNode(bfourteen, 2, new TGeoCombiTrans(-1488 * mm / 2 + 100. * mm / 2, -100. * mm / 2, 0., vinrot));
  TGeoRotation* rotboxbar = new TGeoRotation("rotboxbar");
  rotboxbar->RotateZ(-35);
  TGeoRotation* arotboxbar = new TGeoRotation("arotboxbar");
  arotboxbar->RotateZ(-35);
  arotboxbar->RotateY(180);
  vbox->AddNode(dfourteen, 1, new TGeoCombiTrans(-1488 * mm / 4, -1, 0.4, rotboxbar));
  vbox->AddNode(dfourteen, 2, new TGeoCombiTrans(+1488 * mm / 4, -1, 0.4, arotboxbar));
  // vertical box on the short base of the cradle
  vboxlast->AddNode(afourteen, 1, new TGeoTranslation(0., 487. * mm / 2 - 100. * mm / 2, 0.));
  vboxlast->AddNode(bfourteen, 1, new TGeoCombiTrans(1488 * mm / 2 - 100. * mm / 2, -100. * mm / 2, 0., vinrot));
  vboxlast->AddNode(bfourteen, 2, new TGeoCombiTrans(-1488 * mm / 2 + 100. * mm / 2, -100. * mm / 2, 0., vinrot));
  vboxlast->AddNode(dfourteenlast, 1, new TGeoCombiTrans(-1488 * mm / 4 + 1.7, -3., 0., rotboxbar));
  vboxlast->AddNode(dfourteenlast, 2, new TGeoCombiTrans(+1488 * mm / 4 - 1.7, -3., 0., arotboxbar));

  // POSITIONING IN THE VIRTUAL VOLUME "cradle"

  // long base
  TGeoRotation* rotl = new TGeoRotation("Clongbase");
  rotl->RotateX(90);
  cradle->AddNode(lbase, 0, new TGeoCombiTrans(0 * mm, (1488 - 100) * mm / 2, -(597 - 60) * mm / 2, rotl));
  cradle->AddNode(lbase, 1, new TGeoCombiTrans(0 * mm, -(1488 - 100) * mm / 2, -(597 - 60) * mm / 2, rotl));
  // short base
  TGeoRotation* rots = new TGeoRotation("Cshortbase");
  rots->RotateX(90);
  rots->RotateZ(90);
  cradle->AddNode(sbase, 1, new TGeoCombiTrans((6037 - 100) * mm / 2, 0., -(597 - 60) * mm / 2, rots));
  cradle->AddNode(sbase, 2, new TGeoCombiTrans(-(6037 - 100) * mm / 2, 0., -(597 - 60) * mm / 2, rots));

  // trapezoidal structure
  Double_t origintrapstruct = (6037 - 2 * 60) * mm / 2 - 2288 * mm;

  TGeoRotation* rot1 = new TGeoRotation("inclrot");
  rot1->RotateX(90);
  rot1->RotateY(200);
  TGeoRotation* rot2 = new TGeoRotation("horizrot");
  rot2->RotateX(-90);
  Double_t dx = (1607 - 35) * mm * TMath::Cos(TMath::DegToRad() * 20) / 2 -
                tubeheight / 2 * TMath::Sin(TMath::DegToRad() * 20) + params[5];

  cradle->AddNode(inclin, 1, new TGeoCombiTrans(origintrapstruct + (2288 + 60) * mm - dx, 729 * mm, params[0] + 0.4,
                                                rot1)); //+0.7 added
  cradle->AddNode(horiz, 1, new TGeoCombiTrans(origintrapstruct, 729 * mm, 597 * mm / 2 - tubeheight / 2,
                                               rot2)); // correctly positioned
  TGeoRotation* rot1mirror = new TGeoRotation("inclmirrot");
  rot1mirror->RotateX(90);
  rot1mirror->RotateY(200);
  rot1mirror->RotateZ(180);
  cradle->AddNode(inclin, 2, new TGeoCombiTrans(origintrapstruct - 2345 * mm + dx, 729 * mm, params[0] + 0.4,
                                                rot1mirror)); //+0.7 added
  cradle->AddNode(inclin, 3, new TGeoCombiTrans(origintrapstruct + (2288 + 60) * mm - dx, -729 * mm, params[0] + 0.4,
                                                rot1)); // 0.7 added
  cradle->AddNode(horiz, 2, new TGeoCombiTrans(origintrapstruct, -729 * mm, 597 * mm / 2 - tubeheight / 2,
                                               rot2)); // correctly positioned
  cradle->AddNode(inclin, 4, new TGeoCombiTrans(origintrapstruct - 2345 * mm + dx, -729 * mm, params[0] + 0.4,
                                                rot1mirror)); // 0.7 added

  // inner pieces on one side
  TGeoRotation* rot4 = new TGeoRotation("4rot");
  rot4->RotateX(-90);
  rot4->RotateY(-55);
  rot4->RotateZ(180);
  TGeoRotation* rot4a = new TGeoRotation("4arot");
  rot4a->RotateX(-90);
  rot4a->RotateY(-55);
  cradle->AddNode(four, 1, new TGeoCombiTrans(origintrapstruct -
                                                (39 + (597 - 50 - 60) / (2 * TMath::Tan(TMath::DegToRad() * 55))) * mm -
                                                tubeheight / (2 * TMath::Sin(TMath::DegToRad() * 55)),
                                              -729 * mm, params[3], rot4));

  cradle->AddNode(four, 2, new TGeoCombiTrans(origintrapstruct +
                                                (39 + (597 - 50 - 60) / (2 * TMath::Tan(TMath::DegToRad() * 55))) * mm +
                                                tubeheight / (2 * TMath::Sin(TMath::DegToRad() * 55)),
                                              -729 * mm, params[3], rot4a));

  TGeoRotation* rot5 = new TGeoRotation("5rot");
  rot5->RotateX(-90);
  rot5->RotateY(-75);
  rot5->RotateZ(180);
  TGeoRotation* rot5a = new TGeoRotation("5arot");
  rot5a->RotateX(-90);
  rot5a->RotateY(-75);
  cradle->AddNode(
    five, 1,
    new TGeoCombiTrans(origintrapstruct + (486 + (597 - 50 - 60) / (2 * TMath::Tan(TMath::DegToRad() * 75))) * mm +
                         tubeheight / (2 * TMath::Sin(TMath::DegToRad() * 75)),
                       -729 * mm, 0, rot5));
  cradle->AddNode(
    five, 2,
    new TGeoCombiTrans(origintrapstruct - (486 + (597 - 50 - 60) / (2 * TMath::Tan(TMath::DegToRad() * 75))) * mm -
                         tubeheight / (2 * TMath::Sin(TMath::DegToRad() * 75)),
                       -729 * mm, 0, rot5a));
  cradle->AddNode(six, 1,
                  new TGeoCombiTrans(origintrapstruct + 808 * mm + (480 * mm / 2) * TMath::Cos(TMath::DegToRad() * 55) +
                                       tubeheight / (2 * TMath::Sin(TMath::DegToRad() * 55)) + 2.,
                                     -729 * mm, -params[4] - 0.5, rot4a));
  cradle->AddNode(six, 2,
                  new TGeoCombiTrans(origintrapstruct - 808 * mm - (480 * mm / 2) * TMath::Cos(TMath::DegToRad() * 55) -
                                       tubeheight / (2 * TMath::Sin(TMath::DegToRad() * 55)) - 2.,
                                     -729 * mm, -params[4] - 0.5, rot4));

  TGeoRotation* rot7 = new TGeoRotation("7rot");
  rot7->RotateX(-90);
  rot7->RotateY(130);
  rot7->RotateZ(180);
  TGeoRotation* rot7a = new TGeoRotation("7arot");
  rot7a->RotateX(-90);
  rot7a->RotateY(130);

  cradle->AddNode(
    seven, 1, new TGeoCombiTrans(origintrapstruct + 1478 * mm - (472 * mm / 2) * TMath::Cos(TMath::DegToRad() * 50) +
                                   tubeheight / (2 * TMath::Sin(TMath::DegToRad() * 50)),
                                 -729 * mm, -params[8], rot7));
  cradle->AddNode(
    seven, 2, new TGeoCombiTrans(origintrapstruct - 1478 * mm + (472 * mm / 2) * TMath::Cos(TMath::DegToRad() * 50) -
                                   tubeheight / (2 * TMath::Sin(TMath::DegToRad() * 50)),
                                 -729 * mm, -params[8], rot7a));
  TGeoRotation* rot8 = new TGeoRotation("8rot");
  rot8->RotateX(-90);
  rot8->RotateY(-25);
  TGeoRotation* rot8a = new TGeoRotation("8arot");
  rot8a->RotateX(-90);
  rot8a->RotateY(-25);
  rot8a->RotateZ(180);
  cradle->AddNode(
    eight, 1, new TGeoCombiTrans(origintrapstruct + 1640 * mm + (244 * mm / 2) * TMath::Cos(TMath::DegToRad() * 30) +
                                   tubeheight / (2 * TMath::Sin(TMath::DegToRad() * 30)),
                                 -729 * mm, -20.5, rot8));
  cradle->AddNode(
    eight, 2, new TGeoCombiTrans(origintrapstruct - 1640 * mm - (244 * mm / 2) * TMath::Cos(TMath::DegToRad() * 30) -
                                   tubeheight / (2 * TMath::Sin(TMath::DegToRad() * 30)),
                                 -729 * mm, -20.5, rot8a));
  TGeoRotation* rot9 = new TGeoRotation("9rot");
  rot9->RotateX(-90);
  rot9->RotateY(-90);
  TGeoRotation* rot9a = new TGeoRotation("9arot");
  rot9a->RotateX(-90);
  rot9a->RotateY(-90);
  rot9a->RotateZ(180);
  cradle->AddNode(nine, 1, new TGeoCombiTrans(origintrapstruct + 1960 * mm + 2.5 + 3., -729. * mm, -20., rot9));
  cradle->AddNode(nine, 2, new TGeoCombiTrans(origintrapstruct - 1960 * mm - 2.5 - 3., -729. * mm, -20., rot9a));
  // inner pieces on the other side
  TGeoRotation* rot10 = new TGeoRotation("10rot");
  rot10->RotateX(-90);
  rot10->RotateY(-120);
  TGeoRotation* rot10a = new TGeoRotation("10arot");
  rot10a->RotateX(-90);
  rot10a->RotateY(-120);
  rot10a->RotateZ(180);

  cradle->AddNode(
    ten, 1, new TGeoCombiTrans(origintrapstruct + 1738 * mm + tubeheight / (2 * TMath::Sin(TMath::DegToRad() * 60)) - 2,
                               +729. * mm, -13., rot10));
  cradle->AddNode(
    ten, 2, new TGeoCombiTrans(origintrapstruct - 1738 * mm - tubeheight / (2 * TMath::Sin(TMath::DegToRad() * 60)) + 2,
                               +729. * mm, -13., rot10a));

  TGeoRotation* rot11 = new TGeoRotation("11rot");
  rot11->RotateX(-90);
  rot11->RotateY(50);
  TGeoRotation* rot11a = new TGeoRotation("11arot");
  rot11a->RotateX(-90);
  rot11a->RotateY(50);
  rot11a->RotateZ(180);
  cradle->AddNode(eleven, 1, new TGeoCombiTrans(origintrapstruct - 1738 * mm -
                                                  tubeheight / (2 * TMath::Sin(TMath::DegToRad() * 60)) + 352. * mm,
                                                +729. * mm, -12.7, rot11));
  cradle->AddNode(eleven, 2, new TGeoCombiTrans(origintrapstruct + 1738 * mm +
                                                  tubeheight / (2 * TMath::Sin(TMath::DegToRad() * 60)) - 352. * mm,
                                                +729. * mm, -12.7, rot11a));

  TGeoRotation* rot12 = new TGeoRotation("12rot");
  rot12->RotateX(-90);
  rot12->RotateY(-120);
  TGeoRotation* rot12a = new TGeoRotation("12arot");
  rot12a->RotateX(-90);
  rot12a->RotateY(-120);
  rot12a->RotateZ(180);
  cradle->AddNode(twelve, 1, new TGeoCombiTrans(origintrapstruct + 1065 * mm, +729. * mm, 1., rot12));
  cradle->AddNode(twelve, 2, new TGeoCombiTrans(origintrapstruct - 1065 * mm, +729. * mm, 1., rot12a));

  TGeoRotation* rot13 = new TGeoRotation("13rot");
  rot13->RotateX(-90);
  rot13->RotateY(-43);
  rot13->RotateZ(180);
  TGeoRotation* rot13a = new TGeoRotation("13arot");
  rot13a->RotateX(-90);
  rot13a->RotateY(-43);
  cradle->AddNode(thirteen, 1, new TGeoCombiTrans(origintrapstruct + 572 * mm - 18., +729. * mm, -1.5, rot13));
  cradle->AddNode(thirteen, 2, new TGeoCombiTrans(origintrapstruct - 572 * mm + 18., +729. * mm, -1.5, rot13a));

  // vertical structures
  TGeoRotation* vrot = new TGeoRotation("vertbox");
  vrot->RotateX(90);
  vrot->RotateZ(90);
  cradle->AddNode(vboxlast, 1, new TGeoCombiTrans(-6037 * mm / 2 + 50. * mm / 2, 0., 0.5,
                                                  vrot)); // vertial box on the short cradle base

  cradle->AddNode(vbox, 2, new TGeoCombiTrans(-6037 * mm / 2 + 50. * mm / 2 + 990. * mm, 0., 0.5, vrot));
  cradle->AddNode(cfourteen, 2, new TGeoCombiTrans(-6037 * mm / 2 + 50. * mm / 2 + 990. * mm, 0.,
                                                   -477. * mm / 2 - 20. * mm / 2, vrot));

  cradle->AddNode(
    vbox, 3, new TGeoCombiTrans(origintrapstruct - (1641.36 * mm + params[7]) / 2. + 50. * mm / 2. + 3, 0., 0.5, vrot));
  cradle->AddNode(cfourteen, 3,
                  new TGeoCombiTrans(origintrapstruct - (1641.36 * mm + params[7]) / 2. + 50. * mm / 2. + 3, 0.,
                                     -477. * mm / 2 - 20. * mm / 2, vrot));

  cradle->AddNode(
    vbox, 4, new TGeoCombiTrans(origintrapstruct + (1641.36 * mm + params[7]) / 2. - 50. * mm / 2. - 3, 0., 0.5, vrot));
  cradle->AddNode(cfourteen, 4,
                  new TGeoCombiTrans(origintrapstruct + (1641.36 * mm + params[7]) / 2. - 50. * mm / 2. - 3, 0.,
                                     -477. * mm / 2 - 20. * mm / 2, vrot));

  return cradle;
} // CreateCradle()

//*****************************************************************************************************************

TGeoVolume* Detector::CradleBaseVolume(TGeoMedium* med, Double_t l[7], const char* name)
{
  /*
  The trapezoid is build in the xy plane

      0  ________________ 1
        /       |        \
       /        |         \
      /       (0,0)        \
     /          |           \
  3 /___________|____________\ 2

   01 is right shifted => shift is positive

    //1: small base (0-1); 2: long base (3-2);
    //3: trapezoid height; 4: shift between the two bases;
    //5: height 6: height reduction; 7: z-reduction;
  */

  TGeoXtru* xtruIn = new TGeoXtru(2);
  TGeoXtru* xtruOut = new TGeoXtru(2);
  xtruIn->SetName(Form("%sIN", name));
  xtruOut->SetName(Form("%sOUT", name));

  Double_t xv[4], yv[4];

  xv[0] = -(l[0] / 2 - l[3]);
  yv[0] = l[2] / 2;
  xv[1] = l[0] / 2 + l[3];
  yv[1] = l[2] / 2;
  xv[2] = l[1] / 2;
  yv[2] = -l[2] / 2;
  xv[3] = -l[1] / 2;
  yv[3] = -l[2] / 2;

  xtruOut->DefinePolygon(4, xv, yv);
  xtruOut->DefineSection(0, -l[4] / 2., 0., 0.,
                         1.0); // 0=  I plane z; (0.,0.) = shift wrt centre; 1.= shape scale factor
  xtruOut->DefineSection(1, +l[4] / 2., 0., 0., 1.0); // 1= II plane z;

  Double_t tgalpha = 0;
  if (xv[3] - xv[0] == 0)
    tgalpha = 999999;
  else
    tgalpha = l[2] / TMath::Abs(xv[3] - xv[0]);
  Double_t tgbeta = 0;
  if (xv[2] - xv[1] == 0)
    tgbeta = 999999;
  else
    tgbeta = l[2] / TMath::Abs(xv[2] - xv[1]);

  xv[0] = xv[0] - l[5] / tgalpha;
  yv[0] = l[2] / 2 - l[5];
  xv[1] = xv[1] + l[5] / tgbeta;
  yv[1] = l[2] / 2 - l[5];
  xv[2] = xv[2] + l[5] / tgbeta;
  yv[2] = -l[2] / 2 + l[5];
  xv[3] = xv[3] - l[5] / tgalpha;
  yv[3] = -l[2] / 2 + l[5];

  xtruIn->DefinePolygon(4, xv, yv);
  xtruIn->DefineSection(0, (-l[4] + l[6]) / 2, 0., 0., 1.0);
  xtruIn->DefineSection(1, (+l[4] - l[6]) / 2, 0., 0., 1.0);

  TGeoCompositeShape* shape = new TGeoCompositeShape(name, Form("%sOUT-%sIN", name, name));

  TGeoVolume* vol = new TGeoVolume(name, shape, med);

  return vol;
} // CradleBaseVolume()
//****************************************************************************************************************

void Detector::ConstructGeometry()
{
  // Creates detailed geometry simulation (currently GEANT volumes tree)
  // includind the HMPID cradle

  createMaterials();

  TGeoVolume* hmpcradle = CreateCradle();

  for (Int_t iCh = 0; iCh <= 6; iCh++) { // place 7 chambers
    TGeoVolume* hmpid = createChamber(iCh);
    TGeoHMatrix* pMatrix = new TGeoHMatrix;
    IdealPosition(iCh, pMatrix);
    gGeoManager->GetVolume("cave")->AddNode(hmpid, 0, pMatrix);
    if (iCh == 1 || iCh == 3 || iCh == 5) {
      TGeoHMatrix* pCradleMatrix = new TGeoHMatrix;
      IdealPositionCradle(iCh, pCradleMatrix);
      gGeoManager->GetVolume("cave")->AddNode(hmpcradle, iCh, pCradleMatrix);
    }
  }
  // }
  // AliDebug(1,"Stop v3. HMPID option");
}

//*****************************************************************************************************************
void Detector::DefineOpticalProperties()
{
  // AliDebug(1,"");

  // Optical properties definition.
  const Int_t kNbins = 30;        // number of photon energy points
  Float_t emin = 5.5, emax = 8.5; // Photon energy range,[eV]
  Float_t deltaE = (emax - emin) / kNbins;
  Float_t aEckov[kNbins];
  Double_t dEckov[kNbins];
  Float_t aAbsRad[kNbins], aAbsWin[kNbins], aAbsGap[kNbins], aAbsMet[kNbins];
  Float_t aIdxRad[kNbins], aIdxWin[kNbins], aIdxGap[kNbins], aIdxMet[kNbins], aIdxPc[kNbins];
  Float_t aQeAll[kNbins], aQePc[kNbins];
  Double_t dReflMet[kNbins], dQePc[kNbins];

  TF2* pRaIF = new TF2("HidxRad", "sqrt(1+0.554*(1239.84/x)^2/((1239.84/x)^2-5769)-0.0005*(y-20))", emin, emax, 0,
                       50); // DiMauro mail temp 0-50 degrees C
  TF1* pWiIF = new TF1("HidxWin", "sqrt(1+46.411/(10.666*10.666-x*x)+228.71/(18.125*18.125-x*x))", emin,
                       emax);                                                                  // SiO2 idx TDR p.35
  TF1* pGaIF = new TF1("HidxGap", "1+0.12489e-6/(2.62e-4 - x*x/1239.84/1239.84)", emin, emax); //?????? from where

  TF1* pRaAF =
    new TF1("HabsRad", "(x<7.8)*(gaus+gaus(3))+(x>=7.8)*0.0001", emin, emax); // fit from DiMauro data 28.10.03
  pRaAF->SetParameters(3.20491e16, -0.00917890, 0.742402, 3035.37, 4.81171, 0.626309);
  TF1* pWiAF = new TF1("HabsWin", "(x<8.2)*(818.8638-301.0436*x+36.89642*x*x-1.507555*x*x*x)+(x>=8.2)*0.0001", emin,
                       emax); // fit from DiMauro data 28.10.03
  TF1* pGaAF = new TF1(
    "HabsGap", "(x<7.75)*6512.399+(x>=7.75)*3.90743e-2/(-1.655279e-1+6.307392e-2*x-8.011441e-3*x*x+3.392126e-4*x*x*x)",
    emin, emax); //????? from where

  TF1* pQeF = new TF1("Hqe", "0+(x>6.07267)*0.344811*(1-exp(-1.29730*(x-6.07267)))", emin,
                      emax); // fit from DiMauro data 28.10.03

  TString title = GetTitle();
  Bool_t isFlatIdx = title.Contains("FlatIdx");

  for (Int_t i = 0; i < kNbins; i++) {
    Float_t eV = emin + deltaE * i; // Ckov energy in eV
    aEckov[i] = 1e-9 * eV;          // Ckov energy in GeV
    dEckov[i] = aEckov[i];
    aAbsRad[i] = pRaAF->Eval(eV);
    (isFlatIdx) ? aIdxRad[i] = 1.292 : aIdxRad[i] = pRaIF->Eval(eV, 20);
    aAbsWin[i] = pWiAF->Eval(eV);
    aIdxWin[i] = pWiIF->Eval(eV);
    aAbsGap[i] = pGaAF->Eval(eV);
    aIdxGap[i] = pGaIF->Eval(eV);
    aQeAll[i] = 1; // QE for all other materials except for PC must be 1.
    aAbsMet[i] = 0.0001;
    aIdxMet[i] = 0; // metal ref idx must be 0 in order to reflect photon
    aIdxPc[i] = 1;
    aQePc[i] = pQeF->Eval(eV); // PC ref idx must be 1 in order to apply photon to QE conversion
    dQePc[i] = pQeF->Eval(eV);
    dReflMet[i] = 0.; // no reflection on the surface of the pc (?)
  }
  TVirtualMC::GetMC()->SetCerenkov(getMediumID(kC6F14), kNbins, aEckov, aAbsRad, aQeAll, aIdxRad);
  TVirtualMC::GetMC()->SetCerenkov(getMediumID(kSiO2), kNbins, aEckov, aAbsWin, aQeAll, aIdxWin);
  TVirtualMC::GetMC()->SetCerenkov(getMediumID(kCH4), kNbins, aEckov, aAbsGap, aQeAll, aIdxGap);
  TVirtualMC::GetMC()->SetCerenkov(getMediumID(kCu), kNbins, aEckov, aAbsMet, aQeAll, aIdxMet);
  TVirtualMC::GetMC()->SetCerenkov(getMediumID(kW), kNbins, aEckov, aAbsMet, aQeAll,
                                   aIdxMet); // n=0 means reflect photons
  TVirtualMC::GetMC()->SetCerenkov(getMediumID(kCsI), kNbins, aEckov, aAbsMet, aQePc,
                                   aIdxPc); // n=1 means convert photons
  TVirtualMC::GetMC()->SetCerenkov(getMediumID(kAl), kNbins, aEckov, aAbsMet, aQeAll, aIdxMet);

  // Define a skin surface for the photocatode to enable 'detection' in G4
  for (Int_t i = 0; i < 7; i++) {
    TVirtualMC::GetMC()->DefineOpSurface(Form("surfPc%i", i), kGlisur /*kUnified*/, kDielectric_metal, kPolished, 0.);
    TVirtualMC::GetMC()->SetMaterialProperty(Form("surfPc%i", i), "EFFICIENCY", kNbins, dEckov, dQePc);
    TVirtualMC::GetMC()->SetMaterialProperty(Form("surfPc%i", i), "REFLECTIVITY", kNbins, dEckov, dReflMet);
    TVirtualMC::GetMC()->SetSkinSurface(Form("skinPc%i", i), Form("Hpad%i", i), Form("surfPc%i", i));
  }

  delete pRaAF;
  delete pWiAF;
  delete pGaAF;
  delete pRaIF;
  delete pWiIF;
  delete pGaIF;
  delete pQeF;
}

} // end namespace hmpid
} // end namespace o2

ClassImp(o2::hmpid::Detector);
