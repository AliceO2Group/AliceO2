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
  Int_t nAir = 4; // mixture 0.9999999
  float aC6F14[2] = { 12.01, 18.99 }, zC6F14[2] = { 6, 9 }, wC6F14[2] = { 6, 14 }, dC6F14 = 1.68;
  Int_t nC6F14 = -2;
  float aSiO2[2] = { 28.09, 15.99 }, zSiO2[2] = { 14, 8 }, wSiO2[2] = { 1, 2 }, dSiO2 = 2.64;
  Int_t nSiO2 = -2;
  float aCH4[2] = { 12.01, 1.01 }, zCH4[2] = { 6, 1 }, wCH4[2] = { 1, 4 }, dCH4 = 7.17e-4;
  Int_t nCH4 = -2;

  float aRoha = 12.01, zRoha = 6, dRoha = 0.10, radRoha = 18.80,
        absRoha = 86.3 / dRoha; // special material- quasi quartz
  float aCu = 63.55, zCu = 29, dCu = 8.96, radCu = 1.43, absCu = 134.9 / dCu;
  float aW = 183.84, zW = 74, dW = 19.30, radW = 0.35, absW = 185.0 / dW;
  float aAl = 26.98, zAl = 13, dAl = 2.70, radAl = 8.90, absAl = 106.4 / dAl;
  float aAr = 39.94, zAr = 18, dAr = 1.396e-3, radAr = 14.0, absAr = 117.2 / dAr;

  Int_t matId = 0;            // tmp material id number
  Int_t unsens = 0, sens = 1; // sensitive or unsensitive medium
  Int_t itgfld;
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
  Int_t nG10 = 4;

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

TGeoVolume* Detector::createChamber(int number)
{
  // Single module geometry building

  Double_t cm = 1, mm = 0.1 * cm, um = 0.001 * mm; // default is cm

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

  Double_t cellx = 8.04 * mm, celly = 8.4 * mm;
  Int_t nPadX = 80, nPadY = 48;
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
                                                                                        // Hsec (6.2)   - proxygap2 (34)
                                                                                        // - upper bound of fr4 (9+7.5))

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
                                               2.)); // tot height(81.7) - Hsec - proxygap2 - top edge fr4 at (9+7.5) mm

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

void Detector::ConstructGeometry()
{
  createMaterials();
  // for the moment create just 1 chamber and connect to the cave volume
  auto vol = createChamber(0);
  gGeoManager->GetVolume("cave")->AddNode(vol, 0);
}

} // end namespace hmpid
} // end namespace o2

ClassImp(o2::hmpid::Detector);
