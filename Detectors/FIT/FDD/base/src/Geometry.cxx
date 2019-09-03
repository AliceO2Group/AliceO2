// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include <iomanip>
#include "FDDBase/Geometry.h"

#include <TGeoManager.h>
#include <TGeoBBox.h>
#include <TGeoTube.h>
#include <TGeoCompositeShape.h>
#include <TGeoMedium.h>
#include <TGeoVolume.h>
#include <TGeoMatrix.h>
#include <FairLogger.h>
#include <sstream>

ClassImp(o2::fdd::Geometry);

using namespace o2::fdd;
//_____________________________________________________________________________
Geometry::Geometry(EGeoType initType)
{
  mGeometryType = initType;
  buildGeometry();
}
//_____________________________________________________________________________
Geometry::Geometry(const Geometry& geom)
{
  this->mGeometryType = geom.mGeometryType;
}
//_____________________________________________________________________________
void Geometry::buildGeometry()
{

  // Top volume of FIT FDD detector
  TGeoVolumeAssembly* vFDD = new TGeoVolumeAssembly("FDD");

  LOG(INFO) << "Geometry::buildGeometry()::Volume name = " << vFDD->GetName();

  //Rotations used
  //TGeoRotation* Rx90m = new TGeoRotation("Rx90m", 0., -90., 0.);
  //TGeoRotation* Rx90 = new TGeoRotation("Rx90", 0., 90., 0.);
  TGeoRotation* Rx180 = new TGeoRotation("Rx180", 0., 180., 0.);   //   4    |   1
  TGeoRotation* Rz180 = new TGeoRotation("Rz180", 180., 0., 0.);   // --------------->  x
  TGeoRotation* Ry180 = new TGeoRotation("Ry180", 180., 180., 0.); //   3    |   2
  //TGeoRotation* Ry90m = new TGeoRotation("Ry90m", 90., -90., -90.);
  //TGeoRotation* Ry90 = new TGeoRotation("Ry90", 90., 90., -90.);
  //TGeoRotation* Rz90 = new TGeoRotation("Rz90", 90., 0., 0.);

  const Double_t kZbegFrontBar = 1949.1; // Begining of Front Bar

  //Medium for FDA
  TGeoMedium* medFDASci = gGeoManager->GetMedium("FDD_BC420");
  //Medium for FDC
  TGeoMedium* medFDCSci = gGeoManager->GetMedium("FDD_BC420");

  // FDA Scintillator Pad
  const Double_t kFDACellSideY = 21.6;
  const Double_t kFDACellSideX = 18.1;
  // FDC Scintillator Pad
  const Double_t kFDCCellSideY = 21.6;
  const Double_t kFDCCellSideX = 18.1;
  // WLS bar          :  0.40 cm ( 4.0 mm )
  // Wrapping         :  0.20 cm ( 2.0 mm )
  // Aluminnized Mylar:  0.01 cm ( 0.1 mm )
  // Fishing line     :  0.04 cm ( 0.4 mm )
  // total shift on X :  0.65 cm
  // total shift on Y :  0.21 cm
  const Double_t kShiftX = 0.54;
  const Double_t kShiftY = 0.10;
  const Double_t kFDACelldz = 2.54;
  const Double_t kFDCCelldz = 2.54;
  const Double_t kFDABeamPipeR = 6.20; // Radius of beam pipe hole for FDD_A (Diameter  12.4 cm)
  const Double_t kFDCBeamPipeR = 3.70; // Radius of beam pipe hole for FDD_C (Diameter   7.4 cm)
  const Int_t kColorFDA = kGreen;
  const Int_t kColorFDC = kGreen;
  Double_t X = kShiftX + kFDACellSideX * 0.5;
  Double_t Y = kShiftY + kFDACellSideY * 0.5;
  Double_t WLS_dx = 0.4;
  Double_t WLS_dz = 2.5;
  Double_t WLS_SideA_Long_dy = 24.20;  // 24.2;
  Double_t WLS_SideC_Long_dy = 24.20;  // 24.2;
  Double_t WLS_SideA_Short_dy = 18.20; // 18.41;
  Double_t WLS_SideC_Short_dy = 20.70; // 20.91;

  // Creating FDA WLS bars
  TGeoVolume* vFDA_WLS_s = new TGeoVolume("FDAWLSshort", new TGeoBBox("shFDAWLSbarShort", WLS_dx / 2.0, WLS_SideA_Short_dy / 2.0, WLS_dz / 2.0), medFDASci);
  TGeoVolume* vFDA_WLS_l = new TGeoVolume("FDAWLSlong", new TGeoBBox("shFDAWLSbarLong", WLS_dx / 2.0, WLS_SideA_Long_dy / 2.0, WLS_dz / 2.0), medFDASci);
  vFDA_WLS_l->SetLineColor(kRed);
  vFDA_WLS_s->SetLineColor(kRed);

  // Creating FDC WLS bars
  TGeoVolume* vFDC_WLS_s = new TGeoVolume("FDCWLSshort", new TGeoBBox("shFDCWLSbarShort", WLS_dx / 2.0, WLS_SideC_Short_dy / 2.0, WLS_dz / 2.0), medFDCSci);
  TGeoVolume* vFDC_WLS_l = new TGeoVolume("FDCWLSlong", new TGeoBBox("shFDCWLSbarLong", WLS_dx / 2.0, WLS_SideC_Long_dy / 2.0, WLS_dz / 2.0), medFDCSci);
  vFDC_WLS_l->SetLineColor(kRed);
  vFDC_WLS_s->SetLineColor(kRed);

  // Make FDA scintillator pad
  new TGeoBBox("shFDAbox", kFDACellSideX / 2.0, kFDACellSideY / 2.0, kFDACelldz / 2.0);
  new TGeoTube("shFDAHole", 0., kFDABeamPipeR, kFDACelldz);
  (new TGeoTranslation("trFDAbox", X, Y, 0.))->RegisterYourself();

  TGeoVolume* vFDApad = new TGeoVolume("FDApad", new TGeoCompositeShape("shFDApad", "shFDAbox:trFDAbox-shFDAHole"), medFDASci);
  vFDApad->SetLineColor(kColorFDA);

  TGeoVolume* secFDA = new TGeoVolumeAssembly("FDAsec");
  // Add PAD
  Double_t fX_FDA_WLS_s = 0.1 + WLS_dx / 2.0;
  Double_t fX_FDA_WLS_l = kShiftX + WLS_dx / 2.0 + kFDACellSideX + 0.04;
  secFDA->AddNode(vFDApad, 1);
  secFDA->AddNode(vFDA_WLS_s, 1, new TGeoTranslation(fX_FDA_WLS_s, kFDABeamPipeR + WLS_SideA_Short_dy / 2.0, 0.0));
  secFDA->AddNode(vFDA_WLS_l, 1, new TGeoTranslation(fX_FDA_WLS_l, kShiftY + WLS_SideA_Long_dy / 2.0, 0.0));

  /// Assembling FDA adding 4 sectors                                       //  Sectors
  TGeoVolume* vFDAarray = new TGeoVolumeAssembly("FDA"); //        ^ y
  vFDAarray->AddNode(secFDA, 1);                         //        |
  vFDAarray->AddNode(secFDA, 2, Ry180);                  //   2    |   1
  vFDAarray->AddNode(secFDA, 3, Rz180);                  // --------------->  x
  vFDAarray->AddNode(secFDA, 4, Rx180);                  //   3    |   4

  const Float_t kPosFDA = 1696.67; // z-center of assembly (cm)

  vFDD->AddNode(vFDAarray, 1, new TGeoTranslation(0., 0., kPosFDA - kFDACelldz / 2. - 0.1));
  vFDD->AddNode(vFDAarray, 2, new TGeoTranslation(0., 0., kPosFDA + kFDACelldz / 2. + 0.1));

  /// FDD_C in the tunnel

  new TGeoBBox("shFDCbox", kFDCCellSideX / 2.0, kFDCCellSideY / 2.0, kFDCCelldz / 2.0);
  new TGeoTube("shFDCHole", 0., kFDCBeamPipeR, kFDCCelldz);
  X = kShiftX + kFDCCellSideX * 0.5;
  Y = kShiftY + kFDCCellSideY * 0.5;
  (new TGeoTranslation("trFDCbox", X, Y, 0.))->RegisterYourself();
  TGeoVolume* vFDCpad = new TGeoVolume("FDCpad", new TGeoCompositeShape("shFDCpad", "shFDCbox:trFDCbox-shFDCHole"), medFDCSci);
  vFDCpad->SetLineColor(kColorFDC);

  /// Creating Sector for Tunnel (Asembly:  Scintillator Pad + Light guide)
  TGeoVolume* voFDC = new TGeoVolumeAssembly("FDCsec");
  // Add PAD
  voFDC->AddNode(vFDCpad, 1);
  // Add FDC WLS Short bar
  voFDC->AddNode(vFDC_WLS_s, 1, new TGeoTranslation(0.1 + WLS_dx / 2.0, kFDCBeamPipeR + WLS_SideC_Short_dy / 2.0, 0.0));
  // Add FDC WLS Long  bar
  voFDC->AddNode(vFDC_WLS_l, 1, new TGeoTranslation(0.04 + WLS_dx / 2.0 + kFDCCellSideX + kShiftX, kShiftY + WLS_SideC_Long_dy / 2.0, 0.0));

  /// Assembling FDC adding the 4 sectors                 //  Sectors
  TGeoVolume* vFDCarray = new TGeoVolumeAssembly("FDC"); //        ^ y
  vFDCarray->AddNode(voFDC, 1);                          //        |
  vFDCarray->AddNode(voFDC, 2, Ry180);                   //   2    |   1
  vFDCarray->AddNode(voFDC, 3, Rz180);                   // --------------->  x
  vFDCarray->AddNode(voFDC, 4, Rx180);                   //   3    |   4
                                                         //        |

  const Float_t kPosFDC = -kZbegFrontBar - 2. - 3.0 - 0.3; // 3.0 = (5.6 + 0.2 + 0.2)/2.
  vFDD->AddNode(vFDCarray, 1, new TGeoTranslation(0., 0., kPosFDC - kFDCCelldz / 2. - 0.23));
  vFDD->AddNode(vFDCarray, 2, new TGeoTranslation(0., 0., kPosFDC + kFDCCelldz / 2. + 0.23));

  TGeoVolume* vALIC = gGeoManager->GetVolume("cave");
  if (!vALIC) {
    LOG(FATAL) << "Could not find the top volume";
  }
  vALIC->AddNode(vFDD, 1);
}
