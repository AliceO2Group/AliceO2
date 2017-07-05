// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "TClonesArray.h"
#include "TString.h"
#include "TMath.h"
#include "TGeoManager.h"      // for TGeoManager

#include "FairVolume.h"
#include "FairRootManager.h"

#include "TOFBase/Hit.h"
#include "TOFSimulation/Detector.h"

#include "DetectorsBase/GeometryManager.h"

#include <TVirtualMC.h>  // for TVirtualMC, gMC

using namespace o2::TOF;

ClassImp(Detector)

Detector::Detector(const char* Name, Bool_t Active):
o2::Base::Detector(Name, Active),
  mTOFHoles(kTRUE)
{
  for(Int_t i=0;i < Geo::NSECTORS;i++)
    mTOFSectors[i] = 1;
}

void Detector::Initialize(){
}

Bool_t Detector::ProcessHits( FairVolume* v) {
  return true;
}

Hit *Detector::AddHit(Int_t shunt, Int_t trackID, Int_t parentID, Int_t primary, Double_t initialEnergy,
              Int_t detID, const Point3D<float> &pos, const Vector3D<float> &mom, Double_t time, Double_t eLoss){
  
  return NULL;

}

void Detector::Register(){
//  FairRootManager::Instance()->Register("EMCALHit", "EMCAL", mPointCollection, kTRUE);
}

TClonesArray* Detector::GetCollection(Int_t iColl) const {
  return nullptr;
}

void Detector::Reset() {
}


void Detector::CreateMaterials() {

  //  AliMagF *magneticField = (AliMagF*)((AliMagF*)TGeoGlobalMagField::Instance()->GetField());

  Int_t   isxfld = 0.;//magneticField->Integ();
  Float_t sxmgmx = 0.;//magneticField->Max();

  //--- Quartz (SiO2) ---
  Float_t   aq[2] = { 28.0855,15.9994};
  Float_t   zq[2] = { 14.,8. };
  Float_t   wq[2] = { 1.,2. };
  Float_t   dq = 2.7; // (+5.9%)
  Int_t nq = -2;

  // --- Nomex (C14H22O2N2) ---
  Float_t anox[4] = {12.011,1.00794,15.9994,14.00674};
  Float_t znox[4] = { 6.,  1.,  8.,  7.};
  Float_t wnox[4] = {14., 22., 2., 2.};
  //Float_t dnox  = 0.048; //old value
  Float_t dnox  = 0.22;    // (x 4.6)
  Int_t nnox   = -4;

  // --- G10  {Si, O, C, H, O} ---
  Float_t we[7], na[7];

  Float_t ag10[5] = {28.0855,15.9994,12.011,1.00794,15.9994};
  Float_t zg10[5] = {14., 8., 6., 1., 8.};
  Float_t wmatg10[5];
  Int_t nlmatg10 = 5;
  na[0]= 1. ,   na[1]= 2. ,   na[2]= 0. ,   na[3]= 0. ,   na[4]= 0.;
  MaterialMixer(we,ag10,na,5);
  wmatg10[0]= we[0]*0.6;
  wmatg10[1]= we[1]*0.6;
  na[0]= 0. ,   na[1]= 0. ,   na[2]= 14. ,   na[3]= 20. ,   na[4]= 3.;
  MaterialMixer(we,ag10,na,5);
  wmatg10[2]= we[2]*0.4;
  wmatg10[3]= we[3]*0.4;
  wmatg10[4]= we[4]*0.4;
  //Float_t densg10 = 1.7; //old value
  Float_t densg10 = 2.0; // (+17.8%)

  // --- Water ---
  Float_t awa[2] = {  1.00794, 15.9994 };
  Float_t zwa[2] = {  1.,  8. };
  Float_t wwa[2] = {  2.,  1. };
  Float_t dwa    = 1.0;
  Int_t nwa = -2;

  // --- Air ---
  Float_t aAir[4]={12.011,14.00674,15.9994,39.948};
  Float_t zAir[4]={6.,7.,8.,18.};
  Float_t wAir[4]={0.000124,0.755267,0.231781,0.012827};
  Float_t dAir   = 1.20479E-3;

  // --- Fibre Glass ---
  Float_t afg[4] = {28.0855,15.9994,12.011,1.00794};
  Float_t zfg[4] = {14., 8., 6., 1.};
  Float_t wfg[4] = {0.12906,0.29405,0.51502,0.06187};
  //Float_t dfg    = 1.111;
  Float_t dfg    = 2.05; // (x1.845)
  Int_t nfg      = 4;

  // --- Freon C2F4H2 + SF6 ---
  Float_t afre[4] = {12.011,1.00794,18.9984032,32.0065};
  Float_t zfre[4] = { 6., 1., 9., 16.};
  Float_t wfre[4] = {0.21250,0.01787,0.74827,0.021355};
  Float_t densfre = 0.00375;
  Int_t nfre     = 4;

  // --- Cables and tubes {Al, Cu} ---
  Float_t acbt[2] = {26.981539,63.546};
  Float_t zcbt[2] = {13., 29.};
  Float_t wcbt[2] = {0.407,0.593};
  Float_t decbt   = 0.68;

  // --- Cable {CH2, Al, Cu} ---
  Float_t asc[4] = {12.011, 1.00794, 26.981539,63.546};
  Float_t zsc[4] = { 6., 1., 13., 29.};
  Float_t wsc[4];
  for (Int_t ii=0; ii<4; ii++) wsc[ii]=0.;

  Float_t wDummy[4], nDummy[4];
  for (Int_t ii=0; ii<4; ii++) wDummy[ii]=0.;
  for (Int_t ii=0; ii<4; ii++) nDummy[ii]=0.;
  nDummy[0] = 1.;
  nDummy[1] = 2.;
  MaterialMixer(wDummy,asc,nDummy,2);
  wsc[0] = 0.4375*wDummy[0];
  wsc[1] = 0.4375*wDummy[1];
  wsc[2] = 0.3244;
  wsc[3] = 0.2381;
  Float_t dsc = 1.223;

  // --- Crates boxes {Al, Cu, Fe, Cr, Ni} ---
  Float_t acra[5]= {26.981539,63.546,55.845,51.9961,58.6934};
  Float_t zcra[5]= {13., 29., 26., 24., 28.};
  Float_t wcra[5]= {0.7,0.2,0.07,0.018,0.012};
  Float_t dcra   = 0.77;

  // --- Polietilene CH2 ---
  Float_t aPlastic[2] = {12.011, 1.00794};
  Float_t zPlastic[2] = { 6., 1.};
  Float_t wPlastic[2] = { 1., 2.};
  //Float_t dPlastic = 0.92; // PDB value
  Float_t dPlastic = 0.93; // (~+1.1%)
  Int_t nwPlastic = -2;

  Mixture ( 0, "Air$", aAir, zAir, dAir, 4, wAir);
  Mixture ( 1, "Nomex$", anox, znox, dnox, nnox, wnox);
  Mixture ( 2, "G10$", ag10, zg10, densg10, nlmatg10, wmatg10);
  Mixture ( 3, "fibre glass$", afg, zfg, dfg, nfg, wfg);
  Material( 4, "Al $", 26.981539, 13., 2.7, -8.9, 999.);
  Float_t factor = 0.4/1.5*2./3.;
  Material( 5, "Al honeycomb$", 26.981539, 13., 2.7*factor, -8.9/factor, 999.);
  Mixture ( 6, "Freon$", afre, zfre, densfre, nfre, wfre);
  Mixture ( 7, "Glass$", aq, zq, dq, nq, wq);
  Mixture ( 8, "Water$",  awa, zwa, dwa, nwa, wwa);
  Mixture ( 9, "cables+tubes$", acbt, zcbt, decbt, 2, wcbt);
  Material(10, "Cu $", 63.546, 29., 8.96, -1.43, 999.);
  Mixture (11, "cable$", asc, zsc, dsc, 4, wsc);
  Mixture (12, "Al+Cu+steel$", acra, zcra, dcra, 5, wcra);
  Mixture (13, "plastic$", aPlastic, zPlastic, dPlastic, nwPlastic, wPlastic);
  Float_t factorHoles = 1./36.5;
  Material(14, "Al honey for holes$", 26.981539, 13., 2.7*factorHoles, -8.9/factorHoles, 999.);

  Float_t epsil, stmin, deemax, stemax;

  //   STD data
  //  EPSIL  = 0.1   ! Tracking precision,
  //  STEMAX = 0.1   ! Maximum displacement for multiple scattering
  //  DEEMAX = 0.1   ! Maximum fractional energy loss, DLS
  //  STMIN  = 0.1

  // TOF data
  epsil  = .001;  // Tracking precision,
  stemax = -1.;   // Maximum displacement for multiple scattering
  deemax = -.3;   // Maximum fractional energy loss, DLS
  stmin  = -.8;

  Medium( kAir,"TOF_Air$",          0, 0, isxfld, sxmgmx, 10., stemax, deemax, epsil, stmin);
  Medium( kNomex,"TOF_Nomex$",        1, 0, isxfld, sxmgmx, 10., stemax, deemax, epsil, stmin);
  Medium( kG10,"TOF_G10$",          2, 0, isxfld, sxmgmx, 10., stemax, deemax, epsil, stmin);
  Medium( kFiberGlass,"TOF_fibre glass$",  3, 0, isxfld, sxmgmx, 10., stemax, deemax, epsil, stmin);
  Medium( kAlFrame,"TOF_Al Frame$",     4, 0, isxfld, sxmgmx, 10., stemax, deemax, epsil, stmin);
  Medium( kHoneycomb,"TOF_honeycomb$",    5, 0, isxfld, sxmgmx, 10., stemax, deemax, epsil, stmin);
  Medium( kFre,"TOF_Fre$",          6, 0, isxfld, sxmgmx, 10., stemax, deemax, epsil, stmin);
  Medium( kCuS,"TOF_Cu-S$",        10, 1, isxfld, sxmgmx, 10., stemax, deemax, epsil, stmin);
  Medium( kGlass,"TOF_Glass$",        7, 0, isxfld, sxmgmx, 10., stemax, deemax, epsil, stmin);
  Medium( kWater,"TOF_Water$",        8, 0, isxfld, sxmgmx, 10., stemax, deemax, epsil, stmin);
  Medium( kCable,"TOF_Cable$",       11, 0, isxfld, sxmgmx, 10., stemax, deemax, epsil, stmin);
  Medium( kCableTubes,"TOF_Cables+Tubes$", 9, 0, isxfld, sxmgmx, 10., stemax, deemax, epsil, stmin);
  Medium( kCopper,"TOF_Copper$",      10, 0, isxfld, sxmgmx, 10., stemax, deemax, epsil, stmin);
  Medium( kPlastic,"TOF_Plastic$",     13, 0, isxfld, sxmgmx, 10., stemax, deemax, epsil, stmin);
  Medium( kCrates,"TOF_Crates$",      12, 0, isxfld, sxmgmx, 10., stemax, deemax, epsil, stmin);
  Medium( kHoneyHoles,"TOF_honey_holes$", 14, 0, isxfld, sxmgmx, 10., stemax, deemax, epsil, stmin);

}

void Detector::MaterialMixer(Float_t * p, const Float_t * const a,
			       const Float_t * const m, Int_t n) const
{
  // a[] atomic weights vector      (in)
  //     (atoms present in more compound appear separately)
  // m[] number of corresponding atoms in the compound  (in)
  Float_t t = 0.;
  for (Int_t i = 0; i < n; ++i) {
    p[i] = a[i]*m[i];
    t  += p[i];
  }
  for (Int_t i = 0; i < n; ++i) {
    p[i] = p[i]/t;
  }
}

void Detector::ConstructGeometry() {
  CreateMaterials();

  /*
    xTof = 124.5;//fTOFGeometry->StripLength()+2.*(0.3+0.03); // cm,  x-dimension of FTOA volume
    yTof = fTOFGeometry->Rmax()-fTOFGeometry->Rmin(); // cm,  y-dimension of FTOA volume
    Float_t zTof = fTOFGeometry->ZlenA();             // cm,  z-dimension of FTOA volume
   */

  Float_t xTof=124.5, yTof=Geo::RMAX-Geo::RMIN,zTof=Geo::ZLENA;
  DefineGeometry(xTof,yTof,zTof);
  

}

void Detector::ConstructSuperModule(Int_t imodule){

}

void Detector::DefineGeometry(Float_t xtof, Float_t ytof, Float_t zlenA)
{
  //
  // Definition of the Time Of Fligh Resistive Plate Chambers
  //

  Float_t xFLT, yFLT, zFLTA;
  xFLT  = xtof     - 2.*Geo::MODULEWALLTHICKNESS;
  yFLT  = ytof*0.5 -    Geo::MODULEWALLTHICKNESS;
  zFLTA = zlenA    - 2.*Geo::MODULEWALLTHICKNESS;

  CreateModules(xtof, ytof, zlenA, xFLT, yFLT, zFLTA);
  MakeStripsInModules(ytof, zlenA);

  CreateModuleCovers(xtof, zlenA);

  CreateBackZone(xtof, ytof, zlenA);
  // MakeFrontEndElectronics(xtof);
  // MakeFEACooling(xtof);
  // MakeNinoMask(xtof);
  // MakeSuperModuleCooling(xtof, ytof, zlenA);
  // MakeSuperModuleServices(xtof, ytof, zlenA);

  // MakeModulesInBTOFvolumes(ytof, zlenA);
  // MakeCoversInBTOFvolumes();
  // MakeBackInBTOFvolumes(ytof);

  // MakeReadoutCrates(ytof);

  // Create the 18 sectors
  //  AddAlignableVolumes();

  // put the supervolumes in the 18 sectors
  MakeModulesInBTOFvolumes(ytof, zlenA);
}

void Detector::CreateModules(Float_t xtof,  Float_t ytof, Float_t zlenA,
			       Float_t xFLT,  Float_t yFLT, Float_t zFLTA) const
{
  //
  // Create supermodule volume
  // and wall volumes to separate 5 modules
  //

  Int_t idrotm[8]; for (Int_t ii=0; ii<8; ii++) idrotm[ii]=0;

  // Definition of the of fibre glass modules (FTOA, FTOB and FTOC)
  Float_t  par[3];
  par[0] = xtof * 0.5;
  par[1] = ytof * 0.25;
  par[2] = zlenA * 0.5;
  TVirtualMC::GetMC()->Gsvolu("FTOA", "BOX ", getMedium(kFiberGlass), par, 3);  // Fibre glass

  if (mTOFHoles) {
    par[0] =  xtof * 0.5;
    par[1] =  ytof * 0.25;
    par[2] = (zlenA*0.5 - Geo::INTERCENTRMODBORDER1)*0.5;
    TVirtualMC::GetMC()->Gsvolu("FTOB", "BOX ", getMedium(kFiberGlass), par, 3);  // Fibre glass
    TVirtualMC::GetMC()->Gsvolu("FTOC", "BOX ", getMedium(kFiberGlass), par, 3);  // Fibre glass
  }

  // Definition and positioning
  // of the not sensitive volumes with Insensitive Freon (FLTA, FLTB and FLTC)
  par[0] = xFLT*0.5;
  par[1] = yFLT*0.5;
  par[2] = zFLTA*0.5;
  TVirtualMC::GetMC()->Gsvolu("FLTA", "BOX ", getMedium(kFre), par, 3); // Freon mix

  Float_t xcoor, ycoor, zcoor;
  xcoor = 0.;
  ycoor = Geo::MODULEWALLTHICKNESS*0.5;
  zcoor = 0.;
  TVirtualMC::GetMC()->Gspos ("FLTA", 0, "FTOA", xcoor, ycoor, zcoor, 0, "ONLY");

  if (mTOFHoles) {
    par[2] = (zlenA*0.5 - 2.*Geo::MODULEWALLTHICKNESS - Geo::INTERCENTRMODBORDER1)*0.5;
    TVirtualMC::GetMC()->Gsvolu("FLTB", "BOX ", getMedium(kFre), par, 3); // Freon mix
    TVirtualMC::GetMC()->Gsvolu("FLTC", "BOX ", getMedium(kFre), par, 3); // Freon mix

    //xcoor = 0.;
    //ycoor = Geo::MODULEWALLTHICKNESS*0.5;
    zcoor = Geo::MODULEWALLTHICKNESS;
    TVirtualMC::GetMC()->Gspos ("FLTB", 0, "FTOB", xcoor, ycoor, zcoor, 0, "ONLY");
    TVirtualMC::GetMC()->Gspos ("FLTC", 0, "FTOC", xcoor, ycoor,-zcoor, 0, "ONLY");
  }

  // Definition and positioning
  // of the fibre glass walls between central and intermediate modules (FWZ1 and FWZ2)
  Float_t alpha, tgal, beta, tgbe, trpa[11];
  //tgal  = (yFLT - 2.*Geo::LENGTHINCEMODBORDER)/(Geo::INTERCENTRMODBORDER2 - Geo::INTERCENTRMODBORDER1);
  tgal  = (yFLT - Geo::LENGTHINCEMODBORDERU - Geo::LENGTHINCEMODBORDERD)/(Geo::INTERCENTRMODBORDER2 - Geo::INTERCENTRMODBORDER1);
  alpha = TMath::ATan(tgal);
  beta  = (TMath::Pi()*0.5 - alpha)*0.5;
  tgbe  = TMath::Tan(beta);
  trpa[0]  = xFLT*0.5;
  trpa[1]  = 0.;
  trpa[2]  = 0.;
  trpa[3]  = 2.*Geo::MODULEWALLTHICKNESS;
  //trpa[4]  = (Geo::LENGTHINCEMODBORDER - 2.*Geo::MODULEWALLTHICKNESS*tgbe)*0.5;
  //trpa[5]  = (Geo::LENGTHINCEMODBORDER + 2.*Geo::MODULEWALLTHICKNESS*tgbe)*0.5;
  trpa[4]  = (Geo::LENGTHINCEMODBORDERD - 2.*Geo::MODULEWALLTHICKNESS*tgbe)*0.5;
  trpa[5]  = (Geo::LENGTHINCEMODBORDERD + 2.*Geo::MODULEWALLTHICKNESS*tgbe)*0.5;
  trpa[6]  = TMath::ATan(tgbe*0.5)*TMath::RadToDeg(); //TMath::ATan((trpa[5] - trpa[4])/(2.*trpa[3]))*TMath::RadToDeg();
  trpa[7]  = 2.*Geo::MODULEWALLTHICKNESS;
  trpa[8]  = (Geo::LENGTHINCEMODBORDERD - 2.*Geo::MODULEWALLTHICKNESS*tgbe)*0.5;
  trpa[9]  = (Geo::LENGTHINCEMODBORDERD + 2.*Geo::MODULEWALLTHICKNESS*tgbe)*0.5;
  //trpa[8]  = (Geo::LENGTHINCEMODBORDER - 2.*Geo::MODULEWALLTHICKNESS*tgbe)*0.5;
  //trpa[9]  = (Geo::LENGTHINCEMODBORDER + 2.*Geo::MODULEWALLTHICKNESS*tgbe)*0.5;
  trpa[10] = TMath::ATan(tgbe*0.5)*TMath::RadToDeg(); //TMath::ATan((trpa[5] - trpa[4])/(2.*trpa[3]))*TMath::RadToDeg();
  TVirtualMC::GetMC()->Gsvolu("FWZ1D", "TRAP", getMedium(kFiberGlass), trpa, 11); // Fibre glass

  Matrix (idrotm[0],90., 90.,180.,0.,90.,180.);
  Matrix (idrotm[1],90., 90.,  0.,0.,90.,  0.);

  //xcoor = 0.;
  //ycoor = -(yFLT - Geo::LENGTHINCEMODBORDER)*0.5;
  ycoor = -(yFLT - Geo::LENGTHINCEMODBORDERD)*0.5;
  zcoor = Geo::INTERCENTRMODBORDER1;
  TVirtualMC::GetMC()->Gspos("FWZ1D", 1, "FLTA", xcoor, ycoor, zcoor, idrotm[0], "ONLY");
  TVirtualMC::GetMC()->Gspos("FWZ1D", 2, "FLTA", xcoor, ycoor,-zcoor, idrotm[1], "ONLY");

  Float_t y0B, ycoorB, zcoorB;

  if (mTOFHoles) {
    //y0B = Geo::LENGTHINCEMODBORDER - Geo::MODULEWALLTHICKNESS*tgbe;
    y0B = Geo::LENGTHINCEMODBORDERD - Geo::MODULEWALLTHICKNESS*tgbe;
    trpa[0]  = xFLT*0.5;
    trpa[1]  = 0.;
    trpa[2]  = 0.;
    trpa[3]  = Geo::MODULEWALLTHICKNESS;
    trpa[4]  = (y0B - Geo::MODULEWALLTHICKNESS*tgbe)*0.5;
    trpa[5]  = (y0B + Geo::MODULEWALLTHICKNESS*tgbe)*0.5;
    trpa[6]  = TMath::ATan(tgbe*0.5)*TMath::RadToDeg(); //TMath::ATan((trpa[5] - trpa[4])/(2.*trpa[3]))*TMath::RadToDeg();
    trpa[7]  = Geo::MODULEWALLTHICKNESS;
    trpa[8]  = (y0B - Geo::MODULEWALLTHICKNESS*tgbe)*0.5;
    trpa[9]  = (y0B + Geo::MODULEWALLTHICKNESS*tgbe)*0.5;
    trpa[10] = TMath::ATan(tgbe*0.5)*TMath::RadToDeg(); //TMath::ATan((trpa[5] - trpa[4])/(2.*trpa[3]))*TMath::RadToDeg();
    //xcoor = 0.;
    ycoorB = ycoor - Geo::MODULEWALLTHICKNESS*0.5*tgbe;
    zcoorB = (zlenA*0.5 - 2.*Geo::MODULEWALLTHICKNESS - Geo::INTERCENTRMODBORDER1)*0.5 - 2.*Geo::MODULEWALLTHICKNESS;
    TVirtualMC::GetMC()->Gsvolu("FWZAD", "TRAP", getMedium(kFiberGlass), trpa, 11); // Fibre glass
    TVirtualMC::GetMC()->Gspos("FWZAD", 1, "FLTB", xcoor, ycoorB, zcoorB, idrotm[1], "ONLY");
    TVirtualMC::GetMC()->Gspos("FWZAD", 2, "FLTC", xcoor, ycoorB,-zcoorB, idrotm[0], "ONLY");
  }



  tgal  = (yFLT - Geo::LENGTHINCEMODBORDERU - Geo::LENGTHINCEMODBORDERD)/(Geo::INTERCENTRMODBORDER2 - Geo::INTERCENTRMODBORDER1);
  alpha = TMath::ATan(tgal);
  beta  = (TMath::Pi()*0.5 - alpha)*0.5;
  tgbe  = TMath::Tan(beta);
  trpa[0]  = xFLT*0.5;
  trpa[1]  = 0.;
  trpa[2]  = 0.;
  trpa[3]  = 2.*Geo::MODULEWALLTHICKNESS;
  //trpa[4]  = (Geo::LENGTHINCEMODBORDER - 2.*Geo::MODULEWALLTHICKNESS*tgbe)*0.5;
  //trpa[5]  = (Geo::LENGTHINCEMODBORDER + 2.*Geo::MODULEWALLTHICKNESS*tgbe)*0.5;
  trpa[4]  = (Geo::LENGTHINCEMODBORDERU - 2.*Geo::MODULEWALLTHICKNESS*tgbe)*0.5;
  trpa[5]  = (Geo::LENGTHINCEMODBORDERU + 2.*Geo::MODULEWALLTHICKNESS*tgbe)*0.5;
  trpa[6]  = TMath::ATan(tgbe*0.5)*TMath::RadToDeg(); //TMath::ATan((trpa[5] - trpa[4])/(2.*trpa[3]))*TMath::RadToDeg();
  trpa[7]  = 2.*Geo::MODULEWALLTHICKNESS;
  trpa[8]  = (Geo::LENGTHINCEMODBORDERU - 2.*Geo::MODULEWALLTHICKNESS*tgbe)*0.5;
  trpa[9]  = (Geo::LENGTHINCEMODBORDERU + 2.*Geo::MODULEWALLTHICKNESS*tgbe)*0.5;
  //trpa[8]  = (Geo::LENGTHINCEMODBORDER - 2.*Geo::MODULEWALLTHICKNESS*tgbe)*0.5;
  //trpa[9]  = (Geo::LENGTHINCEMODBORDER + 2.*Geo::MODULEWALLTHICKNESS*tgbe)*0.5;
  trpa[10] = TMath::ATan(tgbe*0.5)*TMath::RadToDeg(); //TMath::ATan((trpa[5] - trpa[4])/(2.*trpa[3]))*TMath::RadToDeg();
  TVirtualMC::GetMC()->Gsvolu("FWZ1U", "TRAP", getMedium(kFiberGlass), trpa, 11); // Fibre glass


  Matrix (idrotm[2],90.,270.,  0.,0.,90.,180.);
  Matrix (idrotm[3],90.,270.,180.,0.,90.,  0.);

  //xcoor = 0.;
  //ycoor = (yFLT - Geo::LENGTHINCEMODBORDER)*0.5;
  ycoor = (yFLT - Geo::LENGTHINCEMODBORDERU)*0.5;
  zcoor = Geo::INTERCENTRMODBORDER2;
  TVirtualMC::GetMC()->Gspos("FWZ1U", 1, "FLTA", xcoor, ycoor, zcoor,idrotm[2], "ONLY");
  TVirtualMC::GetMC()->Gspos("FWZ1U", 2, "FLTA", xcoor, ycoor,-zcoor,idrotm[3], "ONLY");

  if (mTOFHoles) {
    //y0B = Geo::LENGTHINCEMODBORDER + Geo::MODULEWALLTHICKNESS*tgbe;
    y0B = Geo::LENGTHINCEMODBORDERU + Geo::MODULEWALLTHICKNESS*tgbe;
    trpa[0]  = xFLT*0.5;
    trpa[1]  = 0.;
    trpa[2]  = 0.;
    trpa[3]  = Geo::MODULEWALLTHICKNESS;
    trpa[4]  = (y0B - Geo::MODULEWALLTHICKNESS*tgbe)*0.5;
    trpa[5]  = (y0B + Geo::MODULEWALLTHICKNESS*tgbe)*0.5;
    trpa[6]  = TMath::ATan(tgbe*0.5)*TMath::RadToDeg(); //TMath::ATan((trpa[5] - trpa[4])/(2.*trpa[3]))*TMath::RadToDeg();
    trpa[7]  = Geo::MODULEWALLTHICKNESS;
    trpa[8]  = (y0B - Geo::MODULEWALLTHICKNESS*tgbe)*0.5;
    trpa[9]  = (y0B + Geo::MODULEWALLTHICKNESS*tgbe)*0.5;
    trpa[10] = TMath::ATan(tgbe*0.5)*TMath::RadToDeg(); //TMath::ATan((trpa[5] - trpa[4])/(2.*trpa[3]))*TMath::RadToDeg();
    TVirtualMC::GetMC()->Gsvolu("FWZBU", "TRAP", getMedium(kFiberGlass), trpa, 11); // Fibre glass
    //xcoor = 0.;
    ycoorB = ycoor - Geo::MODULEWALLTHICKNESS*0.5*tgbe;
    zcoorB = (zlenA*0.5 - 2.*Geo::MODULEWALLTHICKNESS - Geo::INTERCENTRMODBORDER1)*0.5 -
      (Geo::INTERCENTRMODBORDER2 - Geo::INTERCENTRMODBORDER1) - 2.*Geo::MODULEWALLTHICKNESS;
    TVirtualMC::GetMC()->Gspos("FWZBU", 1, "FLTB", xcoor, ycoorB, zcoorB, idrotm[3], "ONLY");
    TVirtualMC::GetMC()->Gspos("FWZBU", 2, "FLTC", xcoor, ycoorB,-zcoorB, idrotm[2], "ONLY");
  }

  trpa[0] = 0.5*(Geo::INTERCENTRMODBORDER2 - Geo::INTERCENTRMODBORDER1)/TMath::Cos(alpha);
  trpa[1] = 2.*Geo::MODULEWALLTHICKNESS;
  trpa[2] = xFLT*0.5;
  trpa[3] = -beta*TMath::RadToDeg();
  trpa[4] = 0.;
  trpa[5] = 0.;
  TVirtualMC::GetMC()->Gsvolu("FWZ2", "PARA", getMedium(kFiberGlass), trpa, 6); // Fibre glass

  Matrix (idrotm[4],     alpha*TMath::RadToDeg(),90.,90.+alpha*TMath::RadToDeg(),90.,90.,180.);
  Matrix (idrotm[5],180.-alpha*TMath::RadToDeg(),90.,90.-alpha*TMath::RadToDeg(),90.,90.,  0.);

  //xcoor = 0.;
  //ycoor = 0.;
  ycoor = (Geo::LENGTHINCEMODBORDERD - Geo::LENGTHINCEMODBORDERU)*0.5;
  zcoor = (Geo::INTERCENTRMODBORDER2 + Geo::INTERCENTRMODBORDER1)*0.5;
  TVirtualMC::GetMC()->Gspos("FWZ2", 1, "FLTA", xcoor, ycoor, zcoor, idrotm[4], "ONLY");
  TVirtualMC::GetMC()->Gspos("FWZ2", 2, "FLTA", xcoor, ycoor,-zcoor, idrotm[5], "ONLY");

  if (mTOFHoles) {
    trpa[0] = 0.5*(Geo::INTERCENTRMODBORDER2 - Geo::INTERCENTRMODBORDER1)/TMath::Cos(alpha);
    trpa[1] = Geo::MODULEWALLTHICKNESS;
    trpa[2] = xFLT*0.5;
    trpa[3] = -beta*TMath::RadToDeg();
    trpa[4] = 0.;
    trpa[5] = 0.;
    TVirtualMC::GetMC()->Gsvolu("FWZC", "PARA", getMedium(kFiberGlass), trpa, 6); // Fibre glass
    //xcoor = 0.;
    ycoorB = ycoor - Geo::MODULEWALLTHICKNESS*tgbe;
    zcoorB = (zlenA*0.5 - 2.*Geo::MODULEWALLTHICKNESS - Geo::INTERCENTRMODBORDER1)*0.5 -
      (Geo::INTERCENTRMODBORDER2 - Geo::INTERCENTRMODBORDER1)*0.5 - 2.*Geo::MODULEWALLTHICKNESS;
    TVirtualMC::GetMC()->Gspos("FWZC", 1, "FLTB", xcoor, ycoorB, zcoorB, idrotm[5], "ONLY");
    TVirtualMC::GetMC()->Gspos("FWZC", 2, "FLTC", xcoor, ycoorB,-zcoorB, idrotm[4], "ONLY");
  }


  // Definition and positioning
  // of the fibre glass walls between intermediate and lateral modules (FWZ3 and FWZ4)
  tgal  = (yFLT - 2.*Geo::LENGTHEXINMODBORDER)/(Geo::EXTERINTERMODBORDER2 - Geo::EXTERINTERMODBORDER1);
  alpha = TMath::ATan(tgal);
  beta  = (TMath::Pi()*0.5 - alpha)*0.5;
  tgbe  = TMath::Tan(beta);
  trpa[0]  = xFLT*0.5;
  trpa[1]  = 0.;
  trpa[2]  = 0.;
  trpa[3]  = 2.*Geo::MODULEWALLTHICKNESS;
  trpa[4]  = (Geo::LENGTHEXINMODBORDER - 2.*Geo::MODULEWALLTHICKNESS*tgbe)*0.5;
  trpa[5]  = (Geo::LENGTHEXINMODBORDER + 2.*Geo::MODULEWALLTHICKNESS*tgbe)*0.5;
  trpa[6]  = TMath::ATan(tgbe*0.5)*TMath::RadToDeg(); //TMath::ATan((trpa[5] - trpa[4])/(2.*trpa[3]))*TMath::RadToDeg();
  trpa[7]  = 2.*Geo::MODULEWALLTHICKNESS;
  trpa[8]  = (Geo::LENGTHEXINMODBORDER - 2.*Geo::MODULEWALLTHICKNESS*tgbe)*0.5;
  trpa[9]  = (Geo::LENGTHEXINMODBORDER + 2.*Geo::MODULEWALLTHICKNESS*tgbe)*0.5;
  trpa[10] = TMath::ATan(tgbe*0.5)*TMath::RadToDeg(); //TMath::ATan((trpa[5] - trpa[4])/(2.*trpa[3]))*TMath::RadToDeg();
  TVirtualMC::GetMC()->Gsvolu("FWZ3", "TRAP", getMedium(kFiberGlass), trpa, 11); // Fibre glass

  //xcoor = 0.;
  ycoor = (yFLT - Geo::LENGTHEXINMODBORDER)*0.5;
  zcoor = Geo::EXTERINTERMODBORDER1;
  TVirtualMC::GetMC()->Gspos("FWZ3", 1, "FLTA", xcoor, ycoor, zcoor,idrotm[3], "ONLY");
  TVirtualMC::GetMC()->Gspos("FWZ3", 2, "FLTA", xcoor, ycoor,-zcoor,idrotm[2], "ONLY");

  if (mTOFHoles) {
    //xcoor = 0.;
    //ycoor = (yFLT - Geo::LENGTHEXINMODBORDER)*0.5;
    zcoor = -Geo::EXTERINTERMODBORDER1 + (zlenA*0.5 + Geo::INTERCENTRMODBORDER1 - 2.*Geo::MODULEWALLTHICKNESS)*0.5;
    TVirtualMC::GetMC()->Gspos("FWZ3", 5, "FLTB", xcoor, ycoor, zcoor, idrotm[2], "ONLY");
    TVirtualMC::GetMC()->Gspos("FWZ3", 6, "FLTC", xcoor, ycoor,-zcoor, idrotm[3], "ONLY");
  }

  //xcoor = 0.;
  ycoor = -(yFLT - Geo::LENGTHEXINMODBORDER)*0.5;
  zcoor = Geo::EXTERINTERMODBORDER2;
  TVirtualMC::GetMC()->Gspos("FWZ3", 3, "FLTA", xcoor, ycoor, zcoor, idrotm[1], "ONLY");
  TVirtualMC::GetMC()->Gspos("FWZ3", 4, "FLTA", xcoor, ycoor,-zcoor, idrotm[0], "ONLY");

  if (mTOFHoles) {
    //xcoor = 0.;
    //ycoor = -(yFLT - Geo::LENGTHEXINMODBORDER)*0.5;
    zcoor = -Geo::EXTERINTERMODBORDER2 + (zlenA*0.5 + Geo::INTERCENTRMODBORDER1 - 2.*Geo::MODULEWALLTHICKNESS)*0.5;
    TVirtualMC::GetMC()->Gspos("FWZ3", 7, "FLTB", xcoor, ycoor, zcoor, idrotm[0], "ONLY");
    TVirtualMC::GetMC()->Gspos("FWZ3", 8, "FLTC", xcoor, ycoor,-zcoor, idrotm[1], "ONLY");
  }

  trpa[0] = 0.5*(Geo::EXTERINTERMODBORDER2 - Geo::EXTERINTERMODBORDER1)/TMath::Cos(alpha);
  trpa[1] = 2.*Geo::MODULEWALLTHICKNESS;
  trpa[2] = xFLT*0.5;
  trpa[3] = -beta*TMath::RadToDeg();
  trpa[4] = 0.;
  trpa[5] = 0.;
  TVirtualMC::GetMC()->Gsvolu("FWZ4", "PARA", getMedium(kFiberGlass), trpa, 6); // Fibre glass

  Matrix (idrotm[6],alpha*TMath::RadToDeg(),90.,90.+alpha*TMath::RadToDeg(),90.,90.,180.);
  Matrix (idrotm[7],180.-alpha*TMath::RadToDeg(),90.,90.-alpha*TMath::RadToDeg(),90.,90.,0.);

  //xcoor = 0.;
  ycoor = 0.;
  zcoor = (Geo::EXTERINTERMODBORDER2 + Geo::EXTERINTERMODBORDER1)*0.5;
  TVirtualMC::GetMC()->Gspos("FWZ4", 1, "FLTA", xcoor, ycoor, zcoor, idrotm[7], "ONLY");
  TVirtualMC::GetMC()->Gspos("FWZ4", 2, "FLTA", xcoor, ycoor,-zcoor, idrotm[6], "ONLY");

  if (mTOFHoles) {
    //xcoor = 0.;
    //ycoor = 0.;
    zcoor = -(Geo::EXTERINTERMODBORDER2 + Geo::EXTERINTERMODBORDER1)*0.5 +
      (zlenA*0.5 + Geo::INTERCENTRMODBORDER1 - 2.*Geo::MODULEWALLTHICKNESS)*0.5;
    TVirtualMC::GetMC()->Gspos("FWZ4", 3, "FLTB", xcoor, ycoor, zcoor, idrotm[6], "ONLY");
    TVirtualMC::GetMC()->Gspos("FWZ4", 4, "FLTC", xcoor, ycoor,-zcoor, idrotm[7], "ONLY");
  }
}

void Detector::MakeStripsInModules(Float_t ytof, Float_t zlenA) const
{
  //
  // Define MRPC strip volume, called FSTR
  // Insert FSTR volume in FLTA/B/C volumes
  //
  //ciao
  Float_t yFLT  = ytof*0.5 - Geo::MODULEWALLTHICKNESS;

  ///////////////// Detector itself //////////////////////

  const Int_t    knx   = Geo::NPADX;  // number of pads along x
  const Int_t    knz   = Geo::NPADZ;  // number of pads along z
  const Float_t  kPadX = Geo::XPAD;   // pad length along x
  const Float_t  kPadZ = Geo::ZPAD;   // pad length along z

  // new description for strip volume -double stack strip-
  // -- all constants are expressed in cm
  // height of different layers
  const Float_t khhony   = 1.0;       // height of HONY Layer
  const Float_t khpcby   = 0.08;      // height of PCB Layer
  const Float_t khrgly   = 0.055;     // height of RED GLASS Layer

  const Float_t khfiliy  = 0.125;     // height of FISHLINE Layer
  const Float_t khglassy = 0.160*0.5; // semi-height of GLASS Layer
  const Float_t khglfy   = khfiliy+2.*khglassy; // height of GLASS Layer

  const Float_t khcpcby  = 0.16;      // height of PCB  Central Layer
  const Float_t kwhonz   = 8.1;       // z dimension of HONEY Layer
  const Float_t kwpcbz1  = 10.64;     // z dimension of PCB Lower Layer
  const Float_t kwpcbz2  = 11.6;      // z dimension of PCB Upper Layer
  const Float_t kwcpcbz  = 12.4;      // z dimension of PCB Central Layer

  const Float_t kwrglz   = 8.;        // z dimension of RED GLASS Layer
  const Float_t kwglfz   = 7.;        // z dimension of GLASS Layer
  const Float_t klsensmx = knx*kPadX; // length of Sensitive Layer
  const Float_t khsensmy = 0.0105;    // height of Sensitive Layer
  const Float_t kwsensmz = knz*kPadZ; // width of Sensitive Layer

  // height of the FSTR Volume (the strip volume)
  const Float_t khstripy = 2.*khhony+2.*khpcby+4.*khrgly+2.*khglfy+khcpcby;

  // width  of the FSTR Volume (the strip volume)
  const Float_t kwstripz = kwcpcbz;
  // length of the FSTR Volume (the strip volume)
  const Float_t klstripx = Geo::STRIPLENGTH;

  // FSTR volume definition-filling this volume with non sensitive Gas Mixture
  Float_t parfp[3]={static_cast<Float_t>(klstripx*0.5), static_cast<Float_t>(khstripy*0.5), static_cast<Float_t>(kwstripz*0.5)};
  TVirtualMC::GetMC()->Gsvolu("FSTR", "BOX", getMedium(kFre), parfp, 3); // Freon mix

  Float_t posfp[3]={0.,0.,0.};

  // NOMEX (HONEYCOMB) Layer definition
  //parfp[0] = klstripx*0.5;
  parfp[1] = khhony*0.5;
  parfp[2] = kwhonz*0.5;
  TVirtualMC::GetMC()->Gsvolu("FHON", "BOX", getMedium(kNomex), parfp, 3); // Nomex (Honeycomb)
  // positioning 2 NOMEX Layers on FSTR volume
  //posfp[0] = 0.;
  posfp[1] =-khstripy*0.5 + parfp[1];
  //posfp[2] = 0.;
  TVirtualMC::GetMC()->Gspos("FHON", 1, "FSTR", 0., posfp[1], 0., 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("FHON", 2, "FSTR", 0.,-posfp[1], 0., 0, "ONLY");
  
  // Lower PCB Layer definition
  //parfp[0] = klstripx*0.5;
  parfp[1] = khpcby*0.5;
  parfp[2] = kwpcbz1*0.5;
  TVirtualMC::GetMC()->Gsvolu("FPC1", "BOX", getMedium(kG10), parfp, 3); // G10

  // Upper PCB Layer definition
  //parfp[0] = klstripx*0.5;
  //parfp[1] = khpcby*0.5;
  parfp[2] = kwpcbz2*0.5;
  TVirtualMC::GetMC()->Gsvolu("FPC2", "BOX", getMedium(kG10), parfp, 3); // G10

  // positioning 2 external PCB Layers in FSTR volume
  //posfp[0] = 0.;
  posfp[1] =-khstripy*0.5+khhony+parfp[1];
  //posfp[2] = 0.;
  TVirtualMC::GetMC()->Gspos("FPC1", 1, "FSTR", 0.,-posfp[1], 0., 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("FPC2", 1, "FSTR", 0., posfp[1], 0., 0, "ONLY");

  // Central PCB layer definition
  //parfp[0] = klstripx*0.5;
  parfp[1] = khcpcby*0.5;
  parfp[2] = kwcpcbz*0.5;
  TVirtualMC::GetMC()->Gsvolu("FPCB", "BOX", getMedium(kG10), parfp, 3); // G10
  gGeoManager->GetVolume("FPCB")->VisibleDaughters(kFALSE);

  // positioning the central PCB layer
  TVirtualMC::GetMC()->Gspos("FPCB", 1, "FSTR", 0., 0., 0., 0, "ONLY");

  // Sensitive volume definition
  Float_t parfs[3] = {static_cast<Float_t>(klsensmx*0.5), static_cast<Float_t>(khsensmy*0.5), static_cast<Float_t>(kwsensmz*0.5)};
  TVirtualMC::GetMC()->Gsvolu("FSEN", "BOX", getMedium(kCuS), parfs, 3); // Cu sensitive
  // dividing FSEN along z in knz=2 and along x in knx=48
  TVirtualMC::GetMC()->Gsdvn("FSEZ", "FSEN", knz, 3);
  TVirtualMC::GetMC()->Gsdvn("FPAD", "FSEZ", knx, 1);
  // positioning sensitive layer inside FPCB
  TVirtualMC::GetMC()->Gspos("FSEN", 1, "FPCB", 0., 0., 0., 0, "ONLY");

  // RED GLASS Layer definition
  //parfp[0] = klstripx*0.5;
  parfp[1] = khrgly*0.5;
  parfp[2] = kwrglz*0.5;
  TVirtualMC::GetMC()->Gsvolu("FRGL", "BOX", getMedium(kGlass), parfp, 3); // red glass
  // positioning 4 RED GLASS Layers in FSTR volume
  //posfp[0] = 0.;
  posfp[1] = -khstripy*0.5+khhony+khpcby+parfp[1];
  //posfp[2] = 0.;
  TVirtualMC::GetMC()->Gspos("FRGL", 1, "FSTR", 0., posfp[1], 0., 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("FRGL", 4, "FSTR", 0.,-posfp[1], 0., 0, "ONLY");
  //posfp[0] = 0.;
  posfp[1] = (khcpcby+khrgly)*0.5;
  //posfp[2] = 0.;
  TVirtualMC::GetMC()->Gspos("FRGL", 2, "FSTR", 0.,-posfp[1], 0., 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("FRGL", 3, "FSTR", 0., posfp[1], 0., 0, "ONLY");

  // GLASS Layer definition
  //parfp[0] = klstripx*0.5;
  parfp[1] = khglassy;
  parfp[2] = kwglfz*0.5;
  TVirtualMC::GetMC()->Gsvolu("FGLF", "BOX", getMedium(kGlass), parfp, 3); // glass
  // positioning 2 GLASS Layers in FSTR volume
  //posfp[0] = 0.;
  posfp[1] = (khcpcby + khglfy)*0.5 + khrgly;
  //posfp[2] = 0.;
  TVirtualMC::GetMC()->Gspos("FGLF", 1, "FSTR", 0.,-posfp[1], 0., 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("FGLF", 2, "FSTR", 0., posfp[1], 0., 0, "ONLY");

  // Positioning the Strips (FSTR volumes) in the FLT volumes
  Int_t maxStripNumbers [5] ={Geo::NSTRIPC,
			      Geo::NSTRIPB,
			      Geo::NSTRIPA,
			      Geo::NSTRIPB,
			      Geo::NSTRIPC};

  Int_t idrotm[Geo::NSTRIPXSECTOR];
  for (Int_t ii=0; ii<Geo::NSTRIPXSECTOR; ii++) idrotm[ii]=0;

  Int_t totalStrip = 0;
  Float_t xpos, zpos, ypos, ang;
  for(Int_t iplate = 0; iplate < Geo::NPLATES; iplate++){
    if (iplate>0) totalStrip += maxStripNumbers[iplate-1];
    for(Int_t istrip = 0; istrip < maxStripNumbers[iplate]; istrip++){

      ang = Geo::GetAngles(iplate,istrip);
 
      if (ang>0.)       Matrix (idrotm[istrip+totalStrip],90.,0.,90.+ang,90., ang, 90.);
      else if (ang==0.) Matrix (idrotm[istrip+totalStrip],90.,0.,90.,90., 0., 0.);
      else if (ang<0.)  Matrix (idrotm[istrip+totalStrip],90.,0.,90.+ang,90.,-ang,270.);

      xpos = 0.;
      ypos = Geo::GetHeights(iplate,istrip) + yFLT*0.5;
      zpos = Geo::GetDistances(iplate,istrip);
      TVirtualMC::GetMC()->Gspos("FSTR", istrip+totalStrip+1, "FLTA", xpos, ypos,-zpos, idrotm[istrip+totalStrip], "ONLY");

      if (mTOFHoles) {
	if (istrip+totalStrip+1>53)
	  TVirtualMC::GetMC()->Gspos("FSTR", istrip+totalStrip+1, "FLTC", xpos, ypos,-zpos-(zlenA*0.5 - 2.*Geo::MODULEWALLTHICKNESS + Geo::INTERCENTRMODBORDER1)*0.5, idrotm[istrip+totalStrip], "ONLY");
	if (istrip+totalStrip+1<39)
	  TVirtualMC::GetMC()->Gspos("FSTR", istrip+totalStrip+1, "FLTB", xpos, ypos,-zpos+(zlenA*0.5 - 2.*Geo::MODULEWALLTHICKNESS + Geo::INTERCENTRMODBORDER1)*0.5, idrotm[istrip+totalStrip], "ONLY");
      }
    }
  }

}

void Detector::CreateModuleCovers(Float_t xtof, Float_t zlenA) const
{
  //
  // Create covers for module:
  //   per each module zone, defined according to
  //   fgkInterCentrModBorder2, fgkExterInterModBorder1 and zlenA+2 values,
  //   there is a frame of thickness 2cm in Al
  //   and the contained zones in honeycomb of Al.
  //   There is also an interface layer (1.6mm thichness)
  //   and plastic and Cu corresponding to the flat cables.
  //

  Float_t par[3];
  par[0] = xtof*0.5 + 2.;
  par[1] = Geo::MODULECOVERTHICKNESS*0.5;
  par[2] = zlenA*0.5 + 2.;
  TVirtualMC::GetMC()->Gsvolu("FPEA", "BOX ", getMedium(kAir), par, 3); // Air
  if (mTOFHoles) TVirtualMC::GetMC()->Gsvolu("FPEB", "BOX ", getMedium(kAir), par, 3); // Air

  const Float_t kAlCoverThickness = 1.5;
  const Float_t kInterfaceCardThickness = 0.16;
  const Float_t kAlSkinThickness = 0.1;

  //par[0] = xtof*0.5 + 2.;
  par[1] = kAlCoverThickness*0.5;
  //par[2] = zlenA*0.5 + 2.;
  TVirtualMC::GetMC()->Gsvolu("FALT", "BOX ", getMedium(kAlFrame), par, 3); // Al
  if (mTOFHoles) TVirtualMC::GetMC()->Gsvolu("FALB", "BOX ", getMedium(kAlFrame), par, 3); // Al
  Float_t  xcoor, ycoor, zcoor;
  xcoor = 0.;
  ycoor = 0.;
  zcoor = 0.;
  TVirtualMC::GetMC()->Gspos("FALT", 0, "FPEA", xcoor, ycoor, zcoor, 0, "ONLY");
  if (mTOFHoles) TVirtualMC::GetMC()->Gspos("FALB", 0, "FPEB", xcoor, ycoor, zcoor, 0, "ONLY");

  par[0] = xtof*0.5;
  //par[1] = kAlCoverThickness*0.5;
  par[2] = Geo::INTERCENTRMODBORDER2 - 2.;
  TVirtualMC::GetMC()->Gsvolu("FPE1", "BOX ", getMedium(kHoneycomb), par, 3); // Al honeycomb
  //xcoor = 0.;
  //ycoor = 0.;
  //zcoor = 0.;
  TVirtualMC::GetMC()->Gspos("FPE1", 0, "FALT", xcoor, ycoor, zcoor, 0, "ONLY");

  if (mTOFHoles) {
    //par[0] = xtof*0.5;
    par[1] = kAlCoverThickness*0.5 - kAlSkinThickness;
    //par[2] = Geo::INTERCENTRMODBORDER2 - 2.;
    TVirtualMC::GetMC()->Gsvolu("FPE4", "BOX ", getMedium(kHoneyHoles), par, 3); // Al honeycomb for holes
    //xcoor = 0.;
    //ycoor = 0.;
    //zcoor = 0.;
    TVirtualMC::GetMC()->Gspos("FPE4", 0, "FALB", xcoor, ycoor, zcoor, 0, "ONLY");
  }

  //par[0] = xtof*0.5;
  //par[1] = kAlCoverThickness*0.5;
  par[2] = (Geo::EXTERINTERMODBORDER1 - Geo::INTERCENTRMODBORDER2)*0.5 - 2.;
  TVirtualMC::GetMC()->Gsvolu("FPE2", "BOX ", getMedium(kHoneycomb), par, 3); // Al honeycomb
  //xcoor = 0.;
  //ycoor = 0.;
  zcoor = (Geo::EXTERINTERMODBORDER1 + Geo::INTERCENTRMODBORDER2)*0.5;
  TVirtualMC::GetMC()->Gspos("FPE2", 1, "FALT", xcoor, ycoor, zcoor, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("FPE2", 2, "FALT", xcoor, ycoor,-zcoor, 0, "ONLY");

  if (mTOFHoles) {
    //xcoor = 0.;
    //ycoor = 0.;
    //zcoor = (Geo::EXTERINTERMODBORDER1 + Geo::INTERCENTRMODBORDER2)*0.5;
    TVirtualMC::GetMC()->Gspos("FPE2", 1, "FALB", xcoor, ycoor, zcoor, 0, "ONLY");
    TVirtualMC::GetMC()->Gspos("FPE2", 2, "FALB", xcoor, ycoor,-zcoor, 0, "ONLY");
  }

  //par[0] = xtof*0.5;
  //par[1] = kAlCoverThickness*0.5;
  par[2] = (zlenA*0.5 + 2. - Geo::EXTERINTERMODBORDER1)*0.5 - 2.;
  TVirtualMC::GetMC()->Gsvolu("FPE3", "BOX ", getMedium(kHoneycomb), par, 3); // Al honeycomb
  //xcoor = 0.;
  //ycoor = 0.;
  zcoor = (zlenA*0.5 + 2. + Geo::EXTERINTERMODBORDER1)*0.5;
  TVirtualMC::GetMC()->Gspos("FPE3", 1, "FALT", xcoor, ycoor, zcoor, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("FPE3", 2, "FALT", xcoor, ycoor,-zcoor, 0, "ONLY");

  if (mTOFHoles) {
    //xcoor = 0.;
    //ycoor = 0.;
    zcoor = (zlenA*0.5 + 2. + Geo::EXTERINTERMODBORDER1)*0.5;
    TVirtualMC::GetMC()->Gspos("FPE3", 1, "FALB", xcoor, ycoor, zcoor, 0, "ONLY");
    TVirtualMC::GetMC()->Gspos("FPE3", 2, "FALB", xcoor, ycoor,-zcoor, 0, "ONLY");
  }

  // volumes for Interface cards
  par[0] = xtof*0.5;
  par[1] = kInterfaceCardThickness*0.5;
  par[2] = Geo::INTERCENTRMODBORDER2 - 2.;
  TVirtualMC::GetMC()->Gsvolu("FIF1", "BOX ", getMedium(kG10), par, 3); // G10
  //xcoor = 0.;
  ycoor = kAlCoverThickness*0.5 + kInterfaceCardThickness*0.5;
  zcoor = 0.;
  TVirtualMC::GetMC()->Gspos("FIF1", 0, "FPEA", xcoor, ycoor, zcoor, 0, "ONLY");

  //par[0] = xtof*0.5;
  //par[1] = kInterfaceCardThickness*0.5;
  par[2] = (Geo::EXTERINTERMODBORDER1 - Geo::INTERCENTRMODBORDER2)*0.5 - 2.;
  TVirtualMC::GetMC()->Gsvolu("FIF2", "BOX ", getMedium(kG10), par, 3); // G10
  //xcoor = 0.;
  //ycoor = kAlCoverThickness*0.5 + kInterfaceCardThickness*0.5;
  zcoor = (Geo::EXTERINTERMODBORDER1 + Geo::INTERCENTRMODBORDER2)*0.5;
  TVirtualMC::GetMC()->Gspos("FIF2", 1, "FPEA", xcoor, ycoor, zcoor, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("FIF2", 2, "FPEA", xcoor, ycoor,-zcoor, 0, "ONLY");
  if (mTOFHoles) {
    TVirtualMC::GetMC()->Gspos("FIF2", 1, "FPEB", xcoor, ycoor, zcoor, 0, "ONLY");
    TVirtualMC::GetMC()->Gspos("FIF2", 2, "FPEB", xcoor, ycoor,-zcoor, 0, "ONLY");
  }

  //par[0] = xtof*0.5;
  //par[1] = kInterfaceCardThickness*0.5;
  par[2] = (zlenA*0.5 + 2. - Geo::EXTERINTERMODBORDER1)*0.5 - 2.;
  TVirtualMC::GetMC()->Gsvolu("FIF3", "BOX ", getMedium(kG10), par, 3); // G10
  //xcoor = 0.;
  //ycoor = kAlCoverThickness*0.5 + kInterfaceCardThickness*0.5;
  zcoor = (zlenA*0.5 + 2. + Geo::EXTERINTERMODBORDER1)*0.5;
  TVirtualMC::GetMC()->Gspos("FIF3", 1, "FPEA", xcoor, ycoor, zcoor, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("FIF3", 2, "FPEA", xcoor, ycoor,-zcoor, 0, "ONLY");
  if (mTOFHoles) {
    TVirtualMC::GetMC()->Gspos("FIF3", 1, "FPEB", xcoor, ycoor, zcoor, 0, "ONLY");
    TVirtualMC::GetMC()->Gspos("FIF3", 2, "FPEB", xcoor, ycoor,-zcoor, 0, "ONLY");
  }

  // volumes for flat cables
  // plastic
  const Float_t kPlasticFlatCableThickness = 0.25;
  par[0] = xtof*0.5;
  par[1] = kPlasticFlatCableThickness*0.5;
  par[2] = Geo::INTERCENTRMODBORDER2 - 2.;
  TVirtualMC::GetMC()->Gsvolu("FFC1", "BOX ", getMedium(kPlastic), par, 3); // Plastic (CH2)
  //xcoor = 0.;
  ycoor = -kAlCoverThickness*0.5 - kPlasticFlatCableThickness*0.5;
  zcoor = 0.;
  TVirtualMC::GetMC()->Gspos("FFC1", 0, "FPEA", xcoor, ycoor, zcoor, 0, "ONLY");

  //par[0] = xtof*0.5;
  //par[1] = kPlasticFlatCableThickness*0.5;
  par[2] = (Geo::EXTERINTERMODBORDER1 - Geo::INTERCENTRMODBORDER2)*0.5 - 2.;
  TVirtualMC::GetMC()->Gsvolu("FFC2", "BOX ", getMedium(kPlastic), par, 3); // Plastic (CH2)
  //xcoor = 0.;
  //ycoor = -kAlCoverThickness*0.5 - kPlasticFlatCableThickness*0.5;
  zcoor = (Geo::EXTERINTERMODBORDER1 + Geo::INTERCENTRMODBORDER2)*0.5;
  TVirtualMC::GetMC()->Gspos("FFC2", 1, "FPEA", xcoor, ycoor, zcoor, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("FFC2", 2, "FPEA", xcoor, ycoor,-zcoor, 0, "ONLY");
  if (mTOFHoles) {
    TVirtualMC::GetMC()->Gspos("FFC2", 1, "FPEB", xcoor, ycoor, zcoor, 0, "ONLY");
    TVirtualMC::GetMC()->Gspos("FFC2", 2, "FPEB", xcoor, ycoor,-zcoor, 0, "ONLY");
  }

  //par[0] = xtof*0.5;
  //par[1] = kPlasticFlatCableThickness*0.5;
  par[2] = (zlenA*0.5 + 2. - Geo::EXTERINTERMODBORDER1)*0.5 - 2.;
  TVirtualMC::GetMC()->Gsvolu("FFC3", "BOX ", getMedium(kPlastic), par, 3); // Plastic (CH2)
  //xcoor = 0.;
  //ycoor = -kAlCoverThickness*0.5 - kPlasticFlatCableThickness*0.5;
  zcoor = (zlenA*0.5 + 2. + Geo::EXTERINTERMODBORDER1)*0.5;
  TVirtualMC::GetMC()->Gspos("FFC3", 1, "FPEA", xcoor, ycoor, zcoor, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("FFC3", 2, "FPEA", xcoor, ycoor,-zcoor, 0, "ONLY");
  if (mTOFHoles) {
    TVirtualMC::GetMC()->Gspos("FFC3", 1, "FPEB", xcoor, ycoor, zcoor, 0, "ONLY");
    TVirtualMC::GetMC()->Gspos("FFC3", 2, "FPEB", xcoor, ycoor,-zcoor, 0, "ONLY");
  }

  // Cu
  const Float_t kCopperFlatCableThickness = 0.01;
  par[0] = xtof*0.5;
  par[1] = kCopperFlatCableThickness*0.5;
  par[2] = Geo::INTERCENTRMODBORDER2 - 2.;
  TVirtualMC::GetMC()->Gsvolu("FCC1", "BOX ", getMedium(kCopper), par, 3); // Cu
  TVirtualMC::GetMC()->Gspos("FCC1", 0, "FFC1", 0., 0., 0., 0, "ONLY");

  //par[0] = xtof*0.5;
  //par[1] = kCopperFlatCableThickness*0.5;
  par[2] = (Geo::EXTERINTERMODBORDER1 - Geo::INTERCENTRMODBORDER2)*0.5 - 2.;
  TVirtualMC::GetMC()->Gsvolu("FCC2", "BOX ", getMedium(kCopper), par, 3); // Cu
  TVirtualMC::GetMC()->Gspos("FCC2", 0, "FFC2", 0., 0., 0., 0, "ONLY");

  //par[0] = xtof*0.5;
  //par[1] = kCopperFlatCableThickness*0.5;
  par[2] = (zlenA*0.5 + 2. - Geo::EXTERINTERMODBORDER1)*0.5 - 2.;
  TVirtualMC::GetMC()->Gsvolu("FCC3", "BOX ", getMedium(kCopper), par, 3); // Cu
  TVirtualMC::GetMC()->Gspos("FCC3", 0, "FFC3", 0., 0., 0., 0, "ONLY");
}

void Detector::CreateBackZone(Float_t xtof, Float_t ytof, Float_t zlenA) const
{
  //
  // Define:
  //        - containers for FEA cards, cooling system
  //          signal cables and supermodule support structure
  //          (volumes called FAIA/B/C),
  //        - containers for FEA cards and some cooling
  //          elements for a FEA (volumes called FCA1/2).
  //

  Int_t idrotm[1]={0};

  // Definition of the air card containers (FAIA, FAIC and FAIB)

  Float_t  par[3];
  par[0] = xtof*0.5;
  par[1] = (ytof*0.5 - Geo::MODULECOVERTHICKNESS)*0.5;
  par[2] = zlenA*0.5;
  TVirtualMC::GetMC()->Gsvolu("FAIA", "BOX ", getMedium(kAir), par, 3); // Air
  if (mTOFHoles) TVirtualMC::GetMC()->Gsvolu("FAIB", "BOX ", getMedium(kAir), par, 3); // Air
  TVirtualMC::GetMC()->Gsvolu("FAIC", "BOX ", getMedium(kAir), par, 3); // Air

  Float_t feaParam[3] = {Geo::FEAPARAMETERS[0], Geo::FEAPARAMETERS[1], Geo::FEAPARAMETERS[2]};
  Float_t feaRoof1[3] = {Geo::ROOF1PARAMETERS[0], Geo::ROOF1PARAMETERS[1], Geo::ROOF1PARAMETERS[2]};
  Float_t al3[3] = {Geo::AL3PARAMETERS[0], Geo::AL3PARAMETERS[1], Geo::AL3PARAMETERS[2]};
  //Float_t feaRoof2[3] = {Geo::ROOF2PARAMETERS[0], Geo::ROOF2PARAMETERS[1], Geo::ROOF2PARAMETERS[2]};

  // FEA card mother-volume definition
  Float_t carpar[3] = {static_cast<Float_t>(xtof*0.5 - Geo::CBLW - Geo::SAWTHICKNESS),
		       static_cast<Float_t>(feaParam[1] + feaRoof1[1] + Geo::ROOF2PARAMETERS[1]*0.5),
		       static_cast<Float_t>(feaRoof1[2] + Geo::BETWEENLANDMASK*0.5 + al3[2])};
  TVirtualMC::GetMC()->Gsvolu("FCA1", "BOX ", getMedium(kAir), carpar, 3); // Air
  TVirtualMC::GetMC()->Gsvolu("FCA2", "BOX ", getMedium(kAir), carpar, 3); // Air

  // rotation matrix
  Matrix(idrotm[0],  90.,180., 90., 90.,180., 0.);

  // FEA card mother-volume positioning
  Float_t rowstep = 6.66;
  Float_t rowgap[5] = {13.5, 22.9, 16.94, 23.8, 20.4};
  Int_t rowb[5] = {6, 7, 6, 19, 7};
  Float_t carpos[3] = {0.,
		       static_cast<Float_t>(-(ytof*0.5 - Geo::MODULECOVERTHICKNESS)*0.5 + carpar[1]),
		       -0.8};
  TVirtualMC::GetMC()->Gspos("FCA1", 91, "FAIA", carpos[0], carpos[1], carpos[2], 0, "MANY");
  TVirtualMC::GetMC()->Gspos("FCA2", 91, "FAIC", carpos[0], carpos[1], carpos[2], 0, "MANY");

  Int_t row = 1;
  Int_t nrow = 0;
  for (Int_t sg= -1; sg< 2; sg+= 2) {
    carpos[2] = sg*zlenA*0.5 - 0.8;
    for (Int_t nb=0; nb<5; ++nb) {
      carpos[2] = carpos[2] - sg*(rowgap[nb] - rowstep);
      nrow = row + rowb[nb];
      for ( ; row < nrow ; ++row) {

        carpos[2] -= sg*rowstep;

	if (nb==4) {
	  TVirtualMC::GetMC()->Gspos("FCA1", row, "FAIA", carpos[0], carpos[1], carpos[2], 0, "ONLY");
	  TVirtualMC::GetMC()->Gspos("FCA2", row, "FAIC", carpos[0], carpos[1], carpos[2], 0, "ONLY");

	}
	else {
	  switch (sg) {
	  case 1:
	    TVirtualMC::GetMC()->Gspos("FCA1", row, "FAIA", carpos[0], carpos[1], carpos[2], 0, "ONLY");
	    TVirtualMC::GetMC()->Gspos("FCA2", row, "FAIC", carpos[0], carpos[1], carpos[2], 0, "ONLY");
	    break;
	  case -1:
	    TVirtualMC::GetMC()->Gspos("FCA1", row, "FAIA", carpos[0], carpos[1], carpos[2], idrotm[0], "ONLY");
	    TVirtualMC::GetMC()->Gspos("FCA2", row, "FAIC", carpos[0], carpos[1], carpos[2], idrotm[0], "ONLY");
	    break;
	  }

	}

      }
    }
  }

  if (mTOFHoles) {
    row = 1;
    for (Int_t sg= -1; sg< 2; sg+= 2) {
      carpos[2] = sg*zlenA*0.5 - 0.8;
      for (Int_t nb=0; nb<4; ++nb) {
        carpos[2] = carpos[2] - sg*(rowgap[nb] - rowstep);
        nrow = row + rowb[nb];
        for ( ; row < nrow ; ++row) {
          carpos[2] -= sg*rowstep;

	  switch (sg) {
	  case 1:
	    TVirtualMC::GetMC()->Gspos("FCA1", row, "FAIB", carpos[0], carpos[1], carpos[2], 0, "ONLY");
	    break;
	  case -1:
	    TVirtualMC::GetMC()->Gspos("FCA1", row, "FAIB", carpos[0], carpos[1], carpos[2], idrotm[0], "ONLY");
	    break;
	  }
	}
      }
    }
  }

}



void Detector::MakeModulesInBTOFvolumes(Float_t ytof, Float_t zlenA) const
{
  //
  // Fill BTOF_%i (for i=0,...17) volumes
  // with volumes FTOA (MRPC strip container),
  // In case of TOF holes, three sectors (i.e. 13th, 14th and 15th)
  // are filled with volumes: FTOB and FTOC (MRPC containers),
  //

  const Int_t kSize=16;

  Int_t idrotm[1]={0};

  //AliMatrix(idrotm[0], 90.,  0., 0., 0., 90.,-90.);
  Matrix(idrotm[0], 90.,  0., 0., 0., 90.,270.);

  Float_t xcoor, ycoor, zcoor;
  xcoor = 0.;

  // Positioning of fibre glass modules (FTOA, FTOB and FTOC)
  for(Int_t isec=0; isec<Geo::NSECTORS; isec++){
    if(mTOFSectors[isec]==-1)continue;

    char name[kSize];
    snprintf(name, kSize, "BTOF%d",isec);
    if (mTOFHoles && (isec==13 || isec==14 || isec==15)) {
      //xcoor = 0.;
      ycoor = (zlenA*0.5 + Geo::INTERCENTRMODBORDER1)*0.5;
      zcoor = -ytof * 0.25;
      TVirtualMC::GetMC()->Gspos("FTOB", 0, name, xcoor, ycoor, zcoor, idrotm[0], "ONLY");
      TVirtualMC::GetMC()->Gspos("FTOC", 0, name, xcoor,-ycoor, zcoor, idrotm[0], "ONLY");
    }
    else {
      //xcoor = 0.;
      ycoor = 0.;
      zcoor = -ytof * 0.25;
      TVirtualMC::GetMC()->Gspos("FTOA", 0, name, xcoor, ycoor, zcoor, idrotm[0], "ONLY");
    }
  }

}


void Detector::AddAlignableVolumes() const
{
  //
  // Create entries for alignable volumes associating the symbolic volume
  // name with the corresponding volume path. Needs to be syncronized with
  // eventual changes in the geometry.
  //

  o2::Base::DetID::ID idTOF = o2::Base::DetID::TOF;
  Int_t modUID, modnum=0;

  TString volPath;
  TString symName;

  TString vpL0  = "ALIC_1/B077_1/BSEGMO";
  TString vpL1 = "_1/BTOF";
  TString vpL2 = "_1";
  TString vpL3 = "/FTOA_0";
  TString vpL4 = "/FLTA_0/FSTR_";

  TString snSM  = "TOF/sm";
  TString snSTRIP = "/strip";
  
  Int_t nSectors=Geo::NSECTORS;
  Int_t nStrips =Geo::NSTRIPXSECTOR;

  //
  // The TOF MRPC Strips
  // The symbolic names are: TOF/sm00/strip01
  //                           ...
  //                         TOF/sm17/strip91
 
  Int_t imod=0;

  for (Int_t isect = 0; isect < nSectors; isect++) {
    for (Int_t istr = 1; istr <= nStrips; istr++) {

      modUID = o2::Base::GeometryManager::getSensID(idTOF, modnum++);

      if (mTOFSectors[isect]==-1) continue;

      if (mTOFHoles && (isect==13 || isect==14 || isect==15)) {
	if (istr<39) {
	  vpL3 = "/FTOB_0";
	  vpL4 = "/FLTB_0/FSTR_";
	}
	else if (istr>53) {
	  vpL3 = "/FTOC_0";
	  vpL4 = "/FLTC_0/FSTR_";
	}
	else continue;
      }
      else {
	vpL3 = "/FTOA_0";
	vpL4 = "/FLTA_0/FSTR_";
      }

      volPath  = vpL0;
      volPath += isect;
      volPath += vpL1;
      volPath += isect;
      volPath += vpL2;
      volPath += vpL3;
      volPath += vpL4;
      volPath += istr;

      volPath = "";

      symName  = snSM;
      symName += Form("%02d",isect);
      symName += snSTRIP;
      symName += Form("%02d",istr);
            
      // AliDebug(2,"--------------------------------------------"); 
      // AliDebug(2,Form("Alignable object %d", imod)); 
      // AliDebug(2,Form("volPath=%s\n",volPath.Data()));
      // AliDebug(2,Form("symName=%s\n",symName.Data()));
      // AliDebug(2,"--------------------------------------------"); 

      printf("Check for alignable entry: %s\n",symName.Data());
	      
      if(!gGeoManager->SetAlignableEntry(symName.Data(),volPath.Data(),modUID))
	printf("Alignable entry %s not set\n",symName.Data());
	//	AliError(Form("Alignable entry %s not set",symName.Data()));

    }}
      /*
      //T2L matrices for alignment
      TGeoPNEntry *e = gGeoManager->GetAlignableEntryByUID(modUID);
      if (e) {
	TGeoHMatrix *globMatrix = e->GetGlobalOrig();
	Double_t phi = 20.0 * (isect % 18) + 10.0;
	TGeoHMatrix *t2l  = new TGeoHMatrix();
	t2l->RotateZ(phi);
	t2l->MultiplyLeft(&(globMatrix->Inverse()));
	e->SetMatrix(t2l);
      }
      else {
	AliError(Form("Alignable entry %s is not valid!",symName.Data()));
      }
      imod++;
    }
  }


  //
  // The TOF supermodules
  // The symbolic names are: TOF/sm00
  //                           ...
  //                         TOF/sm17
  //
  for (Int_t isect = 0; isect < nSectors; isect++) {

    volPath  = vpL0;
    volPath += isect;
    volPath += vpL1;
    volPath += isect;
    volPath += vpL2;

    symName  = snSM;
    symName += Form("%02d",isect);

    AliDebug(2,"--------------------------------------------"); 
    AliDebug(2,Form("Alignable object %d", isect+imod)); 
    AliDebug(2,Form("volPath=%s\n",volPath.Data()));
    AliDebug(2,Form("symName=%s\n",symName.Data()));
    AliDebug(2,"--------------------------------------------"); 

    gGeoManager->SetAlignableEntry(symName.Data(),volPath.Data());

  }
  */  
}
