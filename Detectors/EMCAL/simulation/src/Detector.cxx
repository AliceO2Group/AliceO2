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
#include "TVirtualMC.h"

#include "FairRootManager.h"
#include "FairVolume.h"

#include "EMCALBase/Geometry.h"
#include "EMCALBase/Hit.h"
#include "EMCALBase/ShishKebabTrd1Module.h"
#include "EMCALSimulation/Detector.h"
#include "EMCALSimulation/SpaceFrame.h"

using namespace o2::EMCAL;

ClassImp(Detector);

Detector::Detector(const char* Name, Bool_t Active)
  : o2::Base::Detector(Name, Active), mPointCollection(new TClonesArray("o2::EMCAL::Hit"))
{
}

void Detector::Initialize() {}

void Detector::ConstructGeometry()
{
  LOG(DEBUG) << "Creating EMCAL geometry\n";
  SpaceFrame emcalframe;
  emcalframe.CreateGeometry();
}

Bool_t Detector::ProcessHits(FairVolume* v) { return true; }

Hit* Detector::AddHit(Int_t shunt, Int_t trackID, Int_t parentID, Int_t primary, Double_t initialEnergy, Int_t detID,
                      const Point3D<float>& pos, const Vector3D<float>& mom, Double_t time, Double_t eLoss)
{
  TClonesArray& refCollection = *mPointCollection;
  Int_t size = refCollection.GetEntriesFast();
  return new (refCollection[size]) Hit(shunt, primary, trackID, parentID, detID, initialEnergy, pos, mom, time, eLoss);
}

void Detector::Register() { FairRootManager::Instance()->Register("EMCALHit", "EMCAL", mPointCollection, kTRUE); }

TClonesArray* Detector::GetCollection(Int_t iColl) const
{
  if (iColl == 0)
    return mPointCollection;
  return nullptr;
}

void Detector::Reset() {}

Geometry* Detector::GetGeometry()
{
  if (!mGeometry) {
    mGeometry = Geometry::GetInstanceFromRunNumber(223409);
  }
  return mGeometry;
}

void Detector::CreateShiskebabGeometry()
{
  // TRD1
  Geometry* g = GetGeometry();
  TString gn(g->GetName());
  gn.ToUpper();
  Double_t trd1Angle = g->GetTrd1Angle() * TMath::DegToRad(), tanTrd1 = TMath::Tan(trd1Angle / 2.);
  // see AliModule::fFIdTmedArr
  //  fIdTmedArr = fIdtmed->GetArray() - 1599 ; // see AliEMCAL::::CreateMaterials()
  //  Int_t kIdAIR=1599, kIdPB = 1600, kIdSC = 1601, kIdSTEEL = 1603;
  //  idAL = 1602;
  Double_t par[10], xpos = 0., ypos = 0., zpos = 0.;

  CreateSmod(g->GetNameOfEMCALEnvelope());

  Int_t* SMTypeList = g->GetEMCSystem();
  Int_t tmpType = -1;
  for (Int_t i = 0; i < g->GetNumberOfSuperModules(); i++) {
    if (SMTypeList[i] == tmpType)
      continue;
    else
      tmpType = SMTypeList[i];

    if (tmpType == Geometry::EMCAL_STANDARD)
      CreateEmod("SMOD", "EMOD"); // 18-may-05
    else if (tmpType == Geometry::EMCAL_HALF)
      CreateEmod("SM10", "EMOD"); // Nov 1,2006 1/2 SM
    else if (tmpType == Geometry::EMCAL_THIRD)
      CreateEmod("SM3rd", "EMOD"); // Feb 1,2012 1/3 SM
    else if (tmpType == Geometry::DCAL_STANDARD)
      CreateEmod("DCSM", "EMOD"); // Mar 13, 2012, 6 or 10 DCSM
    else if (tmpType == Geometry::DCAL_EXT)
      CreateEmod("DCEXT", "EMOD"); // Mar 13, 2012, DCAL extension SM
    else
      LOG(ERROR) << "Unkown SM Type!!\n";
  }

  // Sensitive SC  (2x2 tiles)
  Double_t parSCM0[5] = { 0, 0, 0, 0 }, *dummy = 0, parTRAP[11];
  if (!gn.Contains("V1")) {
    Double_t wallThickness = g->GetPhiModuleSize() / g->GetNPHIdiv() - g->GetPhiTileSize();
    for (Int_t i = 0; i < 3; i++)
      parSCM0[i] = mParEMOD[i] - wallThickness;
    parSCM0[3] = mParEMOD[3];
    TVirtualMC::GetMC()->Gsvolu("SCM0", "TRD1", mIdTmedArr[ID_AIR], parSCM0, 4);
    TVirtualMC::GetMC()->Gspos("SCM0", 1, "EMOD", 0., 0., 0., 0, "ONLY");
  } else {
    Double_t wTh = g->GetLateralSteelStrip();
    parSCM0[0] = mParEMOD[0] - wTh + tanTrd1 * g->GetTrd1AlFrontThick();
    parSCM0[1] = mParEMOD[1] - wTh;
    parSCM0[2] = mParEMOD[2] - wTh;
    parSCM0[3] = mParEMOD[3] - g->GetTrd1AlFrontThick() / 2.;
    TVirtualMC::GetMC()->Gsvolu("SCM0", "TRD1", mIdTmedArr[ID_AIR], parSCM0, 4);
    Double_t zshift = g->GetTrd1AlFrontThick() / 2.;
    TVirtualMC::GetMC()->Gspos("SCM0", 1, "EMOD", 0., 0., zshift, 0, "ONLY");
    //
    CreateAlFrontPlate("EMOD", "ALFP");
  }

  if (g->GetNPHIdiv() == 2 && g->GetNETAdiv() == 2) {
    // Division to tile size - 1-oct-04
    LOG(DEBUG2) << " Divide SCM0 on y-axis " << g->GetNETAdiv() << FairLogger::endl;
    TVirtualMC::GetMC()->Gsdvn("SCMY", "SCM0", g->GetNETAdiv(), 2); // y-axis

    // Trapesoid 2x2
    parTRAP[0] = parSCM0[3];                                                                         // dz
    parTRAP[1] = TMath::ATan2((parSCM0[1] - parSCM0[0]) / 2., 2. * parSCM0[3]) * 180. / TMath::Pi(); // theta
    parTRAP[2] = 0.;                                                                                 // phi

    // bottom
    parTRAP[3] = parSCM0[2] / 2.; // H1
    parTRAP[4] = parSCM0[0] / 2.; // BL1
    parTRAP[5] = parTRAP[4];      // TL1
    parTRAP[6] = 0.0;             // ALP1

    // top
    parTRAP[7] = parSCM0[2] / 2.; // H2
    parTRAP[8] = parSCM0[1] / 2.; // BL2
    parTRAP[9] = parTRAP[8];      // TL2
    parTRAP[10] = 0.0;            // ALP2

    LOG(DEBUG2) << " ** TRAP ** \n";
    for (Int_t i = 0; i < 11; i++)
      LOG(DEBUG3) << " par[" << std::setw(2) << std::setprecision(2) << i << "] " << std::setw(9)
                  << std::setprecision(4) << parTRAP[i] << FairLogger::endl;

    TVirtualMC::GetMC()->Gsvolu("SCMX", "TRAP", mIdTmedArr[ID_SC], parTRAP, 11);
    xpos = +(parSCM0[1] + parSCM0[0]) / 4.;
    TVirtualMC::GetMC()->Gspos("SCMX", 1, "SCMY", xpos, 0.0, 0.0, 0, "ONLY");

    // Using rotation because SCMX should be the same due to Pb tiles
    xpos = -xpos;
    Matrix(mIdRotm, 90.0, 180., 90.0, 270.0, 0.0, 0.0);
    TVirtualMC::GetMC()->Gspos("SCMX", 2, "SCMY", xpos, 0.0, 0.0, mIdRotm, "ONLY");

    // put LED to the SCM0
    ShishKebabTrd1Module* mod = static_cast<ShishKebabTrd1Module*>(mShishKebabModules->At(0));
    Double_t tanBetta = mod->GetTanBetta();

    Int_t nr = 0;
    ypos = 0.0;
    Double_t xCenterSCMX = (parTRAP[4] + parTRAP[8]) / 2.;
    if (!gn.Contains("V1")) {
      par[1] = parSCM0[2] / 2;            // y
      par[2] = g->GetECPbRadThick() / 2.; // z
      TVirtualMC::GetMC()->Gsvolu("PBTI", "BOX", mIdTmedArr[ID_PB], dummy, 0);

      zpos = -mSampleWidth * g->GetNECLayers() / 2. + g->GetECPbRadThick() / 2.;
      LOG(DEBUG2) << " Pb tiles \n";

      for (Int_t iz = 0; iz < g->GetNECLayers(); iz++) {
        par[0] = (parSCM0[0] + tanBetta * mSampleWidth * iz) / 2.;
        xpos = par[0] - xCenterSCMX;
        TVirtualMC::GetMC()->Gsposp("PBTI", ++nr, "SCMX", xpos, ypos, zpos, 0, "ONLY", par, 3);
        LOG(DEBUG3) << iz + 1 << " xpos " << xpos << " zpos " << zpos << " par[0] " << par[0] << FairLogger::endl;
        zpos += mSampleWidth;
      }

      LOG(DEBUG2) << " Number of Pb tiles in SCMX " << nr << FairLogger::endl;
    } else {
      // Oct 26, 2010
      // First sheet of paper
      par[1] = parSCM0[2] / 2.;                 // y
      par[2] = g->GetTrd1BondPaperThick() / 2.; // z
      par[0] = parSCM0[0] / 2.;                 // x
      TVirtualMC::GetMC()->Gsvolu("PAP1", "BOX", mIdTmedArr[ID_PAPER], par, 3);

      xpos = par[0] - xCenterSCMX;
      zpos = -parSCM0[3] + g->GetTrd1BondPaperThick() / 2.;
      TVirtualMC::GetMC()->Gspos("PAP1", 1, "SCMX", xpos, ypos, zpos, 0, "ONLY");

      for (Int_t iz = 0; iz < g->GetNECLayers() - 1; iz++) {
        nr = iz + 1;
        Double_t dz = g->GetECScintThick() + g->GetTrd1BondPaperThick() + mSampleWidth * iz;

        // PB + 2 paper sheets
        par[2] = g->GetECPbRadThick() / 2. + g->GetTrd1BondPaperThick(); // z
        par[0] = (parSCM0[0] + tanBetta * dz) / 2.;
        TString pa(Form("PA%2.2i", nr));
        TVirtualMC::GetMC()->Gsvolu(pa.Data(), "BOX", mIdTmedArr[ID_PAPER], par, 3);

        xpos = par[0] - xCenterSCMX;
        zpos = -parSCM0[3] + dz + par[2];
        TVirtualMC::GetMC()->Gspos(pa.Data(), 1, "SCMX", xpos, ypos, zpos, 0, "ONLY");

        // Pb
        TString pb(Form("PB%2.2i", nr));
        par[2] = g->GetECPbRadThick() / 2.; // z
        TVirtualMC::GetMC()->Gsvolu(pb.Data(), "BOX", mIdTmedArr[ID_PB], par, 3);
        TVirtualMC::GetMC()->Gspos(pb.Data(), 1, pa.Data(), 0.0, 0.0, 0.0, 0, "ONLY");
      }
    }

  }
  //
  // Remove? Next too cases seem early prototype studies
  //
  else if (g->GetNPHIdiv() == 3 && g->GetNETAdiv() == 3) {
    printf(" before AliEMCALv0::Trd1Tower3X3() : parSCM0");
    for (Int_t i = 0; i < 4; i++)
      printf(" %7.4f ", parSCM0[i]);
    printf("\n");
    Trd1Tower3X3(parSCM0);
  } else if (g->GetNPHIdiv() == 1 && g->GetNETAdiv() == 1) {
    // no division in SCM0
    Trd1Tower1X1(parSCM0);
  }
}

void Detector::CreateMaterials()
{
  // media number in idtmed are 1599 to 1698.
  // --- Air ---
  Float_t aAir[4] = { 12.0107, 14.0067, 15.9994, 39.948 };
  Float_t zAir[4] = { 6., 7., 8., 18. };
  Float_t wAir[4] = { 0.000124, 0.755267, 0.231781, 0.012827 };
  Float_t dAir = 1.20479E-3;
  Mixture(0, "Air$", aAir, zAir, dAir, 4, wAir);

  // --- Lead ---
  Material(1, "Pb$", 207.2, 82, 11.35, 0.56, 0., 0, 0);

  // --- The polysterene scintillator (CH) ---
  Float_t aP[2] = { 12.011, 1.00794 };
  Float_t zP[2] = { 6.0, 1.0 };
  Float_t wP[2] = { 1.0, 1.0 };
  Float_t dP = 1.032;

  Mixture(2, "Polystyrene$", aP, zP, dP, -2, wP);

  // --- Aluminium ---
  Material(3, "Al$", 26.98, 13., 2.7, 8.9, 999., 0, 0);
  // ---         Absorption length is ignored ^

  // 25-aug-04 by PAI - see  PMD/AliPMDv0.cxx for STEEL definition
  Float_t asteel[4] = { 55.847, 51.9961, 58.6934, 28.0855 };
  Float_t zsteel[4] = { 26., 24., 28., 14. };
  Float_t wsteel[4] = { .715, .18, .1, .005 };
  Mixture(4, "STAINLESS STEEL$", asteel, zsteel, 7.88, 4, wsteel);

  // Oct 26,2010 : Multipurpose Copy Paper UNV-21200), weiht 75 g/m**2.
  // *Cellulose C6H10O5
  //    Component C  A=12.01   Z=6.    W=6./21.
  //    Component H  A=1.      Z=1.    W=10./21.
  //    Component O  A=16.     Z=8.    W=5./21.
  Float_t apaper[3] = { 12.01, 1.0, 16.0 };
  Float_t zpaper[3] = { 6.0, 1.0, 8.0 };
  Float_t wpaper[3] = { 6. / 21., 10. / 21., 5. / 21. };
  Mixture(5, "BondPaper$", apaper, zpaper, 0.75, 3, wpaper);

  // DEFINITION OF THE TRACKING MEDIA
  // Look to the $ALICE_ROOT/data/galice.cuts for particular values
  // of cuts.
  // Don't forget to add a new tracking medium with non-default cuts

  // for EMCAL: idtmed[1599->1698] equivalent to fIdtmed[0->100]
  // From TPCsimulation/src/Detector.cxx:
  // until we solve the problem of reading the field from files with changed class names we
  //  need to hard code some values here to be able to run the macros  M.Al-Turany (Nov.14)
  Int_t isxfld = 2;
  Float_t sxmgmx = 10.0;

  // Air                                                                         -> idtmed[1599]
  Medium(0, "Air$", 0, 0, isxfld, sxmgmx, 10.0, 1.0, 0.1, 0.1, 10.0, 0, 0);

  // The Lead                                                                      -> idtmed[1600]

  Medium(1, "Lead$", 1, 0, isxfld, sxmgmx, 10.0, 0.1, 0.1, 0.1, 0.1, 0, 0);

  // The scintillator of the CPV made of Polystyrene scintillator                   -> idtmed[1601]
  float deemax = 0.1; // maximum fractional energy loss in one step (0 < DEEMAX < deemax )
  Medium(2, "Scintillator$", 2, 1, isxfld, sxmgmx, 10.0, 0.001, deemax, 0.001, 0.001, 0, 0);

  // Various Aluminium parts made of Al                                            -> idtmed[1602]
  Medium(3, "Al$", 3, 0, isxfld, sxmgmx, 10.0, 0.1, 0.1, 0.001, 0.001, 0, 0);

  // 25-aug-04 by PAI : see  PMD/AliPMDv0.cxx for STEEL definition                 -> idtmed[1603]
  Medium(4, "S steel$", 4, 0, isxfld, sxmgmx, 10.0, 0.01, 0.1, 0.001, 0.001, 0, 0);

  // Oct 26,2010; Nov 24,2010                                                      -> idtmed[1604]
  deemax = 0.01;
  Medium(5, "Paper$", 5, 0, isxfld, sxmgmx, 10.0, deemax, 0.1, 0.001, 0.001, 0, 0);

  // Set constants for Birk's Law implentation
  mBirkC0 = 1;
  mBirkC1 = 0.013 / dP;
  mBirkC2 = 9.6e-6 / (dP * dP);
}

void Detector::CreateSmod(const char* mother)
{
  // 18-may-05; mother="XEN1";
  // child="SMOD" from first to 10th, "SM10" (11th and 12th)
  // "DCSM" from 13th to 18/22th (TRD1 case), "DCEXT"(18th and 19th)  adapted for DCAL, Oct-23-2012
  Geometry* g = GetGeometry();
  TString gn(g->GetName());
  gn.ToUpper();

  Double_t par[3], xpos = 0., ypos = 0., zpos = 0., rpos = 0., dphi = 0., phi = 0.0, phiRad = 0.;
  Double_t parC[3] = { 0 };
  TString smName;
  Int_t tmpType = -1;

  //  ===== define Super Module from air - 14x30 module ==== ;
  LOG(DEBUG2) << "\n ## Super Module | fSampleWidth " << std::setw(5) << std::setprecision(3) << mSampleWidth << " ## "
              << gn << FairLogger::endl;
  par[0] = g->GetShellThickness() / 2.;
  par[1] = g->GetPhiModuleSize() * g->GetNPhi() / 2.;
  par[2] = g->GetEtaModuleSize() * g->GetNEta() / 2.;
  mIdRotm = 0;

  Int_t nSMod = g->GetNumberOfSuperModules();
  Int_t nphism = nSMod / 2; // 20-may-05
  if (nphism > 0) {
    dphi = g->GetEMCGeometry()->GetPhiSuperModule();
    rpos = (g->GetEnvelop(0) + g->GetEnvelop(1)) / 2.;
    LOG(DEBUG2) << " rpos " << std::setw(8) << std::setprecision(2) << rpos << " : dphi " << std::setw(6)
                << std::setprecision(1) << dphi << " degree \n";
  }

  if (gn.Contains("WSUC")) {
    Int_t nr = 0;
    par[0] = g->GetPhiModuleSize() * g->GetNPhi() / 2.;
    par[1] = g->GetShellThickness() / 2.;
    par[2] = g->GetEtaModuleSize() * g->GetNZ() / 2. + 5;

    TVirtualMC::GetMC()->Gsvolu("SMOD", "BOX", mIdTmedArr[ID_AIR], par, 3);

    LOG(DEBUG2) << "SMOD in WSUC : tmed " << mIdTmedArr[ID_AIR] << " | dx " << std::setw(7) << std::setprecision(2)
                << par[0] << " dy " << std::setw(7) << std::setprecision(2) << par[1] << " dz " << std::setw(7)
                << std::setprecision(2) << par[2] << " (SMOD, BOX)\n";
    mSmodPar0 = par[0];
    mSmodPar1 = par[1];
    mSmodPar2 = par[2];
    nphism = g->GetNumberOfSuperModules();
    for (Int_t i = 0; i < nphism; i++) {
      xpos = ypos = zpos = 0.0;
      mIdRotm = 0;
      TVirtualMC::GetMC()->Gspos("SMOD", 1, mother, xpos, ypos, zpos, mIdRotm, "ONLY");

      printf(" fIdRotm %3i phi %6.1f(%5.3f) xpos %7.2f ypos %7.2f zpos %7.2f \n", mIdRotm, phi, phiRad, xpos, ypos,
             zpos);

      nr++;
    }
  } else { // ALICE
    LOG(DEBUG2) << " par[0] " << std::setw(7) << std::setprecision(2) << par[0] << " (old) \n";
    for (Int_t i = 0; i < 3; i++)
      par[i] = g->GetSuperModulesPar(i);
    mSmodPar0 = par[0];
    mSmodPar2 = par[2];

    Int_t SMOrder = -1;
    tmpType = -1;
    for (Int_t smodnum = 0; smodnum < nSMod; ++smodnum) {
      for (Int_t i = 0; i < 3; i++)
        parC[i] = par[i];
      if (g->GetSMType(smodnum) == tmpType) {
        SMOrder++;
      } else {
        tmpType = g->GetSMType(smodnum);
        SMOrder = 1;
      }

      phiRad = g->GetPhiCenterOfSMSec(smodnum); // NEED  phi= 90, 110, 130, 150, 170, 190(not center)...
      phi = phiRad * 180. / TMath::Pi();
      Double_t phiy = 90. + phi;
      Double_t phiz = 0.;

      xpos = rpos * TMath::Cos(phiRad);
      ypos = rpos * TMath::Sin(phiRad);
      zpos = mSmodPar2; // 21-sep-04
      if (tmpType == Geometry::EMCAL_STANDARD) {
        smName = "SMOD";
      } else if (tmpType == Geometry::EMCAL_HALF) {
        smName = "SM10";
        parC[1] /= 2.;
        xpos += (par[1] / 2. * TMath::Sin(phiRad));
        ypos -= (par[1] / 2. * TMath::Cos(phiRad));
      } else if (tmpType == Geometry::EMCAL_THIRD) {
        smName = "SM3rd";
        parC[1] /= 3.;
        xpos += (2. * par[1] / 3. * TMath::Sin(phiRad));
        ypos -= (2. * par[1] / 3. * TMath::Cos(phiRad));
      } else if (tmpType == Geometry::DCAL_STANDARD) {
        smName = "DCSM";
        parC[2] *= 2. / 3.;
        zpos = mSmodPar2 + g->GetDCALInnerEdge() / 2.; // 21-sep-04
      } else if (tmpType == Geometry::DCAL_EXT) {
        smName = "DCEXT";
        parC[1] /= 3.;
        xpos += (2. * par[1] / 3. * TMath::Sin(phiRad));
        ypos -= (2. * par[1] / 3. * TMath::Cos(phiRad));
      } else
        LOG(ERROR) << "Unkown SM Type!!\n";

      if (SMOrder == 1) { // first time, create the SM
        TVirtualMC::GetMC()->Gsvolu(smName.Data(), "BOX", mIdTmedArr[ID_AIR], parC, 3);

        LOG(DEBUG2) << " Super module with name \"" << smName << "\" was created in \"box\" with: par[0] = " << parC[0]
                    << ", par[1] = " << parC[1] << ", par[2] = " << parC[2] << FairLogger::endl;
      }

      if (smodnum % 2 == 1) {
        phiy += 180.;
        if (phiy >= 360.)
          phiy -= 360.;
        phiz = 180.;
        zpos *= -1.;
      }

      Matrix(mIdRotm, 90.0, phi, 90.0, phiy, phiz, 0.0);
      TVirtualMC::GetMC()->Gspos(smName.Data(), SMOrder, mother, xpos, ypos, zpos, mIdRotm, "ONLY");

      LOG(DEBUG3) << smName << " : " << std::setw(2) << SMOrder << ", fIdRotm " << std::setw(3) << mIdRotm << " phi "
                  << std::setw(6) << std::setprecision(1) << phi << "(" << std::setw(5) << std::setprecision(3)
                  << phiRad << ") xpos " << std::setw(7) << std::setprecision(2) << xpos << " ypos " << std::setw(7)
                  << std::setprecision(2) << ypos << " zpos " << std::setw(7) << std::setprecision(2) << zpos << " : i "
                  << smodnum << FairLogger::endl;
    }
  }

  LOG(DEBUG2) << " Number of Super Modules " << nSMod << FairLogger::endl;

  // Steel plate
  if (g->GetSteelFrontThickness() > 0.0) { // 28-mar-05
    par[0] = g->GetSteelFrontThickness() / 2.;
    TVirtualMC::GetMC()->Gsvolu("STPL", "BOX", mIdTmedArr[ID_STEEL], par, 3);

    LOG(DEBUG1) << "tmed " << mIdTmedArr[ID_STEEL] << " | dx " << std::setw(7) << std::setprecision(2) << par[0]
                << " dy " << std::setw(7) << std::setprecision(2) << par[1] << " dz " << std::setw(7)
                << std::setprecision(2) << par[2] << " (STPL) \n";

    xpos = -(g->GetShellThickness() - g->GetSteelFrontThickness()) / 2.;
    TVirtualMC::GetMC()->Gspos("STPL", 1, "SMOD", xpos, 0.0, 0.0, 0, "ONLY");
  }
}

void Detector::CreateEmod(const char* mother, const char* child)
{
  // 17-may-05; mother="SMOD"; child="EMOD"
  // Oct 26,2010
  Geometry* g = GetGeometry();
  TString gn(g->GetName());
  gn.ToUpper();

  // Module definition
  Double_t xpos = 0., ypos = 0., zpos = 0.;
  // Double_t trd1Angle = g->GetTrd1Angle()*TMath::DegToRad();tanTrd1 = TMath::Tan(trd1Angle/2.);

  if (strcmp(mother, "SMOD") == 0) {
    mParEMOD[0] = g->GetEtaModuleSize() / 2.; // dx1
    mParEMOD[1] = g->Get2Trd1Dx2() / 2.;      // dx2
    mParEMOD[2] = g->GetPhiModuleSize() / 2.;
    ;                                          // dy
    mParEMOD[3] = g->GetLongModuleSize() / 2.; // dz
    TVirtualMC::GetMC()->Gsvolu(child, "TRD1", mIdTmedArr[ID_STEEL], mParEMOD, 4);
  }

  Int_t nr = 0;
  mIdRotm = 0;
  // X->Z(0, 0); Y->Y(90, 90); Z->X(90, 0)
  ShishKebabTrd1Module* mod = 0; // current module

  for (Int_t iz = 0; iz < g->GetNZ(); iz++) {
    Double_t angle = 90., phiOK = 0;
    mod = static_cast<ShishKebabTrd1Module*>(mShishKebabModules->At(iz));
    angle = mod->GetThetaInDegree();

    if (!gn.Contains("WSUC")) { // ALICE
      Matrix(mIdRotm, 90. - angle, 180., 90.0, 90.0, angle, 0.);
      phiOK = mod->GetCenterOfModule().Phi() * 180. / TMath::Pi();
      LOG(DEBUG4) << std::setw(2) << iz + 1 << " | angle | " << std::setw(6) << std::setprecision(3) << angle << " - "
                  << std::setw(6) << std::setprecision(3) << phiOK << " = " << std::setw(6) << std::setprecision(3)
                  << angle - phiOK << "(eta " << std::setw(5) << std::setprecision(3) << mod->GetEtaOfCenterOfModule()
                  << ")\n";
      xpos = mod->GetPosXfromR() + g->GetSteelFrontThickness() - mSmodPar0;
      zpos = mod->GetPosZ() - mSmodPar2;

      Int_t iyMax = g->GetNPhi();
      if (strcmp(mother, "SM10") == 0) {
        iyMax /= 2;
      } else if (strcmp(mother, "SM3rd") == 0) {
        iyMax /= 3;
      } else if (strcmp(mother, "DCEXT") == 0) {
        iyMax /= 3;
      } else if (strcmp(mother, "DCSM") == 0) {
        if (iz < 8)
          continue; //!!!DCSM from 8th to 23th
        zpos = mod->GetPosZ() - mSmodPar2 - g->GetDCALInnerEdge() / 2.;
      } else if (strcmp(mother, "SMOD") != 0)
        LOG(ERROR) << "Unknown super module Type!!\n";

      for (Int_t iy = 0; iy < iyMax; iy++) { // flat in phi
        ypos = g->GetPhiModuleSize() * (2 * iy + 1 - iyMax) / 2.;
        TVirtualMC::GetMC()->Gspos(child, ++nr, mother, xpos, ypos, zpos, mIdRotm, "ONLY");

        // printf(" %2i xpos %7.2f ypos %7.2f zpos %7.2f fIdRotm %i\n", nr, xpos, ypos, zpos, fIdRotm);
        LOG(DEBUG3) << std::setw(3) << std::setprecision(3) << nr << "(" << std::setw(2) << std::setprecision(2)
                    << iy + 1 << "," << std::setw(2) << std::setprecision(2) << iz + 1 << ")\n";
      }
      // PH          printf("\n");
    } else { // WSUC
      if (iz == 0)
        Matrix(mIdRotm, 0., 0., 90., 0., 90., 90.); // (x')z; y'(x); z'(y)
      else
        Matrix(mIdRotm, 90 - angle, 270., 90.0, 0.0, angle, 90.);

      phiOK = mod->GetCenterOfModule().Phi() * 180. / TMath::Pi();

      LOG(DEBUG4) << std::setw(2) << iz + 1 << " | angle -phiOK | " << std::setw(6) << std::setprecision(3) << angle
                  << " - " << std::setw(6) << std::setprecision(3) << phiOK << " = " << std::setw(6)
                  << std::setprecision(3) << angle - phiOK << "(eta " << std::setw(5) << std::setprecision(3)
                  << mod->GetEtaOfCenterOfModule() << ")\n";

      zpos = mod->GetPosZ() - mSmodPar2;
      ypos = mod->GetPosXfromR() - mSmodPar1;

      // printf(" zpos %7.2f ypos %7.2f fIdRotm %i\n xpos ", zpos, xpos, fIdRotm);

      for (Int_t ix = 0; ix < g->GetNPhi(); ix++) { // flat in phi
        xpos = g->GetPhiModuleSize() * (2 * ix + 1 - g->GetNPhi()) / 2.;
        TVirtualMC::GetMC()->Gspos(child, ++nr, mother, xpos, ypos, zpos, mIdRotm, "ONLY");
        // printf(" %7.2f ", xpos);
      }
      // printf("\n");
    }
  }

  LOG(DEBUG2) << " Number of modules in Super Module(" << mother << ") " << nr << FairLogger::endl;
}

void Detector::CreateAlFrontPlate(const char* mother, const char* child)
{
  // Oct 26,2010 : Al front plate : ALFP
  Geometry* g = GetGeometry();

  TString gn(g->GetName());
  gn.ToUpper();
  Double_t trd1Angle = g->GetTrd1Angle() * TMath::DegToRad(), tanTrd1 = TMath::Tan(trd1Angle / 2.);
  Double_t parALFP[5], zposALFP = 0.;

  parALFP[0] = g->GetEtaModuleSize() / 2. - g->GetLateralSteelStrip(); // dx1
  parALFP[1] = parALFP[0] + tanTrd1 * g->GetTrd1AlFrontThick();        // dx2
  parALFP[2] = g->GetPhiModuleSize() / 2. - g->GetLateralSteelStrip(); // dy
  parALFP[3] = g->GetTrd1AlFrontThick() / 2.;                          // dz

  TVirtualMC::GetMC()->Gsvolu(child, "TRD1", mIdTmedArr[ID_AL], parALFP, 4);

  zposALFP = -mParEMOD[3] + g->GetTrd1AlFrontThick() / 2.;
  TVirtualMC::GetMC()->Gspos(child, 1, mother, 0.0, 0.0, zposALFP, 0, "ONLY");
}

void Detector::Trd1Tower1X1(Double_t* parSCM0)
{
  // Started Nov 22,2006 by PAI
  LOG(DEBUG1) << " o2::EMCAL::Detector::Trd1Tower1X1() : parSCM0\n";
  for (Int_t i = 0; i < 4; i++)
    LOG(INFO) << " " << std::setw(7) << std::setprecision(4) << parSCM0[i] << " ";
  LOG(INFO) << "\n";

  // No division - keeping the same volume logic
  // and as consequence the same abs is scheme
  LOG(DEBUG2) << "Trd1Tower1X1() : Create SCMX(SCMY) as SCM0\n";

  TVirtualMC::GetMC()->Gsvolu("SCMY", "TRD1", mIdTmedArr[ID_AIR], parSCM0, 4);
  TVirtualMC::GetMC()->Gspos("SCMY", 1, "SCM0", 0.0, 0.0, 0.0, 0, "ONLY");
  TVirtualMC::GetMC()->Gsvolu("SCMX", "TRD1", mIdTmedArr[ID_SC], parSCM0, 4);
  TVirtualMC::GetMC()->Gspos("SCMX", 1, "SCMY", 0.0, 0.0, 0.0, 0, "ONLY");

  // should be defined once
  Double_t* dummy = 0;
  TVirtualMC::GetMC()->Gsvolu("PBTI", "BOX", mIdTmedArr[ID_PB], dummy, 0);

  PbInTrd1(parSCM0, "SCMX");

  LOG(DEBUG1) << "Trd1Tower1X1() : Ver. 0.1 : was tested.\n";
}

void Detector::Trd1Tower3X3(const Double_t* parSCM0)
{
  // Started Dec 8,2004 by PAI
  // Fixed Nov 13,2006
  printf(" o2::EMCAL::Detector::Trd1Tower3X3() : parSCM0");
  for (Int_t i = 0; i < 4; i++)
    printf(" %7.4f ", parSCM0[i]);
  printf("\n");

  // Nov 10, 2006 - different name of SCMX
  Double_t parTRAP[11], *dummy = 0;
  Geometry* g = GetGeometry();

  TString gn(g->GetName()), scmx;
  gn.ToUpper();

  // Division to tile size
  LOG(DEBUG2) << "Trd1Tower3X3() : Divide SCM0 on y-axis " << g->GetNETAdiv() << FairLogger::endl;

  TVirtualMC::GetMC()->Gsdvn("SCMY", "SCM0", g->GetNETAdiv(), 2); // y-axis
  Double_t dx1 = parSCM0[0], dx2 = parSCM0[1], dy = parSCM0[2], dz = parSCM0[3];
  Double_t ndiv = 3., xpos = 0.0;

  // should be defined once
  TVirtualMC::GetMC()->Gsvolu("PBTI", "BOX", mIdTmedArr[ID_PB], dummy, 0);

  for (Int_t ix = 1; ix <= 3; ix++) { // 3X3
    scmx = "SCX";                     // Nov 10,2006
    // ix=1
    parTRAP[0] = dz;
    Double_t xCentBot = 2. * dx1 / 3.;
    Double_t xCentTop = 2. * (dx2 / 4. + dx1 / 12.);
    parTRAP[1] = TMath::ATan2((xCentTop - xCentBot), 2. * dz) * TMath::RadToDeg(); // theta
    parTRAP[2] = 0.;                                                               // phi
    // bottom
    parTRAP[3] = dy / ndiv;  // H1
    parTRAP[4] = dx1 / ndiv; // BL1
    parTRAP[5] = parTRAP[4]; // TL1
    parTRAP[6] = 0.0;        // ALP1
    // top
    parTRAP[7] = dy / ndiv;          // H2
    parTRAP[8] = dx2 / 2 - dx1 / 6.; // BL2
    parTRAP[9] = parTRAP[8];         // TL2
    parTRAP[10] = 0.0;               // ALP2
    xpos = (xCentBot + xCentTop) / 2.;

    if (ix == 3) {
      parTRAP[1] = -parTRAP[1];
      xpos = -xpos;
    } else if (ix == 2) { // central part is box but we treat as trapesoid due to numbering
      parTRAP[1] = 0.;
      parTRAP[8] = dx1 / ndiv; // BL2
      parTRAP[9] = parTRAP[8]; // TL2
      xpos = 0.0;
    }

    LOG(DEBUG2) << " ** TRAP ** xpos " << std::setw(9) << std::setprecision(3) << xpos << FairLogger::endl;
    for (Int_t i = 0; i < 11; i++)
      LOG(DEBUG2) << " par[" << std::setw(2) << std::setprecision(2) << i << "] " << std::setw(9)
                  << std::setprecision(4) << parTRAP[i] << FairLogger::endl;

    scmx += ix;
    TVirtualMC::GetMC()->Gsvolu(scmx.Data(), "TRAP", mIdTmedArr[ID_SC], parTRAP, 11);
    TVirtualMC::GetMC()->Gspos(scmx.Data(), 1, "SCMY", xpos, 0.0, 0.0, 0, "ONLY");

    PbInTrap(parTRAP, scmx);
  }

  LOG(DEBUG2) << "Trd1Tower3X3 - Ver. 1.0 : was tested.\n";
}

void Detector::PbInTrap(const Double_t parTRAP[11], TString n)
{
  // 8-dec-04 by PAI
  // see for example CreateShishKebabGeometry(); just for case TRD1
  Int_t nr = 0;
  LOG(DEBUG2) << " Pb tiles : nrstart " << nr << FairLogger::endl;
  Geometry* g = GetGeometry();

  Double_t par[3];
  Double_t xpos = 0.0, ypos = 0.0;
  Double_t zpos = -mSampleWidth * g->GetNECLayers() / 2. + g->GetECPbRadThick() / 2.;

  Double_t coef = (parTRAP[8] - parTRAP[4]) / (2. * parTRAP[0]);
  Double_t xCenterSCMX = (parTRAP[4] + parTRAP[8]) / 2.; // ??
  //  Double_t tan = TMath::Tan(parTRAP[1]*TMath::DegToRad());

  par[1] = parTRAP[3];                // y
  par[2] = g->GetECPbRadThick() / 2.; // z

  for (Int_t iz = 0; iz < g->GetNECLayers(); iz++) {
    par[0] = parTRAP[4] + coef * mSampleWidth * iz;
    xpos = par[0] - xCenterSCMX;
    if (parTRAP[1] < 0.)
      xpos = -xpos;

    TVirtualMC::GetMC()->Gsposp("PBTI", ++nr, n.Data(), xpos, ypos, zpos, 0, "ONLY", par, 3);

    LOG(DEBUG2) << iz + 1 << " xpos " << std::setw(9) << std::setprecision(3) << xpos << " zpos " << std::setw(9)
                << std::setprecision(3) << zpos << " par[0] " << std::setw(9) << std::setprecision(3) << par[0]
                << " |\n";

    zpos += mSampleWidth;
    if (iz % 2 > 0)
      LOG(DEBUG2) << "\n";
  }

  LOG(DEBUG2) << " Number of Pb tiles in SCMX " << nr << " coef " << std::setw(9) << std::setprecision(7) << coef
              << FairLogger::endl;
  LOG(DEBUG2) << " par[1] " << std::setw(9) << std::setprecision(3) << par[1] << " par[2] " << std::setw(9)
              << std::setprecision(3) << par[2] << " ypos " << std::setw(9) << std::setprecision(3) << ypos
              << FairLogger::endl;
  LOG(DEBUG2) << " PbInTrap Ver. 1.0 : was tested.\n";
}

void Detector::PbInTrd1(const Double_t* parTrd1, TString n)
{
  // see PbInTrap(const Double_t parTrd1[11], TString n)
  Int_t nr = 0;
  LOG(DEBUG2) << " Pb tiles : nrstart " << nr << FairLogger::endl;
  Geometry* g = GetGeometry();

  Double_t par[3];
  Double_t xpos = 0.0, ypos = 0.0;
  Double_t zpos = -mSampleWidth * g->GetNECLayers() / 2. + g->GetECPbRadThick() / 2.;
  Double_t coef = (parTrd1[1] - parTrd1[0]) / (2. * parTrd1[3]);

  par[1] = parTrd1[2];                // y
  par[2] = g->GetECPbRadThick() / 2.; // z

  for (Int_t iz = 0; iz < g->GetNECLayers(); iz++) {
    par[0] = parTrd1[0] + coef * mSampleWidth * iz;
    TVirtualMC::GetMC()->Gsposp("PBTI", ++nr, n.Data(), xpos, ypos, zpos, 0, "ONLY", par, 3);

    LOG(DEBUG2) << iz + 1 << " xpos " << std::setw(9) << std::setprecision(3) << xpos << " zpos " << std::setw(9)
                << std::setprecision(3) << zpos << " par[0] " << std::setw(9) << std::setprecision(3) << par[0]
                << "|\n";

    zpos += mSampleWidth;
    if (iz % 2 > 0)
      LOG(DEBUG2) << "\n";
  }

  LOG(DEBUG2) << " Number of Pb tiles in SCMX " << nr << " coef " << std::setw(9) << std::setprecision(7) << coef
              << FairLogger::endl;
  LOG(DEBUG2) << " PbInTrd1 Ver. 0.1 : was tested.\n";
}
