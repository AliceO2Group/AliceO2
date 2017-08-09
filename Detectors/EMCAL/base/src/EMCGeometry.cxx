// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <algorithm>
#include <cassert>
#include <functional>
#include <iomanip>
#include <iostream>

#include <TObjArray.h>
#include <TObjString.h>
#include <TRegexp.h>

#include <boost/algorithm/string/predicate.hpp>

#include "EMCALBase/EMCGeometry.h"

using namespace o2::EMCAL;

Bool_t EMCGeometry::sInit = kFALSE;
std::string EMCGeometry::sDefaultGeometryName = "EMCAL_COMPLETE12SMV1_DCAL_8SM";

EMCGeometry::EMCGeometry(std::string_view name, std::string_view mcname, std::string_view mctitle)
  : mGeoName(name),
    mArrayOpts(nullptr),
    mNAdditionalOpts(0),
    mECPbRadThickness(0.),
    mECScintThick(0.),
    mNECLayers(0),
    mArm1PhiMin(0.),
    mArm1PhiMax(0.),
    mArm1EtaMin(0.),
    mArm1EtaMax(0.),
    mIPDistance(0.),
    mShellThickness(0.),
    mZLength(0.),
    mDCALInnerEdge(0.),
    mDCALPhiMin(0),
    mDCALPhiMax(0),
    mEMCALPhiMax(0),
    mDCALStandardPhiMax(0),
    mDCALInnerExtandedEta(0),
    mNZ(0),
    mNPhi(0),
    mSampling(0.),
    mNumberOfSuperModules(0),
    mEMCSMSystem(nullptr),
    mFrontSteelStrip(0.),
    mLateralSteelStrip(0.),
    mPassiveScintThick(0.),
    mPhiModuleSize(0.),
    mEtaModuleSize(0.),
    mPhiTileSize(0.),
    mEtaTileSize(0.),
    mLongModuleSize(0.),
    mPhiSuperModule(0),
    mNPhiSuperModule(0),
    mNPHIdiv(0),
    mNETAdiv(0),
    mNCells(0),
    mNCellsInSupMod(0),
    mNCellsInModule(0),
    mTrd1Angle(0.),
    m2Trd1Dx2(0.),
    mPhiGapForSM(0.),
    mKey110DEG(0),
    mnSupModInDCAL(0),
    mPhiBoundariesOfSM(0),
    mPhiCentersOfSM(0),
    mPhiCentersOfSMSec(0),
    mEtaMaxOfTRD1(0),
    mTrd1AlFrontThick(0.0),
    mTrd1BondPaperThick(0.),
    mCentersOfCellsEtaDir(0),
    mCentersOfCellsXDir(0),
    mCentersOfCellsPhiDir(0),
    mEtaCentersOfCells(0),
    mPhiCentersOfCells(0),
    mParSM(),
    mILOSS(-1),
    mIHADR(-1),
    mSteelFrontThick(0.) // obsolete data member?
{
  LOG(DEBUG2) << "EMCGeometry" << name << "," << mcname << "," << mctitle << "\n";

  Init(mcname, mctitle);
}

EMCGeometry::EMCGeometry(const EMCGeometry& geom)
  : mGeoName(geom.mGeoName),
    mArrayOpts(geom.mArrayOpts),
    mNAdditionalOpts(geom.mNAdditionalOpts),
    mECPbRadThickness(geom.mECPbRadThickness),
    mECScintThick(geom.mECScintThick),
    mNECLayers(geom.mNECLayers),
    mArm1PhiMin(geom.mArm1PhiMin),
    mArm1PhiMax(geom.mArm1PhiMax),
    mArm1EtaMin(geom.mArm1EtaMin),
    mArm1EtaMax(geom.mArm1EtaMax),
    mIPDistance(geom.mIPDistance),
    mShellThickness(geom.mShellThickness),
    mZLength(geom.mZLength),
    mDCALInnerEdge(geom.mDCALInnerEdge),
    mDCALPhiMin(geom.mDCALPhiMin),
    mDCALPhiMax(geom.mDCALPhiMax),
    mEMCALPhiMax(geom.mEMCALPhiMax),
    mDCALStandardPhiMax(geom.mDCALStandardPhiMax),
    mDCALInnerExtandedEta(geom.mDCALInnerExtandedEta),
    mNZ(geom.mNZ),
    mNPhi(geom.mNPhi),
    mSampling(geom.mSampling),
    mNumberOfSuperModules(geom.mNumberOfSuperModules),
    mEMCSMSystem(new Int_t[mNumberOfSuperModules]),
    mFrontSteelStrip(geom.mFrontSteelStrip),
    mLateralSteelStrip(geom.mLateralSteelStrip),
    mPassiveScintThick(geom.mPassiveScintThick),
    mPhiModuleSize(geom.mPhiModuleSize),
    mEtaModuleSize(geom.mEtaModuleSize),
    mPhiTileSize(geom.mPhiTileSize),
    mEtaTileSize(geom.mEtaTileSize),
    mLongModuleSize(geom.mLongModuleSize),
    mPhiSuperModule(geom.mPhiSuperModule),
    mNPhiSuperModule(geom.mNPhiSuperModule),
    mNPHIdiv(geom.mNPHIdiv),
    mNETAdiv(geom.mNETAdiv),
    mNCells(geom.mNCells),
    mNCellsInSupMod(geom.mNCellsInSupMod),
    mNCellsInModule(geom.mNCellsInModule),
    mTrd1Angle(geom.mTrd1Angle),
    m2Trd1Dx2(geom.m2Trd1Dx2),
    mPhiGapForSM(geom.mPhiGapForSM),
    mKey110DEG(geom.mKey110DEG),
    mnSupModInDCAL(geom.mnSupModInDCAL),
    mPhiBoundariesOfSM(geom.mPhiBoundariesOfSM),
    mPhiCentersOfSM(geom.mPhiCentersOfSM),
    mPhiCentersOfSMSec(geom.mPhiCentersOfSMSec),
    mEtaMaxOfTRD1(geom.mEtaMaxOfTRD1),
    mTrd1AlFrontThick(geom.mTrd1AlFrontThick),
    mTrd1BondPaperThick(geom.mTrd1BondPaperThick),
    mCentersOfCellsEtaDir(geom.mCentersOfCellsEtaDir),
    mCentersOfCellsXDir(geom.mCentersOfCellsXDir),
    mCentersOfCellsPhiDir(geom.mCentersOfCellsPhiDir),
    mEtaCentersOfCells(geom.mEtaCentersOfCells),
    mPhiCentersOfCells(geom.mPhiCentersOfCells),
    mILOSS(geom.mILOSS),
    mIHADR(geom.mIHADR),
    mSteelFrontThick(geom.mSteelFrontThick) // obsolete data member?
{
  memcpy(mEMCSMSystem, geom.mEMCSMSystem, sizeof(Int_t) * mNumberOfSuperModules);
  memcpy(mParSM, geom.mParSM, sizeof(Float_t) * 3);
  memcpy(mEnvelop, geom.mEnvelop, sizeof(Float_t) * 3);

  for (Int_t i = 0; i < 6; i++)
    mAdditionalOpts[i] = geom.mAdditionalOpts[i];
}

EMCGeometry::~EMCGeometry()
{
  delete[] mEMCSMSystem; // was created with new[], note the brackets
  // TODO, FIXME Hans, Aug 2015: Shouldn't one add
  // if(fArrayOpts){fArrayOpts->Delete();delete fArrayOpts;}
  // End Hans, Aug 2015
}

void EMCGeometry::Init(std::string_view mcname, std::string_view mctitle)
{
  using boost::algorithm::contains;
  mAdditionalOpts[0] = "nl=";       // number of sampling layers (fNECLayers)
  mAdditionalOpts[1] = "pbTh=";     // cm, Thickness of the Pb   (fECPbRadThick)
  mAdditionalOpts[2] = "scTh=";     // cm, Thickness of the Sc    (fECScintThick)
  mAdditionalOpts[3] = "latSS=";    // cm, Thickness of lateral steel strip (fLateralSteelStrip)
  mAdditionalOpts[4] = "allILOSS="; // = 0,1,2,3,4 (4 - energy loss without fluctuation)
  mAdditionalOpts[5] = "allIHADR="; // = 0,1,2 (0 - no hadronic interaction)

  mNAdditionalOpts = sizeof(mAdditionalOpts) / sizeof(char*);

  // geometry
  sInit = kFALSE; // Assume failed until proven otherwise.
  std::transform(mGeoName.begin(), mGeoName.end(), mGeoName.begin(), ::toupper);

  // Convert old geometry names to new ones
  if (contains(mGeoName, "SHISH_77_TRD1_2X2_FINAL_110DEG")) {
    if (contains(mGeoName, "PBTH=0.144") && contains(mGeoName, "SCTH=0.176")) {
      mGeoName = "EMCAL_COMPLETE";
    } else {
      mGeoName = "EMCAL_PDC06";
    }
  }

  if (contains(mGeoName, "WSUC"))
    mGeoName = "EMCAL_WSUC";

  // check that we have a valid geometry name
  if (!(contains(mGeoName, "EMCAL_PDC06") || contains(mGeoName, "EMCAL_WSUC") || contains(mGeoName, "EMCAL_COMPLETE") ||
        contains(mGeoName, "EMCAL_COMPLETEV1") || contains(mGeoName, "EMCAL_COMPLETE12SMV1") ||
        contains(mGeoName, "EMCAL_FIRSTYEAR") || contains(mGeoName, "EMCAL_FIRSTYEARV1"))) {
    LOG(FATAL) << "Init, " << mGeoName << " is an undefined geometry!\n";
  }

  // Option to know whether we have the "half" supermodule(s) or not
  mKey110DEG = 0;
  if (contains(mGeoName, "COMPLETE") || contains(mGeoName, "PDC06") || contains(mGeoName, "12SM"))
    mKey110DEG = 1; // for GetAbsCellId
  if (contains(mGeoName, "COMPLETEV1"))
    mKey110DEG = 0;

  mnSupModInDCAL = 0;
  if (contains(mGeoName, "DCAL_DEV")) {
    mnSupModInDCAL = 10;
  } else if (contains(mGeoName, "DCAL_8SM")) {
    mnSupModInDCAL = 8;
  } else if (contains(mGeoName, "DCAL")) {
    mnSupModInDCAL = 6;
  }

  // JLK 13-Apr-2008
  // default parameters are those of EMCAL_COMPLETE geometry
  // all others render variations from these at the end of
  // geometry-name specific options

  mNumberOfSuperModules = 12; // 12 = 6 * 2 (6 in phi, 2 in Z)
  mNPhi = 12;                 // module granularity in phi within smod (azimuth)
  mNZ = 24;                   // module granularity along Z within smod (eta)
  mNPHIdiv = mNETAdiv = 2;    // tower granularity within module
  mArm1PhiMin = 80.0;         // degrees, Starting EMCAL Phi position
  mArm1PhiMax = 200.0;        // degrees, Ending EMCAL Phi position
  mArm1EtaMin = -0.7;         // pseudorapidity, Starting EMCAL Eta position
  mArm1EtaMax = +0.7;         // pseudorapidity, Ending EMCAL Eta position
  mIPDistance = 428.0;        // cm, radial distance to front face from nominal vertex point
  mPhiGapForSM = 2.;          // cm, only for final TRD1 geometry
  mFrontSteelStrip = 0.025;   // 0.025cm = 0.25mm  (13-may-05 from V.Petrov)
  mPassiveScintThick = 0.8;   // 0.8cm   = 8mm     (13-may-05 from V.Petrov)
  mLateralSteelStrip = 0.01;  // 0.01cm  = 0.1mm   (13-may-05 from V.Petrov) - was 0.025
  mTrd1Angle = 1.5;           // in degrees

  mSampling = 1.;            // should be calculated with call to DefineSamplingFraction()
  mNECLayers = 77;           // (13-may-05 from V.Petrov) - can be changed with additional options
  mECScintThick = 0.176;     // scintillator layer thickness
  mECPbRadThickness = 0.144; // lead layer thickness

  mPhiModuleSize = 12.26 - mPhiGapForSM / Float_t(mNPhi); // first assumption
  mEtaModuleSize = mPhiModuleSize;

  mZLength = 700.;       // Z coverage (cm)
  mPhiSuperModule = 20.; // phi in degree
  mDCALInnerEdge = mIPDistance * TMath::Tan(mTrd1Angle * 8. * TMath::DegToRad());

  // needs to be called for each geometry and before setting geometry
  // parameters which can depend on the outcome
  CheckAdditionalOptions();

  // modifications to the above for PDC06 geometry
  if (contains(mGeoName, "PDC06")) {          // 18-may-05 - about common structure
    mECScintThick = mECPbRadThickness = 0.16; // (13-may-05 from V.Petrov)
    CheckAdditionalOptions();
  }

  // modifications to the above for WSUC geometry
  if (contains(mGeoName, "WSUC")) { // 18-may-05 - about common structure
    mNumberOfSuperModules = 2;      // 27-may-05; Nov 24,2010 for TB
    mNPhi = mNZ = 4;
    mTrd1AlFrontThick = 1.0; // one cm
    // Bond paper - two sheets around Sc tile
    mTrd1BondPaperThick = 0.01; // 0.01cm = 0.1 mm

    mPhiModuleSize = 12.0;
    mEtaModuleSize = mPhiModuleSize;
    mLateralSteelStrip = 0.015; // 0.015cm  = 0.15mm

    CheckAdditionalOptions();
  }

  // In 2009-2010 data taking runs only 4 SM, in the upper position.
  if (contains(mGeoName, "FIRSTYEAR")) {
    mNumberOfSuperModules = 4;
    mArm1PhiMax = 120.0;
    CheckAdditionalOptions();
  }

  if (contains(mGeoName, "FIRSTYEARV1") || contains(mGeoName, "COMPLETEV1") || contains(mGeoName, "COMPLETE12SMV1")) {
    // Oct 26,2010 : First module has tilt = 0.75 degree :
    // look to AliEMCALShishKebabTrd1Module::DefineFirstModule(key)
    // New sizes from production drawing, added Al front plate.
    // The thickness of sampling is change due to existing two sheets of paper.

    // Will replace fFrontSteelStrip
    mTrd1AlFrontThick = 1.0; // one cm
    // Bond paper - two sheets around Sc tile
    mTrd1BondPaperThick = 0.01; // 0.01cm = 0.1 mm

    mPhiModuleSize = 12.0;
    mEtaModuleSize = mPhiModuleSize;
    mLateralSteelStrip = 0.015; // 0.015cm  = 0.15mm

    if (contains(mGeoName, "COMPLETEV1")) {
      mNumberOfSuperModules = 10;
      mArm1PhiMax = 180.0;
    } else if (contains(mGeoName, "COMPLETE12SMV1")) {
      mNumberOfSuperModules = 12;
      mArm1PhiMax = 200.0;
    }
    if (contains(mGeoName, "DCAL")) {
      mNumberOfSuperModules = 12 + mnSupModInDCAL;
      mArm1PhiMax = 320.0;
      if (contains(mGeoName, "DCAL_8SM"))
        mArm1PhiMax = 340.0; // degrees, End of DCAL Phi position
      else if (contains(mGeoName, "DCAL_DEV"))
        mArm1PhiMin = 40.0; // degrees, Starting EMCAL(shifted) Phi position
      mDCALPhiMin = mArm1PhiMax - 10. * mnSupModInDCAL;
    }
    CheckAdditionalOptions();
  }

  //
  // Init EMCal/DCal SMs type array
  if (mEMCSMSystem)
    delete[] mEMCSMSystem;

  mEMCSMSystem = new Int_t[mNumberOfSuperModules];

  for (Int_t i = 0; i < mNumberOfSuperModules; i++)
    mEMCSMSystem[i] = NOT_EXISTENT;

  Int_t iSM = 0;

  //
  // BASIC EMCAL SM
  if (contains(mGeoName, "WSUC")) {
    for (int i = 0; i < 2; i++) {
      mEMCSMSystem[iSM] = EMCAL_STANDARD;
      iSM++;
    }
  } else if (contains(mGeoName, "FIRSTYEAR")) {
    for (int i = 0; i < 4; i++) {
      mEMCSMSystem[iSM] = EMCAL_STANDARD;
      iSM++;
    }
  } else if (contains(mGeoName, "PDC06") || contains(mGeoName, "COMPLETE")) {
    for (int i = 0; i < 10; i++) {
      mEMCSMSystem[iSM] = EMCAL_STANDARD;
      iSM++;
    }
  }

  //
  // EMCAL 110SM
  if (mKey110DEG && contains(mGeoName, "12SM")) {
    for (int i = 0; i < 2; i++) {
      mEMCSMSystem[iSM] = EMCAL_HALF;
      if (contains(mGeoName, "12SMV1")) {
        mEMCSMSystem[iSM] = EMCAL_THIRD;
      }
      iSM++;
    }
  }

  //
  // DCAL SM
  if (mnSupModInDCAL && contains(mGeoName, "DCAL")) {
    if (contains(mGeoName, "8SM")) {
      for (int i = 0; i < mnSupModInDCAL - 2; i++) {
        mEMCSMSystem[iSM] = DCAL_STANDARD;
        iSM++;
      }
      for (int i = 0; i < 2; i++) {
        mEMCSMSystem[iSM] = DCAL_EXT;
        iSM++;
      }
    } else {
      for (int i = 0; i < mnSupModInDCAL; i++) {
        mEMCSMSystem[iSM] = DCAL_STANDARD;
        iSM++;
      }
    }
  }

  // constant for transition absid <--> indexes
  mNCellsInModule = mNPHIdiv * mNETAdiv;
  mNCellsInSupMod = mNCellsInModule * mNPhi * mNZ;
  mNCells = 0;
  for (int i = 0; i < mNumberOfSuperModules; i++) {
    if (GetSMType(i) == EMCAL_STANDARD)
      mNCells += mNCellsInSupMod;
    else if (GetSMType(i) == EMCAL_HALF)
      mNCells += mNCellsInSupMod / 2;
    else if (GetSMType(i) == EMCAL_THIRD)
      mNCells += mNCellsInSupMod / 3;
    else if (GetSMType(i) == DCAL_STANDARD)
      mNCells += 2 * mNCellsInSupMod / 3;
    else if (GetSMType(i) == DCAL_EXT)
      mNCells += mNCellsInSupMod / 3;
    else
      LOG(ERROR) << "Uknown SuperModule Type !!\n";
  }

  mNPhiSuperModule = mNumberOfSuperModules / 2;
  if (mNPhiSuperModule < 1)
    mNPhiSuperModule = 1;

  mPhiTileSize = mPhiModuleSize / double(mNPHIdiv) - mLateralSteelStrip; // 13-may-05
  mEtaTileSize = mEtaModuleSize / double(mNETAdiv) - mLateralSteelStrip; // 13-may-05

  mLongModuleSize = mNECLayers * (mECScintThick + mECPbRadThickness);
  if (contains(mGeoName, "V1")) {
    Double_t ws = mECScintThick + mECPbRadThickness + 2. * mTrd1BondPaperThick; // sampling width
    // Number of Pb tiles = Number of Sc tiles - 1
    mLongModuleSize = mTrd1AlFrontThick + (ws * mNECLayers - mECPbRadThickness);
  }
  m2Trd1Dx2 = mEtaModuleSize + 2. * mLongModuleSize * TMath::Tan(mTrd1Angle * TMath::DegToRad() / 2.);

  if (!contains(mGeoName, "WSUC"))
    mShellThickness = TMath::Sqrt(mLongModuleSize * mLongModuleSize + m2Trd1Dx2 * m2Trd1Dx2);

  // These parameters are used to create the mother volume to hold the supermodules
  // 2cm padding added to allow for misalignments - JLK 30-May-2008
  mEnvelop[0] = mIPDistance - 1.;                   // mother volume inner radius
  mEnvelop[1] = mIPDistance + mShellThickness + 1.; // mother volume outer r.
  mEnvelop[2] = mZLength + 2.;                      // mother volume length

  // Local coordinates
  mParSM[0] = GetShellThickness() / 2.;
  mParSM[1] = GetPhiModuleSize() * GetNPhi() / 2.;
  mParSM[2] = mZLength / 4.; // divide by 4 to get half-length of SM

  // SM phi boundaries - (0,1),(2,3) ... - has the same boundaries;
  mPhiBoundariesOfSM.Set(mNumberOfSuperModules);
  mPhiCentersOfSM.Set(mNumberOfSuperModules / 2);
  mPhiCentersOfSMSec.Set(mNumberOfSuperModules / 2);
  Double_t kfSupermodulePhiWidth = mPhiSuperModule * TMath::DegToRad();
  mPhiCentersOfSM[0] = (mArm1PhiMin + mPhiSuperModule / 2.) * TMath::DegToRad();     // Define from First SM
  mPhiCentersOfSMSec[0] = mPhiCentersOfSM[0];                                        // the same in the First SM
  mPhiBoundariesOfSM[0] = mPhiCentersOfSM[0] - TMath::ATan2(mParSM[1], mIPDistance); // 1th and 2th modules)
  mPhiBoundariesOfSM[1] = mPhiCentersOfSM[0] + TMath::ATan2(mParSM[1], mIPDistance);

  if (mNumberOfSuperModules > 2) { // 2 to Max
    Int_t tmpSMType = GetSMType(2);
    for (int i = 1; i < mNPhiSuperModule; i++) {
      mPhiBoundariesOfSM[2 * i] += mPhiBoundariesOfSM[2 * i - 2] + kfSupermodulePhiWidth;
      if (tmpSMType == GetSMType(2 * i)) {
        mPhiBoundariesOfSM[2 * i + 1] += mPhiBoundariesOfSM[2 * i - 1] + kfSupermodulePhiWidth;
      } else {
        // changed SM Type, redefine the [2*i+1] Boundaries
        tmpSMType = GetSMType(2 * i);
        if (GetSMType(2 * i) == EMCAL_STANDARD) {
          mPhiBoundariesOfSM[2 * i + 1] = mPhiBoundariesOfSM[2 * i] + kfSupermodulePhiWidth;
        } else if (GetSMType(2 * i) == EMCAL_HALF) {
          mPhiBoundariesOfSM[2 * i + 1] = mPhiBoundariesOfSM[2 * i] + 2. * TMath::ATan2((mParSM[1]) / 2, mIPDistance);
        } else if (GetSMType(2 * i) == EMCAL_THIRD) {
          mPhiBoundariesOfSM[2 * i + 1] = mPhiBoundariesOfSM[2 * i] + 2. * TMath::ATan2((mParSM[1]) / 3, mIPDistance);
        } else if (GetSMType(2 * i) == DCAL_STANDARD) { // jump the gap
          mPhiBoundariesOfSM[2 * i] = (mDCALPhiMin - mArm1PhiMin) * TMath::DegToRad() + mPhiBoundariesOfSM[0];
          mPhiBoundariesOfSM[2 * i + 1] = (mDCALPhiMin - mArm1PhiMin) * TMath::DegToRad() + mPhiBoundariesOfSM[1];
        } else if (GetSMType(2 * i) == DCAL_EXT) {
          mPhiBoundariesOfSM[2 * i + 1] = mPhiBoundariesOfSM[2 * i] + 2. * TMath::ATan2((mParSM[1]) / 3, mIPDistance);
        }
      }
      mPhiCentersOfSM[i] = (mPhiBoundariesOfSM[2 * i] + mPhiBoundariesOfSM[2 * i + 1]) / 2.;
      mPhiCentersOfSMSec[i] = mPhiBoundariesOfSM[2 * i] + TMath::ATan2(mParSM[1], mIPDistance);
    }
  }

  // inner extend in eta (same as outer part) for DCal (0.189917), //calculated from the smallest gap (1# cell to the
  // 80-degree-edge),
  Double_t innerExtandedPhi =
    1.102840997; // calculated from the smallest gap (1# cell to the 80-degree-edge), too complicatd to explain...
  mDCALInnerExtandedEta = -TMath::Log(
    TMath::Tan((TMath::Pi() / 2. - 8 * mTrd1Angle * TMath::DegToRad() +
                (TMath::Pi() / 2 - mNZ * mTrd1Angle * TMath::DegToRad() - TMath::ATan(TMath::Exp(mArm1EtaMin)) * 2)) /
               2.));

  mEMCALPhiMax = mArm1PhiMin;
  mDCALPhiMax = mDCALPhiMin; // DCAl extention will not be included
  for (Int_t i = 0; i < mNumberOfSuperModules; i += 2) {
    if (GetSMType(i) == EMCAL_STANDARD)
      mEMCALPhiMax += 20.;
    else if (GetSMType(i) == EMCAL_HALF)
      mEMCALPhiMax += mPhiSuperModule / 2. + innerExtandedPhi;
    else if (GetSMType(i) == EMCAL_THIRD)
      mEMCALPhiMax += mPhiSuperModule / 3. + 4.0 * innerExtandedPhi / 3.0;
    else if (GetSMType(i) == DCAL_STANDARD) {
      mDCALPhiMax += 20.;
      mDCALStandardPhiMax = mDCALPhiMax;
    } else if (GetSMType(i) == DCAL_EXT)
      mDCALPhiMax += mPhiSuperModule / 3. + 4.0 * innerExtandedPhi / 3.0;
    else
      LOG(ERROR) << "Unkown SM Type!!\n";
  }
  // for compatible reason
  // if(fNumberOfSuperModules == 4) {fEMCALPhiMax = fArm1PhiMax ;}
  if (mNumberOfSuperModules == 12) {
    mEMCALPhiMax = mArm1PhiMax;
  }

  // called after setting of scintillator and lead layer parameters
  // called now in AliEMCALv0::CreateGeometry() - 15/03/16
  // DefineSamplingFraction(mcname,mctitle);

  sInit = kTRUE;
}

void EMCGeometry::PrintStream(std::ostream& stream) const
{
  using boost::algorithm::contains;

  // Separate routine is callable from broswer; Nov 7,2006
  stream << "\nInit: geometry of EMCAL named " << mGeoName << " :\n";
  if (mArrayOpts) {
    for (Int_t i = 0; i < mArrayOpts->GetEntries(); i++) {
      TObjString* o = (TObjString*)mArrayOpts->At(i);
      stream << i << ": " << o->String() << std::endl;
    }
  }
  if (contains(mGeoName, "DCAL")) {
    stream << "Phi min of DCAL SuperModule: " << std::setw(7) << std::setprecision(1) << mDCALPhiMin << ", DCAL has "
           << mnSupModInDCAL << "  SuperModule\n";
    stream << "The DCAL inner edge is +- " << std::setw(7) << std::setprecision(1) << mDCALInnerEdge << std::endl;
    if (contains(mGeoName, "DCAL_8SM"))
      stream << "DCAL has its 2 EXTENTION SM\n";
  }
  stream << "Granularity: " << GetNZ() << " in eta and " << GetNPhi() << " in phi\n";
  stream << "Layout: phi = (" << std::setw(7) << std::setprecision(1) << GetArm1PhiMin() << "," << GetArm1PhiMax()
         << "), eta = (" << std::setw(5) << std::setprecision(2) << GetArm1EtaMin() << "," << GetArm1EtaMax()
         << "), IP = " << std::setw(7) << std::setprecision(1) << GetIPDistance() << "-> for EMCAL envelope only\n";

  stream << "               ECAL      : " << GetNECLayers() << " x (" << GetECPbRadThick() << " cm Pb, "
         << GetECScintThick() << " cm Sc) \n",
    stream << "                fSampling " << std::setw(5) << std::setprecision(2) << mSampling << std::endl;
  stream << " fIPDistance       " << std::setw(6) << std::setprecision(3) << mIPDistance << " cm \n";
  stream << " fNPhi " << mNPhi << "   |  fNZ " << mNZ << std::endl;
  stream << " fNCellsInModule " << mNCellsInModule << " : fNCellsInSupMod " << mNCellsInSupMod << " : fNCells "
         << mNCells << std::endl;
  stream << " X:Y module size     " << std::setw(6) << std::setprecision(3) << mPhiModuleSize << ", " << mEtaModuleSize
         << " cm \n";
  stream << " X:Y   tile size     " << std::setw(6) << std::setprecision(3) << mPhiTileSize << ", " << mEtaTileSize
         << " cm \n";
  stream << " #of sampling layers " << mNECLayers << "(mNECLayers) \n";
  stream << " fLongModuleSize     " << std::setw(6) << std::setprecision(3) << mLongModuleSize << " cm \n";
  stream << " #supermodule in phi direction " << mNPhiSuperModule << std::endl;
  stream << " supermodule width in phi direction " << mPhiSuperModule << std::endl;
  stream << " fILOSS " << mILOSS << " : fIHADR " << mIHADR << std::endl;
  stream << " fTrd1Angle " << std::setw(7) << std::setprecision(4) << mTrd1Angle << std::endl;
  stream << " f2Trd1Dx2  " << std::setw(7) << std::setprecision(4) << m2Trd1Dx2 << std::endl;
  stream << " fTrd1AlFrontThick   " << std::setw(7) << std::setprecision(4) << mTrd1AlFrontThick << std::endl;
  stream << " fTrd1BondPaperThick " << std::setw(7) << std::setprecision(4) << mTrd1BondPaperThick << std::endl;
  stream << "SM dimensions(TRD1) : dx " << std::setw(7) << std::setprecision(4) << mParSM[0] << " dy " << mParSM[1]
         << " dz " << mParSM[2] << "(SMOD, BOX)\n";
  stream << " fPhiGapForSM  " << std::setw(7) << std::setprecision(4) << mPhiGapForSM << " cm ("
         << TMath::ATan2(mPhiGapForSM, mIPDistance) * TMath::RadToDeg() << " <- phi size in degree)\n";
  if (mKey110DEG && !contains(mGeoName, "12SMV1"))
    stream << " Last two modules have size 10 degree in  phi (180<phi<190)\n";
  if (mKey110DEG && contains(mGeoName, "12SMV1"))
    stream << " Last two modules have size 6.6 degree in  phi (180<phi<186.6)\n";
  stream << " phi SM boundaries \n";
  for (int i = 0; i < mPhiBoundariesOfSM.GetSize() / 2.; i++) {
    stream << i << " : " << std::setw(7) << std::setprecision(15) << mPhiBoundariesOfSM[2 * i] << "(" << std::setw(7)
           << std::setprecision(12) << mPhiBoundariesOfSM[2 * i] * TMath::RadToDeg() << ") -> " << std::setw(7)
           << std::setprecision(15) << mPhiBoundariesOfSM[2 * i + 1] << "(" << std::setw(7) << std::setprecision(12)
           << mPhiBoundariesOfSM[2 * i + 1] * TMath::RadToDeg() << ") : center " << std::setw(7)
           << std::setprecision(15) << mPhiCentersOfSM[i] << "(" << std::setw(7) << std::setprecision(12)
           << mPhiCentersOfSM[i] * TMath::RadToDeg() << ") \n";
  }
}

void EMCGeometry::CheckAdditionalOptions()
{
  // Dec 27,2006
  // adeed allILOSS= and allIHADR= for MIP investigation
  // TODO, FIXME Hans, Aug 2015: Shouldn't one add
  // if(fArrayOpts){fArrayOpts->Delete();delete fArrayOpts;}
  // This function is called twice in the Init()
  // End Hans, Aug 2015
  mArrayOpts = new TObjArray;
  Int_t nopt = ParseString(mGeoName, *mArrayOpts);
  if (nopt == 1) { // no aditional option(s)
    mArrayOpts->Delete();
    delete mArrayOpts;
    mArrayOpts = nullptr;
    return;
  }

  for (Int_t i = 1; i < nopt; i++) {
    TObjString* o = (TObjString*)mArrayOpts->At(i);

    TString addOpt = o->String();
    Int_t indj = -1;
    for (Int_t j = 0; j < mNAdditionalOpts; j++) {
      TString opt = mAdditionalOpts[j];
      if (addOpt.Contains(opt, TString::kIgnoreCase)) {
        indj = j;
        break;
      }
    }

    if (indj < 0) {
      LOG(DEBUG2) << "<E> option |" << addOpt << "| unavailable : ** look to the file Geometry.h **\n";
      assert(0);
    } else {
      LOG(DEBUG2) << "<I> option |" << addOpt << "| is valid : number " << indj << " : |" << mAdditionalOpts[indj]
                  << "|\n";
      if (addOpt.Contains("NL=", TString::kIgnoreCase)) { // number of sampling layers
        sscanf(addOpt.Data(), "NL=%i", &mNECLayers);
        LOG(DEBUG2) << " mNECLayers " << mNECLayers << " (new) \n";
      } else if (addOpt.Contains("PBTH=", TString::kIgnoreCase)) { // Thickness of the Pb(fECPbRadThicknes)
        sscanf(addOpt.Data(), "PBTH=%f", &mECPbRadThickness);
      } else if (addOpt.Contains("SCTH=", TString::kIgnoreCase)) { // Thickness of the Sc(fECScintThick)
        sscanf(addOpt.Data(), "SCTH=%f", &mECScintThick);
      } else if (addOpt.Contains("LATSS=",
                                 TString::kIgnoreCase)) { // Thickness of lateral steel strip (fLateralSteelStrip)
        sscanf(addOpt.Data(), "LATSS=%f", &mLateralSteelStrip);
        LOG(DEBUG2) << " mLateralSteelStrip " << mLateralSteelStrip << " (new) \n";
      } else if (addOpt.Contains("ILOSS=", TString::kIgnoreCase)) { // As in Geant
        sscanf(addOpt.Data(), "ALLILOSS=%i", &mILOSS);
        LOG(DEBUG2) << " fILOSS " << mILOSS << FairLogger::endl;
      } else if (addOpt.Contains("IHADR=", TString::kIgnoreCase)) { // As in Geant
        sscanf(addOpt.Data(), "ALLIHADR=%i", &mIHADR);
        LOG(DEBUG2) << " fIHADR " << mIHADR << FairLogger::endl;
      }
    }
  }
}

void EMCGeometry::DefineSamplingFraction(const std::string_view mcname, const std::string_view mctitle)
{
  // Jun 05,2006
  // Look http://rhic.physics.wayne.edu/~pavlinov/ALICE/SHISHKEBAB/RES/linearityAndResolutionForTRD1.html
  // Keep for compatibility
  //
  using boost::algorithm::contains;

  // Sampling factor for G3
  mSampling = 10.87;      // Default value - Nov 25,2010
  if (mNECLayers == 69) { // 10% layer reduction
    mSampling = 12.55;
  } else if (mNECLayers == 61) { // 20% layer reduction
    mSampling = 12.80;
  } else if (mNECLayers == 77) {
    if (contains(mGeoName, "V1")) {
      mSampling = 10.87;                                         // Adding paper sheets and cover plate; Nov 25,2010
    } else if (mECScintThick > 0.159 && mECScintThick < 0.161) { // original sampling fraction, equal layers
      mSampling = 12.327;                                        // fECScintThick = fECPbRadThickness = 0.160;
    } else if (mECScintThick > 0.175 && mECScintThick < 0.177) { // 10% Pb thicknes reduction
      mSampling = 10.5;                                          // fECScintThick = 0.176, fECPbRadThickness=0.144;
    } else if (mECScintThick > 0.191 && mECScintThick < 0.193) { // 20% Pb thicknes reduction
      mSampling = 8.93;                                          // fECScintThick = 0.192, fECPbRadThickness=0.128;
    }
  }

  Float_t samplingFactorTranportModel = 1.;
  if (contains(mcname, "Geant3"))
    samplingFactorTranportModel = 1.; // 0.988 // Do nothing
  else if (contains(mcname, "Fluka"))
    samplingFactorTranportModel = 1.; // To be set
  else if (contains(mcname, "Geant4")) {
    if (contains(mctitle, "EMV-EMCAL"))
      samplingFactorTranportModel = 0.821; // EMC list but for EMCal, before 0.86
    else if (contains(mctitle, "EMV"))
      samplingFactorTranportModel = 1.096; // 0.906, 0.896 (OPT)
    else
      samplingFactorTranportModel = 0.821; // 1.15 (CHIPS), 1.149 (BERT), 1.147 (BERT_CHIPS)
  }

  LOG(INFO) << "MC modeler <" << mcname << ">, Title <" << mctitle << ">: Sampling " << std::setw(2)
            << std::setprecision(3) << mSampling << ", model fraction with respect to G3 "
            << samplingFactorTranportModel << ", final sampling " << mSampling * samplingFactorTranportModel
            << FairLogger::endl;

  mSampling *= samplingFactorTranportModel;
}

Double_t EMCGeometry::GetPhiCenterOfSMSec(Int_t nsupmod) const
{
  int i = nsupmod / 2;
  return mPhiCentersOfSMSec[i];
}

Double_t EMCGeometry::GetPhiCenterOfSM(Int_t nsupmod) const
{
  int i = nsupmod / 2;
  return mPhiCentersOfSM[i];
}

std::tuple<double, double> EMCGeometry::GetPhiBoundariesOfSM(Int_t nSupMod) const
{
  int i;
  if (nSupMod < 0 || nSupMod > 12 + mnSupModInDCAL - 1)
    throw InvalidModuleException(nSupMod, 12 + mnSupModInDCAL);
  i = nSupMod / 2;
  return std::make_tuple((Double_t)mPhiBoundariesOfSM[2 * i], (Double_t)mPhiBoundariesOfSM[2 * i + 1]);
}

std::tuple<double, double> EMCGeometry::GetPhiBoundariesOfSMGap(Int_t nPhiSec) const
{
  if (nPhiSec < 0 || nPhiSec > 5 + mnSupModInDCAL / 2 - 1)
    throw InvalidModuleException(nPhiSec, 5 + mnSupModInDCAL / 2);
  return std::make_tuple(mPhiBoundariesOfSM[2 * nPhiSec + 1], mPhiBoundariesOfSM[2 * nPhiSec + 2]);
}

int EMCGeometry::ParseString(const TString& topt, TObjArray& Opt)
{
  Ssiz_t begin, index, end, end2;
  begin = index = end = end2 = 0;
  TRegexp separator(R"([^ ;,\\t\\s/]+)");
  while ((begin < topt.Length()) && (index != kNPOS)) {
    // loop over given options
    index = topt.Index(separator, &end, begin);
    if (index >= 0 && end >= 1) {
      TString substring(topt(index, end));
      Opt.Add(new TObjString(substring.Data()));
    }
    begin += end + 1;
  }
  return Opt.GetEntries();
}

std::ostream& o2::EMCAL::operator<<(std::ostream& stream, const o2::EMCAL::EMCGeometry& geo)
{
  geo.PrintStream(stream);
  return stream;
}

ClassImp(EMCGeometry);
