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

#include <TGeoBBox.h>
#include <TGeoManager.h>
#include <TGeoMatrix.h>
#include <TList.h>

#include <FairLogger.h>

#include "EMCALBase/Geometry.h"
#include "EMCALBase/ShishKebabTrd1Module.h"

#include <boost/algorithm/string/predicate.hpp>

using namespace o2::emcal;

// these initialisations are needed for a singleton
Geometry* Geometry::sGeom = nullptr;

Geometry::Geometry(const Geometry& geo)
  : mGeoName(geo.mGeoName),
    mKey110DEG(geo.mKey110DEG),
    mnSupModInDCAL(geo.mnSupModInDCAL),
    mNCellsInSupMod(geo.mNCellsInSupMod),
    mNETAdiv(geo.mNETAdiv),
    mNPHIdiv(geo.mNPHIdiv),
    mNCellsInModule(geo.mNCellsInModule),
    mPhiBoundariesOfSM(geo.mPhiBoundariesOfSM),
    mPhiCentersOfSM(geo.mPhiCentersOfSM),
    mPhiCentersOfSMSec(geo.mPhiCentersOfSMSec),
    mPhiCentersOfCells(geo.mPhiCentersOfCells),
    mCentersOfCellsEtaDir(geo.mCentersOfCellsEtaDir),
    mCentersOfCellsPhiDir(geo.mCentersOfCellsPhiDir),
    mEtaCentersOfCells(geo.mEtaCentersOfCells),
    mNCells(geo.mNCells),
    mNPhi(geo.mNPhi),
    mCentersOfCellsXDir(geo.mCentersOfCellsXDir),
    mArm1EtaMin(geo.mArm1EtaMin),
    mArm1EtaMax(geo.mArm1EtaMax),
    mArm1PhiMin(geo.mArm1PhiMin),
    mArm1PhiMax(geo.mArm1PhiMax),
    mEtaMaxOfTRD1(geo.mEtaMaxOfTRD1),
    mDCALPhiMin(geo.mDCALPhiMin),
    mDCALPhiMax(geo.mDCALPhiMax),
    mEMCALPhiMax(geo.mEMCALPhiMax),
    mDCALStandardPhiMax(geo.mDCALStandardPhiMax),
    mDCALInnerExtandedEta(geo.mDCALInnerExtandedEta),
    mDCALInnerEdge(geo.mDCALInnerEdge),
    mShishKebabTrd1Modules(geo.mShishKebabTrd1Modules),
    mPhiModuleSize(geo.mPhiModuleSize),
    mEtaModuleSize(geo.mEtaModuleSize),
    mPhiTileSize(geo.mPhiTileSize),
    mEtaTileSize(geo.mEtaTileSize),
    mNZ(geo.mNZ),
    mIPDistance(geo.mIPDistance),
    mLongModuleSize(geo.mLongModuleSize),
    mShellThickness(geo.mShellThickness),
    mZLength(geo.mZLength),
    mSampling(geo.mSampling),
    mECPbRadThickness(geo.mECPbRadThickness),
    mECScintThick(geo.mECScintThick),
    mNECLayers(geo.mNECLayers),
    mNumberOfSuperModules(geo.mNumberOfSuperModules),
    mEMCSMSystem(geo.mEMCSMSystem),
    mFrontSteelStrip(geo.mFrontSteelStrip),
    mLateralSteelStrip(geo.mLateralSteelStrip),
    mPassiveScintThick(geo.mPassiveScintThick),
    mPhiSuperModule(geo.mPhiSuperModule),
    mNPhiSuperModule(geo.mNPhiSuperModule),
    mTrd1Angle(geo.mTrd1Angle),
    m2Trd1Dx2(geo.m2Trd1Dx2),
    mPhiGapForSM(geo.mPhiGapForSM),
    mTrd1AlFrontThick(geo.mTrd1AlFrontThick),
    mTrd1BondPaperThick(geo.mTrd1BondPaperThick),
    mILOSS(geo.mILOSS),
    mIHADR(geo.mIHADR),
    mSteelFrontThick(geo.mSteelFrontThick) // obsolete data member?
{
  memcpy(mEnvelop, geo.mEnvelop, sizeof(Float_t) * 3);
  memcpy(mParSM, geo.mParSM, sizeof(Float_t) * 3);

  memset(SMODULEMATRIX, 0, sizeof(TGeoHMatrix*) * EMCAL_MODULES);
}

Geometry::Geometry(const std::string_view name, const std::string_view mcname, const std::string_view mctitle)
  : mGeoName(name),
    mKey110DEG(0),
    mnSupModInDCAL(0),
    mNCellsInSupMod(0),
    mNETAdiv(0),
    mNPHIdiv(0),
    mNCellsInModule(0),
    mPhiBoundariesOfSM(),
    mPhiCentersOfSM(),
    mPhiCentersOfSMSec(),
    mPhiCentersOfCells(),
    mCentersOfCellsEtaDir(),
    mCentersOfCellsPhiDir(),
    mEtaCentersOfCells(),
    mNCells(0),
    mNPhi(0),
    mCentersOfCellsXDir(),
    mArm1EtaMin(0),
    mArm1EtaMax(0),
    mArm1PhiMin(0),
    mArm1PhiMax(0),
    mEtaMaxOfTRD1(0),
    mDCALPhiMin(0),
    mDCALPhiMax(0),
    mEMCALPhiMax(0),
    mDCALStandardPhiMax(0),
    mDCALInnerExtandedEta(0),
    mDCALInnerEdge(0.),
    mShishKebabTrd1Modules(),
    mPhiModuleSize(0.),
    mEtaModuleSize(0.),
    mPhiTileSize(0.),
    mEtaTileSize(0.),
    mNZ(0),
    mIPDistance(0.),
    mLongModuleSize(0.),
    mShellThickness(0.),
    mZLength(0.),
    mSampling(0.),
    mECPbRadThickness(0.),
    mECScintThick(0.),
    mNECLayers(0),
    mNumberOfSuperModules(0),
    mEMCSMSystem(),
    mFrontSteelStrip(0.),
    mLateralSteelStrip(0.),
    mPassiveScintThick(0.),
    mPhiSuperModule(0),
    mNPhiSuperModule(0),
    mTrd1Angle(0.),
    m2Trd1Dx2(0.),
    mPhiGapForSM(0.),
    mTrd1AlFrontThick(0.0),
    mTrd1BondPaperThick(0.),
    mILOSS(-1),
    mIHADR(-1),
    mSteelFrontThick(0.) // obsolete data member?
{
  DefineEMC(mcname, mctitle);
  mNCellsInModule = mNPHIdiv * mNETAdiv;

  CreateListOfTrd1Modules();

  memset(SMODULEMATRIX, 0, sizeof(TGeoHMatrix*) * EMCAL_MODULES);

  LOG(DEBUG) << "Name <<" << name << ">>";
}

Geometry& Geometry::operator=(const Geometry& /*rvalue*/)
{
  LOG(FATAL) << "assignment operator, not implemented";
  return *this;
}

Geometry::~Geometry()
{
  if (this == sGeom) {
    LOG(ERROR) << "Do not call delete on me";
    return;
  }

  for (Int_t smod = 0; smod < mNumberOfSuperModules; smod++) {
    if (SMODULEMATRIX[smod])
      delete SMODULEMATRIX[smod];
  }
}

Geometry* Geometry::GetInstance()
{
  Geometry* rv = static_cast<Geometry*>(sGeom);
  return rv;
}

Geometry* Geometry::GetInstance(const std::string_view name, const std::string_view mcname,
                                const std::string_view mctitle)
{
  if (!sGeom) {
    if (!name.length()) { // get default geometry
      sGeom = new Geometry(DEFAULT_GEOMETRY, mcname, mctitle);
    } else {
      sGeom = new Geometry(name, mcname, mctitle);
    } // end if strcmp(name,"")
    return sGeom;
  } else {
    if (sGeom->GetName() != name) {
      LOG(INFO) << "\n current geometry is " << sGeom->GetName() << " : you should not call " << name;
    } // end
    return sGeom;
  } // end if sGeom

  return nullptr;
}

Geometry* Geometry::GetInstanceFromRunNumber(Int_t runNumber, const std::string_view geoName,
                                             const std::string_view mcname, const std::string_view mctitle)
{
  using boost::algorithm::contains;

  // printf("AliEMCALGeometry::GetInstanceFromRunNumber() - run %d, geoName <<%s>> \n",runNumber,geoName.Data());

  if (runNumber >= 104064 && runNumber < 140000) {
    // 2009-2010 runs
    // First year geometry, 4 SM.

    if (contains(geoName, "FIRSTYEARV1") && geoName != std::string("")) {
      LOG(INFO) << "o2::emcal::Geometry::GetInstanceFromRunNumber() *** ATTENTION *** \n"
                << "\t Specified geometry name <<" << geoName << ">> for run " << runNumber
                << " is not considered! \n"
                << "\t In use <<EMCAL_FIRSTYEARV1>>, check run number and year";
    } else {
      LOG(INFO)
        << "o2::emcal::Geometry::GetInstanceFromRunNumber() - Initialized geometry with name <<EMCAL_FIRSTYEARV1>>";
    }

    return Geometry::GetInstance("EMCAL_FIRSTYEARV1", mcname, mctitle);
  } else if (runNumber >= 140000 && runNumber <= 170593) {
    // Almost complete EMCAL geometry, 10 SM. Year 2011 configuration

    if (contains(geoName, "COMPLETEV1") && geoName != std::string("")) {
      LOG(INFO) << "o2::emcal::Geometry::GetInstanceFromRunNumber() *** ATTENTION *** \n"
                << "\t Specified geometry name <<" << geoName << ">> for run " << runNumber
                << " is not considered! \n"
                << "\t In use <<EMCAL_COMPLETEV1>>, check run number and year";
    } else {
      LOG(INFO)
        << "o2::emcal::Geometry::GetInstanceFromRunNumber() - Initialized geometry with name <<EMCAL_COMPLETEV1>>";
    }
    return Geometry::GetInstance("EMCAL_COMPLETEV1", mcname, mctitle);
  } else if (runNumber > 176000 && runNumber <= 197692) {
    // Complete EMCAL geometry, 12 SM. Year 2012 and on
    // The last 2 SM were not active, anyway they were there.

    if (contains(geoName, "COMPLETE12SMV1") && geoName != std::string("")) {
      LOG(INFO) << "o2::emcal::Geometry::GetInstanceFromRunNumber() *** ATTENTION *** \n"
                << "\t Specified geometry name <<" << geoName << " >> for run " << runNumber
                << " is not considered! \n"
                << "\t In use <<EMCAL_COMPLETE12SMV1>>, check run number and year";
    } else {
      LOG(INFO) << "o2::emcal::Geometry::GetInstanceFromRunNumber() - Initialized geometry with name "
                   "<<EMCAL_COMPLETE12SMV1>>";
    }
    return Geometry::GetInstance("EMCAL_COMPLETE12SMV1", mcname, mctitle);
  } else // Run 2
  {
    // EMCAL + DCAL geometry, 20 SM. Year 2015 and on

    if (contains(geoName, "DCAL_8SM") && geoName != std::string("")) {
      LOG(INFO) << "o2::emcal::Geometry::GetInstanceFromRunNumber() *** ATTENTION *** \n"
                << "\t Specified geometry name <<" << geoName << ">> for run " << runNumber
                << " is not considered! \n"
                << "\t In use <<EMCAL_COMPLETE12SMV1_DCAL_8SM>>, check run number and year";
    } else {
      LOG(INFO) << "o2::emcal::Geometry::GetInstanceFromRunNumber() - Initialized geometry with name "
                   "<<EMCAL_COMPLETE12SMV1_DCAL_8SM>>";
    }
    return Geometry::GetInstance("EMCAL_COMPLETE12SMV1_DCAL_8SM", mcname, mctitle);
  }
}

void Geometry::DefineSamplingFraction(const std::string_view mcname, const std::string_view mctitle)
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
            << samplingFactorTranportModel << ", final sampling " << mSampling * samplingFactorTranportModel;

  mSampling *= samplingFactorTranportModel;
}

void Geometry::DefineEMC(std::string_view mcname, std::string_view mctitle)
{
  using boost::algorithm::contains;

  // geometry
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

  // modifications to the above for PDC06 geometry
  if (contains(mGeoName, "PDC06")) {          // 18-may-05 - about common structure
    mECScintThick = mECPbRadThickness = 0.16; // (13-may-05 from V.Petrov)
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
  }

  // In 2009-2010 data taking runs only 4 SM, in the upper position.
  if (contains(mGeoName, "FIRSTYEAR")) {
    mNumberOfSuperModules = 4;
    mArm1PhiMax = 120.0;
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
  }

  //
  // Init EMCal/DCal SMs type array
  mEMCSMSystem.clear();
  mEMCSMSystem.resize(mNumberOfSuperModules);

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
  mPhiBoundariesOfSM.resize(mNumberOfSuperModules);
  mPhiCentersOfSM.resize(mNumberOfSuperModules / 2);
  mPhiCentersOfSMSec.resize(mNumberOfSuperModules / 2);
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
}

void Geometry::GetGlobal(const Double_t* loc, Double_t* glob, int iSM) const
{
  const TGeoHMatrix* m = GetMatrixForSuperModule(iSM);
  if (m) {
    m->LocalToMaster(loc, glob);
  } else {
    LOG(FATAL) << "Geo matrixes are not loaded \n";
  }
}

void Geometry::GetGlobal(const TVector3& vloc, TVector3& vglob, int iSM) const
{
  Double_t tglob[3], tloc[3];
  vloc.GetXYZ(tloc);
  GetGlobal(tloc, tglob, iSM);
  vglob.SetXYZ(tglob[0], tglob[1], tglob[2]);
}

void Geometry::GetGlobal(Int_t absId, Double_t glob[3]) const
{
  double loc[3];

  memset(glob, 0, sizeof(Double_t) * 3);
  try {
    auto cellpos = RelPosCellInSModule(absId);
    loc[0] = cellpos.X();
    loc[1] = cellpos.Y();
    loc[2] = cellpos.Z();
  } catch (InvalidCellIDException& e) {
    LOG(ERROR) << e.what();
    return;
  }

  Int_t nSupMod = std::get<0>(GetCellIndex(absId));
  const TGeoHMatrix* m = GetMatrixForSuperModule(nSupMod);
  if (m) {
    m->LocalToMaster(loc, glob);
  } else {
    LOG(FATAL) << "Geo matrixes are not loaded \n";
  }
}

void Geometry::GetGlobal(Int_t absId, TVector3& vglob) const
{
  Double_t glob[3];

  GetGlobal(absId, glob);
  vglob.SetXYZ(glob[0], glob[1], glob[2]);
}

std::tuple<double, double> Geometry::EtaPhiFromIndex(Int_t absId) const
{
  TVector3 vglob;
  GetGlobal(absId, vglob);
  return std::make_tuple(vglob.Eta(), vglob.Phi());
}

Int_t Geometry::GetAbsCellId(Int_t nSupMod, Int_t nModule, Int_t nIphi, Int_t nIeta) const
{
  // 0 <= nSupMod < fNumberOfSuperModules
  // 0 <= nModule  < fNPHI * fNZ ( fNPHI * fNZ/2 for fKey110DEG=1)
  // 0 <= nIphi   < fNPHIdiv
  // 0 <= nIeta   < fNETAdiv
  // 0 <= absid   < fNCells
  Int_t id = 0; // have to change from 0 to fNCells-1
  for (int i = 0; i < nSupMod; i++) {
    if (GetSMType(i) == EMCAL_STANDARD)
      id += mNCellsInSupMod;
    else if (GetSMType(i) == EMCAL_HALF)
      id += mNCellsInSupMod / 2;
    else if (GetSMType(i) == EMCAL_THIRD)
      id += mNCellsInSupMod / 3;
    else if (GetSMType(i) == DCAL_STANDARD)
      id += 2 * mNCellsInSupMod / 3;
    else if (GetSMType(i) == DCAL_EXT)
      id += mNCellsInSupMod / 3;
    else
      throw InvalidSupermoduleTypeException();
  }

  id += mNCellsInModule * nModule;
  id += mNPHIdiv * nIphi;
  id += nIeta;
  if (!CheckAbsCellId(id))
    id = -TMath::Abs(id); // if negative something wrong

  return id;
}

std::tuple<Int_t, Int_t, Int_t> Geometry::GetModuleIndexesFromCellIndexesInSModule(Int_t nSupMod, Int_t iphi, Int_t ieta) const
{
  Int_t nphi = GetNumberOfModuleInPhiDirection(nSupMod);

  Int_t ietam = ieta / mNETAdiv,
        iphim = iphi / mNPHIdiv,
        nModule = ietam * nphi + iphim;
  return std::make_tuple(iphim, ietam, nModule);
}

Int_t Geometry::GetAbsCellIdFromCellIndexes(Int_t nSupMod, Int_t iphi, Int_t ieta) const
{
  // Check if the indeces correspond to existing SM or tower indeces
  if (iphi < 0 || iphi >= EMCAL_ROWS || ieta < 0 || ieta >= EMCAL_COLS || nSupMod < 0 ||
      nSupMod >= GetNumberOfSuperModules()) {
    LOG(DEBUG) << "Wrong cell indexes : SM " << nSupMod << ", column (eta) " << ieta << ", row (phi) " << iphi;
    return -1;
  }

  auto indexmod = GetModuleIndexesFromCellIndexesInSModule(nSupMod, iphi, ieta);

  Int_t nIeta = ieta % mNETAdiv,
        nIphi = iphi % mNPHIdiv;
  nIeta = mNETAdiv - 1 - nIeta;
  return GetAbsCellId(nSupMod, std::get<2>(indexmod), nIphi, nIeta);
}

std::tuple<int, int> Geometry::GlobalRowColFromIndex(int cellID) const
{
  auto indexes = GetCellIndex(cellID);
  auto supermodule = std::get<0>(indexes),
       module = std::get<1>(indexes),
       nPhiInMod = std::get<2>(indexes),
       nEtaInMod = std::get<3>(indexes);
  auto rcSupermodule = GetCellPhiEtaIndexInSModule(supermodule, nPhiInMod, nPhiInMod, nEtaInMod);
  auto row = std::get<0>(rcSupermodule),
       col = std::get<1>(rcSupermodule);
  // add offsets (row / col per supermodule)
  if (supermodule % 2)
    col += mNZ * 2;
  int sector = supermodule / 2;
  if (sector > 0) {
    for (int isec = 0; isec < sector - 1; isec++) {
      auto smtype = GetSMType(isec * 2);
      auto nphism = (smtype == EMCAL_THIRD || smtype == DCAL_EXT) ? GetNPhi() / 3 : GetNPhi();
      row += 2 * nphism;
    }
  }
  return std::make_tuple(row, col);
}

int Geometry::GlobalCol(int cellID) const
{
  return std::get<1>(GlobalRowColFromIndex(cellID));
}

int Geometry::GlobalRow(int cellID) const
{
  return std::get<0>(GlobalRowColFromIndex(cellID));
}

Int_t Geometry::SuperModuleNumberFromEtaPhi(Double_t eta, Double_t phi) const
{
  if (TMath::Abs(eta) > mEtaMaxOfTRD1)
    throw InvalidPositionException(eta, phi);

  phi = TVector2::Phi_0_2pi(phi); // move phi to (0,2pi) boundaries
  Int_t nphism = mNumberOfSuperModules / 2;
  Int_t nSupMod = 0;
  for (Int_t i = 0; i < nphism; i++) {
    LOG(DEBUG) << "Sec " << i << ": Min " << mPhiBoundariesOfSM[2 * i] << ", Max " << mPhiBoundariesOfSM[2 * i + 1];
    if (phi >= mPhiBoundariesOfSM[2 * i] && phi <= mPhiBoundariesOfSM[2 * i + 1]) {
      nSupMod = 2 * i;
      if (eta < 0.0)
        nSupMod++;

      if (GetSMType(nSupMod) == DCAL_STANDARD) { // Gap between DCAL
        if (TMath::Abs(eta) < GetNEta() / 3 * mTrd1Angle * TMath::DegToRad())
          throw InvalidPositionException(eta, phi);
      }

      LOG(DEBUG) << "eta " << eta << " phi " << phi << " (" << std::setw(5) << std::setprecision(2)
                 << phi * TMath::RadToDeg() << ") : nSupMod " << nSupMod << ": #bound " << i;
      return nSupMod;
    }
  }
  throw InvalidPositionException(eta, phi);
}

Int_t Geometry::GetAbsCellIdFromEtaPhi(Double_t eta, Double_t phi) const
{
  Int_t nSupMod = SuperModuleNumberFromEtaPhi(eta, phi);

  // phi index first
  phi = TVector2::Phi_0_2pi(phi);
  Double_t phiLoc = phi - mPhiCentersOfSMSec[nSupMod / 2];
  Int_t nphi = mPhiCentersOfCells.size();
  if (GetSMType(nSupMod) == EMCAL_HALF)
    nphi /= 2;
  else if (GetSMType(nSupMod) == EMCAL_THIRD)
    nphi /= 3;
  else if (GetSMType(nSupMod) == DCAL_EXT)
    nphi /= 3;

  Double_t dmin = TMath::Abs(mPhiCentersOfCells[0] - phiLoc),
           d = 0.;
  Int_t iphi = 0;
  for (Int_t i = 1; i < nphi; i++) {
    d = TMath::Abs(mPhiCentersOfCells[i] - phiLoc);
    if (d < dmin) {
      dmin = d;
      iphi = i;
    }
    // printf(" i %i : d %f : dmin %f : fPhiCentersOfCells[i] %f \n", i, d, dmin, fPhiCentersOfCells[i]);
  }
  // odd SM are turned with respect of even SM - reverse indexes
  LOG(DEBUG2) << " iphi " << iphi << " : dmin " << dmin << " (phi " << phi << ", phiLoc " << phiLoc << ")\n";

  // eta index
  Double_t absEta = TMath::Abs(eta);
  Int_t neta = mCentersOfCellsEtaDir.size(),
        etaShift = iphi * neta,
        ieta = 0;
  if (GetSMType(nSupMod) == DCAL_STANDARD)
    ieta += 16; // jump 16 cells for DCSM
  dmin = TMath::Abs(mEtaCentersOfCells[etaShift + ieta] - absEta);
  for (Int_t i = ieta + 1; i < neta; i++) {
    d = TMath::Abs(mEtaCentersOfCells[i + etaShift] - absEta);
    if (d < dmin) {
      dmin = d;
      ieta = i;
    }
  }

  if (GetSMType(nSupMod) == DCAL_STANDARD)
    ieta -= 16; // jump 16 cells for DCSM

  LOG(DEBUG2) << " ieta " << ieta << " : dmin " << dmin << " (eta=" << eta << ") : nSupMod " << nSupMod;

  // patch for mapping following alice convention
  if (nSupMod % 2 ==
      0) { // 47 + 16 -ieta for DCSM, 47 - ieta for others, revert the ordering on A side in order to keep convention.
    ieta = (neta - 1) - ieta;
    if (GetSMType(nSupMod) == DCAL_STANDARD)
      ieta -= 16; // recover cells for DCSM
  }

  return GetAbsCellIdFromCellIndexes(nSupMod, iphi, ieta);
}

std::tuple<int, int, int, int> Geometry::GetCellIndex(Int_t absId) const
{
  if (!CheckAbsCellId(absId))
    throw InvalidCellIDException(absId);

  Int_t tmp = absId;
  Int_t test = absId;

  Int_t nSupMod;
  for (nSupMod = -1; test >= 0;) {
    nSupMod++;
    tmp = test;
    if (GetSMType(nSupMod) == EMCAL_STANDARD)
      test -= mNCellsInSupMod;
    else if (GetSMType(nSupMod) == EMCAL_HALF)
      test -= mNCellsInSupMod / 2;
    else if (GetSMType(nSupMod) == EMCAL_THIRD)
      test -= mNCellsInSupMod / 3;
    else if (GetSMType(nSupMod) == DCAL_STANDARD)
      test -= 2 * mNCellsInSupMod / 3;
    else if (GetSMType(nSupMod) == DCAL_EXT)
      test -= mNCellsInSupMod / 3;
    else {
      throw InvalidSupermoduleTypeException();
    }
  }

  Int_t nModule = tmp / mNCellsInModule;
  tmp = tmp % mNCellsInModule;
  Int_t nIphi = tmp / mNPHIdiv, nIeta = tmp % mNPHIdiv;
  return std::make_tuple(nSupMod, nModule, nIphi, nIeta);
}

Int_t Geometry::GetSuperModuleNumber(Int_t absId) const { return std::get<0>(GetCellIndex(absId)); }

std::tuple<int, int> Geometry::GetModulePhiEtaIndexInSModule(Int_t nSupMod, Int_t nModule) const
{
  Int_t nphi = -1;
  if (GetSMType(nSupMod) == EMCAL_HALF)
    nphi = mNPhi / 2; // halfSM
  else if (GetSMType(nSupMod) == EMCAL_THIRD)
    nphi = mNPhi / 3; // 1/3 SM
  else if (GetSMType(nSupMod) == DCAL_EXT)
    nphi = mNPhi / 3; // 1/3 SM
  else
    nphi = mNPhi; // full SM

  return std::make_tuple(int(nModule % nphi), int(nModule / nphi));
}

std::tuple<int, int> Geometry::GetCellPhiEtaIndexInSModule(Int_t nSupMod, Int_t nModule, Int_t nIphi,
                                                           Int_t nIeta) const
{
  auto indices = GetModulePhiEtaIndexInSModule(nSupMod, nModule);
  Int_t iphim = std::get<0>(indices), ietam = std::get<1>(indices);

  //  ieta  = ietam*fNETAdiv + (1-nIeta); // x(module) = -z(SM)
  Int_t ieta = ietam * mNETAdiv + (mNETAdiv - 1 - nIeta); // x(module) = -z(SM)
  Int_t iphi = iphim * mNPHIdiv + nIphi;                  // y(module) =  y(SM)

  if (iphi < 0 || ieta < 0)
    LOG(DEBUG) << " nSupMod " << nSupMod << " nModule " << nModule << " nIphi " << nIphi << " nIeta " << nIeta
               << " => ieta " << ieta << " iphi " << iphi;
  return std::make_tuple(iphi, ieta);
}

Point3D<double> Geometry::RelPosCellInSModule(Int_t absId) const
{
  // Shift index taking into account the difference between standard SM
  // and SM of half (or one third) size in phi direction

  Int_t phiindex = mCentersOfCellsPhiDir.size();
  Double_t zshift = 0.5 * GetDCALInnerEdge();
  Double_t xr, yr, zr;

  if (!CheckAbsCellId(absId))
    throw InvalidCellIDException(absId);

  auto cellindex = GetCellIndex(absId);
  Int_t nSupMod = std::get<0>(cellindex), nModule = std::get<1>(cellindex), nIphi = std::get<2>(cellindex),
        nIeta = std::get<3>(cellindex);
  auto indexinsm = GetCellPhiEtaIndexInSModule(nSupMod, nModule, nIphi, nIeta);
  Int_t iphi = std::get<0>(indexinsm), ieta = std::get<1>(indexinsm);

  // Get eta position. Careful with ALICE conventions (increase index decrease eta)
  Int_t ieta2 = ieta;
  if (nSupMod % 2 == 0)
    ieta2 = (mCentersOfCellsEtaDir.size() - 1) -
            ieta; // 47-ieta, revert the ordering on A side in order to keep convention.

  if (GetSMType(nSupMod) == DCAL_STANDARD && nSupMod % 2)
    ieta2 += 16; // DCAL revert the ordering on C side ...
  zr = mCentersOfCellsEtaDir[ieta2];
  if (GetSMType(nSupMod) == DCAL_STANDARD)
    zr -= zshift; // DCAL shift (SMALLER SM)
  xr = mCentersOfCellsXDir[ieta2];

  // Get phi position. Careful with ALICE conventions (increase index increase phi)
  Int_t iphi2 = iphi;
  if (GetSMType(nSupMod) == DCAL_EXT) {
    if (nSupMod % 2 != 0)
      iphi2 = (phiindex / 3 - 1) - iphi; // 7-iphi [1/3SM], revert the ordering on C side in order to keep convention.
    yr = mCentersOfCellsPhiDir[iphi2 + phiindex / 3];
  } else if (GetSMType(nSupMod) == EMCAL_HALF) {
    if (nSupMod % 2 != 0)
      iphi2 = (phiindex / 2 - 1) - iphi; // 11-iphi [1/2SM], revert the ordering on C side in order to keep
                                         // convention.
    yr = mCentersOfCellsPhiDir[iphi2 + phiindex / 4];
  } else if (GetSMType(nSupMod) == EMCAL_THIRD) {
    if (nSupMod % 2 != 0)
      iphi2 = (phiindex / 3 - 1) - iphi; // 7-iphi [1/3SM], revert the ordering on C side in order to keep convention.
    yr = mCentersOfCellsPhiDir[iphi2 + phiindex / 3];
  } else {
    if (nSupMod % 2 != 0)
      iphi2 = (phiindex - 1) - iphi; // 23-iphi, revert the ordering on C side in order to keep conventi
    yr = mCentersOfCellsPhiDir[iphi2];
  }

  LOG(DEBUG) << "absId " << absId << " nSupMod " << nSupMod << " iphi " << iphi << " ieta " << ieta << " xr " << xr
             << " yr " << yr << " zr " << zr;
  return Point3D<double>(xr, yr, zr);
}

Point3D<double> Geometry::RelPosCellInSModule(Int_t absId, Double_t distEff) const
{
  // Shift index taking into account the difference between standard SM
  // and SM of half (or one third) size in phi direction
  Double_t xr, yr, zr;
  Int_t nphiIndex = mCentersOfCellsPhiDir.size();
  Double_t zshift = 0.5 * GetDCALInnerEdge();
  Int_t kDCalshift = 8; // wangml DCal cut first 8 modules(16 cells)

  Int_t iphim = -1, ietam = -1;
  TVector2 v;
  if (!CheckAbsCellId(absId))
    throw InvalidCellIDException(absId);

  auto cellindex = GetCellIndex(absId);
  Int_t nSupMod = std::get<0>(cellindex), nModule = std::get<1>(cellindex), nIphi = std::get<2>(cellindex),
        nIeta = std::get<3>(cellindex);
  auto indmodep = GetModulePhiEtaIndexInSModule(nSupMod, nModule);
  iphim = std::get<0>(indmodep);
  ietam = std::get<1>(indmodep);
  auto indexinsm = GetCellPhiEtaIndexInSModule(nSupMod, nModule, nIphi, nIeta);
  Int_t iphi = std::get<0>(indexinsm), ieta = std::get<1>(indexinsm);

  // Get eta position. Careful with ALICE conventions (increase index decrease eta)
  if (nSupMod % 2 == 0) {
    ietam = (mCentersOfCellsEtaDir.size() / 2 - 1) -
            ietam; // 24-ietam, revert the ordering on A side in order to keep convention.
    if (nIeta == 0)
      nIeta = 1;
    else
      nIeta = 0;
  }

  if (GetSMType(nSupMod) == DCAL_STANDARD && nSupMod % 2)
    ietam += kDCalshift; // DCAL revert the ordering on C side ....
  const ShishKebabTrd1Module& mod = GetShishKebabModule(ietam);
  mod.GetPositionAtCenterCellLine(nIeta, distEff, v);
  xr = v.Y() - mParSM[0];
  zr = v.X() - mParSM[2];
  if (GetSMType(nSupMod) == DCAL_STANDARD)
    zr -= zshift; // DCAL shift (SMALLER SM)

  // Get phi position. Careful with ALICE conventions (increase index increase phi)
  Int_t iphi2 = iphi;
  if (GetSMType(nSupMod) == DCAL_EXT) {
    if (nSupMod % 2 != 0)
      iphi2 = (nphiIndex / 3 - 1) - iphi; // 7-iphi [1/3SM], revert the ordering on C side in order to keep convention.
    yr = mCentersOfCellsPhiDir[iphi2 + nphiIndex / 3];
  } else if (GetSMType(nSupMod) == EMCAL_HALF) {
    if (nSupMod % 2 != 0)
      iphi2 = (nphiIndex / 2 - 1) - iphi; // 11-iphi [1/2SM], revert the ordering on C side in order to keep
                                          // convention.
    yr = mCentersOfCellsPhiDir[iphi2 + nphiIndex / 2];
  } else if (GetSMType(nSupMod) == EMCAL_THIRD) {
    if (nSupMod % 2 != 0)
      iphi2 = (nphiIndex / 3 - 1) - iphi; // 7-iphi [1/3SM], revert the ordering on C side in order to keep convention.
    yr = mCentersOfCellsPhiDir[iphi2 + nphiIndex / 3];
  } else {
    if (nSupMod % 2 != 0)
      iphi2 = (nphiIndex - 1) - iphi; // 23-iphi, revert the ordering on C side in order to keep convention.
    yr = mCentersOfCellsPhiDir[iphi2];
  }

  LOG(DEBUG) << "absId " << absId << " nSupMod " << nSupMod << " iphi " << iphi << " ieta " << ieta << " xr " << xr
             << " yr " << yr << " zr " << zr;
  return Point3D<double>(xr, yr, zr);
}

void Geometry::CreateListOfTrd1Modules()
{
  LOG(DEBUG2) << " o2::emcal::Geometry::CreateListOfTrd1Modules() started\n";

  if (!mShishKebabTrd1Modules.size()) {
    for (int iz = 0; iz < mNZ; iz++) {
      if (iz == 0) {
        //        mod  = new AliEMCALShishKebabTrd1Module(TMath::Pi()/2.,this);
        mShishKebabTrd1Modules.emplace_back(ShishKebabTrd1Module(TMath::Pi() / 2., this));
      } else {
        mShishKebabTrd1Modules.emplace_back(ShishKebabTrd1Module(mShishKebabTrd1Modules.back()));
      }
    }
  } else {
    LOG(DEBUG2) << " Already exits :\n";
  }

  ShishKebabTrd1Module& mod = mShishKebabTrd1Modules.back();
  mEtaMaxOfTRD1 = mod.GetMaxEtaOfModule();
  LOG(DEBUG2) << " mShishKebabTrd1Modules has " << mShishKebabTrd1Modules.size() << " modules : max eta "
              << std::setw(5) << std::setprecision(4) << mEtaMaxOfTRD1;

  // define grid for cells in eta(z) and x directions in local coordinates system of SM
  // Works just for 2x2 case only -- ?? start here
  //
  //
  // Define grid for cells in phi(y) direction in local coordinates system of SM
  // as for 2X2 as for 3X3 - Nov 8,2006
  //
  LOG(DEBUG2) << " Cells grid in phi directions : size " << mCentersOfCellsPhiDir.size();

  Int_t ind = 0; // this is phi index
  Int_t ieta = 0, nModule = 0;
  Double_t xr = 0., zr = 0., theta = 0., phi = 0., eta = 0., r = 0., x = 0., y = 0.;
  TVector3 vglob;
  Double_t ytCenterModule = 0.0, ytCenterCell = 0.0;

  mCentersOfCellsPhiDir.resize(mNPhi * mNPHIdiv);
  mPhiCentersOfCells.resize(mNPhi * mNPHIdiv);

  Double_t r0 = mIPDistance + mLongModuleSize / 2.;
  for (Int_t it = 0; it < mNPhi; it++) {                             // cycle on modules
    ytCenterModule = -mParSM[1] + mPhiModuleSize * (2 * it + 1) / 2; // center of module
    for (Int_t ic = 0; ic < mNPHIdiv; ic++) {                        // cycle on cells in module
      if (mNPHIdiv == 2) {
        ytCenterCell = ytCenterModule + mPhiTileSize * (2 * ic - 1) / 2.;
      } else if (mNPHIdiv == 3) {
        ytCenterCell = ytCenterModule + mPhiTileSize * (ic - 1);
      } else if (mNPHIdiv == 1) {
        ytCenterCell = ytCenterModule;
      }
      mCentersOfCellsPhiDir[ind] = ytCenterCell;
      // Define grid on phi direction
      // Grid is not the same for different eta bin;
      // Effect is small but is still here
      phi = TMath::ATan2(ytCenterCell, r0);
      mPhiCentersOfCells[ind] = phi;

      LOG(DEBUG2) << " ind " << std::setw(2) << std::setprecision(2) << ind << " : y " << std::setw(8)
                  << std::setprecision(3) << mCentersOfCellsPhiDir[ind];
      ind++;
    }
  }

  mCentersOfCellsEtaDir.resize(mNZ * mNETAdiv);
  mCentersOfCellsXDir.resize(mNZ * mNETAdiv);
  mEtaCentersOfCells.resize(mNZ * mNETAdiv * mNPhi * mNPHIdiv);

  LOG(DEBUG2) << " Cells grid in eta directions : size " << mCentersOfCellsEtaDir.size();

  for (Int_t it = 0; it < mNZ; it++) {
    const ShishKebabTrd1Module& trd1 = GetShishKebabModule(it);
    nModule = mNPhi * it;
    for (Int_t ic = 0; ic < mNETAdiv; ic++) {
      if (mNPHIdiv == 2) {
        trd1.GetCenterOfCellInLocalCoordinateofSM(ic, xr, zr); // case of 2X2
        auto indexinsm = GetCellPhiEtaIndexInSModule(0, nModule, 0, ic);
        ieta = std::get<1>(indexinsm);
      }
      if (mNPHIdiv == 3) {
        trd1.GetCenterOfCellInLocalCoordinateofSM3X3(ic, xr, zr); // case of 3X3
        auto indexinsm = GetCellPhiEtaIndexInSModule(0, nModule, 0, ic);
        ieta = std::get<1>(indexinsm);
      }
      if (mNPHIdiv == 1) {
        trd1.GetCenterOfCellInLocalCoordinateofSM1X1(xr, zr); // case of 1X1
        auto indexinsm = GetCellPhiEtaIndexInSModule(0, nModule, 0, ic);
        ieta = std::get<1>(indexinsm);
      }
      mCentersOfCellsXDir[ieta] = float(xr) - mParSM[0];
      mCentersOfCellsEtaDir[ieta] = float(zr) - mParSM[2];
      // Define grid on eta direction for each bin in phi
      for (int iphi = 0; iphi < mCentersOfCellsPhiDir.size(); iphi++) {
        x = xr + trd1.GetRadius();
        y = mCentersOfCellsPhiDir[iphi];
        r = TMath::Sqrt(x * x + y * y + zr * zr);
        theta = TMath::ACos(zr / r);
        eta = ShishKebabTrd1Module::ThetaToEta(theta);
        //        ind   = ieta*fCentersOfCellsPhiDir.GetSize() + iphi;
        ind = iphi * mCentersOfCellsEtaDir.size() + ieta;
        mEtaCentersOfCells[ind] = eta;
      }
      // printf(" ieta %i : xr + trd1->GetRadius() %f : zr %f : eta %f \n", ieta, xr + trd1->GetRadius(), zr, eta);
    }
  }

  for (Int_t i = 0; i < mCentersOfCellsEtaDir.size(); i++) {
    LOG(DEBUG2) << " ind " << std::setw(2) << std::setprecision(2) << i + 1 << " : z " << std::setw(8)
                << std::setprecision(3) << mCentersOfCellsEtaDir[i] << " : x " << std::setw(8)
                << std::setprecision(3) << mCentersOfCellsXDir[i];
  }
}

const ShishKebabTrd1Module& Geometry::GetShishKebabModule(Int_t neta) const
{
  if (mShishKebabTrd1Modules.size() && neta >= 0 && neta < mShishKebabTrd1Modules.size())
    return mShishKebabTrd1Modules.at(neta);
  throw InvalidModuleException(neta, mShishKebabTrd1Modules.size());
}

Bool_t Geometry::Impact(const TParticle* particle) const
{
  Bool_t in = kFALSE;
  Int_t absID = 0;
  Point3D<double> vimpact = {0, 0, 0};

  ImpactOnEmcal({particle->Vx(), particle->Vy(), particle->Vz()}, particle->Theta(), particle->Phi(), absID, vimpact);

  if (absID >= 0)
    in = kTRUE;

  return in;
}

void Geometry::ImpactOnEmcal(const Point3D<double>& vtx, Double_t theta, Double_t phi, Int_t& absId, Point3D<double>& vimpact) const
{
  Vector3D<double> p(TMath::Sin(theta) * TMath::Cos(phi), TMath::Sin(theta) * TMath::Sin(phi), TMath::Cos(theta));

  vimpact.SetXYZ(0, 0, 0);
  absId = -1;
  if (phi == 0 || theta == 0)
    return;

  Vector3D<double> direction;
  Double_t factor = (mIPDistance - vtx.Y()) / p.Y();
  direction = vtx + factor * p;

  // from particle direction -> tower hitted
  absId = GetAbsCellIdFromEtaPhi(direction.Eta(), direction.Phi());

  // tower absID hitted -> tower/module plane (evaluated at the center of the tower)

  Double_t loc[3], loc2[3], loc3[3];
  Double_t glob[3] = {}, glob2[3] = {}, glob3[3] = {};

  try {
    RelPosCellInSModule(absId).GetCoordinates(loc[0], loc[1], loc[2]);
  } catch (InvalidCellIDException& e) {
    LOG(ERROR) << e.what();
    return;
  }

  // loc is cell center of tower
  auto cellindex = GetCellIndex(absId);
  Int_t nSupMod = std::get<0>(cellindex), nModule = std::get<1>(cellindex), nIphi = std::get<2>(cellindex),
        nIeta = std::get<3>(cellindex);
  // look at 2 neighbours-s cell using nIphi={0,1} and nIeta={0,1}
  Int_t nIphi2 = -1, nIeta2 = -1, absId2 = -1, absId3 = -1;
  if (nIeta == 0)
    nIeta2 = 1;
  else
    nIeta2 = 0;
  absId2 = GetAbsCellId(nSupMod, nModule, nIphi, nIeta2);
  if (nIphi == 0)
    nIphi2 = 1;
  else
    nIphi2 = 0;
  absId3 = GetAbsCellId(nSupMod, nModule, nIphi2, nIeta);

  // 2nd point on emcal cell plane
  try {
    RelPosCellInSModule(absId2).GetCoordinates(loc2[0], loc2[1], loc2[2]);
  } catch (InvalidCellIDException& e) {
    LOG(ERROR) << e.what();
    return;
  }

  // 3rd point on emcal cell plane
  try {
    RelPosCellInSModule(absId3).GetCoordinates(loc3[0], loc3[1], loc3[2]);
  } catch (InvalidCellIDException& e) {
    LOG(ERROR) << e.what();
    return;
  }

  // Get Matrix
  const TGeoHMatrix* m = GetMatrixForSuperModule(nSupMod);
  if (m) {
    m->LocalToMaster(loc, glob);
    m->LocalToMaster(loc2, glob2);
    m->LocalToMaster(loc3, glob3);
  } else {
    LOG(FATAL) << "Geo matrixes are not loaded \n";
  }

  // Equation of Plane from glob,glob2,glob3 (Ax+By+Cz+D=0)
  Double_t a = glob[1] * (glob2[2] - glob3[2]) + glob2[1] * (glob3[2] - glob[2]) + glob3[1] * (glob[2] - glob2[2]);
  Double_t b = glob[2] * (glob2[0] - glob3[0]) + glob2[2] * (glob3[0] - glob[0]) + glob3[2] * (glob[0] - glob2[0]);
  Double_t c = glob[0] * (glob2[1] - glob3[1]) + glob2[0] * (glob3[1] - glob[1]) + glob3[0] * (glob[1] - glob2[1]);
  Double_t d = glob[0] * (glob2[1] * glob3[2] - glob3[1] * glob2[2]) +
               glob2[0] * (glob3[1] * glob[2] - glob[1] * glob3[2]) +
               glob3[0] * (glob[1] * glob2[2] - glob2[1] * glob[2]);
  d = -d;

  // shift equation of plane from tower/module center to surface along vector (A,B,C) normal to tower/module plane
  Double_t dist = mLongModuleSize / 2.;
  Double_t norm = TMath::Sqrt(a * a + b * b + c * c);
  Double_t glob4[3] = {};
  Vector3D<double> dir = {a, b, c};
  Point3D<double> point = {glob[0], glob[1], glob[2]};
  if (point.Dot(dir) < 0)
    dist *= -1;
  glob4[0] = glob[0] - dist * a / norm;
  glob4[1] = glob[1] - dist * b / norm;
  glob4[2] = glob[2] - dist * c / norm;
  d = glob4[0] * a + glob4[1] * b + glob4[2] * c;
  d = -d;

  // Line determination (2 points for equation of line : vtx and direction)
  // impact between line (particle) and plane (module/tower plane)
  Double_t den = a * (vtx.X() - direction.X()) + b * (vtx.Y() - direction.Y()) + c * (vtx.Z() - direction.Z());
  if (den == 0) {
    LOG(ERROR) << "ImpactOnEmcal() No solution :\n";
    return;
  }

  Double_t length = a * vtx.X() + b * vtx.Y() + c * vtx.Z() + d;
  length /= den;

  vimpact.SetXYZ(vtx.X() + length * (direction.X() - vtx.X()), vtx.Y() + length * (direction.Y() - vtx.Y()),
                 vtx.Z() + length * (direction.Z() - vtx.Z()));

  // shift vimpact from tower/module surface to center along vector (A,B,C) normal to tower/module plane
  vimpact.SetXYZ(vimpact.Z() + dist * a / norm, vimpact.Y() + dist * b / norm, vimpact.Z() + dist * c / norm);
}

Bool_t Geometry::IsInEMCAL(const Point3D<double>& pnt) const
{
  if (IsInEMCALOrDCAL(pnt) == EMCAL_ACCEPTANCE)
    return kTRUE;
  else
    return kFALSE;
}

Bool_t Geometry::IsInDCAL(const Point3D<double>& pnt) const
{
  if (IsInEMCALOrDCAL(pnt) == DCAL_ACCEPTANCE)
    return kTRUE;
  else
    return kFALSE;
}

o2::emcal::AcceptanceType_t Geometry::IsInEMCALOrDCAL(const Point3D<double>& pnt) const
{
  Double_t r = sqrt(pnt.X() * pnt.X() + pnt.Y() * pnt.Y());

  if (r <= mEnvelop[0])
    return NON_ACCEPTANCE;
  else {
    Double_t theta = TMath::ATan2(r, pnt.Z());
    Double_t eta;
    if (theta == 0)
      eta = 9999;
    else
      eta = -TMath::Log(TMath::Tan(theta / 2.));
    if (eta < mArm1EtaMin || eta > mArm1EtaMax)
      return NON_ACCEPTANCE;

    Double_t phi = TMath::ATan2(pnt.Y(), pnt.X()) * 180. / TMath::Pi();
    if (phi < 0)
      phi += 360; // phi should go from 0 to 360 in this case

    if (phi >= mArm1PhiMin && phi <= mEMCALPhiMax)
      return EMCAL_ACCEPTANCE;
    else if (phi >= mDCALPhiMin && phi <= mDCALStandardPhiMax && TMath::Abs(eta) > mDCALInnerExtandedEta)
      return DCAL_ACCEPTANCE;
    else if (phi > mDCALStandardPhiMax && phi <= mDCALPhiMax)
      return DCAL_ACCEPTANCE;
    return NON_ACCEPTANCE;
  }
}

const TGeoHMatrix* Geometry::GetMatrixForSuperModule(Int_t smod) const
{
  if (smod < 0 || smod > mNumberOfSuperModules)
    LOG(FATAL) << "Wrong supermodule index -> " << smod;

  if (!SMODULEMATRIX[smod]) {
    if (gGeoManager)
      SetMisalMatrix(GetMatrixForSuperModuleFromGeoManager(smod), smod);
    else
      LOG(FATAL) << "Cannot find EMCAL misalignment matrices! Recover them either: \n"
                 << "\t - importing TGeoManager from file geometry.root or \n"
                 << "\t - from OADB in file OADB/EMCAL/EMCALlocal2master.root or \n"
                 << "\t - from OCDB in directory OCDB/EMCAL/Align/Data/ or \n"
                 << "\t - from AliESDs (not in AliAOD) via AliESDRun::GetEMCALMatrix(Int_t superModIndex). \n"
                 << "Store them via AliEMCALGeometry::SetMisalMatrix(Int_t superModIndex)";
  }

  return SMODULEMATRIX[smod];
}

const TGeoHMatrix* Geometry::GetMatrixForSuperModuleFromArray(Int_t smod) const
{
  if (smod < 0 || smod > mNumberOfSuperModules)
    LOG(FATAL) << "Wrong supermodule index -> " << smod;

  return SMODULEMATRIX[smod];
}

const TGeoHMatrix* Geometry::GetMatrixForSuperModuleFromGeoManager(Int_t smod) const
{
  const Int_t buffersize = 255;
  char path[buffersize];
  Int_t tmpType = -1;
  Int_t smOrder = 0;

  // Get the order for SM
  for (Int_t i = 0; i < smod + 1; i++) {
    if (GetSMType(i) == tmpType) {
      smOrder++;
    } else {
      tmpType = GetSMType(i);
      smOrder = 1;
    }
  }

  Int_t smType = GetSMType(smod);
  TString smName = "";

  if (smType == EMCAL_STANDARD)
    smName = "SMOD";
  else if (smType == EMCAL_HALF)
    smName = "SM10";
  else if (smType == EMCAL_THIRD)
    smName = "SM3rd";
  else if (smType == DCAL_STANDARD)
    smName = "DCSM";
  else if (smType == DCAL_EXT)
    smName = "DCEXT";
  else
    LOG(ERROR) << "Unkown SM Type!!\n";

  snprintf(path, buffersize, "/ALIC_1/XEN1_1/%s_%d", smName.Data(), smOrder);

  if (!gGeoManager->cd(path))
    LOG(FATAL) << "Geo manager can not find path " << path << "!\n";

  return gGeoManager->GetCurrentMatrix();
}

void Geometry::RecalculateTowerPosition(Float_t drow, Float_t dcol, const Int_t sm, const Float_t depth,
                                        const Float_t misaligTransShifts[15], const Float_t misaligRotShifts[15],
                                        Float_t global[3]) const
{
  // To use in a print later
  Float_t droworg = drow;
  Float_t dcolorg = dcol;

  if (gGeoManager) {
    // Recover some stuff

    const Int_t nSMod = mNumberOfSuperModules;

    gGeoManager->cd("ALIC_1/XEN1_1");
    TGeoNode* geoXEn1 = gGeoManager->GetCurrentNode();
    TGeoNodeMatrix* geoSM[nSMod];
    TGeoVolume* geoSMVol[nSMod];
    TGeoShape* geoSMShape[nSMod];
    TGeoBBox* geoBox[nSMod];
    TGeoMatrix* geoSMMatrix[nSMod];

    for (int iSM = 0; iSM < nSMod; iSM++) {
      geoSM[iSM] = dynamic_cast<TGeoNodeMatrix*>(geoXEn1->GetDaughter(iSM));
      geoSMVol[iSM] = geoSM[iSM]->GetVolume();
      geoSMShape[iSM] = geoSMVol[iSM]->GetShape();
      geoBox[iSM] = dynamic_cast<TGeoBBox*>(geoSMShape[iSM]);
      geoSMMatrix[iSM] = geoSM[iSM]->GetMatrix();
    }

    if (sm % 2 == 0) {
      dcol = 47. - dcol;
      drow = 23. - drow;
    }

    Int_t istrip = 0;
    Float_t z0 = 0;
    Float_t zb = 0;
    Float_t zIs = 0;

    Float_t x, y, z; // return variables in terry's RF

    //***********************************************************
    // Do not like this: too many hardcoded values, is it not already stored somewhere else?
    //                : need more comments in the code
    //***********************************************************

    Float_t dz = 6.0;   // base cell width in eta
    Float_t dx = 6.004; // base cell width in phi

    // Float_t L = 26.04; // active tower length for hadron (lead+scint+paper)
    // we use the geant numbers 13.87*2=27.74
    Float_t teta1 = 0.;

    // Do some basic checks
    if (dcol >= 47.5 || dcol < -0.5) {
      LOG(ERROR) << "Bad tower coordinate dcol=" << dcol << ", where dcol >= 47.5 || dcol<-0.5; org: " << dcolorg;
      return;
    }
    if (drow >= 23.5 || drow < -0.5) {
      LOG(ERROR) << "Bad tower coordinate drow=" << drow << ", where drow >= 23.5 || drow<-0.5; org: " << droworg;
      return;
    }
    if (sm >= nSMod || sm < 0) {
      LOG(ERROR) << "Bad SM number sm=" << nSMod << ", where sm >= " << sm << " || sm < 0\n";
      return;
    }

    istrip = int((dcol + 0.5) / 2);

    // tapering angle
    teta1 = TMath::DegToRad() * istrip * 1.5;

    // calculation of module corner along z
    // as a function of strip

    for (int is = 0; is <= istrip; is++) {
      teta1 = TMath::DegToRad() * (is * 1.5 + 0.75);
      if (is == 0)
        zIs = zIs + 2 * dz * TMath::Cos(teta1);
      else
        zIs =
          zIs + 2 * dz * TMath::Cos(teta1) + 2 * dz * TMath::Sin(teta1) * TMath::Tan(teta1 - 0.75 * TMath::DegToRad());
    }

    z0 = dz * (dcol - 2 * istrip + 0.5);
    zb = (2 * dz - z0 - depth * TMath::Tan(teta1));

    z = zIs - zb * TMath::Cos(teta1);
    y = depth / TMath::Cos(teta1) + zb * TMath::Sin(teta1);

    x = (drow + 0.5) * dx;

    // moving the origin from terry's RF
    // to the GEANT one

    double xx = y - geoBox[sm]->GetDX();
    double yy = -x + geoBox[sm]->GetDY();
    double zz = z - geoBox[sm]->GetDZ();
    const double localIn[3] = {xx, yy, zz};
    double dglobal[3];
    // geoSMMatrix[sm]->Print();
    // printf("TFF Local    (row = %d, col = %d, x = %3.2f,  y = %3.2f, z = %3.2f)\n", iroworg, icolorg, localIn[0],
    // localIn[1], localIn[2]);
    geoSMMatrix[sm]->LocalToMaster(localIn, dglobal);
    // printf("TFF Global   (row = %2.0f, col = %2.0f, x = %3.2f,  y = %3.2f, z = %3.2f)\n", drow, dcol, dglobal[0],
    // dglobal[1], dglobal[2]);

    // apply global shifts
    if (sm == 2 || sm == 3) { // sector 1
      global[0] = dglobal[0] + misaligTransShifts[3] + misaligRotShifts[3] * TMath::Sin(TMath::DegToRad() * 20);
      global[1] = dglobal[1] + misaligTransShifts[4] + misaligRotShifts[4] * TMath::Cos(TMath::DegToRad() * 20);
      global[2] = dglobal[2] + misaligTransShifts[5];
    } else if (sm == 0 || sm == 1) { // sector 0
      global[0] = dglobal[0] + misaligTransShifts[0];
      global[1] = dglobal[1] + misaligTransShifts[1];
      global[2] = dglobal[2] + misaligTransShifts[2];
    } else {
      LOG(INFO) << "Careful, correction not implemented yet!\n";
      global[0] = dglobal[0];
      global[1] = dglobal[1];
      global[2] = dglobal[2];
    }
  } else {
    LOG(FATAL) << "Geometry boxes information, check that geometry.root is loaded\n";
  }
}

void Geometry::SetMisalMatrix(const TGeoHMatrix* m, Int_t smod) const
{
  if (smod >= 0 && smod < mNumberOfSuperModules) {
    if (!SMODULEMATRIX[smod])
      SMODULEMATRIX[smod] = new TGeoHMatrix(*m); // Set only if not set yet
  } else {
    LOG(FATAL) << "Wrong supermodule index -> " << smod << std::endl;
  }
}

Bool_t Geometry::IsDCALSM(Int_t iSupMod) const
{
  if (mEMCSMSystem[iSupMod] == DCAL_STANDARD || mEMCSMSystem[iSupMod] == DCAL_EXT)
    return kTRUE;

  return kFALSE;
}

Bool_t Geometry::IsDCALExtSM(Int_t iSupMod) const
{
  if (mEMCSMSystem[iSupMod] == DCAL_EXT)
    return kTRUE;

  return kFALSE;
}

Double_t Geometry::GetPhiCenterOfSMSec(Int_t nsupmod) const
{
  int i = nsupmod / 2;
  return mPhiCentersOfSMSec[i];
}

Double_t Geometry::GetPhiCenterOfSM(Int_t nsupmod) const
{
  int i = nsupmod / 2;
  return mPhiCentersOfSM[i];
}

std::tuple<double, double> Geometry::GetPhiBoundariesOfSM(Int_t nSupMod) const
{
  int i;
  if (nSupMod < 0 || nSupMod > 12 + mnSupModInDCAL - 1)
    throw InvalidModuleException(nSupMod, 12 + mnSupModInDCAL);
  i = nSupMod / 2;
  return std::make_tuple((Double_t)mPhiBoundariesOfSM[2 * i], (Double_t)mPhiBoundariesOfSM[2 * i + 1]);
}

std::tuple<double, double> Geometry::GetPhiBoundariesOfSMGap(Int_t nPhiSec) const
{
  if (nPhiSec < 0 || nPhiSec > 5 + mnSupModInDCAL / 2 - 1)
    throw InvalidModuleException(nPhiSec, 5 + mnSupModInDCAL / 2);
  return std::make_tuple(mPhiBoundariesOfSM[2 * nPhiSec + 1], mPhiBoundariesOfSM[2 * nPhiSec + 2]);
}
