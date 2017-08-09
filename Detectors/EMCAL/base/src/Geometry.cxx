// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
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

using namespace o2::EMCAL;

// these initialisations are needed for a singleton
Geometry* Geometry::sGeom = nullptr;
std::string Geometry::sDefaultGeometryName = "EMCAL_COMPLETE12SMV1_DCAL_8SM";

Geometry::Geometry(const Geometry& geo)
  : mEMCGeometry(geo.mEMCGeometry),
    mGeoName(geo.mGeoName),
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
    mSampling(geo.mSampling)
{
  memcpy(mEnvelop, geo.mEnvelop, sizeof(Float_t) * 3);
  memcpy(mParSM, geo.mParSM, sizeof(Float_t) * 3);

  memset(SMODULEMATRIX, 0, sizeof(TGeoHMatrix*) * EMCAL_MODULES);
}

Geometry::Geometry(const std::string_view name, const std::string_view mcname, const std::string_view mctitle)
  : mEMCGeometry(name, mcname, mctitle),
    mGeoName(name),
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
    mSampling(0.)
{
  mGeoName = mEMCGeometry.GetGeoName();
  mKey110DEG = mEMCGeometry.GetKey110DEG();
  mnSupModInDCAL = mEMCGeometry.GetnSupModInDCAL();
  mNCellsInSupMod = mEMCGeometry.GetNCellsInSupMod();
  mNETAdiv = mEMCGeometry.GetNETAdiv();
  mNPHIdiv = mEMCGeometry.GetNPHIdiv();
  mNCellsInModule = mNPHIdiv * mNETAdiv;
  Int_t nSMod = mEMCGeometry.GetNumberOfSuperModules();
  mPhiBoundariesOfSM.Set(nSMod);
  mPhiCentersOfSM.Set(nSMod / 2);
  mPhiCentersOfSMSec.Set(nSMod / 2);
  Int_t i = 0;
  for (Int_t sm = 0; sm < nSMod; sm++) {
    i = sm / 2;
    auto boundaries = mEMCGeometry.GetPhiBoundariesOfSM(sm);
    mPhiBoundariesOfSM[2 * i] = std::get<0>(boundaries);
    mPhiBoundariesOfSM[2 * i + 1] = std::get<1>(boundaries);
  }

  for (Int_t sm = 0; sm < nSMod; sm++) {
    i = sm / 2;
    mPhiCentersOfSM[i] = mEMCGeometry.GetPhiCenterOfSM(sm);
    mPhiCentersOfSMSec[i] = mEMCGeometry.GetPhiCenterOfSMSec(sm);
  }

  mNCells = mEMCGeometry.GetNCells();
  mNPhi = mEMCGeometry.GetNPhi();
  mEnvelop[0] = mEMCGeometry.GetEnvelop(0);
  mEnvelop[1] = mEMCGeometry.GetEnvelop(1);
  mEnvelop[2] = mEMCGeometry.GetEnvelop(2);
  mParSM[0] = mEMCGeometry.GetSuperModulesPar(0);
  mParSM[1] = mEMCGeometry.GetSuperModulesPar(1);
  mParSM[2] = mEMCGeometry.GetSuperModulesPar(2);
  mArm1EtaMin = mEMCGeometry.GetArm1EtaMin();
  mArm1EtaMax = mEMCGeometry.GetArm1EtaMax();
  mArm1PhiMin = mEMCGeometry.GetArm1PhiMin();
  mArm1PhiMax = mEMCGeometry.GetArm1PhiMax();
  mDCALPhiMin = mEMCGeometry.GetDCALPhiMin();
  mDCALPhiMax = mEMCGeometry.GetDCALPhiMax();
  mEMCALPhiMax = mEMCGeometry.GetEMCALPhiMax();
  mDCALStandardPhiMax = mEMCGeometry.GetDCALStandardPhiMax();
  mDCALInnerExtandedEta = mEMCGeometry.GetDCALInnerExtandedEta();
  mShellThickness = mEMCGeometry.GetShellThickness();
  mZLength = mEMCGeometry.GetZLength();
  mSampling = mEMCGeometry.GetSampling();
  mEtaModuleSize = mEMCGeometry.GetEtaModuleSize();
  mPhiModuleSize = mEMCGeometry.GetPhiModuleSize();
  mEtaTileSize = mEMCGeometry.GetEtaTileSize();
  mPhiTileSize = mEMCGeometry.GetPhiTileSize();
  mNZ = mEMCGeometry.GetNZ();
  mIPDistance = mEMCGeometry.GetIPDistance();
  mLongModuleSize = mEMCGeometry.GetLongModuleSize();

  CreateListOfTrd1Modules();

  memset(SMODULEMATRIX, 0, sizeof(TGeoHMatrix*) * EMCAL_MODULES);

  LOG(DEBUG2) << mEMCGeometry << FairLogger::endl;

  LOG(INFO) << "Name <<" << name << ">>" << FairLogger::endl;
}

Geometry& Geometry::operator=(const Geometry& /*rvalue*/)
{
  LOG(FATAL) << "assignment operator, not implemented\n";
  return *this;
}

Geometry::~Geometry()
{
  if (this == sGeom) {
    LOG(ERROR) << "Do not call delete on me\n";
    return;
  }

  for (Int_t smod = 0; smod < mEMCGeometry.GetNumberOfSuperModules(); smod++) {
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
  Geometry* rv = nullptr;

  if (!sGeom) {
    if (name != std::string("")) { // get default geometry
      sGeom = new Geometry(sDefaultGeometryName.c_str(), mcname, mctitle);
    } else {
      sGeom = new Geometry(name, mcname, mctitle);
    } // end if strcmp(name,"")

    if (EMCGeometry::sInit)
      rv = static_cast<Geometry*>(sGeom);
    else {
      rv = nullptr;
      delete sGeom;
      sGeom = nullptr;
    } // end if fgInit
  } else {
    if (sGeom->GetName() != name) {
      LOG(INFO) << "\n current geometry is " << sGeom->GetName() << " : you should not call " << name
                << FairLogger::endl;

    } // end

    rv = static_cast<Geometry*>(sGeom);
  } // end if fgGeom

  return rv;
}

Geometry* Geometry::GetInstanceFromRunNumber(Int_t runNumber, const std::string_view geoName,
                                             const std::string_view mcname, const std::string_view mctitle)
{
  using boost::algorithm::contains;

  // printf("AliEMCALGeometry::GetInstanceFromRunNumber() - run %d, geoName <<%s>> \n",runNumber,geoName.Data());

  bool showInfo = !(getenv("HLT_ONLINE_MODE") && strcmp(getenv("HLT_ONLINE_MODE"), "on") == 0);

  if (runNumber >= 104064 && runNumber < 140000) {
    // 2009-2010 runs
    // First year geometry, 4 SM.

    if (showInfo) {
      if (contains(geoName, "FIRSTYEARV1") && geoName != std::string("")) {
        LOG(INFO) << "o2::EMCAL::Geometry::GetInstanceFromRunNumber() *** ATTENTION *** \n"
                  << "\t Specified geometry name <<" << geoName << ">> for run " << runNumber
                  << " is not considered! \n"
                  << "\t In use <<EMCAL_FIRSTYEARV1>>, check run number and year\n";
      } else {
        LOG(INFO)
          << "o2::EMCAL::Geometry::GetInstanceFromRunNumber() - Initialized geometry with name <<EMCAL_FIRSTYEARV1>>\n";
      }
    }

    return Geometry::GetInstance("EMCAL_FIRSTYEARV1", mcname, mctitle);
  } else if (runNumber >= 140000 && runNumber <= 170593) {
    // Almost complete EMCAL geometry, 10 SM. Year 2011 configuration

    if (showInfo) {
      if (contains(geoName, "COMPLETEV1") && geoName != std::string("")) {
        LOG(INFO) << "o2::EMCAL::Geometry::GetInstanceFromRunNumber() *** ATTENTION *** \n"
                  << "\t Specified geometry name <<" << geoName << ">> for run " << runNumber
                  << " is not considered! \n"
                  << "\t In use <<EMCAL_COMPLETEV1>>, check run number and year\n";
      } else {
        LOG(INFO)
          << "o2::EMCAL::Geometry::GetInstanceFromRunNumber() - Initialized geometry with name <<EMCAL_COMPLETEV1>>\n";
      }
    }
    return Geometry::GetInstance("EMCAL_COMPLETEV1", mcname, mctitle);
  } else if (runNumber > 176000 && runNumber <= 197692) {
    // Complete EMCAL geometry, 12 SM. Year 2012 and on
    // The last 2 SM were not active, anyway they were there.

    if (showInfo) {
      if (contains(geoName, "COMPLETE12SMV1") && geoName != std::string("")) {
        LOG(INFO) << "o2::EMCAL::Geometry::GetInstanceFromRunNumber() *** ATTENTION *** \n"
                  << "\t Specified geometry name <<" << geoName << " >> for run " << runNumber
                  << " is not considered! \n"
                  << "\t In use <<EMCAL_COMPLETE12SMV1>>, check run number and year\n";
      } else {
        LOG(INFO) << "o2::EMCAL::Geometry::GetInstanceFromRunNumber() - Initialized geometry with name "
                     "<<EMCAL_COMPLETE12SMV1>>\n";
      }
    }
    return Geometry::GetInstance("EMCAL_COMPLETE12SMV1", mcname, mctitle);
  } else // Run 2
  {
    // EMCAL + DCAL geometry, 20 SM. Year 2015 and on

    if (showInfo) {
      if (contains(geoName, "DCAL_8SM") && geoName != std::string("")) {
        LOG(INFO) << "o2::EMCAL::Geometry::GetInstanceFromRunNumber() *** ATTENTION *** \n"
                  << "\t Specified geometry name <<" << geoName << ">> for run " << runNumber
                  << " is not considered! \n"
                  << "\t In use <<EMCAL_COMPLETE12SMV1_DCAL_8SM>>, check run number and year\n";
      } else {
        LOG(INFO) << "o2::EMCAL::Geometry::GetInstanceFromRunNumber() - Initialized geometry with name "
                     "<<EMCAL_COMPLETE12SMV1_DCAL_8SM>>\n";
      }
    }
    return Geometry::GetInstance("EMCAL_COMPLETE12SMV1_DCAL_8SM", mcname, mctitle);
  }
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
  static Double_t tglob[3], tloc[3];
  vloc.GetXYZ(tloc);
  GetGlobal(tloc, tglob, iSM);
  vglob.SetXYZ(tglob[0], tglob[1], tglob[2]);
}

void Geometry::GetGlobal(Int_t absId, Double_t glob[3]) const
{
  double loc[3];

  memset(glob, 0, sizeof(Double_t) * 3);
  try {
    RelPosCellInSModule(absId, loc);
  } catch (InvalidCellIDException& e) {
    LOG(ERROR) << e.what() << FairLogger::endl;
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

void Geometry::GetModuleIndexesFromCellIndexesInSModule(Int_t nSupMod, Int_t iphi, Int_t ieta, Int_t& iphim,
                                                        Int_t& ietam, Int_t& nModule) const
{
  Int_t nphi = -1;
  nphi = GetNumberOfModuleInPhiDirection(nSupMod);

  ietam = ieta / mNETAdiv;
  iphim = iphi / mNPHIdiv;
  nModule = ietam * nphi + iphim;
}

Int_t Geometry::GetAbsCellIdFromCellIndexes(Int_t nSupMod, Int_t iphi, Int_t ieta) const
{
  // Check if the indeces correspond to existing SM or tower indeces
  if (iphi < 0 || iphi >= EMCAL_ROWS || ieta < 0 || ieta >= EMCAL_COLS || nSupMod < 0 ||
      nSupMod >= GetNumberOfSuperModules()) {
    LOG(DEBUG) << "Wrong cell indexes : SM " << nSupMod << ", column (eta) " << ieta << ", row (phi) " << iphi
               << FairLogger::endl;
    return -1;
  }

  static Int_t ietam = -1, iphim = -1, nModule = -1;
  static Int_t nIeta = -1, nIphi = -1; // cell indexes in module

  GetModuleIndexesFromCellIndexesInSModule(nSupMod, iphi, ieta, ietam, iphim, nModule);

  nIeta = ieta % mNETAdiv;
  nIeta = mNETAdiv - 1 - nIeta;
  nIphi = iphi % mNPHIdiv;

  return GetAbsCellId(nSupMod, nModule, nIphi, nIeta);
}

Int_t Geometry::SuperModuleNumberFromEtaPhi(Double_t eta, Double_t phi) const
{
  Int_t i = 0;

  if (TMath::Abs(eta) > mEtaMaxOfTRD1)
    throw InvalidPositionException(eta, phi);

  phi = TVector2::Phi_0_2pi(phi); // move phi to (0,2pi) boundaries
  Int_t nphism = mEMCGeometry.GetNumberOfSuperModules() / 2;
  Int_t nSupMod = 0;
  for (i = 0; i < nphism; i++) {
    if (phi >= mPhiBoundariesOfSM[2 * i] && phi <= mPhiBoundariesOfSM[2 * i + 1]) {
      nSupMod = 2 * i;
      if (eta < 0.0)
        nSupMod++;

      if (GetSMType(nSupMod) == DCAL_STANDARD) { // Gap between DCAL
        if (TMath::Abs(eta) < GetNEta() / 3 * (mEMCGeometry.GetTrd1Angle()) * TMath::DegToRad())
          throw InvalidPositionException(eta, phi);
      }

      LOG(DEBUG) << "eta " << eta << " phi " << phi << " (" << std::setw(5) << std::setprecision(2)
                 << phi * TMath::RadToDeg() << ") : nSupMod " << nSupMod << ": #bound " << i << FairLogger::endl;
      return nSupMod;
    }
  }
  throw InvalidPositionException(eta, phi);
}

void Geometry::ShiftOnlineToOfflineCellIndexes(Int_t sm, Int_t& iphi, Int_t& ieta) const
{
  if (sm == 13 || sm == 15 || sm == 17) {
    // DCal odd SMs
    ieta -= 16; // Same cabling mapping as for EMCal, not considered offline.
  } else if (sm == 18 || sm == 19) {
    // DCal 1/3 SMs
    iphi -= 16; // Needed due to cabling mistake.
  }
}

void Geometry::ShiftOfflineToOnlineCellIndexes(Int_t sm, Int_t& iphi, Int_t& ieta) const
{
  if (sm == 13 || sm == 15 || sm == 17) {
    // DCal odd SMs
    ieta += 16; // Same cabling mapping as for EMCal, not considered offline.
  } else if (sm == 18 || sm == 19) {
    // DCal 1/3 SMs
    iphi += 16; // Needed due to cabling mistake.
  }
}

Int_t Geometry::GetAbsCellIdFromEtaPhi(Double_t eta, Double_t phi) const
{
  Int_t i = 0, ieta = -1, iphi = -1, etaShift = 0, neta = -1, nphi = -1;
  Double_t absEta = 0.0, d = 0.0, dmin = 0.0, phiLoc = 0;
  Int_t nSupMod = SuperModuleNumberFromEtaPhi(eta, phi);

  // phi index first
  phi = TVector2::Phi_0_2pi(phi);
  phiLoc = phi - mPhiCentersOfSMSec[nSupMod / 2];
  nphi = mPhiCentersOfCells.GetSize();
  if (GetSMType(nSupMod) == EMCAL_HALF)
    nphi /= 2;
  else if (GetSMType(nSupMod) == EMCAL_THIRD)
    nphi /= 3;
  else if (GetSMType(nSupMod) == DCAL_EXT)
    nphi /= 3;

  dmin = TMath::Abs(mPhiCentersOfCells[0] - phiLoc);
  iphi = 0;
  for (i = 1; i < nphi; i++) {
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
  absEta = TMath::Abs(eta);
  neta = mCentersOfCellsEtaDir.GetSize();
  etaShift = iphi * neta;
  ieta = 0;
  if (GetSMType(nSupMod) == DCAL_STANDARD)
    ieta += 16; // jump 16 cells for DCSM
  dmin = TMath::Abs(mEtaCentersOfCells[etaShift + ieta] - absEta);
  for (i = ieta + 1; i < neta; i++) {
    d = TMath::Abs(mEtaCentersOfCells[i + etaShift] - absEta);
    if (d < dmin) {
      dmin = d;
      ieta = i;
    }
  }

  if (GetSMType(nSupMod) == DCAL_STANDARD)
    ieta -= 16; // jump 16 cells for DCSM

  LOG(DEBUG2) << " ieta " << ieta << " : dmin " << dmin << " (eta=" << eta << ") : nSupMod " << nSupMod
              << FairLogger::endl;

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

  return std::make_tuple(int(nModule / nphi), int(nModule % nphi));
}

std::tuple<double, double> Geometry::GetCellPhiEtaIndexInSModule(Int_t nSupMod, Int_t nModule, Int_t nIphi,
                                                                 Int_t nIeta) const
{
  auto indices = GetModulePhiEtaIndexInSModule(nSupMod, nModule);
  Int_t iphim = std::get<0>(indices), ietam = std::get<1>(indices);

  //  ieta  = ietam*fNETAdiv + (1-nIeta); // x(module) = -z(SM)
  Int_t ieta = ietam * mNETAdiv + (mNETAdiv - 1 - nIeta); // x(module) = -z(SM)
  Int_t iphi = iphim * mNPHIdiv + nIphi;                  // y(module) =  y(SM)

  if (iphi < 0 || ieta < 0)
    LOG(DEBUG) << " nSupMod " << nSupMod << " nModule " << nModule << " nIphi " << nIphi << " nIeta " << nIeta
               << " => ieta " << ieta << " iphi " << iphi << FairLogger::endl;
  return std::make_tuple(iphi, ieta);
}

void Geometry::RelPosCellInSModule(Int_t absId, Double_t& xr, Double_t& yr, Double_t& zr) const
{
  // Shift index taking into account the difference between standard SM
  // and SM of half (or one third) size in phi direction

  const Int_t PHIINDEX = mCentersOfCellsPhiDir.GetSize();
  Double_t zshift = 0.5 * GetDCALInnerEdge();

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
    ieta2 = (mCentersOfCellsEtaDir.GetSize() - 1) -
            ieta; // 47-ieta, revert the ordering on A side in order to keep convention.

  if (GetSMType(nSupMod) == DCAL_STANDARD && nSupMod % 2)
    ieta2 += 16; // DCAL revert the ordering on C side ...
  zr = mCentersOfCellsEtaDir.At(ieta2);
  if (GetSMType(nSupMod) == DCAL_STANDARD)
    zr -= zshift; // DCAL shift (SMALLER SM)
  xr = mCentersOfCellsXDir.At(ieta2);

  // Get phi position. Careful with ALICE conventions (increase index increase phi)
  Int_t iphi2 = iphi;
  if (GetSMType(nSupMod) == DCAL_EXT) {
    if (nSupMod % 2 != 0)
      iphi2 = (PHIINDEX / 3 - 1) - iphi; // 7-iphi [1/3SM], revert the ordering on C side in order to keep convention.
    yr = mCentersOfCellsPhiDir.At(iphi2 + PHIINDEX / 3);
  } else if (GetSMType(nSupMod) == EMCAL_HALF) {
    if (nSupMod % 2 != 0)
      iphi2 = (PHIINDEX / 2 - 1) - iphi; // 11-iphi [1/2SM], revert the ordering on C side in order to keep
                                         // convention.
    yr = mCentersOfCellsPhiDir.At(iphi2 + PHIINDEX / 4);
  } else if (GetSMType(nSupMod) == EMCAL_THIRD) {
    if (nSupMod % 2 != 0)
      iphi2 = (PHIINDEX / 3 - 1) - iphi; // 7-iphi [1/3SM], revert the ordering on C side in order to keep convention.
    yr = mCentersOfCellsPhiDir.At(iphi2 + PHIINDEX / 3);
  } else {
    if (nSupMod % 2 != 0)
      iphi2 = (PHIINDEX - 1) - iphi; // 23-iphi, revert the ordering on C side in order to keep conventi
    yr = mCentersOfCellsPhiDir.At(iphi2);
  }

  LOG(DEBUG) << "absId " << absId << " nSupMod " << nSupMod << " iphi " << iphi << " ieta " << ieta << " xr " << xr
             << " yr " << yr << " zr " << zr << FairLogger::endl;
}

void Geometry::RelPosCellInSModule(Int_t absId, Double_t loc[3]) const
{
  memset(loc, 0, sizeof(Double_t) * 3);
  RelPosCellInSModule(absId, loc[0], loc[1], loc[2]);
}

void Geometry::RelPosCellInSModule(Int_t absId, TVector3& vloc) const
{
  Double_t loc[3];
  try {
    RelPosCellInSModule(absId, loc);
    vloc.SetXYZ(loc[0], loc[1], loc[2]);
  } catch (InvalidCellIDException& e) {
    vloc.SetXYZ(0, 0, 0);
  }
}

void Geometry::RelPosCellInSModule(Int_t absId, Double_t distEff, Double_t& xr, Double_t& yr, Double_t& zr) const
{
  // Shift index taking into account the difference between standard SM
  // and SM of half (or one third) size in phi direction

  const Int_t kNphiIndex = mCentersOfCellsPhiDir.GetSize();
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
    ietam = (mCentersOfCellsEtaDir.GetSize() / 2 - 1) -
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
      iphi2 = (kNphiIndex / 3 - 1) - iphi; // 7-iphi [1/3SM], revert the ordering on C side in order to keep convention.
    yr = mCentersOfCellsPhiDir.At(iphi2 + kNphiIndex / 3);
  } else if (GetSMType(nSupMod) == EMCAL_HALF) {
    if (nSupMod % 2 != 0)
      iphi2 = (kNphiIndex / 2 - 1) - iphi; // 11-iphi [1/2SM], revert the ordering on C side in order to keep
                                           // convention.
    yr = mCentersOfCellsPhiDir.At(iphi2 + kNphiIndex / 2);
  } else if (GetSMType(nSupMod) == EMCAL_THIRD) {
    if (nSupMod % 2 != 0)
      iphi2 = (kNphiIndex / 3 - 1) - iphi; // 7-iphi [1/3SM], revert the ordering on C side in order to keep convention.
    yr = mCentersOfCellsPhiDir.At(iphi2 + kNphiIndex / 3);
  } else {
    if (nSupMod % 2 != 0)
      iphi2 = (kNphiIndex - 1) - iphi; // 23-iphi, revert the ordering on C side in order to keep convention.
    yr = mCentersOfCellsPhiDir.At(iphi2);
  }

  LOG(DEBUG) << "absId " << absId << " nSupMod " << nSupMod << " iphi " << iphi << " ieta " << ieta << " xr " << xr
             << " yr " << yr << " zr " << zr << FairLogger::endl;
}

void Geometry::CreateListOfTrd1Modules()
{
  LOG(DEBUG2) << " o2::EMCAL::Geometry::CreateListOfTrd1Modules() started\n";

  if (!mShishKebabTrd1Modules.size()) {
    for (int iz = 0; iz < mEMCGeometry.GetNZ(); iz++) {
      if (iz == 0) {
        //        mod  = new AliEMCALShishKebabTrd1Module(TMath::Pi()/2.,this);
        mShishKebabTrd1Modules.emplace_back(ShishKebabTrd1Module(TMath::Pi() / 2., &mEMCGeometry));
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
              << std::setw(5) << std::setprecision(4) << mEtaMaxOfTRD1 << FairLogger::endl;

  // define grid for cells in eta(z) and x directions in local coordinates system of SM
  // Works just for 2x2 case only -- ?? start here
  //
  //
  // Define grid for cells in phi(y) direction in local coordinates system of SM
  // as for 2X2 as for 3X3 - Nov 8,2006
  //
  LOG(DEBUG2) << " Cells grid in phi directions : size " << mCentersOfCellsPhiDir.GetSize() << FairLogger::endl;

  Int_t ind = 0; // this is phi index
  Int_t ieta = 0, nModule = 0;
  Double_t xr = 0., zr = 0., theta = 0., phi = 0., eta = 0., r = 0., x = 0., y = 0.;
  TVector3 vglob;
  Double_t ytCenterModule = 0.0, ytCenterCell = 0.0;

  mCentersOfCellsPhiDir.Set(mNPhi * mNPHIdiv);
  mPhiCentersOfCells.Set(mNPhi * mNPHIdiv);

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
      mCentersOfCellsPhiDir.AddAt(ytCenterCell, ind);
      // Define grid on phi direction
      // Grid is not the same for different eta bin;
      // Effect is small but is still here
      phi = TMath::ATan2(ytCenterCell, r0);
      mPhiCentersOfCells.AddAt(phi, ind);

      LOG(DEBUG2) << " ind " << std::setw(2) << std::setprecision(2) << ind << " : y " << std::setw(8)
                  << std::setprecision(3) << mCentersOfCellsPhiDir.At(ind) << FairLogger::endl;
      ind++;
    }
  }

  mCentersOfCellsEtaDir.Set(mNZ * mNETAdiv);
  mCentersOfCellsXDir.Set(mNZ * mNETAdiv);
  mEtaCentersOfCells.Set(mNZ * mNETAdiv * mNPhi * mNPHIdiv);

  LOG(DEBUG2) << " Cells grid in eta directions : size " << mCentersOfCellsEtaDir.GetSize() << FairLogger::endl;

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
      mCentersOfCellsXDir.AddAt(float(xr) - mParSM[0], ieta);
      mCentersOfCellsEtaDir.AddAt(float(zr) - mParSM[2], ieta);
      // Define grid on eta direction for each bin in phi
      for (int iphi = 0; iphi < mCentersOfCellsPhiDir.GetSize(); iphi++) {
        x = xr + trd1.GetRadius();
        y = mCentersOfCellsPhiDir[iphi];
        r = TMath::Sqrt(x * x + y * y + zr * zr);
        theta = TMath::ACos(zr / r);
        eta = ShishKebabTrd1Module::ThetaToEta(theta);
        //        ind   = ieta*fCentersOfCellsPhiDir.GetSize() + iphi;
        ind = iphi * mCentersOfCellsEtaDir.GetSize() + ieta;
        mEtaCentersOfCells.AddAt(eta, ind);
      }
      // printf(" ieta %i : xr + trd1->GetRadius() %f : zr %f : eta %f \n", ieta, xr + trd1->GetRadius(), zr, eta);
    }
  }

  for (Int_t i = 0; i < mCentersOfCellsEtaDir.GetSize(); i++) {
    LOG(DEBUG2) << " ind " << std::setw(2) << std::setprecision(2) << i + 1 << " : z " << std::setw(8)
                << std::setprecision(3) << mCentersOfCellsEtaDir.At(i) << " : x " << std::setw(8)
                << std::setprecision(3) << mCentersOfCellsXDir.At(i) << FairLogger::endl;
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
  TVector3 vtx(particle->Vx(), particle->Vy(), particle->Vz());
  TVector3 vimpact(0, 0, 0);

  ImpactOnEmcal(vtx, particle->Theta(), particle->Phi(), absID, vimpact);

  if (absID >= 0)
    in = kTRUE;

  return in;
}

void Geometry::ImpactOnEmcal(TVector3 vtx, Double_t theta, Double_t phi, Int_t& absId, TVector3& vimpact) const
{
  TVector3 p(TMath::Sin(theta) * TMath::Cos(phi), TMath::Sin(theta) * TMath::Sin(phi), TMath::Cos(theta));

  vimpact.SetXYZ(0, 0, 0);
  absId = -1;
  if (phi == 0 || theta == 0)
    return;

  TVector3 direction;
  Double_t factor = (mIPDistance - vtx[1]) / p[1];
  direction = vtx + factor * p;

  // from particle direction -> tower hitted
  absId = GetAbsCellIdFromEtaPhi(direction.Eta(), direction.Phi());

  // tower absID hitted -> tower/module plane (evaluated at the center of the tower)

  Double_t loc[3], loc2[3], loc3[3];
  Double_t glob[3] = {}, glob2[3] = {}, glob3[3] = {};

  try {
    RelPosCellInSModule(absId, loc);
  } catch (InvalidCellIDException& e) {
    LOG(ERROR) << e.what() << FairLogger::endl;
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
    RelPosCellInSModule(absId2, loc2);
  } catch (InvalidCellIDException& e) {
    LOG(ERROR) << e.what() << FairLogger::endl;
    return;
  }

  // 3rd point on emcal cell plane
  try {
    RelPosCellInSModule(absId3, loc3);
  } catch (InvalidCellIDException& e) {
    LOG(ERROR) << e.what() << FairLogger::endl;
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
  TVector3 dir(a, b, c);
  TVector3 point(glob[0], glob[1], glob[2]);
  if (point.Dot(dir) < 0)
    dist *= -1;
  glob4[0] = glob[0] - dist * a / norm;
  glob4[1] = glob[1] - dist * b / norm;
  glob4[2] = glob[2] - dist * c / norm;
  d = glob4[0] * a + glob4[1] * b + glob4[2] * c;
  d = -d;

  // Line determination (2 points for equation of line : vtx and direction)
  // impact between line (particle) and plane (module/tower plane)
  Double_t den = a * (vtx(0) - direction(0)) + b * (vtx(1) - direction(1)) + c * (vtx(2) - direction(2));
  if (den == 0) {
    LOG(ERROR) << "ImpactOnEmcal() No solution :\n";
    return;
  }

  Double_t length = a * vtx(0) + b * vtx(1) + c * vtx(2) + d;
  length /= den;

  vimpact.SetXYZ(vtx(0) + length * (direction(0) - vtx(0)), vtx(1) + length * (direction(1) - vtx(1)),
                 vtx(2) + length * (direction(2) - vtx(2)));

  // shift vimpact from tower/module surface to center along vector (A,B,C) normal to tower/module plane
  vimpact.SetXYZ(vimpact(0) + dist * a / norm, vimpact(1) + dist * b / norm, vimpact(2) + dist * c / norm);
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

o2::EMCAL::AcceptanceType_t Geometry::IsInEMCALOrDCAL(const Point3D<double>& pnt) const
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
  if (smod < 0 || smod > mEMCGeometry.GetNumberOfSuperModules())
    LOG(FATAL) << "Wrong supermodule index -> " << smod << FairLogger::endl;

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
  if (smod < 0 || smod > mEMCGeometry.GetNumberOfSuperModules())
    LOG(FATAL) << "Wrong supermodule index -> " << smod << FairLogger::endl;

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

    const Int_t nSMod = mEMCGeometry.GetNumberOfSuperModules();

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
      LOG(ERROR) << "Bad tower coordinate dcol=" << dcol << ", where dcol >= 47.5 || dcol<-0.5; org: " << dcolorg
                 << FairLogger::endl;
      return;
    }
    if (drow >= 23.5 || drow < -0.5) {
      LOG(ERROR) << "Bad tower coordinate drow=" << drow << ", where drow >= 23.5 || drow<-0.5; org: " << droworg
                 << FairLogger::endl;
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
    const double localIn[3] = { xx, yy, zz };
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
  if (smod >= 0 && smod < mEMCGeometry.GetNumberOfSuperModules()) {
    if (!SMODULEMATRIX[smod])
      SMODULEMATRIX[smod] = new TGeoHMatrix(*m); // Set only if not set yet
  } else {
    LOG(FATAL) << "Wrong supermodule index -> " << smod << std::endl;
  }
}

Bool_t Geometry::IsDCALSM(Int_t iSupMod) const
{
  if (mEMCGeometry.GetEMCSystem()[iSupMod] == DCAL_STANDARD || mEMCGeometry.GetEMCSystem()[iSupMod] == DCAL_EXT)
    return kTRUE;

  return kFALSE;
}

Bool_t Geometry::IsDCALExtSM(Int_t iSupMod) const
{
  if (mEMCGeometry.GetEMCSystem()[iSupMod] == DCAL_EXT)
    return kTRUE;

  return kFALSE;
}
