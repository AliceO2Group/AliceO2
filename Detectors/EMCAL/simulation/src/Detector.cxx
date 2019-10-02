// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <algorithm>
#include <iomanip>

#include "TGeoManager.h"
#include "TGeoVolume.h"
#include "TVirtualMC.h"

#include "FairGeoNode.h"
#include "FairRootManager.h"
#include "FairVolume.h"

#include "EMCALBase/Geometry.h"
#include "EMCALBase/Hit.h"
#include "EMCALBase/ShishKebabTrd1Module.h"
#include "EMCALSimulation/Detector.h"
#include "EMCALSimulation/SpaceFrame.h"

#include "SimulationDataFormat/Stack.h"

#include <boost/algorithm/string/predicate.hpp>
#include <boost/range/irange.hpp>

using namespace o2::emcal;

ClassImp(Detector);

Detector::Detector(Bool_t active)
  : o2::base::DetImpl<Detector>("EMC", active),
    mBirkC0(0),
    mBirkC1(0.),
    mBirkC2(0.),
    mHits(o2::utils::createSimVector<Hit>()),
    mGeometry(nullptr),
    mCurrentPrimaryID(-1),
    mCurrentParentID(-1),
    mParentEnergy(0.),
    mParentHasTrackRef(false),
    mSampleWidth(0.),
    mSmodPar0(0.),
    mSmodPar1(0.),
    mSmodPar2(0.),
    mInnerEdge(0.)

{
  using boost::algorithm::contains;
  memset(mParEMOD, 0, sizeof(Double_t) * 5);

  Geometry* geo = GetGeometry();
  if (!geo)
    LOG(FATAL) << "Geometry is nullptr";
  std::string gn = geo->GetName();
  std::transform(gn.begin(), gn.end(), gn.begin(), ::toupper);

  mSampleWidth = Double_t(geo->GetECPbRadThick() + geo->GetECScintThick());

  if (contains(gn, "V1"))
    mSampleWidth += 2. * geo->GetTrd1BondPaperThick();
}

Detector::Detector(const Detector& rhs)
  : o2::base::DetImpl<Detector>(rhs),
    mBirkC0(rhs.mBirkC0),
    mBirkC1(rhs.mBirkC1),
    mBirkC2(rhs.mBirkC2),
    mHits(o2::utils::createSimVector<Hit>()),
    mGeometry(rhs.mGeometry),
    mCurrentPrimaryID(-1),
    mCurrentParentID(-1),
    mParentEnergy(0.),
    mParentHasTrackRef(false),
    mSampleWidth(rhs.mSampleWidth),
    mSmodPar0(rhs.mSmodPar0),
    mSmodPar1(rhs.mSmodPar1),
    mSmodPar2(rhs.mSmodPar2),
    mInnerEdge(rhs.mInnerEdge)

{
  for (int i = 0; i < 5; ++i) {
    mParEMOD[i] = rhs.mParEMOD[i];
  }
}

Detector::~Detector()
{
  o2::utils::freeSimVector(mHits);
}

void Detector::InitializeO2Detector()
{
  // Define sensitive volume
  TGeoVolume* vsense = gGeoManager->GetVolume("SCMX");
  if (vsense)
    AddSensitiveVolume(vsense);
  else
    LOG(ERROR) << "EMCAL Sensitive volume SCMX not found ... No hit creation!";
}

void Detector::EndOfEvent() { Reset(); }

void Detector::ConstructGeometry()
{
  using boost::algorithm::contains;
  LOG(DEBUG) << "Creating EMCAL geometry";

  Geometry* geom = GetGeometry();
  if (!(geom->IsInitialized())) {
    LOG(ERROR) << "ConstructGeometry: EMCAL Geometry class has not been set up.";
  }

  CreateMaterials();

  SpaceFrame emcalframe;
  emcalframe.CreateGeometry();

  CreateEmcalEnvelope();

  // COMPACT, TRD1
  LOG(DEBUG2) << "Shish-Kebab geometry : " << GetTitle();
  CreateShiskebabGeometry();

  geom->DefineSamplingFraction(TVirtualMC::GetMC()->GetName(), TVirtualMC::GetMC()->GetTitle());

  gGeoManager->CheckGeometry();
}

Bool_t Detector::ProcessHits(FairVolume* v)
{
  // TODO Implement handling of parents and primary particle
  Double_t eloss = fMC->Edep();
  if (eloss < DBL_EPSILON)
    return false; // only process hits which actually deposit some energy in the EMCAL
  Geometry* geom = GetGeometry();
  // Obtain detector ID
  // This is not equal to the volume ID of the fair volume
  // EMCAL geometry implementation in VMC works with a copy of the volume and placing it n-times into the mother volume
  // via translation / rotation, so the copy index is the index of a tower / module / supermodule node within a mother
  // volume Additional care needs to be taken for the supermodule index: The copy is connected to a certain supermodule
  // type, which differs for various parts of the detector
  Int_t copyEta, copyPhi, copyMod, copySmod;
  fMC->CurrentVolID(copyEta);       // Tower in module - x-direction
  fMC->CurrentVolOffID(1, copyPhi); // Tower in module - y-direction
  fMC->CurrentVolOffID(3, copyMod); // Module in supermodule
  fMC->CurrentVolOffID(
    4, copySmod); // Supermodule in EMCAL - attention, with respect to a given supermodule type (offsets needed)
  std::string smtype(fMC->CurrentVolOffName(4));
  int offset(0);
  if (smtype == "SM3rd") {
    offset = 10;
  } else if (smtype == "DCSM") {
    offset = 12;
  } else if (smtype == "DCEXT") {
    offset = 18;
  }
  LOG(DEBUG3) << "Supermodule copy " << copySmod << ", module copy " << copyMod << ", y-dir " << copyPhi << ", x-dir "
              << copyEta << ", supermodule ID " << copySmod + offset - 1;
  LOG(DEBUG3) << "path " << fMC->CurrentVolPath();
  LOG(DEBUG3) << "Name of the supermodule type " << fMC->CurrentVolOffName(4) << ", Module name "
              << fMC->CurrentVolOffName(3);

  // porting from aliroot: circumvention of the index problem. GetAbsCellID cannot be used
  // Int_t detID = geom->GetAbsCellId(offset + copySmod - 1, copyMod - 1, copyPhi - 1, copyEta - 1);
  //
  // Swap A side in Phi and C side in Eta due to wrong indexing.
  //
  Int_t smNumber = offset + copySmod - 1, smTypeID = 1;
  auto [iphi, ieta] = geom->GetCellPhiEtaIndexInSModule(smNumber, copyMod - 1, copyPhi - 1, copyEta - 1);
  //iphi = std::get<0>(posetaphi);
  //ieta = std::get<1>(posetaphi);
  if (smNumber % 2 == 0) {
    if (strcmp(fMC->CurrentVolOffName(4), "DCSM") == 0)
      smTypeID = 3; //DCal supermodule. previous design/idea
    else
      smTypeID = 2;
    ieta = ((geom->GetCentersOfCellsEtaDir()).size() * 2 / smTypeID - 1) - ieta; // 47/31-ieta, revert the ordering on A side in order to keep convention.
  } else {
    if (strcmp(fMC->CurrentVolOffName(4), "SM10") == 0)
      smTypeID = 2; //half supermodule. previous design/idea
    if (strcmp(fMC->CurrentVolOffName(4), "SM3rd") == 0)
      smTypeID = 3; //one third (installed in 2012) supermodule
    if (strcmp(fMC->CurrentVolOffName(4), "DCEXT") == 0)
      smTypeID = 3;                                                          //one third (installed in 2012) supermodule
    iphi = ((geom->GetCentersOfCellsPhiDir()).size() / smTypeID - 1) - iphi; // 23/7-iphi, revert the ordering on C side in order to keep convention.
  }

  // Once we know the indexes, calculate the absolute ID
  Int_t detID = geom->GetAbsCellIdFromCellIndexes(smNumber, iphi, ieta);

  Double_t lightyield(eloss);
  if (fMC->TrackCharge())
    lightyield = CalculateLightYield(eloss, fMC->TrackStep(), fMC->TrackCharge());
  lightyield *= geom->GetSampling();

  auto o2stack = static_cast<o2::data::Stack*>(fMC->GetStack());
  auto currenthit = FindHit(detID, mCurrentParentID);
  if (!currenthit) {
    // Condition for new hit:
    // - Processing different partent track (parent track must be produced outside EMCAL)
    // - Inside different cell
    // - First track of the event
    Float_t posX, posY, posZ, momX, momY, momZ, energy;
    fMC->TrackPosition(posX, posY, posZ);
    fMC->TrackMomentum(momX, momY, momZ, energy);
    Double_t time = fMC->TrackTime() * 1e9; // time in ns
    LOG(DEBUG4) << "Adding new hit for parent " << mCurrentParentID << " and cell " << detID << std::endl;

    /// check handling of primary particles
    AddHit(mCurrentParentID, mCurrentPrimaryID, mParentEnergy, detID, Point3D<float>(posX, posY, posZ),
           Vector3D<float>(momX, momY, momZ), time, lightyield);
    o2stack->addHit(GetDetId());
  } else {
    LOG(DEBUG4) << "Adding energy to the current hit" << std::endl;
    currenthit->SetEnergyLoss(currenthit->GetEnergyLoss() + lightyield);
  }

  // In case of new parent track create new track reference
  if (!mParentHasTrackRef) {
    float x, y, z, px, py, pz, e;
    fMC->TrackPosition(x, y, z);
    fMC->TrackMomentum(px, py, pz, e);
    o2::TrackReference trackref(x, y, z, px, py, pz, fMC->TrackLength(), fMC->TrackTime(), mCurrentParentID, GetDetId());
    o2stack->addTrackReference(trackref);
    mParentHasTrackRef = true;
  }

  return true;
}

Hit* Detector::AddHit(Int_t trackID, Int_t primary, Double_t initialEnergy, Int_t detID,
                      const Point3D<float>& pos, const Vector3D<float>& mom, Double_t time, Double_t eLoss)
{
  LOG(DEBUG4) << "Adding hit for track " << trackID << " with position (" << pos.X() << ", "
              << pos.Y() << ", " << pos.Z() << ") and momentum (" << mom.X() << ", " << mom.Y() << ", " << mom.Z()
              << ")  with energy " << initialEnergy << " loosing " << eLoss;
  mHits->emplace_back(primary, trackID, detID, initialEnergy, pos, mom, time, eLoss);
  return &(mHits->back());
}

Double_t Detector::CalculateLightYield(Double_t energydeposit, Double_t tracklength, Int_t charge) const
{
  if (charge == 0)
    return energydeposit; // full energy deposit for neutral particles (photons)
  // Apply Birk's law (copied from G3BIRK)

  Float_t birkC1Mod = 0;
  if (mBirkC0 == 1) { // Apply correction for higher charge states
    if (std::abs(charge) >= 2)
      birkC1Mod = mBirkC1 * 7.2 / 12.6;
    else
      birkC1Mod = mBirkC1;
  }

  Float_t dedxcm = 0.;
  if (tracklength > 0)
    dedxcm = 1000. * energydeposit / tracklength;
  else
    dedxcm = 0;

  return energydeposit / (1. + birkC1Mod * dedxcm + mBirkC2 * dedxcm * dedxcm);
}

Hit* Detector::FindHit(int cellID, int parentID)
{
  auto result = std::find_if(mHits->begin(), mHits->end(), [cellID, parentID](const Hit& hit) { return hit.GetTrackID() == parentID && hit.GetDetectorID() == cellID; });
  if (result == mHits->end())
    return nullptr;
  return &(*result);
}

void Detector::Register()
{
  FairRootManager::Instance()->RegisterAny(addNameTo("Hit").data(), mHits, kTRUE);
}

void Detector::Reset()
{
  LOG(DEBUG) << "Cleaning EMCAL hits ...";
  if (!o2::utils::ShmManager::Instance().isOperational()) {
    mHits->clear();
  }
}

Geometry* Detector::GetGeometry()
{
  if (!mGeometry) {
    mGeometry = Geometry::GetInstanceFromRunNumber(223409);
  }
  if (!mGeometry)
    LOG(ERROR) << "Failure accessing geometry";
  return mGeometry;
}

void Detector::CreateEmcalEnvelope()
{
  using boost::algorithm::contains;

  Geometry* geom = GetGeometry();
  std::string gn = geom->GetName();
  std::transform(gn.begin(), gn.end(), gn.begin(), ::toupper);

  Int_t rotMatrixID(-1); // Will be assigned by the simulation engine
  //  TVirtualMC::GetMC()->Matrix(nmat, theta1, phi1, theta2, phi2, theta3, phi3) - see AliModule
  Matrix(rotMatrixID, 90.0, 0., 90.0, 90.0, 0.0, 0.0);

  // Create the EMCAL Mother Volume (a polygone) within which to place the Detector and named XEN1
  if (contains(gn, "WSUC")) { // TRD1 for WSUC facility
    // Nov 25,2010
    Float_t envelopA[3];
    envelopA[0] = 30.;
    envelopA[1] = 30;
    envelopA[2] = 20;

    TVirtualMC::GetMC()->Gsvolu(geom->GetNameOfEMCALEnvelope(), "BOX", getMediumID(ID_SC), envelopA, 3);
    for (Int_t i = 0; i < 3; i++)
      // Position the EMCAL Mother Volume (XEN1) in WSUC.
      // Look to AliEMCALWsuCosmicRaySetUp.
      TVirtualMC::GetMC()->Gspos(geom->GetNameOfEMCALEnvelope(), 1, "WSUC", 0.0, 0.0, +265., rotMatrixID, "ONLY");
  } else {
    Double_t envelopA[10];
    envelopA[0] = geom->GetArm1PhiMin();                         // minimum phi angle
    envelopA[1] = geom->GetArm1PhiMax() - geom->GetArm1PhiMin(); // angular range in phi
    envelopA[2] = envelopA[1] / geom->GetPhiSuperModule();       // Section of that
    envelopA[3] = 2;                                             // 2: z coordinates
    envelopA[4] = -geom->GetEnvelop(2) / 2.;                     // zmin - includes padding
    envelopA[5] = geom->GetEnvelop(0);                           // rmin at z1 - includes padding
    envelopA[6] = geom->GetEnvelop(1);                           // rmax at z1 - includes padding
    envelopA[7] = geom->GetEnvelop(2) / 2.;                      // zmax includes padding

    envelopA[8] = envelopA[5]; // radii are the same.
    envelopA[9] = envelopA[6];
    // radii are the same.
    TVirtualMC::GetMC()->Gsvolu(geom->GetNameOfEMCALEnvelope(), "PGON", getMediumID(ID_AIR), envelopA,
                                10); // Polygone filled with air

    LOG(DEBUG2) << "ConstructGeometry: " << geom->GetNameOfEMCALEnvelope() << " = " << envelopA[5] << ", "
                << envelopA[6];
    LOG(DEBUG2) << "ConstructGeometry: XU0 = " << envelopA[5] << ", " << envelopA[6];

    // Position the EMCAL Mother Volume (XEN1) in Alice (ALIC)
    TVirtualMC::GetMC()->Gspos(geom->GetNameOfEMCALEnvelope(), 1, "cave", 0.0, 0.0, 0.0, rotMatrixID, "ONLY");
  }
}

void Detector::CreateShiskebabGeometry()
{
  // TRD1
  using boost::algorithm::contains;
  Geometry* g = GetGeometry();
  std::string gn = g->GetName();
  std::transform(gn.begin(), gn.end(), gn.begin(), ::toupper);

  Double_t trd1Angle = g->GetTrd1Angle() * TMath::DegToRad(), tanTrd1 = TMath::Tan(trd1Angle / 2.);
  // see AliModule::fFIdTmedArr
  //  fIdTmedArr = fIdtmed->GetArray() - 1599 ; // see AliEMCAL::::CreateMaterials()
  //  Int_t kIdAIR=1599, kIdPB = 1600, kIdSC = 1601, kIdSTEEL = 1603;
  //  idAL = 1602;
  Double_t par[10], xpos = 0., ypos = 0., zpos = 0.;

  LOG(DEBUG2) << "Name of mother volume: " << g->GetNameOfEMCALEnvelope();
  CreateSupermoduleGeometry(g->GetNameOfEMCALEnvelope());

  auto SMTypeList = g->GetEMCSystem();
  auto tmpType = NOT_EXISTENT;
  std::string namesmtype;
  for (auto i : boost::irange(0, g->GetNumberOfSuperModules())) {
    if (SMTypeList[i] == tmpType)
      continue;
    else
      tmpType = SMTypeList[i];

    switch (tmpType) {
      case EMCAL_STANDARD:
        namesmtype = "SMOD";
        break;
      case EMCAL_HALF:
        namesmtype = "SM10";
        break;
      case EMCAL_THIRD:
        namesmtype = "SM3rd";
        break;
      case DCAL_STANDARD:
        namesmtype = "DCSM";
        break;
      case DCAL_EXT:
        namesmtype = "DCEXT";
        break;
      default:
        LOG(ERROR) << "Unkown SM Type!!";
    };
    LOG(DEBUG2) << "Creating EMCAL module for SM " << namesmtype << std::endl;
    if (namesmtype.length())
      CreateEmcalModuleGeometry(namesmtype, "EMOD");
  }

  // Sensitive SC  (2x2 tiles)
  Double_t parSCM0[5] = {0, 0, 0, 0}, *dummy = nullptr, parTRAP[11];
  if (!contains(gn, "V1")) {
    Double_t wallThickness = g->GetPhiModuleSize() / g->GetNPHIdiv() - g->GetPhiTileSize();
    for (Int_t i = 0; i < 3; i++)
      parSCM0[i] = mParEMOD[i] - wallThickness;
    parSCM0[3] = mParEMOD[3];
    TVirtualMC::GetMC()->Gsvolu("SCM0", "TRD1", getMediumID(ID_AIR), parSCM0, 4);
    TVirtualMC::GetMC()->Gspos("SCM0", 1, "EMOD", 0., 0., 0., 0, "ONLY");
  } else {
    Double_t wTh = g->GetLateralSteelStrip();
    parSCM0[0] = mParEMOD[0] - wTh + tanTrd1 * g->GetTrd1AlFrontThick();
    parSCM0[1] = mParEMOD[1] - wTh;
    parSCM0[2] = mParEMOD[2] - wTh;
    parSCM0[3] = mParEMOD[3] - g->GetTrd1AlFrontThick() / 2.;
    TVirtualMC::GetMC()->Gsvolu("SCM0", "TRD1", getMediumID(ID_AIR), parSCM0, 4);
    Double_t zshift = g->GetTrd1AlFrontThick() / 2.;
    TVirtualMC::GetMC()->Gspos("SCM0", 1, "EMOD", 0., 0., zshift, 0, "ONLY");
    //
    CreateAlFrontPlate("EMOD", "ALFP");
  }

  if (g->GetNPHIdiv() == 2 && g->GetNETAdiv() == 2) {
    // Division to tile size - 1-oct-04
    LOG(DEBUG2) << " Divide SCM0 on y-axis " << g->GetNETAdiv();
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

    LOG(DEBUG2) << " ** TRAP ** ";
    for (Int_t i = 0; i < 11; i++)
      LOG(DEBUG3) << " par[" << std::setw(2) << std::setprecision(2) << i << "] " << std::setw(9)
                  << std::setprecision(4) << parTRAP[i];

    TVirtualMC::GetMC()->Gsvolu("SCMX", "TRAP", getMediumID(ID_SC), parTRAP, 11);
    xpos = +(parSCM0[1] + parSCM0[0]) / 4.;
    TVirtualMC::GetMC()->Gspos("SCMX", 1, "SCMY", xpos, 0.0, 0.0, 0, "ONLY");

    // Using rotation because SCMX should be the same due to Pb tiles
    xpos = -xpos;
    Int_t rotMatrixID(-1);
    Matrix(rotMatrixID, 90.0, 180., 90.0, 270.0, 0.0, 0.0);
    TVirtualMC::GetMC()->Gspos("SCMX", 2, "SCMY", xpos, 0.0, 0.0, rotMatrixID, "ONLY");

    // put LED to the SCM0
    const ShishKebabTrd1Module& mod = g->GetShishKebabTrd1Modules()[0];
    Double_t tanBetta = mod.GetTanBetta();

    Int_t nr = 0;
    ypos = 0.0;
    Double_t xCenterSCMX = (parTRAP[4] + parTRAP[8]) / 2.;
    if (!contains(gn, "V1")) {
      par[1] = parSCM0[2] / 2;            // y
      par[2] = g->GetECPbRadThick() / 2.; // z
      TVirtualMC::GetMC()->Gsvolu("PBTI", "BOX", getMediumID(ID_PB), dummy, 0);

      zpos = -mSampleWidth * g->GetNECLayers() / 2. + g->GetECPbRadThick() / 2.;
      LOG(DEBUG2) << " Pb tiles ";

      for (Int_t iz = 0; iz < g->GetNECLayers(); iz++) {
        par[0] = (parSCM0[0] + tanBetta * mSampleWidth * iz) / 2.;
        xpos = par[0] - xCenterSCMX;
        TVirtualMC::GetMC()->Gsposp("PBTI", ++nr, "SCMX", xpos, ypos, zpos, 0, "ONLY", par, 3);
        LOG(DEBUG3) << iz + 1 << " xpos " << xpos << " zpos " << zpos << " par[0] " << par[0];
        zpos += mSampleWidth;
      }

      LOG(DEBUG2) << " Number of Pb tiles in SCMX " << nr;
    } else {
      // Oct 26, 2010
      // First sheet of paper
      par[1] = parSCM0[2] / 2.;                 // y
      par[2] = g->GetTrd1BondPaperThick() / 2.; // z
      par[0] = parSCM0[0] / 2.;                 // x
      TVirtualMC::GetMC()->Gsvolu("PAP1", "BOX", getMediumID(ID_PAPER), par, 3);

      xpos = par[0] - xCenterSCMX;
      zpos = -parSCM0[3] + g->GetTrd1BondPaperThick() / 2.;
      TVirtualMC::GetMC()->Gspos("PAP1", 1, "SCMX", xpos, ypos, zpos, 0, "ONLY");

      for (auto iz : boost::irange(0, g->GetNECLayers() - 1)) {
        nr = iz + 1;
        Double_t dz = g->GetECScintThick() + g->GetTrd1BondPaperThick() + mSampleWidth * iz;

        // PB + 2 paper sheets
        par[2] = g->GetECPbRadThick() / 2. + g->GetTrd1BondPaperThick(); // z
        par[0] = (parSCM0[0] + tanBetta * dz) / 2.;
        std::string pa(Form("PA%2.2i", nr));
        TVirtualMC::GetMC()->Gsvolu(pa.data(), "BOX", getMediumID(ID_PAPER), par, 3);

        xpos = par[0] - xCenterSCMX;
        zpos = -parSCM0[3] + dz + par[2];
        TVirtualMC::GetMC()->Gspos(pa.data(), 1, "SCMX", xpos, ypos, zpos, 0, "ONLY");

        // Pb
        std::string pb(Form("PB%2.2i", nr));
        par[2] = g->GetECPbRadThick() / 2.; // z
        TVirtualMC::GetMC()->Gsvolu(pb.data(), "BOX", getMediumID(ID_PB), par, 3);
        TVirtualMC::GetMC()->Gspos(pb.data(), 1, pa.data(), 0.0, 0.0, 0.0, 0, "ONLY");
      }
    }
  }
}

void Detector::CreateMaterials()
{
  // media number in idtmed are 1599 to 1698.
  // --- Air ---
  Float_t aAir[4] = {12.0107, 14.0067, 15.9994, 39.948};
  Float_t zAir[4] = {6., 7., 8., 18.};
  Float_t wAir[4] = {0.000124, 0.755267, 0.231781, 0.012827};
  Float_t dAir = 1.20479E-3;
  Mixture(0, "Air$", aAir, zAir, dAir, 4, wAir);

  // --- Lead ---
  Material(1, "Pb$", 207.2, 82, 11.35, 0.56, 0., nullptr, 0);

  // --- The polysterene scintillator (CH) ---
  Float_t aP[2] = {12.011, 1.00794};
  Float_t zP[2] = {6.0, 1.0};
  Float_t wP[2] = {1.0, 1.0};
  Float_t dP = 1.032;

  Mixture(2, "Polystyrene$", aP, zP, dP, -2, wP);

  // --- Aluminium ---
  Material(3, "Al$", 26.98, 13., 2.7, 8.9, 999., nullptr, 0);
  // ---         Absorption length is ignored ^

  // 25-aug-04 by PAI - see  PMD/AliPMDv0.cxx for STEEL definition
  Float_t asteel[4] = {55.847, 51.9961, 58.6934, 28.0855};
  Float_t zsteel[4] = {26., 24., 28., 14.};
  Float_t wsteel[4] = {.715, .18, .1, .005};
  Mixture(4, "STAINLESS STEEL$", asteel, zsteel, 7.88, 4, wsteel);

  // Oct 26,2010 : Multipurpose Copy Paper UNV-21200), weiht 75 g/m**2.
  // *Cellulose C6H10O5
  //    Component C  A=12.01   Z=6.    W=6./21.
  //    Component H  A=1.      Z=1.    W=10./21.
  //    Component O  A=16.     Z=8.    W=5./21.
  Float_t apaper[3] = {12.01, 1.0, 16.0};
  Float_t zpaper[3] = {6.0, 1.0, 8.0};
  Float_t wpaper[3] = {6. / 21., 10. / 21., 5. / 21.};
  Mixture(5, "BondPaper$", apaper, zpaper, 0.75, 3, wpaper);

  // DEFINITION OF THE TRACKING MEDIA
  // Look to the $ALICE_ROOT/data/galice.cuts for particular values
  // of cuts.
  // Don't forget to add a new tracking medium with non-default cuts

  // for EMCAL: idtmed[1599->1698] equivalent to fIdtmed[0->100]

  Int_t isxfld = 2;
  Float_t sxmgmx = 10.0;
  o2::base::Detector::initFieldTrackingParams(isxfld, sxmgmx);

  // Air                                                                         -> idtmed[1599]
  Medium(ID_AIR, "Air$", 0, 0, isxfld, sxmgmx, 10.0, 1.0, 0.1, 0.1, 10.0, nullptr, 0);

  // The Lead                                                                      -> idtmed[1600]
  Medium(ID_PB, "Lead$", 1, 0, isxfld, sxmgmx, 10.0, 0.1, 0.1, 0.1, 0.1, nullptr, 0);

  // The scintillator of the CPV made of Polystyrene scintillator                   -> idtmed[1601]
  float deemax = 0.1; // maximum fractional energy loss in one step (0 < DEEMAX < deemax )
  Medium(ID_SC, "Scintillator$", 2, 1, isxfld, sxmgmx, 10.0, 0.001, deemax, 0.001, 0.001, nullptr, 0);

  // Various Aluminium parts made of Al                                            -> idtmed[1602]
  Medium(ID_AL, "Al$", 3, 0, isxfld, sxmgmx, 10.0, 0.1, 0.1, 0.001, 0.001, nullptr, 0);

  // 25-aug-04 by PAI : see  PMD/AliPMDv0.cxx for STEEL definition                 -> idtmed[1603]
  Medium(ID_STEEL, "S steel$", 4, 0, isxfld, sxmgmx, 10.0, 0.01, 0.1, 0.001, 0.001, nullptr, 0);

  // Oct 26,2010; Nov 24,2010                                                      -> idtmed[1604]
  deemax = 0.01;
  Medium(ID_PAPER, "Paper$", 5, 0, isxfld, sxmgmx, 10.0, deemax, 0.1, 0.001, 0.001, nullptr, 0);

  // Set constants for Birk's Law implentation
  mBirkC0 = 1;
  mBirkC1 = 0.013 / dP;
  mBirkC2 = 9.6e-6 / (dP * dP);

  std::array<std::string, 6> materialNames = {"Air", "Pb", "Scintillator", "Aluminium", "Steel", "Paper"};
  for (int i = 0; i < 6; i++)
    LOG(DEBUG) << "Created material of type " << materialNames[i] << " with global index " << getMediumID(i);
}

void Detector::CreateSupermoduleGeometry(const std::string_view mother)
{
  // 18-may-05; mother="XEN1";
  // child="SMOD" from first to 10th, "SM10" (11th and 12th)
  // "DCSM" from 13th to 18/22th (TRD1 case), "DCEXT"(18th and 19th)  adapted for DCAL, Oct-23-2012

  using boost::algorithm::contains;
  Geometry* g = GetGeometry();
  std::string gn = g->GetName();
  std::transform(gn.begin(), gn.end(), gn.begin(), ::toupper);

  Double_t par[3], xpos = 0., ypos = 0., zpos = 0., rpos = 0., dphi = 0., phi = 0.0, phiRad = 0.;
  Double_t parC[3] = {0};
  std::string smName;
  Int_t tmpType = -1;

  //  ===== define Super Module from air - 14x30 module ==== ;
  LOG(DEBUG2) << "\n ## Super Module | fSampleWidth " << std::setw(5) << std::setprecision(3) << mSampleWidth << " ## "
              << gn;
  par[0] = g->GetShellThickness() / 2.;               // radial
  par[1] = g->GetPhiModuleSize() * g->GetNPhi() / 2.; // phi
  par[2] = g->GetEtaModuleSize() * g->GetNEta() / 2.; // eta

  Int_t nSMod = g->GetNumberOfSuperModules();
  Int_t nphism = nSMod / 2; // 20-may-05
  if (nphism > 0) {
    dphi = g->GetPhiSuperModule();
    rpos = (g->GetEnvelop(0) + g->GetEnvelop(1)) / 2.;
    LOG(DEBUG2) << " rpos " << std::setw(8) << std::setprecision(2) << rpos << " : dphi " << std::setw(6)
                << std::setprecision(1) << dphi << " degree ";
  }

  if (contains(gn, "WSUC")) {
    Int_t nr = 0;
    par[0] = g->GetPhiModuleSize() * g->GetNPhi() / 2.;
    par[1] = g->GetShellThickness() / 2.;
    par[2] = g->GetEtaModuleSize() * g->GetNZ() / 2. + 5;

    TVirtualMC::GetMC()->Gsvolu("SMOD", "BOX", getMediumID(ID_AIR), par, 3);

    LOG(DEBUG2) << "SMOD in WSUC : tmed " << getMediumID(ID_AIR) << " | dx " << std::setw(7) << std::setprecision(2)
                << par[0] << " dy " << std::setw(7) << std::setprecision(2) << par[1] << " dz " << std::setw(7)
                << std::setprecision(2) << par[2] << " (SMOD, BOX)";
    mSmodPar0 = par[0];
    mSmodPar1 = par[1];
    mSmodPar2 = par[2];
    nphism = g->GetNumberOfSuperModules();
    for (Int_t i = 0; i < nphism; i++) {
      xpos = ypos = zpos = 0.0;
      TVirtualMC::GetMC()->Gspos("SMOD", 1, mother.data(), xpos, ypos, zpos, 0, "ONLY");

      LOG(DEBUG2) << " fIdRotm " << std::setw(3) << 0 << " phi " << std::setw(7) << std::setprecision(1) << phi << "("
                  << std::setw(5) << std::setprecision(3) << phiRad << ") xpos " << std::setw(7) << std::setprecision(2)
                  << xpos << " ypos " << std::setw(7) << std::setprecision(2) << ypos << " zpos " << std::setw(7)
                  << std::setprecision(2) << zpos;

      nr++;
    }
  } else { // ALICE
    LOG(DEBUG2) << " par[0] " << std::setw(7) << std::setprecision(2) << par[0] << " (old) ";
    for (Int_t i = 0; i < 3; i++)
      par[i] = g->GetSuperModulesPar(i);
    mSmodPar0 = par[0];
    mSmodPar2 = par[2];

    Int_t SMOrder = -1;
    tmpType = -1;
    for (auto smodnum : boost::irange(0, nSMod)) {
      memcpy(parC, par, sizeof(double) * 3);
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
      switch (tmpType) {
        case EMCAL_STANDARD: {
          smName = "SMOD";
          break;
        }
        case EMCAL_HALF: {
          smName = "SM10";
          parC[1] /= 2.;
          xpos += (par[1] / 2. * TMath::Sin(phiRad));
          ypos -= (par[1] / 2. * TMath::Cos(phiRad));
          break;
        }
        case EMCAL_THIRD: {
          smName = "SM3rd";
          parC[1] /= 3.;
          xpos += (2. * par[1] / 3. * TMath::Sin(phiRad));
          ypos -= (2. * par[1] / 3. * TMath::Cos(phiRad));
          break;
        }
        case DCAL_STANDARD: {
          smName = "DCSM";
          parC[2] *= 2. / 3.;
          zpos = mSmodPar2 + g->GetDCALInnerEdge() / 2.; // 21-sep-04
          break;
        }
        case DCAL_EXT: {
          smName = "DCEXT";
          parC[1] /= 3.;
          xpos += (2. * par[1] / 3. * TMath::Sin(phiRad));
          ypos -= (2. * par[1] / 3. * TMath::Cos(phiRad));
          break;
        }
        default:
          LOG(ERROR) << "Unkown SM Type!!";
          break;
      };

      if (SMOrder == 1) { // first time, create the SM
        TVirtualMC::GetMC()->Gsvolu(smName.data(), "BOX", getMediumID(ID_AIR), parC, 3);

        LOG(DEBUG2) << R"( Super module with name \")" << smName << R"(\" was created in \"box\" with: par[0] = )"
                    << parC[0] << ", par[1] = " << parC[1] << ", par[2] = " << parC[2];
      }

      if (smodnum % 2 == 1) {
        phiy += 180.;
        if (phiy >= 360.)
          phiy -= 360.;
        phiz = 180.;
        zpos *= -1.;
      }

      Int_t rotMatrixID(-1);
      Matrix(rotMatrixID, 90.0, phi, 90.0, phiy, phiz, 0.0);
      TVirtualMC::GetMC()->Gspos(smName.data(), SMOrder, mother.data(), xpos, ypos, zpos, rotMatrixID, "ONLY");

      LOG(DEBUG3) << smName << " : " << std::setw(2) << SMOrder << ", fIdRotm " << std::setw(3) << rotMatrixID
                  << " phi " << std::setw(6) << std::setprecision(1) << phi << "(" << std::setw(5)
                  << std::setprecision(3) << phiRad << ") xpos " << std::setw(7) << std::setprecision(2) << xpos
                  << " ypos " << std::setw(7) << std::setprecision(2) << ypos << " zpos " << std::setw(7)
                  << std::setprecision(2) << zpos << " : i " << smodnum;
    }
  }

  LOG(DEBUG2) << " Number of Super Modules " << nSMod;

  // Steel plate
  if (g->GetSteelFrontThickness() > 0.0) { // 28-mar-05
    par[0] = g->GetSteelFrontThickness() / 2.;
    TVirtualMC::GetMC()->Gsvolu("STPL", "BOX", getMediumID(ID_STEEL), par, 3);

    LOG(DEBUG1) << "tmed " << getMediumID(ID_STEEL) << " | dx " << std::setw(7) << std::setprecision(2) << par[0]
                << " dy " << std::setw(7) << std::setprecision(2) << par[1] << " dz " << std::setw(7)
                << std::setprecision(2) << par[2] << " (STPL) ";

    xpos = -(g->GetShellThickness() - g->GetSteelFrontThickness()) / 2.;
    TVirtualMC::GetMC()->Gspos("STPL", 1, "SMOD", xpos, 0.0, 0.0, 0, "ONLY");
  }
}

void Detector::CreateEmcalModuleGeometry(const std::string_view mother, const std::string_view child)
{
  // 17-may-05; mother="SMOD"; child="EMOD"
  // Oct 26,2010
  using boost::algorithm::contains;
  Geometry* g = GetGeometry();
  std::string gn = g->GetName();
  std::transform(gn.begin(), gn.end(), gn.begin(), ::toupper);

  // Module definition
  Double_t xpos = 0., ypos = 0., zpos = 0.;
  // Double_t trd1Angle = g->GetTrd1Angle()*TMath::DegToRad();tanTrd1 = TMath::Tan(trd1Angle/2.);

  if (mother == "SMOD") {
    mParEMOD[0] = g->GetEtaModuleSize() / 2.;  // dx1
    mParEMOD[1] = g->Get2Trd1Dx2() / 2.;       // dx2
    mParEMOD[2] = g->GetPhiModuleSize() / 2.;  // dy
    mParEMOD[3] = g->GetLongModuleSize() / 2.; // dz
    TVirtualMC::GetMC()->Gsvolu(child.data(), "TRD1", getMediumID(ID_STEEL), mParEMOD, 4);
  }

  Int_t nr = 0;
  Int_t rotMatrixID(-1);
  // X->Z(0, 0); Y->Y(90, 90); Z->X(90, 0)

  for (auto iz : boost::irange(0, g->GetNZ())) {
    const ShishKebabTrd1Module& mod = g->GetShishKebabTrd1Modules()[iz];
    Double_t angle(mod.GetThetaInDegree()), phiOK(0.);

    if (!contains(gn, "WSUC")) { // ALICE
      Matrix(rotMatrixID, 90. - angle, 180., 90.0, 90.0, angle, 0.);
      phiOK = mod.GetCenterOfModule().Phi() * 180. / TMath::Pi();
      LOG(DEBUG4) << std::setw(2) << iz + 1 << " | angle | " << std::setw(6) << std::setprecision(3) << angle << " - "
                  << std::setw(6) << std::setprecision(3) << phiOK << " = " << std::setw(6) << std::setprecision(3)
                  << angle - phiOK << "(eta " << std::setw(5) << std::setprecision(3) << mod.GetEtaOfCenterOfModule()
                  << ")";
      xpos = mod.GetPosXfromR() + g->GetSteelFrontThickness() - mSmodPar0;
      zpos = mod.GetPosZ() - mSmodPar2;

      Int_t iyMax = g->GetNPhi();
      if (mother == "SM10") {
        iyMax /= 2;
      } else if (mother == "SM3rd") {
        iyMax /= 3;
      } else if (mother == "DCEXT") {
        iyMax /= 3;
      } else if (mother == "DCSM") {
        if (iz < 8)
          continue; //!!!DCSM from 8th to 23th
        zpos = mod.GetPosZ() - mSmodPar2 - g->GetDCALInnerEdge() / 2.;
      } else if (mother.compare("SMOD"))
        LOG(ERROR) << "Unknown super module Type!!";

      for (auto iy : boost::irange(0, iyMax)) { // flat in phi
        ypos = g->GetPhiModuleSize() * (2 * iy + 1 - iyMax) / 2.;
        TVirtualMC::GetMC()->Gspos(child.data(), ++nr, mother.data(), xpos, ypos, zpos, rotMatrixID, "ONLY");

        // printf(" %2i xpos %7.2f ypos %7.2f zpos %7.2f fIdRotm %i\n", nr, xpos, ypos, zpos, fIdRotm);
        LOG(DEBUG3) << std::setw(3) << std::setprecision(3) << nr << "(" << std::setw(2) << std::setprecision(2)
                    << iy + 1 << "," << std::setw(2) << std::setprecision(2) << iz + 1 << ")";
      }
      // PH          printf("\n");
    } else { // WSUC
      if (iz == 0)
        Matrix(rotMatrixID, 0., 0., 90., 0., 90., 90.); // (x')z; y'(x); z'(y)
      else
        Matrix(rotMatrixID, 90 - angle, 270., 90.0, 0.0, angle, 90.);

      phiOK = mod.GetCenterOfModule().Phi() * 180. / TMath::Pi();

      LOG(DEBUG4) << std::setw(2) << iz + 1 << " | angle -phiOK | " << std::setw(6) << std::setprecision(3) << angle
                  << " - " << std::setw(6) << std::setprecision(3) << phiOK << " = " << std::setw(6)
                  << std::setprecision(3) << angle - phiOK << "(eta " << std::setw(5) << std::setprecision(3)
                  << mod.GetEtaOfCenterOfModule() << ")";

      zpos = mod.GetPosZ() - mSmodPar2;
      ypos = mod.GetPosXfromR() - mSmodPar1;

      // printf(" zpos %7.2f ypos %7.2f fIdRotm %i\n xpos ", zpos, xpos, fIdRotm);

      for (auto ix : boost::irange(0, g->GetNPhi())) { // flat in phi
        xpos = g->GetPhiModuleSize() * (2 * ix + 1 - g->GetNPhi()) / 2.;
        TVirtualMC::GetMC()->Gspos(child.data(), ++nr, mother.data(), xpos, ypos, zpos, rotMatrixID, "ONLY");
        // printf(" %7.2f ", xpos);
      }
      // printf("");
    }
  }

  LOG(DEBUG2) << " Number of modules in Super Module(" << mother << ") " << nr;
}

void Detector::CreateAlFrontPlate(const std::string_view mother, const std::string_view child)
{
  // Oct 26,2010 : Al front plate : ALFP
  Geometry* g = GetGeometry();

  Double_t trd1Angle = g->GetTrd1Angle() * TMath::DegToRad(), tanTrd1 = TMath::Tan(trd1Angle / 2.);
  Double_t parALFP[5], zposALFP = 0.;

  parALFP[0] = g->GetEtaModuleSize() / 2. - g->GetLateralSteelStrip(); // dx1
  parALFP[1] = parALFP[0] + tanTrd1 * g->GetTrd1AlFrontThick();        // dx2
  parALFP[2] = g->GetPhiModuleSize() / 2. - g->GetLateralSteelStrip(); // dy
  parALFP[3] = g->GetTrd1AlFrontThick() / 2.;                          // dz

  TVirtualMC::GetMC()->Gsvolu(child.data(), "TRD1", getMediumID(ID_AL), parALFP, 4);

  zposALFP = -mParEMOD[3] + g->GetTrd1AlFrontThick() / 2.;
  TVirtualMC::GetMC()->Gspos(child.data(), 1, mother.data(), 0.0, 0.0, zposALFP, 0, "ONLY");
}

void Detector::BeginPrimary()
{
  mCurrentPrimaryID = fMC->GetStack()->GetCurrentTrackNumber();
  mCurrentParentID = mCurrentPrimaryID;
  mParentEnergy = fMC->GetStack()->GetCurrentTrack()->Energy();
  mParentHasTrackRef = false;
  LOG(DEBUG) << "Starting primary " << mCurrentPrimaryID << " with energy " << fMC->GetStack()->GetCurrentTrack()->Energy();
}

void Detector::PreTrack()
{
  auto currenttrack = fMC->GetStack()->GetCurrentTrackNumber();
  LOG(DEBUG) << "Starting new track " << currenttrack << std::endl;
  auto current = fMC->GetStack()->GetCurrentTrack();
  // check if produced outside EMCAL - if yes set as current parent
  Point3D<double> prodvertex{current->Vx(), current->Vy(), current->Vz()};
  if (mGeometry->IsInEMCALOrDCAL(prodvertex) == AcceptanceType_t::NON_ACCEPTANCE) {
    LOG(DEBUG) << "Produced outside EMCAL - Setting new parent track " << currenttrack << " of type " << current->GetPdgCode() << " with energy " << current->Energy()
               << "(R=" << TMath::Sqrt(current->Vx() * current->Vx() + current->Vy() * current->Vy()) << ")" << std::endl;
    mCurrentParentID = currenttrack;
    mParentEnergy = current->Energy();
    mParentHasTrackRef = false;
  }
}

void Detector::FinishPrimary()
{
  LOG(DEBUG) << "Finishing primary " << mCurrentPrimaryID << std::endl;
  // Resetting primary and parent ID
  mCurrentPrimaryID = -1;
  mCurrentParentID = -1;
  mParentHasTrackRef = false;
}