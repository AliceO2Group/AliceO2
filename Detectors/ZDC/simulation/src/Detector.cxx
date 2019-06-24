// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

// FairRoot includes
#include "FairLogger.h"      // for LOG, LOG_IF
#include "FairRootManager.h" // for FairRootManager
#include "FairVolume.h"      // for FairVolume
#include "DetectorsBase/MaterialManager.h"
#include "SimulationDataFormat/Stack.h"
#include "ZDCSimulation/Detector.h"
#include "ZDCSimulation/Hit.h"

#include "TMath.h"
#include "TGeoManager.h"        // for TGeoManager, gGeoManager
#include "TGeoVolume.h"         // for TGeoVolume, TGeoVolumeAssembly
#include "TGeoTube.h"           // for TGeoTube
#include "TGeoCone.h"           // for TGeoCone
#include "TGeoCompositeShape.h" // for TGeoCone
#include "TVirtualMC.h"         // for gMC, TVirtualMC
#include "TString.h"            // for TString, operator+
#include <TRandom.h>
#include <cassert>
#include <fstream>

using namespace o2::zdc;

ClassImp(o2::zdc::Detector);
#define kRaddeg TMath::RadToDeg()

//_____________________________________________________________________________
Detector::Detector(Bool_t active)
  : o2::base::DetImpl<Detector>("ZDC", active),
    mHits(new std::vector<o2::zdc::Hit>),
    mXImpact(-999, -999, -999)
{
  mTrackEta = 999;
  resetHitIndices();
}

//_____________________________________________________________________________
Detector::Detector(const Detector& rhs)
  : o2::base::DetImpl<Detector>(rhs),
    mHits(new std::vector<o2::zdc::Hit>)
{
}

//_____________________________________________________________________________
template <typename T>
int loadLightTable(T& table, int beta, int NRADBINS, std::string filename)
{
  // Retrieve the light yield table
  std::string data;
  std::ifstream input(filename);
  int radiusbin = 0;
  int anglebin = 0;
  int counter = 0;
  float value;
  if (input.is_open()) {
    while (input >> value) {
      counter++;
      table[beta][radiusbin][anglebin] = value;
      radiusbin++;
      if (radiusbin % NRADBINS == 0) {
        radiusbin = 0;
        anglebin++;
      }
    }
    LOG(DEBUG) << "Read " << counter << " values from ZDC data file " << filename;
    input.close();
    return counter;
  } else {
    LOG(ERROR) << "Could not open file " << filename;
    return 0;
  }
}

//_____________________________________________________________________________
void Detector::InitializeO2Detector()
{
  // Define the list of sensitive volumes
  defineSensitiveVolumes();

  std::string inputDir;
  const char* aliceO2env = std::getenv("O2_ROOT");
  if (aliceO2env)
    inputDir = std::string(aliceO2env);
  inputDir += "/share/Detectors/ZDC/simulation/data/";
  //ZN case
  loadLightTable(mLightTableZN, 0, ZNRADIUSBINS, inputDir + "light22620362207s");
  loadLightTable(mLightTableZN, 1, ZNRADIUSBINS, inputDir + "light22620362208s");
  loadLightTable(mLightTableZN, 2, ZNRADIUSBINS, inputDir + "light22620362209s");
  auto elements = loadLightTable(mLightTableZN, 3, ZNRADIUSBINS, inputDir + "light22620362210s");
  assert(elements == ZNRADIUSBINS * ANGLEBINS);
  // check a few values to test correctness of reading from file light22620362207s
  assert(std::abs(mLightTableZN[0][ZNRADIUSBINS - 1][0] - 1.39742) < 1.E-4); // beta=0; radius = ZNRADIUSBINS - 1; anglebin = 2;
  assert(std::abs(mLightTableZN[0][ZNRADIUSBINS - 1][1] - .45017) < 1.E-4);  // beta=1; radius = ZNRADIUSBINS - 1; anglebin = 2;
  assert(std::abs(mLightTableZN[0][0][2] - .47985) < 1.E-4);                 // beta=0; radius = 0; anglebin = 2;
  assert(std::abs(mLightTableZN[0][0][11] - .01358) < 1.E-4);                // beta=0; radius = 0; anglebin = 11;

  //ZP case
  loadLightTable(mLightTableZP, 0, ZPRADIUSBINS, inputDir + "light22620552207s");
  loadLightTable(mLightTableZP, 1, ZPRADIUSBINS, inputDir + "light22620552208s");
  loadLightTable(mLightTableZP, 2, ZPRADIUSBINS, inputDir + "light22620552209s");
  elements = loadLightTable(mLightTableZP, 3, ZPRADIUSBINS, inputDir + "light22620552210s");
  assert(elements == ZPRADIUSBINS * ANGLEBINS);
}

//_____________________________________________________________________________
void Detector::ConstructGeometry()
{
  LOG(DEBUG) << "Creating ZDC  geometry\n";

  createMaterials();

  createAsideBeamLine();
  createCsideBeamLine();
  createMagnets();
  createDetectors();
}

//_____________________________________________________________________________
void Detector::defineSensitiveVolumes()
{
  LOG(INFO) << "defining sensitive for ZDC";
  auto vol = gGeoManager->GetVolume("ZNENV");
  if (vol) {
    AddSensitiveVolume(vol);
    mZNENVVolID = vol->GetNumber(); // initialize id

    AddSensitiveVolume(gGeoManager->GetVolume("ZNF1"));
    AddSensitiveVolume(gGeoManager->GetVolume("ZNF2"));
    AddSensitiveVolume(gGeoManager->GetVolume("ZNF3"));
    AddSensitiveVolume(gGeoManager->GetVolume("ZNF4"));
  } else {
    LOG(FATAL) << "can't find volume ZNENV";
  }
  vol = gGeoManager->GetVolume("ZPENV");
  if (vol) {
    AddSensitiveVolume(vol);
    mZPENVVolID = vol->GetNumber(); // initialize id

    AddSensitiveVolume(gGeoManager->GetVolume("ZPF1"));
    AddSensitiveVolume(gGeoManager->GetVolume("ZPF2"));
    AddSensitiveVolume(gGeoManager->GetVolume("ZPF3"));
    AddSensitiveVolume(gGeoManager->GetVolume("ZPF4"));
  } else {
    LOG(FATAL) << "can't find volume ZPENV";
  }
  // em calorimeter
  vol = gGeoManager->GetVolume("ZEM ");
  if (vol) {
    AddSensitiveVolume(vol);
    mZEMVolID = vol->GetNumber();
    AddSensitiveVolume(gGeoManager->GetVolume("ZEMF"));
  } else {
    LOG(FATAL) << "can't find volume ZEM";
  }
}

// determines detectorID and sectorID from volume and coordinates
void Detector::getDetIDandSecID(TString const& volname, Vector3D<float> const& x,
                                Vector3D<float>& xDet, int& detector, int& sector) const
{
  if (volname.BeginsWith("ZN")) {
    // for the neutron calorimeter

    if (x.Z() > 0) {
      detector = 1; //ZNA
      xDet = x - Vector3D<float>(Geometry::ZNAPOSITION[0], Geometry::ZNAPOSITION[1], Geometry::ZNAPOSITION[2]);

    } else if (x.Z() < 0) {
      detector = 4; //ZNC
      xDet = x - Vector3D<float>(Geometry::ZNCPOSITION[0], Geometry::ZNCPOSITION[1], Geometry::ZNCPOSITION[2]);
    }
    // now determine sector/tower
    if (xDet.X() <= 0.) {
      if (xDet.Y() <= 0.) {
        sector = 1;
      } else
        sector = 3;
    } else {
      if (xDet.Y() <= 0.) {
        sector = 2;
      } else {
        sector = 4;
      }
    }
    return;

  } else if (volname.BeginsWith("ZP")) {
    // proton calorimeter
    if (x.Z() > 0) {
      detector = 2; //ZPA (NB -> DIFFERENT FROM AliRoot!!!)
      xDet = x - Vector3D<float>(Geometry::ZPAPOSITION[0], Geometry::ZPAPOSITION[1], Geometry::ZPAPOSITION[2]);
    } else if (x.Z() < 0) {
      detector = 5; //ZPC (NB -> DIFFERENT FROM AliRoot!!!)
      xDet = x - Vector3D<float>(Geometry::ZPCPOSITION[0], Geometry::ZPCPOSITION[1], Geometry::ZPCPOSITION[2]);
    }

    // determine sector/tower
    if (xDet.X() >= Geometry::ZPDIMENSION[0]) {
      xDet.SetX(Geometry::ZPDIMENSION[0] - 0.01);
    } else if (xDet.X() <= -Geometry::ZPDIMENSION[0]) {
      xDet.SetX(-Geometry::ZPDIMENSION[0] + 0.01);
    }

    float xTow = 2. * xDet.X() / (Geometry::ZPDIMENSION[0]);
    for (int i = 1; i <= 4; i++) {
      if (xTow >= (i - 3) && xTow < (i - 2)) {
        sector = i;
        break;
      }
    }
    return;

  } else if (volname.BeginsWith("ZE")) {
    // electromagnetic calorimeter
    detector = 3;
    xDet = x - Vector3D<float>(Geometry::ZEMPOSITION[0], Geometry::ZEMPOSITION[1], Geometry::ZEMPOSITION[2]);
    sector = (x.X() > 0.) ? 1 : 2;
    return;
  }

  assert(false);
}

void Detector::resetHitIndices()
{
  // reinit hit buffer to null (because we make new hits for each principal track)
  for (int det = 0; det < NUMDETS; ++det) {
    for (int sec = 0; sec < NUMSECS; ++sec) {
      mCurrentHitsIndices[det][sec] = -1;
    }
  }
}

//_____________________________________________________________________________
Bool_t Detector::ProcessHits(FairVolume* v)
{
  // Method called from MC stepping for the sensitive volumes
  TString volname = fMC->CurrentVolName();
  Float_t x[3] = { 0., 0., 0. };
  fMC->TrackPosition(x[0], x[1], x[2]);

  // determine detectorID and sectorID
  int detector = -1;
  int sector = -1;
  Vector3D<float> xImp;
  getDetIDandSecID(volname, Vector3D<float>(x[0], x[1], x[2]),
                   xImp, detector, sector);
  //printf("ProcessHits:  x=(%f, %f, %f)  DET %d  SEC %d\n",x[0], x[1], x[2],detector,sector);
  //printf("                  XImpact=(%f, %f, %f)\n",xImp.X(), xImp.Y(), xImp.Z());

  auto stack = (o2::data::Stack*)fMC->GetStack();
  int trackn = stack->GetCurrentTrackNumber();

  // find out if we are entering into the detector NEU or PRO for the first time
  int volID, copy;
  volID = fMC->CurrentVolID(copy);
  //printf("--- track %d in vol. %d %d  trackMother %d \n",trackn, detector, sector, stack->GetCurrentTrack()->GetMother(0));

  // a new principal track is a track which previously was not seen by any ZDC detector
  // we will account all detector response associated to principal tracks only
  if ((volID == mZNENVVolID || volID == mZPENVVolID || volID == mZEMVolID) && fMC->IsTrackEntering()) {
    if ((mLastPrincipalTrackEntered == -1) || !(stack->isTrackDaughterOf(trackn, mLastPrincipalTrackEntered))) {
      mLastPrincipalTrackEntered = trackn;
      resetHitIndices();

      // there is nothing more to do here as we are not
      // in the fiber volumes
      return false;
    }
  }

  // it could be that the entering track was not noticed
  // (tracking precision problems); warn about it for the moment until we have
  // a better solution (like checking the origin coordinates of the track)
  if (mLastPrincipalTrackEntered == -1) {
    LOG(WARN) << "Problem with principal track detection ; now in " << volname;
    // if we come here we are definitely in a sensitive volume !!
    mLastPrincipalTrackEntered = trackn;
    resetHitIndices();
  }

  Float_t p[3] = { 0., 0., 0. };
  Float_t trackenergy = 0.;
  fMC->TrackMomentum(p[0], p[1], p[2], trackenergy);
  Float_t eDep = fMC->Edep();

  int pdgCode = fMC->TrackPid();
  float lightoutput = 0.;
  auto currentMediumid = fMC->CurrentMedium();
  int nphe = 0;
  if (((currentMediumid == mMediumPMCid) || (currentMediumid == mMediumPMQid))) {
    int ibeta = 0, iangle = 0, iradius = 0;
    Bool_t isLightProduced = calculateTableIndexes(ibeta, iangle, iradius);
    if (isLightProduced) {
      int charge = 0;
      if (pdgCode < 10000) {
        charge = fMC->TrackCharge();
      } else {
        charge = TMath::Abs(pdgCode / 10000 - 100000);
      }

      //look into the light tables if the particle is charged
      if (TMath::Abs(charge) > 0) {
        if (detector == 1 || detector == 4) {
          iradius = std::min((int)Geometry::ZNFIBREDIAMETER, iradius);
          lightoutput = charge * charge * mLightTableZN[ibeta][iradius][iangle];
        } else {
          iradius = std::min((int)Geometry::ZPFIBREDIAMETER, iradius);
          lightoutput = charge * charge * mLightTableZP[ibeta][iradius][iangle];
        }
        //printf(".....beta %d  alpha %d radius %d  light output %f\n", ibeta, iangle, iradius, lightoutput);
        if (lightoutput > 0) {
          nphe = gRandom->Poisson(lightoutput);
          //printf(".. nphe %d  \n", nphe);
        }
      }
    }
  }

  // A new hit is created when there is nothing yet for this det + sector
  if (mCurrentHitsIndices[detector - 1][sector - 1] == -1) {

    auto tof = 1.e09 * fMC->TrackTime(); //TOF in ns
    bool issecondary = trackn != stack->getCurrentPrimaryIndex();
    //if(!issecondary) printf("!!! primary track (index %d)\n",stack->getCurrentPrimaryIndex());

    if (currentMediumid == mMediumPMCid) {
      mTotLightPMC = nphe;
    } else if (currentMediumid == mMediumPMQid) {
      mTotLightPMQ = nphe;
    }

    Vector3D<float> pos(x[0], x[1], x[2]);
    Vector3D<float> mom(p[0], p[1], p[2]);
    addHit(trackn, mLastPrincipalTrackEntered, issecondary, trackenergy, detector, sector,
           pos, mom, tof, xImp, eDep, mTotLightPMC, mTotLightPMQ);
    stack->addHit(GetDetId());
    mCurrentHitsIndices[detector - 1][sector - 1] = mHits->size() - 1;

    mXImpact = xImp;
    //printf("### NEW HITS CREATED in vol %d %d for track %d daughter of track %d\n", detector, sector, trackn, stack->GetCurrentTrack()->GetMother(0));
    return true;

  } else {
    auto& curHit = (*mHits)[mCurrentHitsIndices[detector - 1][sector - 1]];
    // summing variables that needs to be updated (Eloss and light yield)
    curHit.setNoNumContributingSteps(curHit.getNumContributingSteps() + 1);
    if (currentMediumid == mMediumPMCid) {
      mTotLightPMC += nphe;
    } else if (currentMediumid == mMediumPMQid) {
      mTotLightPMQ += nphe;
    }
    float incenloss = curHit.GetEnergyLoss() + eDep;
    if (incenloss > 0 || nphe > 0) {
      curHit.SetEnergyLoss(incenloss);
      curHit.setPMCLightYield(mTotLightPMC);
      curHit.setPMQLightYield(mTotLightPMQ);
      //printf("   >>> Hit updated in vol %d %d  for track %d (%d)    E %f  light %1.0f %1.0f \n",detector, sector, trackn, stack->GetCurrentTrack()->GetMother(0),curHit.GetEnergyLoss()+ eDep,mTotLightPMC,mTotLightPMQ);
    }
    return true;
  }
  return false;
}

//_____________________________________________________________________________
o2::zdc::Hit* Detector::addHit(Int_t trackID, Int_t parentID, Int_t sFlag, Float_t primaryEnergy, Int_t detID,
                               Int_t secID, Vector3D<float> pos, Vector3D<float> mom, Float_t tof, Vector3D<float> xImpact, Double_t energyloss, Int_t nphePMC, Int_t nphePMQ)
{
  LOG(DEBUG4) << "Adding hit for track " << trackID << " X (" << pos.X() << ", " << pos.Y() << ", "
              << pos.Z() << ") P (" << mom.X() << ", " << mom.Y() << ", " << mom.Z() << ")  Ekin "
              << primaryEnergy << " lightPMC  " << nphePMC << " lightPMQ  " << nphePMQ << std::endl;
  mHits->emplace_back(trackID, parentID, sFlag, primaryEnergy, detID, secID, pos, mom,
                      tof, xImpact, energyloss, nphePMC, nphePMQ);
  return &(mHits->back());
}

//_____________________________________________________________________________
void Detector::createMaterials()
{
  Int_t ifield = 2;
  Float_t fieldm = 10.0;
  o2::base::Detector::initFieldTrackingParams(ifield, fieldm);
  LOG(INFO) << "Detector::CreateMaterials >>>>> magnetic field: type " << ifield << " max " << fieldm << "\n";

  // ******** MATERIAL DEFINITION ********
  // --- W alloy -> ZN passive material
  Float_t aW[3] = { 183.85, 55.85, 58.71 };
  Float_t zW[3] = { 74., 26., 28. };
  Float_t wW[3] = { 0.93, 0.03, 0.04 };
  Float_t dW = 17.6;

  // --- Brass (CuZn)  -> ZP passive material
  Float_t aCuZn[2] = { 63.546, 65.39 };
  Float_t zCuZn[2] = { 29., 30. };
  Float_t wCuZn[2] = { 0.63, 0.37 };
  Float_t dCuZn = 8.48;

  // --- SiO2 -> fibres
  Float_t aq[2] = { 28.0855, 15.9994 };
  Float_t zq[2] = { 14., 8. };
  Float_t wq[2] = { 1., 2. };
  Float_t dq = 2.64;

  // --- Lead -> ZEM passive material
  Float_t aPb = 207.2;
  Float_t zPb = 82.;
  Float_t dPb = 11.35;
  Float_t radPb = 6.37 / dPb;
  Float_t absPb = 199.6 / dPb;

  // --- Copper -> beam pipe
  Float_t aCu = 63.546;
  Float_t zCu = 29.;
  Float_t dCu = 8.96;
  Float_t radCu = 12.86 / dCu;
  Float_t absCu = 137.3 / dCu;
  // Int_t nCu = 1.10;

  // --- Iron -> beam pipe
  Float_t aFe = 55.845;
  Float_t zFe = 26.;
  Float_t dFe = 7.874;
  Float_t radFe = 13.84 / dFe;
  Float_t absFe = 132.1 / dFe;

  // --- Aluminum -> beam pipe
  Float_t aAl = 26.98;
  Float_t zAl = 13.;
  Float_t dAl = 2.699;
  Float_t radAl = 24.01 / dAl;
  Float_t absAl = 107.2 / dAl;

  // --- Carbon -> beam pipe
  Float_t aCarb = 12.01;
  Float_t zCarb = 6.;
  Float_t dCarb = 2.265;
  Float_t radCarb = 18.8;
  Float_t absCarb = 49.9;

  // --- Residual gas -> inside beam pipe
  Float_t aResGas[3] = { 1.008, 12.0107, 15.9994 };
  Float_t zResGas[3] = { 1., 6., 8. };
  Float_t wResGas[3] = { 0.28, 0.28, 0.44 };
  Float_t dResGas = 3.2E-14;

  // --- Air
  Float_t aAir[4] = { 12.0107, 14.0067, 15.9994, 39.948 };
  Float_t zAir[4] = { 6., 7., 8., 18. };
  Float_t wAir[4] = { 0.000124, 0.755267, 0.231781, 0.012827 };
  Float_t dAir = 1.20479E-3;

  // ******** TRACKING MEDIA PARAMETERS ********
  Int_t notactiveMed = 0, sensMed = 1; // sensitive or not sensitive medium

  // field integration 0 no field -1 user in guswim 1 Runge Kutta 2 helix 3 const field along z
  Int_t inofld = 0; // Max. field value (no field)
  Int_t ifld = 2;   //TODO: ????CHECK!!!! secondo me va -1!!!!!
  Float_t nofieldm = 0.;

  Float_t maxnofld = 0.; // max field value (no field)
  Float_t maxfld = 45.;  // max field value (with field)
  Float_t tmaxnofd = 0.; // max deflection angle due to magnetic field in one step
  Float_t tmaxfd = 0.1;  // max deflection angle due to magnetic field in one step
  Float_t deemax = -1.;  // maximum fractional energy loss in one step 0<deemax<=1
  Float_t epsil = 0.001; // tracking precision [cm]
  Float_t stemax = 1.;   // max step allowed [cm] ????CHECK!!!!
  Float_t stmin = 0.01;  // minimum step due to continuous processes [cm] (negative value: choose it automatically) ????CHECK!!!! 0.01 in aliroot

  // ******** MATERIAL DEFINITION ********
  Mixture(0, "Walloy$", aW, zW, dW, 3, wW);
  Mixture(1, "CuZn$", aCuZn, zCuZn, dCuZn, 2, wCuZn);
  Mixture(2, "SiO2$", aq, zq, dq, -2, wq);
  Material(3, "Pb $", aPb, zPb, dPb, radPb, absPb);
  Material(4, "Cu $", aCu, zCu, dCu, radCu, absCu);
  Material(5, "Fe $", aFe, zFe, dFe, radFe, absFe);
  Material(6, "Al $", aAl, zAl, dAl, radAl, absAl);
  Material(7, "graphite$", aCarb, zCarb, dCarb, radCarb, absCarb);
  Mixture(8, "residualGas$", aResGas, zResGas, dResGas, 3, wResGas);
  Mixture(9, "Air$", aAir, zAir, dAir, 4, wAir);

  // ******** MEDIUM DEFINITION ********
  Medium(kWalloy, "Walloy$", 0, notactiveMed, inofld, nofieldm, tmaxnofd, stemax, deemax, epsil, stmin);
  Medium(kCuZn, "CuZn$", 1, notactiveMed, inofld, nofieldm, tmaxnofd, stemax, deemax, epsil, stmin);
  Medium(kSiO2pmc, "quartzPMC$", 2, sensMed, inofld, nofieldm, tmaxnofd, stemax, deemax, epsil, stmin);
  Medium(kSiO2pmq, "quartzPMQ$", 2, sensMed, inofld, nofieldm, tmaxnofd, stemax, deemax, epsil, stmin);
  Medium(kPb, "Lead$", 3, notactiveMed, inofld, nofieldm, tmaxnofd, stemax, deemax, epsil, stmin);
  Medium(kCu, "Copper$", 4, notactiveMed, inofld, nofieldm, tmaxnofd, stemax, deemax, epsil, stmin);
  Medium(kCuLumi, "CopperLowTh$", 4, notactiveMed, inofld, nofieldm, tmaxnofd, stemax, deemax, epsil, stmin);
  Medium(kFe, "Iron$", 5, notactiveMed, inofld, nofieldm, tmaxnofd, stemax, deemax, epsil, stmin);
  Medium(kFeLowTh, "IronLowTh$", 5, notactiveMed, inofld, nofieldm, tmaxnofd, stemax, deemax, epsil, stmin);
  Medium(kAl, "Aluminum$", 6, notactiveMed, inofld, nofieldm, tmaxnofd, stemax, deemax, epsil, stmin);
  Medium(kGraphite, "Graphite$", 7, notactiveMed, inofld, nofieldm, tmaxnofd, stemax, deemax, epsil, stmin);
  Medium(kVoidNoField, "VoidNoField$", 8, notactiveMed, inofld, nofieldm, tmaxnofd, stemax, deemax, epsil, stmin);
  Medium(kVoidwField, "VoidwField$", 8, notactiveMed, ifld, maxfld, tmaxfd, stemax, deemax, epsil, stmin);
  Medium(kAir, "Air$", 9, notactiveMed, inofld, nofieldm, tmaxnofd, stemax, deemax, epsil, stmin);
}

//_____________________________________________________________________________
void Detector::createAsideBeamLine()
{

  Double_t tubpar[3] = { 0., 0., 0 };
  Float_t boxpar[3] = { 0., 0., 0 };
  Double_t tubspar[5] = { 0., 0., 0., 0., 0. };
  Double_t conpar[15] = { 0. }; // all elements will be 0

  Float_t zA = 1910.2;

  conpar[0] = 0.;
  conpar[1] = 360.;
  conpar[2] = 2.;
  conpar[3] = zA;
  conpar[4] = 0.;
  conpar[5] = 55.;
  conpar[6] = 13500.;
  conpar[7] = 0.;
  conpar[8] = 55.;
  TVirtualMC::GetMC()->Gsvolu("ZDCA", "PCON", getMediumID(kVoidNoField), conpar, 9);
  TVirtualMC::GetMC()->Gspos("ZDCA", 1, "cave", 0., 0., 0., 0, "ONLY");

  // BEAM PIPE from 19.10 m to inner triplet beginning (22.965 m)
  tubpar[0] = 6.0 / 2.;
  tubpar[1] = 6.4 / 2.;
  tubpar[2] = 386.28 / 2.;
  TVirtualMC::GetMC()->Gsvolu("QA01", "TUBE", getMediumID(kFe), tubpar, 3);
  TVirtualMC::GetMC()->Gspos("QA01", 1, "ZDCA", 0., 0., tubpar[2] + zA, 0, "ONLY");

  zA += 2. * tubpar[2];

  // -- FIRST SECTION OF THE BEAM PIPE (from beginning of inner triplet to beginning of D1)
  tubpar[0] = 6.3 / 2.;
  tubpar[1] = 6.7 / 2.;
  tubpar[2] = 3541.8 / 2.;
  TVirtualMC::GetMC()->Gsvolu("QA02", "TUBE", getMediumID(kFe), tubpar, 3);
  TVirtualMC::GetMC()->Gspos("QA02", 1, "ZDCA", 0., 0., tubpar[2] + zA, 0, "ONLY");

  zA += 2. * tubpar[2];

  // -- SECOND SECTION OF THE BEAM PIPE (from the beginning of D1 to the beginning of D2)
  //  FROM (MAGNETIC) BEGINNING OF D1 TO THE (MAGNETIC) END OF D1 + 126.5 cm
  //  CYLINDRICAL PIPE of diameter increasing from 6.75 cm up to 8.0 cm
  //  from magnetic end :
  //  1) 80.1 cm still with ID = 6.75 radial beam screen
  //  2) 2.5 cm conical section from ID = 6.75 to ID = 8.0 cm
  //  3) 43.9 cm straight section (tube) with ID = 8.0 cm

  tubpar[0] = 6.75 / 2.;
  tubpar[1] = 7.15 / 2.;
  tubpar[2] = (945.0 + 80.1) / 2.;
  TVirtualMC::GetMC()->Gsvolu("QA03", "TUBE", getMediumID(kFe), tubpar, 3);
  TVirtualMC::GetMC()->Gspos("QA03", 1, "ZDCA", 0., 0., tubpar[2] + zA, 0, "ONLY");

  zA += 2. * tubpar[2];

  // Transition Cone from ID=67.5 mm  to ID=80 mm
  conpar[0] = 2.5 / 2.;
  conpar[1] = 6.75 / 2.;
  conpar[2] = 7.15 / 2.;
  conpar[3] = 8.0 / 2.;
  conpar[4] = 8.4 / 2.;
  TVirtualMC::GetMC()->Gsvolu("QA04", "CONE", getMediumID(kFe), conpar, 5);
  TVirtualMC::GetMC()->Gspos("QA04", 1, "ZDCA", 0., 0., conpar[0] + zA, 0, "ONLY");

  zA += 2. * conpar[0];

  tubpar[0] = 8.0 / 2.;
  tubpar[1] = 8.4 / 2.;
  tubpar[2] = (43.9 + 20. + 28.5 + 28.5) / 2.;
  TVirtualMC::GetMC()->Gsvolu("QA05", "TUBE", getMediumID(kFe), tubpar, 3);
  TVirtualMC::GetMC()->Gspos("QA05", 1, "ZDCA", 0., 0., tubpar[2] + zA, 0, "ONLY");

  zA += 2. * tubpar[2];

  // Second section of VAEHI (transition cone from ID=80mm to ID=98mm)
  conpar[0] = 4.0 / 2.;
  conpar[1] = 8.0 / 2.;
  conpar[2] = 8.4 / 2.;
  conpar[3] = 9.8 / 2.;
  conpar[4] = 10.2 / 2.;
  TVirtualMC::GetMC()->Gsvolu("QAV1", "CONE", getMediumID(kFe), conpar, 5);
  TVirtualMC::GetMC()->Gspos("QAV1", 1, "ZDCA", 0., 0., conpar[0] + zA, 0, "ONLY");

  zA += 2. * conpar[0];

  //Third section of VAEHI (transition cone from ID=98mm to ID=90mm)
  conpar[0] = 1.0 / 2.;
  conpar[1] = 9.8 / 2.;
  conpar[2] = 10.2 / 2.;
  conpar[3] = 9.0 / 2.;
  conpar[4] = 9.4 / 2.;
  TVirtualMC::GetMC()->Gsvolu("QAV2", "CONE", getMediumID(kFe), conpar, 5);
  TVirtualMC::GetMC()->Gspos("QAV2", 1, "ZDCA", 0., 0., conpar[0] + zA, 0, "ONLY");

  zA += 2. * conpar[0];

  // Fourth section of VAEHI (tube ID=90mm)
  tubpar[0] = 9.0 / 2.;
  tubpar[1] = 9.4 / 2.;
  tubpar[2] = 31.0 / 2.;
  TVirtualMC::GetMC()->Gsvolu("QAV3", "TUBE", getMediumID(kFe), tubpar, 3);
  TVirtualMC::GetMC()->Gspos("QAV3", 1, "ZDCA", 0., 0., tubpar[2] + zA, 0, "ONLY");

  zA += 2. * tubpar[2];

  //---------------------------- TCDD beginning ----------------------------------
  // space for the insertion of the collimator TCDD (2 m)
  // TCDD ZONE - 1st volume
  conpar[0] = 1.3 / 2.;
  conpar[1] = 9.0 / 2.;
  conpar[2] = 13.0 / 2.;
  conpar[3] = 9.6 / 2.;
  conpar[4] = 13.0 / 2.;
  TVirtualMC::GetMC()->Gsvolu("Q01T", "CONE", getMediumID(kFe), conpar, 5);
  TVirtualMC::GetMC()->Gspos("Q01T", 1, "ZDCA", 0., 0., conpar[0] + zA, 0, "ONLY");

  zA += 2. * conpar[0];

  // TCDD ZONE - 2nd volume
  tubpar[0] = 9.6 / 2.;
  tubpar[1] = 10.0 / 2.;
  tubpar[2] = 1.0 / 2.;
  TVirtualMC::GetMC()->Gsvolu("Q02T", "TUBE", getMediumID(kFe), tubpar, 3);
  TVirtualMC::GetMC()->Gspos("Q02T", 1, "ZDCA", 0., 0., tubpar[2] + zA, 0, "ONLY");

  zA += 2. * tubpar[2];

  // TCDD ZONE - third volume
  conpar[0] = 9.04 / 2.;
  conpar[1] = 9.6 / 2.;
  conpar[2] = 10.0 / 2.;
  conpar[3] = 13.8 / 2.;
  conpar[4] = 14.2 / 2.;
  TVirtualMC::GetMC()->Gsvolu("Q03T", "CONE", getMediumID(kFe), conpar, 5);
  TVirtualMC::GetMC()->Gspos("Q03T", 1, "ZDCA", 0., 0., conpar[0] + zA, 0, "ONLY");

  zA += 2. * conpar[0];

  // TCDD ZONE - 4th volume
  tubpar[0] = 13.8 / 2.;
  tubpar[1] = 14.2 / 2.;
  tubpar[2] = 38.6 / 2.;
  TVirtualMC::GetMC()->Gsvolu("Q04T", "TUBE", getMediumID(kFe), tubpar, 3);
  TVirtualMC::GetMC()->Gspos("Q04T", 1, "ZDCA", 0., 0., tubpar[2] + zA, 0, "ONLY");

  zA += 2. * tubpar[2];

  // TCDD ZONE - 5th volume
  tubpar[0] = 21.0 / 2.;
  tubpar[1] = 21.4 / 2.;
  tubpar[2] = 100.12 / 2.;
  TVirtualMC::GetMC()->Gsvolu("Q05T", "TUBE", getMediumID(kFe), tubpar, 3);
  TVirtualMC::GetMC()->Gspos("Q05T", 1, "ZDCA", 0., 0., tubpar[2] + zA, 0, "ONLY");

  zA += 2. * tubpar[2];

  // TCDD ZONE - 6th volume
  tubpar[0] = 13.8 / 2.;
  tubpar[1] = 14.2 / 2.;
  tubpar[2] = 38.6 / 2.;
  TVirtualMC::GetMC()->Gsvolu("Q06T", "TUBE", getMediumID(kFe), tubpar, 3);
  TVirtualMC::GetMC()->Gspos("Q06T", 1, "ZDCA", 0., 0., tubpar[2] + zA, 0, "ONLY");

  zA += 2. * tubpar[2];

  // TCDD ZONE - 7th volume
  conpar[0] = 11.34 / 2.;
  conpar[1] = 13.8 / 2.;
  conpar[2] = 14.2 / 2.;
  conpar[3] = 18.0 / 2.;
  conpar[4] = 18.4 / 2.;
  TVirtualMC::GetMC()->Gsvolu("Q07T", "CONE", getMediumID(kFe), conpar, 5);
  TVirtualMC::GetMC()->Gspos("Q07T", 1, "ZDCA", 0., 0., conpar[0] + zA, 0, "ONLY");

  zA += 2. * conpar[0];

  // Upper section : one single phi segment of a tube
  //  5 parameters for tubs: inner radius = 0.,
  //	outer radius = 7. cm, half length = 50 cm
  //	phi1 = 0., phi2 = 180.
  tubspar[0] = 0.0 / 2.;
  tubspar[1] = 14.0 / 2.;
  tubspar[2] = 100.0 / 2.;
  tubspar[3] = 0.;
  tubspar[4] = 180.;
  TVirtualMC::GetMC()->Gsvolu("Q08T", "TUBS", getMediumID(kFe), tubspar, 5);

  // rectangular beam pipe inside TCDD upper section (Vacuum)
  boxpar[0] = 7.0 / 2.;
  boxpar[1] = 2.2 / 2.;
  boxpar[2] = 100. / 2.;
  TVirtualMC::GetMC()->Gsvolu("Q09T", "BOX ", getMediumID(kVoidNoField), boxpar, 3);
  // positioning vacuum box in the upper section of TCDD
  TVirtualMC::GetMC()->Gspos("Q09T", 1, "Q08T", 0., 1.1, 0., 0, "ONLY");

  // lower section : one single phi segment of a tube
  tubspar[0] = 0.0 / 2.;
  tubspar[1] = 14.0 / 2.;
  tubspar[2] = 100.0 / 2.;
  tubspar[3] = 180.;
  tubspar[4] = 360.;
  TVirtualMC::GetMC()->Gsvolu("Q10T", "TUBS", getMediumID(kFe), tubspar, 5);
  // rectangular beam pipe inside TCDD lower section (Vacuum)
  boxpar[0] = 7.0 / 2.;
  boxpar[1] = 2.2 / 2.;
  boxpar[2] = 100. / 2.;
  TVirtualMC::GetMC()->Gsvolu("Q11T", "BOX ", getMediumID(kVoidNoField), boxpar, 3);
  // positioning vacuum box in the lower section of TCDD
  TVirtualMC::GetMC()->Gspos("Q11T", 1, "Q10T", 0., -1.1, 0., 0, "ONLY");

  // positioning  TCDD elements in ZDCA, (inside TCDD volume)
  // TODO: think about making those parameters tunable/settable from outside
  double TCDDAperturePos = 2.2;
  double TCDDApertureNeg = 2.4;
  TVirtualMC::GetMC()->Gspos("Q08T", 1, "ZDCA", 0., TCDDAperturePos, -100. + zA, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("Q10T", 1, "ZDCA", 0., -TCDDApertureNeg, -100. + zA, 0, "ONLY");

  // RF screen
  boxpar[0] = 0.2 / 2.;
  boxpar[1] = 4.0 / 2.;
  boxpar[2] = 100. / 2.;
  TVirtualMC::GetMC()->Gsvolu("Q12T", "BOX ", getMediumID(kFe), boxpar, 3);
  // positioning RF screen at both sides of TCDD
  TVirtualMC::GetMC()->Gspos("Q12T", 1, "ZDCA", tubspar[1] + boxpar[0], 0., -100. + zA, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("Q12T", 2, "ZDCA", -tubspar[1] - boxpar[0], 0., -100. + zA, 0, "ONLY");
  //---------------------------- TCDD end ---------------------------------------

  // The following elliptical tube 180 mm x 70 mm (obtained positioning the void QA06 in QA07)
  // represents VAMTF + first part of VCTCP (93 mm)

  tubpar[0] = 18.4 / 2.;
  tubpar[1] = 7.4 / 2.;
  tubpar[2] = (78 + 9.3) / 2.;
  TVirtualMC::GetMC()->Gsvolu("QA06", "ELTU", getMediumID(kFe), tubpar, 3);
  // AliRoot: temporary replace with a scaled tube (AG) ????????????
  /*TGeoTube *tubeQA06 = new TGeoTube(0.,tubpar[0],tubpar[2]);
  TGeoScale *scaleQA06 = new TGeoScale(1., tubpar[1]/tubpar[0], 1.);
  TGeoScaledShape *sshapeQA06 = new TGeoScaledShape(tubeQA06, scaleQA06);
  new TGeoVolume("QA06", sshapeQA06, gGeoManager->GetMedium(getMediumID(kVoidNoField)));*/

  tubpar[0] = 18.0 / 2.;
  tubpar[1] = 7.0 / 2.;
  tubpar[2] = (78 + 9.3) / 2.;
  TVirtualMC::GetMC()->Gsvolu("QA07", "ELTU", getMediumID(kVoidNoField), tubpar, 3);
  // temporary replace with a scaled tube (AG) ????????????
  /*TGeoTube *tubeQA07 = new TGeoTube(0.,tubpar[0],tubpar[2]);
  TGeoScale *scaleQA07 = new TGeoScale(1., tubpar[1]/tubpar[0], 1.);
  TGeoScaledShape *sshapeQA07 = new TGeoScaledShape(tubeQA07, scaleQA07);
  new TGeoVolume("QA07", sshapeQA07, gGeoManager->GetMedium(getMediumID(k10]));*/
  TVirtualMC::GetMC()->Gspos("QA06", 1, "ZDCA", 0., 0., tubpar[2] + zA, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("QA07", 1, "QA06", 0., 0., 0., 0, "ONLY");

  zA += 2. * tubpar[2];

  // VCTCP second part: transition cone from ID=180 to ID=212.7
  conpar[0] = 31.5 / 2.;
  conpar[1] = 18.0 / 2.;
  conpar[2] = 18.6 / 2.;
  conpar[3] = 21.27 / 2.;
  conpar[4] = 21.87 / 2.;
  TVirtualMC::GetMC()->Gsvolu("QA08", "CONE", getMediumID(kFe), conpar, 5);
  TVirtualMC::GetMC()->Gspos("QA08", 1, "ZDCA", 0., 0., conpar[0] + zA, 0, "ONLY");

  zA += 2. * conpar[0];

  //-- rotation matrices for the tilted cone after the TDI to recenter vacuum chamber
  Int_t irotpipe3, irotpipe4, irotpipe5;
  double rang3[6] = { 90. - 1.8934, 0., 90., 90., 1.8934, 180. };
  double rang4[6] = { 90. - 3.8, 0., 90., 90., 3.8, 180. };
  double rang5[6] = { 90. + 9.8, 0., 90., 90., 9.8, 0. };
  TVirtualMC::GetMC()->Matrix(irotpipe3, rang3[0], rang3[1], rang3[2], rang3[3], rang3[4], rang3[5]);
  TVirtualMC::GetMC()->Matrix(irotpipe4, rang4[0], rang4[1], rang4[2], rang4[3], rang4[4], rang4[5]);
  TVirtualMC::GetMC()->Matrix(irotpipe5, rang5[0], rang5[1], rang5[2], rang5[3], rang5[4], rang5[5]);

  // Tube ID 212.7 mm
  // Represents VCTCP third part (92 mm) + VCDWB (765 mm) + VMBGA (400 mm) +
  // VCDWE (300 mm) + VMBGA (400 mm) + TCTVB space + VAMTF space
  tubpar[0] = 21.27 / 2.;
  tubpar[1] = 21.87 / 2.;
  tubpar[2] = (195.7 + 148. + 78.) / 2.;
  TVirtualMC::GetMC()->Gsvolu("QA09", "TUBE", getMediumID(kFe), tubpar, 3);
  TVirtualMC::GetMC()->Gspos("QA09", 1, "ZDCA", 0., 0., tubpar[2] + zA, 0, "ONLY");

  zA += 2. * tubpar[2];

  // skewed transition piece (ID=212.7 mm to 332 mm) (before TDI)
  conpar[0] = (50.0 - 0.73 - 1.13) / 2.;
  conpar[1] = 21.27 / 2.;
  conpar[2] = 21.87 / 2.;
  conpar[3] = 33.2 / 2.;
  conpar[4] = 33.8 / 2.;
  TVirtualMC::GetMC()->Gsvolu("QA10", "CONE", getMediumID(kFe), conpar, 5);
  TVirtualMC::GetMC()->Gspos("QA10", 1, "ZDCA", -1.66, 0., conpar[0] + 0.73 + zA, irotpipe4, "ONLY");

  zA += 2. * conpar[0] + 0.73 + 1.13;

  // Vacuum chamber containing TDI
  tubpar[0] = 0.;
  tubpar[1] = 54.6 / 2.;
  tubpar[2] = 540.0 / 2.;
  TVirtualMC::GetMC()->Gsvolu("Q13TM", "TUBE", getMediumID(kVoidNoField), tubpar, 3);
  TVirtualMC::GetMC()->Gspos("Q13TM", 1, "ZDCA", 0., 0., tubpar[2] + zA, 0, "ONLY");
  tubpar[0] = 54.0 / 2.;
  tubpar[1] = 54.6 / 2.;
  tubpar[2] = 540.0 / 2.;
  TVirtualMC::GetMC()->Gsvolu("Q13T", "TUBE", getMediumID(kFe), tubpar, 3);
  TVirtualMC::GetMC()->Gspos("Q13T", 1, "Q13TM", 0., 0., 0., 0, "ONLY");

  zA += 2. * tubpar[2];

  //---------------- INSERT TDI INSIDE Q13T -----------------------------------
  boxpar[0] = 11.0 / 2.;
  boxpar[1] = 9.0 / 2.;
  boxpar[2] = 418.5 / 2.;
  double TDIAperturePos = 6.;
  TVirtualMC::GetMC()->Gsvolu("QTD1", "BOX ", getMediumID(kFe), boxpar, 3);
  TVirtualMC::GetMC()->Gspos("QTD1", 1, "Q13TM", -3.8, boxpar[1] + TDIAperturePos, 0., 0, "ONLY");
  boxpar[0] = 11.0 / 2.;
  boxpar[1] = 9.0 / 2.;
  boxpar[2] = 418.5 / 2.;
  TVirtualMC::GetMC()->Gsvolu("QTD2", "BOX ", getMediumID(kFe), boxpar, 3);
  double TDIApertureNeg = 6.;
  TVirtualMC::GetMC()->Gspos("QTD2", 1, "Q13TM", -3.8, -boxpar[1] - TDIApertureNeg, 0., 0, "ONLY");
  boxpar[0] = 5.1 / 2.;
  boxpar[1] = 0.2 / 2.;
  boxpar[2] = 418.5 / 2.;
  TVirtualMC::GetMC()->Gsvolu("QTD3", "BOX ", getMediumID(kFe), boxpar, 3);
  TVirtualMC::GetMC()->Gspos("QTD3", 1, "Q13TM", -3.8 + 5.5 + boxpar[0], TDIAperturePos, 0., 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("QTD3", 2, "Q13TM", -3.8 + 5.5 + boxpar[0], -TDIApertureNeg, 0., 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("QTD3", 3, "Q13TM", -3.8 - 5.5 - boxpar[0], TDIAperturePos, 0., 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("QTD3", 4, "Q13TM", -3.8 - 5.5 - boxpar[0], -TDIApertureNeg, 0., 0, "ONLY");
  //
  tubspar[0] = 12.0 / 2.;
  tubspar[1] = 12.4 / 2.;
  tubspar[2] = 418.5 / 2.;
  tubspar[3] = 90.;
  tubspar[4] = 270.;
  TVirtualMC::GetMC()->Gsvolu("QTD4", "TUBS", getMediumID(kCu), tubspar, 5);
  TVirtualMC::GetMC()->Gspos("QTD4", 1, "Q13TM", -3.8 - 10.6, 0., 0., 0, "ONLY");
  tubspar[0] = 12.0 / 2.;
  tubspar[1] = 12.4 / 2.;
  tubspar[2] = 418.5 / 2.;
  tubspar[3] = -90.;
  tubspar[4] = 90.;
  TVirtualMC::GetMC()->Gsvolu("QTD5", "TUBS", getMediumID(kCu), tubspar, 5);
  TVirtualMC::GetMC()->Gspos("QTD5", 1, "Q13TM", -3.8 + 10.6, 0., 0., 0, "ONLY");
  //---------------- END DEFINING TDI INSIDE Q13T -------------------------------

  // VCTCG skewed transition piece (ID=332 mm to 212.7 mm) (after TDI)
  conpar[0] = (50.0 - 2.92 - 1.89) / 2.;
  conpar[1] = 33.2 / 2.;
  conpar[2] = 33.8 / 2.;
  conpar[3] = 21.27 / 2.;
  conpar[4] = 21.87 / 2.;
  TVirtualMC::GetMC()->Gsvolu("QA11", "CONE", getMediumID(kFe), conpar, 5);
  TVirtualMC::GetMC()->Gspos("QA11", 1, "ZDCA", 4.32 - 3.8, 0., conpar[0] + 2.92 + zA, irotpipe5, "ONLY");

  zA += 2. * conpar[0] + 2.92 + 1.89;

  // The following tube ID 212.7 mm
  // represents VMBGA (400 mm) + VCDWE (300 mm) + VMBGA (400 mm) +
  //            BTVTS (600 mm) + VMLGB (400 mm)
  tubpar[0] = 21.27 / 2.;
  tubpar[1] = 21.87 / 2.;
  tubpar[2] = 210.0 / 2.;
  TVirtualMC::GetMC()->Gsvolu("QA12", "TUBE", getMediumID(kFe), tubpar, 3);
  TVirtualMC::GetMC()->Gspos("QA12", 1, "ZDCA", 4., 0., tubpar[2] + zA, 0, "ONLY");

  zA += 2. * tubpar[2];

  // First part of VCTCC
  // skewed transition cone from ID=212.7 mm to ID=797 mm
  conpar[0] = (121.0 - 0.37 - 1.35) / 2.;
  conpar[1] = 21.27 / 2.;
  conpar[2] = 21.87 / 2.;
  conpar[3] = 79.7 / 2.;
  conpar[4] = 81.3 / 2.;
  TVirtualMC::GetMC()->Gsvolu("QA13", "CONE", getMediumID(kFe), conpar, 5);
  TVirtualMC::GetMC()->Gspos("QA13", 1, "ZDCA", 4. - 2., 0., conpar[0] + 0.37 + zA, irotpipe3, "ONLY");

  zA += 2. * conpar[0] + 0.37 + 1.35;

  // The following tube ID 797 mm  represents the second part of VCTCC (4272 mm) +
  //            4 x VCDGA (4 x 4272 mm) + the first part of VCTCR (850 mm)
  tubpar[0] = 79.7 / 2.;
  tubpar[1] = 81.3 / 2.;
  tubpar[2] = (2221. - 136.) / 2.;
  TVirtualMC::GetMC()->Gsvolu("QA14", "TUBE", getMediumID(kFe), tubpar, 3);
  TVirtualMC::GetMC()->Gspos("QA14", 1, "ZDCA", 0., 0., tubpar[2] + zA, 0, "ONLY");

  zA += 2. * tubpar[2];

  // Second part of VCTCR
  // Transition from ID=797 mm to ID=196 mm. To simulate the thin window opened in the transition cone
  // we divide the transition cone in three cones:
  // (1) 8 mm thick (2) 3 mm thick (3) the third 8 mm thick

  // (1) 8 mm thick
  conpar[0] = 9.09 / 2.; // 15 degree
  conpar[1] = 79.7 / 2.;
  conpar[2] = 81.3 / 2.; // thickness 8 mm
  conpar[3] = 74.82868 / 2.;
  conpar[4] = 76.42868 / 2.; // thickness 8 mm
  TVirtualMC::GetMC()->Gsvolu("QA15", "CONE", getMediumID(kFe), conpar, 5);
  TVirtualMC::GetMC()->Gspos("QA15", 1, "ZDCA", 0., 0., conpar[0] + zA, 0, "ONLY");

  zA += 2. * conpar[0];

  // (2) 3 mm thick
  conpar[0] = 96.2 / 2.; // 15 degree
  conpar[1] = 74.82868 / 2.;
  conpar[2] = 75.42868 / 2.; // thickness 3 mm
  conpar[3] = 23.19588 / 2.;
  conpar[4] = 23.79588 / 2.; // thickness 3 mm
  TVirtualMC::GetMC()->Gsvolu("QA16", "CONE", getMediumID(kFe), conpar, 5);
  TVirtualMC::GetMC()->Gspos("QA16", 1, "ZDCA", 0., 0., conpar[0] + zA, 0, "ONLY");

  zA += 2. * conpar[0];

  // (3) 8 mm thick
  conpar[0] = 6.71 / 2.; // 15 degree
  conpar[1] = 23.19588 / 2.;
  conpar[2] = 24.79588 / 2.; // thickness 8 mm
  conpar[3] = 19.6 / 2.;
  conpar[4] = 21.2 / 2.; // thickness 8 mm
  TVirtualMC::GetMC()->Gsvolu("QA17", "CONE", getMediumID(kFe), conpar, 5);
  TVirtualMC::GetMC()->Gspos("QA17", 1, "ZDCA", 0., 0., conpar[0] + zA, 0, "ONLY");

  zA += 2. * conpar[0];

  // Third part of VCTCR: tube (ID=196 mm)
  tubpar[0] = 19.6 / 2.;
  tubpar[1] = 21.2 / 2.;
  tubpar[2] = 9.55 / 2.;
  TVirtualMC::GetMC()->Gsvolu("QA18", "TUBE", getMediumID(kFe), tubpar, 3);
  TVirtualMC::GetMC()->Gspos("QA18", 1, "ZDCA", 0., 0., tubpar[2] + zA, 0, "ONLY");

  zA += 2. * tubpar[2];

  // Flange (ID=196 mm) (last part of VCTCR and first part of VMZAR)
  tubpar[0] = 19.6 / 2.;
  tubpar[1] = 25.3 / 2.;
  tubpar[2] = 4.9 / 2.;
  TVirtualMC::GetMC()->Gsvolu("QF01", "TUBE", getMediumID(kFe), tubpar, 3);
  TVirtualMC::GetMC()->Gspos("QF01", 1, "ZDCA", 0., 0., tubpar[2] + zA, 0, "ONLY");

  zA += 2. * tubpar[2];

  // VMZAR (5 volumes)
  tubpar[0] = 20.2 / 2.;
  tubpar[1] = 20.6 / 2.;
  tubpar[2] = 2.15 / 2.;
  TVirtualMC::GetMC()->Gsvolu("QA19", "TUBE", getMediumID(kFe), tubpar, 3);
  TVirtualMC::GetMC()->Gspos("QA19", 1, "ZDCA", 0., 0., tubpar[2] + zA, 0, "ONLY");

  zA += 2. * tubpar[2];

  conpar[0] = 6.9 / 2.;
  conpar[1] = 20.2 / 2.;
  conpar[2] = 20.6 / 2.;
  conpar[3] = 23.9 / 2.;
  conpar[4] = 24.3 / 2.;
  TVirtualMC::GetMC()->Gsvolu("QA20", "CONE", getMediumID(kFe), conpar, 5);
  TVirtualMC::GetMC()->Gspos("QA20", 1, "ZDCA", 0., 0., conpar[0] + zA, 0, "ONLY");

  zA += 2. * conpar[0];

  tubpar[0] = 23.9 / 2.;
  tubpar[1] = 25.5 / 2.;
  tubpar[2] = 17.0 / 2.;
  TVirtualMC::GetMC()->Gsvolu("QA21", "TUBE", getMediumID(kFe), tubpar, 3);
  TVirtualMC::GetMC()->Gspos("QA21", 1, "ZDCA", 0., 0., tubpar[2] + zA, 0, "ONLY");

  zA += 2. * tubpar[2];

  conpar[0] = 6.9 / 2.;
  conpar[1] = 23.9 / 2.;
  conpar[2] = 24.3 / 2.;
  conpar[3] = 20.2 / 2.;
  conpar[4] = 20.6 / 2.;
  TVirtualMC::GetMC()->Gsvolu("QA22", "CONE", getMediumID(kFe), conpar, 5);
  TVirtualMC::GetMC()->Gspos("QA22", 1, "ZDCA", 0., 0., conpar[0] + zA, 0, "ONLY");

  zA += 2. * conpar[0];

  tubpar[0] = 20.2 / 2.;
  tubpar[1] = 20.6 / 2.;
  tubpar[2] = 2.15 / 2.;
  TVirtualMC::GetMC()->Gsvolu("QA23", "TUBE", getMediumID(kFe), tubpar, 3);
  TVirtualMC::GetMC()->Gspos("QA23", 1, "ZDCA", 0., 0., tubpar[2] + zA, 0, "ONLY");

  zA += 2. * tubpar[2];

  // Flange (ID=196 mm)(last part of VMZAR and first part of VCTYD)
  tubpar[0] = 19.6 / 2.;
  tubpar[1] = 25.3 / 2.;
  tubpar[2] = 4.9 / 2.;
  TVirtualMC::GetMC()->Gsvolu("QF02", "TUBE", getMediumID(kFe), tubpar, 3);
  TVirtualMC::GetMC()->Gspos("QF02", 1, "ZDCA", 0., 0., tubpar[2] + zA, 0, "ONLY");

  zA += 2. * tubpar[2];

  // simulation of the trousers (VCTYB)
  tubpar[0] = 19.6 / 2.;
  tubpar[1] = 20.0 / 2.;
  tubpar[2] = 3.9 / 2.;
  TVirtualMC::GetMC()->Gsvolu("QA24", "TUBE", getMediumID(kFe), tubpar, 3);
  TVirtualMC::GetMC()->Gspos("QA24", 1, "ZDCA", 0., 0., tubpar[2] + zA, 0, "ONLY");

  zA += 2. * tubpar[2];

  // transition cone from ID=196. to ID=216.6
  conpar[0] = 32.55 / 2.;
  conpar[1] = 19.6 / 2.;
  conpar[2] = 20.0 / 2.;
  conpar[3] = 21.66 / 2.;
  conpar[4] = 22.06 / 2.;
  TVirtualMC::GetMC()->Gsvolu("QA25", "CONE", getMediumID(kFe), conpar, 5);
  TVirtualMC::GetMC()->Gspos("QA25", 1, "ZDCA", 0., 0., conpar[0] + zA, 0, "ONLY");

  zA += 2. * conpar[0];

  // tube
  tubpar[0] = 21.66 / 2.;
  tubpar[1] = 22.06 / 2.;
  tubpar[2] = 28.6 / 2.;
  TVirtualMC::GetMC()->Gsvolu("QA26", "TUBE", getMediumID(kFe), tubpar, 3);
  TVirtualMC::GetMC()->Gspos("QA26", 1, "ZDCA", 0., 0., tubpar[2] + zA, 0, "ONLY");

  zA += 2. * tubpar[2];

  // --------------------------------------------------------
  // RECOMBINATION CHAMBER
  // TRANSFORMATION MATRICES
  double dx = -3.970000;
  double dy = 0.000000;
  double dz = 0.0;
  // Rotation:
  double thx = 84.989100;
  double phx = 0.000000;
  double thy = 90.000000;
  double phy = 90.000000;
  double thz = 5.010900;
  double phz = 180.000000;
  TGeoRotation* rotMatrix1 = new TGeoRotation("", thx, phx, thy, phy, thz, phz);
  // Combi transformation:
  TGeoCombiTrans* rotMatrix2 = new TGeoCombiTrans("ZDC_c1", dx, dy, dz, rotMatrix1);
  rotMatrix2->RegisterYourself();
  // Combi transformation:
  // Rotation:
  double thx3 = 95.010900;
  double phx3 = 0.000000;
  double thy3 = 90.000000;
  double phy3 = 90.000000;
  double thz3 = 5.010900;
  double phz3 = 0.000000;
  TGeoRotation* rotMatrix3 = new TGeoRotation("", thx3, phx3, thy3, phy3, thz3, phz3);
  TGeoCombiTrans* rotMatrix4 = new TGeoCombiTrans("ZDC_c2", -dx, dy, dz, rotMatrix3);
  rotMatrix4->RegisterYourself();

  //-- rotation matrices for the legs
  Int_t irotpipe1, irotpipe2;
  double rang1[6] = { 90. - 1.0027, 0., 90., 90., 1.0027, 180. };
  double rang2[6] = { 90. + 1.0027, 0., 90., 90., 1.0027, 0. };
  TVirtualMC::GetMC()->Matrix(irotpipe1, rang1[0], rang1[1], rang1[2], rang1[3], rang1[4], rang1[5]);
  TVirtualMC::GetMC()->Matrix(irotpipe2, rang2[0], rang2[1], rang2[2], rang2[3], rang2[4], rang2[5]);

  // VOLUMES DEFINITION
  // Volume: ZDCA
  TGeoVolume* pZDCA = gGeoManager->GetVolume("ZDCA");

  conpar[0] = (90.1 - 0.95 - 0.26) / 2.;
  conpar[1] = 0.0 / 2.;
  conpar[2] = 21.6 / 2.;
  conpar[3] = 0.0 / 2.;
  conpar[4] = 5.8 / 2.;
  new TGeoCone("QALext", conpar[0], conpar[1], conpar[2], conpar[3], conpar[4]);

  conpar[0] = (90.1 - 0.95 - 0.26) / 2.;
  conpar[1] = 0.0 / 2.;
  conpar[2] = 21.2 / 2.;
  conpar[3] = 0.0 / 2.;
  conpar[4] = 5.4 / 2.;
  new TGeoCone("QALint", conpar[0], conpar[1], conpar[2], conpar[3], conpar[4]);

  // Outer trousers
  TGeoCompositeShape* pOutTrousers = new TGeoCompositeShape("outTrousers", "QALext:ZDC_c1+QALext:ZDC_c2");

  auto& matmgr = o2::base::MaterialManager::Instance();

  // Volume: QALext
  TGeoVolume* pQALext = new TGeoVolume("QALext", pOutTrousers, matmgr.getTGeoMedium("ZDC", kFeLowTh));
  pQALext->SetLineColor(kBlue);
  pQALext->SetVisLeaves(kTRUE);
  //
  TGeoTranslation* tr1 = new TGeoTranslation(0., 0., (Double_t)conpar[0] + 0.95 + zA);
  pZDCA->AddNode(pQALext, 1, tr1);
  // Inner trousers
  TGeoCompositeShape* pIntTrousers = new TGeoCompositeShape("intTrousers", "QALint:ZDC_c1+QALint:ZDC_c2");
  // Volume: QALint
  TGeoVolume* pQALint = new TGeoVolume("QALint", pIntTrousers, matmgr.getTGeoMedium("ZDC", kVoidNoField));
  pQALint->SetLineColor(kAzure);
  pQALint->SetVisLeaves(kTRUE);
  pQALext->AddNode(pQALint, 1);

  zA += 90.1;

  //  second section : 2 tubes (ID = 54. OD = 58.)
  tubpar[0] = 5.4 / 2.;
  tubpar[1] = 5.8 / 2.;
  tubpar[2] = 40.0 / 2.;
  TVirtualMC::GetMC()->Gsvolu("QA27", "TUBE", getMediumID(kFe), tubpar, 3);
  TVirtualMC::GetMC()->Gspos("QA27", 1, "ZDCA", -15.8 / 2., 0., tubpar[2] + zA, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("QA27", 2, "ZDCA", 15.8 / 2., 0., tubpar[2] + zA, 0, "ONLY");

  zA += 2. * tubpar[2];

  // transition x2zdc to recombination chamber : skewed cone
  conpar[0] = (10. - 1.) / 2.;
  conpar[1] = 5.4 / 2.;
  conpar[2] = 5.8 / 2.;
  conpar[3] = 6.3 / 2.;
  conpar[4] = 7.0 / 2.;
  TVirtualMC::GetMC()->Gsvolu("QA28", "CONE", getMediumID(kFe), conpar, 5);
  TVirtualMC::GetMC()->Gspos("QA28", 1, "ZDCA", -7.9 - 0.175, 0., conpar[0] + 0.5 + zA, irotpipe1, "ONLY");
  TVirtualMC::GetMC()->Gspos("QA28", 2, "ZDCA", 7.9 + 0.175, 0., conpar[0] + 0.5 + zA, irotpipe2, "ONLY");

  zA += 2. * conpar[0] + 1.;

  // 2 tubes (ID = 63 mm OD=70 mm)
  tubpar[0] = 6.3 / 2.;
  tubpar[1] = 7.0 / 2.;
  tubpar[2] = (342.5 + 498.3) / 2.;
  TVirtualMC::GetMC()->Gsvolu("QA29", "TUBE", getMediumID(kFe), tubpar, 3);
  TVirtualMC::GetMC()->Gspos("QA29", 1, "ZDCA", -16.5 / 2., 0., tubpar[2] + zA, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("QA29", 2, "ZDCA", 16.5 / 2., 0., tubpar[2] + zA, 0, "ONLY");
  //printf("	QA29 TUBE from z = %1.2f to z= %1.2f (separate pipes)\n",zA,2*tubpar[2]+zA);

  zA += 2. * tubpar[2];

  // -- Luminometer (Cu box) in front of ZN - side A
  if (mLumiLength > 0.) { // FIX IT!!!!!!!!!!!!!!!
    boxpar[0] = 8.0 / 2.;
    boxpar[1] = 8.0 / 2.;
    boxpar[2] = mLumiLength / 2.;
    TVirtualMC::GetMC()->Gsvolu("QLUA", "BOX ", getMediumID(kCuLumi), boxpar, 3);
    TVirtualMC::GetMC()->Gspos("QLUA", 1, "ZDCA", 0., 0., Geometry::ZNAPOSITION[1] /*fPosZNA[2]*/ - 66. - boxpar[2], 0, "ONLY");
    LOG(DEBUG) << "	A-side luminometer positioned in front of ZNA\n";
  }
}

//_____________________________________________________________________________
void Detector::createCsideBeamLine()
{
  Double_t tubpar[3] = { 0., 0., 0 };
  Float_t boxpar[3] = { 0., 0., 0 };
  Double_t tubspar[5] = { 0., 0., 0., 0., 0. };
  Double_t conpar[15] = {
    0.,
  };

  Float_t zC = 1947.2;
  Float_t zCompensator = 1974.;

  conpar[0] = 0.;
  conpar[1] = 360.;
  conpar[2] = 4.; // Num radius specifications: 4
  conpar[3] = -13500.;
  conpar[4] = 0.;
  conpar[5] = 55.;
  conpar[6] = -zCompensator;
  conpar[7] = 0.;
  conpar[8] = 55.;
  conpar[9] = -zCompensator;
  conpar[10] = 0.;
  conpar[11] = 6.7 / 2.;
  conpar[12] = -zC; // (4) Beginning of ZDCC mother volume
  conpar[13] = 0.;
  conpar[14] = 6.7 / 2.;
  TVirtualMC::GetMC()->Gsvolu("ZDCC", "PCON", getMediumID(kVoidNoField), conpar, 15);
  TVirtualMC::GetMC()->Gspos("ZDCC", 1, "cave", 0., 0., 0., 0, "ONLY");

  // -- BEAM PIPE from compensator dipole to the beginning of D1
  tubpar[0] = 6.3 / 2.;
  tubpar[1] = 6.7 / 2.;
  // From beginning of ZDC volumes to beginning of D1
  tubpar[2] = (5838.3 - zC) / 2.;
  TVirtualMC::GetMC()->Gsvolu("QT01", "TUBE", getMediumID(kFe), tubpar, 3);
  TVirtualMC::GetMC()->Gspos("QT01", 1, "ZDCC", 0., 0., -tubpar[2] - zC, 0, "ONLY");

  zC += 2. * tubpar[2];

  //-- BEAM PIPE from the end of D1 to the beginning of D2
  //-- FROM MAGNETIC BEGINNING OF D1 TO MAGNETIC END OF D1
  //-- 	Cylindrical pipe (r = 3.47) + conical flare
  tubpar[0] = 6.94 / 2.;
  tubpar[1] = 7.34 / 2.;
  tubpar[2] = (6909.8 - zC) / 2.;
  TVirtualMC::GetMC()->Gsvolu("QT02", "TUBE", getMediumID(kFe), tubpar, 3);
  TVirtualMC::GetMC()->Gspos("QT02", 1, "ZDCC", 0., 0., -tubpar[2] - zC, 0, "ONLY");

  zC += 2. * tubpar[2];

  tubpar[0] = 8. / 2.;
  tubpar[1] = 8.6 / 2.;
  tubpar[2] = (6958.3 - zC) / 2.;
  TVirtualMC::GetMC()->Gsvolu("QT0B", "TUBE", getMediumID(kFe), tubpar, 3);
  TVirtualMC::GetMC()->Gspos("QT0B", 1, "ZDCC", 0., 0., -tubpar[2] - zC, 0, "ONLY");

  zC += 2. * tubpar[2];

  tubpar[0] = 9. / 2.;
  tubpar[1] = 9.6 / 2.;
  tubpar[2] = (7022.8 - zC) / 2.;
  TVirtualMC::GetMC()->Gsvolu("QT03", "TUBE", getMediumID(kFe), tubpar, 3);
  TVirtualMC::GetMC()->Gspos("QT03", 1, "ZDCC", 0., 0., -tubpar[2] - zC, 0, "ONLY");

  zC += 2. * tubpar[2];

  conpar[0] = 39.2 / 2.;
  conpar[1] = 18. / 2.;
  conpar[2] = 18.6 / 2.;
  conpar[3] = 9. / 2.;
  conpar[4] = 9.6 / 2.;
  TVirtualMC::GetMC()->Gsvolu("QC01", "CONE", getMediumID(kFe), conpar, 5);
  TVirtualMC::GetMC()->Gspos("QC01", 1, "ZDCC", 0., 0., -conpar[0] - zC, 0, "ONLY");

  zC += conpar[0] * 2.;

  // 2nd section of    VCTCQ+VAMTF+TCLIA+VAMTF+1st part of VCTCP
  Float_t totLength1 = 160.8 + 78. + 148. + 78. + 9.3;
  //
  tubpar[0] = 18.6 / 2.;
  tubpar[1] = 7.6 / 2.;
  tubpar[2] = totLength1 / 2.;
  TVirtualMC::GetMC()->Gsvolu("QE01", "ELTU", getMediumID(kFe), tubpar, 3);
  // in AliRoot: temporary replace with a scaled tube (AG) ??????????
  /*TGeoTube *tubeQE01 = new TGeoTube(0.,tubpar[0],tubpar[2]);
  TGeoScale *scaleQE01 = new TGeoScale(1., tubpar[1]/tubpar[0], 1.);
  TGeoScaledShape *sshapeQE01 = new TGeoScaledShape(tubeQE01, scaleQE01);
  new TGeoVolume("QE01", sshapeQE01, gGeoManager->GetMedium(getMediumID(kVoidNoField)));*/

  tubpar[0] = 18.0 / 2.;
  tubpar[1] = 7.0 / 2.;
  tubpar[2] = totLength1 / 2.;
  TVirtualMC::GetMC()->Gsvolu("QE02", "ELTU", getMediumID(kVoidNoField), tubpar, 3);
  // in AliRoot: temporary replace with a scaled tube (AG) ??????????
  /*TGeoTube *tubeQE02 = new TGeoTube(0.,tubpar[0],tubpar[2]);
  TGeoScale *scaleQE02 = new TGeoScale(1., tubpar[1]/tubpar[0], 1.);
  TGeoScaledShape *sshapeQE02 = new TGeoScaledShape(tubeQE02, scaleQE02);
  new TGeoVolume("QE02", sshapeQE02, gGeoManager->GetMedium(getMediumID(k10]));*/

  TVirtualMC::GetMC()->Gspos("QE01", 1, "ZDCC", 0., 0., -tubpar[2] - zC, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("QE02", 1, "QE01", 0., 0., 0., 0, "ONLY");

  // TCLIA collimator jaws (defined ONLY if aperture<3.5!)
  if (mTCLIAAPERTURE < 3.5) {
    boxpar[0] = 5.4 / 2.;
    boxpar[1] = (3.5 - mTCLIAAPERTURE - mVCollSideCCentreY - 0.7) / 2.; // FIX IT!!!!!!!!
    if (boxpar[1] < 0.)
      boxpar[1] = 0.;
    boxpar[2] = 124.4 / 2.;
    TVirtualMC::GetMC()->Gsvolu("QCVC", "BOX ", getMediumID(kGraphite), boxpar, 3);
    TVirtualMC::GetMC()->Gspos("QCVC", 1, "QE02", -boxpar[0], mTCLIAAPERTURE + mVCollSideCCentreY + boxpar[1], -totLength1 / 2. + 160.8 + 78. + 148. / 2., 0, "ONLY");     // FIX IT!!!!!!!!
    TVirtualMC::GetMC()->Gspos("QCVC", 2, "QE02", -boxpar[0], -mTCLIAAPERTURENEG + mVCollSideCCentreY - boxpar[1], -totLength1 / 2. + 160.8 + 78. + 148. / 2., 0, "ONLY"); // FIX IT!!!!!!!!
  }

  zC += tubpar[2] * 2.;

  // 2nd part of VCTCP
  conpar[0] = 31.5 / 2.;
  conpar[1] = 21.27 / 2.;
  conpar[2] = 21.87 / 2.;
  conpar[3] = 18.0 / 2.;
  conpar[4] = 18.6 / 2.;
  TVirtualMC::GetMC()->Gsvolu("QC02", "CONE", getMediumID(kFe), conpar, 5);
  TVirtualMC::GetMC()->Gspos("QC02", 1, "ZDCC", 0., 0., -conpar[0] - zC, 0, "ONLY");

  zC += conpar[0] * 2.;

  // 3rd section of VCTCP+VCDWC+VMLGB
  Float_t totLenght2 = (8373.3 - zC);
  tubpar[0] = 21.2 / 2.;
  tubpar[1] = 21.9 / 2.;
  tubpar[2] = totLenght2 / 2.;
  TVirtualMC::GetMC()->Gsvolu("QT04", "TUBE", getMediumID(kVoidNoField), tubpar, 3);
  TVirtualMC::GetMC()->Gspos("QT04", 1, "ZDCC", 0., 0., -tubpar[2] - zC, 0, "ONLY");

  zC += tubpar[2] * 2.;

  // First part of VCTCD
  // skewed transition cone from ID=212.7 mm to ID=797 mm
  conpar[0] = 121. / 2.;
  conpar[1] = 79.7 / 2.;
  conpar[2] = 81.3 / 2.;
  conpar[3] = 21.27 / 2.;
  conpar[4] = 21.87 / 2.;
  TVirtualMC::GetMC()->Gsvolu("QC03", "CONE", getMediumID(kVoidNoField), conpar, 5);
  TVirtualMC::GetMC()->Gspos("QC03", 1, "ZDCC", 0., 0., -conpar[0] - zC, 0, "ONLY");

  zC += 2. * conpar[0];

  // VCDGB + 1st part of VCTCH
  tubpar[0] = 79.7 / 2.;
  tubpar[1] = 81.3 / 2.;
  tubpar[2] = (5 * 475.2 + 97. - 136) / 2.;
  TVirtualMC::GetMC()->Gsvolu("QT05", "TUBE", getMediumID(kVoidNoField), tubpar, 3);
  TVirtualMC::GetMC()->Gspos("QT05", 1, "ZDCC", 0., 0., -tubpar[2] - zC, 0, "ONLY");

  zC += 2. * tubpar[2];

  // 2nd part of VCTCH
  // Transition from ID=797 mm to ID=196 mm:
  // in order to simulate the thin window opened in the transition cone
  // we divide the transition cone in three cones:
  // (1) 8 mm thick (2) 3 mm thick (3) the third 8 mm thick

  // (1) 8 mm thick
  conpar[0] = 9.09 / 2.; // 15 degree
  conpar[1] = 74.82868 / 2.;
  conpar[2] = 76.42868 / 2.; // thickness 8 mm
  conpar[3] = 79.7 / 2.;
  conpar[4] = 81.3 / 2.; // thickness 8 mm
  TVirtualMC::GetMC()->Gsvolu("QC04", "CONE", getMediumID(kVoidNoField), conpar, 5);
  TVirtualMC::GetMC()->Gspos("QC04", 1, "ZDCC", 0., 0., -conpar[0] - zC, 0, "ONLY");

  zC += 2. * conpar[0];

  // (2) 3 mm thick
  conpar[0] = 96.2 / 2.; // 15 degree
  conpar[1] = 23.19588 / 2.;
  conpar[2] = 23.79588 / 2.; // thickness 3 mm
  conpar[3] = 74.82868 / 2.;
  conpar[4] = 75.42868 / 2.; // thickness 3 mm
  TVirtualMC::GetMC()->Gsvolu("QC05", "CONE", getMediumID(kVoidNoField), conpar, 5);
  TVirtualMC::GetMC()->Gspos("QC05", 1, "ZDCC", 0., 0., -conpar[0] - zC, 0, "ONLY");

  zC += 2. * conpar[0];

  // (3) 8 mm thick
  conpar[0] = 6.71 / 2.; // 15 degree
  conpar[1] = 19.6 / 2.;
  conpar[2] = 21.2 / 2.; // thickness 8 mm
  conpar[3] = 23.19588 / 2.;
  conpar[4] = 24.79588 / 2.; // thickness 8 mm
  TVirtualMC::GetMC()->Gsvolu("QC06", "CONE", getMediumID(kVoidNoField), conpar, 5);
  TVirtualMC::GetMC()->Gspos("QC06", 1, "ZDCC", 0., 0., -conpar[0] - zC, 0, "ONLY");

  zC += 2. * conpar[0];

  // VMZAR (5 volumes)
  tubpar[0] = 20.2 / 2.;
  tubpar[1] = 20.6 / 2.;
  tubpar[2] = 2.15 / 2.;
  TVirtualMC::GetMC()->Gsvolu("QT06", "TUBE", getMediumID(kVoidNoField), tubpar, 3);
  TVirtualMC::GetMC()->Gspos("QT06", 1, "ZDCC", 0., 0., -tubpar[2] - zC, 0, "ONLY");

  zC += 2. * tubpar[2];

  conpar[0] = 6.9 / 2.;
  conpar[1] = 23.9 / 2.;
  conpar[2] = 24.3 / 2.;
  conpar[3] = 20.2 / 2.;
  conpar[4] = 20.6 / 2.;
  TVirtualMC::GetMC()->Gsvolu("QC07", "CONE", getMediumID(kVoidNoField), conpar, 5);
  TVirtualMC::GetMC()->Gspos("QC07", 1, "ZDCC", 0., 0., -conpar[0] - zC, 0, "ONLY");

  zC += 2. * conpar[0];

  tubpar[0] = 23.9 / 2.;
  tubpar[1] = 25.5 / 2.;
  tubpar[2] = 17.0 / 2.;
  TVirtualMC::GetMC()->Gsvolu("QT07", "TUBE", getMediumID(kVoidNoField), tubpar, 3);
  TVirtualMC::GetMC()->Gspos("QT07", 1, "ZDCC", 0., 0., -tubpar[2] - zC, 0, "ONLY");

  zC += 2. * tubpar[2];

  conpar[0] = 6.9 / 2.;
  conpar[1] = 20.2 / 2.;
  conpar[2] = 20.6 / 2.;
  conpar[3] = 23.9 / 2.;
  conpar[4] = 24.3 / 2.;
  TVirtualMC::GetMC()->Gsvolu("QC08", "CONE", getMediumID(kVoidNoField), conpar, 5);
  TVirtualMC::GetMC()->Gspos("QC08", 1, "ZDCC", 0., 0., -conpar[0] - zC, 0, "ONLY");

  zC += 2. * conpar[0];

  tubpar[0] = 20.2 / 2.;
  tubpar[1] = 20.6 / 2.;
  tubpar[2] = 2.15 / 2.;
  TVirtualMC::GetMC()->Gsvolu("QT08", "TUBE", getMediumID(kVoidNoField), tubpar, 3);
  TVirtualMC::GetMC()->Gspos("QT08", 1, "ZDCC", 0., 0., -tubpar[2] - zC, 0, "ONLY");

  zC += 2. * tubpar[2];

  // Flange (ID=196 mm)(last part of VMZAR and first part of VCTYB)
  tubpar[0] = 19.6 / 2.;
  tubpar[1] = 25.3 / 2.;
  tubpar[2] = 4.9 / 2.;
  TVirtualMC::GetMC()->Gsvolu("QT09", "TUBE", getMediumID(kVoidNoField), tubpar, 3);
  TVirtualMC::GetMC()->Gspos("QT09", 1, "ZDCC", 0., 0., -tubpar[2] - zC, 0, "ONLY");

  zC += 2. * tubpar[2];

  // simulation of the trousers (VCTYB)
  tubpar[0] = 19.6 / 2.;
  tubpar[1] = 20.0 / 2.;
  tubpar[2] = 3.9 / 2.;
  TVirtualMC::GetMC()->Gsvolu("QT10", "TUBE", getMediumID(kVoidNoField), tubpar, 3);
  TVirtualMC::GetMC()->Gspos("QT10", 1, "ZDCC", 0., 0., -tubpar[2] - zC, 0, "ONLY");

  zC += 2. * tubpar[2];

  // transition cone from ID=196. to ID=216.6
  conpar[0] = 32.55 / 2.;
  conpar[1] = 21.66 / 2.;
  conpar[2] = 22.06 / 2.;
  conpar[3] = 19.6 / 2.;
  conpar[4] = 20.0 / 2.;
  TVirtualMC::GetMC()->Gsvolu("QC09", "CONE", getMediumID(kVoidNoField), conpar, 5);
  TVirtualMC::GetMC()->Gspos("QC09", 1, "ZDCC", 0., 0., -conpar[0] - zC, 0, "ONLY");

  zC += 2. * conpar[0];

  // tube
  tubpar[0] = 21.66 / 2.;
  tubpar[1] = 22.06 / 2.;
  tubpar[2] = 28.6 / 2.;
  TVirtualMC::GetMC()->Gsvolu("QT11", "TUBE", getMediumID(kVoidNoField), tubpar, 3);
  TVirtualMC::GetMC()->Gspos("QT11", 1, "ZDCC", 0., 0., -tubpar[2] - zC, 0, "ONLY");

  zC += 2. * tubpar[2];

  // --------------------------------------------------------
  // RECOMBINATION CHAMBER
  // TRANSFORMATION MATRICES
  Double_t dx = -3.970000;
  Double_t dy = 0.000000;
  Double_t dz = 0.0;
  // Rotation:
  Double_t thx = 84.989100;
  Double_t phx = 180.000000;
  Double_t thy = 90.000000;
  Double_t phy = 90.000000;
  Double_t thz = 185.010900;
  Double_t phz = 0.000000;
  TGeoRotation* rotMatrix1c = new TGeoRotation("c", thx, phx, thy, phy, thz, phz);
  // Combi transformation:
  dx = -3.970000;
  dy = 0.000000;
  dz = 0.0;
  TGeoCombiTrans* rotMatrix2c = new TGeoCombiTrans("ZDCC_c1", dx, dy, dz, rotMatrix1c);
  rotMatrix2c->RegisterYourself();
  // Combi transformation:
  dx = 3.970000;
  dy = 0.000000;
  dz = 0.0;
  // Rotation:
  thx = 95.010900;
  phx = 180.000000;
  thy = 90.000000;
  phy = 90.000000;
  thz = 180. - 5.010900;
  phz = 0.000000;
  TGeoRotation* rotMatrix3c = new TGeoRotation("", thx, phx, thy, phy, thz, phz);
  TGeoCombiTrans* rotMatrix4c = new TGeoCombiTrans("ZDCC_c2", dx, dy, dz, rotMatrix3c);
  rotMatrix4c->RegisterYourself();

  // VOLUMES DEFINITION
  // Volume: ZDCC
  TGeoVolume* pZDCC = gGeoManager->GetVolume("ZDCC");

  conpar[0] = (90.1 - 0.95 - 0.26 - 0.0085) / 2.;
  conpar[1] = 0.0 / 2.;
  conpar[2] = 21.6 / 2.;
  conpar[3] = 0.0 / 2.;
  conpar[4] = 5.8 / 2.;
  new TGeoCone("QCLext", conpar[0], conpar[1], conpar[2], conpar[3], conpar[4]);

  conpar[0] = (90.1 - 0.95 - 0.26 - 0.0085) / 2.;
  conpar[1] = 0.0 / 2.;
  conpar[2] = 21.2 / 2.;
  conpar[3] = 0.0 / 2.;
  conpar[4] = 5.4 / 2.;
  new TGeoCone("QCLint", conpar[0], conpar[1], conpar[2], conpar[3], conpar[4]);

  // Outer trousers
  TGeoCompositeShape* pOutTrousersC = new TGeoCompositeShape("outTrousersC", "QCLext:ZDCC_c1+QCLext:ZDCC_c2");

  //auto& matmgr = o2::base::MaterialManager::Instance();

  // Volume: QCLext
  TGeoMedium* medZDCFeLowTh = gGeoManager->GetMedium("ZDC_IronLowTh$");
  TGeoVolume* pQCLext = new TGeoVolume("QCLext", pOutTrousersC, medZDCFeLowTh);
  pQCLext->SetLineColor(kAzure);
  pQCLext->SetVisLeaves(kTRUE);
  //
  TGeoTranslation* tr1c = new TGeoTranslation(0., 0., (Double_t)-conpar[0] - 0.95 - zC);
  //
  pZDCC->AddNode(pQCLext, 1, tr1c);
  // Inner trousers
  TGeoCompositeShape* pIntTrousersC = new TGeoCompositeShape("intTrousersC", "QCLint:ZDCC_c1+QCLint:ZDCC_c2");
  // Volume: QCLint
  TGeoMedium* medZDCvoid = gGeoManager->GetMedium("ZDC_VoidNoField$");
  TGeoVolume* pQCLint = new TGeoVolume("QCLint", pIntTrousersC, medZDCvoid);
  pQCLint->SetLineColor(kBlue);
  pQCLint->SetVisLeaves(kTRUE);
  pQCLext->AddNode(pQCLint, 1);

  zC += 90.1;
  Double_t offset = 0.5;
  zC = zC + offset;

  //  second section : 2 tubes (ID = 54. OD = 58.)
  tubpar[0] = 5.4 / 2.;
  tubpar[1] = 5.8 / 2.;
  tubpar[2] = 40.0 / 2.;
  TVirtualMC::GetMC()->Gsvolu("QT12", "TUBE", getMediumID(kVoidNoField), tubpar, 3);
  TVirtualMC::GetMC()->Gspos("QT12", 1, "ZDCC", -15.8 / 2., 0., -tubpar[2] - zC, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("QT12", 2, "ZDCC", 15.8 / 2., 0., -tubpar[2] - zC, 0, "ONLY");

  zC += 2. * tubpar[2];

  //-- rotation matrices for the legs
  Int_t irotpipe1, irotpipe2;
  double rang1[6] = { 90. - 1.0027, 0., 90., 90., 1.0027, 180. };
  double rang2[6] = { 90. + 1.0027, 0., 90., 90., 1.0027, 0. };
  TVirtualMC::GetMC()->Matrix(irotpipe1, rang1[0], rang1[1], rang1[2], rang1[3], rang1[4], rang1[5]);
  TVirtualMC::GetMC()->Matrix(irotpipe2, rang2[0], rang2[1], rang2[2], rang2[3], rang2[4], rang2[5]);

  // transition x2zdc to recombination chamber : skewed cone
  conpar[0] = (10. - 0.2 - offset) / 2.;
  conpar[1] = 6.3 / 2.;
  conpar[2] = 7.0 / 2.;
  conpar[3] = 5.4 / 2.;
  conpar[4] = 5.8 / 2.;
  TVirtualMC::GetMC()->Gsvolu("QC10", "CONE", getMediumID(kVoidNoField), conpar, 5);
  TVirtualMC::GetMC()->Gspos("QC10", 1, "ZDCC", -7.9 - 0.175, 0., -conpar[0] - 0.1 - zC, irotpipe1, "ONLY");
  TVirtualMC::GetMC()->Gspos("QC10", 2, "ZDCC", 7.9 + 0.175, 0., -conpar[0] - 0.1 - zC, irotpipe2, "ONLY");

  zC += 2. * conpar[0] + 0.2;

  // 2 tubes (ID = 63 mm OD=70 mm)
  tubpar[0] = 6.3 / 2.;
  tubpar[1] = 7.0 / 2.;
  tubpar[2] = 639.8 / 2.;
  TVirtualMC::GetMC()->Gsvolu("QT13", "TUBE", getMediumID(kVoidNoField), tubpar, 3);
  TVirtualMC::GetMC()->Gspos("QT13", 1, "ZDCC", -16.5 / 2., 0., -tubpar[2] - zC, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("QT13", 2, "ZDCC", 16.5 / 2., 0., -tubpar[2] - zC, 0, "ONLY");

  zC += 2. * tubpar[2];

  // -- Luminometer (Cu box) in front of ZN - side C
  if (mLumiLength > 0.) { // FIX IT!!!!!!!!!!!!!!!!!!!!!!!!
    boxpar[0] = 8.0 / 2.;
    boxpar[1] = 8.0 / 2.;
    boxpar[2] = mLumiLength / 2.; // FIX IT!!!!!!!!!!!!!!!!!!!!!!!!
    TVirtualMC::GetMC()->Gsvolu("QLUC", "BOX ", getMediumID(kCuLumi), boxpar, 3);
    TVirtualMC::GetMC()->Gspos("QLUC", 1, "ZDCC", 0., 0., Geometry::ZNCPOSITION[1] + 66. + boxpar[2], 0, "ONLY");
    LOG(DEBUG) << "	C-side luminometer positioned in front of ZNC\n";
  }
}

//_____________________________________________________________________________
void Detector::createMagnets()
{
  Float_t tubpar[3] = { 0., 0., 0. };
  Float_t boxpar[3] = { 0., 0., 0. };
  // Parameters from magnet DEFINITION
  double zCompensatorField = 1972.5;
  double zITField = 2296.5;
  double zD1Field = 5838.3001;
  double zD2Field = 12167.8;

  // ***************************************************************
  //		SIDE C
  // ***************************************************************
  // --  COMPENSATOR DIPOLE (MBXW)
  // --  GAP (VACUUM WITH MAGNETIC FIELD)
  tubpar[0] = 0.;
  tubpar[1] = 3.14;
  // New -> Added to accomodate AD (A. Morsch)
  // Updated -> The field must be 1.53 m long
  tubpar[2] = 153. / 2.;
  TVirtualMC::GetMC()->Gsvolu("MBXW", "TUBE", getMediumID(kVoidwField), tubpar, 3);
  TVirtualMC::GetMC()->Gspos("MBXW", 1, "ZDCC", 0., 0., -tubpar[2] - zCompensatorField, 0, "ONLY");
  // --  YOKE
  tubpar[0] = 4.5;
  tubpar[1] = 55.;
  // Updated -> The yoke can be 1.50 m to avoid overlaps
  tubpar[2] = 150. / 2.;
  TVirtualMC::GetMC()->Gsvolu("YMBX", "TUBE", getMediumID(kVoidNoField), tubpar, 3);
  TVirtualMC::GetMC()->Gspos("YMBX", 1, "ZDCC", 0., 0., -tubpar[2] - zCompensatorField - 1.5, 0, "ONLY");

  // -- INNER TRIPLET
  // -- DEFINE MQXL AND MQX QUADRUPOLE ELEMENT
  // --  MQXL
  // --  GAP (VACUUM WITH MAGNETIC FIELD)
  tubpar[0] = 0.;
  tubpar[1] = 3.14;
  tubpar[2] = 637. / 2.;
  TVirtualMC::GetMC()->Gsvolu("MQXL", "TUBE", getMediumID(kVoidwField), tubpar, 3);

  // --  YOKE
  tubpar[0] = 3.5;
  tubpar[1] = 22.;
  tubpar[2] = 637. / 2.;
  TVirtualMC::GetMC()->Gsvolu("YMQL", "TUBE", getMediumID(kVoidNoField), tubpar, 3);

  TVirtualMC::GetMC()->Gspos("MQXL", 1, "ZDCC", 0., 0., -tubpar[2] - zITField, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("YMQL", 1, "ZDCC", 0., 0., -tubpar[2] - zITField, 0, "ONLY");

  TVirtualMC::GetMC()->Gspos("MQXL", 2, "ZDCC", 0., 0., -tubpar[2] - zITField - 2400., 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("YMQL", 2, "ZDCC", 0., 0., -tubpar[2] - zITField - 2400., 0, "ONLY");

  // --  MQX
  // --  GAP (VACUUM WITH MAGNETIC FIELD)
  tubpar[0] = 0.;
  tubpar[1] = 3.14;
  tubpar[2] = 550. / 2.;
  TVirtualMC::GetMC()->Gsvolu("MQX ", "TUBE", getMediumID(kVoidwField), tubpar, 3);

  // --  YOKE
  tubpar[0] = 3.5;
  tubpar[1] = 22.;
  tubpar[2] = 550. / 2.;
  TVirtualMC::GetMC()->Gsvolu("YMQ ", "TUBE", getMediumID(kVoidNoField), tubpar, 3);

  TVirtualMC::GetMC()->Gspos("MQX ", 1, "ZDCC", 0., 0., -tubpar[2] - zITField - 908.5, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("YMQ ", 1, "ZDCC", 0., 0., -tubpar[2] - zITField - 908.5, 0, "ONLY");

  TVirtualMC::GetMC()->Gspos("MQX ", 2, "ZDCC", 0., 0., -tubpar[2] - zITField - 1558.5, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("YMQ ", 2, "ZDCC", 0., 0., -tubpar[2] - zITField - 1558.5, 0, "ONLY");

  // -- SEPARATOR DIPOLE D1
  // --  GAP (VACUUM WITH MAGNETIC FIELD)
  tubpar[0] = 0.;
  tubpar[1] = 3.46;
  tubpar[2] = 945. / 2.;
  TVirtualMC::GetMC()->Gsvolu("MD1 ", "TUBE", getMediumID(kVoidwField), tubpar, 3);

  // --  Insert horizontal Cu plates inside D1
  // --   (to simulate the vacuum chamber)
  boxpar[0] = TMath::Sqrt(tubpar[1] * tubpar[1] - (2.98 + 0.2) * (2.98 + 0.2)) - 0.05;
  boxpar[1] = 0.2 / 2.;
  boxpar[2] = 945. / 2.;
  TVirtualMC::GetMC()->Gsvolu("MD1V", "BOX ", getMediumID(kCu), boxpar, 3);
  TVirtualMC::GetMC()->Gspos("MD1V", 1, "MD1 ", 0., 2.98 + boxpar[1], 0., 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("MD1V", 2, "MD1 ", 0., -2.98 - boxpar[1], 0., 0, "ONLY");

  // --  YOKE
  tubpar[0] = 3.68;
  tubpar[1] = 110. / 2.;
  tubpar[2] = 945. / 2.;
  TVirtualMC::GetMC()->Gsvolu("YD1 ", "TUBE", getMediumID(kVoidNoField), tubpar, 3);

  TVirtualMC::GetMC()->Gspos("YD1 ", 1, "ZDCC", 0., 0., -tubpar[2] - zD1Field, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("MD1 ", 1, "ZDCC", 0., 0., -tubpar[2] - zD1Field, 0, "ONLY");

  // -- DIPOLE D2
  // --  GAP (VACUUM WITH MAGNETIC FIELD)
  tubpar[0] = 0.;
  tubpar[1] = 7.5 / 2.;
  tubpar[2] = 945. / 2.;
  TVirtualMC::GetMC()->Gsvolu("MD2 ", "TUBE", getMediumID(kVoidwField), tubpar, 3);

  // --  YOKE
  tubpar[0] = 0.;
  tubpar[1] = 55.;
  tubpar[2] = 945. / 2.;
  TVirtualMC::GetMC()->Gsvolu("YD2 ", "TUBE", getMediumID(kVoidNoField), tubpar, 3);
  TVirtualMC::GetMC()->Gspos("YD2 ", 1, "ZDCC", 0., 0., -tubpar[2] - zD2Field, 0, "ONLY");

  TVirtualMC::GetMC()->Gspos("MD2 ", 1, "YD2 ", -9.4, 0., 0., 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("MD2 ", 2, "YD2 ", 9.4, 0., 0., 0, "ONLY");

  // ***************************************************************
  //		SIDE A
  // ***************************************************************

  // COMPENSATOR DIPOLE (MCBWA) (2nd compensator)
  // --  GAP (VACUUM WITH MAGNETIC FIELD)
  tubpar[0] = 0.;
  tubpar[1] = 3.;
  tubpar[2] = 153. / 2.;
  TVirtualMC::GetMC()->Gsvolu("MCBW", "TUBE", getMediumID(kVoidwField), tubpar, 3);
  TVirtualMC::GetMC()->Gspos("MCBW", 1, "ZDCA", 0., 0., tubpar[2] + zCompensatorField, 0, "ONLY");

  // --  YOKE
  tubpar[0] = 4.5;
  tubpar[1] = 55.;
  tubpar[2] = 153. / 2.;
  TVirtualMC::GetMC()->Gsvolu("YMCB", "TUBE", getMediumID(kVoidNoField), tubpar, 3);
  TVirtualMC::GetMC()->Gspos("YMCB", 1, "ZDCA", 0., 0., tubpar[2] + zCompensatorField, 0, "ONLY");

  // -- INNER TRIPLET
  // -- DEFINE MQX1 AND MQX2 QUADRUPOLE ELEMENT
  // --  MQX1
  // --  GAP (VACUUM WITH MAGNETIC FIELD)
  tubpar[0] = 0.;
  tubpar[1] = 3.14;
  tubpar[2] = 637. / 2.;
  TVirtualMC::GetMC()->Gsvolu("MQX1", "TUBE", getMediumID(kVoidwField), tubpar, 3);
  TVirtualMC::GetMC()->Gsvolu("MQX4", "TUBE", getMediumID(kVoidwField), tubpar, 3);

  // --  YOKE
  tubpar[0] = 3.5;
  tubpar[1] = 22.;
  tubpar[2] = 637. / 2.;
  TVirtualMC::GetMC()->Gsvolu("YMQ1", "TUBE", getMediumID(kVoidNoField), tubpar, 3);

  // -- Q1
  TVirtualMC::GetMC()->Gspos("MQX1", 1, "ZDCA", 0., 0., tubpar[2] + zITField, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("YMQ1", 1, "ZDCA", 0., 0., tubpar[2] + zITField, 0, "ONLY");

  // -- BEAM SCREEN FOR Q1
  tubpar[0] = 4.78 / 2.;
  tubpar[1] = 5.18 / 2.;
  tubpar[2] = 637. / 2.;
  TVirtualMC::GetMC()->Gsvolu("QBS1", "TUBE", getMediumID(kCu), tubpar, 3);
  TVirtualMC::GetMC()->Gspos("QBS1", 1, "MQX1", 0., 0., 0., 0, "ONLY");
  // INSERT VERTICAL PLATE INSIDE Q1
  boxpar[0] = 0.2 / 2.0;
  boxpar[1] = TMath::Sqrt(tubpar[0] * tubpar[0] - (1.9 + 0.2) * (1.9 + 0.2));
  boxpar[2] = 637. / 2.;
  TVirtualMC::GetMC()->Gsvolu("QBS2", "BOX ", getMediumID(kCu), boxpar, 3);
  TVirtualMC::GetMC()->Gspos("QBS2", 1, "MQX1", 1.9 + boxpar[0], 0., 0., 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("QBS2", 2, "MQX1", -1.9 - boxpar[0], 0., 0., 0, "ONLY");

  // -- Q3
  TVirtualMC::GetMC()->Gspos("MQX4", 1, "ZDCA", 0., 0., tubpar[2] + zITField + 2400., 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("YMQ1", 2, "ZDCA", 0., 0., tubpar[2] + zITField + 2400., 0, "ONLY");

  // -- BEAM SCREEN FOR Q3
  tubpar[0] = 5.79 / 2.;
  tubpar[1] = 6.14 / 2.;
  tubpar[2] = 637. / 2.;
  TVirtualMC::GetMC()->Gsvolu("QBS3", "TUBE", getMediumID(kCu), tubpar, 3);
  TVirtualMC::GetMC()->Gspos("QBS3", 1, "MQX4", 0., 0., 0., 0, "ONLY");
  // INSERT VERTICAL PLATE INSIDE Q3
  boxpar[0] = 0.2 / 2.0;
  boxpar[1] = TMath::Sqrt(tubpar[0] * tubpar[0] - (2.405 + 0.2) * (2.405 + 0.2));
  boxpar[2] = 637. / 2.;
  TVirtualMC::GetMC()->Gsvolu("QBS4", "BOX ", getMediumID(kCu), boxpar, 3);
  TVirtualMC::GetMC()->Gspos("QBS4", 1, "MQX4", 2.405 + boxpar[0], 0., 0., 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("QBS4", 2, "MQX4", -2.405 - boxpar[0], 0., 0., 0, "ONLY");

  // --  MQX2
  // --  GAP (VACUUM WITH MAGNETIC FIELD)
  tubpar[0] = 0.;
  tubpar[1] = 3.14;
  tubpar[2] = 550. / 2.;
  TVirtualMC::GetMC()->Gsvolu("MQX2", "TUBE", getMediumID(kVoidwField), tubpar, 3);
  TVirtualMC::GetMC()->Gsvolu("MQX3", "TUBE", getMediumID(kVoidwField), tubpar, 3);

  // --  YOKE
  tubpar[0] = 3.5;
  tubpar[1] = 22.;
  tubpar[2] = 550. / 2.;
  TVirtualMC::GetMC()->Gsvolu("YMQ2", "TUBE", getMediumID(kVoidNoField), tubpar, 3);

  // -- BEAM SCREEN FOR Q2
  tubpar[0] = 5.79 / 2.;
  tubpar[1] = 6.14 / 2.;
  tubpar[2] = 550. / 2.;
  TVirtualMC::GetMC()->Gsvolu("QBS5", "TUBE", getMediumID(kCu), tubpar, 3);
  //    VERTICAL PLATE INSIDE Q2
  boxpar[0] = 0.2 / 2.0;
  boxpar[1] = TMath::Sqrt(tubpar[0] * tubpar[0] - (2.405 + 0.2) * (2.405 + 0.2));
  boxpar[2] = 550. / 2.;
  TVirtualMC::GetMC()->Gsvolu("QBS6", "BOX ", getMediumID(kCu), boxpar, 3);

  // -- Q2A
  TVirtualMC::GetMC()->Gspos("MQX2", 1, "ZDCA", 0., 0., tubpar[2] + zITField + 908.5, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("QBS5", 1, "MQX2", 0., 0., 0., 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("QBS6", 1, "MQX2", 2.405 + boxpar[0], 0., 0., 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("QBS6", 2, "MQX2", -2.405 - boxpar[0], 0., 0., 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("YMQ2", 1, "ZDCA", 0., 0., tubpar[2] + zITField + 908.5, 0, "ONLY");

  // -- Q2B
  TVirtualMC::GetMC()->Gspos("MQX3", 1, "ZDCA", 0., 0., tubpar[2] + zITField + 1558.5, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("QBS5", 2, "MQX3", 0., 0., 0., 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("QBS6", 3, "MQX3", 2.405 + boxpar[0], 0., 0., 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("QBS6", 4, "MQX3", -2.405 - boxpar[0], 0., 0., 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("YMQ2", 2, "ZDCA", 0., 0., tubpar[2] + zITField + 1558.5, 0, "ONLY");

  // -- SEPARATOR DIPOLE D1
  // --  GAP (VACUUM WITH MAGNETIC FIELD)
  tubpar[0] = 0.;
  tubpar[1] = 6.75 / 2.; //3.375
  tubpar[2] = 945. / 2.;
  TVirtualMC::GetMC()->Gsvolu("MD1L", "TUBE", getMediumID(kVoidwField), tubpar, 3);

  // --  The beam screen tube is provided by the beam pipe in D1 (QA03 volume)
  // --  Insert the beam screen horizontal Cu plates inside D1 to simulate the vacuum chamber
  boxpar[0] = TMath::Sqrt(tubpar[1] * tubpar[1] - (2.885 + 0.2) * (2.885 + 0.2));
  boxpar[1] = 0.2 / 2.;
  boxpar[2] = 945. / 2.;
  TVirtualMC::GetMC()->Gsvolu("QBS7", "BOX ", getMediumID(kCu), boxpar, 3);
  TVirtualMC::GetMC()->Gspos("QBS7", 1, "MD1L", 0., 2.885 + boxpar[1], 0., 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("QBS7", 2, "MD1L", 0., -2.885 - boxpar[1], 0., 0, "ONLY");

  // --  YOKE
  tubpar[0] = 3.68;
  tubpar[1] = 110. / 2;
  tubpar[2] = 945. / 2.;
  TVirtualMC::GetMC()->Gsvolu("YD1L", "TUBE", getMediumID(kVoidNoField), tubpar, 3);

  TVirtualMC::GetMC()->Gspos("YD1L", 1, "ZDCA", 0., 0., tubpar[2] + zD1Field, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("MD1L", 1, "ZDCA", 0., 0., tubpar[2] + zD1Field, 0, "ONLY");

  // -- DIPOLE D2
  // --  GAP (VACUUM WITH MAGNETIC FIELD)
  tubpar[0] = 0.;
  tubpar[1] = 7.5 / 2.; // this has to be checked
  tubpar[2] = 945. / 2.;
  TVirtualMC::GetMC()->Gsvolu("MD2L", "TUBE", getMediumID(kVoidwField), tubpar, 3);

  // --  YOKE
  tubpar[0] = 0.;
  tubpar[1] = 55.;
  tubpar[2] = 945. / 2.;
  TVirtualMC::GetMC()->Gsvolu("YD2L", "TUBE", getMediumID(kVoidNoField), tubpar, 3);
  TVirtualMC::GetMC()->Gspos("YD2L", 1, "ZDCA", 0., 0., tubpar[2] + zD2Field, 0, "ONLY");

  TVirtualMC::GetMC()->Gspos("MD2L", 1, "YD2L", -9.4, 0., 0., 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("MD2L", 2, "YD2L", 9.4, 0., 0., 0, "ONLY");
}
//_____________________________________________________________________________
void Detector::createDetectors()
{
  // Create the ZDCs

  double znSupportBase[3] = { 6.3, 4.57, 71.2 }; //Basement of ZN table (thick one)
  double znSupportBasePos[3] = { 0., -14., 21.2 };
  double znSupportScintillH[3] = { 4.32 - 0.8, 0.8, 50. }; //Scintillator container: top&bottom
  double znSupportScintillV[3] = { 0.8, 1.955, 50. };      //Scintillator container: sides
  double znSupportWallsud[3] = { 3.52, 1., 50. };          //Top and bottom walls
  double znSupportWallside[3] = { 0.4, 5.52, 50. };        //Side walls

  Float_t dimPb[6], dimVoid[6];

  // -------------------------------------------------------------------------------
  //--> Neutron calorimeter (ZN)
  mMediumPMCid = getMediumID(kSiO2pmc);
  mMediumPMQid = getMediumID(kSiO2pmq);

  // an envelop volume for the purpose of registering particles entering the detector
  double eps = 0.1; // 1 mm
  double neu_envelopdim[3] = { Geometry::ZNDIMENSION[0] + eps, Geometry::ZNDIMENSION[1] + eps, Geometry::ZNDIMENSION[2] + eps };
  TVirtualMC::GetMC()->Gsvolu("ZNENV", "BOX ", getMediumID(kVoidNoField), neu_envelopdim, 3);

  TVirtualMC::GetMC()->Gsvolu("ZNEU", "BOX ", getMediumID(kWalloy), const_cast<double*>(Geometry::ZNDIMENSION), 3); // Passive material
  TVirtualMC::GetMC()->Gsvolu("ZNF1", "TUBE", mMediumPMCid, const_cast<double*>(Geometry::ZNFIBRE), 3);             // Active material
  TVirtualMC::GetMC()->Gsvolu("ZNF2", "TUBE", mMediumPMQid, const_cast<double*>(Geometry::ZNFIBRE), 3);
  TVirtualMC::GetMC()->Gsvolu("ZNF3", "TUBE", mMediumPMQid, const_cast<double*>(Geometry::ZNFIBRE), 3);
  TVirtualMC::GetMC()->Gsvolu("ZNF4", "TUBE", mMediumPMCid, const_cast<double*>(Geometry::ZNFIBRE), 3);
  TVirtualMC::GetMC()->Gsvolu("ZNG1", "BOX ", getMediumID(kAir), const_cast<double*>(Geometry::ZNGROOVES), 3); // Empty grooves
  TVirtualMC::GetMC()->Gsvolu("ZNG2", "BOX ", getMediumID(kAir), const_cast<double*>(Geometry::ZNGROOVES), 3);
  TVirtualMC::GetMC()->Gsvolu("ZNG3", "BOX ", getMediumID(kAir), const_cast<double*>(Geometry::ZNGROOVES), 3);
  TVirtualMC::GetMC()->Gsvolu("ZNG4", "BOX ", getMediumID(kAir), const_cast<double*>(Geometry::ZNGROOVES), 3);

  // Divide ZNEU in quadrants
  TVirtualMC::GetMC()->Gsdvn("ZNTX", "ZNEU", Geometry::ZNSECTORS[0], 1); // x-tower
  TVirtualMC::GetMC()->Gsdvn("ZN1 ", "ZNTX", Geometry::ZNSECTORS[1], 2); // y-tower

  //-- Divide ZN1 in minitowers (4 fibres per minitower)
  //  ZNDIVISION[0]= NUMBER OF FIBERS PER TOWER ALONG X-AXIS,
  //  ZNDIVISION[1]= NUMBER OF FIBERS PER TOWER ALONG Y-AXIS
  TVirtualMC::GetMC()->Gsdvn("ZNSL", "ZN1 ", Geometry::ZNDIVISION[1], 2); // Slices
  TVirtualMC::GetMC()->Gsdvn("ZNST", "ZNSL", Geometry::ZNDIVISION[0], 1); // Sticks

  // --- Position the empty grooves in the sticks (4 grooves per stick)
  Float_t dx = Geometry::ZNDIMENSION[0] / Geometry::ZNDIVISION[0] / 4.;
  Float_t dy = Geometry::ZNDIMENSION[1] / Geometry::ZNDIVISION[1] / 4.;

  TVirtualMC::GetMC()->Gspos("ZNG1", 1, "ZNST", 0. - dx, 0. + dy, 0., 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("ZNG2", 1, "ZNST", 0. + dx, 0. + dy, 0., 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("ZNG3", 1, "ZNST", 0. - dx, 0. - dy, 0., 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("ZNG4", 1, "ZNST", 0. + dx, 0. - dy, 0., 0, "ONLY");

  // --- Position the fibers in the grooves
  TVirtualMC::GetMC()->Gspos("ZNF1", 1, "ZNG1", 0., 0., 0., 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("ZNF2", 1, "ZNG2", 0., 0., 0., 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("ZNF3", 1, "ZNG3", 0., 0., 0., 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("ZNF4", 1, "ZNG4", 0., 0., 0., 0, "ONLY");

  // --- Position the neutron calorimeter in ZDC
  // -- Rotation of C side ZN
  Int_t irotznc;
  double rangznc[6] = { 90., 180., 90., 90., 180., 0. };
  TVirtualMC::GetMC()->Matrix(irotznc, rangznc[0], rangznc[1], rangznc[2], rangznc[3], rangznc[4], rangznc[5]);
  //
  TVirtualMC::GetMC()->Gspos("ZNEU", 1, "ZNENV", 0., 0., 0., 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("ZNENV", 1, "ZDCC", Geometry::ZNCPOSITION[0], Geometry::ZNCPOSITION[1], Geometry::ZNCPOSITION[2] - Geometry::ZNDIMENSION[2], irotznc, "ONLY");

  // --- Position the neutron calorimeter on the A side
  TVirtualMC::GetMC()->Gspos("ZNENV", 2, "ZDCA", Geometry::ZNAPOSITION[0], Geometry::ZNAPOSITION[1], Geometry::ZNAPOSITION[2] + Geometry::ZNDIMENSION[2], 0, "ONLY");

  // -------------------------------------------------------------------------------
  // -> ZN supports

  // Basements (A and C sides)
  TVirtualMC::GetMC()->Gsvolu("ZNBASE", "BOX ", getMediumID(kAl), znSupportBase, 3);
  TVirtualMC::GetMC()->Gspos("ZNBASE", 1, "ZDCC", Geometry::ZNCPOSITION[0] + znSupportBasePos[0],
                             Geometry::ZNCPOSITION[1] + znSupportBasePos[1], Geometry::ZNCPOSITION[2] - znSupportBase[2], 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("ZNBASE", 2, "ZDCA", Geometry::ZNAPOSITION[0] + znSupportBasePos[0],
                             Geometry::ZNAPOSITION[1] + znSupportBasePos[1], Geometry::ZNAPOSITION[2] + znSupportBase[2], 0, "ONLY");

  // Box containing scintillators (C side)
  TVirtualMC::GetMC()->Gsvolu("ZNSCH", "BOX ", getMediumID(kAl), znSupportScintillH, 3);
  TVirtualMC::GetMC()->Gspos("ZNSCH", 1, "ZDCC", Geometry::ZNCPOSITION[0] + znSupportBasePos[0],
                             Geometry::ZNCPOSITION[1] + znSupportBasePos[1] + znSupportBase[1] + znSupportScintillH[1], Geometry::ZNCPOSITION[2] - znSupportScintillH[2], 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("ZNSCH", 2, "ZDCC", Geometry::ZNCPOSITION[0] + znSupportBasePos[0],
                             Geometry::ZNCPOSITION[1] - Geometry::ZNDIMENSION[1] - znSupportScintillV[1] + znSupportScintillH[1], Geometry::ZNCPOSITION[2] - znSupportScintillH[2], 0, "ONLY");

  TVirtualMC::GetMC()->Gsvolu("ZNSCV", "BOX ", getMediumID(kAl), znSupportScintillV, 3);
  TVirtualMC::GetMC()->Gspos("ZNSCV", 1, "ZDCC", Geometry::ZNCPOSITION[0] + znSupportBasePos[0] + znSupportScintillH[0] + znSupportScintillV[0],
                             Geometry::ZNCPOSITION[1] + znSupportBasePos[1] + znSupportBase[1] + znSupportScintillV[1], Geometry::ZNCPOSITION[2] - znSupportScintillV[2], 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("ZNSCV", 2, "ZDCC", Geometry::ZNCPOSITION[0] + znSupportBasePos[0] - znSupportScintillH[0] - znSupportScintillV[0],
                             Geometry::ZNCPOSITION[1] + znSupportBasePos[1] + znSupportBase[1] + znSupportScintillV[1], Geometry::ZNCPOSITION[2] - znSupportScintillV[2], 0, "ONLY");

  // Box containing scintillators (A side)
  TVirtualMC::GetMC()->Gsvolu("ZNSCH", "BOX ", getMediumID(kAl), znSupportScintillH, 3);
  TVirtualMC::GetMC()->Gspos("ZNSCH", 1, "ZDCA", Geometry::ZNAPOSITION[0] + znSupportBasePos[0],
                             Geometry::ZNAPOSITION[1] + znSupportBasePos[1] + znSupportBase[1] + znSupportScintillH[1], Geometry::ZNAPOSITION[2] + znSupportScintillH[2], 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("ZNSCH", 2, "ZDCA", Geometry::ZNAPOSITION[0] + znSupportBasePos[0],
                             Geometry::ZNAPOSITION[1] - Geometry::ZNDIMENSION[1] - znSupportScintillV[1] + znSupportScintillH[1], Geometry::ZNAPOSITION[2] + znSupportScintillH[2], 0, "ONLY");

  TVirtualMC::GetMC()->Gsvolu("ZNSCV", "BOX ", getMediumID(kAl), const_cast<double*>(znSupportScintillV), 3);
  TVirtualMC::GetMC()->Gspos("ZNSCV", 1, "ZDCA", Geometry::ZNAPOSITION[0] + znSupportBasePos[0] + znSupportScintillH[0] + znSupportScintillV[0],
                             Geometry::ZNAPOSITION[1] + znSupportBasePos[1] + znSupportBase[1] + znSupportScintillV[1], Geometry::ZNAPOSITION[2] + znSupportScintillV[2], 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("ZNSCV", 2, "ZDCA", Geometry::ZNAPOSITION[0] + znSupportBasePos[0] - znSupportScintillH[0] - znSupportScintillV[0],
                             Geometry::ZNAPOSITION[1] + znSupportBasePos[1] + znSupportBase[1] + znSupportScintillV[1], Geometry::ZNAPOSITION[2] + znSupportScintillV[2], 0, "ONLY");

  // ZNC Box (A and C sides)
  // Top & bottom walls
  TVirtualMC::GetMC()->Gsvolu("ZNBH", "BOX ", getMediumID(kAl), const_cast<double*>(znSupportWallsud), 3);
  TVirtualMC::GetMC()->Gspos("ZNBH", 1, "ZDCC", Geometry::ZNCPOSITION[0] + znSupportBasePos[0],
                             Geometry::ZNCPOSITION[1] - Geometry::ZNDIMENSION[1] - znSupportWallsud[1], Geometry::ZNCPOSITION[2] - znSupportWallsud[2], 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("ZNBH", 2, "ZDCC", Geometry::ZNCPOSITION[0] + znSupportBasePos[0],
                             Geometry::ZNCPOSITION[1] + Geometry::ZNDIMENSION[1] + znSupportWallsud[1], Geometry::ZNCPOSITION[2] - znSupportWallsud[2], 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("ZNBH", 3, "ZDCA", Geometry::ZNAPOSITION[0] + znSupportBasePos[0],
                             Geometry::ZNAPOSITION[1] - Geometry::ZNDIMENSION[1] - znSupportWallsud[1], Geometry::ZNAPOSITION[2] + znSupportWallsud[2], 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("ZNBH", 4, "ZDCA", Geometry::ZNAPOSITION[0] + znSupportBasePos[0],
                             Geometry::ZNAPOSITION[1] + Geometry::ZNDIMENSION[1] + znSupportWallsud[1], Geometry::ZNAPOSITION[2] + znSupportWallsud[2], 0, "ONLY");

  // Side walls
  TVirtualMC::GetMC()->Gsvolu("ZNBS", "BOX ", getMediumID(kAl), const_cast<double*>(znSupportWallside), 3);
  TVirtualMC::GetMC()->Gspos("ZNBS", 1, "ZDCC", Geometry::ZNCPOSITION[0] + Geometry::ZNDIMENSION[0] + znSupportWallside[0],
                             Geometry::ZNCPOSITION[1], Geometry::ZNCPOSITION[2] - znSupportWallside[2], 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("ZNBS", 2, "ZDCC", Geometry::ZNCPOSITION[0] - Geometry::ZNDIMENSION[0] - znSupportWallside[0],
                             Geometry::ZNCPOSITION[1], Geometry::ZNCPOSITION[2] - znSupportWallsud[2], 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("ZNBS", 3, "ZDCA", Geometry::ZNAPOSITION[0] + Geometry::ZNDIMENSION[0] + znSupportWallside[0],
                             Geometry::ZNAPOSITION[1], Geometry::ZNAPOSITION[2] + znSupportWallside[2], 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("ZNBS", 4, "ZDCA", Geometry::ZNAPOSITION[0] - Geometry::ZNDIMENSION[0] - znSupportWallside[0],
                             Geometry::ZNAPOSITION[1], Geometry::ZNAPOSITION[2] + znSupportWallsud[2], 0, "ONLY");

  // -------------------------------------------------------------------------------
  //--> Proton calorimeter (ZP)
  double zpSupportBase1[3] = { 12.5, 1.4, 75. }; //Bottom basement of ZP table (thinner one)
  double zpSupportBase1Pos[3] = { 0., -17., 0. };
  double zpSupportBase2[3] = { 12.5, 2.5, 75. }; //Upper basement of ZP table (thicker one)
  double zpSupportBase2Pos[3] = { 0., -9., 0. };
  double zpSupportBase3[3] = { 1.5, 2.05, 75. };       //support table heels (piedini)
  double zpSupportWallBottom[3] = { 11.2, 0.25, 75. }; //Bottom wall
  double zpSupportWallup[3] = { 11.2, 1., 75. };       //Top wall
  //double zpSupportWallside[3] = {0.5, 7.25, 75.}; //Side walls (original)
  double zpSupportWallside[3] = { 0.5, 6., 75. }; //Side walls (modified)

  double pro_envelopdim[3] = { Geometry::ZPDIMENSION[0] + eps, Geometry::ZPDIMENSION[1] + eps, Geometry::ZPDIMENSION[2] + eps };
  TVirtualMC::GetMC()->Gsvolu("ZPENV", "BOX", getMediumID(kVoidNoField), pro_envelopdim, 3);

  TVirtualMC::GetMC()->Gsvolu("ZPRO", "BOX ", getMediumID(kCuZn), const_cast<double*>(Geometry::ZPDIMENSION), 3); // Passive material
  TVirtualMC::GetMC()->Gsvolu("ZPF1", "TUBE", getMediumID(kSiO2pmc), const_cast<double*>(Geometry::ZPFIBRE), 3);  // Active material
  TVirtualMC::GetMC()->Gsvolu("ZPF2", "TUBE", getMediumID(kSiO2pmq), const_cast<double*>(Geometry::ZPFIBRE), 3);
  TVirtualMC::GetMC()->Gsvolu("ZPF3", "TUBE", getMediumID(kSiO2pmq), const_cast<double*>(Geometry::ZPFIBRE), 3);
  TVirtualMC::GetMC()->Gsvolu("ZPF4", "TUBE", getMediumID(kSiO2pmc), const_cast<double*>(Geometry::ZPFIBRE), 3);
  TVirtualMC::GetMC()->Gsvolu("ZPG1", "BOX ", getMediumID(kAir), const_cast<double*>(Geometry::ZPGROOVES), 3); // Empty grooves
  TVirtualMC::GetMC()->Gsvolu("ZPG2", "BOX ", getMediumID(kAir), const_cast<double*>(Geometry::ZPGROOVES), 3);
  TVirtualMC::GetMC()->Gsvolu("ZPG3", "BOX ", getMediumID(kAir), const_cast<double*>(Geometry::ZPGROOVES), 3);
  TVirtualMC::GetMC()->Gsvolu("ZPG4", "BOX ", getMediumID(kAir), const_cast<double*>(Geometry::ZPGROOVES), 3);

  //-- Divide ZPRO in towers
  TVirtualMC::GetMC()->Gsdvn("ZPTX", "ZPRO", Geometry::ZPSECTORS[0], 1); // x-tower
  TVirtualMC::GetMC()->Gsdvn("ZP1 ", "ZPTX", Geometry::ZPSECTORS[1], 2); // y-tower

  //-- Divide ZP1 in minitowers (4 fiber per minitower)
  //  ZPDIVISION[0]= NUMBER OF FIBERS ALONG X-AXIS PER MINITOWER,
  //  ZPDIVISION[1]= NUMBER OF FIBERS ALONG Y-AXIS PER MINITOWER
  TVirtualMC::GetMC()->Gsdvn("ZPSL", "ZP1 ", Geometry::ZPDIVISION[1], 2); // Slices
  TVirtualMC::GetMC()->Gsdvn("ZPST", "ZPSL", Geometry::ZPDIVISION[0], 1); // Sticks

  // --- Position the empty grooves in the sticks (4 grooves per stick)
  dx = Geometry::ZPDIMENSION[0] / Geometry::ZPSECTORS[0] / Geometry::ZPDIVISION[0] / 2.;
  dy = Geometry::ZPDIMENSION[1] / Geometry::ZPSECTORS[1] / Geometry::ZPDIVISION[1] / 2.;

  TVirtualMC::GetMC()->Gspos("ZPG1", 1, "ZPST", 0. - dx, 0. + dy, 0., 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("ZPG2", 1, "ZPST", 0. + dx, 0. + dy, 0., 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("ZPG3", 1, "ZPST", 0. - dx, 0. - dy, 0., 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("ZPG4", 1, "ZPST", 0. + dx, 0. - dy, 0., 0, "ONLY");

  // --- Position the fibers in the grooves
  TVirtualMC::GetMC()->Gspos("ZPF1", 1, "ZPG1", 0., 0., 0., 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("ZPF2", 1, "ZPG2", 0., 0., 0., 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("ZPF3", 1, "ZPG3", 0., 0., 0., 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("ZPF4", 1, "ZPG4", 0., 0., 0., 0, "ONLY");

  // --- Position the proton calorimeter in ZDCC
  // -- Rotation of C side ZP
  TVirtualMC::GetMC()->Gspos("ZPRO", 1, "ZPENV", 0., 0., 0., 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("ZPENV", 1, "ZDCC", Geometry::ZPCPOSITION[0], Geometry::ZPCPOSITION[1], Geometry::ZPCPOSITION[2] - Geometry::ZPDIMENSION[2], irotznc, "ONLY");

  // --- Position the proton calorimeter in ZDCA
  TVirtualMC::GetMC()->Gspos("ZPENV", 2, "ZDCA", Geometry::ZPAPOSITION[0], Geometry::ZPAPOSITION[1], Geometry::ZPAPOSITION[2] + Geometry::ZPDIMENSION[2], 0, "ONLY");

  // -------------------------------------------------------------------------------
  // -> ZP supports

  // Bottom basements (A and C sides)
  TVirtualMC::GetMC()->Gsvolu("ZPBASE1", "BOX ", getMediumID(kAl), const_cast<double*>(zpSupportBase1), 3);
  TVirtualMC::GetMC()->Gspos("ZPBASE1", 1, "ZDCC", Geometry::ZPCPOSITION[0] + zpSupportBase1Pos[0],
                             Geometry::ZPCPOSITION[1] + zpSupportBase1Pos[1], Geometry::ZPCPOSITION[2] - zpSupportBase1[2], 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("ZPBASE1", 2, "ZDCA", Geometry::ZPAPOSITION[0] + zpSupportBase1Pos[0],
                             Geometry::ZPAPOSITION[1] + zpSupportBase1Pos[1], Geometry::ZPAPOSITION[2] + zpSupportBase1[2], 0, "ONLY");

  // Bottom foot between 2 basements (A and C sides)
  TVirtualMC::GetMC()->Gsvolu("ZPFOOT", "BOX ", getMediumID(kAl), const_cast<double*>(zpSupportBase3), 3);
  TVirtualMC::GetMC()->Gspos("ZPFOOT", 1, "ZDCC", Geometry::ZPCPOSITION[0] + zpSupportBase1Pos[0] - zpSupportBase1[0] + zpSupportBase3[0], Geometry::ZPCPOSITION[1] + zpSupportBase1Pos[1] + zpSupportBase1[1] + zpSupportBase3[1], Geometry::ZPCPOSITION[2] - zpSupportBase3[2], 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("ZPFOOT", 2, "ZDCC", Geometry::ZPCPOSITION[0] + zpSupportBase1Pos[0] + zpSupportBase1[0] - zpSupportBase3[0], Geometry::ZPCPOSITION[1] + zpSupportBase1Pos[1] + zpSupportBase1[1] + zpSupportBase3[1], Geometry::ZPCPOSITION[2] - zpSupportBase3[2], 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("ZPFOOT", 3, "ZDCA", Geometry::ZPAPOSITION[0] + zpSupportBase1Pos[0] - zpSupportBase1[0] + zpSupportBase3[0], Geometry::ZPAPOSITION[1] + zpSupportBase1Pos[1] + zpSupportBase1[1] + zpSupportBase3[1], Geometry::ZPAPOSITION[2] + zpSupportBase3[2], 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("ZPFOOT", 4, "ZDCA", Geometry::ZPAPOSITION[0] + zpSupportBase1Pos[0] + zpSupportBase1[0] - zpSupportBase3[0], Geometry::ZPAPOSITION[1] + zpSupportBase1Pos[1] + zpSupportBase1[1] + zpSupportBase3[1], Geometry::ZPAPOSITION[2] + zpSupportBase3[2], 0, "ONLY");

  // Upper basements (A and C sides)
  TVirtualMC::GetMC()->Gsvolu("ZPBASE2", "BOX ", getMediumID(kAl), const_cast<double*>(zpSupportBase2), 3);
  TVirtualMC::GetMC()->Gspos("ZPBASE2", 1, "ZDCC", Geometry::ZPCPOSITION[0] + zpSupportBase2Pos[0],
                             Geometry::ZPCPOSITION[1] + zpSupportBase2Pos[1], Geometry::ZPCPOSITION[2] - zpSupportBase2[2], 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("ZPBASE2", 2, "ZDCA", Geometry::ZPAPOSITION[0] + zpSupportBase2Pos[0],
                             Geometry::ZPAPOSITION[1] + zpSupportBase2Pos[1], Geometry::ZPAPOSITION[2] + zpSupportBase2[2], 0, "ONLY");

  // ZPC Box (A and C sides)
  // Bottom walls
  TVirtualMC::GetMC()->Gsvolu("ZPBB", "BOX ", getMediumID(kAl), const_cast<double*>(zpSupportWallBottom), 3);
  TVirtualMC::GetMC()->Gspos("ZPBB", 1, "ZDCC", Geometry::ZPCPOSITION[0],
                             Geometry::ZPCPOSITION[1] - Geometry::ZPDIMENSION[1] - zpSupportWallBottom[1], Geometry::ZPCPOSITION[2] - zpSupportWallBottom[2], 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("ZPBB", 2, "ZDCA", Geometry::ZPAPOSITION[0],
                             Geometry::ZPAPOSITION[1] - Geometry::ZPDIMENSION[1] - zpSupportWallBottom[1], Geometry::ZPAPOSITION[2] + zpSupportWallBottom[2], 0, "ONLY");

  // Top walls
  TVirtualMC::GetMC()->Gsvolu("ZPBT", "BOX ", getMediumID(kAl), const_cast<double*>(zpSupportWallup), 3);
  TVirtualMC::GetMC()->Gspos("ZPBT", 1, "ZDCC", Geometry::ZPCPOSITION[0],
                             Geometry::ZPCPOSITION[1] + Geometry::ZPDIMENSION[1] + zpSupportWallup[1], Geometry::ZPCPOSITION[2] - zpSupportWallup[2], 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("ZPBT", 2, "ZDCA", Geometry::ZPAPOSITION[0],
                             Geometry::ZPAPOSITION[1] + Geometry::ZPDIMENSION[1] + zpSupportWallup[1], Geometry::ZPAPOSITION[2] + zpSupportWallup[2], 0, "ONLY");

  // Side walls
  TVirtualMC::GetMC()->Gsvolu("ZPBS", "BOX ", getMediumID(kAl), const_cast<double*>(zpSupportWallside), 3);
  TVirtualMC::GetMC()->Gspos("ZPBS", 1, "ZDCC", Geometry::ZPCPOSITION[0] + Geometry::ZPDIMENSION[0] + zpSupportWallside[0], Geometry::ZPCPOSITION[1] + 0.75, Geometry::ZPCPOSITION[2] - zpSupportWallside[2], 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("ZPBS", 2, "ZDCC", Geometry::ZPCPOSITION[0] - Geometry::ZPDIMENSION[0] - zpSupportWallside[0], Geometry::ZPCPOSITION[1] + 0.75, Geometry::ZPCPOSITION[2] - zpSupportWallside[2], 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("ZPBS", 3, "ZDCA", Geometry::ZPAPOSITION[0] + Geometry::ZPDIMENSION[0] + zpSupportWallside[0], Geometry::ZPAPOSITION[1] + 0.75, Geometry::ZPAPOSITION[2] + zpSupportWallside[2], 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("ZPBS", 4, "ZDCA", Geometry::ZPAPOSITION[0] - Geometry::ZPDIMENSION[0] - zpSupportWallside[0], Geometry::ZPAPOSITION[1] + 0.75, Geometry::ZPAPOSITION[2] + zpSupportWallside[2], 0, "ONLY");

  // -------------------------------------------------------------------------------
  // -> EM calorimeter (ZEM)
  Int_t irotzem1, irotzem2;
  double rangzem1[6] = { 0., 0., 90., 90., -90., 0. };
  double rangzem2[6] = { 180., 0., 90., 45. + 90., 90., 45. };
  TVirtualMC::GetMC()->Matrix(irotzem1, rangzem1[0], rangzem1[1], rangzem1[2], rangzem1[3], rangzem1[4], rangzem1[5]);
  TVirtualMC::GetMC()->Matrix(irotzem2, rangzem2[0], rangzem2[1], rangzem2[2], rangzem2[3], rangzem2[4], rangzem2[5]);

  double zemPbSlice[6] = { 0.15 * TMath::Sqrt(2), 3.5, 3.5, 45., 0., 0. };
  double zemVoidLayer[6] = { (20.62 / 20.) / 2., 3.5, 3.5, 45., 0., 0. };
  double zemSupportTable[3] = { 55. / 2., 1.5 / 2., 110. / 2. };
  double zemSupportBox[6] = { 10.5 / 2., 100. / 2., 95. / 2., 0.25 / 2., 2. / 2., 2. / 2. };
  double zemSupport1[3] = { 15. / 2, 3. / 2., 95. / 2. };             //support table
  double zemSupport2[3] = { 2. / 2, 5. / 2., 95. / 2. };              //support table heels (piedini)
  double zemSupport3[3] = { 3.5, 2. / 2., 20. / 2. };                 //screens around ZEM
  double zemSupport4[6] = { 20. / 2., 3.5, 1.5 / 2., 45., 0., 0. };   //detector box walls (side)
  double zemWallH[3] = { 10.5 / 2., /*bthickness[1]*/ 1., 95. / 2. }; //box walls
  double zemWallVfwd[3] = { 10.5 / 2., (100. - 2.) / 2., 0.2 };
  double zemWallVbkw[3] = { 10.5 / 2., (100. - 2.) / 2., 2. / 2. };
  double zemWallVside[3] = { 0.25 / 2., (100. - 2.) / 2., (95. - 2.) / 2. };

  TVirtualMC::GetMC()->Gsvolu("ZEM ", "PARA", getMediumID(kVoidNoField), const_cast<double*>(Geometry::ZEMDIMENSION), 6);
  TVirtualMC::GetMC()->Gsvolu("ZEMF", "TUBE", getMediumID(kSiO2pmc), const_cast<double*>(Geometry::ZEMFIBRE), 3); // Active material
  TVirtualMC::GetMC()->Gsdvn("ZETR", "ZEM ", Geometry::ZEMDIVISION[2], 1);                                        // Tranches

  TVirtualMC::GetMC()->Gsvolu("ZEL0", "PARA", getMediumID(kPb), const_cast<double*>(zemPbSlice), 6); // Lead slices
  TVirtualMC::GetMC()->Gsvolu("ZEL1", "PARA", getMediumID(kPb), const_cast<double*>(zemPbSlice), 6);
  TVirtualMC::GetMC()->Gsvolu("ZEL2", "PARA", getMediumID(kPb), const_cast<double*>(zemPbSlice), 6);

  // --- Position the lead slices in the tranche
  TVirtualMC::GetMC()->Gspos("ZEL0", 1, "ZETR", -zemVoidLayer[0] + zemPbSlice[0], 0., 0., 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("ZEL1", 1, "ZETR", zemPbSlice[0], 0., 0., 0, "ONLY");

  // --- Vacuum zone (to be filled with fibres)
  TVirtualMC::GetMC()->Gsvolu("ZEV0", "PARA", getMediumID(kVoidNoField), const_cast<double*>(zemVoidLayer), 6);
  TVirtualMC::GetMC()->Gsvolu("ZEV1", "PARA", getMediumID(kVoidNoField), const_cast<double*>(zemVoidLayer), 6);

  // --- Divide the vacuum slice into sticks along x axis
  TVirtualMC::GetMC()->Gsdvn("ZES0", "ZEV0", Geometry::ZEMDIVISION[0], 3);
  TVirtualMC::GetMC()->Gsdvn("ZES1", "ZEV1", Geometry::ZEMDIVISION[0], 3);

  // --- Positioning the fibers into the sticks
  TVirtualMC::GetMC()->Gspos("ZEMF", 1, "ZES0", 0., 0., 0., irotzem2, "ONLY");
  TVirtualMC::GetMC()->Gspos("ZEMF", 1, "ZES1", 0., 0., 0., irotzem2, "ONLY");

  // --- Positioning the vacuum slice into the tranche
  //Float_t displFib = fDimZEM[1]/fDivZEM[0];
  TVirtualMC::GetMC()->Gspos("ZEV0", 1, "ZETR", -zemVoidLayer[0], 0., 0., 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("ZEV1", 1, "ZETR", -zemVoidLayer[0] + zemPbSlice[0], 0., 0., 0, "ONLY");

  // --- Positioning the ZEM into the ZDC - rotation for 90 degrees
  // NB -> ZEM is positioned in cave volume
  TVirtualMC::GetMC()->Gspos("ZEM ", 1, "cave", -Geometry::ZEMPOSITION[0], Geometry::ZEMPOSITION[1], Geometry::ZEMPOSITION[2] + Geometry::ZEMDIMENSION[0], irotzem1, "ONLY");

  // Second EM ZDC (same side w.r.t. IP, just on the other side w.r.t. beam pipe)
  TVirtualMC::GetMC()->Gspos("ZEM ", 2, "cave", Geometry::ZEMPOSITION[0], Geometry::ZEMPOSITION[1], Geometry::ZEMPOSITION[2] + Geometry::ZEMDIMENSION[0], irotzem1, "ONLY");

  // --- Adding last slice at the end of the EM calorimeter
  Float_t zLastSlice = Geometry::ZEMPOSITION[2] + zemPbSlice[0] + 2 * Geometry::ZEMDIMENSION[0];
  TVirtualMC::GetMC()->Gspos("ZEL2", 1, "cave", Geometry::ZEMPOSITION[0], Geometry::ZEMPOSITION[1], zLastSlice, irotzem1, "ONLY");

  // -------------------------------------------------------------------------------
  // -> ZEM supports

  // Platform and supports
  Float_t ybox = Geometry::ZEMPOSITION[1] - Geometry::ZEMDIMENSION[1] - 2. * 2. * zemSupportBox[3 + 1] + zemSupportBox[1];
  Float_t zSupport = Geometry::ZEMPOSITION[2] - 3.5; //to take into account the titlted front face
  Float_t zbox = zSupport + zemSupportBox[2];

  // Bridge
  TVirtualMC::GetMC()->Gsvolu("ZESH", "BOX ", getMediumID(kAl), const_cast<double*>(zemSupport1), 3);
  Float_t ybridge = Geometry::ZEMPOSITION[1] - Geometry::ZEMDIMENSION[1] - 2. * 2. * zemSupportBox[3 + 1] - 5. - zemSupport1[1];
  TVirtualMC::GetMC()->Gspos("ZESH", 1, "cave", Geometry::ZEMPOSITION[0], ybridge, zbox, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("ZESH", 2, "cave", -Geometry::ZEMPOSITION[0], ybridge, zbox, 0, "ONLY");
  //
  TVirtualMC::GetMC()->Gsvolu("ZESV", "BOX ", getMediumID(kAl), const_cast<double*>(zemSupport2), 3);
  TVirtualMC::GetMC()->Gspos("ZESV", 1, "cave", Geometry::ZEMPOSITION[0] - zemSupportBox[0] + zemSupport2[0], ybox - zemSupportBox[1] - zemSupport2[1], zbox, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("ZESV", 2, "cave", Geometry::ZEMPOSITION[0] + zemSupportBox[0] - zemSupport2[0], ybox - zemSupportBox[1] - zemSupport2[1], zbox, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("ZESV", 3, "cave", -(Geometry::ZEMPOSITION[0] - zemSupportBox[0] + zemSupport2[0]), ybox - zemSupportBox[1] - zemSupport2[1], zbox, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("ZESV", 4, "cave", -(Geometry::ZEMPOSITION[0] + zemSupportBox[0] - zemSupport2[0]), ybox - zemSupportBox[1] - zemSupport2[1], zbox, 0, "ONLY");

  // Table
  TVirtualMC::GetMC()->Gsvolu("ZETA", "BOX ", getMediumID(kAl), const_cast<double*>(zemSupportTable), 3);
  Float_t ytable = ybridge - zemSupport1[1] - zemSupportTable[1];
  TVirtualMC::GetMC()->Gspos("ZETA", 1, "cave", 0.0, ytable, zbox, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("ZETA", 2, "cave", 0.0, ytable - 13. + 2. * zemSupportTable[1], zbox, 0, "ONLY");

  //Screens around ZEM
  TVirtualMC::GetMC()->Gsvolu("ZEFL", "BOX ", getMediumID(kAl), const_cast<double*>(zemSupport3), 3);
  TVirtualMC::GetMC()->Gspos("ZEFL", 1, "cave", Geometry::ZEMPOSITION[0], -Geometry::ZEMDIMENSION[1] - zemSupport3[1], zSupport + zemSupport3[2], 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("ZEFL", 2, "cave", -Geometry::ZEMPOSITION[0], -Geometry::ZEMDIMENSION[1] - zemSupport3[1], zSupport + zemSupport3[2], 0, "ONLY");

  TVirtualMC::GetMC()->Gsvolu("ZELA", "PARA", getMediumID(kAl), const_cast<double*>(zemSupport4), 6);
  TVirtualMC::GetMC()->Gspos("ZELA", 1, "cave", Geometry::ZEMPOSITION[0] - Geometry::ZEMDIMENSION[2] - zemSupport4[2], Geometry::ZEMPOSITION[1], Geometry::ZEMPOSITION[2] + zemSupport4[0], irotzem1, "ONLY");
  TVirtualMC::GetMC()->Gspos("ZELA", 2, "cave", Geometry::ZEMPOSITION[0] + Geometry::ZEMDIMENSION[2] + zemSupport4[2], Geometry::ZEMPOSITION[1], Geometry::ZEMPOSITION[2] + zemSupport4[0], irotzem1, "ONLY");
  TVirtualMC::GetMC()->Gspos("ZELA", 3, "cave", -(Geometry::ZEMPOSITION[0] - Geometry::ZEMDIMENSION[2] - zemSupport4[2]), Geometry::ZEMPOSITION[1], Geometry::ZEMPOSITION[2] + zemSupport4[0], irotzem1, "ONLY");
  TVirtualMC::GetMC()->Gspos("ZELA", 4, "cave", -(Geometry::ZEMPOSITION[0] + Geometry::ZEMDIMENSION[2] + zemSupport4[2]), Geometry::ZEMPOSITION[1], Geometry::ZEMPOSITION[2] + zemSupport4[0], irotzem1, "ONLY");

  // Containers for ZEM calorimeters
  TVirtualMC::GetMC()->Gsvolu("ZEW1", "BOX ", getMediumID(kAl), const_cast<double*>(zemWallH), 3);
  TVirtualMC::GetMC()->Gsvolu("ZEW2", "BOX ", getMediumID(kAl), const_cast<double*>(zemWallVfwd), 3);
  TVirtualMC::GetMC()->Gsvolu("ZEW3", "BOX ", getMediumID(kAl), const_cast<double*>(zemWallVbkw), 3);
  TVirtualMC::GetMC()->Gsvolu("ZEW4", "BOX ", getMediumID(kFe), const_cast<double*>(zemWallVside), 3);
  //
  Float_t yh1 = Geometry::ZEMPOSITION[1] - Geometry::ZEMDIMENSION[1] - 2 * zemSupport3[1] - zemWallH[1];
  Float_t zh1 = zSupport + zemWallH[2];
  TVirtualMC::GetMC()->Gspos("ZEW1", 1, "cave", Geometry::ZEMPOSITION[0], yh1, zh1, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("ZEW1", 2, "cave", Geometry::ZEMPOSITION[0], yh1 + 2 * zemSupportBox[1], zh1, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("ZEW1", 3, "cave", -Geometry::ZEMPOSITION[0], yh1, zh1, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("ZEW1", 4, "cave", -Geometry::ZEMPOSITION[0], yh1 + 2 * zemSupportBox[1], zh1, 0, "ONLY");
  //
  TVirtualMC::GetMC()->Gspos("ZEW2", 1, "cave", Geometry::ZEMPOSITION[0], yh1 + zemSupportBox[1], zSupport - zemWallVfwd[2], 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("ZEW3", 1, "cave", Geometry::ZEMPOSITION[0], yh1 + zemSupportBox[1], zSupport + 2 * zemWallH[2], 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("ZEW2", 2, "cave", -Geometry::ZEMPOSITION[0], yh1 + zemSupportBox[1], zSupport - zemWallVfwd[2], 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("ZEW3", 2, "cave", -Geometry::ZEMPOSITION[0], yh1 + zemSupportBox[1], zSupport + 2 * zemWallH[2], 0, "ONLY");
  //
  Float_t xl1 = Geometry::ZEMPOSITION[0] - Geometry::ZEMDIMENSION[2] - 2. * zemSupport4[2] - zemWallVside[0];
  Float_t xl2 = Geometry::ZEMPOSITION[0] + Geometry::ZEMDIMENSION[2] + 2. * zemSupport4[2] + zemWallVside[0];
  TVirtualMC::GetMC()->Gspos("ZEW4", 1, "cave", xl1, yh1 + zemSupportBox[1], zh1, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("ZEW4", 2, "cave", xl2, yh1 + zemSupportBox[1], zh1, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("ZEW4", 3, "cave", -xl1, yh1 + zemSupportBox[1], zh1, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("ZEW4", 4, "cave", -xl2, yh1 + zemSupportBox[1], zh1, 0, "ONLY");
}

//_____________________________________________________________________________
Bool_t Detector::calculateTableIndexes(int& ibeta, int& iangle, int& iradius)
{
  double x[3] = { 0., 0., 0. }, xDet[3] = { 0., 0., 0. }, p[3] = { 0., 0., 0. }, energy = 0.;
  fMC->TrackPosition(x[0], x[1], x[2]);
  fMC->TrackMomentum(p[0], p[1], p[2], energy);

  //particle velocity
  float ptot = TMath::Sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
  float beta = 0.;
  if (energy > 0.)
    beta = ptot / energy;
  if (beta >= 0.67) {
    if (beta <= 0.75)
      ibeta = 0;
    else if (beta > 0.75 && beta <= 0.85)
      ibeta = 1;
    else if (beta > 0.85 && beta <= 0.95)
      ibeta = 2;
    else if (beta > 0.95)
      ibeta = 3;
  } else
    return kFALSE;
  //track angle wrt fibre axis (||LHC axis)
  double umom[3] = { 0., 0., 0. }, udet[3] = { 0., 0., 0. };
  umom[0] = p[0] / ptot;
  umom[1] = p[1] / ptot;
  umom[2] = p[2] / ptot;
  fMC->Gmtod(umom, udet, 2);
  double angleRad = TMath::ACos(udet[2]);
  double angleDeg = angleRad * kRaddeg;
  if (angleDeg < 110.)
    iangle = int(0.5 + angleDeg / 2.);
  else
    return kFALSE;
  //radius from fibre axis
  fMC->Gmtod(x, xDet, 1);
  float radius = 0.;
  if (TMath::Abs(udet[0]) > 0) {
    float dcoeff = udet[1] / udet[0];
    radius = TMath::Abs((xDet[1] - dcoeff * xDet[0]) / TMath::Sqrt(dcoeff * dcoeff + 1.));
  } else
    radius = TMath::Abs(udet[0]);
  iradius = int(radius * 1000. + 1.);
  return kTRUE;
}

//_____________________________________________________________________________
void Detector::EndOfEvent()
{
  Reset();
}

//_____________________________________________________________________________
void Detector::FinishPrimary()
{
  // after each primary we should definitely reset
  mLastPrincipalTrackEntered = -1;
  resetHitIndices();
}

//_____________________________________________________________________________
void Detector::Register()
{
  // This will create a branch in the output tree called Hit, setting the last
  // parameter to kFALSE means that this collection will not be written to the file,
  // it will exist only during the simulation

  if (FairRootManager::Instance()) {
    FairRootManager::Instance()->RegisterAny(addNameTo("Hit").data(), mHits, kTRUE);
  }
}

//_____________________________________________________________________________
void Detector::Reset()
{
  mHits->clear();
  mLastPrincipalTrackEntered = -1;
  resetHitIndices();
}
