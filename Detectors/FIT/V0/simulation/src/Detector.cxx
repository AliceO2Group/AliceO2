// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "TGeoManager.h" // for TGeoManager
#include "TMath.h"
#include "TGraph.h"
#include "TString.h"
#include "TSystem.h"
#include "TVirtualMC.h"
#include "TVector3.h"
#include "TLorentzVector.h"

#include "FairRootManager.h" // for FairRootManager
#include "FairLogger.h"
#include "FairVolume.h"

#include "FairRootManager.h"
#include "FairVolume.h"

#include <sstream>
#include "V0Simulation/Detector.h"
#include "V0Base/Geometry.h"
#include "SimulationDataFormat/Stack.h"

using namespace o2::v0;
using o2::v0::Geometry;

ClassImp(Detector);

Detector::Detector()
  : o2::Base::DetImpl<Detector>("V0", kTRUE),
    mHits(o2::utils::createSimVector<o2::v0::Hit>()),
    mGeometry(nullptr),
    mTrackData(){
  // Empty
}

Detector::~Detector() {
  if (mHits) {
    o2::utils::freeSimVector(mHits); // delete mHits;
  }
  if (mGeometry){
    delete mGeometry;
  }
}

Detector::Detector(Bool_t isActive)
  : o2::Base::DetImpl<Detector> ("V0", isActive),
    mHits(o2::utils::createSimVector<o2::v0::Hit>()),
    mGeometry(nullptr),
    mTrackData(){
  // Empty
}

void Detector::InitializeO2Detector()
{
  LOG(INFO) << "Initializing FIT V0 geometry\n";

  TGeoVolume* volSensitive = gGeoManager->GetVolume("cell");
  if (!volSensitive) {
    LOG(FATAL) << "Can't find FIT V0 sensitive volume: cell";
  }
  else {
    AddSensitiveVolume(volSensitive);
    LOG(INFO) << "FIT-V0: Sensitive volume: " << volSensitive->GetName() << "   " << volSensitive->GetNumber();
    // TODO: Code from MFT
//    if (!mftGeom->getSensorVolumeID()) {
//      mftGeom->setSensorVolumeID(vol->GetNumber());
//    }
//    else if (mftGeom->getSensorVolumeID() != vol->GetNumber()) {
//      LOG(FATAL) << "CreateSensors: different Sensor volume ID !!!!";
//    }
  }
}

// TODO: Check if it works and remove some fields if same in MFT base as in T0
Bool_t Detector::ProcessHits(FairVolume* v){
  // This method is called from the MC stepping

  // Track only charged particles and photons
  bool isPhotonTrack = false;
  Int_t particleId = fMC->TrackPid();
  if (particleId == 50000050){ // If particle is photon
    isPhotonTrack = true;
  }
  if (!(isPhotonTrack || fMC->TrackCharge())) {
    return kFALSE;
  }

  // TODO: Uncomment or change the approach after geometry is ready
//  Geometry* v0Geo = Geometry::instance();
//  Int_t copy;
  // Check if hit is into a FIT-V0 sensitive volume
//  if (fMC->CurrentVolID(copy) != v0Geo->getSensorVolumeID())
//    return kFALSE;

  // Get unique ID of the cell
  Int_t cellId = -1;
  fMC->CurrentVolOffID(1, cellId);

  // Check track status to define when hit is started and when it is stopped
  bool startHit = false, stopHit = false;
  if ((fMC->IsTrackEntering()) || (fMC->IsTrackInside() && !mTrackData.mHitStarted)) {
    startHit = true;
  } else if ((fMC->IsTrackExiting() || fMC->IsTrackOut() || fMC->IsTrackStop())) {
    stopHit = true;
  }

  // Track is entering or created in the volume
  if (startHit) {
    mTrackData.mEnergyLoss = 0.;
    fMC->TrackMomentum(mTrackData.mMomentumStart);
    fMC->TrackPosition(mTrackData.mPositionStart);
    mTrackData.mHitStarted = true;
  }
  // Track is exiting or stopped within the volume
  if (stopHit) {
    TLorentzVector positionStop;
    fMC->TrackPosition(positionStop);
    Int_t trackID = fMC->GetStack()->GetCurrentTrackNumber();

    // TODO: compare this base with methods used by T0 (3 lines below)
    float etot = fMC->Etot();
    float eDep = fMC->Edep();
    addHit(trackID, cellId, particleId,
        mTrackData.mPositionStart.Vect(), positionStop.Vect(),
        mTrackData.mMomentumStart.Vect(), mTrackData.mMomentumStart.E(),
        positionStop.T(), mTrackData.mEnergyLoss, etot, eDep);
  }
  return kTRUE;
}

o2::v0::Hit* Detector::addHit(Int_t trackId, Int_t cellId, Int_t particleId,
    TVector3 startPos, TVector3 endPos,
    TVector3 startMom, double startE,
    double endTime, double eLoss, float eTot, float eDep){

  mHits->emplace_back(trackId, cellId, startPos, endPos, startMom, startE, endTime, eLoss, eTot, eDep);
  auto stack = (o2::Data::Stack*)fMC->GetStack();
  stack->addHit(GetDetId());
  return &(mHits->back());
}

// TODO: -> verify Todos inside the function
void Detector::createMaterials()
{
  // Air mixture
  const Int_t nAir = 4;
  Float_t aAir[nAir] = { 12.0107,  14.0067,  15.9994,  39.948 };
  Float_t zAir[nAir] = { 6,        7,        8,        18 };
  Float_t wAir[nAir] = { 0.000124, 0.755267, 0.231781, 0.012827 };
  Float_t dAir = 0.00120479;

  // Scintillator mixture; TODO: Looks very rough, improve these numbers?
  const Int_t nScint = 2;
  Float_t aScint[nScint] = { 1,     12.01};
  Float_t zScint[nScint] = { 1,     6,   };
  Float_t wScint[nScint] = { 0.016, 0.984};
  Float_t dScint = 1.023;

  // Aluminum
//  Float_t aAlu = 26.98;
//  Float_t zAlu = 13.;
//  Float_t dAlu = 2.70;     // density [gr/cm^3]
//  Float_t radAlu = 8.897;  // rad len [cm]
//  Float_t absAlu = 39.70;  // abs len [cm]


  Int_t matId = 0;            // tmp material id number
  const Int_t unsens = 0, sens = 1; // sensitive or unsensitive medium

  // TODO: After the simulation is running cross run for both sets of numbers and verify if they matter to us -> choose faster solution
  Float_t tmaxfd = -10.0; // in t0: 10   // max deflection angle due to magnetic field in one step
  Float_t stemax = 0.001; // in t0: 0.1  // max step allowed [cm]
  Float_t deemax = -0.2;  // in t0: 1.0  // maximum fractional energy loss in one step 0<deemax<=1
  Float_t epsil = 0.001;  // in t0: 0.03 // tracking precision [cm]
  Float_t stmin = -0.001; // in t0: 0.03 // minimum step due to continuous processes [cm] (negative value: choose it automatically)

  Int_t fieldType;
  Float_t maxField;
  o2::Base::Detector::initFieldTrackingParams(fieldType, maxField);
  LOG(DEBUG) << "Detector::createMaterials >>>>> fieldType " << fieldType << " maxField " << maxField;

  // TODO: Comment out two lines below once tested that the above function assigns field type and max correctly
  fieldType = 2;
  maxField = 10.;

  o2::Base::Detector::Mixture(++matId, "Air$", aAir, zAir, dAir, nAir, wAir);
  o2::Base::Detector::Medium(Air, "Air$", matId, unsens, fieldType, maxField, tmaxfd, stemax, deemax, epsil, stmin);

  o2::Base::Detector::Mixture(++matId, "Scintillator", aScint, zScint, dScint, nScint, wScint);
  o2::Base::Detector::Medium(Scintillator, "Scintillator$", matId, unsens, fieldType, maxField, tmaxfd, stemax, deemax, epsil, stmin);

//  o2::Base::Detector::Material(++matId, "Alu$", aAlu, zAlu, dAlu, radAlu, absAlu);
//  o2::Base::Detector::Medium(Alu, "Alu$", matId, unsens, fieldType, maxField, tmaxfd, stemax, deemax, epsil, stmin);

  LOG(DEBUG) << "Detector::createMaterials -----> matId = " << matId;
}

void Detector::ConstructGeometry()
{
  LOG(DEBUG) << "Creating FIT V0 geometry\n";
  createMaterials();
  mGeometry = new Geometry(Geometry::eOnlySensitive);
}

void Detector::Register()
{
  // This will create a branch in the output tree called Hit, setting the last
  // parameter to kFALSE means that this collection will not be written to the file,
  // it will exist only during the simulation

  if (FairRootManager::Instance()) {
    FairRootManager::Instance()->RegisterAny(addNameTo("Hit").data(), mHits, kTRUE);
  }
}

void Detector::EndOfEvent()
{
  Reset();
}

void Detector::Reset()
{
  if (!o2::utils::ShmManager::Instance().isOperational()) {
    mHits->clear();
  }
}






/*void Detector::DefineOpticalProperties()
{
  // Path of the optical properties input file
  TString inputDir;
  const char* aliceO2env = std::getenv("O2_ROOT");
  if (aliceO2env)
    inputDir = aliceO2env;
  inputDir += "/share/Detectors/T0/files/";

  TString optPropPath = inputDir + "quartzOptProperties.txt";
  optPropPath = gSystem->ExpandPathName(optPropPath.Data()); // Expand $(ALICE_ROOT) into real system path

  if (ReadOptProperties(optPropPath.Data()) < 0) {
    // Error reading file
    LOG(ERROR) << "Could not read FIT optical properties" << FairLogger::endl;
    return;
  }
  Int_t nBins = mPhotonEnergyD.size();
  // set QE
  mPMTeff = new TGraph(nBins, &(mPhotonEnergyD[0]), &(mQuantumEfficiency[0]));

  // Prepare pointers for arrays with constant and hardcoded values (independent on wavelength)
  FillOtherOptProperties();

  // Quick conversion from vector<Double_t> to Double_t*: photonEnergyD -> &(photonEnergyD[0])
  TVirtualMC::GetMC()->SetCerenkov(getMediumID(kOpGlass), nBins, &(mPhotonEnergyD[0]), &(mAbsorptionLength[0]),
                                   &(mEfficAll[0]), &(mRefractionIndex[0]));
  // TVirtualMC::GetMC()->SetCerenkov (getMediumID(kOpGlassCathode), kNbins, aPckov, aAbsSiO2, effCathode, rindexSiO2);
  TVirtualMC::GetMC()->SetCerenkov(getMediumID(kOpGlassCathode), nBins, &(mPhotonEnergyD[0]), &(mAbsorptionLength[0]),
                                   &(mEfficAll[0]), &(mRefractionIndex[0]));

  // Define a border for radiator optical properties
  //TVirtualMC::GetMC()->DefineOpSurface("surfRd", kGlisur, kDielectric_metal, kPolished, 0.);
  TVirtualMC::GetMC()->DefineOpSurface("surfRd", kUnified, kDielectric_metal, kPolished, 0.);
  TVirtualMC::GetMC()->SetMaterialProperty("surfRd", "EFFICIENCY", nBins, &(mPhotonEnergyD[0]), &(mEfficMet[0]));
  TVirtualMC::GetMC()->SetMaterialProperty("surfRd", "REFLECTIVITY", nBins, &(mPhotonEnergyD[0]), &(mReflMet[0]));
}

void Detector::FillOtherOptProperties()
{
  // Set constant values to the other arrays
  for (Int_t i = 0; i < mPhotonEnergyD.size(); i++) {
    mEfficAll.push_back(1.);
    mRindexAir.push_back(1.);
    mAbsorAir.push_back(0.3);
    mRindexCathodeNext.push_back(0.);
    mAbsorbCathodeNext.push_back(0.);
    mEfficMet.push_back(0.);
    mReflMet.push_back(1.);
  }
}

//------------------------------------------------------------------------
Bool_t Detector::RegisterPhotoE(float energy)
{
  //  Float_t hc=197.326960*1.e6; //mev*nm
  float hc = 1.973 * 1.e-6; // gev*nm
  float lambda = hc / energy;
  float eff = mPMTeff->Eval(lambda);
  float p = gRandom->Rndm();

  if (p > eff)
    return kFALSE;

  return kTRUE;
}

Int_t Detector::ReadOptProperties(const std::string filePath)
{
  std::ifstream infile;
  infile.open(filePath.c_str());

  // Check if file is opened correctly
  if (infile.fail() == true) {
    // AliFatal(Form("Error opening ascii file: %s", filePath.c_str()));
    return -1;
  }

  std::string comment;             // dummy, used just to read 4 first lines and move the cursor to the 5th, otherwise unused
  if (!getline(infile, comment)) { // first comment line
    //         AliFatal(Form("Error opening ascii file (it is probably a folder!): %s", filePath.c_str()));
    return -2;
  }
  getline(infile, comment); // 2nd comment line

  // Get number of elements required for the array
  Int_t nLines;
  infile >> nLines;
  if (nLines < 0 || nLines > 1e4) {
    //   AliFatal(Form("Input arraySize out of range 0..1e4: %i. Check input file: %s", kNbins, filePath.c_str()));
    return -4;
  }

  getline(infile, comment); // finish 3rd line after the nEntries are read
  getline(infile, comment); // 4th comment line

  // read the main body of the file (table of values: energy, absorption length and refractive index)
  Int_t iLine = 0;
  std::string sLine;
  getline(infile, sLine);
  while (!infile.eof()) {
    if (iLine >= nLines) {
      //      AliFatal(Form("Line number: %i reaches range of declared arraySize: %i. Check input file: %s", iLine,
      //      kNbins, filePath.c_str()));
      return -5;
    }
    std::stringstream ssLine(sLine);
    // First column:
    Double_t energy;
    ssLine >> energy;
    energy *= 1e-9; // Convert eV -> GeV immediately
    mPhotonEnergyD.push_back(energy);
    // Second column:
    Double_t absorption;
    ssLine >> absorption;
    mAbsorptionLength.push_back(absorption);
    // Third column:
    Double_t refraction;
    ssLine >> refraction;
    mRefractionIndex.push_back(refraction);
    // Fourth column:
    Double_t efficiency;
    ssLine >> efficiency;
    mQuantumEfficiency.push_back(efficiency);
    if (!(ssLine.good() || ssLine.eof())) { // check if there were problems with numbers conversion
      //    AliFatal(Form("Error while reading line %i: %s", iLine, ssLine.str().c_str()));
      return -6;
    }
    getline(infile, sLine);
    iLine++;
  }
  if (iLine != mPhotonEnergyD.size()) {
    //    AliFatal(Form("Total number of lines %i is different than declared %i. Check input file: %s", iLine, kNbins,
    //    filePath.c_str()));
    return -7;
  }

  //  AliInfo(Form("Optical properties taken from the file: %s. Number of lines read: %i",filePath.c_str(),iLine));
  return 0;
}
*/

