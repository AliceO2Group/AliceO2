// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "TClonesArray.h"
#include "TGeoManager.h" // for TGeoManager
#include "TLorentzVector.h"
#include "TMath.h"
#include "TGraph.h"
#include "TString.h"
#include "TSystem.h"
#include "TVirtualMC.h"

#include "FairRootManager.h" // for FairRootManager
#include "FairLogger.h"
#include "FairVolume.h"

#include "FairRootManager.h"
#include "FairVolume.h"

#include <sstream>
#include "FITBase/Geometry.h"
#include "FITSimulation/Detector.h"

using namespace o2::fit;
using o2::fit::Geometry;

ClassImp(Detector);

Detector::Detector(Bool_t Active)
  : o2::Base::DetImpl<Detector>("FIT", Active), mIdSens1(0), mPMTeff(nullptr), mHits(new std::vector<o2::fit::HitType>)
{
  //  Geometry *geo  = GetGeometry() ;

  //  TString gn(geo->GetName());
}

void Detector::Initialize() { o2::Base::Detector::Initialize(); }

void Detector::ConstructGeometry()
{
  LOG(DEBUG) << "Creating FIT geometry\n";
  CreateMaterials();

  Float_t zdetA = 333;
  Float_t zdetC = 82;

  Int_t idrotm[999];
  Double_t x, y, z;
  Float_t pstartC[3] = { 20., 20, 5 };
  Float_t pstartA[3] = { 20, 20, 5 };
  Float_t pinstart[3] = { 2.95, 2.95, 4.34 };
  Float_t pmcp[3] = { 2.949, 2.949, 2.8 }; // MCP

  Matrix(idrotm[901], 90., 0., 90., 90., 180., 0.);

  // C side Concave Geometry

  Double_t crad = 82.; // define concave c-side radius here

  Double_t dP = pmcp[0]; // side length of mcp divided by 2

  // uniform angle between detector faces==
  Double_t btta = 2 * TMath::ATan(dP / crad);

  // get noncompensated translation data
  Double_t grdin[6] = { -3, -2, -1, 1, 2, 3 };
  Double_t gridpoints[6];
  for (Int_t i = 0; i < 6; i++) {
    gridpoints[i] = crad * TMath::Sin((1 - 1 / (2 * TMath::Abs(grdin[i]))) * grdin[i] * btta);
  }

  Double_t xi[Geometry::NCellsC] = { gridpoints[1], gridpoints[2], gridpoints[3], gridpoints[4], gridpoints[0],
                                     gridpoints[1], gridpoints[2], gridpoints[3], gridpoints[4], gridpoints[5],
                                     gridpoints[0], gridpoints[1], gridpoints[4], gridpoints[5], gridpoints[0],
                                     gridpoints[1], gridpoints[4], gridpoints[5], gridpoints[0], gridpoints[1],
                                     gridpoints[2], gridpoints[3], gridpoints[4], gridpoints[5], gridpoints[1],
                                     gridpoints[2], gridpoints[3], gridpoints[4] };
  Double_t yi[Geometry::NCellsC] = { gridpoints[5], gridpoints[5], gridpoints[5], gridpoints[5], gridpoints[4],
                                     gridpoints[4], gridpoints[4], gridpoints[4], gridpoints[4], gridpoints[4],
                                     gridpoints[3], gridpoints[3], gridpoints[3], gridpoints[3], gridpoints[2],
                                     gridpoints[2], gridpoints[2], gridpoints[2], gridpoints[1], gridpoints[1],
                                     gridpoints[1], gridpoints[1], gridpoints[1], gridpoints[1], gridpoints[0],
                                     gridpoints[0], gridpoints[0], gridpoints[0] };
  Double_t zi[Geometry::NCellsC];
  for (Int_t i = 0; i < Geometry::NCellsC; i++) {
    zi[i] = TMath::Sqrt(TMath::Power(crad, 2) - TMath::Power(xi[i], 2) - TMath::Power(yi[i], 2));
  }

  // get rotation data
  Double_t ac[Geometry::NCellsC], bc[Geometry::NCellsC], gc[Geometry::NCellsC];
  for (Int_t i = 0; i < Geometry::NCellsC; i++) {
    ac[i] = TMath::ATan(yi[i] / xi[i]) - TMath::Pi() / 2 + 2 * TMath::Pi();
    if (xi[i] < 0) {
      bc[i] = TMath::ACos(zi[i] / crad);
    } else {
      bc[i] = -1 * TMath::ACos(zi[i] / crad);
    }
  }
  Double_t xc2[Geometry::NCellsC], yc2[Geometry::NCellsC], zc2[Geometry::NCellsC];

  // compensation based on node position within individual detector geometries
  // determine compensated radius
  Double_t rcomp = crad + pstartC[2] / 2.0; //
  for (Int_t i = 0; i < Geometry::NCellsC; i++) {
    // Get compensated translation data
    xc2[i] = rcomp * TMath::Cos(ac[i] + TMath::Pi() / 2) * TMath::Sin(-1 * bc[i]);
    yc2[i] = rcomp * TMath::Sin(ac[i] + TMath::Pi() / 2) * TMath::Sin(-1 * bc[i]);
    zc2[i] = rcomp * TMath::Cos(bc[i]);

    // Convert angles to degrees
    ac[i] *= 180 / TMath::Pi();
    bc[i] *= 180 / TMath::Pi();
    gc[i] = -1 * ac[i];
  }
  // A Side

  Float_t xa[Geometry::NCellsA] = { -11.8, -5.9, 0,     5.9,  11.8, -11.8, -5.9, 0,     5.9,  11.8, -12.8, -6.9,
                                    6.9,   12.8, -11.8, -5.9, 0,    5.9,   11.8, -11.8, -5.9, 0,    5.9,   11.8 };

  Float_t ya[Geometry::NCellsA] = { 11.9, 11.9, 12.9, 11.9, 11.9, 6.0,  6.0,  7.0,   6.0,   6.0,   -0.1,  -0.1,
                                    0.1,  0.1,  -6.0, -6.0, -7.0, -6.0, -6.0, -11.9, -11.9, -12.9, -11.9, -11.9 };

  TGeoVolumeAssembly* stlinA = new TGeoVolumeAssembly("0STL"); // A side mother
  TGeoVolumeAssembly* stlinC = new TGeoVolumeAssembly("0STR"); // C side mother

  // FIT interior
  TVirtualMC::GetMC()->Gsvolu("0INS", "BOX", getMediumID(kAir), pinstart, 3);
  TGeoVolume* ins = gGeoManager->GetVolume("0INS");
  //
  TGeoTranslation* tr[Geometry::NCellsA + Geometry::NCellsC];
  TString nameTr;

  // A side Translations
  for (Int_t itr = 0; itr < Geometry::NCellsA; itr++) {
    nameTr = Form("0TR%i", itr + 1);
    z = -pstartA[2] + pinstart[2];
    tr[itr] = new TGeoTranslation(nameTr.Data(), xa[itr], ya[itr], z);
    printf(" itr %i A %f %f %f \n", itr, xa[itr], ya[itr], z + zdetA);
    tr[itr]->RegisterYourself();
    stlinA->AddNode(ins, itr, tr[itr]);
  }

  TGeoRotation* rot[Geometry::NCellsC];
  TString nameRot;

  TGeoCombiTrans* com[Geometry::NCellsC];
  TString nameCom;

  // C Side Transformations
  for (Int_t itr = Geometry::NCellsA; itr < Geometry::NCellsA + Geometry::NCellsC; itr++) {
    nameTr = Form("0TR%i", itr + 1);
    nameRot = Form("0Rot%i", itr + 1);
    int ic = itr - Geometry::NCellsA;
    // nameCom = Form("0Com%i",itr+1);
    rot[ic] = new TGeoRotation(nameRot.Data(), ac[ic], bc[ic], gc[ic]);
    rot[ic]->RegisterYourself();

    tr[itr] = new TGeoTranslation(nameTr.Data(), xc2[ic], yc2[ic], (zc2[ic] - 80.));
    tr[itr]->RegisterYourself();

    //   com[itr-Geometry::NCellsA] = new TGeoCombiTrans(tr[itr],rot[itr-Geometry::NCellsA]);
    com[ic] = new TGeoCombiTrans(xc2[ic], yc2[ic], (zc2[ic] - 80), rot[ic]);
    TGeoHMatrix hm = *com[ic];
    TGeoHMatrix* ph = new TGeoHMatrix(hm);
    stlinC->AddNode(ins, itr, ph);
  }

  TGeoVolume* alice = gGeoManager->GetVolume("cave");
  alice->AddNode(stlinA, 1, new TGeoTranslation(0, 0, zdetA));
  // alice->AddNode(stlinC,1,new TGeoTranslation(0,0, -zdetC ) );
  TGeoRotation* rotC = new TGeoRotation("rotC", 90., 0., 90., 90., 180., 0.);
  alice->AddNode(stlinC, 1, new TGeoCombiTrans(0., 0., -zdetC, rotC));

  // MCP + 4 x wrapped radiator + 4xphotocathod + MCP + Al top in front of radiators
  SetOneMCP(ins);

  // FIXME: we need to register the sensitive volumes with FairRoot
  TGeoVolume* v = gGeoManager->GetVolume("0REG");
  if (v == nullptr)
    printf("Sensitive volume 0REG not found!!!!!!!!");
  else {
    AddSensitiveVolume(v);
  }
}

//_________________________________________
void Detector::SetOneMCP(TGeoVolume* ins)
{
  Double_t x, y, z;
  Float_t pinstart[3] = { 2.95, 2.95, 4.34 };
  Float_t pmcp[3] = { 2.949, 2.949, 2.8 };     // MCP
  Float_t ptop[3] = { 1.324, 1.324, 1. };      // cherenkov radiator
  Float_t ptopref[3] = { 1.3241, 1.3241, 1. }; // cherenkov radiator wrapped with reflection
  Float_t preg[3] = { 1.324, 1.324, 0.005 };   // photcathode
  Double_t prfv[3] = { 0.0002, 1.323,
                       1. }; // vertical refracting layer bettwen radiators and bettwen radiator and not optical Air
  Double_t prfh[3] = { 1.323, 0.0002, 1. }; // horizontal refracting layer bettwen radiators a
  Double_t pal[3] = { 2.648, 2.648, 0.25 }; // 5mm Al top on th eeach radiator
  // Entry window (glass)
  TVirtualMC::GetMC()->Gsvolu("0TOP", "BOX", getMediumID(kOpGlass), ptop, 3); // glass radiator
  TGeoVolume* top = gGeoManager->GetVolume("0TOP");
  TVirtualMC::GetMC()->Gsvolu("0TRE", "BOX", getMediumID(kAir), ptopref, 3); // air: wrapped  radiator
  TGeoVolume* topref = gGeoManager->GetVolume("0TRE");
  TVirtualMC::GetMC()->Gsvolu("0REG", "BOX", getMediumID(kOpGlassCathode), preg, 3);
  TGeoVolume* cat = gGeoManager->GetVolume("0REG");
  TVirtualMC::GetMC()->Gsvolu("0MCP", "BOX", getMediumID(kGlass), pmcp, 3); // glass
  TGeoVolume* mcp = gGeoManager->GetVolume("0MCP");
  TVirtualMC::GetMC()->Gsvolu("0RFV", "BOX", getMediumID(kOpAir), prfv, 3); // optical Air vertical
  TGeoVolume* rfv = gGeoManager->GetVolume("0RFV");
  TVirtualMC::GetMC()->Gsvolu("0RFH", "BOX", getMediumID(kOpAir), prfh, 3); // optical Air horizontal
  TGeoVolume* rfh = gGeoManager->GetVolume("0RFH");

  TVirtualMC::GetMC()->Gsvolu("0PAL", "BOX", getMediumID(kAl), pal, 3); // 5mmAl top on the radiator
  TGeoVolume* altop = gGeoManager->GetVolume("0PAL");

  // wrapped radiator +  reflectiong layers
  Int_t ntops = 0, nrfvs = 0, nrfhs = 0;
  Float_t xin = 0, yin = 0, xinv = 0, yinv = 0, xinh = 0, yinh = 0;
  x = y = z = 0;
  topref->AddNode(top, 1, new TGeoTranslation(0, 0, 0));
  xinv = -ptop[0] - prfv[0];
  topref->AddNode(rfv, 1, new TGeoTranslation(xinv, 0, 0));
  printf(" GEOGEO  refv %f ,  0,0 \n", xinv);
  xinv = ptop[0] + prfv[0];
  topref->AddNode(rfv, 2, new TGeoTranslation(xinv, 0, 0));
  printf(" GEOGEO  refv %f ,  0,0 \n", xinv);
  yinv = -ptop[1] - prfh[1];
  topref->AddNode(rfh, 1, new TGeoTranslation(0, yinv, 0));
  printf(" GEOGEO  refh  ,  0, %f, 0 \n", yinv);
  yinv = ptop[1] + prfh[1];
  topref->AddNode(rfh, 2, new TGeoTranslation(0, yinv, 0));

  // container for radiator, cathod
  for (Int_t ix = 0; ix < 2; ix++) {
    xin = -pinstart[0] + 0.3 + (ix + 0.5) * 2 * ptopref[0];
    for (Int_t iy = 0; iy < 2; iy++) {
      z = -pinstart[2] + 2 * pal[2] + ptopref[2];
      yin = -pinstart[1] + 0.3 + (iy + 0.5) * 2 * ptopref[1];
      ntops++;
      ins->AddNode(topref, ntops, new TGeoTranslation(xin, yin, z));
      printf(" 0TOP  full %i x %f y %f z %f \n", ntops, xin, yin, z);
      z = -pinstart[2] + 2 * pal[2] + 2 * ptopref[2] + preg[2];
      ins->AddNode(cat, ntops, new TGeoTranslation(xin, yin, z));
      // cat->Print();
      printf(" GEOGEO  CATHOD x=%f , y= %f z= %f num  %i\n", xin, yin, z, ntops);
    }
  }
  // Al top
  z = -pinstart[2] + pal[2];
  ins->AddNode(altop, 1, new TGeoTranslation(0, 0, z));

  // MCP
  z = -pinstart[2] + 2 * pal[2] + 2 * ptopref[2] + 2 * preg[2] + pmcp[2];
  //   z=-pinstart[2] + 2*ptopref[2] + preg[2];
  ins->AddNode(mcp, 1, new TGeoTranslation(0, 0, z));
}

Bool_t Detector::ProcessHits(FairVolume* v)
{

  TLorentzVector position;
  static auto* refMC = TVirtualMC::GetMC();
  if (refMC->IsTrackEntering()) {
    refMC->TrackPosition(position);
    float x = position.X();
    float y = position.Y();
    float z = position.Z();

    float time = refMC->TrackTime() * 1.0e12;
    int trackID = refMC->GetStack()->GetCurrentTrackNumber();
    int detID = v->getMCid();
    float etot = refMC->Etot();
    int iPart = refMC->TrackPid();
    float enDep = refMC->Edep();
    if (iPart == 50000050) // If particles is photon then ...
    {
      if (RegisterPhotoE(etot)) {
        AddHit(x, y, z, time, enDep, trackID, detID);
      }
    }
    return kTRUE;
  }
  return kFALSE;
}

HitType* Detector::AddHit(float x, float y, float z, float time, float energy, Int_t trackId, Int_t detId)
{
  mHits->emplace_back(x, y, z, time, energy, trackId, detId);
  return &(mHits->back());
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

void Detector::Reset() { mHits->clear(); }
void Detector::CreateMaterials()
{
  Int_t isxfld = 2;     // magneticField->Integ();
  Float_t sxmgmx = 10.; // magneticField->Max();
  // FIXME: use o2::Base::Detector::initFieldTrack to init mag field params
  
  //   Float_t a,z,d,radl,absl,buf[1];
  // Int_t nbuf;
  // AIR

  Float_t aAir[4] = { 12.0107, 14.0067, 15.9994, 39.948 };
  Float_t zAir[4] = { 6., 7., 8., 18. };
  Float_t wAir[4] = { 0.000124, 0.755267, 0.231781, 0.012827 };
  Float_t dAir = 1.20479E-3;
  Float_t dAir1 = 1.20479E-11;
  // Radiator  glass SiO2
  Float_t aglass[2] = { 28.0855, 15.9994 };
  Float_t zglass[2] = { 14., 8. };
  Float_t wglass[2] = { 1., 2. };
  Float_t dglass = 2.65;
  // MCP glass SiO2
  Float_t dglass_mcp = 1.3;
  //*** Definition Of avaible FIT materials ***
  Mixture(1, "Vacuum$", aAir, zAir, dAir1, 4, wAir);
  Mixture(2, "Air$", aAir, zAir, dAir, 4, wAir);
  Mixture(4, "MCP glass   $", aglass, zglass, dglass_mcp, -2, wglass);
  Mixture(24, "Radiator Optical glass$", aglass, zglass, dglass, -2, wglass);
  Material(11, "Aliminium$", 26.98, 13.0, 2.7, 8.9, 999);

  Medium(1, "Air$", 2, 0, isxfld, sxmgmx, 10., .1, 1., .003, .003);
  Medium(3, "Vacuum$", 1, 0, isxfld, sxmgmx, 10., .01, .1, .003, .003);
  Medium(6, "Glass$", 4, 0, isxfld, sxmgmx, 10., .01, .1, .003, .003);

  Medium(7, "OpAir$", 2, 0, isxfld, sxmgmx, 10., .1, 1., .003, .003);

  Medium(15, "Aluminium$", 11, 0, isxfld, sxmgmx, 10., .01, 1., .003, .003);
  Medium(16, "OpticalGlass$", 24, 1, isxfld, sxmgmx, 10., .01, .1, .003, .003);
  Medium(19, "OpticalGlassCathode$", 24, 1, isxfld, sxmgmx, 10., .01, .1, .003, .003);
  Medium(22, "SensAir$", 2, 1, isxfld, sxmgmx, 10., .1, 1., .003, .003);
}

//-------------------------------------------------------------------
void Detector::DefineOpticalProperties()
{
  // Path of the optical properties input file
  TString inputDir;
  const char* aliceO2env = std::getenv("O2_ROOT");
  if (aliceO2env)
    inputDir = aliceO2env;
  inputDir += "/share/Detectors/FIT/files/";

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
  TVirtualMC::GetMC()->DefineOpSurface("surfRd", kUnified /*kGlisur*/, kDielectric_metal, kPolished, 0.);
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

  std::string comment; // dummy, used just to read 4 first lines and move the cursor to the 5th, otherwise unused
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
