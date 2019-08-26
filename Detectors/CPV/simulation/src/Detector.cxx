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
#include <map>

#include "TGeoManager.h"
#include "TGeoVolume.h"
#include "TVirtualMC.h"
#include "TGeoPhysicalNode.h"

#include "FairGeoNode.h"
#include "FairRootManager.h"
#include "FairVolume.h"

#include "CPVBase/Geometry.h"
#include "CPVBase/Hit.h"
#include "CPVSimulation/Detector.h"
#include "CPVSimulation/GeometryParams.h"
#include "CPVSimulation/CPVSimParams.h"

#include "DetectorsBase/GeometryManager.h"
#include "SimulationDataFormat/Stack.h"

#include <boost/algorithm/string/predicate.hpp>
#include <boost/range/irange.hpp>

using namespace o2::cpv;

ClassImp(Detector);

Detector::Detector(Bool_t active)
  : o2::base::DetImpl<Detector>("CPV", active),
    mGeom(nullptr),
    mHits(o2::utils::createSimVector<o2::cpv::Hit>())
{
}

Detector::Detector(const Detector& rhs)
  : o2::base::DetImpl<Detector>(rhs),
    mHits(o2::utils::createSimVector<o2::cpv::Hit>())
{
}

Detector::~Detector()
{
  o2::utils::freeSimVector(mHits);
}

void Detector::InitializeO2Detector()
{
  Reset();

  // Define sensitive volumes
  defineSensitiveVolumes();
}

void Detector::EndOfEvent()
{
  Reset();
}
void Detector::FinishEvent()
{
  // Sort Hits
  // Add duplicates if any and remove them
  // TODO:
  if (!mHits || mHits->size() == 0) {
    return;
  }

  auto first = mHits->begin();
  auto last = mHits->end();

  std::sort(first, last);

  first = mHits->begin();
  last = mHits->end();

  // this is copy of std::unique() method with addition: adding identical Hits
  auto itr = first;
  while (++first != last) {
    if (*itr == *first) {
      *itr += *first;
    } else {
      *(++itr) = *first;
    }
  }
  ++itr;

  mHits->erase(itr, mHits->end());

  //       std::ostream stream(nullptr);
  //       stream.rdbuf(std::cout.rdbuf()); // uses cout's buffer
  //      stream.rdbuf(LOG(DEBUG2));
  //      for (int i = 0; i < mHits->size(); i++) {
  //         mHits->at(i).PrintStream(stream);
  //       }
}
void Detector::Reset()
{
  if (!o2::utils::ShmManager::Instance().isOperational()) {
    mHits->clear();
  }
}

void Detector::Register() { FairRootManager::Instance()->RegisterAny(addNameTo("Hit").data(), mHits, kTRUE); }

Bool_t Detector::ProcessHits(FairVolume* v)
{

  // ------------------------------------------------------------------------
  // Digitize one CPV hit:
  // use exact 4-momentum p and position zxhit of the hit,
  // find the pad response around this hit and
  // put the amplitudes in the pads into array digits
  //
  // Author: Yuri Kharlov (after Serguei Sadovsky)
  // 2 October 2000
  // ported from AliRoot 2019
  // ------------------------------------------------------------------------

  //analyze only charged
  if (fMC->TrackCharge() == 0) {
    fMC->SetMaxStep(1.e10);
    return false;
  }

  if (!fMC->IsTrackEntering()) { //simulate once per track
    return false;
  }

  if (!mGeom) {
    mGeom = Geometry::GetInstance();
  }

  int moduleNumber;
  fMC->CurrentVolOffID(2, moduleNumber);

  double time = fMC->TrackTime() * 1.e+9; // time in ns?? To be consistent with PHOS/EMCAL

  float xyzm[3], xyzd[3];
  fMC->TrackPosition(xyzm[0], xyzm[1], xyzm[2]);

  fMC->Gmtod(xyzm, xyzd, 1); // transform coordinate from master to daughter system

  // Current momentum of the hit's track in the local ref. system
  float pm[3], pd[3], energy = 0.;
  fMC->TrackMomentum(pm[0], pm[1], pm[2], energy);
  fMC->Gmtod(pm, pd, 2); // transform 3-momentum from master to daughter system

  // Just a reminder on axes notation in the CPV module:
  // axis Z goes along the beam
  // axis X goes across the beam in the module plane
  // axis Y is a normal to the module plane showing from the IP

  auto& cpvparam = o2::cpv::CPVSimParams::Instance();
  // Digitize the current CPV hit:
  // find pad response
  float hitX = xyzd[0]; //hit coordinate in daugter frame
  float hitZ = xyzd[2]; // z-coordinate of track in daugter frame
  float pX = pd[0];
  float pZ = pd[2];
  float pNorm = -pd[1];
  float dZY = pZ / pNorm * cpvparam.mCPVGasThickness;
  float dXY = pX / pNorm * cpvparam.mCPVGasThickness;
  float rnor1 = 0., rnor2 = 0.;
  gRandom->Rannor(rnor1, rnor2);
  float eloss = cpvparam.mdEdx * (1 + cpvparam.mDetR * rnor1) *
                TMath::Sqrt((1. + (pow(dZY, 2) + pow(dXY, 2)) / pow(cpvparam.mCPVGasThickness, 2)));
  float zhit1 = hitZ + cpvparam.mPadSizeZ * cpvparam.mnCellZ / 2 - dZY / 2;
  float xhit1 = hitX + cpvparam.mPadSizeX * cpvparam.mnCellX / 2 - dXY / 2;
  float zhit2 = zhit1 + dZY;
  float xhit2 = xhit1 + dXY;

  int iwht1 = (int)(xhit1 / cpvparam.CellWr()); // wire (x) coordinate "in"
  int iwht2 = (int)(xhit2 / cpvparam.CellWr()); // wire (x) coordinate "out"

  int nIter;
  float zxe[3][5];
  if (iwht1 == iwht2) { // incline 1-wire hit
    nIter = 2;
    zxe[0][0] = (zhit1 + zhit2 - dZY * 0.57735) / 2;
    zxe[1][0] = (iwht1 + 0.5) * cpvparam.CellWr();
    zxe[2][0] = eloss / 2;
    zxe[0][1] = (zhit1 + zhit2 + dZY * 0.57735) / 2;
    zxe[1][1] = (iwht1 + 0.5) * cpvparam.CellWr();
    zxe[2][1] = eloss / 2;
  } else if (TMath::Abs(iwht1 - iwht2) != 1) { // incline 3-wire hit
    nIter = 3;
    int iwht3 = (iwht1 + iwht2) / 2;
    float xwht1 = (iwht1 + 0.5) * cpvparam.CellWr(); // wire 1
    float xwht2 = (iwht2 + 0.5) * cpvparam.CellWr(); // wire 2
    float xwht3 = (iwht3 + 0.5) * cpvparam.CellWr(); // wire 3
    float xwr13 = (xwht1 + xwht3) / 2;               // center 13
    float xwr23 = (xwht2 + xwht3) / 2;               // center 23
    float dxw1 = xhit1 - xwr13;
    float dxw2 = xhit2 - xwr23;
    float egm1 = TMath::Abs(dxw1) / (TMath::Abs(dxw1) + TMath::Abs(dxw2) + cpvparam.CellWr());
    float egm2 = TMath::Abs(dxw2) / (TMath::Abs(dxw1) + TMath::Abs(dxw2) + cpvparam.CellWr());
    float egm3 = cpvparam.CellWr() / (TMath::Abs(dxw1) + TMath::Abs(dxw2) + cpvparam.CellWr());
    zxe[0][0] = (dXY * (xwr13 - xwht1) / dXY + zhit1 + zhit1) / 2;
    zxe[1][0] = xwht1;
    zxe[2][0] = eloss * egm1;
    zxe[0][1] = (dXY * (xwr23 - xwht1) / dXY + zhit1 + zhit2) / 2;
    zxe[1][1] = xwht2;
    zxe[2][1] = eloss * egm2;
    zxe[0][2] = dXY * (xwht3 - xwht1) / dXY + zhit1;
    zxe[1][2] = xwht3;
    zxe[2][2] = eloss * egm3;
  } else { // incline 2-wire hit
    nIter = 2;
    float xwht1 = (iwht1 + 0.5) * cpvparam.CellWr();
    float xwht2 = (iwht2 + 0.5) * cpvparam.CellWr();
    float xwr12 = (xwht1 + xwht2) / 2;
    float dxw1 = xhit1 - xwr12;
    float dxw2 = xhit2 - xwr12;
    float egm1 = TMath::Abs(dxw1) / (TMath::Abs(dxw1) + TMath::Abs(dxw2));
    float egm2 = TMath::Abs(dxw2) / (TMath::Abs(dxw1) + TMath::Abs(dxw2));
    zxe[0][0] = (zhit1 + zhit2 - dZY * egm1) / 2;
    zxe[1][0] = xwht1;
    zxe[2][0] = eloss * egm1;
    zxe[0][1] = (zhit1 + zhit2 + dZY * egm2) / 2;
    zxe[1][1] = xwht2;
    zxe[2][1] = eloss * egm2;
  }

  // Finite size of ionization region

  int nz3 = (cpvparam.mNgamz + 1) / 2;
  int nx3 = (cpvparam.mNgamx + 1) / 2;

  TVirtualMCStack* stack = fMC->GetStack();
  const Int_t partID = stack->GetCurrentTrackNumber();

  for (int iter = 0; iter < nIter; iter++) {

    float zhit = zxe[0][iter];
    float xhit = zxe[1][iter];
    float qhit = zxe[2][iter];
    float zcell = zhit / cpvparam.mPadSizeZ;
    float xcell = xhit / cpvparam.mPadSizeX;
    if (zcell <= 0 || xcell <= 0 ||
        zcell >= cpvparam.mnCellZ || xcell >= cpvparam.mnCellX) {
      return true; //beyond CPV
    }
    int izcell = (int)zcell;
    int ixcell = (int)xcell;
    float zc = zcell - izcell - 0.5;
    float xc = xcell - ixcell - 0.5;
    for (int iz = 1; iz <= cpvparam.mNgamz; iz++) {
      int kzg = izcell + iz - nz3;
      if (kzg <= 0 || kzg > cpvparam.mnCellZ) {
        continue;
      }
      float zg = (float)(iz - nz3) - zc;
      for (int ix = 1; ix <= cpvparam.mNgamx; ix++) {
        int kxg = ixcell + ix - nx3;
        if (kxg <= 0 || kxg > cpvparam.mnCellX) {
          continue;
        }
        float xg = (float)(ix - nx3) - xc;

        // Now calculate pad response
        float qpad = PadResponseFunction(qhit, zg, xg);
        qpad += cpvparam.mNoise * rnor2;
        if (qpad < 0) {
          continue;
        }
        // Fill hit with pad response ID and amplitude
        // hist will be sorted and merged later if necessary
        int detID = mGeom->RelToAbsId(moduleNumber, kxg, kzg);
        AddHit(partID, detID, Point3D<float>(xyzm[0], xyzm[1], xyzm[2]), time, qpad);
      }
    }
  }

  return true;
}

double Detector::PadResponseFunction(float qhit, float zhit, float xhit)
{
  // ------------------------------------------------------------------------
  // Calculate the amplitude in one CPV pad using the
  // cumulative pad response function
  // Author: Yuri Kharlov (after Serguei Sadovski)
  // 3 October 2000
  // ------------------------------------------------------------------------

  auto& cpvparam = o2::cpv::CPVSimParams::Instance();
  double dz = cpvparam.mPadSizeZ / 2;
  double dx = cpvparam.mPadSizeX / 2;
  double z = zhit * cpvparam.mPadSizeZ;
  double x = xhit * cpvparam.mPadSizeX;
  double amplitude = qhit *
                     (CPVCumulPadResponse(z + dz, x + dx) - CPVCumulPadResponse(z + dz, x - dx) -
                      CPVCumulPadResponse(z - dz, x + dx) + CPVCumulPadResponse(z - dz, x - dx));
  return amplitude;
}

double Detector::CPVCumulPadResponse(double x, double y)
{
  // ------------------------------------------------------------------------
  // Cumulative pad response function
  // It includes several terms from the CF decomposition in electrostatics
  // Note: this cumulative function is wrong since omits some terms
  //       but the cell amplitude obtained with it is correct because
  //       these omitting terms cancel
  // Author: Yuri Kharlov (after Serguei Sadovski)
  // 3 October 2000
  // ------------------------------------------------------------------------

  auto& cpvparam = o2::cpv::CPVSimParams::Instance();
  double r2 = x * x + y * y;
  double xy = x * y;
  double cumulPRF = 0;
  for (Int_t i = 0; i <= 4; i++) {
    double b1 = (2 * i + 1) * cpvparam.mB;
    cumulPRF += TMath::Power(-1, i) * TMath::ATan(xy / (b1 * TMath::Sqrt(b1 * b1 + r2)));
  }
  cumulPRF *= cpvparam.mA / (2 * TMath::Pi());
  return cumulPRF;
}

void Detector::AddHit(int trackID, int detID, const Point3D<float>& pos, double time, double qdep)
{
  LOG(DEBUG4) << "Adding hit for track " << trackID << " in a pad " << detID << " with position (" << pos.X() << ", "
              << pos.Y() << ", " << pos.Z() << "), time" << time << ", qdep =" << qdep << std::endl;
  mHits->emplace_back(trackID, detID, pos, time, qdep);
  // register hit creation with MCStack
  static_cast<o2::data::Stack*>(fMC->GetStack())->addHit(GetDetId());
}

void Detector::ConstructGeometry()
{
  // Create geometry description of CPV depector for Geant simulations.

  using boost::algorithm::contains;
  LOG(DEBUG) << "Creating CPV geometry\n";

  cpv::GeometryParams* geomParams = cpv::GeometryParams::GetInstance("CPVRun3Params");

  if (!geomParams) {
    LOG(ERROR) << "ConstructGeometry: CPV Geometry class has not been set up.\n";
  }

  if (!fMC) {
    fMC = TVirtualMC::GetMC();
  }

  // Configure geometry So far we have only one: Run3
  {
    mActiveModule[0] = kFALSE;
    mActiveModule[1] = kTRUE;
    mActiveModule[2] = kTRUE;
    mActiveModule[3] = kTRUE;
    mActiveModule[4] = kFALSE;
    mActiveModule[5] = kFALSE;
  }

  // First create necessary materials
  CreateMaterials();

  // Create a CPV modules-containers which will be filled with the stuff later.
  float par[3], x, y, z;

  // The box containing all CPV filled with air
  par[0] = geomParams->GetCPVBoxSize(0) / 2.0;
  par[1] = geomParams->GetCPVBoxSize(1) / 2.0;
  par[2] = geomParams->GetCPVBoxSize(2) / 2.0;
  fMC->Gsvolu("CPV", "BOX ", getMediumID(ID_AIR), par, 3);

  // --- Position  CPV modules in ALICE setup ---
  int idrotm[5];
  int iXYZ, iAngle;
  char im[5];
  for (int iModule = 0; iModule < 5; iModule++) {
    if (!mActiveModule[iModule + 1]) {
      continue;
    }
    float angle[3][2] = {0};
    geomParams->GetModuleAngle(iModule, angle);
    Matrix(idrotm[iModule], angle[0][0], angle[0][1], angle[1][0], angle[1][1], angle[2][0], angle[2][1]);

    float pos[3] = {0};
    geomParams->GetModuleCenter(iModule, pos);

    fMC->Gspos("CPV", iModule + 1, "cave", pos[0], pos[1], pos[2], idrotm[iModule], "ONLY");
  }

  //start filling CPV moodules
  // Gassiplex board
  par[0] = geomParams->GetGassiplexChipSize(0) / 2.;
  par[1] = geomParams->GetGassiplexChipSize(1) / 2.;
  par[2] = geomParams->GetGassiplexChipSize(2) / 2.;
  fMC->Gsvolu("CPVG", "BOX ", ID_TEXTOLIT, par, 3);

  // Cu+Ni foil covers Gassiplex board
  par[1] = geomParams->GetCPVCuNiFoilThickness() / 2;
  fMC->Gsvolu("CPVC", "BOX ", ID_CU, par, 3);
  y = -(geomParams->GetGassiplexChipSize(1) / 2 - par[1]);
  fMC->Gspos("CPVC", 1, "CPVG", 0, y, 0, 0, "ONLY");

  // Position of the chip inside CPV
  float xStep = geomParams->GetCPVActiveSize(0) / (geomParams->GetNumberOfCPVChipsPhi() + 1);
  float zStep = geomParams->GetCPVActiveSize(1) / (geomParams->GetNumberOfCPVChipsZ() + 1);
  int copy = 0;
  y = geomParams->GetCPVFrameSize(1) / 2 - geomParams->GetFTPosition(0) +
      geomParams->GetCPVTextoliteThickness() / 2 + geomParams->GetGassiplexChipSize(1) / 2 + 0.1;
  for (int ix = 0; ix < geomParams->GetNumberOfCPVChipsPhi(); ix++) {
    x = xStep * (ix + 1) - geomParams->GetCPVActiveSize(0) / 2;
    for (int iz = 0; iz < geomParams->GetNumberOfCPVChipsZ(); iz++) {
      copy++;
      z = zStep * (iz + 1) - geomParams->GetCPVActiveSize(1) / 2;
      fMC->Gspos("CPVG", copy, "CPV", x, y, z, 0, "ONLY");
    }
  }

  // Foiled textolite (1 mm of textolite + 50 mkm of Cu + 6 mkm of Ni)
  par[0] = geomParams->GetCPVActiveSize(0) / 2;
  par[1] = geomParams->GetCPVTextoliteThickness() / 2;
  par[2] = geomParams->GetCPVActiveSize(1) / 2;
  fMC->Gsvolu("CPVF", "BOX ", ID_TEXTOLIT, par, 3);

  // Argon gas volume
  par[1] = (geomParams->GetFTPosition(2) - geomParams->GetFTPosition(1) - geomParams->GetCPVTextoliteThickness()) / 2;
  fMC->Gsvolu("CPVAr", "BOX ", ID_AR, par, 3);

  for (int i = 0; i < 4; i++) {
    y = geomParams->GetCPVFrameSize(1) / 2 - geomParams->GetFTPosition(i) + geomParams->GetCPVTextoliteThickness() / 2;
    fMC->Gspos("CPVF", i + 1, "CPV", 0, y, 0, 0, "ONLY");
    if (i == 1) {
      y -= (geomParams->GetFTPosition(2) - geomParams->GetFTPosition(1)) / 2;
      fMC->Gspos("CPVAr", 1, "CPV ", 0, y, 0, 0, "ONLY");
    }
  }

  // Dummy sensitive plane in the middle of argone gas volume
  par[1] = 0.001;
  fMC->Gsvolu("CPVQ", "BOX ", ID_AR, par, 3);
  fMC->Gspos("CPVQ", 1, "CPVAr", 0, 0, 0, 0, "ONLY");

  // Cu+Ni foil covers textolite
  par[1] = geomParams->GetCPVCuNiFoilThickness() / 2;
  fMC->Gsvolu("CPVP1", "BOX ", ID_CU, par, 3);
  y = geomParams->GetCPVTextoliteThickness() / 2 - par[1];
  fMC->Gspos("CPVP1", 1, "CPVF", 0, y, 0, 0, "ONLY");

  // Aluminum frame around CPV
  par[0] = geomParams->GetCPVFrameSize(0) / 2;
  par[1] = geomParams->GetCPVFrameSize(1) / 2;
  par[2] = geomParams->GetCPVBoxSize(2) / 2;
  fMC->Gsvolu("CPVF1", "BOX ", ID_AL, par, 3);

  par[0] = geomParams->GetCPVBoxSize(0) / 2 - geomParams->GetCPVFrameSize(0);
  par[1] = geomParams->GetCPVFrameSize(1) / 2;
  par[2] = geomParams->GetCPVFrameSize(2) / 2;
  fMC->Gsvolu("CPVF2", "BOX ", ID_AL, par, 3);

  for (int j = 0; j <= 1; j++) {
    x = TMath::Sign(1, 2 * j - 1) * (geomParams->GetCPVBoxSize(0) - geomParams->GetCPVFrameSize(0)) / 2;
    fMC->Gspos("CPVF1", j + 1, "CPV", x, 0, 0, 0, "ONLY");
    z = TMath::Sign(1, 2 * j - 1) * (geomParams->GetCPVBoxSize(2) - geomParams->GetCPVFrameSize(2)) / 2;
    fMC->Gspos("CPVF2", j + 1, "CPV", 0, 0, z, 0, "ONLY");
  }

  gGeoManager->CheckGeometry();
}
//-----------------------------------------
void Detector::CreateMaterials()
{
  // Definitions of materials to build CPV and associated tracking media.

  int isxfld = 2;
  float sxmgmx = 10.0;
  o2::base::Detector::initFieldTrackingParams(isxfld, sxmgmx);

  // --- Air ---
  float aAir[4] = {12.0107, 14.0067, 15.9994, 39.948};
  float zAir[4] = {6., 7., 8., 18.};
  float wAir[4] = {0.000124, 0.755267, 0.231781, 0.012827};
  float dAir = 1.20479E-3;

  Mixture(ID_AIR, "Air", aAir, zAir, dAir, 4, wAir);
  Medium(ID_AIR, "Air", ID_AIR, 0, isxfld, sxmgmx, 10.0, 1.0, 0.1, 0.1, 10.0, nullptr, 0);

  // The Textolit which makes up the box which contains the calorimeter module      -> idtmed[707]
  float aTX[4] = {16.0, 28.09, 12.011, 1.00794};
  float zTX[4] = {8.0, 14.0, 6.0, 1.0};
  float wTX[4] = {292.0, 68.0, 462.0, 736.0};
  float dTX = 1.75;

  Mixture(ID_TEXTOLIT, "Textolit", aTX, zTX, dTX, -4, wTX);
  Medium(ID_TEXTOLIT, "Textolit", ID_TEXTOLIT, 0, isxfld, sxmgmx, 10.0, 0.1, 0.1, 0.1, 0.1, nullptr, 0);

  // Copper                                                                         -> idtmed[710]
  Material(ID_CU, "Cupr", 63.546, 29, 8.96, 1.43, 14.8, nullptr, 0);
  Medium(ID_CU, "Cupr", ID_CU, 0, isxfld, sxmgmx, 10.0, 0.1, 0.1, 0.1, 0.0001, nullptr, 0);

  // The gas mixture: ArCo2                                                         -> idtmed[715]
  // Ar+CO2 Mixture (80% / 20%)
  // --- The gas mixture ---
  // Co2
  //  float aCO[2] = {12.0, 16.0} ;
  // float zCO[2] = {6.0, 8.0} ;
  //  float wCO[2] = {1.0, 2.0} ;
  float dCO = 0.001977; //Co2 density
  float dAr = 0.001782; //Argon density

  float arContent = 0.80; // Ar-content of the ArCO2-mixture
  float aArCO[3] = {39.948, 12.0, 16.0};
  float zArCO[3] = {18.0, 6.0, 8.0};
  float wArCO[3];
  wArCO[0] = arContent;
  wArCO[1] = (1 - arContent) * 1;
  wArCO[2] = (1 - arContent) * 2;
  float dArCO = arContent * dAr + (1 - arContent) * dCO;
  Mixture(ID_AR, "ArCo2", aArCO, zArCO, dArCO, -3, wArCO);
  Medium(ID_AR, "ArCo2", ID_AR, 1, isxfld, sxmgmx, 10.0, 0.1, 0.1, 0.1, 0.01, nullptr, 0);

  // Various Aluminium parts made of Al                                             -> idtmed[701]
  Material(ID_AL, "Al", 26.98, 13., 2.7, 8.9, 999., nullptr, 0);
  Medium(ID_AL, "Alparts", ID_AL, 0, isxfld, sxmgmx, 10.0, 0.1, 0.1, 0.001, 0.001, nullptr, 0);

  // --- Stainless steel (let it be pure iron) ---
  Material(ID_FE, "Steel", 55.845, 26, 7.87, 1.76, 0., nullptr, 0);
  Medium(ID_FE, "Steel", ID_FE, 0, isxfld, sxmgmx, 10.0, 0.1, 0.1, 0.1, 0.0001, nullptr, 0);

  // Fibergalss                                                                     -> getMediumID(ID_FIBERGLASS)
  //  Medium(ID_FIBERGLASS, "Fiberglass", ID_FIBERGLASS, 0, isxfld, sxmgmx, 10.0, 0.1, 0.1, 0.1, 0.1, nullptr, 0);

  // The scintillator of the CPV made of Polystyrene scintillator                   -> idtmed[700]
  //  Medium(ID_SC, "CPVscint", ID_SC, 1,  isxfld, sxmgmx, 10.0, 0.1, 0.1, 0.1, 0.1, 0, 0) ;
}

//-----------------------------------------
void Detector::defineSensitiveVolumes()
{
  if (fActive) {
    TGeoVolume* vsense = gGeoManager->GetVolume("CPVQ");
    if (vsense) {
      AddSensitiveVolume(vsense);
    } else {
      LOG(ERROR) << "CPV Sensitive volume CPVQ not found ... No hit creation!\n";
    }
  }
}
//-----------------------------------------
void Detector::addAlignableVolumes() const
{
  //
  // Create entries for alignable volumes associating the symbolic volume
  // name with the corresponding volume path.

  cpv::GeometryParams* geom = cpv::GeometryParams::GetInstance();

  // Alignable modules
  // Volume path /cave_1/CPV_<i> => symbolic name /CPV/Module<i>, <i>=1,2,3,4,5

  o2::detectors::DetID::ID idCPV = o2::detectors::DetID::CPV;

  TString physModulePath = "/cave_1/CPV_";

  TString symbModuleName = "CPV/Module";

  for (Int_t iModule = 1; iModule <= geom->GetNModules(); iModule++) {

    TString volPath(physModulePath);
    volPath += iModule;

    TString symName(symbModuleName);
    symName += iModule;

    int modUID = o2::base::GeometryManager::getSensID(idCPV, iModule - 1);

    LOG(DEBUG) << "--------------------------------------------"
               << "\n";
    LOG(DEBUG) << "Alignable object" << iModule << "\n";
    LOG(DEBUG) << "volPath=" << volPath << "\n";
    LOG(DEBUG) << "symName=" << symName << "\n";
    LOG(DEBUG) << "--------------------------------------------"
               << "\n";

    LOG(DEBUG) << "Check for alignable entry: " << symName;

    if (!gGeoManager->SetAlignableEntry(symName.Data(), volPath.Data(), modUID)) {
      LOG(ERROR) << "Alignable entry " << symName << " NOT set";
    }
    LOG(DEBUG) << "Alignable entry " << symName << " set";

    // Create the Tracking to Local transformation matrix for PHOS modules
    TGeoPNEntry* alignableEntry = gGeoManager->GetAlignableEntryByUID(modUID);
    LOG(DEBUG) << "Got TGeoPNEntry " << alignableEntry;

    if (alignableEntry) {
      Float_t angle = geom->GetCPVAngle(iModule);
      TGeoHMatrix* globMatrix = alignableEntry->GetGlobalOrig();

      TGeoHMatrix* matTtoL = new TGeoHMatrix;
      matTtoL->RotateZ(270. + angle);
      const TGeoHMatrix& globmatrixi = globMatrix->Inverse();
      matTtoL->MultiplyLeft(&globmatrixi);
      alignableEntry->SetMatrix(matTtoL);
    }
  }
}
