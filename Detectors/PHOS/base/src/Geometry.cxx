// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "PHOSBase/Geometry.h"
#include <fairlogger/Logger.h>
#include "TSystem.h"
#include "TFile.h"

using namespace o2::phos;

ClassImp(Geometry);

// module numbering:
//  start from module 0 (non-existing), 1 (half-module), 2 (bottom),... 4(highest)
// absId:
// start from 1 till 4*64*56=14336. Numbering in each module starts at bottom left and first go in z direction:
//  56   112   3584
//  ...  ...    ...
//  1    57 ...3529
//  relid[3]: (module number[1...4], iphi[1...64], iz[1...56])
//
//  TRUabsId channels go from getTotalNCells()+1, 112 per branch, 2 branches per 14 ddl
//  Converting tru absId to relId one gets bottom left corner of 2x2 or 4x4 box

// these initialisations are needed for a singleton
Geometry* Geometry::sGeom = nullptr;

Geometry::Geometry(const std::string_view name) : mGeoName(name)
{
  std::string p = gSystem->Getenv("O2_ROOT");
  p += "/share/Detectors/PHOS/files/alignment.root";
  TFile fin(p.data());

  // try reading rotation mathices
  for (int m = 1; m < 5; m++) {
    mPHOS[m] = *static_cast<TGeoHMatrix*>(fin.Get(Form("Module%d", m)));
  }
  fin.Close();
}

short Geometry::relToAbsId(char moduleNumber, int strip, int cell)
{
  // calculates absolute cell Id from moduleNumber, strip (number) and cell (number)
  // PHOS layout parameters:
  const short nStrpZ = 28;                 // Number of strips along z-axis
  const short nCrystalsInModule = 56 * 64; // Total number of crystals in module
  const short nCellsXInStrip = 8;          // Number of crystals in strip unit along x-axis
  const short nZ = 56;                     // nStripZ * nCellsZInStrip

  short row = nStrpZ - (strip - 1) % nStrpZ;
  short col = (int)std::ceil((float)strip / (nStrpZ)) - 1;

  return (moduleNumber - 1) * nCrystalsInModule + row * 2 + (col * nCellsXInStrip + (cell - 1) / 2) * nZ -
         (cell & 1 ? 1 : 0);
}

bool Geometry::absToRelNumbering(short absId, char* relid)
{
  // Converts the absolute numbering into the following array
  //  relid[0] = PHOS Module number 1:fNModules
  //  relid[1] = Row number inside a PHOS module (Z coordinate)
  //  relid[2] = Column number inside a PHOS module (Phi coordinate)
  const short nZ = 56;   // nStripZ * nCellsZInStrip
  const short nPhi = 64; // nStripZ * nCellsZInStrip
  absId--;
  short phosmodulenumber = absId / (nZ * nPhi);

  relid[0] = phosmodulenumber + 1;
  absId -= phosmodulenumber * nPhi * nZ;
  relid[1] = 1 + absId / nZ;
  relid[2] = absId - (relid[1] - 1) * nZ + 1;

  return true;
}
bool Geometry::truAbsToRelNumbering(short truId, short trigType, char* relid)
{
  // convert trigger cell Id 1..224*14 to relId (same relId schema as for readout channels)
  // short trigType=0 for 2x2, short trigType=1 for 4x4 trigger
  //   Converting tru absId to relId one gets bottom left corner of 2x2 or 4x4 box
  truId -= getTotalNCells() + 1; // 1 to start from zero
  short ddl = truId / 224;       // 2*112 channels // DDL id
  relid[0] = 1 + (ddl + 2) / 4;
  truId = truId % 224;
  if (trigType == 1) { // 4x4 trigger
    if ((truId > 90 && truId < 112) || truId > 202) {
      LOG(error) << "Wrong TRU id channel " << truId << " should be <91";
      relid[0] = 0;
      relid[1] = 0;
      relid[2] = 0;
      return false;
    }
    relid[1] = 16 * ((2 + ddl) % 4) + 1 + 2 * (truId % 7); // x index
    if (truId < 112) {
      relid[2] = 53 - 2 * (truId / 7); // z index branch 0
    } else {
      truId -= 112;
      relid[2] = 25 - 2 * (truId / 7); // z index branch 1
    }
  } else {                                                 // 2x2 trigger
    relid[1] = 16 * ((2 + ddl) % 4) + 1 + 2 * (truId % 8); // x index
    if (truId < 112) {
      relid[2] = 55 - 2 * (truId / 8); // z index branch 0
    } else {
      truId -= 112;
      relid[2] = 27 - 2 * (truId / 8); // z index branch 1
    }
  }
  return true;
}
short Geometry::truRelToAbsNumbering(const char* relId, short trigType)
{
  // Convert position in PHOS module to TRU id for 4x4 or 2x2 tiles

  short absId = 0;
  short ddl = relId[0] * 4 + relId[1] / 16 - 6;
  if (trigType == 1) { // 4x4 trigger
    if (relId[2] > 28) {
      absId = ((53 - relId[2]) / 2) * 7;
    } else {
      absId = 112 + ((25 - relId[2]) / 2) * 7;
    }
    absId += (((relId[1] - 1) % 16) / 2) % 7;
    absId += ddl * 224;
    return getTotalNCells() + absId + 1;
  } else { // 2x2
    if (relId[2] > 28) {
      absId = ((55 - relId[2]) / 2) * 8;
    } else {
      absId = 112 + ((27 - relId[2]) / 2) * 8;
    }
    absId += (((relId[1] - 1) % 16) / 2) % 8;
    absId += ddl * 224;
    return getTotalNCells() + absId + 1;
  }
}
short Geometry::relPosToTruId(char mod, float x, float z, short trigType)
{
  // tranform local cluster coordinates to truId
  char relid[3] = {mod, static_cast<char>(ceil(x / CELLSTEP + 32.499)), static_cast<char>(ceil(z / CELLSTEP + 28.499))};
  return truRelToAbsNumbering(relid, trigType);
}

char Geometry::absIdToModule(short absId)
{
  const short nZ = 56;
  const short nPhi = 64;

  return 1 + (absId - 1) / (nZ * nPhi);
}

int Geometry::areNeighbours(short absId1, short absId2)
{

  // Gives the neighbourness of two digits = 0 are not neighbour but continue searching
  //                                       = 1 are neighbour
  //                                       = 2 are not neighbour but do not continue searching
  //                                       =-1 are not neighbour, continue searching, but do not look before d2 next
  //                                       time
  // neighbours are defined as digits having at least a common vertex
  // The order of d1 and d2 is important: first (d1) should be a digit already in a cluster
  //                                      which is compared to a digit (d2)  not yet in a cluster

  char relid1[3];
  absToRelNumbering(absId1, relid1);

  char relid2[3];
  absToRelNumbering(absId2, relid2);

  if (relid1[0] == relid2[0]) { // inside the same PHOS module
    char rowdiff = TMath::Abs(relid1[1] - relid2[1]);
    char coldiff = TMath::Abs(relid1[2] - relid2[2]);

    if (coldiff + rowdiff <= 1) { // Common side
      return 1;
    } else {
      if ((relid2[1] > relid1[1]) && (relid2[2] > relid1[2] + 1)) {
        return 2; //  Difference in row numbers is too large to look further
      }
    }
    return 0;

  } else {
    if (relid1[0] > relid2[0]) { // we switched to the next module
      return -1;
    }
    return 2;
  }
  return 0;
}
void Geometry::absIdToRelPosInModule(short absId, float& x, float& z)
{

  char relid[3];
  absToRelNumbering(absId, relid);

  x = (relid[1] - 32 - 0.5) * CELLSTEP;
  z = (relid[2] - 28 - 0.5) * CELLSTEP;
}
bool Geometry::relToAbsNumbering(const char* relId, short& absId)
{
  const short nZ = 56;   // nStripZ * nCellsZInStrip
  const short nPhi = 64; // nStripZ * nCellsZInStrip

  absId =
    (relId[0] - 1) * nPhi * nZ + // the offset of PHOS modules
    (relId[1] - 1) * nZ +        // the offset along phi
    relId[2];                    // the offset along z

  return true;
}
// local position to absId
void Geometry::relPosToAbsId(char module, float x, float z, short& absId)
{
  // adding 32.5 instead of 32.499 leads to (rare) rounding errors before ceil()
  char relid[3] = {module, static_cast<char>(ceil(x / CELLSTEP + 32.499)), static_cast<char>(ceil(z / CELLSTEP + 28.499))};
  relToAbsNumbering(relid, absId);
}
void Geometry::relPosToRelId(short module, float x, float z, char* relId)
{
  relId[0] = module;
  relId[1] = static_cast<char>(ceil(x / CELLSTEP + 32.499));
  relId[2] = static_cast<char>(ceil(z / CELLSTEP + 28.499));
}

// convert local position in module to global position in ALICE
void Geometry::local2Global(char module, float x, float z, TVector3& globaPos) const
{
  // constexpr float shiftY=-10.76; Run2
  constexpr float shiftY = -1.26; // Depth-optimized
  Double_t posL[3] = {x, z, shiftY};
  Double_t posG[3];
  mPHOS[module].LocalToMaster(posL, posG);
  globaPos.SetXYZ(posG[0], posG[1], posG[2]);
}

bool Geometry::impactOnPHOS(const TVector3& vtx, const TVector3& p,
                            short& module, float& z, float& x) const
{
  // calculates the impact coordinates on PHOS of a neutral particle
  // emitted in the vertex vtx with 3-momentum p
  constexpr float shiftY = -1.26;          // Depth-optimized
  constexpr float moduleXhalfSize = 72.16; // 18.04 / 2 * 8
  constexpr float moduleZhalfSize = 64.14; // 4.51 / 2 * 28

  for (short mod = 1; mod < 5; mod++) {
    // create vector from (0,0,0) to center of crystal surface of imod module
    double tmp[3] = {0., 0., shiftY};
    double posG[3] = {0., 0., 0.};
    mPHOS[mod].LocalToMaster(tmp, posG);
    TVector3 n(posG[0], posG[1], posG[2]);
    double direction = n.Dot(p);
    if (direction <= 0.) {
      continue; // momentum directed FROM module
    }
    double fr = (n.Mag2() - n.Dot(vtx)) / direction;
    // Calculate direction in module plane
    n -= vtx + fr * p;
    n *= -1.;
    if (TMath::Abs(n.Z()) < moduleZhalfSize && n.Pt() < moduleXhalfSize) {
      module = mod;
      z = n.Z();
      x = TMath::Sign(n.Pt(), n.X());
      // no need to return to local system since we calcilated distance from module center
      // and tilts can not be significant.
      return true;
    }
  }
  // Not in acceptance
  x = 0;
  z = 0;
  module = 0;
  return false;
}
