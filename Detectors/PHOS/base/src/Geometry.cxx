// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "PHOSBase/Geometry.h"
#include "FairLogger.h"
#include "TSystem.h"
#include "TFile.h"

using namespace o2::phos;

ClassImp(Geometry);

// module numbering:
//  start from module 0 (non-existing), 1 (half-module), 2 (bottom),... 4(highest)
// absId:
// start from 1 till 4*64*56. Numbering in each module starts at bottom left and first go in z direction:
//  56   112   3584
//  ...  ...    ...
//  1    57 ...3529
//  relid[3]: (module number[1...4], iphi[1...64], iz[1...56])
//
//  Then TRU channels go from 1 to 112 per branch, 2 branches per ddl
//  absId = getTotalNCells() + TRUabsId ;
//  relId for TRU
//  relid: [DDL id=0..13] [x in 2x2 system: 1..8] [z in 2x2 system 1..28] TODO: verify with real TRU data!!!

// these initialisations are needed for a singleton
Geometry* Geometry::sGeom = nullptr;

Geometry::Geometry(const std::string_view name) : mGeoName(name)
{
  std::string p = gSystem->Getenv("O2_ROOT");
  p += "/share/Detectors/PHOS/files/alignment.root";
  TFile fin(p.data());

  //try reading rotation mathices
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
bool Geometry::truAbsToRelNumbering(short truId, char* relid)
{
  //convert trigger cell Id to
  truId--;
  relid[0] = truId / 224; //2*112 channels // DDL id
  truId = truId % 224;
  relid[1] = 1 + truId % 8; // x index in TRU internal 2x2 coordinate system
  relid[2] = 1 + truId / 8; // z index in TRU internal 2x2 coordinate system
  return true;
}
short Geometry::truRelToAbsNumbering(const char* relId)
{
  return relId[0] * 224 +    // the offset of PHOS modules
         relId[1] - 1 +      // the offset along phi
         (relId[2] - 1) * 8; // the offset along z
}
bool Geometry::truRelId2RelId(const char* truRelId, char* relId)
{
  relId[0] = 1 + (truRelId[0] + 2) / 4;
  relId[1] = ((truRelId[0] + 2) % 4) * 16 + truRelId[1] * 2 - 1;
  relId[2] = truRelId[2] * 2 - 1;
  return true;
}
short Geometry::relPosToTruId(char mod, float x, float z, short& ddl)
{
  //tranform local cluster coordinates to truId
  char relid[3] = {mod, static_cast<char>(ceil(x / CELLSTEP + 32.5)), static_cast<char>(ceil(z / CELLSTEP + 28.5))};
  ddl = (mod - 1) * 4 + relid[1] / 16 - 2;
  char truid[3] = {static_cast<char>(ddl), static_cast<char>(1 + ((relid[1] - 1) % 16) / 2), static_cast<char>(1 + (relid[2] - 1) / 2)};
  return truRelToAbsNumbering(truid);
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

    if ((coldiff <= 1) && (rowdiff <= 1)) { // At least common vertex
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
//local position to absId
void Geometry::relPosToAbsId(char module, float x, float z, short& absId)
{
  char relid[3] = {module, static_cast<char>(ceil(x / CELLSTEP + 32.5)), static_cast<char>(ceil(z / CELLSTEP + 28.5))};
  relToAbsNumbering(relid, absId);
}
void Geometry::relPosToRelId(short module, float x, float z, char* relId)
{
  relId[0] = module;
  relId[1] = static_cast<char>(ceil(x / CELLSTEP + 32.5));
  relId[2] = static_cast<char>(ceil(z / CELLSTEP + 28.5));
}

// convert local position in module to global position in ALICE
void Geometry::local2Global(char module, float x, float z, TVector3& globaPos) const
{
  // constexpr float shiftY=-10.76; Run2
  constexpr float shiftY = -1.26; //Depth-optimized
  Double_t posL[3] = {x, z, shiftY};
  Double_t posG[3];
  mPHOS[module].LocalToMaster(posL, posG);
  globaPos.SetXYZ(posG[0], posG[1], posG[2]);
}
