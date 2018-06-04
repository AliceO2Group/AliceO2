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

using namespace o2::phos;

ClassImp(Geometry);

// these initialisations are needed for a singleton
Geometry* Geometry::sGeom = nullptr;

Geometry::Geometry(const std::string_view name) : mGeoName(name) {}

// static Geometry* Geometry::GetInstance(const std::string_view name)
// {
//     if(sGeom){
//       if(sGeom->GetName()==name){
//         return sGeom;
//       }
//       else{
//         delete sGeom ;
//       }
//     }
//     sGeom = new Geometry(name) ;
//     return sGeom;
// }

Int_t Geometry::RelToAbsId(Int_t moduleNumber, Int_t strip, Int_t cell) const
{
  // calculates absolute cell Id from moduleNumber, strip (number) and cell (number)
  // PHOS layout parameters:
  const Int_t nStrpZ = 28;                 // Number of strips along z-axis
  const Int_t nCrystalsInModule = 56 * 64; // Total number of crystals in module
  const Int_t nCellsXInStrip = 8;          // Number of crystals in strip unit along x-axis
  const Int_t nZ = 56;                     // nStripZ * nCellsZInStrip

  Int_t row = nStrpZ - (strip - 1) % nStrpZ;
  Int_t col = (Int_t)std::ceil((Double_t)strip / (nStrpZ)) - 1;

  return (moduleNumber - 1) * nCrystalsInModule + row * 2 + (col * nCellsXInStrip + (cell - 1) / 2) * nZ -
         (cell & 1 ? 1 : 0);
}

Bool_t Geometry::AbsToRelNumbering(Int_t absId, Int_t* relid) const
{
  // Converts the absolute numbering into the following array
  //  relid[0] = PHOS Module number 1:fNModules
  //  relid[1] = Row number inside a PHOS module (Z coordinate)
  //  relid[2] = Column number inside a PHOS module (Phi coordinate)
  const Int_t nZ = 56;   // nStripZ * nCellsZInStrip
  const Int_t nPhi = 64; // nStripZ * nCellsZInStrip

  Int_t phosmodulenumber = (absId - 1) / (nZ * nPhi);

  relid[0] = phosmodulenumber + 1;
  absId -= phosmodulenumber * nPhi * nZ;
  relid[1] = 1 + (absId - 1) / nZ;
  relid[2] = absId - (relid[1] - 1) * nZ;

  return true;
}
Int_t Geometry::AbsIdToModule(Int_t absId)
{
  const Int_t nZ = 56;
  const Int_t nPhi = 64;

  return 1 + (absId - 1) / (nZ * nPhi);
}

int Geometry::AreNeighbours(Int_t absId1, Int_t absId2) const
{

  // Gives the neighbourness of two digits = 0 are not neighbour but continue searching
  //                                       = 1 are neighbour
  //                                       = 2 are not neighbour but do not continue searching
  //                                       =-1 are not neighbour, continue searching, but do not look before d2 next
  //                                       time
  // neighbours are defined as digits having at least a common vertex
  // The order of d1 and d2 is important: first (d1) should be a digit already in a cluster
  //                                      which is compared to a digit (d2)  not yet in a cluster

  Int_t relid1[3];
  AbsToRelNumbering(absId1, relid1);

  Int_t relid2[3];
  AbsToRelNumbering(absId2, relid2);

  if (relid1[0] == relid2[0]) { // inside the same PHOS module
    Int_t rowdiff = TMath::Abs(relid1[1] - relid2[1]);
    Int_t coldiff = TMath::Abs(relid1[2] - relid2[2]);

    if ((coldiff <= 1) && (rowdiff <= 1)) { // At least common vertex
      return 1;
    } else {
      if ((relid2[1] > relid1[1]) && (relid2[2] > relid1[2] + 1))
        return 2; //  Difference in row numbers is too large to look further
    }
    return 0;

  } else {
    if (relid1[0] > relid2[0]) // we switched to the next module
      return -1;
    return 2;
  }
  return 0;
}
void Geometry::AbsIdToRelPosInModule(int absId, double& x, double& z) const
{

  const double cellStep = 2.25;

  Int_t relid[3];
  AbsToRelNumbering(absId, relid);

  x = (relid[1] - 28 - 0.5) * cellStep;
  z = (relid[2] - 32 - 0.5) * cellStep;
}
