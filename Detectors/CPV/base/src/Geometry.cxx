// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "CPVBase/Geometry.h"

using namespace o2::cpv;

ClassImp(Geometry);

// these initialisations are needed for a singleton
Geometry* Geometry::sGeom = nullptr;

Geometry::Geometry(const std::string_view name) : mGeoName(name),
                                                  mNumberOfCPVPadsPhi(128),
                                                  mNumberOfCPVPadsZ(60),
                                                  mCPVPadSizePhi(1.13),
                                                  mCPVPadSizeZ(2.1093)
{
}

int Geometry::RelToAbsId(int moduleNumber, int iphi, int iz) const
{
  //converts module number, phi and z coordunates to absId
  return mNumberOfCPVPadsPhi * mNumberOfCPVPadsZ * (moduleNumber - 1) + mNumberOfCPVPadsZ * (iz - 1) + iphi;
}

bool Geometry::AbsToRelNumbering(int absId, int* relid) const
{
  // Converts the absolute numbering into the following array
  //  relid[0] = CPV Module number 1:fNModules
  //  relid[1] = Column number inside a CPV module (Phi coordinate)
  //  relid[2] = Row number inside a CPV module (Z coordinate)

  double nCPV = mNumberOfCPVPadsPhi * mNumberOfCPVPadsZ;
  relid[0] = (absId - 1) / nCPV + 1;
  absId -= (relid[0] - 1) * nCPV;
  relid[2] = absId / mNumberOfCPVPadsZ + 1;
  relid[1] = absId - (relid[2] - 1) * mNumberOfCPVPadsZ;

  return true;
}
int Geometry::AbsIdToModule(int absId)
{

  return (int)TMath::Ceil(absId / (mNumberOfCPVPadsPhi * mNumberOfCPVPadsZ));
}

int Geometry::AreNeighbours(int absId1, int absId2) const
{

  // Gives the neighbourness of two digits = 0 are not neighbour but continue searching
  //                                       = 1 are neighbour
  //                                       = 2 are not neighbour but do not continue searching
  //                                       =-1 are not neighbour, continue searching, but do not look before d2 next
  //                                       time
  // neighbours are defined as digits having at least a common vertex
  // The order of d1 and d2 is important: first (d1) should be a digit already in a cluster
  //                                      which is compared to a digit (d2)  not yet in a cluster

  int relid1[3];
  AbsToRelNumbering(absId1, relid1);

  int relid2[3];
  AbsToRelNumbering(absId2, relid2);

  if (relid1[0] == relid2[0]) { // inside the same CPV module
    int rowdiff = TMath::Abs(relid1[1] - relid2[1]);
    int coldiff = TMath::Abs(relid1[2] - relid2[2]);

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
void Geometry::AbsIdToRelPosInModule(int absId, double& x, double& z) const
{
  //Calculate from absId of a cell its position in module

  int relid[3];
  AbsToRelNumbering(absId, relid);

  x = (relid[1] - mNumberOfCPVPadsPhi / 2 - 0.5) * mCPVPadSizePhi;
  z = (relid[2] - mNumberOfCPVPadsPhi / 2 - 0.5) * mCPVPadSizeZ;
}
