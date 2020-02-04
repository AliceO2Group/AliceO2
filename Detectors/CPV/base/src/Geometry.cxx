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

short Geometry::relToAbsId(char moduleNumber, int iphi, int iz)
{
  //converts module number, phi and z coordunates to absId
  return kNumberOfCPVPadsPhi * kNumberOfCPVPadsZ * (moduleNumber - 1) + kNumberOfCPVPadsZ * (iz - 1) + iphi;
}

bool Geometry::absToRelNumbering(short absId, short* relid)
{
  // Converts the absolute numbering into the following array
  //  relid[0] = CPV Module number 1:fNModules
  //  relid[1] = Column number inside a CPV module (Phi coordinate)
  //  relid[2] = Row number inside a CPV module (Z coordinate)

  short nCPV = kNumberOfCPVPadsPhi * kNumberOfCPVPadsZ;
  relid[0] = (absId - 1) / nCPV + 1;
  absId -= (relid[0] - 1) * nCPV;
  relid[2] = absId / kNumberOfCPVPadsZ + 1;
  relid[1] = absId - (relid[2] - 1) * kNumberOfCPVPadsZ;

  return true;
}
char Geometry::absIdToModule(short absId)
{

  return 1 + (absId - 1) / (kNumberOfCPVPadsPhi * kNumberOfCPVPadsZ);
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

  short relid1[3];
  absToRelNumbering(absId1, relid1);

  short relid2[3];
  absToRelNumbering(absId2, relid2);

  if (relid1[0] == relid2[0]) { // inside the same CPV module
    short rowdiff = TMath::Abs(relid1[1] - relid2[1]);
    short coldiff = TMath::Abs(relid1[2] - relid2[2]);

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
  //Calculate from absId of a cell its position in module

  short relid[3];
  absToRelNumbering(absId, relid);

  x = (relid[1] - kNumberOfCPVPadsPhi / 2 - 0.5) * kCPVPadSizePhi;
  z = (relid[2] - kNumberOfCPVPadsZ / 2 - 0.5) * kCPVPadSizeZ;
}
bool Geometry::relToAbsNumbering(const short* relId, short& absId)
{

  absId =
    (relId[0] - 1) * kNumberOfCPVPadsPhi * kNumberOfCPVPadsZ + // the offset of PHOS modules
    (relId[2] - 1) * kNumberOfCPVPadsZ +                       // the offset along phi
    relId[1];                                                  // the offset along z

  return true;
}
