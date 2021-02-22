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
#include "FairLogger.h"

using namespace o2::cpv;

ClassImp(Geometry);

unsigned short Geometry::relToAbsId(short moduleNumber, short iphi, short iz)
{
  //converts module number, phi and z coordunates to absId
  return kNumberOfCPVPadsPhi * kNumberOfCPVPadsZ * (moduleNumber - 2) + kNumberOfCPVPadsZ * iphi + iz;
}

bool Geometry::absToRelNumbering(unsigned short absId, short* relid)
{
  // Converts the absolute numbering into the following array
  //  relid[0] = CPV Module number 1:fNModules
  //  relid[1] = Column number inside a CPV module (Phi coordinate)
  //  relid[2] = Row number inside a CPV module (Z coordinate)

  const short nCPV = kNumberOfCPVPadsPhi * kNumberOfCPVPadsZ;
  relid[0] = absId / nCPV + 2;
  absId -= (relid[0] - 2) * nCPV;
  relid[1] = absId / kNumberOfCPVPadsZ;
  relid[2] = absId % kNumberOfCPVPadsZ;

  return true;
}
short Geometry::absIdToModule(unsigned short absId)
{

  return 2 + absId / (kNumberOfCPVPadsPhi * kNumberOfCPVPadsZ);
}

short Geometry::areNeighbours(unsigned short absId1, unsigned short absId2)
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
void Geometry::absIdToRelPosInModule(unsigned short absId, float& x, float& z)
{
  //Calculate from absId of a cell its position in module

  short relid[3];
  absToRelNumbering(absId, relid);

  x = (relid[1] - kNumberOfCPVPadsPhi / 2 + 0.5) * kCPVPadSizePhi;
  z = (relid[2] - kNumberOfCPVPadsZ / 2 + 0.5) * kCPVPadSizeZ;
}
bool Geometry::relToAbsNumbering(const short* relId, unsigned short& absId)
{

  absId =
    (relId[0] - 2) * kNumberOfCPVPadsPhi * kNumberOfCPVPadsZ + // the offset of PHOS modules
    relId[1] * kNumberOfCPVPadsZ +                             // the offset along phi
    relId[2];                                                  // the offset along z

  return true;
}
void Geometry::hwaddressToAbsId(short /*ddl*/, short row, short dilog, short hw, unsigned short& absId)
{
  short mod = row / 16;
  row = row % 16;
  short relid[3] = {short(mod + 2), short(8 * row + hw % 8), short(6 * dilog + hw / 8)};

  relToAbsNumbering(relid, absId);
}

void Geometry::absIdToHWaddress(unsigned short absId, short& mod, short& row, short& dilogic, short& hw)
{
  // Convert absId to hw address
  // Arguments: w32,mod,row,dilogic,address where to write the results
  // return row in range 0...47, packing rows from all modules in common numeration

  short relid[3];
  absToRelNumbering(absId, relid);

  mod = relid[0];                         // DDL# 2..4
  row = relid[1] / 8;                     // row# 0..16
  dilogic = relid[2] / 6;                 // Dilogic# 0..10
  hw = relid[1] % 8 + 8 * (relid[2] % 6); // Address 0..47

  if (hw < 0 || hw > kNPAD) {
    LOG(ERROR) << "Wrong hw address: hw=" << hw << " > kNPAD=" << kNPAD;
    hw = 0;
    dilogic = 0;
    row = 0;
    mod = 0;
    return;
  }
  if (dilogic < 0 || dilogic > kNDilogic) {
    LOG(ERROR) << "Wrong dilogic address: dilogic=" << dilogic << " > kNDilogic=" << kNDilogic;
    hw = 0;
    dilogic = 0;
    row = 0;
    mod = 0;
    return;
  }
  if (row < 0 || row > kNRow) {
    LOG(ERROR) << "Wrong row address: row=" << row << " > kNRow=" << kNRow;
    hw = 0;
    dilogic = 0;
    row = 0;
    mod = 0;
    return;
  }
  row += (mod - 2) * 16;
}
