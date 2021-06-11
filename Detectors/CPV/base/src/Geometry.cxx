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
void Geometry::hwaddressToAbsId(short ccId, short dil, short gas, short pad, unsigned short& absId)
{

  short pZ = mPadToZ[pad];
  short pPhi = mPadToPhi[pad];
  short relid[3] = {short(ccId / 8 + 2), short((ccId % 8) * 16 + (dil / 2) * 8 + 7 - pPhi), short((dil % 2) * 30 + gas * 6 + pZ)};

  relToAbsNumbering(relid, absId);
}

void Geometry::absIdToHWaddress(unsigned short absId, short& ccId, short& dil, short& gas, short& pad)
{
  // Convert absId to hw address
  // Arguments: ccId:  0 -- 7 - mod 2;  8...15 mod 3; 16...23 mod 4
  //dilogic: 0..3, gas=0..5, pad:0..47

  short relid[3];
  absToRelNumbering(absId, relid);

  ccId = (relid[0] - 2) * 8 + relid[1] / 16;
  dil = 2 * ((relid[1] % 16) / 8) + relid[2] / 30; // Dilogic# 0..3
  gas = (relid[2] % 30) / 6;                       // gasiplex# 0..4
  pad = mPadMap[relid[2] % 6][7 - relid[1] % 8];   // pad 0..47

  if (pad < 0 || pad > kNPAD) {
    LOG(ERROR) << "Wrong pad address: pad=" << pad << " > kNPAD=" << kNPAD;
    pad = 0;
    dil = 0;
    gas = 0;
    ccId = 0;
    return;
  }
  if (dil < 0 || dil >= kNDilogic) {
    LOG(ERROR) << "Wrong dil address: dil=" << dil << " > kNDilogic=" << kNDilogic;
    pad = 0;
    dil = 0;
    gas = 0;
    ccId = 0;
    return;
  }
  if (gas < 0 || gas >= kNGas) {
    LOG(ERROR) << "Wrong gasiplex address: gas=" << gas << " > kNGas=" << kNGas;
    pad = 0;
    dil = 0;
    gas = 0;
    ccId = 0;
    return;
  }
}
