// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "DataFormatsPHOS/Cluster.h"
#include "PHOSBase/Geometry.h"
#include "PHOSBase/PHOSSimParams.h"

using namespace o2::phos;

ClassImp(Cluster);

//____________________________________________________________________________
bool Cluster::operator<(const Cluster& other) const
{
  // Compares two Clusters according to their position in the PHOS modules

  char phosmod1 = module();
  char phosmod2 = other.module();
  if (phosmod1 != phosmod2) {
    return phosmod1 < phosmod2;
  }

  float posX, posZ;
  getLocalPosition(posX, posZ);
  float posOtherX, posOtherZ;
  other.getLocalPosition(posOtherX, posOtherZ);
  int rowdifX = (int)std::ceil(posX / o2::phos::PHOSSimParams::Instance().mSortingDelta) -
                (int)std::ceil(posOtherX / o2::phos::PHOSSimParams::Instance().mSortingDelta);
  if (rowdifX == 0) {
    return posZ > posOtherZ;
  } else {
    return rowdifX > 0;
  }
}
//____________________________________________________________________________
/// \brief Comparison oparator, based on time and absId
/// \param another PHOS Cluster
/// \return result of comparison: x and z coordinates
bool Cluster::operator>(const Cluster& other) const
{
  // Compares two Clusters according to their position in the PHOS modules

  char phosmod1 = module();
  char phosmod2 = other.module();
  if (phosmod1 != phosmod2) {
    return phosmod1 > phosmod2;
  }

  float posX, posZ;
  getLocalPosition(posX, posZ);
  float posOtherX, posOtherZ;
  other.getLocalPosition(posOtherX, posOtherZ);
  int rowdifX = (int)std::ceil(posX / o2::phos::PHOSSimParams::Instance().mSortingDelta) -
                (int)std::ceil(posOtherX / o2::phos::PHOSSimParams::Instance().mSortingDelta);
  if (rowdifX == 0) {
    return posZ < posOtherZ;
  } else {
    return rowdifX < 0;
  }
}
