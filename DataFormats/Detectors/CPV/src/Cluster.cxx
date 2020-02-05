// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "DataFormatsCPV/Cluster.h"
#include "CPVBase/Geometry.h"
#include "CPVBase/CPVSimParams.h"

using namespace o2::cpv;

ClassImp(Cluster);

//____________________________________________________________________________
bool Cluster::operator<(const Cluster& other) const
{
  // Compares two Clusters according to their position in the CPV modules

  char cpvmod1 = getModule();
  char cpvmod2 = other.getModule();
  if (cpvmod1 != cpvmod2) {
    return cpvmod1 < cpvmod2;
  }

  float posX, posZ;
  getLocalPosition(posX, posZ);
  float posOtherX, posOtherZ;
  other.getLocalPosition(posOtherX, posOtherZ);
  int rowdifX = (int)std::ceil(posX * o2::cpv::CPVSimParams::Instance().mSortingDelta) -
                (int)std::ceil(posOtherX * o2::cpv::CPVSimParams::Instance().mSortingDelta);
  if (rowdifX == 0) {
    return posZ > posOtherZ;
  } else {
    return rowdifX > 0;
  }
}
//____________________________________________________________________________
/// \brief Comparison oparator, based on time and absId
/// \param another CPV Cluster
/// \return result of comparison: x and z coordinates
bool Cluster::operator>(const Cluster& other) const
{
  // Compares two Clusters according to their position in the CPV modules

  if (mModule != other.mModule) {
    return mModule > other.mModule;
  }

  int rowdifX = (int)std::ceil(mLocalPosX * o2::cpv::CPVSimParams::Instance().mSortingDelta) -
                (int)std::ceil(other.mLocalPosX * o2::cpv::CPVSimParams::Instance().mSortingDelta);
  if (rowdifX == 0) {
    return mLocalPosZ < other.mLocalPosZ;
  } else {
    return rowdifX < 0;
  }
}
