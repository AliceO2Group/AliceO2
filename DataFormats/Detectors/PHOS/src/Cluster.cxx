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

Cluster::Cluster(const Cluster& clu)
  : mMulDigit(clu.mMulDigit),
    mModule(clu.mModule),
    mLabel(clu.mLabel),
    mNExMax(clu.mNExMax),
    mLocalPosX(clu.mLocalPosX),
    mLocalPosZ(clu.mLocalPosZ),
    mFullEnergy(clu.mFullEnergy),
    mCoreEnergy(clu.mCoreEnergy),
    mLambdaLong(clu.mLambdaLong),
    mLambdaShort(clu.mLambdaShort),
    mDispersion(clu.mDispersion),
    mTime(clu.mTime),
    mDistToBadChannel(clu.mDistToBadChannel)
{
}

//____________________________________________________________________________
bool Cluster::operator<(const Cluster& other) const
{
  // Compares two Clusters according to their position in the PHOS modules

  Int_t phosmod1 = getPHOSMod();
  Int_t phosmod2 = other.getPHOSMod();
  if (phosmod1 != phosmod2) {
    return phosmod1 < phosmod2;
  }

  double posX, posZ;
  getLocalPosition(posX, posZ);
  double posOtherX, posOtherZ;
  other.getLocalPosition(posOtherX, posOtherZ);
  Int_t rowdifX = (Int_t)std::ceil(posX / o2::phos::PHOSSimParams::Instance().mSortingDelta) -
                  (Int_t)std::ceil(posOtherX / o2::phos::PHOSSimParams::Instance().mSortingDelta);
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

  Int_t phosmod1 = getPHOSMod();
  Int_t phosmod2 = other.getPHOSMod();
  if (phosmod1 != phosmod2) {
    return phosmod1 > phosmod2;
  }

  double posX, posZ;
  getLocalPosition(posX, posZ);
  double posOtherX, posOtherZ;
  other.getLocalPosition(posOtherX, posOtherZ);
  Int_t rowdifX = (Int_t)std::ceil(posX / o2::phos::PHOSSimParams::Instance().mSortingDelta) -
                  (Int_t)std::ceil(posOtherX / o2::phos::PHOSSimParams::Instance().mSortingDelta);
  if (rowdifX == 0) {
    return posZ < posOtherZ;
  } else {
    return rowdifX < 0;
  }
}
