// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "PHOSReconstruction/Cluster.h"
#include "PHOSBase/Geometry.h"

using namespace o2::phos;

ClassImp(Cluster);

Cluster::Cluster(int digitAbsId, double energy, double time)
  : mMulDigit(0),
    mModule(0),
    mLocalPosX(-999.),
    mLocalPosZ(-999.),
    mFullEnergy(0.),
    mCoreEnergy(0.),
    mLambdaLong(0.),
    mLambdaShort(0.),
    mDispersion(0.),
    mTime(0.),
    mNExMax(0),
    mDistToBadChannel(0.)
{
  AddDigit(digitAbsId, energy, time);
}

//____________________________________________________________________________
bool Cluster::operator<(const Cluster& other) const
{
  // Compares two Clusters according to their position in the PHOS modules

  Int_t phosmod1 = GetPHOSMod();
  Int_t phosmod2 = other.GetPHOSMod();
  if (phosmod1 != phosmod2) {
    return phosmod1 < phosmod2;
  }

  double posX, posZ;
  GetLocalPosition(posX, posZ);
  double posOtherX, posOtherZ;
  other.GetLocalPosition(posOtherX, posOtherZ);
  Int_t rowdifX = (Int_t)std::ceil(posX / kSortingDelta) - (Int_t)std::ceil(posOtherX / kSortingDelta);
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

  Int_t phosmod1 = GetPHOSMod();
  Int_t phosmod2 = other.GetPHOSMod();
  if (phosmod1 != phosmod2) {
    return phosmod1 > phosmod2;
  }

  double posX, posZ;
  GetLocalPosition(posX, posZ);
  double posOtherX, posOtherZ;
  other.GetLocalPosition(posOtherX, posOtherZ);
  Int_t rowdifX = (Int_t)std::ceil(posX / kSortingDelta) - (Int_t)std::ceil(posOtherX / kSortingDelta);
  if (rowdifX == 0) {
    return posZ < posOtherZ;
  } else {
    return rowdifX < 0;
  }
}
//____________________________________________________________________________
void Cluster::AddDigit(int digitIndex, double energy, double time)
{

  // Adds digit to the Cluster
  // and accumulates the total amplitude and the multiplicity, list of primaries
  mDigitsIdList.push_back(digitIndex);
  mEnergyList.push_back(energy);
  mTimeList.push_back(int(time * 10.)); // store time in units ot 100 ps
  mMulDigit++;
}
//____________________________________________________________________________
void Cluster::EvalAll(const std::vector<Digit>* digits)
{
  // Evaluate cluster parameters
  double rCore = 3.5; // TODO: should be stored in recoParams
  mPHOSGeom = Geometry::GetInstance();
  EvalCoreEnergy(rCore);
  EvalLocalPosition();
  EvalElipsAxis();
  EvalDispersion();
  EvalTime();
  EvalPrimaries(digits);
}
//____________________________________________________________________________
void Cluster::Purify(double threshold)
{
  // Removes digits below threshold

  std::vector<float>::iterator itE = mEnergyList.begin();
  std::vector<int>::iterator itId = mDigitsIdList.begin();
  std::vector<int>::iterator itTime = mTimeList.begin();
  std::vector<int>::iterator jtId;
  while (itE != mEnergyList.end()) {
    if (*itE < threshold) {
      itE = mEnergyList.erase(itE);
      itId = mDigitsIdList.erase(itId);
      itTime = mTimeList.erase(itTime);
    } else {
      itId++;
      itE++;
      itTime++;
    }
  }

  mMulDigit = mEnergyList.size();

  if (mMulDigit == 0) { // too soft cluster
    mFullEnergy = 0.;
    return;
  }

  // Remove non-connected cells
  if (mMulDigit > 1) {
    mFullEnergy = 0.; // Recalculate total energy
    itE = mEnergyList.begin();
    for (itId = mDigitsIdList.begin(); itId != mDigitsIdList.end();) {
      bool hasNeighbours = false;
      for (jtId = mDigitsIdList.begin(); jtId != mDigitsIdList.end(); jtId++) {
        if (itId == jtId) {
          continue;
        }
        if (mPHOSGeom->AreNeighbours(*itId, *jtId) == 1) {
          hasNeighbours = true;
          break;
        }
      }
      if (!hasNeighbours) {
        itE = mEnergyList.erase(itE);
        itId = mDigitsIdList.erase(itId);
        itTime = mTimeList.erase(itId);
      } else {
        mFullEnergy += *itE;
        ++itId;
        ++itTime;
        ++itE;
      }
    }
  } else {
    mFullEnergy = mEnergyList[0];
  }

  mDigitsIdList.shrink_to_fit();
  mTimeList.shrink_to_fit();
  mEnergyList.shrink_to_fit();
}
//____________________________________________________________________________
void Cluster::EvalCoreEnergy(double coreRadius)
{
  // This function calculates energy in the core,
  // i.e. within a radius coreRadius (~3cm) around the center. Beyond this radius
  // in accordance with shower profile the energy deposition
  // should be less than 2%

  if (mLocalPosX < -900.) // local position was not calculated yiet
  {
    EvalLocalPosition();
  }

  std::vector<float>::iterator itE = mEnergyList.begin();
  for (std::vector<int>::iterator i = mDigitsIdList.begin(); i != mDigitsIdList.end(); i++) {
    double xi = 0., zi = 0.;
    mPHOSGeom->AbsIdToRelPosInModule(*i, xi, zi);
    Float_t distance = std::sqrt((xi - mLocalPosX) * (xi - mLocalPosX) + (zi - mLocalPosZ) * (zi - mLocalPosX));
    if (distance < coreRadius) {
      mCoreEnergy += *itE;
    }
    ++itE;
  }
}
//____________________________________________________________________________
void Cluster::EvalLocalPosition()
{
  // Calculates the center of gravity in the local PHOS-module coordinates
  // Note that correction for non-perpendicular incidence will be applied later
  // when vertex will be known.

  if (mFullEnergy <= 0.) { // zero energy cluster, position undefined
    mLocalPosX = -999.;
    mLocalPosZ = -999.;
    return;
  }

  // find module number
  mModule = mPHOSGeom->AbsIdToModule(mDigitsIdList[0]);

  double wtot = 0.;
  mLocalPosX = 0.;
  mLocalPosZ = 0.;
  std::vector<float>::iterator itE = mEnergyList.begin();
  for (std::vector<int>::iterator i = mDigitsIdList.begin(); i != mDigitsIdList.end(); i++) {
    double xi = 0., zi = 0.;
    mPHOSGeom->AbsIdToRelPosInModule(*i, xi, zi);
    if (*itE > 0) {
      double w = std::max(0., kLogWeight + std::log(*itE / mFullEnergy));
      mLocalPosX += xi * w;
      mLocalPosZ += zi * w;
      wtot += w;
    }
    ++itE;
  }
  if (wtot > 0) {
    mLocalPosX /= wtot;
    mLocalPosZ /= wtot;
  }
}
//____________________________________________________________________________
void Cluster::EvalDispersion()
{ // computes the dispersion of the shower

  mDispersion = 0.;
  if (mFullEnergy <= 0.) { // zero energy cluster, dispersion undefined
    return;
  }
  if (mLocalPosX < -900.) // local position was not calculated yiet
  {
    EvalLocalPosition();
  }

  double wtot = 0.;
  std::vector<float>::iterator itE = mEnergyList.begin();
  for (std::vector<int>::iterator i = mDigitsIdList.begin(); i != mDigitsIdList.end(); i++) {
    double xi = 0., zi = 0.;
    mPHOSGeom->AbsIdToRelPosInModule(*i, xi, zi);
    if (*itE > 0) {
      double w = std::max(0., kLogWeight + std::log(*itE / mFullEnergy));
      mDispersion += w * ((xi - mLocalPosX) * (xi - mLocalPosX) + (zi - mLocalPosZ) * (zi - mLocalPosZ));
      wtot += w;
    }
    ++itE;
  }

  if (wtot > 0) {
    mDispersion /= wtot;
  }

  if (mDispersion >= 0)
    mDispersion = std::sqrt(mDispersion);
  else
    mDispersion = 0.;
}
//____________________________________________________________________________
void Cluster::EvalElipsAxis()
{ // computes the axis of shower ellipsoide
  // Calculates the axis of the shower ellipsoid

  if (mFullEnergy <= 0.) { // zero energy cluster, dispersion undefined
    mLambdaLong = mLambdaShort = 0.;
    return;
  }

  double wtot = 0., x = 0., z = 0., dxx = 0., dxz = 0., dzz = 0.;
  std::vector<float>::iterator itE = mEnergyList.begin();
  for (std::vector<int>::iterator i = mDigitsIdList.begin(); i != mDigitsIdList.end(); i++) {
    double xi = 0., zi = 0.;
    mPHOSGeom->AbsIdToRelPosInModule(*i, xi, zi);
    if (*itE > 0) {
      double w = std::max(0., kLogWeight + std::log(*itE / mFullEnergy));
      dxx += w * xi * xi;
      x += w * xi;
      dzz += w * zi * zi;
      z += w * zi;
      dxz += w * xi * zi;
      wtot += w;
    }
    ++itE;
  }
  if (wtot > 0) {
    dxx /= wtot;
    x /= wtot;
    dxx -= x * x;
    dzz /= wtot;
    z /= wtot;
    dzz -= z * z;
    dxz /= wtot;
    dxz -= x * z;

    mLambdaLong = 0.5 * (dxx + dzz) + std::sqrt(0.25 * (dxx - dzz) * (dxx - dzz) + dxz * dxz);
    if (mLambdaLong > 0)
      mLambdaLong = std::sqrt(mLambdaLong);

    mLambdaShort = 0.5 * (dxx + dzz) - std::sqrt(0.25 * (dxx - dzz) * (dxx - dzz) + dxz * dxz);
    if (mLambdaShort > 0) // To avoid exception if numerical errors lead to negative lambda.
      mLambdaShort = std::sqrt(mLambdaShort);
    else
      mLambdaShort = 0.;
  } else {
    mLambdaLong = mLambdaShort = 0.;
  }
}
//____________________________________________________________________________
void Cluster::EvalPrimaries(const std::vector<Digit>* digits)
{
  // Constructs the list of primary particles (tracks) which have contributed to this RecPoint

  if (mFullEnergy <= 0.) { // zero energy cluster, skip primary calculation
    return;
  }
  std::vector<float>::iterator itE = mEnergyList.begin();
  for (std::vector<int>::iterator i = mDigitsIdList.begin(); i != mDigitsIdList.end(); i++) {

    Digit testDigit(*i, 0., 0., 0);                                               // Digit with correct AbsId
    std::vector<Digit>::const_iterator foundIt = BinarySearch(digits, testDigit); // Find this digit in the total list
    if (foundIt == digits->end()) {                                               // should not happen
      continue;
    }

    //int lab = (*foundIt).getLabel();  //index of entry in MCLabels array
    //Add Labels to list of primaries
    //....
    //TODO!!!!
  }
  // Vectors will not be modified any more.
  //TODO: sort and add labels
  mLabels.shrink_to_fit();
  mLabelsEProp.shrink_to_fit();
}
//____________________________________________________________________________
void Cluster::EvalTime()
{

  // Calculate time as time in the digit with maximal energy

  double eMax = 0.;
  std::vector<float>::iterator itE;
  std::vector<int>::iterator itTime = mTimeList.begin();
  for (itE = mEnergyList.begin(); itE != mEnergyList.end(); itE++) {
    if (*itE > eMax) {
      mTime = 0.1 * (*itTime);
      eMax = *itE;
    }
    itTime++;
  }
}
//____________________________________________________________________________
std::vector<Digit>::const_iterator Cluster::BinarySearch(const std::vector<Digit>* container, Digit& element)
{
  const std::vector<Digit>::const_iterator endIt = container->end();

  std::vector<Digit>::const_iterator left = container->begin();
  std::vector<Digit>::const_iterator right = endIt;

  if (container->size() == 0 || container->front() > element || container->back() < element) {
    return endIt;
  }

  while (distance(left, right) > 0) {
    const std::vector<Digit>::const_iterator mid = left + distance(left, right) / 2;

    if (element > *mid) {
      left = mid + 1;
    } else {
      right = mid;
    }
  }

  if (*right == element) {
    return right;
  }

  return endIt;
}
