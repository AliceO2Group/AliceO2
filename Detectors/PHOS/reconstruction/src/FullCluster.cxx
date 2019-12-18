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
#include "PHOSReconstruction/FullCluster.h"
#include "PHOSBase/Geometry.h"
#include "PHOSBase/PHOSSimParams.h"

#include "FairLogger.h" // for LOG

using namespace o2::phos;

ClassImp(FullCluster);

FullCluster::FullCluster(int digitAbsId, double energy, double time, int label, float scale)
  : Cluster()
{
  addDigit(digitAbsId, energy, time, label, scale);
}
//____________________________________________________________________________
void FullCluster::addDigit(int digitIndex, double energy, double time, int label, float scale)
{

  // Adds digit to the FullCluster
  // and accumulates the total amplitude and the multiplicity, list of primaries
  mDigitsIdList.push_back(digitIndex);
  mEnergyList.push_back(energy);
  mTimeList.push_back(int(time * 10.)); // store time in units ot 100 ps
  if (label >= 0) {
    mLabels.push_back(std::pair(label, scale));
  }
  mMulDigit++;
}
//____________________________________________________________________________
void FullCluster::evalAll(const std::vector<Digit>* digits)
{
  // Evaluate cluster parameters
  mPHOSGeom = Geometry::GetInstance();
  evalCoreEnergy();
  evalLocalPosition();
  evalElipsAxis();
  evalDispersion();
  evalTime();
}
//____________________________________________________________________________
void FullCluster::purify(double threshold)
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
        if (mPHOSGeom->areNeighbours(*itId, *jtId) == 1) {
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
void FullCluster::evalCoreEnergy()
{
  // This function calculates energy in the core,
  // i.e. within a radius coreRadius (~3cm) around the center. Beyond this radius
  // in accordance with shower profile the energy deposition
  // should be less than 2%

  if (mLocalPosX < -900.) // local position was not calculated yiet
  {
    evalLocalPosition();
  }

  float coreRadius = o2::phos::PHOSSimParams::Instance().mCoreR;
  std::vector<float>::iterator itE = mEnergyList.begin();
  for (std::vector<int>::iterator i = mDigitsIdList.begin(); i != mDigitsIdList.end(); i++) {
    double xi = 0., zi = 0.;
    mPHOSGeom->absIdToRelPosInModule(*i, xi, zi);
    Float_t distance = std::sqrt((xi - mLocalPosX) * (xi - mLocalPosX) + (zi - mLocalPosZ) * (zi - mLocalPosX));
    if (distance < coreRadius) {
      mCoreEnergy += *itE;
    }
    ++itE;
  }
}
//____________________________________________________________________________
void FullCluster::evalLocalPosition()
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
  mModule = mPHOSGeom->absIdToModule(mDigitsIdList[0]);

  double wtot = 0.;
  mLocalPosX = 0.;
  mLocalPosZ = 0.;
  std::vector<float>::iterator itE = mEnergyList.begin();
  for (std::vector<int>::iterator i = mDigitsIdList.begin(); i != mDigitsIdList.end(); i++) {
    double xi = 0., zi = 0.;
    mPHOSGeom->absIdToRelPosInModule(*i, xi, zi);
    if (*itE > 0) {
      double w = std::max(0., o2::phos::PHOSSimParams::Instance().mLogWeight + std::log(*itE / mFullEnergy));
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
void FullCluster::evalDispersion()
{ // computes the dispersion of the shower

  mDispersion = 0.;
  if (mFullEnergy <= 0.) { // zero energy cluster, dispersion undefined
    return;
  }
  if (mLocalPosX < -900.) // local position was not calculated yiet
  {
    evalLocalPosition();
  }

  double wtot = 0.;
  std::vector<float>::iterator itE = mEnergyList.begin();
  for (std::vector<int>::iterator i = mDigitsIdList.begin(); i != mDigitsIdList.end(); i++) {
    double xi = 0., zi = 0.;
    mPHOSGeom->absIdToRelPosInModule(*i, xi, zi);
    if (*itE > 0) {
      double w = std::max(0., o2::phos::PHOSSimParams::Instance().mLogWeight + std::log(*itE / mFullEnergy));
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
void FullCluster::evalElipsAxis()
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
    mPHOSGeom->absIdToRelPosInModule(*i, xi, zi);
    if (*itE > 0) {
      double w = std::max(0., o2::phos::PHOSSimParams::Instance().mLogWeight + std::log(*itE / mFullEnergy));
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
void FullCluster::evalTime()
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
std::vector<Digit>::const_iterator FullCluster::BinarySearch(const std::vector<Digit>* container, Digit& element)
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
//____________________________________________________________________________
int FullCluster::getNumberOfLocalMax(int* maxAt, float* maxAtEnergy) const
{
  // Calculates the number of local maxima in the cluster using LocalMaxCut as the minimum
  // energy difference between maximum and surrounding digits

  int n = getMultiplicity();
  float locMaxCut = o2::phos::PHOSSimParams::Instance().mLocalMaximumCut;

  bool* isLocalMax = new bool[n];
  for (int i = 0; i < n; i++) {
    isLocalMax[i] = false;
    float en1 = mEnergyList.at(i);
    if (en1 > o2::phos::PHOSSimParams::Instance().mClusteringThreshold)
      isLocalMax[i] = true;
  }

  for (int i = 0; i < n; i++) {
    int detId1 = mDigitsIdList.at(i);
    float en1 = mEnergyList.at(i);

    for (int j = i + 1; j < n; j++) {
      int detId2 = mDigitsIdList.at(j);
      float en2 = mEnergyList.at(j);

      if (Geometry::GetInstance()->areNeighbours(detId1, detId2) == 1) {
        if (en1 > en2) {
          isLocalMax[j] = false;
          // but may be digit too is not local max ?
          if (en2 > en1 - locMaxCut) {
            isLocalMax[i] = false;
          }
        } else {
          isLocalMax[i] = false;
          // but may be digitN is not local max too?
          if (en1 > en2 - locMaxCut) {
            isLocalMax[j] = false;
          }
        }
      } // if areneighbours
    }   // digit j
  }     // digit i

  int iDigitN = 0;
  for (int i = 0; i < n; i++) {
    if (isLocalMax[i]) {
      maxAt[iDigitN] = i;
      maxAtEnergy[iDigitN] = mEnergyList.at(i);
      iDigitN++;
      if (iDigitN >= o2::phos::PHOSSimParams::Instance().mNLMMax) { // Note that size of output arrays is limited:
        LOG(ERROR) << "Too many local maxima, cluster multiplicity " << n;
        delete[] isLocalMax;
        return 0;
      }
    }
  }
  delete[] isLocalMax;
  return iDigitN;
}
