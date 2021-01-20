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

FullCluster::FullCluster(short digitAbsId, float energy, float time, int label, float scale)
  : Cluster()
{
  addDigit(digitAbsId, energy, time, label, scale);
}
//____________________________________________________________________________
void FullCluster::addDigit(short digitIndex, float energy, float time, int label, float scale)
{

  // Adds digit to the FullCluster
  // and accumulates the total amplitude and the multiplicity, list of primaries
  float x = 0., z = 0.;
  Geometry* phosGeom = Geometry::GetInstance();
  phosGeom->absIdToRelPosInModule(digitIndex, x, z);

  mElementList.emplace_back(digitIndex, energy, time, x, z, label, scale);
  mFullEnergy += energy; //To be updated when calculate cluster properties.
  mMulDigit++;
}
//____________________________________________________________________________
void FullCluster::evalAll()
{
  // Evaluate cluster parameters
  evalCoreEnergy();
  evalLocalPosition();
  evalElipsAxis();
  evalDispersion();
  evalTime();
}
//____________________________________________________________________________
void FullCluster::purify()
{
  // Removes digits below threshold

  float threshold = o2::phos::PHOSSimParams::Instance().mDigitMinEnergy;
  Geometry* phosGeom = Geometry::GetInstance();

  std::vector<CluElement>::iterator itEl = mElementList.begin();
  while (itEl != mElementList.end()) {
    if ((*itEl).energy < threshold) { //very rare case
      itEl = mElementList.erase(itEl);
    } else {
      ++itEl;
    }
  }

  mMulDigit = mElementList.size();

  if (mMulDigit == 0) { // too soft cluster
    mFullEnergy = 0.;
    return;
  }

  // Remove non-connected cells
  if (mMulDigit > 1) {
    mFullEnergy = 0.; // Recalculate total energy
    auto it = mElementList.begin();
    while (it != mElementList.end()) {
      bool hasNeighbours = false;
      for (auto jt = mElementList.begin(); jt != mElementList.end(); ++jt) {
        if (it == jt) {
          continue;
        }
        if (phosGeom->areNeighbours((*it).absId, (*jt).absId) == 1) {
          hasNeighbours = true;
          break;
        }
      }
      if (!hasNeighbours) { //Isolated digits are rare
        it = mElementList.erase(it);
        --mMulDigit;
      } else {
        mFullEnergy += (*it).energy;
        ++it;
      }
    }
  } else {
    mFullEnergy = mElementList[0].energy;
  }
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
  for (auto it : mElementList) {
    Float_t distance = std::sqrt((it.localX - mLocalPosX) * (it.localX - mLocalPosX) +
                                 (it.localZ - mLocalPosZ) * (it.localZ - mLocalPosX));
    if (distance < coreRadius) {
      mCoreEnergy += it.energy;
    }
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
  Geometry* phosGeom = Geometry::GetInstance();
  mModule = phosGeom->absIdToModule(mElementList[0].absId);

  float wtot = 0.;
  mLocalPosX = 0.;
  mLocalPosZ = 0.;
  float invE = 1. / mFullEnergy;
  for (auto it : mElementList) {
    float w = std::max(float(0.), o2::phos::PHOSSimParams::Instance().mLogWeight + std::log(it.energy * invE));
    mLocalPosX += it.localX * w;
    mLocalPosZ += it.localZ * w;
    wtot += w;
  }
  if (wtot > 0) {
    wtot = 1. / wtot;
    mLocalPosX *= wtot;
    mLocalPosZ *= wtot;
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

  float wtot = 0.;
  float invE = 1. / mFullEnergy;
  for (auto it : mElementList) {
    float w = std::max(float(0.), o2::phos::PHOSSimParams::Instance().mLogWeight + std::log(it.energy * invE));
    mDispersion += w * ((it.localX - mLocalPosX) * (it.localX - mLocalPosX) +
                        (it.localZ - mLocalPosZ) * (it.localZ - mLocalPosZ));
    wtot += w;
  }

  if (wtot > 0) {
    mDispersion /= wtot;
  }

  if (mDispersion >= 0) {
    mDispersion = std::sqrt(mDispersion);
  } else {
    mDispersion = 0.;
  }
}
//____________________________________________________________________________
void FullCluster::evalElipsAxis()
{ // computes the axis of shower ellipsoide
  // Calculates the axis of the shower ellipsoid

  if (mFullEnergy <= 0.) { // zero energy cluster, dispersion undefined
    mLambdaLong = mLambdaShort = 0.;
    return;
  }

  float wtot = 0., x = 0., z = 0., dxx = 0., dxz = 0., dzz = 0.;
  float invE = 1. / mFullEnergy;
  for (auto it : mElementList) {
    float xi = 0., zi = 0.;
    float w = std::max(float(0.), o2::phos::PHOSSimParams::Instance().mLogWeight + std::log(it.energy * invE));
    dxx += w * it.localX * it.localX;
    x += w * it.localX;
    dzz += w * it.localZ * it.localZ;
    z += w * it.localZ;
    dxz += w * it.localX * it.localZ;
    wtot += w;
  }
  if (wtot > 0) {
    wtot = 1. / wtot;
    dxx *= wtot;
    x *= wtot;
    dxx -= x * x;
    dzz *= wtot;
    z *= wtot;
    dzz -= z * z;
    dxz *= wtot;
    dxz -= x * z;

    mLambdaLong = 0.5 * (dxx + dzz) + std::sqrt(0.25 * (dxx - dzz) * (dxx - dzz) + dxz * dxz);
    if (mLambdaLong > 0) {
      mLambdaLong = std::sqrt(mLambdaLong);
    }

    mLambdaShort = 0.5 * (dxx + dzz) - std::sqrt(0.25 * (dxx - dzz) * (dxx - dzz) + dxz * dxz);
    if (mLambdaShort > 0) { // To avoid exception if numerical errors lead to negative lambda.
      mLambdaShort = std::sqrt(mLambdaShort);
    } else {
      mLambdaShort = 0.;
    }
  } else {
    mLambdaLong = mLambdaShort = 0.;
  }
}
//____________________________________________________________________________
void FullCluster::evalTime()
{

  // Calculate time as time in the digit with maximal energy

  float eMax = 0.;
  for (auto it : mElementList) {
    if (it.energy > eMax) {
      mTime = it.time;
      eMax = it.energy;
    }
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
char FullCluster::getNumberOfLocalMax(gsl::span<int> maxAt) const
{
  // Calculates the number of local maxima in the cluster using LocalMaxCut as the minimum
  // energy difference between maximum and surrounding digits

  float locMaxCut = o2::phos::PHOSSimParams::Instance().mLocalMaximumCut;

  std::unique_ptr<bool[]> isLocalMax = std::make_unique<bool[]>(mMulDigit);

  for (int i = 0; i < mMulDigit; i++) {
    if (mElementList[i].energy > o2::phos::PHOSSimParams::Instance().mClusteringThreshold) {
      isLocalMax[i] = true;
    } else {
      isLocalMax[i] = false;
    }
  }
  for (int i = 0; i < mMulDigit; i++) {

    for (int j = i + 1; j < mMulDigit; j++) {

      if (Geometry::GetInstance()->areNeighbours(mElementList[i].absId, mElementList[j].absId) == 1) {
        if (mElementList[i].energy > mElementList[j].energy) {
          isLocalMax[j] = false;
          // but may be digit too is not local max ?
          if (mElementList[j].energy > mElementList[i].energy - locMaxCut) {
            isLocalMax[i] = false;
          }
        } else {
          isLocalMax[i] = false;
          // but may be digitN is not local max too?
          if (mElementList[i].energy > mElementList[j].energy - locMaxCut) {
            isLocalMax[j] = false;
          }
        }
      } // if areneighbours
    }   // digit j
  }     // digit i

  int iDigitN = 0;
  for (int i = 0; i < mMulDigit; i++) {
    if (isLocalMax[i]) {
      maxAt[iDigitN] = i;
      iDigitN++;
      if (iDigitN >= o2::phos::PHOSSimParams::Instance().mNLMMax) { // Note that size of output arrays is limited:
        LOG(ERROR) << "Too many local maxima, cluster multiplicity " << mMulDigit;
        return 0;
      }
    }
  }

  return iDigitN;
}

//____________________________________________________________________________
void FullCluster::reset()
{
  //clean up everething
  mElementList.clear();
  mFullEnergy = 0.;
  mCoreEnergy = 0.;
  mMulDigit = 0;
  mNExMax = -1;
  //other cluster parameters will be re-calculated
}