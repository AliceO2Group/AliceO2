// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "DataFormatsCPV/Cluster.h"
#include "CPVReconstruction/FullCluster.h"
#include "CPVBase/Geometry.h"
#include "CPVBase/CPVSimParams.h"

#include "FairLogger.h" // for LOG

using namespace o2::cpv;

ClassImp(FullCluster);

FullCluster::FullCluster(short digitAbsId, float energy, int label)
  : Cluster()
{
  addDigit(digitAbsId, energy, label);
}
//____________________________________________________________________________
void FullCluster::addDigit(short digitIndex, float energy, int label)
{

  // Adds digit to the FullCluster
  // and accumulates the total amplitude and the multiplicity, list of primaries
  float x = 0., z = 0.;
  Geometry::absIdToRelPosInModule(digitIndex, x, z);

  mElementList.emplace_back(digitIndex, energy, x, z, label);
  mEnergy += energy; // To be updated when calculate cluster properties.
  mMulDigit++;
}
//____________________________________________________________________________
void FullCluster::evalAll()
{
  // Evaluate cluster parameters
  evalLocalPosition();
  //  evalGlobalPosition();
}
//____________________________________________________________________________
void FullCluster::purify()
{
  // Removes digits below threshold

  float threshold = o2::cpv::CPVSimParams::Instance().mDigitMinEnergy;

  std::vector<CluElement>::iterator itEl = mElementList.begin();
  while (itEl != mElementList.end()) {
    if ((*itEl).energy < threshold) { // very rare case
      itEl = mElementList.erase(itEl);
    } else {
      ++itEl;
    }
  }

  mMulDigit = mElementList.size();

  if (mMulDigit == 0) { // too soft cluster
    mEnergy = 0.;
    return;
  }

  // Remove non-connected cells
  if (mMulDigit > 1) {
    mEnergy = 0.; // Recalculate total energy
    auto it = mElementList.begin();
    while (it != mElementList.end()) {
      bool hasNeighbours = false;
      for (auto jt = mElementList.begin(); jt != mElementList.end(); ++jt) {
        if (it == jt) {
          continue;
        }
        if (Geometry::areNeighbours((*it).absId, (*jt).absId) == 1) {
          hasNeighbours = true;
          break;
        }
      }
      if (!hasNeighbours) { // Isolated digits are rare
        it = mElementList.erase(it);
        --mMulDigit;
      } else {
        mEnergy += (*it).energy;
        ++it;
      }
    }
  } else {
    mEnergy = mElementList[0].energy;
  }
}
//____________________________________________________________________________
void FullCluster::evalLocalPosition()
{
  // Calculates the center of gravity in the local CPV-module coordinates
  // Note that correction for non-perpendicular incidence will be applied later
  // when vertex will be known.

  if (mEnergy <= 0.) { // zero energy cluster, position undefined
    mLocalPosX = -999.;
    mLocalPosZ = -999.;
    return;
  }

  // find module number
  mModule = Geometry::absIdToModule(mElementList[0].absId);

  float wtot = 0.;
  mLocalPosX = 0.;
  mLocalPosZ = 0.;
  float invE = 1. / mEnergy;
  for (auto it : mElementList) {
    float w = std::max(float(0.), o2::cpv::CPVSimParams::Instance().mLogWeight + std::log(it.energy * invE));
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
char FullCluster::getNumberOfLocalMax(gsl::span<int> maxAt) const
{
  // Calculates the number of local maxima in the cluster using LocalMaxCut as the minimum
  // energy difference between maximum and surrounding digits

  float locMaxCut = o2::cpv::CPVSimParams::Instance().mLocalMaximumCut;

  std::unique_ptr<bool[]> isLocalMax = std::make_unique<bool[]>(mMulDigit);

  for (int i = 0; i < mMulDigit; i++) {
    if (mElementList[i].energy > o2::cpv::CPVSimParams::Instance().mClusteringThreshold) {
      isLocalMax[i] = true;
    } else {
      isLocalMax[i] = false;
    }
  }
  for (int i = mMulDigit; i--;) {

    for (int j = i; j--;) {

      if (Geometry::areNeighbours(mElementList[i].absId, mElementList[j].absId) == 1) {
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

  unsigned int iDigitN = 0;
  for (int i = mMulDigit; i--;) {
    if (isLocalMax[i]) {
      maxAt[iDigitN] = i;
      iDigitN++;
      if (iDigitN >= maxAt.size()) { // Note that size of output arrays is limited:
        static int nAlarms = 0;
        if (nAlarms++ < 5) {
          LOG(alarm) << "Too many local maxima, cluster multiplicity " << mMulDigit;
        }
        return 0;
      }
    }
  }

  return iDigitN;
}
