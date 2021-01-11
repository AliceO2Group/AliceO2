// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Clusterer.cxx
/// \brief Implementation of the CPV cluster finder
#include <memory>

#include "CPVReconstruction/Clusterer.h" // for LOG
#include "CPVBase/Geometry.h"
#include "DataFormatsCPV/Cluster.h"
#include "CPVReconstruction/FullCluster.h"
#include "DataFormatsCPV/Digit.h"
#include "CCDB/CcdbApi.h"
#include "CPVBase/CPVSimParams.h"

#include "FairLogger.h" // for LOG

using namespace o2::cpv;

ClassImp(Clusterer);

//____________________________________________________________________________
void Clusterer::initialize()
{
  mFirstDigitInEvent = 0;
  mLastDigitInEvent = -1;
}

//____________________________________________________________________________
void Clusterer::process(gsl::span<const Digit> digits, gsl::span<const TriggerRecord> dtr,
                        const o2::dataformats::MCTruthContainer<o2::MCCompLabel>& dmc,
                        std::vector<Cluster>* clusters, std::vector<TriggerRecord>* trigRec,
                        o2::dataformats::MCTruthContainer<o2::MCCompLabel>* cluMC)
{
  clusters->clear(); //final out list of clusters
  trigRec->clear();
  if (mRunMC) {
    cluMC->clear();
  }

  for (const auto& tr : dtr) {

    mFirstDigitInEvent = tr.getFirstEntry();
    mLastDigitInEvent = mFirstDigitInEvent + tr.getNumberOfObjects();
    int indexStart = clusters->size();
    mClusters.clear(); // internal list of FullClusters

    LOG(DEBUG) << "Starting clusteriztion digits from " << mFirstDigitInEvent << " to " << mLastDigitInEvent;

    // Collect digits to clusters
    makeClusters(digits);

    // // Unfold overlapped clusters
    // // Split clusters with several local maxima if necessary
    // if (o2::cpv::CPVSimParams::Instance().mUnfoldClusters) {
    //   makeUnfoldings(digits);
    // }

    // Calculate properties of collected clusters (Local position, energy, disp etc.)
    evalCluProperties(digits, clusters, dmc, cluMC);

    LOG(DEBUG) << "Found clusters from " << indexStart << " to " << clusters->size();

    trigRec->emplace_back(tr.getBCData(), indexStart, clusters->size());
  }
}
//____________________________________________________________________________
void Clusterer::makeClusters(gsl::span<const Digit> digits)
{
  // A cluster is defined as a list of neighbour digits

  // Mark all digits as unused yet
  const int maxNDigits = 23040;       // There is no digits more than in CPV modules ;)
  std::bitset<maxNDigits> digitsUsed; ///< Container for bad cells, 1 means bad sell
  digitsUsed.reset();

  int iFirst = mFirstDigitInEvent; // first index of digit which potentially can be a part of cluster

  for (int i = iFirst; i < mLastDigitInEvent; i++) {
    if (digitsUsed.test(i - mFirstDigitInEvent)) {
      continue;
    }

    const Digit& digitSeed = digits[i];
    float digitSeedEnergy = digitSeed.getAmplitude(); //already calibrated digits
    if (digitSeedEnergy < o2::cpv::CPVSimParams::Instance().mDigitMinEnergy) {
      continue;
    }

    // is this digit so energetic that start cluster?
    if (digitSeedEnergy < o2::cpv::CPVSimParams::Instance().mClusteringThreshold) {
      continue;
    }
    // start new cluster
    mClusters.emplace_back(digitSeed.getAbsId(), digitSeedEnergy, digitSeed.getLabel());
    FullCluster& clu = mClusters.back();
    digitsUsed.set(i - mFirstDigitInEvent, true);
    int iDigitInCluster = 1;

    // Now scan remaining digits in list to find neigbours of our seed
    int index = 0;
    while (index < iDigitInCluster) { // scan over digits already in cluster
      short digitSeedAbsId = clu.getDigitAbsId(index);
      index++;
      bool runLoop = true;
      for (Int_t j = iFirst; runLoop && (j < mLastDigitInEvent); j++) {
        if (digitsUsed.test(j - mFirstDigitInEvent)) {
          continue; // look through remaining digits
        }
        const Digit& digitN = digits[j];
        float digitNEnergy = digitN.getAmplitude(); //Already calibrated digits!
        if (digitNEnergy < o2::cpv::CPVSimParams::Instance().mDigitMinEnergy) {
          continue;
        }

        // call (digit,digitN) in THAT oder !!!!!
        Int_t ineb = Geometry::areNeighbours(digitSeedAbsId, digitN.getAbsId());
        switch (ineb) {
          case -1: // too early (e.g. previous module), do not look before j at subsequent passes
            iFirst = j;
            break;
          case 0: // not a neighbour
            break;
          case 1: // are neighbours
            clu.addDigit(digitN.getAbsId(), digitNEnergy, digitN.getLabel());
            iDigitInCluster++;
            digitsUsed.set(j - mFirstDigitInEvent, true);
            break;
          case 2: // too far from each other
          default:
            runLoop = false;
            break;
        } // switch
      }
    } // loop over cluster
  }   // energy theshold
}
//__________________________________________________________________________
void Clusterer::makeUnfoldings(gsl::span<const Digit> digits)
{
  //Split cluster if several local maxima are found

  std::array<int, NLMMax> maxAt; // NLMMax:Maximal number of local maxima

  int numberOfNotUnfolded = mClusters.size();

  for (int i = 0; i < numberOfNotUnfolded; i++) { //can not use iterator here as list can expand
    FullCluster& clu = mClusters[i];
    if (clu.getNExMax() > -1) { //already unfolded
      continue;
    }
    char nMultipl = clu.getMultiplicity();
    char nMax = clu.getNumberOfLocalMax(maxAt);
    if (nMax > 1) {
      unfoldOneCluster(clu, nMax, maxAt, digits);
      clu.setEnergy(0); // will be skipped later
    } else {
      clu.setNExMax(nMax); // Only one local maximum
    }
  }
}
//____________________________________________________________________________
void Clusterer::unfoldOneCluster(FullCluster& iniClu, char nMax, gsl::span<int> digitId, gsl::span<const Digit> digits)
{
  // Performs the unfolding of a cluster with nMax overlapping showers
  // Parameters: iniClu cluster to be unfolded
  //             nMax number of local maxima found (this is the number of new clusters)
  //             digitId: index of digits, corresponding to local maxima
  //             maxAtEnergy: energies of digits, corresponding to local maxima

  // Take initial cluster and calculate local coordinates of digits
  // To avoid multiple re-calculation of same parameters
  char mult = iniClu.getMultiplicity();
  if (meInClusters.capacity() < mult) {
    meInClusters.reserve(mult);
    mfij.reserve(mult);
  }

  const std::vector<FullCluster::CluElement>* cluElist = iniClu.getElementList();

  // Coordinates of centers of clusters
  std::array<float, NLMMax> xMax;
  std::array<float, NLMMax> zMax;
  std::array<float, NLMMax> eMax;
  std::array<float, NLMMax> deNew;

  //transient variables
  std::array<float, NLMMax> a;
  std::array<float, NLMMax> b;
  std::array<float, NLMMax> c;

  for (int iclu = 0; iclu < nMax; iclu++) {
    xMax[iclu] = (*cluElist)[digitId[iclu]].localX;
    zMax[iclu] = (*cluElist)[digitId[iclu]].localZ;
    eMax[iclu] = 2. * (*cluElist)[digitId[iclu]].energy;
  }

  std::array<float, NLMMax> prop; // proportion of clusters in the current digit

  // Try to decompose cluster to contributions
  int nIterations = 0;
  bool insuficientAccuracy = true;
  while (insuficientAccuracy && nIterations < o2::cpv::CPVSimParams::Instance().mNMaxIterations) {
    insuficientAccuracy = false; // will be true if at least one parameter changed too much
    std::memset(&a, 0, sizeof a);
    std::memset(&b, 0, sizeof b);
    std::memset(&c, 0, sizeof c);
    //First calculate shower shapes
    for (int idig = 0; idig < mult; idig++) {
      auto it = (*cluElist)[idig];
      for (int iclu = 0; iclu < nMax; iclu++) {
        mfij[idig][iclu] = responseShape(it.localX - xMax[iclu], it.localZ - zMax[iclu]);
      }
    }

    //Fit energies
    for (int idig = 0; idig < mult; idig++) {
      auto it = (*cluElist)[idig];
      for (int iclu = 0; iclu < nMax; iclu++) {
        a[iclu] += mfij[idig][iclu] * mfij[idig][iclu];
        b[iclu] += it.energy * mfij[idig][iclu];
        for (int kclu = 0; kclu < nMax; kclu++) {
          if (iclu == kclu) {
            continue;
          }
          c[iclu] += eMax[kclu] * mfij[idig][iclu] * mfij[idig][kclu];
        }
      }
    }
    //Evaluate new maximal energies
    for (int iclu = 0; iclu < nMax; iclu++) {
      if (a[iclu] != 0.) {
        float eNew = (b[iclu] - c[iclu]) / a[iclu];
        insuficientAccuracy += (std::abs(eMax[iclu] - eNew) > eNew * o2::cpv::CPVSimParams::Instance().mUnfogingEAccuracy);
        eMax[iclu] = eNew;
      }
    } // otherwise keep old value

    // Loop over all digits of parent cluster and split their energies between daughter clusters
    // according to shower shape
    // then re-evaluate local position of clusters
    for (int idig = 0; idig < mult; idig++) {
      float eEstimated = 0;
      for (int iclu = 0; iclu < nMax; iclu++) {
        prop[iclu] = eMax[iclu] * mfij[idig][iclu];
        eEstimated += prop[iclu];
      }
      if (eEstimated == 0.) { // numerical accuracy
        continue;
      }
      // Split energy of digit according to contributions
      for (int iclu = 0; iclu < nMax; iclu++) {
        meInClusters[idig][iclu] = (*cluElist)[idig].energy * prop[iclu] / eEstimated;
      }
    }

    // Recalculate parameters of clusters and check relative variation of energy and absolute of position
    for (int iclu = 0; iclu < nMax; iclu++) {
      float oldX = xMax[iclu];
      float oldZ = zMax[iclu];
      // full energy, need for weight
      float eTotNew = 0;
      for (int idig = 0; idig < mult; idig++) {
        eTotNew += meInClusters[idig][iclu];
      }
      xMax[iclu] = 0;
      zMax[iclu] = 0.;
      float wtot = 0.;
      for (int idig = 0; idig < mult; idig++) {
        if (meInClusters[idig][iclu] > 0) {
          // In unfolding it is better to use linear weight to reduce contribution of unfolded tails
          float w = meInClusters[idig][iclu] / eTotNew;
          // float w = std::max(std::log(eInClusters[idig][iclu] / eTotNew) + o2::cpv::CPVSimParams::Instance().mLogWeight, float(0.));
          xMax[iclu] += (*cluElist)[idig].localX * w;
          zMax[iclu] += (*cluElist)[idig].localZ * w;
          wtot += w;
        }
      }
      if (wtot > 0.) {
        wtot = 1. / wtot;
        xMax[iclu] *= wtot;
        zMax[iclu] *= wtot;
      }
      // Compare variation of parameters
      insuficientAccuracy += (std::abs(xMax[iclu] - oldX) > o2::cpv::CPVSimParams::Instance().mUnfogingXZAccuracy);
      insuficientAccuracy += (std::abs(zMax[iclu] - oldZ) > o2::cpv::CPVSimParams::Instance().mUnfogingXZAccuracy);
    }
    nIterations++;
  }
  // Iterations finished, add new clusters
  for (int iclu = 0; iclu < nMax; iclu++) {
    mClusters.emplace_back();
    FullCluster& clu = mClusters.back();
    clu.setNExMax(nMax);
    int idig = 0;
    for (int idig = 0; idig < mult; idig++) {
      float eDigit = meInClusters[idig][iclu];
      idig++;
      if (eDigit < o2::cpv::CPVSimParams::Instance().mDigitMinEnergy) {
        continue;
      }
      clu.addDigit((*cluElist)[idig].absId, eDigit, (*cluElist)[idig].label);
    }
  }
}

//____________________________________________________________________________
void Clusterer::evalCluProperties(gsl::span<const Digit> digits, std::vector<Cluster>* clusters,
                                  const o2::dataformats::MCTruthContainer<o2::MCCompLabel>& dmc,
                                  o2::dataformats::MCTruthContainer<o2::MCCompLabel>* cluMC)
{

  if (clusters->capacity() - clusters->size() < mClusters.size()) { //avoid expanding vector per element
    clusters->reserve(clusters->size() + mClusters.size());
  }

  int labelIndex = 0;
  if (mRunMC) {
    labelIndex = cluMC->getIndexedSize();
  }

  auto clu = mClusters.begin();

  while (clu != mClusters.end()) {

    if (clu->getEnergy() < 1.e-4) { //Marked earlier for removal
      ++clu;
      continue;
    }

    // may be soft digits remain after unfolding
    clu->purify();

    //  LOG(DEBUG) << "Purify done";
    clu->evalAll();

    if (clu->getEnergy() > 1.e-4) { //Non-empty cluster
      clusters->emplace_back(*clu);

      if (mRunMC) { //Handle labels
        //Calculate list of primaries
        //loop over entries in digit MCTruthContainer
        const std::vector<FullCluster::CluElement>* vl = clu->getElementList();
        auto ll = vl->begin();
        while (ll != vl->end()) {
          int i = (*ll).label; //index
          if (i < 0) {
            ++ll;
            continue;
          }
          gsl::span<const o2::MCCompLabel> spDigList = dmc.getLabels(i);
          gsl::span<o2::MCCompLabel> spCluList = cluMC->getLabels(labelIndex); //get updated list
          auto digL = spDigList.begin();
          while (digL != spDigList.end()) {
            bool exist = false;
            auto cluL = spCluList.begin();
            while (cluL != spCluList.end()) {
              if (*digL == *cluL) { //exist
                exist = true;
                break;
              }
              ++cluL;
            }
            if (!exist) { //just add label
              cluMC->addElement(labelIndex, (*digL));
            }
            ++digL;
          }
          ++ll;
        }
        labelIndex++;
      } // Work with MC
    }

    ++clu;
  }
}
//____________________________________________________________________________
float Clusterer::responseShape(float x, float z)
{
  // Shape of the shower (see CPV TDR)
  // we neglect dependence on the incident angle.

  const float width = 1. / (2. * 2.32 * 2.32 * 2.32 * 2.32 * 2.32 * 2.32);
  float r2 = x * x + z * z;
  return TMath::Exp(-r2 * r2 * r2 * width);
/*  float r2 = x * x + z * z;
  float r4 = r2 * r2;
  float r295 = TMath::Power(r2, 2.95 / 2.);
  float shape = TMath::Exp(-r4 * (1. / (2.32 + 0.26 * r4) + 0.0316 / (1 + 0.0652 * r295)));
  return shape;
*/}
