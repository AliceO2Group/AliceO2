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

/// \file Clusterer.cxx
/// \brief Implementation of the PHOS cluster finder
#include <memory>
#include "TDecompBK.h"

#include "PHOSReconstruction/Clusterer.h" // for LOG
#include "PHOSBase/PHOSSimParams.h"
#include "DataFormatsPHOS/Cluster.h"
#include "DataFormatsPHOS/Digit.h"

#include <fairlogger/Logger.h> // for LOG

using namespace o2::phos;

ClassImp(Clusterer);

//____________________________________________________________________________
void Clusterer::initialize()
{
  if (!mPHOSGeom) {
    mPHOSGeom = Geometry::GetInstance();
  }
  mFirstElememtInEvent = 0;
  mLastElementInEvent = -1;
  LOG(info) << "Clusterizer parameters";
  const PHOSSimParams& sp = o2::phos::PHOSSimParams::Instance();
  LOG(info) << "mLogWeight = " << sp.mLogWeight;
  LOG(info) << "mDigitMinEnergy = " << sp.mDigitMinEnergy;
  LOG(info) << "mClusteringThreshold = " << sp.mClusteringThreshold;
  LOG(info) << "mLocalMaximumCut = " << sp.mLocalMaximumCut;
  LOG(info) << "mUnfoldMaxSize = " << sp.mUnfoldMaxSize;
  LOG(info) << "mUnfoldClusters = " << sp.mUnfoldClusters;
  LOG(info) << "mUnfogingEAccuracy = " << sp.mUnfogingEAccuracy;
  LOG(info) << "mUnfogingXZAccuracy = " << sp.mUnfogingXZAccuracy;
  LOG(info) << "mUnfogingChi2Accuracy = " << sp.mUnfogingChi2Accuracy;
  LOG(info) << "mNMaxIterations = " << sp.mNMaxIterations;
}
//____________________________________________________________________________
void Clusterer::process(gsl::span<const Digit> digits, gsl::span<const TriggerRecord> dtr,
                        const o2::dataformats::MCTruthContainer<MCLabel>* dmc,
                        std::vector<Cluster>& clusters, std::vector<CluElement>& cluelements, std::vector<TriggerRecord>& trigRec,
                        o2::dataformats::MCTruthContainer<MCLabel>& cluMC)
{
  clusters.clear(); // final out list of clusters
  cluelements.clear();
  cluelements.reserve(digits.size());
  trigRec.clear();
  cluMC.clear();
  mProcessMC = (dmc != nullptr);

  for (const auto& tr : dtr) {
    int indexStart = clusters.size(); // final out list of clusters

    LOG(debug) << "Starting clusteriztion digits from " << mFirstElememtInEvent << " to " << mLastElementInEvent;
    // Convert digits to cluelements
    int firstDigitInEvent = tr.getFirstEntry();
    int lastDigitInEvent = firstDigitInEvent + tr.getNumberOfObjects();
    mFirstElememtInEvent = cluelements.size();
    mCluEl.clear();
    mTrigger.clear();
    for (int i = firstDigitInEvent; i < lastDigitInEvent; i++) {
      const Digit& digitSeed = digits[i];
      short absId = digitSeed.getAbsId();
      if (digitSeed.isTRU()) {
        mTrigger.emplace_back(digitSeed);
        continue;
      }
      if (isBadChannel(absId)) {
        continue;
      }
      float energy = calibrate(digitSeed.getAmplitude(), absId, digitSeed.isHighGain());
      if (energy < o2::phos::PHOSSimParams::Instance().mDigitMinEnergy) {
        continue;
      }
      float x = 0., z = 0.;
      Geometry::absIdToRelPosInModule(digits[i].getAbsId(), x, z);
      mCluEl.emplace_back(absId, digitSeed.isHighGain(), energy, calibrateT(digitSeed.getTime(), absId, digitSeed.isHighGain(), tr.getBCData().bc),
                          x, z, digitSeed.getLabel(), 1.);
    }
    mLastElementInEvent = cluelements.size();

    // Collect digits to clusters
    makeClusters(clusters, cluelements);

    LOG(debug) << "Found clusters from " << indexStart << " to " << clusters.size();
    trigRec.emplace_back(tr.getBCData(), indexStart, clusters.size() - indexStart);
  }
  if (mProcessMC) {
    evalLabels(clusters, cluelements, dmc, cluMC);
  }
}
//____________________________________________________________________________
void Clusterer::processCells(gsl::span<const Cell> cells, gsl::span<const TriggerRecord> ctr,
                             const o2::dataformats::MCTruthContainer<MCLabel>* dmc,
                             std::vector<Cluster>& clusters, std::vector<CluElement>& cluelements, std::vector<TriggerRecord>& trigRec,
                             o2::dataformats::MCTruthContainer<MCLabel>& cluMC)
{
  // Transform input Cells to digits and run standard recontruction
  clusters.clear(); // final out list of clusters
  cluelements.clear();
  cluelements.reserve(cells.size());
  trigRec.clear();
  cluMC.clear();
  mProcessMC = (dmc != nullptr);
  miCellLabel = 0;
  for (const auto& tr : ctr) {
    int firstCellInEvent = tr.getFirstEntry();
    int lastCellInEvent = firstCellInEvent + tr.getNumberOfObjects();
    int indexStart = clusters.size(); // final out list of clusters
    LOG(debug) << "Starting clusteriztion cells from " << firstCellInEvent << " to " << lastCellInEvent;
    // convert cells to cluelements
    mFirstElememtInEvent = cluelements.size();
    mCluEl.clear();
    mTrigger.clear();
    for (int i = firstCellInEvent; i < lastCellInEvent; i++) {
      const Cell c = cells[i];
      short absId = c.getAbsId();
      if (c.getTRU()) {
        mTrigger.emplace_back(c.getTRUId(), c.getEnergy(), c.getTime(), 0);
        continue;
      }
      if (isBadChannel(absId)) {
        continue;
      }
      float energy = calibrate(c.getEnergy(), absId, c.getHighGain());
      if (energy < o2::phos::PHOSSimParams::Instance().mDigitMinEnergy) {
        continue;
      }
      float x = 0., z = 0.;
      Geometry::absIdToRelPosInModule(absId, x, z);
      mCluEl.emplace_back(absId, c.getHighGain(), energy, calibrateT(c.getTime(), absId, c.getHighGain(), tr.getBCData().bc),
                          x, z, i, 1.);
    }
    mLastElementInEvent = cluelements.size();
    makeClusters(clusters, cluelements);
    trigRec.emplace_back(tr.getBCData(), indexStart, clusters.size() - indexStart);
  }
  if (mProcessMC) {
    evalLabels(clusters, cluelements, dmc, cluMC);
  }
}
//____________________________________________________________________________
void Clusterer::makeClusters(std::vector<Cluster>& clusters, std::vector<CluElement>& cluelements)
{
  // A cluster is defined as a list of neighbour digits (as defined in Geometry::areNeighbours)
  // Cluster contains first and (next-to) last index of the combined list of clusterelements, so
  // add elements to final list and mark element in internal list as used (zero energy)

  LOG(debug) << "makeClusters: clusters size=" << clusters.size() << " elements=" << cluelements.size();
  int iFirst = 0; // first index of digit which potentially can be a part of cluster
  int n = mCluEl.size();
  for (int i = iFirst; i < n; i++) {
    if (mCluEl[i].energy == 0) { // already used
      continue;
    }

    CluElement& digitSeed = mCluEl[i];

    // is this digit so energetic that start cluster?
    Cluster* clu = nullptr;
    int iDigitInCluster = 0;
    if (digitSeed.energy > o2::phos::PHOSSimParams::Instance().mClusteringThreshold) {
      // start new cluster
      clusters.emplace_back();
      clu = &(clusters.back());
      clu->setFirstCluEl(cluelements.size());
      cluelements.emplace_back(digitSeed);
      digitSeed.energy = 0;
      iDigitInCluster = 1;
    } else {
      continue;
    }
    // Now scan remaining digits in list to find neigbours of our seed
    int index = 0;
    while (index < iDigitInCluster) { // scan over digits already in cluster
      short digitSeedAbsId = cluelements.at(clu->getFirstCluEl() + index).absId;
      index++;
      for (int j = iFirst; j < n; j++) {
        if (mCluEl[j].energy == 0) {
          continue; // look through remaining digits
        }
        CluElement& digitN = mCluEl[j];

        // call (digit,digitN) in THAT oder !!!!!
        Int_t ineb = Geometry::areNeighbours(digitSeedAbsId, digitN.absId);
        switch (ineb) {
          case -1: // too early (e.g. previous module), do not look before j at subsequent passes
            iFirst = j;
            break;
          case 0: // not a neighbour
            break;
          case 1: // are neighbours
            cluelements.emplace_back(digitN);
            digitN.energy = 0;
            iDigitInCluster++;
            break;
          case 2: // too far from each other
          default:
            break;
        } // switch
      }
    } // loop over cluster
    clu->setLastCluEl(cluelements.size());
    LOG(debug) << "Cluster: elements from " << clu->getFirstCluEl() << " last=" << cluelements.size();

    // Unfold overlapped clusters
    // Split clusters with several local maxima if necessary
    if (o2::phos::PHOSSimParams::Instance().mUnfoldClusters &&
        clu->getMultiplicity() < o2::phos::PHOSSimParams::Instance().mUnfoldMaxSize) { // Do not unfold huge clusters
      makeUnfolding(*clu, clusters, cluelements);
    } else {
      evalAll(*clu, cluelements);
      if (clu->getEnergy() < 1.e-4) { // remove cluster and belonging to it elements
        for (int i = clu->getMultiplicity(); i--;) {
          cluelements.pop_back();
        }
        clusters.pop_back();
      }
    }

  } // energy theshold
}
//__________________________________________________________________________
void Clusterer::makeUnfolding(Cluster& clu, std::vector<Cluster>& clusters, std::vector<CluElement>& cluelements)
{
  // Split cluster if several local maxima are found
  if (clu.getNExMax() > -1) { // already unfolded
    return;
  }

  char nMax = getNumberOfLocalMax(clu, cluelements);
  if (nMax > 1) {
    unfoldOneCluster(clu, nMax, clusters, cluelements);
  } else {
    clu.setNExMax(nMax); // Only one local maximum
    evalAll(clu, cluelements);
    if (clu.getEnergy() < 1.e-4) { // remove cluster and belonging to it elements
      for (int i = clu.getMultiplicity(); i--;) {
        cluelements.pop_back();
      }
      clusters.pop_back();
    }
  }
}
//____________________________________________________________________________
void Clusterer::unfoldOneCluster(Cluster& iniClu, char nMax, std::vector<Cluster>& clusters, std::vector<CluElement>& cluelements)
{
  // Performs the unfolding of a cluster with nMax overlapping showers
  // Parameters: iniClu cluster to be unfolded
  //             nMax number of local maxima found (this is the number of new clusters)
  //             digitId: index of digits, corresponding to local maxima
  //             maxAtEnergy: energies of digits, corresponding to local maxima

  // Take initial cluster and calculate local coordinates of digits
  // To avoid multiple re-calculation of same parameters
  short mult = iniClu.getMultiplicity();
  std::vector<std::vector<float>> eInClusters(mult, std::vector<float>(nMax));
  uint32_t firstCE = iniClu.getFirstCluEl();
  uint32_t lastCE = iniClu.getLastCluEl();

  mProp.reserve(mult * nMax);

  for (int iclu = nMax; iclu--;) {
    CluElement& ce = cluelements[mMaxAt[iclu]];
    mxMax[iclu] = ce.localX;
    mzMax[iclu] = ce.localZ;
    meMax[iclu] = ce.energy;
    mxMaxPrev[iclu] = mxMax[iclu];
    mzMaxPrev[iclu] = mzMax[iclu];
  }

  TMatrixDSym B(nMax);
  TVectorD C(nMax);
  TDecompBK bk(nMax);

  // Try to decompose cluster to contributions
  int nIterations = 0;
  bool insuficientAccuracy = true;
  double chi2Previous = 1.e+6;
  double step = 0.2;
  while (insuficientAccuracy && nIterations < o2::phos::PHOSSimParams::Instance().mNMaxIterations) {
    insuficientAccuracy = false; // will be true if at least one parameter changed too much
    B.Zero();
    C.Zero();
    mProp.clear();
    double chi2 = 0.;
    for (int iclu = nMax; iclu--;) {
      mA[iclu] = 0;
      mxB[iclu] = 0;
      mzB[iclu] = 0;
    }
    // Fill matrix and vector
    for (uint32_t idig = firstCE; idig < lastCE; idig++) {
      CluElement& ce = cluelements[idig];
      double sumA = 0.;
      for (int iclu = nMax; iclu--;) {
        double lx = ce.localX - mxMax[iclu];
        double lz = ce.localZ - mzMax[iclu];
        double r2 = lx * lx + lz * lz;
        double deriv = 0;
        double ss = showerShape(r2, deriv);
        mfij[iclu] = ss;
        mfijr[iclu] = deriv;
        mfijx[iclu] = deriv * ce.localX; // derivatives
        mfijz[iclu] = deriv * ce.localZ;
        sumA += ss * meMax[iclu];
        C(iclu) += ce.energy * ss;
      }
      double dE = ce.energy - sumA;
      chi2 += dE * dE;
      for (int iclu = 0; iclu < nMax; iclu++) {
        for (int jclu = iclu; jclu < nMax; jclu++) {
          B(iclu, jclu) += mfij[iclu] * mfij[jclu];
        }
        mA[iclu] += mfijr[iclu] * dE;
        mxB[iclu] += mfijx[iclu] * dE;
        mzB[iclu] += mfijz[iclu] * dE;
        mProp[(idig - firstCE) * nMax + iclu] = mfij[iclu] * meMax[iclu] / sumA;
      }
    }
    if (nIterations > 0 && chi2 > chi2Previous) { // too big step
      step = 0.5 * step;
      for (int iclu = nMax; iclu--;) {
        mxMax[iclu] = mxMaxPrev[iclu] + step * mdx[iclu];
        mzMax[iclu] = mzMaxPrev[iclu] + step * mdz[iclu];
      }
      nIterations++;
      insuficientAccuracy = true;
      continue;
    }
    // Good iteration, move further
    step = 0.2;
    chi2Previous = chi2;
    for (int iclu = nMax; iclu--;) {
      mxMaxPrev[iclu] = mxMax[iclu];
      mzMaxPrev[iclu] = mzMax[iclu];
    }

    // calculate next step using derivative
    // fill remaning part of B
    for (int iclu = 1; iclu < nMax; iclu++) {
      for (int jclu = 0; jclu < iclu; jclu++) {
        B(iclu, jclu) = B(jclu, iclu);
      }
    }
    for (int iclu = nMax; iclu--;) {
      if (mA[iclu] != 0) {
        mdx[iclu] = mxB[iclu] / mA[iclu] - mxMaxPrev[iclu];
        mdz[iclu] = mzB[iclu] / mA[iclu] - mzMaxPrev[iclu];
      }
    }

    for (int iclu = nMax; iclu--;) {
      // a-la Fletcher-Rivs algorithm
      mdx[iclu] += 0.2 * mdxprev[iclu];
      mdz[iclu] += 0.2 * mdzprev[iclu];
      mdxprev[iclu] = mdx[iclu];
      mdzprev[iclu] = mdz[iclu];
      insuficientAccuracy |= fabs(step * mdx[iclu]) > o2::phos::PHOSSimParams::Instance().mUnfogingXZAccuracy;
      insuficientAccuracy |= fabs(step * mdz[iclu]) > o2::phos::PHOSSimParams::Instance().mUnfogingXZAccuracy;
      mxMax[iclu] = mxMaxPrev[iclu] + step * mdx[iclu];
      mzMax[iclu] = mzMaxPrev[iclu] + step * mdz[iclu];
    }
    // now exact solution for amplitudes
    bk.SetMatrix(B);
    if (bk.Decompose()) {
      if (bk.Solve(C)) {
        for (int iclu = 0; iclu < nMax; iclu++) {
          meMax[iclu] = C(iclu);
          // double eOld = meMax[iclu];
          // insuficientAccuracy|=fabs(meMax[iclu]-eOld)> meMax[iclu]*o2::phos::PHOSSimParams::Instance().mUnfogingEAccuracy ;
        }
      } else {
        //        LOG(warning) << "Failed to decompose matrix of size " << int(nMax) << " Clusters mult=" << lastCE-firstCE;
      }
    } else {
      //      LOG(warning) << "Failed to decompose matrix of size " << int(nMax);
    }
    insuficientAccuracy &= (chi2 > o2::phos::PHOSSimParams::Instance().mUnfogingChi2Accuracy * nMax);
    nIterations++;
  }

  // Iterations finished, put first new cluster into place of mother one, others to the end of list
  for (int iclu = 0; iclu < nMax; iclu++) {
    // copy cluElements to the final list
    int start = cluelements.size();
    int nce = 0;
    for (uint32_t idig = firstCE; idig < lastCE; idig++) {
      CluElement& el = cluelements[idig];
      float ei = el.energy * mProp[(idig - firstCE) * nMax + iclu];
      if (ei > o2::phos::PHOSSimParams::Instance().mDigitMinEnergy) {
        cluelements.emplace_back(el);
        cluelements.back().energy = ei;
        cluelements.back().fraction = mProp[(idig - firstCE) * nMax + iclu];
        nce++;
      }
    }
    if (iclu == 0) { // replace parent
      iniClu.setNExMax(nMax);
      iniClu.setFirstCluEl(start);
      iniClu.setLastCluEl(start + nce);
      evalAll(iniClu, cluelements);
      if (iniClu.getEnergy() < 1.e-4) { // remove cluster and belonging to it elements
        for (int i = iniClu.getMultiplicity(); i--;) {
          cluelements.pop_back();
        }
        clusters.pop_back();
      }
    } else {
      clusters.emplace_back();
      Cluster& clu = clusters.back();
      clu.setNExMax(nMax);
      clu.setFirstCluEl(start);
      clu.setLastCluEl(start + nce);
      evalAll(clu, cluelements);
      if (clu.getEnergy() < 1.e-4) { // remove cluster and belonging to it elements
        for (int i = clu.getMultiplicity(); i--;) {
          cluelements.pop_back();
        }
        clusters.pop_back();
      }
    }
  }
}

//____________________________________________________________________________
void Clusterer::evalLabels(std::vector<Cluster>& clusters, std::vector<CluElement>& cluElements,
                           const o2::dataformats::MCTruthContainer<MCLabel>* dmc,
                           o2::dataformats::MCTruthContainer<MCLabel>& cluMC)
{

  int labelIndex = cluMC.getIndexedSize();
  auto clu = clusters.begin();

  while (clu != clusters.end()) {
    // Calculate list of primaries
    // loop over entries in digit MCTruthContainer
    for (uint32_t id = clu->getFirstCluEl(); id < clu->getLastCluEl(); id++) {
      CluElement& ll = cluElements[id];
      int i = ll.label; // index
      float sc = ll.fraction;
      gsl::span<const MCLabel> spDigList = dmc->getLabels(i);
      if (spDigList.size() == 0 || spDigList.begin()->isFake()) {
        continue;
      }
      gsl::span<MCLabel> spCluList = cluMC.getLabels(labelIndex); // get updated list
      auto digL = spDigList.begin();
      while (digL != spDigList.end()) {
        if (digL->isFake()) {
          digL++;
          continue;
        }
        bool merged = false;
        auto cluL = spCluList.begin();
        while (cluL != spCluList.end()) {
          if (*digL == *cluL) {
            (*cluL).add(*digL, sc);
            merged = true;
            break;
          }
          ++cluL;
        }
        if (!merged) { // just add label
          if (sc == 1.) {
            cluMC.addElement(labelIndex, (*digL));
          } else { // rare case of unfolded clusters
            MCLabel tmpL = (*digL);
            tmpL.scale(sc);
            cluMC.addElement(labelIndex, tmpL);
          }
        }
        ++digL;
      }
    }
    labelIndex++;
    ++clu;
  }
}
//____________________________________________________________________________
double Clusterer::showerShape(double r2, double& deriv)
{
  // Shape of the shower (see PHOS TDR)
  // we neglect dependence on the incident angle.

  // const float width = 1. / (2. * 2.32 * 2.32 * 2.32 * 2.32 * 2.32 * 2.32);
  // const float width = 1. / (2. * 2.32 * 2.32 * 2.32 * 2.32 );
  // return TMath::Exp(-r2 * r2 * width);
  if (r2 == 0.) {
    deriv = 0;
    return 1.;
  }
  double r4 = r2 * r2;
  double r295 = TMath::Power(r2, 2.95 / 2.);
  double a = 2.32 + 0.26 * r4;
  double b = 31.645570 + 2.0632911 * r295;
  double s = TMath::Exp(-r4 * ((a + b) / (a * b)));
  deriv = -2. * s * r2 * (2.32 / (a * a) + (0.54161392 * r295 + 31.645570) / (b * b));
  return s;
}

//____________________________________________________________________________
void Clusterer::evalAll(Cluster& clu, std::vector<CluElement>& cluel) const
{
  // position, energy, coreEnergy, dispersion, time,

  // Calculates the center of gravity in the local PHOS-module coordinates
  // Note that correction for non-perpendicular incidence will be applied later
  // when vertex will be known.
  float fullEnergy = 0.;
  float time = 0.;
  float eMax = 0.;
  uint32_t iFirst = clu.getFirstCluEl(), iLast = clu.getLastCluEl();
  clu.setModule(Geometry::absIdToModule(cluel[iFirst].absId));
  float eMin = o2::phos::PHOSSimParams::Instance().mDigitMinEnergy;
  for (uint32_t i = iFirst; i < iLast; i++) {
    float ei = cluel[i].energy;
    if (ei < eMin) {
      continue;
    }
    fullEnergy += ei;
    if (ei > eMax) {
      time = cluel[i].time;
      eMax = ei;
    }
  }
  clu.setEnergy(fullEnergy);
  if (fullEnergy <= 0) {
    return;
  }
  // Calculate time as time in the digit with maximal energy
  clu.setTime(time);

  float localPosX = 0., localPosZ = 0.;
  float wtot = 0.;
  float invE = 1. / fullEnergy;
  for (uint32_t i = iFirst; i < iLast; i++) {
    CluElement& ce = cluel[i];
    if (ce.energy < eMin) {
      continue;
    }
    float w = std::max(0.f, o2::phos::PHOSSimParams::Instance().mLogWeight + std::log(ce.energy * invE));
    localPosX += ce.localX * w;
    localPosZ += ce.localZ * w;
    wtot += w;
  }
  if (wtot > 0) {
    wtot = 1. / wtot;
    localPosX *= wtot;
    localPosZ *= wtot;
  }
  clu.setLocalPosition(localPosX, localPosZ);

  // Dispersion, core energy
  float coreRadius2 = o2::phos::PHOSSimParams::Instance().mCoreR;
  coreRadius2 *= coreRadius2;
  float coreE = 0.;
  float dispersion = 0.;
  float dxx = 0., dxz = 0., dzz = 0., lambdaLong = 0., lambdaShort = 0.;
  for (uint32_t i = iFirst; i < iLast; i++) {
    CluElement& ce = cluel[i];
    float ei = ce.energy;
    if (ei < eMin) {
      continue;
    }
    float x = ce.localX - localPosX;
    float z = ce.localZ - localPosZ;
    float distance = x * x + z * z;

    float w = std::max(0.f, o2::phos::PHOSSimParams::Instance().mLogWeight + std::log(ei * invE));
    dispersion += w * distance;
    dxx += w * x * x;
    dzz += w * z * z;
    dxz += w * x * z;
    if (distance < coreRadius2) {
      coreE += ei;
    }
  }
  clu.setCoreEnergy(coreE);
  // dispersion NB! wtot here already inverse
  dispersion *= wtot;

  dxx *= wtot;
  dzz *= wtot;
  dxz *= wtot;

  lambdaLong = 0.5 * (dxx + dzz) + std::sqrt(0.25 * (dxx - dzz) * (dxx - dzz) + dxz * dxz);
  if (lambdaLong > 0) {
    lambdaLong = std::sqrt(lambdaLong);
  }

  lambdaShort = 0.5 * (dxx + dzz) - std::sqrt(0.25 * (dxx - dzz) * (dxx - dzz) + dxz * dxz);
  if (lambdaShort > 0) { // To avoid exception if numerical errors lead to negative lambda.
    lambdaShort = std::sqrt(lambdaShort);
  } else {
    lambdaShort = 0.;
  }

  if (dispersion >= 0) {
    clu.setDispersion(std::sqrt(dispersion));
  } else {
    clu.setDispersion(0.);
  }
  clu.setElipsAxis(lambdaShort, lambdaLong);

  // Test trigger
  char relId[3];
  Geometry::relPosToRelId(clu.module(), localPosX, localPosZ, relId);

  for (auto& trd : mTrigger) {
    char trurelid[3];
    short trtype = trd.is2x2Tile() ? 0 : 1;
    Geometry::truAbsToRelNumbering(trd.getAbsId(), trtype, trurelid);

    // Trigger tile coordinates of lower left corner (smallest x,z)
    int dx = relId[1] - trurelid[1];
    int dz = relId[2] - trurelid[2];
    if (trtype == 0) { // 2x2
      if (dx >= 0 && dx < 2 && dz >= 0 && dz < 2) {
        clu.setFiredTrigger(trd.isHighGain());
        break;
      }
    } else { // 4x4
      if (dx >= 0 && dx < 4 && dz >= 0 && dz < 4) {
        clu.setFiredTrigger(trd.isHighGain());
        break;
      }
    }
  }
}
//____________________________________________________________________________
char Clusterer::getNumberOfLocalMax(Cluster& clu, std::vector<CluElement>& cluel)
{
  // Calculates the number of local maxima in the cluster using LocalMaxCut as the minimum
  // energy difference between maximum and surrounding digits

  float locMaxCut = o2::phos::PHOSSimParams::Instance().mLocalMaximumCut;
  float cluSeed = o2::phos::PHOSSimParams::Instance().mClusteringThreshold;
  mIsLocalMax.clear();
  mIsLocalMax.reserve(clu.getMultiplicity());

  uint32_t iFirst = clu.getFirstCluEl(), iLast = clu.getLastCluEl();
  for (uint32_t i = iFirst; i < iLast; i++) {
    mIsLocalMax.push_back(cluel[i].energy > cluSeed);
  }

  for (uint32_t i = iFirst; i < iLast - 1; i++) {
    for (uint32_t j = i + 1; j < iLast; j++) {

      if (Geometry::areNeighbours(cluel[i].absId, cluel[j].absId) == 1) {
        if (cluel[i].energy > cluel[j].energy) {
          mIsLocalMax[j - iFirst] = false;
          // but may be digit too is not local max ?
          if (cluel[j].energy > cluel[i].energy - locMaxCut) {
            mIsLocalMax[i - iFirst] = false;
          }
        } else {
          mIsLocalMax[i - iFirst] = false;
          // but may be digitN is not local max too?
          if (cluel[i].energy > cluel[j].energy - locMaxCut) {
            mIsLocalMax[j - iFirst] = false;
          }
        }
      } // if areneighbours
    }   // digit j
  }     // digit i

  int iDigitN = 0;
  for (std::size_t i = 0; i < mIsLocalMax.size(); i++) {
    if (mIsLocalMax[i]) {
      mMaxAt[iDigitN] = i + iFirst;
      iDigitN++;
      if (iDigitN >= NLOCMAX) { // Note that size of output arrays is limited:
        static int nAlarms = 0;
        if (nAlarms++ < 5) {
          LOG(alarm) << "Too many local maxima, cluster multiplicity " << mIsLocalMax.size();
        }
        return -2;
      }
    }
  }

  return iDigitN;
}
