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
/// \brief Implementation of the PHOS cluster finder
#include <memory>

#include "PHOSReconstruction/Clusterer.h" // for LOG
#include "PHOSBase/Geometry.h"
#include "PHOSBase/PHOSSimParams.h"
#include "DataFormatsPHOS/Cluster.h"
#include "PHOSReconstruction/FullCluster.h"
#include "DataFormatsPHOS/Digit.h"
#include "CCDB/CcdbApi.h"

#include "FairLogger.h" // for LOG

using namespace o2::phos;

ClassImp(Clusterer);

//____________________________________________________________________________
void Clusterer::initialize()
{
  if (!mPHOSGeom) {
    mPHOSGeom = Geometry::GetInstance();
  }
  mFirstDigitInEvent = 0;
  mLastDigitInEvent = -1;
}
//____________________________________________________________________________
void Clusterer::process(gsl::span<const Digit> digits, gsl::span<const TriggerRecord> dtr,
                        const o2::dataformats::MCTruthContainer<MCLabel>* dmc,
                        std::vector<Cluster>* clusters, std::vector<TriggerRecord>* trigRec,
                        o2::dataformats::MCTruthContainer<MCLabel>* cluMC)
{
  clusters->clear(); //final out list of clusters
  trigRec->clear();
  cluMC->clear();

  for (const auto& tr : dtr) {
    mFirstDigitInEvent = tr.getFirstEntry();
    mLastDigitInEvent = mFirstDigitInEvent + tr.getNumberOfObjects();
    int indexStart = clusters->size();
    mClusters.clear(); // internal list of FullClusters

    LOG(DEBUG) << "Starting clusteriztion digits from " << mFirstDigitInEvent << " to " << mLastDigitInEvent;

    if (!mBadMap) {
      if (o2::phos::PHOSSimParams::Instance().mCCDBPath.compare("localtest") == 0) {
        mBadMap = new BadChannelMap(1);    // test default map
        mCalibParams = new CalibParams(1); //test calibration map
        LOG(INFO) << "No reading BadMap/Calibration from ccdb requested, set default";
      } else {
        LOG(INFO) << "Getting BadMap object from ccdb";
        o2::ccdb::CcdbApi ccdb;
        std::map<std::string, std::string> metadata; // do we want to store any meta data?
        ccdb.init("http://ccdb-test.cern.ch:8080");  // or http://localhost:8080 for a local installation
        long bcTime = 1;                             //TODO!!! Convert BC time to time o2::InteractionRecord bcTime = digitsTR.front().getBCData() ;
        mBadMap = ccdb.retrieveFromTFileAny<o2::phos::BadChannelMap>("PHOS/BadMap", metadata, bcTime);
        mCalibParams = ccdb.retrieveFromTFileAny<o2::phos::CalibParams>("PHOS/Calib", metadata, bcTime);
        if (!mBadMap) {
          LOG(FATAL) << "[PHOSCellConverter - run] can not get Bad Map";
        }
        if (!mCalibParams) {
          LOG(FATAL) << "[PHOSCellConverter - run] can not get CalibParams";
        }
      }
    }

    // Collect digits to clusters
    makeClusters(digits);

    // Unfold overlapped clusters
    // Split clusters with several local maxima if necessary
    if (o2::phos::PHOSSimParams::Instance().mUnfoldClusters) {
      makeUnfoldings(digits);
    }

    // Calculate properties of collected clusters (Local position, energy, disp etc.)
    evalCluProperties(digits, clusters, dmc, cluMC);

    LOG(DEBUG) << "Found clusters from " << indexStart << " to " << clusters->size();

    trigRec->emplace_back(tr.getBCData(), indexStart, clusters->size());
  }
}
//____________________________________________________________________________
void Clusterer::processCells(gsl::span<const Cell> cells, gsl::span<const TriggerRecord> ctr,
                             const o2::dataformats::MCTruthContainer<MCLabel>* dmc, gsl::span<const uint> mcmap,
                             std::vector<Cluster>* clusters, std::vector<TriggerRecord>* trigRec,
                             o2::dataformats::MCTruthContainer<MCLabel>* cluMC)
{
  // Transform input Cells to digits and run standard recontruction
  clusters->clear(); //final out list of clusters
  trigRec->clear();
  cluMC->clear();

  for (const auto& tr : ctr) {
    int firstCellInEvent = tr.getFirstEntry();
    int lastCellInEvent = firstCellInEvent + tr.getNumberOfObjects();
    int indexStart = clusters->size();
    mClusters.clear(); // internal list of FullClusters

    LOG(DEBUG) << "Starting clusteriztion cells from " << mFirstDigitInEvent << " to " << mLastDigitInEvent;

    if (!mBadMap) {
      if (o2::phos::PHOSSimParams::Instance().mCCDBPath.compare("localtest") == 0) {
        mBadMap = new BadChannelMap(1);    // test default map
        mCalibParams = new CalibParams(1); //test calibration map
        LOG(INFO) << "No reading BadMap/Calibration from ccdb requested, set default";
      } else {
        LOG(INFO) << "Getting BadMap object from ccdb";
        o2::ccdb::CcdbApi ccdb;
        std::map<std::string, std::string> metadata; // do we want to store any meta data?
        ccdb.init("http://ccdb-test.cern.ch:8080");  // or http://localhost:8080 for a local installation
        long bcTime = 1;                             //TODO!!! Convert BC time to time o2::InteractionRecord bcTime = digitsTR.front().getBCData() ;
        mBadMap = ccdb.retrieveFromTFileAny<o2::phos::BadChannelMap>("PHOS/BadMap", metadata, bcTime);
        mCalibParams = ccdb.retrieveFromTFileAny<o2::phos::CalibParams>("PHOS/Calib", metadata, bcTime);
        if (!mBadMap) {
          LOG(FATAL) << "[PHOSCellConverter - run] can not get Bad Map";
        }
        if (!mCalibParams) {
          LOG(FATAL) << "[PHOSCellConverter - run] can not get CalibParams";
        }
      }
    }

    convertCellsToDigits(cells, firstCellInEvent, lastCellInEvent, mcmap);

    // Collect digits to clusters
    makeClusters(mDigits);

    // Unfold overlapped clusters
    // Split clusters with several local maxima if necessary
    if (o2::phos::PHOSSimParams::Instance().mUnfoldClusters) {
      makeUnfoldings(mDigits);
    }

    // Calculate properties of collected clusters (Local position, energy, disp etc.)
    evalCluProperties(mDigits, clusters, dmc, cluMC);

    LOG(DEBUG) << "Found clusters from " << indexStart << " to " << clusters->size();

    trigRec->emplace_back(tr.getBCData(), indexStart, clusters->size());
  }
}
//____________________________________________________________________________
void Clusterer::convertCellsToDigits(gsl::span<const Cell> cells, int firstCellInEvent, int lastCellInEvent, gsl::span<const uint> mcmap)
{

  mDigits.clear();
  if (mDigits.capacity() < lastCellInEvent - firstCellInEvent) {
    mDigits.reserve(lastCellInEvent - firstCellInEvent);
  }
  int iLab = 0, nLab = mcmap.size();
  while (iLab < nLab) {
    if (mcmap[iLab] >= firstCellInEvent) {
      break;
    }
    ++iLab;
  }

  for (int i = firstCellInEvent; i < lastCellInEvent; i++) {
    const Cell c = cells[i];
    //short cell, float amplitude, float time, int label
    int label = -1;
    if (mcmap[iLab] == i) {
      label = iLab;
      ++iLab;
      if (iLab >= nLab)
        --iLab;
    }
    mDigits.emplace_back(c.getAbsId(), c.getEnergy(), c.getTime(), label);
    mDigits.back().setHighGain(c.getHighGain());
  }
  mFirstDigitInEvent = 0;
  mLastDigitInEvent = mDigits.size();
}
//____________________________________________________________________________
void Clusterer::makeClusters(gsl::span<const Digit> digits)
{
  // A cluster is defined as a list of neighbour digits

  // Mark all digits as unused yet
  const int maxNDigits = 12546; // There is no digits more than in PHOS modules ;)
  bool digitsUsed[maxNDigits];
  memset(digitsUsed, 0, sizeof(bool) * maxNDigits);

  int iFirst = mFirstDigitInEvent; // first index of digit which potentially can be a part of cluster

  for (int i = iFirst; i < mLastDigitInEvent; i++) {
    if (digitsUsed[i - mFirstDigitInEvent])
      continue;

    const Digit& digitSeed = digits[i];
    float digitSeedEnergy = calibrate(digitSeed.getAmplitude(), digitSeed.getAbsId());
    if (isBadChannel(digitSeed.getAbsId())) {
      digitSeedEnergy = 0.;
    }
    if (digitSeedEnergy < o2::phos::PHOSSimParams::Instance().mDigitMinEnergy) {
      continue;
    }

    // is this digit so energetic that start cluster?
    FullCluster* clu = nullptr;
    int iDigitInCluster = 0;
    if (digitSeedEnergy > o2::phos::PHOSSimParams::Instance().mClusteringThreshold) {
      // start new cluster
      mClusters.emplace_back(digitSeed.getAbsId(), digitSeedEnergy,
                             calibrateT(digitSeed.getTime(), digitSeed.getAbsId(), digitSeed.isHighGain()),
                             digitSeed.getLabel(), 1.);
      clu = &(mClusters.back());

      digitsUsed[i - mFirstDigitInEvent] = true;
      iDigitInCluster = 1;
    } else {
      continue;
    }
    // Now scan remaining digits in list to find neigbours of our seed
    int index = 0;
    while (index < iDigitInCluster) { // scan over digits already in cluster
      short digitSeedAbsId = clu->getDigitAbsId(index);
      index++;
      for (Int_t j = iFirst; j < mLastDigitInEvent; j++) {
        if (digitsUsed[j - mFirstDigitInEvent])
          continue; // look through remaining digits
        const Digit* digitN = &(digits[j]);
        float digitNEnergy = calibrate(digitN->getAmplitude(), digitN->getAbsId());
        if (isBadChannel(digitN->getAbsId())) { //remove digit
          digitNEnergy = 0.;
        }
        if (digitNEnergy < o2::phos::PHOSSimParams::Instance().mDigitMinEnergy) {
          continue;
        }

        // call (digit,digitN) in THAT oder !!!!!
        Int_t ineb = mPHOSGeom->areNeighbours(digitSeedAbsId, digitN->getAbsId());
        switch (ineb) {
          case -1: // too early (e.g. previous module), do not look before j at subsequent passes
            iFirst = j;
            break;
          case 0: // not a neighbour
            break;
          case 1: // are neighbours
            clu->addDigit(digitN->getAbsId(), digitNEnergy, calibrateT(digitN->getTime(), digitN->getAbsId(), digitN->isHighGain()), digitN->getLabel(), 1.);
            iDigitInCluster++;
            digitsUsed[j - mFirstDigitInEvent] = true;
            break;
          case 2: // too far from each other
          default:
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

  std::vector<int> maxAt(o2::phos::PHOSSimParams::Instance().mNLMMax); // NLMMax:Maximal number of local maxima

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
  std::vector<std::vector<float>> eInClusters(mult, std::vector<float>(nMax));
  std::vector<std::vector<float>> fij(mult, std::vector<float>(nMax));

  const std::vector<FullCluster::CluElement>* cluElist = iniClu.getElementList();

  // Coordinates of centers of clusters
  std::vector<float> xMax(nMax);
  std::vector<float> zMax(nMax);
  std::vector<float> eMax(nMax);
  std::vector<float> deNew(nMax);

  //transient variables
  std::vector<float> a(nMax);
  std::vector<float> b(nMax);
  std::vector<float> c(nMax);

  for (int iclu = 0; iclu < nMax; iclu++) {
    xMax[iclu] = (*cluElist)[digitId[iclu]].localX;
    zMax[iclu] = (*cluElist)[digitId[iclu]].localZ;
    eMax[iclu] = 2. * (*cluElist)[digitId[iclu]].energy;
  }

  std::vector<float> prop(nMax); // proportion of clusters in the current digit

  // Try to decompose cluster to contributions
  int nIterations = 0;
  bool insuficientAccuracy = true;
  while (insuficientAccuracy && nIterations < o2::phos::PHOSSimParams::Instance().mNMaxIterations) {
    insuficientAccuracy = false; // will be true if at least one parameter changed too much
    a.clear();
    b.clear();
    c.clear();
    //First calculate shower shapes
    for (int idig = 0; idig < mult; idig++) {
      auto it = (*cluElist)[idig];
      for (int iclu = 0; iclu < nMax; iclu++) {
        fij[idig][iclu] = showerShape(it.localX - xMax[iclu], it.localZ - zMax[iclu]);
      }
    }

    //Fit energies
    for (int idig = 0; idig < mult; idig++) {
      auto it = (*cluElist)[idig];
      for (int iclu = 0; iclu < nMax; iclu++) {
        a[iclu] += fij[idig][iclu] * fij[idig][iclu];
        b[iclu] += it.energy * fij[idig][iclu];
        for (int kclu = 0; kclu < nMax; kclu++) {
          if (iclu == kclu)
            continue;
          c[iclu] += eMax[kclu] * fij[idig][iclu] * fij[idig][kclu];
        }
      }
    }
    //Evaluate new maximal energies
    for (int iclu = 0; iclu < nMax; iclu++) {
      if (a[iclu] != 0.) {
        float eNew = (b[iclu] - c[iclu]) / a[iclu];
        insuficientAccuracy += (std::abs(eMax[iclu] - eNew) > eNew * o2::phos::PHOSSimParams::Instance().mUnfogingEAccuracy);
        eMax[iclu] = eNew;
      }
    } // otherwise keep old value

    // Loop over all digits of parent cluster and split their energies between daughter clusters
    // according to shower shape
    // then re-evaluate local position of clusters
    for (int idig = 0; idig < mult; idig++) {
      float eEstimated = 0;
      for (int iclu = 0; iclu < nMax; iclu++) {
        prop[iclu] = eMax[iclu] * fij[idig][iclu];
        eEstimated += prop[iclu];
      }
      if (eEstimated == 0.) { // numerical accuracy
        continue;
      }
      // Split energy of digit according to contributions
      for (int iclu = 0; iclu < nMax; iclu++) {
        eInClusters[idig][iclu] = (*cluElist)[idig].energy * prop[iclu] / eEstimated;
      }
    }

    // Recalculate parameters of clusters and check relative variation of energy and absolute of position
    for (int iclu = 0; iclu < nMax; iclu++) {
      float oldX = xMax[iclu];
      float oldZ = zMax[iclu];
      // full energy, need for weight
      float eTotNew = 0;
      for (int idig = 0; idig < mult; idig++) {
        eTotNew += eInClusters[idig][iclu];
      }
      xMax[iclu] = 0;
      zMax[iclu] = 0.;
      float wtot = 0.;
      for (int idig = 0; idig < mult; idig++) {
        if (eInClusters[idig][iclu] > 0) {
          // In unfolding it is better to use linear weight to reduce contribution of unfolded tails
          float w = eInClusters[idig][iclu] / eTotNew;
          // float w = std::max(std::log(eInClusters[idig][iclu] / eTotNew) + o2::phos::PHOSSimParams::Instance().mLogWeight, float(0.));
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
      insuficientAccuracy += (std::abs(xMax[iclu] - oldX) > o2::phos::PHOSSimParams::Instance().mUnfogingXZAccuracy);
      insuficientAccuracy += (std::abs(zMax[iclu] - oldZ) > o2::phos::PHOSSimParams::Instance().mUnfogingXZAccuracy);
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
      float eDigit = eInClusters[idig][iclu];
      idig++;
      if (eDigit < o2::phos::PHOSSimParams::Instance().mDigitMinEnergy) {
        continue;
      }
      clu.addDigit((*cluElist)[idig].absId, eDigit, (*cluElist)[idig].time, (*cluElist)[idig].label, eDigit / (*cluElist)[idig].energy);
    }
  }
}

//____________________________________________________________________________
void Clusterer::evalCluProperties(gsl::span<const Digit> digits, std::vector<Cluster>* clusters,
                                  const o2::dataformats::MCTruthContainer<MCLabel>* dmc,
                                  o2::dataformats::MCTruthContainer<MCLabel>* cluMC)
{

  if (clusters->capacity() - clusters->size() < mClusters.size()) { //avoid expanding vector per element
    clusters->reserve(clusters->size() + mClusters.size());
  }

  int labelIndex = 0;
  if (cluMC) {
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

      if (cluMC) { //Handle labels
        //Calculate list of primaries
        //loop over entries in digit MCTruthContainer
        const std::vector<FullCluster::CluElement>* vl = clu->getElementList();
        auto ll = vl->begin();
        while (ll != vl->end()) {
          int i = (*ll).label; //index
          float sc = (*ll).scale;
          if (i < 0) {
            ++ll;
            continue;
          }
          gsl::span<const MCLabel> spDigList = dmc->getLabels(i);
          gsl::span<MCLabel> spCluList = cluMC->getLabels(labelIndex); //get updated list
          auto digL = spDigList.begin();
          while (digL != spDigList.end()) {
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
            if (!merged) { //just add label
              if (sc == 1.) {
                cluMC->addElement(labelIndex, (*digL));
              } else { //rare case of unfolded clusters
                MCLabel tmpL = (*digL);
                tmpL.scale(sc);
                cluMC->addElement(labelIndex, tmpL);
              }
            }
            ++digL;
          }
          ++ll;
        }
        clusters->back().setLabel(labelIndex);
        labelIndex++;
      } // Work with MC
    }

    ++clu;
  }
}
//____________________________________________________________________________
float Clusterer::showerShape(float x, float z)
{
  // Shape of the shower (see PHOS TDR)
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
