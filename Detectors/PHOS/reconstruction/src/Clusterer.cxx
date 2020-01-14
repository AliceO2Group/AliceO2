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
void Clusterer::process(const std::vector<Digit>* digits, const std::vector<TriggerRecord>* dtr,
                        const o2::dataformats::MCTruthContainer<MCLabel>* dmc,
                        std::vector<Cluster>* clusters, std::vector<TriggerRecord>* trigRec,
                        o2::dataformats::MCTruthContainer<MCLabel>* cluMC)
{

  clusters->clear(); //final out list of clusters
  trigRec->clear();
  cluMC->clear();

  for (const auto& tr : (*dtr)) {
    mFirstDigitInEvent = tr.getFirstEntry();
    mLastDigitInEvent = mFirstDigitInEvent + tr.getNumberOfObjects();
    int indexStart = clusters->size();
    mClusters.clear(); // internal list of FullClusters

    if (!mBadMap) {
      long bcTime = 1; //TODO!!! Convert BC time to time o2::InteractionRecord bcTime = digitsTR.front().getBCData() ;
      if (o2::phos::PHOSSimParams::Instance().mCCDBPath.compare("localtest") == 0) {
        mBadMap = new BadChannelMap(1);    // test default map
        mCalibParams = new CalibParams(1); //test calibration map
        LOG(INFO) << "No reading BadMap/Calibration from ccdb requested, set default";
      } else {
        LOG(INFO) << "Getting BadMap object from ccdb";
        o2::ccdb::CcdbApi ccdb;
        std::map<std::string, std::string> metadata; // do we want to store any meta data?
        ccdb.init("http://ccdb-test.cern.ch:8080");  // or http://localhost:8080 for a local installation
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

    LOG(DEBUG) << "Number of PHOS clusters" << clusters->size();
    // Unfold overlapped clusters
    // Split clusters with several local maxima if necessary
    if (o2::phos::PHOSSimParams::Instance().mUnfoldClusters) {
      makeUnfoldings(digits);
    }
    // Calculate properties of collected clusters (Local position, energy, disp etc.)
    evalCluProperties(digits, clusters, dmc, cluMC);

    trigRec->emplace_back(tr.getBCData(), indexStart, clusters->size());
  }
  LOG(DEBUG) << "PHOS clustrization done";
}
//____________________________________________________________________________
void Clusterer::makeClusters(const std::vector<Digit>* digits)
{
  // A cluster is defined as a list of neighbour digits

  // Mark all digits as unused yet
  const int maxNDigits = 12546; // There is no digits more than in PHOS modules ;)
  bool digitsUsed[maxNDigits];

  for (int i = 0; i < maxNDigits; i++) {
    digitsUsed[i] = false;
  }
  int iFirst = mFirstDigitInEvent; // first index of digit which potentially can be a part of cluster

  for (int i = iFirst; i < mLastDigitInEvent; i++) {
    if (digitsUsed[i - mFirstDigitInEvent])
      continue;

    const Digit* digitSeed = &(digits->at(i));
    float digitSeedEnergy = calibrate(digitSeed->getAmplitude(), digitSeed->getAbsId());
    if (isBadChannel(digitSeed->getAbsId())) {
      digitSeedEnergy = 0.;
    }
    if (digitSeedEnergy < o2::phos::PHOSSimParams::Instance().mDigitMinEnergy) {
      continue;
    }

    // is this digit so energetic that start cluster?
    FullCluster* clu = nullptr;
    int iDigitInCluster = 0;
    if (digitSeedEnergy > o2::phos::PHOSSimParams::Instance().mClusteringThreshold) {
      // start a new EMC RecPoint
      FullCluster cluTmp(digitSeed->getAbsId(), digitSeedEnergy, calibrateT(digitSeed->getTime(), digitSeed->getAbsId(), digitSeed->isHighGain()),
                         digitSeed->getLabel(), 1.);
      mClusters.push_back(cluTmp);
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
        const Digit* digitN = &(digits->at(j));
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
void Clusterer::makeUnfoldings(const std::vector<Digit>* digits)
{
  //Split cluster if several local maxima are found

  int* maxAt = new int[o2::phos::PHOSSimParams::Instance().mNLMMax]; // NLMMax:Maximal number of local maxima
  float* maxAtEnergy = new float[o2::phos::PHOSSimParams::Instance().mNLMMax];

  int numberOfNotUnfolded = mClusters.size();

  auto clu = mClusters.begin();

  for (int i = 0; i < numberOfNotUnfolded; i++) {
    if (clu->getNExMax() > -1) { //already unfolded
      continue;
    }
    char nMultipl = clu->getMultiplicity();
    char nMax = clu->getNumberOfLocalMax(maxAt, maxAtEnergy);
    if (nMax > 1) {
      unfoldOneCluster(&(*clu), nMax, maxAt, maxAtEnergy, digits);

      clu = mClusters.erase(clu); //remove current and move iterator to next cluster
    } else {
      clu->setNExMax(1); // Only one local maximum
      clu++;
    }
  }

  delete[] maxAt;
  delete[] maxAtEnergy;
}
//____________________________________________________________________________
void Clusterer::unfoldOneCluster(FullCluster* iniClu, char nMax, int* digitId, float* maxAtEnergy, const std::vector<Digit>* digits)
{
  // Performs the unfolding of a cluster with nMax overlapping showers
  // Parameters: iniClu cluster to be unfolded
  //             nMax number of local maxima found (this is the number of new clusters)
  //             digitId: index of digits, corresponding to local maxima
  //             maxAtEnergy: energies of digits, corresponding to local maxima

  // Take initial cluster and calculate local coordinates of digits
  // To avoid multiple re-calculation of same parameters
  char mult = iniClu->getMultiplicity();
  std::vector<float> x(mult);
  std::vector<float> z(mult);
  std::vector<float> e(mult);
  std::vector<float> t(mult);
  std::vector<int> lbl(mult);
  std::vector<std::vector<float>> eInClusters(mult, std::vector<float>(nMax));

  const std::vector<float>* cluElist = iniClu->getEnergyList();
  // gets the list of energies of digits making this recpoint

  for (int idig = 0; idig < mult; idig++) {
    short absID = iniClu->getDigitAbsId(idig);
    float eDigit = cluElist->at(idig);
    e[idig] = eDigit;
    float lx, lz;
    mPHOSGeom->absIdToRelPosInModule(absID, lx, lz);
    x[idig] = lx;
    z[idig] = lz;
    //Extract time from digits: first find digit in list
    Digit testDigit(absID, 0., 0., 0.);
    auto dIter = std::lower_bound(digits->begin(), digits->end(), testDigit); //finds first (and the only) digit with same absId. Binary search is used
    if (dIter != digits->end()) {
      t[idig] = (*dIter).getTime();
      lbl[idig] = (*dIter).getLabel();
    }
  }

  // Coordinates of centers of clusters
  std::vector<float> xMax(nMax);
  std::vector<float> zMax(nMax);
  std::vector<float> eMax(nMax);

  for (int iclu = 0; iclu < nMax; iclu++) {
    xMax[iclu] = x[digitId[iclu]];
    zMax[iclu] = z[digitId[iclu]];
    eMax[iclu] = e[digitId[iclu]];
  }

  std::vector<float> prop(nMax); // proportion of clusters in the current digit

  // Try to decompose cluster to contributions
  int nIterations = 0;
  bool insuficientAccuracy = true;
  while (insuficientAccuracy && nIterations < o2::phos::PHOSSimParams::Instance().mNMaxIterations) {
    // Loop over all digits of parent cluster and split their energies between daughter clusters
    // according to shower shape
    for (int idig = 0; idig < mult; idig++) {
      float eEstimated = 0;
      for (int iclu = 0; iclu < nMax; iclu++) {
        prop[iclu] = eMax[iclu] * showerShape(x[idig] - xMax[iclu],
                                              z[idig] - zMax[iclu]);
        eEstimated += prop[iclu];
      }
      if (eEstimated == 0.) { // numerical accuracy
        continue;
      }
      // Split energy of digit according to contributions
      for (int iclu = 0; iclu < nMax; iclu++) {
        eInClusters[idig][iclu] = e[idig] * prop[iclu] / eEstimated;
      }
    }

    // Recalculate parameters of clusters and check relative variation of energy and absolute of position
    insuficientAccuracy = false; // will be true if at least one parameter changed too much
    for (int iclu = 0; iclu < nMax; iclu++) {
      float oldX = xMax[iclu];
      float oldZ = zMax[iclu];
      float oldE = eMax[iclu];
      // new energy, need for weight
      eMax[iclu] = 0;
      for (int idig = 0; idig < mult; idig++) {
        eMax[iclu] += eInClusters[idig][iclu];
      }
      xMax[iclu] = 0;
      zMax[iclu] = 0.;
      float wtot = 0.;
      for (int idig = 0; idig < mult; idig++) {
        float w = std::max(std::log(eInClusters[idig][iclu] / eMax[iclu]) + o2::phos::PHOSSimParams::Instance().mLogWeight, float(0.));
        xMax[iclu] += x[idig] * w;
        zMax[iclu] += z[idig] * w;
        wtot += w;
      }
      if (wtot > 0.) {
        xMax[iclu] /= wtot;
        zMax[iclu] /= wtot;
      }
      // Compare variation of parameters
      insuficientAccuracy += (std::abs(eMax[iclu] - oldE) > o2::phos::PHOSSimParams::Instance().mUnfogingEAccuracy);
      insuficientAccuracy += (std::abs(xMax[iclu] - oldX) > o2::phos::PHOSSimParams::Instance().mUnfogingXZAccuracy);
      insuficientAccuracy += (std::abs(zMax[iclu] - oldZ) > o2::phos::PHOSSimParams::Instance().mUnfogingXZAccuracy);
    }
    nIterations++;
  }
  // Iterations finished, add new clusters
  auto lastInList = mClusters.end();
  lastInList--; // last cluster before adding new
  for (int iclu = 0; iclu < nMax; iclu++) {
    mClusters.emplace_back();
    FullCluster clu = mClusters.back();
    clu.setNExMax(nMax);

    for (int idig = 0; idig < mult; idig++) {
      float eDigit = eInClusters[idig][iclu];
      if (eDigit < o2::phos::PHOSSimParams::Instance().mDigitMinEnergy) {
        continue;
      }
      int absID = iniClu->getDigitAbsId(idig);
      clu.addDigit(absID, eDigit, t[idig], lbl[idig], eDigit / e[idig]);
    }
  }
}

//____________________________________________________________________________
void Clusterer::evalCluProperties(const std::vector<Digit>* digits, std::vector<Cluster>* clusters,
                                  const o2::dataformats::MCTruthContainer<MCLabel>* dmc,
                                  o2::dataformats::MCTruthContainer<MCLabel>* cluMC)
{
  LOG(DEBUG) << "EvalCluProperties: nclu=" << mClusters.size();

  if (clusters->capacity() - clusters->size() < mClusters.size()) { //avoid expanding vector per element
    clusters->reserve(clusters->size() + mClusters.size());
  }

  int labelIndex = 0;
  if (cluMC) {
    labelIndex = cluMC->getIndexedSize();
  }
  auto clu = mClusters.begin();

  while (clu != mClusters.end()) {

    clu->purify(o2::phos::PHOSSimParams::Instance().mDigitMinEnergy);

    //  LOG(DEBUG) << "Purify done";
    clu->evalAll(digits);

    if (clu->getEnergy() > 1.e-4) { //Non-empty cluster
      clusters->emplace_back(*clu);

      if (cluMC) { //Handle labels
        //Calculate list of primaries
        //loop over entries in digit MCTruthContainer
        const std::vector<std::pair<int, float>>* vl = clu->getLabels();
        auto ll = vl->begin();
        while (ll != vl->end()) {
          int i = (*ll).first; //index
          float scale = (*ll).second;
          if (i < 0)
            continue;
          gsl::span<const MCLabel> spDigList = dmc->getLabels(i);
          gsl::span<MCLabel> spCluList = cluMC->getLabels(labelIndex); //get updated list
          auto digL = spDigList.begin();
          while (digL != spDigList.end()) {
            bool merged = false;
            auto cluL = spCluList.begin();
            while (cluL != spCluList.end()) {
              if (*digL == *cluL) {
                (*cluL).add(*digL, scale);
                merged = true;
                break;
              }
              cluL++;
            }
            if (!merged) { //just add label
              if (scale == 1.) {
                cluMC->addElement(labelIndex, (*digL));
              } else { //rare case of unfolded clusters
                MCLabel tmpL = (*digL);
                tmpL.scale(scale);
                cluMC->addElement(labelIndex, tmpL);
              }
            }
            digL++;
          }
          ll++;
        }
        clusters->back().setLabel(labelIndex);
        labelIndex++;
      } // Work with MC
    }

    clu++;
  }
}
//____________________________________________________________________________
float Clusterer::showerShape(float x, float z)
{
  // Shape of the shower (see PHOS TDR)
  // If you change this function, change also the gradient evaluation in ChiSquare()

  //for the moment we neglect dependence on the incident angle.

  float r2 = x * x + z * z;
  float r4 = r2 * r2;
  float r295 = TMath::Power(r2, 2.95 / 2.);
  float shape = TMath::Exp(-r4 * (1. / (2.32 + 0.26 * r4) + 0.0316 / (1 + 0.0652 * r295)));
  return shape;
}
//____________________________________________________________________________
float Clusterer::calibrate(float amp, short absId)
{
  return amp * mCalibParams->getGain(absId);
}
//Calibrate energy
//____________________________________________________________________________
float Clusterer::calibrateT(float time, short absId, bool isHG)
{
  //Calibrate time
  if (isHG) {
    return time - mCalibParams->getHGTimeCalib(absId);
  } else {
    return time - mCalibParams->getLGTimeCalib(absId);
  }
}
//____________________________________________________________________________
bool Clusterer::isBadChannel(short absId)
{
  return (!mBadMap->isChannelGood(absId));
}
