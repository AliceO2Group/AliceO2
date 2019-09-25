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
#include "PHOSReconstruction/Cluster.h"
#include "PHOSBase/Geometry.h"
#include "PHOSBase/Digit.h"

#include "FairLogger.h" // for LOG

using namespace o2::phos;

ClassImp(Clusterer);

//____________________________________________________________________________
void Clusterer::process(const std::vector<Digit>* digits, std::vector<Cluster>* clusters)
{

  if (!mPHOSGeom) {
    mPHOSGeom = Geometry::GetInstance();
  }

  // Collect digits to clusters
  MakeClusters(digits, clusters);

  LOG(DEBUG) << "Number of PHOS clusters" << clusters->size();

  // Unfold overlapped clusters
  // TODO:
  // MakeUnfolding();

  // Calculate properties of collected clusters (Local position, energy, disp etc.)
  EvalCluProperties(digits, clusters);
  LOG(DEBUG) << "PHOS clustrization done";
}
//____________________________________________________________________________
void Clusterer::MakeClusters(const std::vector<Digit>* digits, std::vector<Cluster>* clusters)
{
  // A cluster is defined as a list of neighbour digits
  const double kClusteringThreshold = 0.050; // TODO: To be read from RecoParam
  const double kDigitMinEnergy = 0.010;      // TODO: to be implemented as a digit energy cut

  // Mark all digits as unused yet
  const int maxNDigits = 3584; // There is no clusters larger than PHOS module ;)
  int nDigits = digits->size();
  bool digitsUsed[maxNDigits];

  for (int i = 0; i < nDigits; i++) {
    digitsUsed[i] = false;
  }
  int iFirst = 0; // first index of digit which potentially can be a part of cluster
                  // e.g. first digit in this module, first CPV digit etc.

  for (int i = 0; i < nDigits; i++) {
    if (digitsUsed[i])
      continue;

    const Digit* digitSeed = &(digits->at(i));

    // is this digit so energetic that start cluster?
    Cluster* clu = nullptr;
    int iDigitInCluster = 0;
    if (digitSeed->getAmplitude() > kClusteringThreshold) {
      // start a new EMC RecPoint
      Cluster cluTmp(digitSeed->getAbsId(), digitSeed->getAmplitude(), digitSeed->getTime());
      clusters->push_back(cluTmp);
      clu = &(clusters->back());

      digitsUsed[i] = true;
      iDigitInCluster = 1;
    } else {
      continue;
    }

    // Now scan remaining digits in list to find neigbours of our seed
    int index = 0;
    while (index < iDigitInCluster) { // scan over digits already in cluster
      int digitSeedAbsId = clu->GetDigitAbsId(index);
      index++;
      for (Int_t j = iFirst; j < nDigits; j++) {
        if (digitsUsed[j])
          continue; // look through remaining digits
        const Digit* digitN = &(digits->at(j));
        ;
        // call (digit,digitN) in THAT oder !!!!!
        Int_t ineb = mPHOSGeom->AreNeighbours(digitSeedAbsId, digitN->getAbsId());
        switch (ineb) {
          case -1: // too early (e.g. previous module), do not look before j at subsequent passes
            iFirst = j;
            break;
          case 0: // not a neighbour
            break;
          case 1: // are neighbours
            clu->AddDigit(digitN->getAbsId(), digitN->getAmplitude(), digitN->getTime());
            iDigitInCluster++;
            digitsUsed[j] = true;
            break;
          case 2: // too far from each other
          default:
            break;
        } // switch
      }
    } // loop over cluster
  }   // energy theshold
}
//____________________________________________________________________________
void Clusterer::EvalCluProperties(const std::vector<Digit>* digits, std::vector<Cluster>* clusters)
{
  const double kThreshold = 0.020; // TODO: Should be in RecoParams
  LOG(DEBUG) << "EvalCluProperties: nclu=" << clusters->size();

  for (int i = 0; i < clusters->size(); i++) {
    LOG(DEBUG) << "   i=" << i;

    Cluster* clu = &(clusters->at(i));
    LOG(DEBUG) << " clu=" << clu;

    clu->Purify(kThreshold);
    //  LOG(DEBUG) << "Purify done";
    clu->EvalAll(digits);
    double PosX, PosZ;
    clu->GetLocalPosition(PosX, PosZ);
    LOG(DEBUG) << "EvalALl done: clu E=" << clu->GetEnergy() << " pos = (" << PosX << "," << PosZ << ")";
  }
}
