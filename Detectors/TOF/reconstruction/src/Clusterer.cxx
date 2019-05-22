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
/// \brief Implementation of the TOF cluster finder
#include <algorithm>
#include "FairLogger.h" // for LOG
#include "DataFormatsTOF/Cluster.h"
#include "TOFReconstruction/Clusterer.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include <TStopwatch.h>

using namespace o2::tof;

//__________________________________________________
void Clusterer::process(DataReader& reader, std::vector<Cluster>& clusters, MCLabelContainer const* digitMCTruth)
{
  TStopwatch timerProcess;
  timerProcess.Start();

  reader.init();
  int totNumDigits = 0;

  while (reader.getNextStripData(mStripData)) {
    LOG(DEBUG) << "TOFClusterer got Strip " << mStripData.stripID << " with Ndigits "
               << mStripData.digits.size();
    totNumDigits += mStripData.digits.size();

    processStrip(clusters, digitMCTruth);
  }

  LOG(DEBUG) << "We had " << totNumDigits << " digits in this event";
  timerProcess.Stop();
  printf("Timing:\n");
  printf("Clusterer::process:        ");
  timerProcess.Print();
}

//__________________________________________________
void Clusterer::processStrip(std::vector<Cluster>& clusters, MCLabelContainer const* digitMCTruth)
{
  // method to clusterize the current strip

  Int_t detId[5];
  Int_t chan, chan2, chan3;
  Int_t strip1, strip2;
  Int_t iphi, iphi2, iphi3;
  Int_t ieta, ieta2, ieta3; // it is the number of padz-row increasing along the various strips

  for (int idig = 0; idig < mStripData.digits.size(); idig++) {
    //    LOG(DEBUG) << "Checking digit " << idig;
    Digit* dig = &mStripData.digits[idig];
    if (dig->isUsedInCluster())
      continue; // the digit was already used to build a cluster

    mNumberOfContributingDigits = 0;
    dig->getPhiAndEtaIndex(iphi, ieta);
    if (mStripData.digits.size() > 1)
      LOG(DEBUG) << "idig = " << idig;

    // first we make a cluster out of the digit
    int noc = clusters.size();
    //    LOG(DEBUG) << "noc = " << noc << "\n";
    clusters.emplace_back();
    Cluster& c = clusters[noc];
    addContributingDigit(dig);
    float timeDig = dig->getTDC() * Geo::TDCBIN;

    for (int idigNext = idig + 1; idigNext < mStripData.digits.size(); idigNext++) {
      Digit* digNext = &mStripData.digits[idigNext];
      if (digNext->isUsedInCluster())
        continue; // the digit was already used to build a cluster
      // check if the TOF time are close enough to be merged; if not, it means that nothing else will contribute to the cluster (since digits are ordered in time)
      float timeDigNext = digNext->getTDC() * Geo::TDCBIN; // we assume it calibrated (for now); in ps
      LOG(DEBUG) << "Time difference = " << timeDigNext - timeDig;
      if (timeDigNext - timeDig > 500 /*in ps*/)
        break;
      digNext->getPhiAndEtaIndex(iphi2, ieta2);

      // check if the fired pad are close in space
      LOG(DEBUG) << "phi difference = " << iphi - iphi2;
      LOG(DEBUG) << "eta difference = " << ieta - ieta2;
      if ((TMath::Abs(iphi - iphi2) > 1) || (TMath::Abs(ieta - ieta2) > 1))
        continue;

      // if we are here, the digit contributes to the cluster
      addContributingDigit(digNext);

    } // loop on the second digit

    buildCluster(c, digitMCTruth);

  } // loop on the first digit
}
//______________________________________________________________________
void Clusterer::addContributingDigit(Digit* dig)
{

  // adding a digit to the array that stores the contributing ones

  if (mNumberOfContributingDigits == 6) {
    LOG(ERROR) << "The cluster has already 6 digits associated to it, we cannot add more; returning without doing anything";

    int phi, eta;
    for (int i = 0; i < mNumberOfContributingDigits; i++) {
      mContributingDigit[i]->getPhiAndEtaIndex(phi, eta);
      LOG(ERROR) << "digit already in " << i << ", channel = " << mContributingDigit[i]->getChannel() << ",phi,eta = (" << phi << "," << eta << "), TDC = " << mContributingDigit[i]->getTDC();
    }

    dig->getPhiAndEtaIndex(phi, eta);
    LOG(ERROR) << "skipped digit"
               << ", channel = " << dig->getChannel() << ",phi,eta = (" << phi << "," << eta << "), TDC = " << dig->getTDC();

    dig->setIsUsedInCluster(); // flag is at used in any case

    return;
  }
  mContributingDigit[mNumberOfContributingDigits] = dig;
  mNumberOfContributingDigits++;
  dig->setIsUsedInCluster();

  return;
}

//_____________________________________________________________________
void Clusterer::buildCluster(Cluster& c, MCLabelContainer const* digitMCTruth)
{
  static const float inv12 = 1. / 12.;

  // here we finally build the cluster from all the digits contributing to it

  Digit* temp;
  for (int idig = 1; idig < mNumberOfContributingDigits; idig++) {
    // the digit[0] will be the main one
    if (mContributingDigit[idig]->getTOT() > mContributingDigit[0]->getTOT()) {
      temp = mContributingDigit[0];
      mContributingDigit[0] = mContributingDigit[idig];
      mContributingDigit[idig] = temp;
    }
  }

  c.setMainContributingChannel(mContributingDigit[0]->getChannel());
  c.setTime(mContributingDigit[0]->getTDC() * Geo::TDCBIN + double(mContributingDigit[0]->getBC() * 25000.)); // time in ps (for now we assume it calibrated)
  c.setTimeRaw(mContributingDigit[0]->getTDC() * Geo::TDCBIN + double(mContributingDigit[0]->getBC() * 25000.)); // time in ps (for now we assume it calibrated)

  c.setTot(mContributingDigit[0]->getTOT() * Geo::TOTBIN * 1E-3); // TOT in ns (for now we assume it calibrated)
  //setL0L1Latency(); // to be filled (maybe)
  //setDeltaBC(); // to be filled (maybe)

  int chan1, chan2;
  int phi1, phi2;
  int eta1, eta2;
  int deltaPhi, deltaEta;
  int mask;

  mContributingDigit[0]->getPhiAndEtaIndex(phi1, eta1);
  // now set the mask with the secondary digits
  for (int idig = 1; idig < mNumberOfContributingDigits; idig++) {
    mContributingDigit[idig]->getPhiAndEtaIndex(phi2, eta2);
    deltaPhi = phi1 - phi2;
    deltaEta = eta1 - eta2;
    if (deltaPhi == 1) {   // the digit is to the LEFT of the cluster; let's check about UP/DOWN/Same Line
      if (deltaEta == 1) { // the digit is DOWN LEFT wrt the cluster
        mask = Cluster::kDownLeft;
      } else if (deltaEta == -1) { // the digit is UP LEFT wrt the cluster
        mask = Cluster::kUpLeft;
      } else { // the digit is LEFT wrt the cluster
        mask = Cluster::kLeft;
      }
    } else if (deltaPhi == -1) { // the digit is to the RIGHT of the cluster; let's check about UP/DOWN/Same Line
      if (deltaEta == 1) {       // the digit is DOWN RIGHT wrt the cluster
        mask = Cluster::kDownRight;
      } else if (deltaEta == -1) { // the digit is UP RIGHT wrt the cluster
        mask = Cluster::kUpRight;
      } else { // the digit is RIGHT wrt the cluster
        mask = Cluster::kRight;
      }
    } else if (deltaPhi == 0) { // the digit is on the same column as the cluster; is it UP or Down?
      if (deltaEta == 1) {      // the digit is DOWN wrt the cluster
        mask = Cluster::kDown;
      } else if (deltaEta == -1) { // the digit is UP wrt the cluster
        mask = Cluster::kUp;
      } else { // impossible!!
        LOG(DEBUG) << " Check what is going on, the digit you are trying to merge to the cluster must be in a different channels... ";
      }
    } else { // impossible!!! We checked above...
      LOG(DEBUG) << " Check what is going on, the digit you are trying to merge to the cluster is too far from the cluster, you should have not got here... ";
    }
    c.addBitInContributingChannels(mask);
  }

  // filling the MC labels of this cluster; the first will be those of the main digit; then the others
  if (digitMCTruth != nullptr) {
    int lbl = mClsLabels->getIndexedSize(); // this should correspond to the number of digits also;
    printf("lbl = %d\n", lbl);
    for (int i = 0; i < mNumberOfContributingDigits; i++) {
      printf("contributing digit = %d\n", i);
      int digitLabel = mContributingDigit[i]->getLabel();
      printf("digitLabel = %d\n", digitLabel);
      gsl::span<const o2::MCCompLabel> mcArray = digitMCTruth->getLabels(digitLabel);
      for (int j = 0; j < static_cast<int>(mcArray.size()); j++) {
        printf("checking element %d in the array of labels\n", j);
        auto label = digitMCTruth->getElement(digitMCTruth->getMCTruthHeader(digitLabel).index + j);
        printf("EventID = %d\n", label.getEventID());
        mClsLabels->addElement(lbl, label);
      }
    }
  }

  // set geometrical variables
  int det[5];
  Geo::getVolumeIndices(c.getMainContributingChannel(), det);
  float pos[3];
  Geo::getPos(det, pos);
  Geo::rotateToSector(pos, c.getSector());
  c.setXYZ(pos[2], pos[0], pos[1]); // storing coordinates in sector frame: note that the rotation above puts z in pos[1], the radial coordinate in pos[2], and the tangent coordinate in pos[0] (this is to match the TOF residual system, where we don't use the radial component), so we swap their positions.

  c.setR(TMath::Sqrt(pos[0] * pos[0] + pos[1] * pos[1])); // it is the R in the sector frame
  c.setPhi(TMath::ATan2(pos[1], pos[0]));

  float errY2 = Geo::XPAD * Geo::XPAD * inv12;
  float errZ2 = Geo::ZPAD * Geo::ZPAD * inv12;
  float errYZ = 0;

  //if(c.getNumOfContributingChannels() > 1){
  // set errors according to a model not yet defined!
  // TO DO
  //}
  c.setErrors(errY2, errZ2, errYZ);

  return;
}
