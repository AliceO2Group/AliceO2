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

#include <algorithm>
#include <stdexcept>
#include <cstring>
#include <vector>

#include "MCHClustering/ClusterConfig.h"
#include "MCHClustering/clusterProcessing.h"
#include "mathUtil.h"
#include "mathieson.h"
// Used to analyse the clustering/fitting
#include "InspectModel.h"

// To keep internal data
#define CHECK 1

// Type of projection
// Here add single cathode pads
// (No intersection with pads
// in other plane)
static int includeSinglePads = 1;

namespace o2
{
namespace mch
{
extern ClusterConfig clusterConfig;
}
} // namespace o2

using namespace o2::mch;

// Total number of hits/seeds (number of mathieson)
// found in the precluster;
static int nbrOfHits = 0;
// Storage of the seeds found
static struct Results_t {
  std::vector<DataBlock_t> seedList;
  // mapping pads - groups
  Groups_t* padToGroups;
} clusterResults;

// Release memory and reset the seed list
void cleanClusterResults()
{
  for (int i = 0; i < clusterResults.seedList.size(); i++) {
    delete[] clusterResults.seedList[i].second;
  }
  clusterResults.seedList.clear();
  //
  deleteShort(clusterResults.padToGroups);
}

void o2::mch::collectGroupMapping(o2::mch::Mask_t* padToMGrp, int nPads)
{

  if (clusterConfig.processingLog >= ClusterConfig::info) {
    printf("collectGroupMapping nPads=%d\n", nPads);
  }
  o2::mch::vectorCopyShort(clusterResults.padToGroups, nPads, padToMGrp);
}

void storeGroupMapping(const o2::mch::Groups_t* cath0Grp,
                       const o2::mch::PadIdx_t* mapCath0PadIdxToPadIdx, int nCath0,
                       const o2::mch::Groups_t* cath1Grp,
                       const o2::mch::PadIdx_t* mapCath1PadIdxToPadIdx, int nCath1)
{
  clusterResults.padToGroups = new Groups_t[nCath0 + nCath1];
  if (cath0Grp != nullptr) {
    for (int p = 0; p < nCath0; p++) {
      clusterResults.padToGroups[mapCath0PadIdxToPadIdx[p]] = cath0Grp[p];
    }
  }
  if (cath1Grp != nullptr) {
    for (int p = 0; p < nCath1; p++) {
      // printf("savePadToCathGroup p[cath1 idx]=%d mapCath1PadIdxToPadIdx[p]=
      // %d, grp=%d\n", p, mapCath1PadIdxToPadIdx[p], cath1Grp[p]);
      clusterResults.padToGroups[mapCath1PadIdxToPadIdx[p]] = cath1Grp[p];
    }
  }
}

void o2::mch::collectSeeds(double* theta, o2::mch::Groups_t* thetaToGroup, int K)
{
  int sumK = 0;

  // printf("collectSeeds : nbrOfGroups with clusters = %d\n", clusterResults.seedList.size());
  for (int h = 0; h < clusterResults.seedList.size(); h++) {
    int k = clusterResults.seedList[h].first;
    // if (clusterConfig.inspectModelLog >= ClusterConfig.info) {
    //  o2::mch::printTheta("  ",
    //                    clusterResults.seedList[h].second,
    //                    clusterResults.seedList[h].first);
    //}
    o2::mch::copyTheta(clusterResults.seedList[h].second, k,
                       &theta[sumK], K, k);
    if (thetaToGroup) {
      o2::mch::vectorSetShort(&thetaToGroup[sumK], h + 1, k);
    }
    sumK += k;
    // if (clusterConfig.inspectModelLog >= ClusterConfig.info) {
    //  printf("collect theta grp=%d,  grpSize=%d, adress=%p\n", h, k,
    //         clusterResults.seedList[h].second);
    //}
    // delete[] clusterResults.seedList[h].second;
  }
  if (sumK > K) {
    printf("Bad allocation for collectTheta sumK=%d greater than K=%d\n", sumK,
           K);
    throw std::overflow_error("Bad Allocation");
  }
}

// Extract hits/seeds of a pre-cluster
int clusterProcess(const double* xyDxyi_, const Mask_t* cathi_,
                   const o2::mch::Mask_t* saturated_, const double* zi_, int chId,
                   int nPads)
{

  nbrOfHits = 0;
  cleanClusterResults();
  // if (INSPECTMODEL) {
  cleanInspectModel();
  InspectModelChrono(0, false);
  //}

  const double* xyDxyi;
  const double* zi;
  const Mask_t* cathi;
  const Mask_t* saturated;

  // Large and Noisy  clusters
  double* xyDxyi__;
  double* zi__;
  Mask_t* cathi__;
  Mask_t* saturated__;
  Mask_t noiseMask[nPads];
  int nNewPads = 0;
  double qCutOff = 0.0;

  // Pad filter when there are a too large number of pads
  if (nPads > clusterConfig.nbrPadLimit) {
    // Remove noisy event
    if (clusterConfig.processingLog >= ClusterConfig::info) {
      printf("WARNING: remove noisy pads nPads=%d, <q>=%8.1f, min/max q=%8.1f, %8.1f\n",
             nPads, vectorSum(zi_, nPads) / nPads, vectorMin(zi_, nPads),
             vectorMax(zi_, nPads));
    }
    // Select pads which q > qCutOff
    double ratioStep = clusterConfig.ratioStepForLargeCluster;
    double ratio = 1.;
    double qMax = vectorMax(zi_, nPads);
    int nPadsTest = nPads;

    while (nPadsTest > clusterConfig.nbrPadLimit) {
      ratio -= ratioStep;
      qCutOff = ratio * qMax;
      vectorBuildMaskGreater(zi_, qCutOff, nPads, noiseMask);
      nPadsTest = vectorSumShort(noiseMask, nPads);
    }
    nNewPads = nPadsTest;
    if (clusterConfig.processingLog >= clusterConfig.info) {
      printf("WARNING: remove noisy pads qCutOff=%8.1f, nbr of kept Pads=%d/%d\n",
             qCutOff, nNewPads, nPads);
    }
    xyDxyi__ = new double[nNewPads * 4];
    zi__ = new double[nNewPads];
    saturated__ = new Mask_t[nNewPads];
    cathi__ = new Mask_t[nNewPads];
    //
    vectorGather(zi_, noiseMask, nPads, zi__);
    vectorGatherShort(saturated_, noiseMask, nPads, saturated__);
    vectorGatherShort(cathi_, noiseMask, nPads, cathi__);
    maskedCopyXYdXY(xyDxyi_, nPads, noiseMask, nPads, xyDxyi__, nNewPads);
    xyDxyi = xyDxyi__;
    zi = zi__;
    cathi = cathi__;
    saturated = saturated__;
    nPads = nNewPads;
  } else {
    xyDxyi = xyDxyi_;
    zi = zi_;
    cathi = cathi_;
    saturated = saturated_;
  }
  // Build a cluster object
  ClusterPEM cluster(getConstX(xyDxyi, nPads), getConstY(xyDxyi, nPads),
                     getConstDX(xyDxyi, nPads), getConstDY(xyDxyi, nPads), zi,
                     cathi, saturated, chId, nPads);

  // Compute the underlying geometry (cathode plae superposition
  int nProjPads = cluster.buildProjectedGeometry(includeSinglePads);

  if (nProjPads == 0) {
    throw std::range_error("No projected pads !!!");
  }

  // Build geometric groups of pads
  // which constitute sub-clusters
  // A sub-cluster can contain several seeds
  int nGroups = cluster.buildGroupOfPads();

  // Store the mapping in ClusterResults
  storeGroupMapping(cluster.getCathGroup(0), cluster.getMapCathPadToPad(0),
                    cluster.getNbrOfPads(0), cluster.getCathGroup(1),
                    cluster.getMapCathPadToPad(1), cluster.getNbrOfPads(1));

  if (clusterConfig.inspectModel >= clusterConfig.active) {
    // Compute the charge on projected geometry
    double* qProj = cluster.projectChargeOnProjGeometry(includeSinglePads);
    // Save the projection with projected pads
    saveProjectedPads(cluster.getProjectedPads(), qProj);
    // Save the final groups (or cath-groups)
    savePadToCathGroup(cluster.getCathGroup(0), cluster.getMapCathPadToPad(0),
                       cluster.getNbrOfPads(0), cluster.getCathGroup(1),
                       cluster.getMapCathPadToPad(1), cluster.getNbrOfPads(1));
  }
  //
  // Sub-Cluster loop
  //
  int nbrOfProjPadsInTheGroup = 0;
  // Group allocations
  // ??? double *xyDxyGrp=nullptr;
  // ??? double *chGrp=nullptr;

  // EM allocations
  double* thetaEMFinal = nullptr;
  int finalK = 0;

  //
  // Find local maxima (seeds)
  //
  for (int g = 1; g <= nGroups; g++) {
    InspectModelChrono(1, false);
    //
    //  Exctract the current group
    //
    if (clusterConfig.processingLog >= clusterConfig.info) {
      printf("----------------\n");
      printf("Group %d/%d \n", g, nGroups);
      printf("----------------\n");
    }
    //
    // Number of seeds in this group
    int kEM;

    ClusterPEM* subCluster = nullptr;
    // Extract the sub-cluster
    if ((nGroups == 1) && (cluster.getNbrOfPadsInGroup(g) == cluster.getNbrOfPads())) {
      subCluster = &cluster;
    } else {
      subCluster = new ClusterPEM(cluster, g);
    }

    // To do something ???
    double meanCharge = 0.5 * (subCluster->getTotalCharge(0) + subCluster->getTotalCharge(1));

    if (clusterConfig.processingLog >= clusterConfig.info) {
      printf("[clusterProcessing] charge= (%7.0f %2.0f) nPads=(%d, %d)\n",
             subCluster->getTotalCharge(0), subCluster->getTotalCharge(1),
             subCluster->getNbrOfPads(0), subCluster->getNbrOfPads(1));
    }
    int nbrOfPadsInTheGroup = subCluster->getNbrOfPads();

    // Allocation of possible nbr of seeds
    // (.i.e the nbr of Pads)
    double thetaL[nbrOfPadsInTheGroup * 5];

    if (clusterConfig.inspectModel >= clusterConfig.active) {
      // Compute the local max with laplacian method
      // Used only to give insights of the cluster
      subCluster->buildProjectedGeometry(includeSinglePads);
      kEM =
        subCluster->findLocalMaxWithBothCathodes(thetaL, nbrOfPadsInTheGroup);
      double thetaExtra[kEM * 5];
      copyTheta(thetaL, nbrOfPadsInTheGroup, thetaExtra, kEM, kEM);
      saveThetaExtraInGroupList(thetaExtra, kEM);
      if (clusterConfig.inspectModelLog > clusterConfig.info) {
        printTheta("Theta findLocalMaxWithBothCathodes", meanCharge, thetaExtra, kEM);
      }
    }
    // Add null pads in the neighboring of the sub-cluster
    // ???
    subCluster->addBoundaryPads();
    //
    // Search for seeds on this sub-cluster
    kEM = subCluster->findLocalMaxWithPEM(thetaL, nbrOfPadsInTheGroup);
    if (kEM != 0) {
      double thetaEM[kEM * 5];
      copyTheta(thetaL, nbrOfPadsInTheGroup, thetaEM, kEM, kEM);

      if (clusterConfig.processingLog >= clusterConfig.info) {
        printf("[clusterProcessing] Find %2d PEM local maxima : \n", kEM);
        printTheta("ThetaEM", meanCharge, thetaEM, kEM);
      }

      //
      // EM
      //
      // ??? double *projXc = getX( xyDxyGrp, nbrOfProjPadsInTheGroup);
      // ??? double *projYc = getY( xyDxyGrp, nbrOfProjPadsInTheGroup);
      /*
      if (VERBOSE > 1) {
        printf("projPads in the group=%d, Xmin/max = %f %f, min/max = %f %f\n",
      g, vectorMin( projXc, nbrOfProjPadsInTheGroup),vectorMax( projXc,
      nbrOfProjPadsInTheGroup), vectorMin( projYc, nbrOfProjPadsInTheGroup),
      vectorMax( projYc, nbrOfProjPadsInTheGroup));
      }
      */
      if (clusterConfig.inspectModel >= clusterConfig.active) {
        // Save the seed founds by the EM algorithm
        saveThetaEMInGroupList(thetaEM, kEM);
      }
      InspectModelChrono(1, true);

      //
      //
      //
      // Perform the fitting if the sub-cluster g
      // is well separated at the 2 planes level (cath0, cath1)
      // If not the EM result is kept
      //
      InspectModelChrono(2, false);

      DataBlock_t newSeeds = subCluster->fit(thetaEM, kEM);
      finalK = newSeeds.first;
      nbrOfHits += finalK;
      //
      // Store result (hits/seeds)
      clusterResults.seedList.push_back(newSeeds);
      //
      if (clusterConfig.inspectModel >= clusterConfig.active) {
        saveThetaFitInGroupList(newSeeds.second, newSeeds.first);
      }
      InspectModelChrono(2, true);
    } else {
      // No EM seeds
      finalK = kEM;
      nbrOfHits += finalK;
      // Save the result of EM
      DataBlock_t newSeeds = std::make_pair(finalK, nullptr);
      clusterResults.seedList.push_back(newSeeds);
    }
    if (clusterConfig.processingLog >= clusterConfig.info) {
      printTheta("ThetaFit:", meanCharge, clusterResults.seedList.back().second, clusterResults.seedList.back().first);
    }
    // Release pointer for group
    // deleteDouble( xyDxyGrp );
    // deleteDouble( chGrp );
    if (nGroups > 1) {
      delete subCluster;
    }
  } // next group

  // Finalise inspectModel
  if (clusterConfig.inspectModel >= clusterConfig.active) {
    finalizeInspectModel();
  }
  InspectModelChrono(0, true);
  InspectModelChrono(-1, true);

  if (nNewPads) {
    delete[] xyDxyi__;
    delete[] cathi__;
    delete[] saturated__;
    delete[] zi__;
  }
  return nbrOfHits;
}
/*
} // namespace mch
} // namespace o2
*/