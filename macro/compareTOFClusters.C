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

#if !defined(__CLING__) || defined(__ROOTCLING__)

#include <TTree.h>
#include <TMath.h>
#include <TFile.h>
#include <vector>
#include <string>

#include "DataFormatsTOF/Cluster.h"

#endif

bool compareTOFClusters(std::string inpName1 = "tofclustersOr.root", std::string inpName2 = "tofclusters.root")
{
  bool status = true;
  int ngood = 0;
  int nfake = 0;

  TFile* f1 = TFile::Open(inpName1.c_str());
  TFile* f2 = TFile::Open(inpName2.c_str());

  TTree* t1 = (TTree*)f1->Get("o2sim");
  TTree* t2 = (TTree*)f2->Get("o2sim");

  std::vector<o2::tof::Cluster> clusters1, *pClusters1 = &clusters1;
  t1->SetBranchAddress("TOFCluster", &pClusters1);
  std::vector<o2::tof::Cluster> clusters2, *pClusters2 = &clusters2;
  t2->SetBranchAddress("TOFCluster", &pClusters2);

  t1->GetEvent(0);
  t2->GetEvent(0);

  int ncl1 = clusters1.size();
  int ncl2 = clusters2.size();

  if (ncl1 != ncl2) {
    printf("N clusters different!!!! %d != %d \n", ncl1, ncl2);
    status = false;
    return status;
  }

  printf("N clusters = %d\n", ncl1);

  for (int i = 0; i < ncl1; i++) {
    auto cl1 = clusters1.at(i);
    auto cl2 = clusters2.at(i);

    bool clstatus = true;

    if (cl1.getMainContributingChannel() != cl2.getMainContributingChannel()) {
      printf("cluster %d - Different Main Contributing Channels %d != %d \n", i, cl1.getMainContributingChannel(), cl2.getMainContributingChannel());
      clstatus = false;
    }

    if (cl1.getNumOfContributingChannels() != cl2.getNumOfContributingChannels()) {
      printf("cluster %d - Different N contributing channels %d != %d \n", i, cl1.getNumOfContributingChannels(), cl2.getNumOfContributingChannels());
      clstatus = false;
    }

    if (std::abs(cl1.getTimeRaw() - cl2.getTimeRaw()) > 1E-6) {
      printf("cluster %d - Different Raw Times %lf != %lf \n", i, cl1.getTimeRaw(), cl2.getTimeRaw());
      clstatus = false;
    }

    if (std::abs(cl1.getTime() - cl2.getTime()) > 1E-6) {
      printf("cluster %d - Different Calbrated Times %lf != %lf \n", i, cl1.getTime(), cl2.getTime());
      clstatus = false;
    }

    if (std::abs(cl1.getTot() - cl2.getTot()) > 1E-6) {
      printf("cluster %d - Different ToTs %f != %f \n", i, cl1.getTot(), cl2.getTot());
      clstatus = false;
    }

    if (cl1.getBC() != cl2.getBC()) {
      printf("cluster %d - Different Bunch Crossing IDs %d != %d \n", i, cl1.getBC(), cl2.getBC());
      clstatus = false;
    }

    if (cl1.getL0L1Latency() != cl2.getL0L1Latency()) {
      printf("cluster %d - Different L0L1 Latencies %d != %d \n", i, cl1.getL0L1Latency(), cl2.getL0L1Latency());
      clstatus = false;
    }

    if (std::abs(cl1.getX() - cl2.getX()) > 1E-6) {
      printf("cluster %d - Different X positions %lf != %lf \n", i, cl1.getX(), cl2.getX());
      clstatus = false;
    }

    if (std::abs(cl1.getY() - cl2.getY()) > 1E-6) {
      printf("cluster %d - Different Y positions %lf != %lf \n", i, cl1.getY(), cl2.getY());
      clstatus = false;
    }

    if (std::abs(cl1.getZ() - cl2.getZ()) > 1E-6) {
      printf("cluster %d - Different Z positions %lf != %lf \n", i, cl1.getZ(), cl2.getZ());
      clstatus = false;
    }

    if (std::abs(cl1.getSigmaY2() - cl2.getSigmaY2()) > 1E-6) {
      printf("cluster %d - Different Y2 sigmas %lf != %lf \n", i, cl1.getSigmaY2(), cl2.getSigmaY2());
      clstatus = false;
    }

    if (std::abs(cl1.getSigmaZ2() - cl2.getSigmaZ2()) > 1E-6) {
      printf("cluster %d - Different Z2 sigmas %lf != %lf \n", i, cl1.getSigmaZ2(), cl2.getSigmaZ2());
      clstatus = false;
    }

    if (std::abs(cl1.getSigmaYZ() - cl2.getSigmaYZ()) > 1E-6) {
      printf("cluster %d - Different YZ sigmas %lf != %lf \n", i, cl1.getSigmaYZ(), cl2.getSigmaYZ());
      clstatus = false;
    }

    if (cl1.getCount() != cl2.getCount()) {
      printf("cluster %d - Different Sensor IDs %d != %d \n", i, cl1.getCount(), cl2.getCount());
      clstatus = false;
    }

    if (cl1.getBits() != cl2.getBits()) {
      printf("cluster %d - Different Sensor IDs %d != %d \n", i, cl1.getBits(), cl2.getBits());
      clstatus = false;
    }

    if (cl1.getSensorID() != cl2.getSensorID()) {
      printf("cluster %d - Different Sensor IDs %d != %d \n", i, cl1.getSensorID(), cl2.getSensorID());
      clstatus = false;
    }

    if (!clstatus) {
      status = false;
      nfake++;
    } else
      ngood++;
  }

  printf("Clusters good = %d\n", ngood);
  printf("Clusters fake = %d\n", nfake);

  return status;
}
