// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <vector>
#include <fstream>
#include <iostream>
#include "TSystem.h"

#include "TROOT.h"
#include "TTree.h"
#include "TFile.h"
#include "TString.h"

#include "DataFormatsTPC/ClusterNative.h"
#include "DataFormatsTPC/ClusterHardware.h"
#include "DataFormatsTPC/Helpers.h"
#include "TPCReconstruction/HardwareClusterDecoder.h"
#include "TPCBase/Constants.h"
#include "TPCBase/CRU.h"
#endif

using namespace o2::TPC;
using namespace o2::DataFormat::TPC;
using namespace std;

ClusterHardwareContainer8kb* clusterContainerMemory;

void runHardwareClusterDecoderRoot(TString infile = "clusterHardware.root", TString outfile = "clusterNative.root") {
  gSystem->Load("libTPCReconstruction.so");
  HardwareClusterDecoder decoder;

  std::vector<std::pair<const ClusterHardwareContainer*, std::size_t>> inputList;
  std::vector<ClusterHardwareContainer8kb> inputBuffer;

  TFile fin(infile);
  TTree* tin = (TTree*) fin.FindObjectAny("clustersHardware");
  if (tin == NULL) {printf("Error reading input\n"); return;}
  tin->SetBranchAddress("clusters", &clusterContainerMemory);
  
  inputBuffer.reserve(tin->GetEntries());
  inputList.reserve(tin->GetEntries());
  for (int i = 0;i < tin->GetEntries();i++)
  {
    tin->GetEntry(i);
    inputBuffer.push_back(*clusterContainerMemory);
    inputList.emplace_back(inputBuffer[i].getContainer(), 1);
  }
  fin.Close();

  std::vector<ClusterNativeContainer> cont;
  decoder.decodeClusters(inputList, cont);
  
  TFile fout(outfile, "recreate");
  int nClustersTotal = 0;
  for (unsigned int i = 0;i < cont.size();i++)
  {
    nClustersTotal += cont[i].mClusters.size();
    fprintf(stderr, "\tSector %d, Row %d, Clusters %d\n", (int) cont[i].mSector, (int) cont[i].mGlobalPadRow, (int) cont[i].mClusters.size());
    TString contName = Form("clusters_sector_%d_row_%d", (int) cont[i].mSector, (int) cont[i].mGlobalPadRow);
    fout.WriteObject(&cont[i], contName);
  }

  printf("Total clusters: %d\n", nClustersTotal);
  fout.Close();
}
