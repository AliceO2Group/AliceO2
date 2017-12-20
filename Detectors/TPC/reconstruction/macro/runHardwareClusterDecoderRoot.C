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

#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "DataFormatsTPC/ClusterNative.h"
#include "DataFormatsTPC/ClusterHardware.h"
#include "DataFormatsTPC/Helpers.h"
#include "TPCReconstruction/HardwareClusterDecoder.h"
#include "TPCBase/Constants.h"
#include "TPCBase/CRU.h"
#else
#pragma cling load("libTPCReconstruction")
#pragma cling load("libDataFormatsTPC")
#endif

using namespace o2;
using namespace o2::DataFormat::TPC;
using namespace o2::TPC;
using namespace o2::dataformats;
using namespace std;

using MCLabelContainer = MCTruthContainer<MCCompLabel>;

int runHardwareClusterDecoderRoot(TString infile = "", TString outfile = "") {
  if (infile.EqualTo("") || outfile.EqualTo("")) {printf("Filename missing\n");return(1);}
  HardwareClusterDecoder decoder;

  ClusterHardwareContainer8kb* clusterContainerMemory = nullptr;
  std::vector<std::pair<const ClusterHardwareContainer*, std::size_t>> inputList;
  std::vector<ClusterHardwareContainer8kb> inputBuffer;
  MCLabelContainer* inMCLabels = nullptr;
  std::vector<MCLabelContainer> inputBufferMC;
  
  std::vector<MCLabelContainer> outMCLabels;

  TFile fin(infile);
  TTree* tin = (TTree*) fin.FindObjectAny("clustersHardware");
  if (tin == NULL) {printf("Error reading input\n"); return(1);}
  tin->SetBranchAddress("clusters", &clusterContainerMemory);
  tin->SetBranchAddress("clustersMCTruth", &inMCLabels);
  
  inputBuffer.reserve(tin->GetEntries());
  inputList.reserve(tin->GetEntries());
  if (inMCLabels) inputBufferMC.reserve(tin->GetEntries());
  for (int i = 0;i < tin->GetEntries();i++)
  {
    tin->GetEntry(i);
    inputBuffer.push_back(*clusterContainerMemory);
    inputList.emplace_back(inputBuffer[i].getContainer(), 1);
    if (inMCLabels) inputBufferMC.emplace_back(std::move(*inMCLabels));
  }
  fin.Close();

  std::vector<ClusterNativeContainer> cont;
  decoder.decodeClusters(inputList, cont, inMCLabels ? &inputBufferMC : nullptr, &outMCLabels);
  
  TFile fout(outfile, "recreate");
  int nClustersTotal = 0;
  for (unsigned int i = 0;i < cont.size();i++)
  {
    nClustersTotal += cont[i].clusters.size();
    fprintf(stderr, "\tSector %d, Row %d, Clusters %d\n", (int) cont[i].sector, (int) cont[i].globalPadRow, (int) cont[i].clusters.size());
    fout.WriteObject(&cont[i], Form("clusters_sector_%d_row_%d", (int) cont[i].sector, (int) cont[i].globalPadRow));
    if (inMCLabels) fout.WriteObject(&outMCLabels[i], Form("clustersMCTruth_sector_%d_row_%d", (int) cont[i].sector, (int) cont[i].globalPadRow));
  }

  printf("Total clusters: %d\n", nClustersTotal);
  fout.Close();
  return(0);
}
