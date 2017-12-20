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
#include "TFile.h"
#include "TString.h"

#include "DataFormatsTPC/ClusterNative.h"
#include "DataFormatsTPC/ClusterHardware.h"
#include "TPCReconstruction/HardwareClusterDecoder.h"
#include "TPCBase/Constants.h"
#include "TPCBase/CRU.h"
#else
#pragma cling load("libTPCReconstruction")
#pragma cling load("libDataFormatsTPC")
#endif

using namespace o2::TPC;
using namespace o2::DataFormat::TPC;
using namespace std;

int runHardwareClusterDecoderRaw(TString outfile = "", int tf = 0) {
  if (outfile.EqualTo("")) {printf("Filename missing\n");return(1);}
  HardwareClusterDecoder decoder;

  TFile file(outfile, "recreate");
  int nClustersTotal = 0;
  for (int iCRU = 0;iCRU < CRU::MaxCRU;iCRU++)
  {
    CRU cru(iCRU);
    Sector sec = cru.sector();
    int region = cru.region();
    
    TString fname = Form("tf_%d_sec_%d_region_%d.raw", tf, (int) sec, region);
    ifstream inFile;
    inFile.open(fname.Data(), std::ios::binary);
    if (inFile.fail()) continue;
    stringstream strStream;
    strStream << inFile.rdbuf();
    string str = strStream.str();
    fprintf(stderr, "Processing sector %d, region %d, len %d\n", (int) sec, region, (int) str.size());
    
    std::vector<std::pair<const ClusterHardwareContainer*, std::size_t>> inputList = {{reinterpret_cast<const ClusterHardwareContainer*> (str.c_str()), str.size() / 8192}};
    std::vector<ClusterNativeContainer> cont;
    decoder.decodeClusters(inputList, cont);
    for (unsigned int i = 0;i < cont.size();i++)
    {
      nClustersTotal += cont[i].clusters.size();
      fprintf(stderr, "\tSector %d, Row %d, Clusters %d\n", (int) cont[i].sector, (int) cont[i].globalPadRow, (int) cont[i].clusters.size());
      TString contName = Form("clusters_sector_%d_row_%d", (int) cont[i].sector, (int) cont[i].globalPadRow);
      file.WriteObject(&cont[i], contName);
    }
  }
  printf("Total clusters: %d\n", nClustersTotal);
  file.Close();
  return(0);
}
