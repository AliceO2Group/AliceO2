// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <vector>
#include <fstream>
#include <iostream>

#include "TROOT.h"
#include "TFile.h"
#include "TTree.h"
#include "TChain.h"
#include "TClonesArray.h"

#include "TPCBase/Defs.h"
#include "TPCBase/CRU.h"
#include "TPCBase/Sector.h"
#include "TPCBase/Mapper.h"
#include "TPCBase/PadRegionInfo.h"
#include "TPCSimulation/Cluster.h"
#include "TPCSimulation/Constants.h"

/*
gSystem->AddIncludePath("-I$O2_ROOT/include -I$FAIRROOT_ROOT/include");
.L convertClusters.C+
*/

using namespace o2::TPC;

struct ClusterData {
	int fId;
	int fRow;
	float fX;
	float fY;
	float fZ;
	float fAmp;
};

ClusterData fclDat;
ClusterData fclDatGlobal;

Int_t feventNumber=0;
Int_t fsector=0;

TTree *fOutTree = 0x0;

void dumpData(std::ofstream &fout, std::vector<ClusterData> &data);
void addCluster(std::vector<ClusterData> &data, Cluster& cluster);

void convertClusters(TString filename, Int_t nmaxEvent=-1, Int_t startEvent=0)
{

  // ===| input chain initialisation |==========================================
  TChain c("cbmsim");
  c.AddFile(filename);

  TClonesArray *clusters=0x0;
  c.SetBranchAddress("TPCClusterHW", &clusters);
  //c.SetBranchAddress("TPC_HW_Cluster", &clusters);

  // ===| event ranges |========================================================
  const Int_t nentries = c.GetEntries();
  const Int_t start = startEvent>nentries?0:startEvent;
  const Int_t max   = nmaxEvent>0 ? TMath::Min(nmaxEvent, nentries-startEvent) : nentries;

  TFile f("data.root", "recreate");
  fOutTree = new TTree("data","data");
  fOutTree->Branch("ev", &feventNumber);
  fOutTree->Branch("sector", &fsector);
  fOutTree->Branch("cl", &fclDat, "id/I:row/I:x/F:y:z:a");
  fOutTree->Branch("clg", &fclDatGlobal, "id/I:row/I:x/F:y:z:a");
  gROOT->cd();

  // ===| loop over events |====================================================
  for (Int_t iEvent=0; iEvent<max; ++iEvent) {
    c.GetEntry(start+iEvent);
    feventNumber=iEvent;

    printf("Processing event %d with %d number of clusters\n", iEvent, clusters->GetEntries());
    if (!clusters->GetEntries()) continue;

    // ---| output file |-------------------------------------------------------
    TString outputFileName = TString::Format("event.%d.dump", iEvent);
 
    std::ofstream fout(outputFileName, ios::out | ios::binary);

    // ---| output array |------------------------------------------------------
    // must be sorted in sectors
    std::vector<ClusterData> data;

    for (int iSector = 0; iSector<36; ++iSector){
      fsector = iSector;
      Sector thisSector(iSector);
      std::cout << "Sector: " << iSector << std::endl;
      // ---| loop over clusters |------------------------------------------------
      for (Int_t icluster=0; icluster<clusters->GetEntries(); ++icluster) {
        Cluster& cluster = *static_cast<Cluster*>(clusters->At(icluster));
        const Sector sector = CRU(cluster.getCRU()).sector();

        if (sector != thisSector) continue;
        std::cout << cluster << std::endl;

        cluster.SetUniqueID(icluster);
        addCluster(data, cluster);

      }
      printf("  clusters in sector: %zu\n", data.size());
      dumpData(fout, data);
    }
/*
    Sector lastSector;
    // ---| loop over clusters |------------------------------------------------
    for (const auto object : *clusters) {
      Cluster& cluster = *static_cast<Cluster*>(object);
      cluster.SetUniqueID(clusterId++);
      
      const Sector sector = CRU(cluster.getCRU()).sector();
      if (sector != lastSector) {
        dumpData(fout, data);
        lastSector=sector;
      }
      addCluster(data, cluster);

    }
    dumpData(fout, data);
    */
    fout.close();
  }
  f.Write();
  f.Close();
}

//______________________________________________________________________________
void addCluster(std::vector<ClusterData> &data, Cluster& cluster)
{
  const CRU cru(cluster.getCRU());

  // ===| mapper |==============================================================
  Mapper &mapper = Mapper::instance();
  const PadRegionInfo& region = mapper.getPadRegionInfo(cru.region());
  const int rowInSector       = cluster.getRow() + region.getGlobalRowOffset();
  const float padY            = cluster.getPadMean();
  const int padNumber         = int(padY);
  const GlobalPadNumber pad   = mapper.globalPadNumber(PadPos(rowInSector, padNumber));
  const PadCentre& padCentre  = mapper.padCentre(pad);
  const float localY          = padCentre.getY() - (padY - padNumber - 0.5) * region.getPadWidth();
  const float localYfactor    = (cru.side()==Side::A)?-1.f:1.f;
        float zPosition         = cluster.getTimeMean()*ZBINWIDTH*DRIFTV;
        //float zPosition         = TPCLENGTH - cluster.getTimeMean()*ZBINWIDTH*DRIFTV;

  Point2D<float> clusterPos(padCentre.getX(), localY); 

        printf("zPosition: %.2f\n", zPosition);
  // sanity checks
  if (zPosition<0) return;
  if (zPosition>TPCLENGTH) return;

  ClusterData cl;
  cl.fId  = cluster.GetUniqueID();
  cl.fRow = rowInSector;
  cl.fX    = clusterPos.getX();
  cl.fY    = clusterPos.getY()*(localYfactor);
  cl.fZ    = zPosition*(-localYfactor);
  cl.fAmp  = cluster.getQmax();

  data.push_back(cl);
}

//______________________________________________________________________________
void dumpData(std::ofstream &fout, std::vector<ClusterData> &data)
{
  Mapper &mapper = Mapper::instance();

  const int clustersInSector = data.size();
  fout.write((char*)&clustersInSector, sizeof(clustersInSector));
  if (clustersInSector) {
    fout.write((char*)&data[0], clustersInSector * sizeof(ClusterData));
  }
  std::cout << clustersInSector << " " << std::endl;
  for (auto& cl : data) {
    std::cout << cl.fId <<" " << cl.fRow << " " 
              << cl.fX << " " << cl.fY << " " << cl.fZ << " "
              << cl.fAmp << std::endl;
    fclDat = cl;
    fclDatGlobal = cl;
    LocalPosition3D posLoc(cl.fX, cl.fY, cl.fZ);
    GlobalPosition3D posGlob = Mapper::LocalToGlobal(posLoc, fsector);
    fclDatGlobal.fX = posGlob.getX();
    fclDatGlobal.fY = posGlob.getY();
    fOutTree->Fill();
  }
  data.clear();
}
