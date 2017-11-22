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

#include "TROOT.h"
#include "TFile.h"
#include "TTree.h"
#include "TChain.h"
#include "TClonesArray.h"

#include "TPCReconstruction/TrackTPC.h"
#include "DetectorsBase/Track.h"
#include "TPCReconstruction/Cluster.h"
#endif

using namespace o2::TPC;
using namespace o2::Base::Track;

struct OutputTrack
{
	float Alpha;
	float X;
	float Y;
	float Z;
	float SinPhi;
	float DzDs;
	float QPt;
	int NClusters;
	int FitOK;
};

struct EventHeader
{
  int run;
  float cherenkovValue;

};

void convertTracks(TString inputBinaryFile, TString inputClusters, TString cherenkovFile, TString outputFile)
{

  // ===| input chain initialisation |==========================================
  TChain c("cbmsim");
  c.AddFile(inputClusters);

  float cherenkovValue = 0.;
  int runNumber = 0;

  std::vector<o2::TPC::Cluster> *clusters=nullptr;
  c.SetBranchAddress("TPCClusterHW", &clusters);
  //c.SetBranchAddress("TPC_Cluster", &clusters);
  c.SetBranchAddress("cherenkovValue", &cherenkovValue);
  c.SetBranchAddress("runNumber",      &runNumber);

  // ===| output tree |=========================================================
  TFile fout(outputFile, "recreate");
  TTree tout("events","events");

  // ---| output data |---------------------------------------------------------
  vector<TrackTPC> arrTracks;
  //TClonesArray *arrTracksPtr = new TClonesArray("TrackTPC");
  //TClonesArray &arrTracks = *arrTracksPtr;
  EventHeader eventHeader;
  eventHeader.run = 0;
  eventHeader.cherenkovValue = 0;

  tout.Branch("header", &eventHeader, "run/I:cherenkovValue/F");
  tout.Branch("Tracks", &arrTracks);
  
  // ===| input binary file |===================================================
  FILE* fpInput = fopen(inputBinaryFile, "rb");
  if (fpInput == NULL)
  {
    printf("Error opening input file\n");
    exit(1);
  }
  // ===| input cherenkov file |================================================
  ifstream istr(cherenkovFile.Data());

  // ===| Loop over all events in the input file |==============================
  //Number of events is not stored int the file, but we just read until we reach the end of file.
  //All values stored in the file are either int or float, so 32 bit.
  //Thus, we do not need to care about alignment. The same data structures must be used again for reading the file.
  int nEvents = 0;
  std::vector<unsigned int> ClusterIDs;
  while (!feof(fpInput))
  {
    int numTracks; //Must be int!
    OutputTrack track;
    size_t count = fread(&numTracks, sizeof(numTracks), 1, fpInput);
    if (!count) break;
    printf("Event: %d, Number of tracks: %d, %zu\n", nEvents, numTracks, count);

    // ---| read cluster tree |-------------------------------------------------
    c.GetEntry(nEvents);

    // ---| set event information from cluster file |---------------------------
    eventHeader.run = runNumber;
    eventHeader.cherenkovValue = cherenkovValue;

    // ---| loop over tracks |--------------------------------------------------
    arrTracks.clear();
    for (int iTrack = 0;iTrack < numTracks;iTrack++)
    {
      count = fread(&track, sizeof(track), 1, fpInput);
      //printf("Track %d Parameters: Alpha %f, X %f, Y %f, Z %f, SinPhi %f, DzDs %f, Q/Pt %f, Number of clusters %d, Fit OK %d\n", iTrack, track.Alpha, track.X, track.Y, track.Z, track.SinPhi, track.DzDs, track.QPt, track.NClusters, track.FitOK);

      //TrackTPC* track = new(arrTracks[iTrack]) TrackTPC();
      TrackTPC trackTPC(track.X, track.Alpha, {track.Y, track.Z, track.SinPhi, track.DzDs, track.QPt}, {0, 0});
      arrTracks.push_back(trackTPC);
      TrackTPC & storedTrack = arrTracks.back();

      // ---| read cluster IDs |---
      if (size_t(track.NClusters) > ClusterIDs.size()) ClusterIDs.resize(track.NClusters);
      count = fread(&ClusterIDs[0], sizeof(ClusterIDs[0]), track.NClusters, fpInput);
      //printf("Cluster IDs:");

      // ---| loop over clusters |---
      for (int iCluster = 0;iCluster < track.NClusters;iCluster++)
      {
        //printf(" %d", ClusterIDs[iCluster]);

        const auto& tempCluster = (*clusters)[ClusterIDs[iCluster]];
        storedTrack.addCluster(tempCluster);
      }
      //printf("\n");
    }

    nEvents++;
    tout.Fill();
  }
  fclose(fpInput);

  fout.Write();
  fout.Close();
}
