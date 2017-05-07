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
#include "TPCSimulation/Cluster.h"

#pragma link C++ class std::vector<o2::Base::Track::TrackPar>+;
#pragma link C++ class std::vector<o2::TPC::TrackTPC>+;

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

};

void convertTracks(TString inputBinaryFile, TString inputClusters, TString outputFile)
{

  // ===| input chain initialisation |==========================================
  TChain c("cbmsim");
  c.AddFile(inputClusters);

  TClonesArray *clusters=0x0;
  c.SetBranchAddress("TPCCluster", &clusters);
  //c.SetBranchAddress("TPC_Cluster", &clusters);

  // ===| output tree |=========================================================
  TFile fout(outputFile, "recreate");
  TTree tout("events","events");

  // ---| output data |---------------------------------------------------------
  vector<TrackTPC> arrTracks;
  //TClonesArray *arrTracksPtr = new TClonesArray("TrackTPC");
  //TClonesArray &arrTracks = *arrTracksPtr;
  EventHeader eventHeader;

  tout.Branch("header", &eventHeader, "run/I");
  tout.Branch("Tracks", &arrTracks);
  
  // ===| input binary file |===================================================
  FILE* fpInput = fopen(inputBinaryFile, "rb");
  if (fpInput == NULL)
  {
    printf("Error opening input file\n");
    exit(1);
  }

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

    // ---| set event information |---------------------------------------------
    eventHeader.run = 12345;

    // ---| read cluster tree |-------------------------------------------------
    c.GetEntry(nEvents);

    // ---| loop over tracks |--------------------------------------------------
    arrTracks.clear();
    for (int iTrack = 0;iTrack < numTracks;iTrack++)
    {
      count = fread(&track, sizeof(track), 1, fpInput);
      printf("Track %d Parameters: Alpha %f, X %f, Y %f, Z %f, SinPhi %f, DzDs %f, Q/Pt %f, Number of clusters %d, Fit OK %d\n", iTrack, track.Alpha, track.X, track.Y, track.Z, track.SinPhi, track.DzDs, track.QPt, track.NClusters, track.FitOK);

      //TrackTPC* track = new(arrTracks[iTrack]) TrackTPC();
      TrackTPC trackTPC(track.X, track.Alpha, {track.Y, track.Z, track.SinPhi, track.DzDs, track.QPt});
      arrTracks.push_back(trackTPC);
      TrackTPC & storedTrack = arrTracks.back();

      // ---| read cluster IDs |---
      if (size_t(track.NClusters) > ClusterIDs.size()) ClusterIDs.resize(track.NClusters);
      count = fread(&ClusterIDs[0], sizeof(ClusterIDs[0]), track.NClusters, fpInput);
      printf("Cluster IDs:");

      // ---| loop over clusters |---
      for (int iCluster = 0;iCluster < track.NClusters;iCluster++)
      {
        printf(" %d", ClusterIDs[iCluster]);
        storedTrack.AddCluster(static_cast<Cluster*>(clusters->At(ClusterIDs[iCluster])));
      }
      printf("\n");
    }

    nEvents++;
    tout.Fill();
  }
  fclose(fpInput);

  fout.Write();
  fout.Close();
}
