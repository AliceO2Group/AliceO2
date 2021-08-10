#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <vector>
#include <TFile.h>
#include <TTree.h>
#endif

void CheckMCTrack()
{
  // Reconstructed tracks
  TFile* file1 = TFile::Open(tracfile.data());
  TTree* recTree = (TTree*)gFile->Get("o2sim");
  std::vector<TrackITS>* recArr = nullptr;
  recTree->SetBranchAddress("ITSTrack", &recArr);
  // Track MC labels
  std::vector<o2::MCCompLabel>* trkLabArr = nullptr;
  recTree->SetBranchAddress("ITSTrackMCTruth", &trkLabArr);
}