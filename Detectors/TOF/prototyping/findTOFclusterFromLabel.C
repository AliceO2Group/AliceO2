#if !defined(__CLING__) || defined(__ROOTCLING__)
#include "TFile.h"
#include "TTree.h"
#include "DataFormatsTOF/Cluster.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTrack.h"
#include "DataFormatsTPC/TrackTPC.h"
#include "ReconstructionDataFormats/TrackTPCITS.h"
#include "CommonDataFormat/EvIndex.h"
#include "TOFBase/Geo.h"
#include "TOFBase/Digit.h"
#endif

void findTOFclusterFromLabel(int trackID, int eventID = 0, int sourceID = 0)
{

  // macro to find the labels of a TPCITS track and the corresponding TOF cluster

  // getting the TOF clusters
  TFile* fclustersTOF = new TFile("tofclusters.root");
  TTree* tofClTree = (TTree*)fclustersTOF->Get("o2sim");
  std::vector<o2::tof::Cluster>* mTOFClustersArrayInp = new std::vector<o2::tof::Cluster>;
  tofClTree->SetBranchAddress("TOFCluster", &mTOFClustersArrayInp);
  o2::dataformats::MCTruthContainer<o2::MCCompLabel>* mcTOF = new o2::dataformats::MCTruthContainer<o2::MCCompLabel>();
  tofClTree->SetBranchAddress("TOFClusterMCTruth", &mcTOF);

  tofClTree->GetEntry(0);

  // now checking if we have a corresponding cluster
  bool found = false;
  for (int tofClIndex = 0; tofClIndex < mTOFClustersArrayInp->size(); tofClIndex++) {
    o2::tof::Cluster tofCluster = mTOFClustersArrayInp->at(tofClIndex);
    const auto& labelsTOF = mcTOF->getLabels(tofClIndex);
    int trackIdTOF;
    int eventIdTOF;
    int sourceIdTOF;
    for (int ilabel = 0; ilabel < labelsTOF.size(); ilabel++) {
      if (trackID == labelsTOF[ilabel].getTrackID() && eventID == labelsTOF[ilabel].getEventID() && sourceID == labelsTOF[ilabel].getSourceID()) {
        Printf("The corresponding TOF cluster is %d", tofClIndex);
        Printf("TOF label %d: trackID = %d, eventID = %d, sourceID = %d", ilabel, labelsTOF[ilabel].getTrackID(), labelsTOF[ilabel].getEventID(), labelsTOF[ilabel].getSourceID());
        found = true;
        int nContrChannels = tofCluster.getNumOfContributingChannels();
        int mainContrChannel = tofCluster.getMainContributingChannel();
        int* indices = new int[5];
        o2::tof::Geo::getVolumeIndices(mainContrChannel, indices);
        Printf("main contributing channel: sector = %d, plate = %d, strip = %d, padz = %d, padx = %d", indices[0], indices[1], indices[2], indices[3], indices[4]);
        break;
      }
    }
  }
  if (!found)
    Printf("No TOF cluster corresponding to this label was found");

  TFile* fdigitsTOF = new TFile("tofdigits.root");
  TTree* tofDigTree = (TTree*)fdigitsTOF->Get("o2sim");
  std::vector<std::vector<o2::tof::Digit>>* mTOFDigitsArrayInp = nullptr;
  tofDigTree->SetBranchAddress("TOFDigit", &mTOFDigitsArrayInp);
  std::vector<o2::dataformats::MCTruthContainer<o2::MCCompLabel>>* mcTOFDig = nullptr;
  tofDigTree->SetBranchAddress("TOFDigitMCTruth", &mcTOFDig);
  bool foundInDigits = false;
  Printf("Now looking in the digits");
  for (int ientry = 0; ientry < tofDigTree->GetEntries(); ientry++) {
    //Printf("\n\nEntry in tree %d", ientry);
    tofDigTree->GetEntry(ientry);
    for (int iVect = 0; iVect < mTOFDigitsArrayInp->size(); iVect++) {
      //Printf("\nEntry in vector of digits and MC truth %d", iVect);
      std::vector<o2::tof::Digit> digitVector = mTOFDigitsArrayInp->at(iVect);
      o2::dataformats::MCTruthContainer<o2::MCCompLabel> digitMCTruth = mcTOFDig->at(iVect);
      for (int iDig = 0; iDig < digitVector.size(); iDig++) {
        //Printf("Digit %d", iDig);
        o2::tof::Digit digit = digitVector.at(iDig);
        int digitLabel = digit.getLabel();
        gsl::span<const o2::MCCompLabel> mcArray = digitMCTruth.getLabels(digitLabel);
        for (int j = 0; j < static_cast<int>(mcArray.size()); j++) {
          //printf("checking element %d in the array of labels\n", j);
          auto label = digitMCTruth.getElement(digitMCTruth.getMCTruthHeader(digitLabel).index + j);
          //printf("TrackID = %d, EventID = %d, SourceID = %d\n", label.getTrackID(), label.getEventID(), label.getSourceID());
          if (label.getTrackID() == trackID && label.getEventID() == eventID && label.getSourceID() == sourceID) {
            Printf("We found the label that we were looking for! tree entry = %d, vector entry = %d, digit = %d", ientry, iVect, iDig);
            foundInDigits = true;
          }
        }
      }
    }
  }
  if (!foundInDigits)
    Printf("The label was NEVER found in the digits");

  TFile* fKine = new TFile("o2sim_Kine.root");
  TTree* tKine = (TTree*)fKine->Get("o2sim");
  std::vector<o2::MCTrack>* mcArr = nullptr;
  tKine->SetBranchAddress("MCTrack", &mcArr);
  tKine->GetEntry(eventID);
  for (int i = 0; i < mcArr->size(); ++i) {
    const auto& mcTrack = (*mcArr)[i];
    if (i == trackID) {
      Printf("Particle %d: pdg = %d, pT = %f, px = %f, py = %f, pz = %f, vx = %f, vy = %f, vz = %f", i, mcTrack.GetPdgCode(), TMath::Abs(mcTrack.GetStartVertexMomentumX() * mcTrack.GetStartVertexMomentumX() + mcTrack.GetStartVertexMomentumY() * mcTrack.GetStartVertexMomentumY()), mcTrack.GetStartVertexMomentumX(), mcTrack.GetStartVertexMomentumY(), mcTrack.GetStartVertexMomentumZ(), mcTrack.GetStartVertexCoordinatesX(), mcTrack.GetStartVertexCoordinatesY(), mcTrack.GetStartVertexCoordinatesZ());
    }
  }

  TFile* fmatch = new TFile("o2match_itstpc.root");
  TTree* matchTPCITS = (TTree*)fmatch->Get("matchTPCITS");
  std::vector<o2::dataformats::TrackTPCITS>* mTracksArrayInp = new std::vector<o2::dataformats::TrackTPCITS>;
  matchTPCITS->SetBranchAddress("TPCITS", &mTracksArrayInp);
  matchTPCITS->GetEntry(eventID);

  // getting the TPC tracks
  TFile* ftracksTPC = new TFile("tpctracks.root");
  TTree* tpcTree = (TTree*)ftracksTPC->Get("tpcrec");
  std::vector<o2::tpc::TrackTPC>* mTPCTracksArrayInp = new std::vector<o2::tpc::TrackTPC>;
  tpcTree->SetBranchAddress("TPCTracks", &mTPCTracksArrayInp);
  o2::dataformats::MCTruthContainer<o2::MCCompLabel>* mcTPC = new o2::dataformats::MCTruthContainer<o2::MCCompLabel>();
  tpcTree->SetBranchAddress("TPCTracksMCTruth", &mcTPC);
  tpcTree->GetEntry(eventID);

  for (int i = 0; i < mTracksArrayInp->size(); i++) {
    o2::dataformats::TrackTPCITS trackITSTPC = mTracksArrayInp->at(i);
    int evIdxTPC = trackITSTPC.getRefTPC();
    const auto& labelsTPC = mcTPC->getLabels(evIdxTPC);
    for (int ilabel = 0; ilabel < labelsTPC.size(); ilabel++) {
      //Printf("TPC label %d: trackID = %d, eventID = %d, sourceID = %d", ilabel, labelsTPC[ilabel].getTrackID(), labelsTPC[ilabel].getEventID(), labelsTPC[ilabel].getSourceID());
      if (labelsTPC[ilabel].getTrackID() == trackID && labelsTPC[ilabel].getEventID() == eventID)
        Printf("TPC track found");
    }
  }

  return;
}
