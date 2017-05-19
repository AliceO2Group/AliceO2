#ifndef ALICEO2_TPC_CONVERTRAWCLUSTERS_C_
#define ALICEO2_TPC_CONVERTRAWCLUSTERS_C_

/// \file   RawClusterFinder.h
/// \author Jens Wiechula, Jens.Wiechula@ikf.uni-frankfurt.de

#include <vector>
#include <memory>
#include <fstream>

#include "Rtypes.h"
#include "TClonesArray.h"
#include "TFile.h"
#include "TSystem.h"
#include "TGraph.h"
#include "TLinearFitter.h"

#include "TPCBase/Defs.h"
#include "TPCBase/CalDet.h"
#include "TPCBase/CRU.h"
#include "TPCCalibration/CalibRawBase.h"
#include "TPCSimulation/HwClusterer.h"
#include "TPCSimulation/BoxClusterer.h"
#include "TPCSimulation/ClusterContainer.h"
#include "TPCReconstruction/DigitData.h"
#include "TPCBase/Mapper.h"
#include "TPCReconstruction/TrackTPC.h"
#include "TPCSimulation/Cluster.h"

#include "TCanvas.h"


namespace o2
{
namespace TPC
{


/// \brief Raw cluster conversion
///
/// This class is used to produce pad wise pedestal and noise calibration data
///
/// origin: TPC
/// \author Jens Wiechula, Jens.Wiechula@ikf.uni-frankfurt.de

class RawClusterFinder : public CalibRawBase
{
  public:
    enum class ClustererType : char {
      Box,  ///< use box clusterer
      HW    ///< use HW clusterer
    };


    struct EventHeader
    {
      int run;
      float cherenkovValue;

    };

    using vectorType = std::vector<float>;

    /// default constructor
    RawClusterFinder(PadSubset padSubset = PadSubset::ROC) : CalibRawBase(padSubset), mClustererType(ClustererType::HW), mPedestals(nullptr), mVectorDigits() {;}

    /// default destructor
    virtual ~RawClusterFinder() = default;

    /// set clusterer type
    void setClustererType(ClustererType clustererType) { mClustererType = clustererType; }

    /// not used
    Int_t UpdateROC(const Int_t sector, const Int_t row, const Int_t pad,
                    const Int_t timeBin, const Float_t signal) final { return 0;}

    /// 
    Int_t UpdateCRU(const CRU& cru, const Int_t row, const Int_t pad,
                    const Int_t timeBin, const Float_t signal) final;


    void TrackFinder(int nTries, int allowedMisses, int minClusters, bool reverse, TClonesArray *pclusterArray, vector<TrackTPC> &arrTracks);
    static void GetFit(TGraph *trackGraph, float& slope, float& offset, float& slopeError, float& offsetError, float& chi2);

    void setPedestals(CalPad* pedestals) { mPedestals = pedestals; }
    static void ProcessEvents(TString fileInfo, TString pedestalFile, TString outputFileName="clusters.root", Int_t maxEvents=-1, TString cherenkovFile="", ClustererType clustererType=ClustererType::HW);

    std::vector<std::unique_ptr<Digit>>& getDigitVector() { return mVectorDigits; }

    /// Dummy end event
    virtual void EndEvent() final {};

    TTree *mTOut;

  private:
    ClustererType     mClustererType;
    CalPad       *mPedestals;
    std::vector<std::unique_ptr<Digit>> mVectorDigits;

    /// dummy reset
    void ResetEvent() final { mVectorDigits.clear(); }
};

Int_t RawClusterFinder::UpdateCRU(const CRU& cru, const Int_t row, const Int_t pad,
                                     const Int_t timeBin, const Float_t signal)
{
  float corrSignal = signal;

  // ===| get pedestal |========================================================
  if (mPedestals) {
    corrSignal -= mPedestals->getValue(cru, row, pad);
  }

  // ===| add new digit |=======================================================
  mVectorDigits.emplace_back(new DigitData(cru, corrSignal, row, pad, timeBin));

  return 1;
}

void RawClusterFinder::ProcessEvents(TString fileInfo, TString pedestalFile, TString outputFileName, Int_t maxEvents, TString cherenkovFile, ClustererType clustererType)
{

  // ===| create raw converter |================================================
  RawClusterFinder converter;
  converter.setupContainers(fileInfo);

  // ===| load pedestals |======================================================
  TFile f(pedestalFile);
  CalDet<float> *pedestal = nullptr;
  if (f.IsOpen() && !f.IsZombie()) {
    f.GetObject("Pedestals", pedestal);
    printf("pedestal: %.2f\n", pedestal->getValue(CRU(0), 0, 0));
  }

  // ===| output file and container |===========================================
  TClonesArray arrCluster("o2::TPC::Cluster");
  float cherenkovValue = 0.;
  int runNumber = 0;

  TFile fout(outputFileName,"recreate");
  TTree t("cbmsim","cbmsim");
  t.Branch("TPCClusterHW", &arrCluster);
  t.Branch("cherenkovValue", &cherenkovValue);
  t.Branch("runNumber", &runNumber);

  // ===| output track file and container |======================================
  TFile foutTracks("tracksLinear.root", "recreate");
  TTree tOut("events","events");

  vector<TrackTPC> arrTracks;
  EventHeader eventHeader;

  tOut.Branch("header", &eventHeader, "run/I:cherenkovValue/F");
  tOut.Branch("Tracks", &arrTracks);


  // ===| input cherenkov file |================================================
  ifstream istr(cherenkovFile.Data());

  // ===| fill header |=========================================================
  eventHeader.run = TString(gSystem->Getenv("RUN_NUMBER")).Atoi();
  runNumber = eventHeader.run;

  // ===| cluster finder |======================================================
  // HW cluster finder
  std::unique_ptr<Clusterer> cl;
  if (clustererType == ClustererType::HW) {
    HwClusterer *hwCl = new HwClusterer(HwClusterer::Processing::Parallel, 0, 0, 4);
    hwCl->setContinuousReadout(false);
    hwCl->setPedestalObject(pedestal);
    cl = std::unique_ptr<Clusterer>(hwCl);
  }
  else if (clustererType == ClustererType::Box) {
    BoxClusterer *boxCl = new BoxClusterer;
    boxCl->setPedestals(pedestal);
    cl = std::unique_ptr<Clusterer>(boxCl);
  }
  else {
    return;
  }
    
  //cl->setRequirePositiveCharge(false);
  cl->Init();

  // Box cluster finder
  
  // ===| loop over all data |==================================================
  int events = 0;
  bool data = true;
  printf("max events %d\n", maxEvents);
  CalibRawBase::ProcessStatus status;
  //while (((status = converter.ProcessEvent()) == CalibRawBase::ProcessStatus::Ok) && (maxEvents>0)?events<maxEvents:1) {
  // skip synch event
  converter.ProcessEvent();
  while (converter.ProcessEvent() == CalibRawBase::ProcessStatus::Ok) {
    if (maxEvents>0 && events>=maxEvents) break;

    printf("========| Event %4zu %d %d %d |========\n", converter.getPresentEventNumber(), events, maxEvents, status);

    auto &arr = converter.getDigitVector();
    if (!arr.size()) {++events; continue;}
    //printf("Converted digits: %zu %f\n", arr.size(), arr.at(0)->getChargeFloat());

    ClusterContainer* clCont = cl->Process(arr);

    clCont->FillOutputContainer(&arrCluster);

    // ---| set cherenkov value|---------------------------------------------
    if (istr.is_open()) {
      float value=0.f;
      istr >> value;
      cherenkovValue = TMath::Abs(value);
    }
    t.Fill();

    printf("Found clusters: %d\n", arrCluster.GetEntriesFast());

    Int_t nTries = -1; // defaults to number of clusters in the roc
    Int_t allowedMisses = 3;  //number of allowed holes in track before discarding
    Int_t minClusters = 10;
    Bool_t reverse = kFALSE; //start search from pad row opposite to beam entry side
    TString TrackerOut = "tracksLinear.root";
    //converter.TrackFinder(nTries, allowedMisses, minClusters, reverse, &arrCluster, arrTracks);

    tOut.Fill();
    arrCluster.Clear();
    ++events;
  }

  fout.Write();
  fout.Close();
  foutTracks.Write();
  foutTracks.Close();
}

void RawClusterFinder::TrackFinder(int nTries, int allowedMisses, int minClusters, bool reverse, TClonesArray *pclusterArray, vector<TrackTPC> &arrTracks)
{

int loopcounter = 0;

  float zPosition = 0.;
  float fWindowPad = 4.;
  float fWindowTime = 4.;
 

  Mapper &mapper = Mapper::instance();
  TClonesArray &clusterArray = *pclusterArray;
  std::vector<int> trackNumbers;
  trackNumbers.resize(clusterArray.GetEntriesFast());

  arrTracks.clear();

  if (nTries < 0) nTries = clusterArray.GetEntriesFast();

  int nTracks = 0;

  if(clusterArray.GetEntriesFast()>=1000) {
    printf("Very large event, no tracking will be performed!!");
    return;
  }

  uint nclustersUsed = 0;

  //loop over the n (given bu nTries) clusters and use iFirst as starting point for tracking
  for(Int_t iFirst = 0; (iFirst<nTries&&iFirst<clusterArray.GetEntriesFast()); iFirst++){

    Int_t nClusters = clusterArray.GetEntriesFast();
    Int_t searchIndexArray[1000]; //array of cluster indices to be used in the tracking
    for(Int_t i = 0; i<1000;i++){
      searchIndexArray[i] = -1;
    }
    Int_t freeClusters = 0; //no of clusters not already associated to a track

    for(Int_t iCluster = iFirst; iCluster<nClusters;iCluster++) {
      Cluster *cluster = 0;
      if (reverse) cluster =  (Cluster*) clusterArray.At(clusterArray.GetEntriesFast()-1-iCluster);
      else cluster =  (Cluster*) clusterArray.At(iCluster);
      //if(cluster->getTrackNr() ==0) {
      if(trackNumbers[iCluster] ==0) {
	if (reverse) searchIndexArray[freeClusters] =clusterArray.GetEntriesFast()-1- iCluster;
	else searchIndexArray[freeClusters] = iCluster;
	freeClusters++;
      }
    }
    if(freeClusters < minClusters) continue;

    Int_t iStartCluster = -1;
    Int_t trackIndexArray[100];
    Int_t nClustersInTrackCand = 0;

    TGraph *trackCandidateGraph = new TGraph();
    TGraph *trackCandidateGraphYZ = new TGraph();
    TGraph *zGraph = new TGraph();


    for(Int_t i = 0; i<100;i++) trackIndexArray[i] = -1;

    iStartCluster = searchIndexArray[0];
    Cluster *startCluster = (Cluster*) clusterArray.At(iStartCluster);

    // Cluster *test = 0;
    //if(reverse) test =(Cluster*) clusterArray.At(clusterArray.GetEntriesFast()-1-iFirst);
    //else test = (Cluster*) clusterArray.At(iFirst);
    //if(startCluster->getTrackNr() != 0) continue;
    if(trackNumbers[iStartCluster] != 0) continue;

//     Double_t oldX = startCluster->GetRow();
    DigitPos pos(startCluster->getCRU(), PadPos(startCluster->getRow(), startCluster->getPadMean()));
    float oldRow = pos.getPadSecPos().getPadPos().getRow();
    float oldY = pos.getPadSecPos().getPadPos().getPad();
    //Double_t oldY = startCluster->GetPad();
    float oldZ = startCluster->getTimeMean();
    //Double_t oldZ = startCluster->GetTimeBinW();
    //Int_t oldRow = Int_t(startCluster->GetRow());
    const CRU cru(startCluster->getCRU());

    const PadRegionInfo& region = mapper.getPadRegionInfo(cru.region());
    const int rowInSector       = startCluster->getRow() + region.getGlobalRowOffset();
    const GlobalPadNumber pad   = mapper.globalPadNumber(PadPos(rowInSector, startCluster->getPadMean()));
    const PadCentre& padCentre  = mapper.padCentre(pad);
    const float localYfactor    = (cru.side()==Side::A)?-1.f:1.f;

    LocalPosition3D clusLoc(padCentre.getX(), localYfactor*padCentre.getY(), zPosition);
    trackIndexArray[0] = iStartCluster;
    nClustersInTrackCand = 1;
    Int_t totalMisses = 0;
    Int_t currentSearchRow = oldRow + 1;
    trackCandidateGraph->SetPoint(0,clusLoc.getX(),clusLoc.getY());
    trackCandidateGraphYZ->SetPoint(0,clusLoc.getY(),startCluster->getTimeMean());
    zGraph->SetPoint(0,clusLoc.getX(),startCluster->getTimeMean());
    Bool_t candidateFound = kFALSE;
    const Double_t searchWindow = fWindowPad;
    const Double_t timeWindow = fWindowTime;


    for(Int_t iCluster = 1; iCluster < freeClusters; iCluster++) {


      Cluster *currentCluster = (Cluster*) clusterArray.At(searchIndexArray[iCluster]);
      //DigitPos currentpos(currentCluster->getCRU(), PadPos(currentCluster->getRow(), currentCluster->getPadMean()));
      //Int_t clusterRow = currentpos.getPadSecPos().getPadPos().getRow();
      const PadRegionInfo& region = mapper.getPadRegionInfo(currentCluster->getCRU());
      Int_t clusterRow = currentCluster->getRow() + region.getGlobalRowOffset();
      if(clusterRow == oldRow) continue;
      //if(clusterRow > currentSearchRow) {
      if(clusterRow != currentSearchRow) {
        totalMisses=totalMisses+(TMath::Abs(clusterRow-oldRow)-1);
        currentSearchRow = clusterRow;
      }
      Double_t closest = 1000.;
      Double_t zClosest = 1000.;
      Int_t iClosest = -1;

      while (clusterRow == currentSearchRow && iCluster < freeClusters ) { //search currentSearchRow to find the cluster closest to the last accepted cluster


        Double_t currentZ = currentCluster -> getTimeMean();
        Double_t currentY = currentCluster -> getPadMean(); //currentpos.getPadSecPos().getPadPos().getPad();

        Double_t distance;
        distance = (TMath::Abs(oldY - currentY));

        Double_t zDistance = (TMath::Abs(oldZ - currentZ));

        if(distance < closest && zDistance<=timeWindow) {

          closest = distance;
          zClosest = zDistance;
          iClosest = iCluster;
        }
        if (searchIndexArray[iCluster+1]==-1) break;
        currentCluster = (Cluster*) clusterArray.At(searchIndexArray[iCluster+1]);
        DigitPos newcurrentpos(currentCluster->getCRU(), PadPos(currentCluster->getRow(), currentCluster->getPadMean()));

        clusterRow = Int_t(newcurrentpos.getPadSecPos().getPadPos().getRow());
        if (clusterRow == currentSearchRow) iCluster++;
      }

      Int_t miss = TMath::Abs(currentSearchRow-oldRow)-1;
      if(miss>0) totalMisses++;
      if(nClustersInTrackCand<6 && miss>1) break;//seems to lower risk for bad seed cluster being accepted
      if(miss>allowedMisses)  break;


      //Now check if cluster is close enough
      if( closest <=searchWindow && zClosest <=timeWindow){
        trackIndexArray[nClustersInTrackCand] = searchIndexArray[iClosest];
        nClustersInTrackCand++;

        Cluster *acceptedCluster = (Cluster*) clusterArray.At(searchIndexArray[iClosest]);
        DigitPos acceptedpos(currentCluster->getCRU(), PadPos(currentCluster->getRow(), currentCluster->getPadMean()));

        // 	oldX = acceptedCluster->GetRow();
        oldY = acceptedpos.getPadSecPos().getPadPos().getPad();
        oldZ = acceptedCluster->getTimeMean();
        oldRow = Int_t(acceptedpos.getPadSecPos().getPadPos().getRow());
        const CRU acceptedcru(acceptedCluster->getCRU());

        const PadRegionInfo& acceptedregion = mapper.getPadRegionInfo(acceptedcru.region());
        const int rowInSector       = acceptedCluster->getRow() + acceptedregion.getGlobalRowOffset();
        const GlobalPadNumber acceptedpad   = mapper.globalPadNumber(PadPos(rowInSector, acceptedCluster->getPadMean()));
        const PadCentre& acceptedpadCentre  = mapper.padCentre(acceptedpad);
        const float acceptedlocalYfactor    = (acceptedcru.side()==Side::A)?-1.f:1.f;

        LocalPosition3D acceptedclusLoc(acceptedpadCentre.getX(), acceptedlocalYfactor*acceptedpadCentre.getY(), zPosition);
        GlobalPosition3D acceptedclusGlob = Mapper::LocalToGlobal(acceptedclusLoc, acceptedcru.sector());



        trackCandidateGraph->SetPoint(trackCandidateGraph->GetN(),acceptedclusLoc.getX(),acceptedclusLoc.getY());
        trackCandidateGraphYZ->SetPoint(trackCandidateGraph->GetN(),acceptedclusLoc.getY(), acceptedCluster->getTimeMean());
        zGraph->SetPoint(zGraph->GetN(),acceptedclusLoc.getX(),acceptedCluster->getTimeMean());
      }

      if(reverse) currentSearchRow--;
      else currentSearchRow++;

      if(nClustersInTrackCand>=minClusters) candidateFound = kTRUE;
    }

    if(candidateFound) {

      float slope=0., offset=0., chi2=0., slopeError=0., offsetError=0.;
      float slopeZ=0., offsetZ=0., chi2Z=0., slopeErrorZ=0., offsetErrorZ=0.;
      GetFit(trackCandidateGraph, slope, offset, slopeError, offsetError, chi2);
      GetFit(trackCandidateGraph, slope, offset, slopeError, offsetError, chi2);
      GetFit(zGraph, slopeZ, offsetZ, slopeErrorZ, offsetErrorZ, chi2Z);

     if(loopcounter == 0) {
    	TCanvas *c1 = new TCanvas(); 	       
	trackCandidateGraph->Draw("APE");
	c1->Update();
        c1->Print("test_cand_1_xy.root");
        TCanvas *c2 = new TCanvas(); 	       
	trackCandidateGraphYZ->Draw("APE");
	c2->Update();
        c2->Print("test_cand_1_yz.root");
        TCanvas *c3 = new TCanvas(); 	       
	zGraph->Draw("APE");
	c3->Update();
        c3->Print("test_cand_1_xz.root");
      }
      if(loopcounter == 1) {
    	TCanvas *c4 = new TCanvas(); 	       
	trackCandidateGraph->Draw("APE");
	c4->Update();
        c4->Print("test_cand_2_xy.root");
        TCanvas *c5 = new TCanvas(); 	       
	trackCandidateGraphYZ->Draw("APE");
	c5->Update();
        c5->Print("test_cand_2_yz.root");
        TCanvas *c6 = new TCanvas(); 	       
	zGraph->Draw("APE");
	c6->Update();
        c6->Print("test_cand_2_xz.root");
      }
      if(loopcounter == 2) {
        TCanvas *c7 = new TCanvas(); 	       
	trackCandidateGraph->Draw("APE");
	c7->Update();
        c7->Print("test_cand_3_xy.root");
        TCanvas *c8 = new TCanvas(); 	       
	trackCandidateGraphYZ->Draw("APE");
	c8->Update();
        c8->Print("test_cand_3_yz.root");
        TCanvas *c9 = new TCanvas(); 	       
	zGraph->Draw("APE");
	c9->Update();
        c9->Print("test_cand_3_xz.root");
      }

      ++loopcounter;
      TrackTPC trackTPC(offset, slope, {offsetError, slopeError, chi2, offsetZ, slopeZ}, {0, 0});
      arrTracks.push_back(trackTPC);
      TrackTPC & storedTrack = arrTracks.back();

      for(Int_t i = 0; i<nClustersInTrackCand;i++) {


        Cluster *chosen = (Cluster*) clusterArray.At(trackIndexArray[i]);
        trackNumbers[trackIndexArray[i]]=nTracks+1;
        //chosen->setTrackNr(nTracks+1);


//        if(fRunResUBiased){

//          TGraph tempGraphXY(*trackCandidateGraph);
//          TGraph tempGraphXZ(*zGraph);
//          tempGraphXY.RemovePoint(i);
//          tempGraphXZ.RemovePoint(i);

//          Double_t slopeTemp=0., offsetTemp=0., chi2Temp=0., slopeErrorTemp=0., offsetErrorTemp=0.;
//          Double_t slopeZTemp=0., offsetZTemp=0., chi2ZTemp=0., slopeErrorZTemp=0., offsetErrorZTemp=0.;

//          GetFit(&tempGraphXY, slopeTemp, offsetTemp, slopeErrorTemp, offsetErrorTemp, chi2Temp);
//          GetFit(&tempGraphXZ, slopeZTemp, offsetZTemp, slopeErrorZTemp, offsetErrorZTemp, chi2ZTemp);

//          chosen->SetResidualYUnBiased(slopeTemp*chosen->GetX() + offsetTemp - chosen->GetY());

//        }

        storedTrack.addCluster(*chosen);

        ++nclustersUsed;
      }

//       trackCandidateGraph->Delete();
      nTracks++;
    }
    delete zGraph;
    delete trackCandidateGraph;
    delete trackCandidateGraphYZ;
  }
  std::cout << arrTracks.size() << " tracks found \n";
}


void RawClusterFinder::GetFit(TGraph *trackGraph, float& slope, float& offset, float& slopeError, float& offsetError, float& chi2) {
   //printf("begin of get fit");
   TLinearFitter fitter(1,"pol1","D");
   //printf("A");
   fitter.AssignData(trackGraph->GetN(), 1, trackGraph->GetX(), trackGraph->GetY());
   //printf("B");
   fitter . Eval();//Robust(0.8); //at least 80% good points, return 0 if fit is ok
   //printf("C");
   fitter . Chisquare();
   chi2 = fitter.GetChisquare();
   offset = fitter.GetParameter(0);
   slope = fitter.GetParameter(1);
   slopeError = fitter.GetParError(1);
   offsetError = fitter.GetParError(0);
   //printf("D");
   //if(degree > 1) track->SetP2(fitter.GetParameter(2));
   //if(degree > 2) track->SetP3(fitter.GetParameter(3));

//    fitter.Delete();
    //printf("end of get fit");
}


} // namespace TPC

} // namespace o2
#endif

void RawClusterFinder(TString fileInfo, TString pedestalFile, TString outputFileName="clusters.root", Int_t maxEvents=-1, TString cherenkovFile="cherenkov.txt", o2::TPC::RawClusterFinder::ClustererType clustererType=o2::TPC::RawClusterFinder::ClustererType::HW)
{
   using namespace o2::TPC;
   RawClusterFinder::ProcessEvents(fileInfo, pedestalFile, outputFileName, maxEvents, cherenkovFile, clustererType);
}
