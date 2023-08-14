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

/// \file PtResolution.cxx.cxx
/// \brief Study on Pt resolution per track in the ITS
/// \author Roberta Ferioli roberta.ferioli@cern.ch
#include <ITSStudies/PtResolution.h>
#include "ITSStudies/ITSStudiesConfigParam.h"

#include "DataFormatsITS/TrackITS.h"
#include "SimulationDataFormat/MCTrack.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"
#include "CommonUtils/TreeStreamRedirector.h"

#include "Framework/Task.h"
#include "Steer/MCKinematicsReader.h"
#include "ITSBase/GeometryTGeo.h"
#include "DetectorsBase/GRPGeomHelper.h"

#include <TH1D.h>
#include <TF1.h>
#include <TH2D.h>
#include <TCanvas.h>
#include <TEfficiency.h>
#include <TStyle.h>
#include <TLegend.h>
#include <TGraphErrors.h>
namespace o2
{
namespace its
{
namespace study
{
using namespace o2::framework;
using namespace o2::globaltracking;

using GTrackID = o2::dataformats::GlobalTrackID;
using o2::steer::MCKinematicsReader;
class PtResolutionStudy : public Task
{
  struct ParticleInfo {
    int event;
    int pdg;
    float pt;
    float eta;
    float phi;
    int mother;
    int first;
    float vx;
    float vy;
    float vz;
    unsigned short clusters = 0u;
    unsigned char isReco = 0u;
    unsigned char isFake = 0u;
    bool isPrimary = 0u;
    unsigned char storedStatus = 2; /// not stored = 2, fake = 1, good = 0
    o2::its::TrackITS track;
  };

 public:
  PtResolutionStudy(std::shared_ptr<DataRequest> dr,
                  mask_t src,
                  bool useMC,
                  std::shared_ptr<o2::steer::MCKinematicsReader> kineReader,
                  std::shared_ptr<o2::base::GRPGeomRequest> gr) : mDataRequest(dr), mTracksSrc(src), mKineReader(kineReader), mGGCCDBRequest(gr)
  {
    if (useMC) {
      LOGP(info, "Read MCKine reader with {} sources", mKineReader->getNSources());
    }
  }
  ~PtResolutionStudy() final = default;
  void init(InitContext&) final;
  void run(ProcessingContext&) final;
  void endOfStream(EndOfStreamContext&) final;
  void finaliseCCDB(ConcreteDataMatcher&, void*) final;
  void initialiseRun(o2::globaltracking::RecoContainer&);
  void process();

 private:
  void updateTimeDependentParams(ProcessingContext& pc);
  std::string mOutFileName = "TrackPtResolutionStudy.root";
  std::shared_ptr<MCKinematicsReader> mKineReader;
  GeometryTGeo* mGeometry;

  // Spans
  gsl::span<const o2::itsmft::ROFRecord> mTracksROFRecords;
  gsl::span<const o2::its::TrackITS> mTracks;
  gsl::span<const o2::MCCompLabel> mTracksMCLabels;
  gsl::span<const o2::itsmft::CompClusterExt> mClusters;
  const o2::dataformats::MCLabelContainer* mClustersMCLCont;

  // Data
  GTrackID::mask_t mTracksSrc{};
  std::shared_ptr<DataRequest> mDataRequest;
  std::vector<std::vector<std::vector<ParticleInfo>>> mParticleInfo; // src/event/track
  unsigned short mMask = 0x7f;

  // Utils
  std::shared_ptr<o2::base::GRPGeomRequest> mGGCCDBRequest;

  // Histos
  std::unique_ptr<TH1D> mPtResolution;
  std::unique_ptr<TH2D> mPtResolution2D;
  std::unique_ptr<TH1D> mPtResolutionSec;
  std::unique_ptr<TH1D> mPtResolutionPrim;
  std::unique_ptr<TH2D> mPtResolutionSec2D;
  std::unique_ptr<TGraphErrors> g1;

  //Canvas & decorations
  std::unique_ptr<TLegend> mLegendPt;
  std::unique_ptr<TLegend> mLegendPt4;
  std::unique_ptr<TCanvas> mCanvasPt;
  std::unique_ptr<TCanvas> mCanvasPt2;
  std::unique_ptr<TCanvas> mCanvasPt3;
  std::unique_ptr<TCanvas> mCanvasPt4;
  // Debug output tree
  std::unique_ptr<o2::utils::TreeStreamRedirector> mDBGOut;
  double sigma[100];
  double sigmaerr[100];
  double meanPt[100];
  double aa[100];
  int bb=0;
};


  void PtResolutionStudy::init(InitContext& ic)
{
  o2::base::GRPGeomHelper::instance().setRequest(mGGCCDBRequest);

  auto& pars = o2::its::study::ITSPtTracksResParamConfig::Instance();
  mOutFileName = pars.outFileName;
  mMask = pars.trackLengthMask;

  
  mPtResolution = std::make_unique<TH1D>("PtResolution", ";#it{p}_{T} ;Den", 100, -1,1);
  mPtResolutionSec = std::make_unique<TH1D>("PtResolutionSec", ";#it{p}_{T} ;Den", 100, -1,1);
  mPtResolutionPrim = std::make_unique<TH1D>("PtResolutionPrim", ";#it{p}_{T} ;Den", 100, -1,1);
  mPtResolution2D = std::make_unique<TH2D>("#it{p}_{T} Resolution vs #it{p}_{T}", ";#it{p}_{T} (GeV/#it{c});#Delta p_{T}/p_{T_{MC}", 100, 0,10,100,-1,1);
  mPtResolutionSec2D=std::make_unique<TH2D>("#it{p}_{T} Resolution vs #it{p}_{T} sec ", ";#it{p}_{T} (GeV/#it{c});#Delta p_{T}/p_{T_{MC}", 100, 0,10,100,-1,1);
  mPtResolution->Sumw2();
  mPtResolutionSec->Sumw2();
  mPtResolutionPrim->Sumw2();

}

void PtResolutionStudy::run(ProcessingContext& pc)
{
  o2::globaltracking::RecoContainer recoData;
  recoData.collectData(pc, *mDataRequest.get());
  LOGP(info,"*****RUN*****");
  updateTimeDependentParams(pc); // Make sure this is called after recoData.collectData, which may load some conditions
  initialiseRun(recoData);
  process();
}

void PtResolutionStudy::initialiseRun(o2::globaltracking::RecoContainer& recoData)
{
  mTracksROFRecords = recoData.getITSTracksROFRecords();
  mTracks = recoData.getITSTracks();
  mTracksMCLabels = recoData.getITSTracksMCLabels();
  LOGP(info,"***** INITI RUN*****");
  LOGP(info, "** Found in {} rofs:\n\t-  {} tracks with {} labels",
       mTracksROFRecords.size(), mTracks.size(), mTracksMCLabels.size());
  LOGP(info, "** Found {} sources from kinematic files", mKineReader->getNSources());
}

void PtResolutionStudy::process()
{
  LOGP(info, "** Filling particle table ... ");
  mParticleInfo.resize(mKineReader->getNSources()); // sources
  for (int iSource{0}; iSource < mKineReader->getNSources(); ++iSource) {
    mParticleInfo[iSource].resize(mKineReader->getNEvents(iSource)); // events
    for (int iEvent{0}; iEvent < mKineReader->getNEvents(iSource); ++iEvent) {
      mParticleInfo[iSource][iEvent].resize(mKineReader->getTracks(iSource, iEvent).size()); // tracks
      for (auto iPart{0}; iPart < mKineReader->getTracks(iEvent).size(); ++iPart) {
        auto& part = mKineReader->getTracks(iSource, iEvent)[iPart];
        mParticleInfo[iSource][iEvent][iPart].event = iEvent;
        mParticleInfo[iSource][iEvent][iPart].pdg = part.GetPdgCode();
        mParticleInfo[iSource][iEvent][iPart].pt = part.GetPt();
        mParticleInfo[iSource][iEvent][iPart].phi = part.GetPhi();
        mParticleInfo[iSource][iEvent][iPart].eta = part.GetEta();
        mParticleInfo[iSource][iEvent][iPart].isPrimary = part.isPrimary();
        mParticleInfo[iSource][iEvent][iPart].vx = part.Vx();
        mParticleInfo[iSource][iEvent][iPart].vy = part.Vy();
        mParticleInfo[iSource][iEvent][iPart].vz = part.Vz();
        mParticleInfo[iSource][iEvent][iPart].first = part.getFirstDaughterTrackId();
      }
    }
  }
  LOGP(info, "** Analysing tracks ... ");
  int  good{0}, fakes{0}, total{0};
  for (auto iTrack{0}; iTrack < mTracks.size(); ++iTrack) {
    auto& lab = mTracksMCLabels[iTrack];
    if (!lab.isSet() || lab.isNoise()) {
      continue;
    }
    int trackID, evID, srcID;
    bool fake;
    const_cast<o2::MCCompLabel&>(lab).get(trackID, evID, srcID, fake);
    bool pass{true};

    if (srcID == 99) { // skip QED
      continue;
    }

    mParticleInfo[srcID][evID][trackID].isReco += !fake;
    mParticleInfo[srcID][evID][trackID].isFake += fake;
    if (mTracks[iTrack].isBetter(mParticleInfo[srcID][evID][trackID].track, 1.e9)) {
      mParticleInfo[srcID][evID][trackID].storedStatus = fake;
      mParticleInfo[srcID][evID][trackID].track = mTracks[iTrack];
    }
    fakes += fake;
    good += !fake;
  }
  LOGP(info, "** Analysing pT resolution...");
  for (auto iTrack{0}; iTrack < mTracks.size(); ++iTrack) {
        auto& lab = mTracksMCLabels[iTrack];
        if (!lab.isSet() || lab.isNoise()) continue;
        int trackID, evID, srcID;
        bool fake;
        const_cast<o2::MCCompLabel&>(lab).get(trackID, evID, srcID, fake);
        bool pass{true};
        if (srcID == 99) continue;// skip QED
        //PtResVec[iTrack]=(mParticleInfo[srcID][evID][trackID].pt-mTracks[iTrack].getPt())/mParticleInfo[srcID][evID][trackID].pt;
        mPtResolution->Fill((mParticleInfo[srcID][evID][trackID].pt-mTracks[iTrack].getPt())/mParticleInfo[srcID][evID][trackID].pt);
        mPtResolution2D->Fill(mParticleInfo[srcID][evID][trackID].pt,(mParticleInfo[srcID][evID][trackID].pt-mTracks[iTrack].getPt())/mParticleInfo[srcID][evID][trackID].pt);
        if(!mParticleInfo[srcID][evID][trackID].isPrimary)  mPtResolutionSec->Fill((mParticleInfo[srcID][evID][trackID].pt-mTracks[iTrack].getPt())/mParticleInfo[srcID][evID][trackID].pt);
        if(mParticleInfo[srcID][evID][trackID].isPrimary)  mPtResolutionPrim->Fill((mParticleInfo[srcID][evID][trackID].pt-mTracks[iTrack].getPt())/mParticleInfo[srcID][evID][trackID].pt);
   }
    for(int yy=0;yy<100;yy++)
   {
       aa[yy]=0.;
       sigma[yy]=0.;
       sigmaerr[yy]=0.;
       meanPt[yy]=0.;
   }
  
   for(int yy=0;yy<100;yy++)
   {
       TH1D * projh2X = mPtResolution2D->ProjectionY("projh2X",yy,yy+1,"");
       TF1 *f1 = new TF1("f1","gaus",-0.2,0.2);
       projh2X->Fit("f1");
       if(f1->GetParameter(2)>0. && f1->GetParameter(2)<1. && f1->GetParameter(1)<1.)
       {
       sigma[yy]=f1->GetParameter(2);
       sigmaerr[yy]=f1->GetParError(2);
       meanPt[yy]=((8./100.)*yy+(8./100.)*(yy+1))/2;
       aa[yy]=0.0125;
       }
   }
   
    


}

void PtResolutionStudy::updateTimeDependentParams(ProcessingContext& pc)
{
  static bool initOnceDone = false;
  o2::base::GRPGeomHelper::instance().checkUpdates(pc);
  if (!initOnceDone) { // this params need to be queried only once
    initOnceDone = true;
    mGeometry = GeometryTGeo::Instance();
    mGeometry->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L, o2::math_utils::TransformType::T2GRot, o2::math_utils::TransformType::T2G));
  }
}
void PtResolutionStudy::endOfStream(EndOfStreamContext& ec)
{
  TFile fout(mOutFileName.c_str(), "recreate");
  mPtResolution->SetName("#it{p}_{T} resolution");
  mPtResolution->SetTitle(";#Delta p_{T}/p_{T_{MC}} ;Entries");
  mPtResolution->SetFillColor(kAzure + 4);
  mPtResolutionPrim->SetFillColor(kRed);
  mPtResolutionSec->SetFillColor(kOrange);
  mPtResolutionPrim->SetTitle(";#Delta p_{T}/p_{T_{MC}} ;Entries");
  mPtResolutionSec->SetTitle(";#Delta #it{p}_{T}/#it{p}_{T_{MC}} ;Entries");
  mPtResolution2D->SetTitle(";#it{p}_{T_{MC}} [GeV];#Delta #it{p}_{T}/#it{p}_{T_{MC}}");
  
  
  fout.WriteTObject(mPtResolution.get());
  fout.WriteTObject(mPtResolutionPrim.get());
  fout.WriteTObject(mPtResolutionSec.get());
  fout.WriteTObject(mPtResolution2D.get());
  
  mCanvasPt = std::make_unique<TCanvas>("cPt", "cPt", 1600, 1200);
  mCanvasPt->cd();
  mPtResolution->Draw("HIST");
  mLegendPt = std::make_unique<TLegend>(0.19, 0.8, 0.40, 0.96);
  mLegendPt->SetHeader(Form("%zu events PP min bias", mKineReader->getNEvents(0)), "C");
  mLegendPt->AddEntry("mPtResolution", "All events", "lep");
  mLegendPt->Draw();
  mCanvasPt->SaveAs("ptRes.png");
  fout.cd();
  mCanvasPt->Write();
  mCanvasPt2 = std::make_unique<TCanvas>("cPt2", "cPt2", 1600, 1200);
  mCanvasPt2->cd();
  mPtResolution2D->Draw();
  mCanvasPt2->SaveAs("ptRes2.png");
  fout.cd();
  mCanvasPt2->Write();
  mCanvasPt3 = std::make_unique<TCanvas>("cPt3", "cPt3", 1600, 1200);
  mCanvasPt3->cd();
 
  TGraphErrors *g1 = new TGraphErrors(100,meanPt,sigma,aa,sigmaerr);
  g1->SetMarkerStyle(8);
  g1->SetMarkerColor(kGreen);
  g1->GetXaxis()->SetTitle("Pt [GeV]");
  g1->GetYaxis()->SetTitle("#sigma #Delta #it{p}_{T}/#it{p}_{T_{MC}}");
  g1->GetYaxis()->SetLimits(0,1);
  g1->GetXaxis()->SetLimits(0,10.);
  g1->Draw("AP");
  g1->GetYaxis()->SetRangeUser(0, 1);
  g1->GetXaxis()->SetRangeUser(0, 10.);
  mCanvasPt3->SaveAs("ptRes3.png");
  fout.cd();
  mCanvasPt3->Write();

 mCanvasPt4 = std::make_unique<TCanvas>("cPt4", "cPt4", 1600, 1200);
 mCanvasPt4->cd();
 mPtResolutionPrim->SetName("mPtResolutionPrim");
 mPtResolutionSec->SetName("mPtResolutionSec");
 mPtResolutionPrim->Draw("same hist");
 mPtResolutionSec->Draw("same hist");
 mLegendPt4 = std::make_unique<TLegend>(0.19, 0.8, 0.40, 0.96);

 mLegendPt4->SetHeader(Form("%zu events PP", mKineReader->getNEvents(0)), "C");
 mLegendPt4->AddEntry("mPtResolutionPrim", "Primary events","f");
 mLegendPt4->AddEntry("mPtResolutionSec", "Secondary events","f");
 mLegendPt4->Draw("same");
 mCanvasPt4->SaveAs("ptRes4.png");
 fout.cd();
 mCanvasPt4->Write();

  fout.Close();
}

void PtResolutionStudy::finaliseCCDB(ConcreteDataMatcher& matcher, void* obj)
{
}

DataProcessorSpec getPtResolutionStudy(mask_t srcTracksMask, mask_t srcClustersMask, bool useMC, std::shared_ptr<o2::steer::MCKinematicsReader> kineReader)
{
  std::vector<OutputSpec> outputs;
  auto dataRequest = std::make_shared<DataRequest>();
  dataRequest->requestTracks(srcTracksMask, useMC);

  auto ggRequest = std::make_shared<o2::base::GRPGeomRequest>(false,                             // orbitResetTime
                                                              true,                              // GRPECS=true
                                                              false,                             // GRPLHCIF
                                                              true,                              // GRPMagField
                                                              true,                              // askMatLUT
                                                              o2::base::GRPGeomRequest::Aligned, // geometry
                                                              dataRequest->inputs,
                                                              true);

  return DataProcessorSpec{
    "its-study-tracks-pt-resolution",
    dataRequest->inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<PtResolutionStudy>(dataRequest, srcTracksMask, useMC, kineReader, ggRequest)},
    Options{}};
}

} // namespace study
} // namespace its
} // namespace o2