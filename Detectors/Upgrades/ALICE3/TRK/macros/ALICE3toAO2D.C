//******************************************************************
// ALICE 3 hits to AO2D converter tool
//
// This macro imports ALICE 3 tracker hits, runs the ITS tracker
// on them and then saves the tracks and primary vertex in a
// AO2D.root file that is compliant with the O2 framework.
//
// More specifically, it mimics Run 2-converted data so that
// any analysis geared towards that can run on the output of this
// conversion tool.
//
// To be used compiled ('root.exe -q -b ALICE3toAO2D.C+')
//
// Comments, complaints, suggestions? Please write to:
// --- david.dobrigkeit.chinellato@cern.ch
//******************************************************************

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <string>
#include <TFile.h>
#include <TChain.h>
#include <TTree.h>
#include <TBranch.h>
#include <TH2D.h>
#include <TProfile.h>
#include <TBranch.h>
#include <TRandom3.h>
#include <TGeoGlobalMagField.h>
#include <vector>
#include <TTimeStamp.h>
#include "DataFormatsITSMFT/ROFRecord.h"
#include "ITSMFTSimulation/Hit.h"
#include "ITStracking/Configuration.h"
#include "ITStracking/IOUtils.h"
#include "ITStracking/Tracker.h"
#include "ITStracking/TrackerTraitsCPU.h"
#include "ITStracking/TimeFrame.h"
#include "ITStracking/Vertexer.h"
#include "ITStracking/VertexerTraits.h"
#include "DataFormatsITSMFT/Cluster.h"
#include "DataFormatsITSMFT/TopologyDictionary.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "SimulationDataFormat/MCTrack.h"
#include "MathUtils/Cartesian.h"
#include "ReconstructionDataFormats/TrackParametrization.h"
#include "ReconstructionDataFormats/TrackParametrizationWithError.h"
#include "SimulationDataFormat/MCEventHeader.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "ReconstructionDataFormats/DCA.h"
#include "ReconstructionDataFormats/Vertex.h"
#include "Framework/DataTypes.h"
#include "UpgradesAODUtils/Run2LikeAO2D.h"
#endif

using o2::its::MemoryParameters;
using o2::its::TrackingParameters;
using o2::itsmft::Hit;
using std::string;

using Vertex = o2::dataformats::Vertex<o2::dataformats::TimeStamp<int>>;

constexpr bool kUseSmearing{true};

namespace o2::upgrades_utils
{
float getDetLengthFromEta(const float eta, const float radius)
{
  return 10. * (10. + radius * std::cos(2 * std::atan(std::exp(-eta))));
}

//+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+

void ALICE3toAO2D()
{
  std::cout << "\e[1;31m***********************************************\e[0;00m" << std::endl;
  std::cout << "\e[1;31m      ALICE 3 hits to AO2D converter tool      \e[0;00m" << std::endl;
  std::cout << "\e[1;31m                12-layer version               \e[0;00m" << std::endl;
  std::cout << "\e[1;31m***********************************************\e[0;00m" << std::endl;

  std::cout << "*- Starting..." << std::endl;
  const string hitsFileName = "o2sim_HitsTRK.root";
  TChain mcTree("o2sim");
  mcTree.AddFile("o2sim_Kine.root");
  mcTree.SetBranchStatus("*", 0); //disable all branches
  mcTree.SetBranchStatus("MCTrack*", 1);
  mcTree.SetBranchStatus("MCEventHeader.", 1);

  std::vector<o2::MCTrack>* mcArr = nullptr;
  mcTree.SetBranchAddress("MCTrack", &mcArr);

  //o2::dataformats::MCEventHeader* mcHead;
  //FairMCEventHeader *mcHead;
  auto mcHead = new o2::dataformats::MCEventHeader;
  mcTree.SetBranchAddress("MCEventHeader.", &mcHead);

  o2::its::Vertexer vertexer(new o2::its::VertexerTraits());

  TChain itsHits("o2sim");

  itsHits.AddFile(hitsFileName.data());

  o2::its::TimeFrame tf;
  o2::its::Tracker tracker(new o2::its::TrackerTraitsCPU(&tf));
  tracker.setBz(5.f);

  std::uint32_t roFrame;
  std::vector<Hit>* hits = nullptr;
  itsHits.SetBranchAddress("TRKHit", &hits);

  std::vector<TrackingParameters> trackParams(4);

  //Tracking parameters for 12 layer setup
  trackParams[0].NLayers = 12;
  trackParams[0].MinTrackLength = 12; //this is the one with fixed params
  std::vector<float> LayerRadii = {0.5f, 1.2f, 2.5f, 3.75f, 7.0f, 12.0f, 20.0f, 30.0f, 45.0f, 60.0f, 80.0f, 100.0f};
  std::vector<float> LayerZ(12);
  for (int i{0}; i < 12; ++i)
    LayerZ[i] = getDetLengthFromEta(1.44, LayerRadii[i]) + 1.;

  //loosely based on run_trac_alice3.C but with extra stuff for the innermost layers
  //FIXME: This may be subject to further tuning and is only a first guess
  std::vector<float> TrackletMaxDeltaZ = {0.1f, 0.1f, 0.1f, 0.1f, 0.3f, 0.3f, 0.3f, 0.3f, 0.5f, 0.5f, 0.5f};
  std::vector<float> CellMaxDCA = {0.05f, 0.05f, 0.05f, 0.04f, 0.05f, 0.2f, 0.4f, 0.5f, 0.5f, 0.5f};
  std::vector<float> CellMaxDeltaZ = {0.2f, 0.2f, 0.2f, 0.4f, 0.5f, 0.6f, 3.0f, 3.0f, 3.0f, 3.0f};
  std::vector<float> NeighbourMaxDeltaCurvature = {0.012f, 0.010f, 0.008f, 0.0025f, 0.003f, 0.0035f, 0.004f, 0.004f, 0.005f};
  std::vector<float> NeighbourMaxDeltaN = {0.002f, 0.002f, 0.002f, 0.0090f, 0.002f, 0.005f, 0.005f, 0.005f, 0.005f};

  trackParams[0].LayerRadii = LayerRadii;
  trackParams[0].LayerZ = LayerZ;
  trackParams[0].TrackletMaxDeltaPhi = 0.3;
  trackParams[0].CellMaxDeltaPhi = 0.15;
  trackParams[0].CellMaxDeltaTanLambda = 0.03;
  trackParams[0].TrackletMaxDeltaZ = TrackletMaxDeltaZ;
  trackParams[0].CellMaxDCA = CellMaxDCA;
  trackParams[0].CellMaxDeltaZ = CellMaxDeltaZ;
  trackParams[0].NeighbourMaxDeltaCurvature = NeighbourMaxDeltaCurvature;
  trackParams[0].NeighbourMaxDeltaN = NeighbourMaxDeltaN;

  std::vector<MemoryParameters> memParams(4);
  std::vector<float> CellsMemoryCoefficients = {2.3208e-08f * 300, 2.104e-08f * 300, 1.6432e-08f * 300, 1.2412e-08f * 300, 1.3543e-08f * 300, 1.5e-08f * 300, 1.6e-08f * 300, 1.7e-08f * 300};
  std::vector<float> TrackletsMemoryCoefficients = {0.0016353f * 15000, 0.0013627f * 15000, 0.000984f * 15000, 0.00078135f * 15000, 0.00057934f * 15000, 0.00052217f * 15000, 0.00052217f * 15000, 0.00052217f * 15000, 0.00052217f * 15000};
  memParams[0].CellsMemoryCoefficients = CellsMemoryCoefficients;
  memParams[0].TrackletsMemoryCoefficients = TrackletsMemoryCoefficients;
  memParams[0].MemoryOffset = 120000;

  float loosening = 3.;
  for (int i = 1; i < 4; ++i) {
    memParams[i] = memParams[i - 1];
    trackParams[i] = trackParams[i - 1];
    // trackParams[i].MinTrackLength -= 2;
    trackParams[i].TrackletMaxDeltaPhi = trackParams[i].TrackletMaxDeltaPhi * 3 > TMath::Pi() ? TMath::Pi() : trackParams[i].TrackletMaxDeltaPhi * 3;
    trackParams[i].CellMaxDeltaPhi = trackParams[i].CellMaxDeltaPhi * 3 > TMath::Pi() ? TMath::Pi() : trackParams[i].CellMaxDeltaPhi * 3;
    trackParams[i].CellMaxDeltaTanLambda *= loosening;
    for (auto& val : trackParams[i].TrackletMaxDeltaZ)
      val *= loosening;
    for (auto& val : trackParams[i].CellMaxDCA)
      val *= loosening;
    for (auto& val : trackParams[i].CellMaxDeltaZ)
      val *= loosening;
    for (auto& val : trackParams[i].NeighbourMaxDeltaCurvature)
      val *= loosening;
    for (auto& val : trackParams[i].NeighbourMaxDeltaN)
      val *= loosening;
  }

  tracker.setParameters(memParams, trackParams);

  constexpr int nBins = 100;
  constexpr float minPt = 0.01;
  constexpr float maxPt = 10;
  double newBins[nBins + 1];
  newBins[0] = minPt;
  double factor = pow(maxPt / minPt, 1. / nBins);
  for (int i = 1; i <= nBins; i++) {
    newBins[i] = factor * newBins[i - 1];
  }

  Double_t ptbinlimits[] = {0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2., 2.2, 2.4, 2.6, 2.8, 3.0,
                            3.3, 3.6, 3.9, 4.2, 4.6, 5, 5.4, 5.9, 6.5, 7, 7.5, 8, 8.5, 9.2, 10, 11, 12, 13.5, 15, 17, 20};
  Long_t ptbinnumb = sizeof(ptbinlimits) / sizeof(Double_t) - 1;

  //Debug output
  TH1D* hNVertices = new TH1D("hNVertices", "", 10, 0, 10);
  TH1D* hNTracks = new TH1D("hNTracks", "", 100, 0, 100);

  TH1D* hPtSpectra = new TH1D("hPtSpectra", "", ptbinnumb, ptbinlimits);
  TH1D* hPtSpectraFake = new TH1D("hPtSpectraFake", "", ptbinnumb, ptbinlimits);

  //+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
  std::cout << "*- Setting up output file..." << std::endl;
  // Setup output
  UInt_t fCompress = 101;
  int fBasketSizeEvents = 1000000;                                                // Maximum basket size of the trees for events
  int fBasketSizeTracks = 10000000;                                               // Maximum basket size of the trees for tracks
  TFile* fOutputFile = TFile::Open("AO2D.root", "RECREATE", "O2 AOD", fCompress); // File to store the trees of time frames

  //setup timestamp for output
  TTimeStamp ts0(2020, 11, 1, 0, 0, 0);
  TTimeStamp ts1;
  UInt_t tfId = ts1.GetSec() - ts0.GetSec();

  // Create the output directory for the current time frame
  TDirectory* fOutputDir = 0x0; ///! Pointer to the output Root subdirectory
  fOutputDir = fOutputFile->mkdir(Form("DF_%d", tfId));
  fOutputDir->cd();

  //Create output trees in file
  TTree* fTree[kTrees];
  for (Int_t ii = 0; ii < kTrees; ii++) {
    if (gSaveTree[ii]) {
      std::cout << "*- Creating tree " << gTreeName[ii] << "..." << std::endl;
      fTree[ii] = new TTree(gTreeName[ii], gTreeTitle[ii]);
      fTree[ii]->SetAutoFlush(0);
    }
  }
  if (gSaveTree[kEvents]) {
    fTree[kEvents]->Branch("fBCsID", &collision.fBCsID, "fBCsID/I");
    fTree[kEvents]->Branch("fPosX", &collision.fPosX, "fPosX/F");
    fTree[kEvents]->Branch("fPosY", &collision.fPosY, "fPosY/F");
    fTree[kEvents]->Branch("fPosZ", &collision.fPosZ, "fPosZ/F");
    fTree[kEvents]->Branch("fCovXX", &collision.fCovXX, "fCovXX/F");
    fTree[kEvents]->Branch("fCovXY", &collision.fCovXY, "fCovXY/F");
    fTree[kEvents]->Branch("fCovXZ", &collision.fCovXZ, "fCovXZ/F");
    fTree[kEvents]->Branch("fCovYY", &collision.fCovYY, "fCovYY/F");
    fTree[kEvents]->Branch("fCovYZ", &collision.fCovYZ, "fCovYZ/F");
    fTree[kEvents]->Branch("fCovZZ", &collision.fCovZZ, "fCovZZ/F");
    fTree[kEvents]->Branch("fChi2", &collision.fChi2, "fChi2/F");
    fTree[kEvents]->Branch("fNumContrib", &collision.fN, "fNumContrib/i");
    fTree[kEvents]->Branch("fCollisionTime", &collision.fCollisionTime, "fCollisionTime/F");
    fTree[kEvents]->Branch("fCollisionTimeRes", &collision.fCollisionTimeRes, "fCollisionTimeRes/F");
    fTree[kEvents]->Branch("fCollisionTimeMask", &collision.fCollisionTimeMask, "fCollisionTimeMask/b");
    fTree[kEvents]->SetBasketSize("*", fBasketSizeEvents);
  }

  if (gSaveTree[kTracks]) {
    fTree[kTracks]->Branch("fCollisionsID", &tracks.fCollisionsID, "fCollisionsID/I");
    fTree[kTracks]->Branch("fTrackType", &tracks.fTrackType, "fTrackType/b");
    //    fTree[kTracks]->Branch("fTOFclsIndex", &tracks.fTOFclsIndex, "fTOFclsIndex/I");
    //    fTree[kTracks]->Branch("fNTOFcls", &tracks.fNTOFcls, "fNTOFcls/I");
    fTree[kTracks]->Branch("fX", &tracks.fX, "fX/F");
    fTree[kTracks]->Branch("fAlpha", &tracks.fAlpha, "fAlpha/F");
    fTree[kTracks]->Branch("fY", &tracks.fY, "fY/F");
    fTree[kTracks]->Branch("fZ", &tracks.fZ, "fZ/F");
    fTree[kTracks]->Branch("fSnp", &tracks.fSnp, "fSnp/F");
    fTree[kTracks]->Branch("fTgl", &tracks.fTgl, "fTgl/F");
    fTree[kTracks]->Branch("fSigned1Pt", &tracks.fSigned1Pt, "fSigned1Pt/F");
    // Modified covariance matrix
    fTree[kTracks]->Branch("fSigmaY", &tracks.fSigmaY, "fSigmaY/F");
    fTree[kTracks]->Branch("fSigmaZ", &tracks.fSigmaZ, "fSigmaZ/F");
    fTree[kTracks]->Branch("fSigmaSnp", &tracks.fSigmaSnp, "fSigmaSnp/F");
    fTree[kTracks]->Branch("fSigmaTgl", &tracks.fSigmaTgl, "fSigmaTgl/F");
    fTree[kTracks]->Branch("fSigma1Pt", &tracks.fSigma1Pt, "fSigma1Pt/F");
    fTree[kTracks]->Branch("fRhoZY", &tracks.fRhoZY, "fRhoZY/B");
    fTree[kTracks]->Branch("fRhoSnpY", &tracks.fRhoSnpY, "fRhoSnpY/B");
    fTree[kTracks]->Branch("fRhoSnpZ", &tracks.fRhoSnpZ, "fRhoSnpZ/B");
    fTree[kTracks]->Branch("fRhoTglY", &tracks.fRhoTglY, "fRhoTglY/B");
    fTree[kTracks]->Branch("fRhoTglZ", &tracks.fRhoTglZ, "fRhoTglZ/B");
    fTree[kTracks]->Branch("fRhoTglSnp", &tracks.fRhoTglSnp, "fRhoTglSnp/B");
    fTree[kTracks]->Branch("fRho1PtY", &tracks.fRho1PtY, "fRho1PtY/B");
    fTree[kTracks]->Branch("fRho1PtZ", &tracks.fRho1PtZ, "fRho1PtZ/B");
    fTree[kTracks]->Branch("fRho1PtSnp", &tracks.fRho1PtSnp, "fRho1PtSnp/B");
    fTree[kTracks]->Branch("fRho1PtTgl", &tracks.fRho1PtTgl, "fRho1PtTgl/B");
    //
    fTree[kTracks]->Branch("fTPCInnerParam", &tracks.fTPCinnerP, "fTPCInnerParam/F");
    fTree[kTracks]->Branch("fFlags", &tracks.fFlags, "fFlags/i");
    fTree[kTracks]->Branch("fITSClusterMap", &tracks.fITSClusterMap, "fITSClusterMap/b");
    fTree[kTracks]->Branch("fTPCNClsFindable", &tracks.fTPCNClsFindable, "fTPCNClsFindable/b");
    fTree[kTracks]->Branch("fTPCNClsFindableMinusFound", &tracks.fTPCNClsFindableMinusFound, "fTPCNClsFindableMinusFound/B");
    fTree[kTracks]->Branch("fTPCNClsFindableMinusCrossedRows", &tracks.fTPCNClsFindableMinusCrossedRows, "fTPCNClsFindableMinusCrossedRows/B");
    fTree[kTracks]->Branch("fTPCNClsShared", &tracks.fTPCNClsShared, "fTPCNClsShared/b");
    fTree[kTracks]->Branch("fTRDPattern", &tracks.fTRDPattern, "fTRDPattern/b");
    fTree[kTracks]->Branch("fITSChi2NCl", &tracks.fITSChi2NCl, "fITSChi2NCl/F");
    fTree[kTracks]->Branch("fTPCChi2NCl", &tracks.fTPCChi2NCl, "fTPCChi2NCl/F");
    fTree[kTracks]->Branch("fTRDChi2", &tracks.fTRDChi2, "fTRDChi2/F");
    fTree[kTracks]->Branch("fTOFChi2", &tracks.fTOFChi2, "fTOFChi2/F");
    fTree[kTracks]->Branch("fTPCSignal", &tracks.fTPCSignal, "fTPCSignal/F");
    fTree[kTracks]->Branch("fTRDSignal", &tracks.fTRDSignal, "fTRDSignal/F");
    fTree[kTracks]->Branch("fTOFSignal", &tracks.fTOFSignal, "fTOFSignal/F");
    fTree[kTracks]->Branch("fLength", &tracks.fLength, "fLength/F");
    fTree[kTracks]->Branch("fTOFExpMom", &tracks.fTOFExpMom, "fTOFExpMom/F");
    fTree[kTracks]->Branch("fTrackEtaEMCAL", &tracks.fTrackEtaEMCAL, "fTrackEtaEMCAL/F");
    fTree[kTracks]->Branch("fTrackPhiEMCAL", &tracks.fTrackPhiEMCAL, "fTrackPhiEMCAL/F");
    fTree[kTracks]->SetBasketSize("*", fBasketSizeTracks);
  }

  if (gSaveTree[kMcTrackLabel]) {
    fTree[kMcTrackLabel]->Branch("fLabel", &mctracklabel.fLabel, "fLabel/i");
    fTree[kMcTrackLabel]->Branch("fLabelMask", &mctracklabel.fLabelMask, "fLabelMask/s");
    fTree[kMcTrackLabel]->SetBasketSize("*", fBasketSizeTracks);
  }

  if (gSaveTree[kMcCollision]) {
    fTree[kMcCollision]->Branch("fBCsID", &mccollision.fBCsID, "fBCsID/I");
    fTree[kMcCollision]->Branch("fGeneratorsID", &mccollision.fGeneratorsID, "fGeneratorsID/S");
    fTree[kMcCollision]->Branch("fPosX", &mccollision.fPosX, "fPosX/F");
    fTree[kMcCollision]->Branch("fPosY", &mccollision.fPosY, "fPosY/F");
    fTree[kMcCollision]->Branch("fPosZ", &mccollision.fPosZ, "fPosZ/F");
    fTree[kMcCollision]->Branch("fT", &mccollision.fT, "fT/F");
    fTree[kMcCollision]->Branch("fWeight", &mccollision.fWeight, "fWeight/F");
    fTree[kMcCollision]->Branch("fImpactParameter", &mccollision.fImpactParameter, "fImpactParameter/F");
    fTree[kMcCollision]->SetBasketSize("*", fBasketSizeEvents);
  }

  if (gSaveTree[kMcParticle]) {
    fTree[kMcParticle]->Branch("fMcCollisionsID", &mcparticle.fMcCollisionsID, "fMcCollisionsID/I");
    fTree[kMcParticle]->Branch("fPdgCode", &mcparticle.fPdgCode, "fPdgCode/I");
    fTree[kMcParticle]->Branch("fStatusCode", &mcparticle.fStatusCode, "fStatusCode/I");
    fTree[kMcParticle]->Branch("fFlags", &mcparticle.fFlags, "fFlags/b");
    fTree[kMcParticle]->Branch("fMother0", &mcparticle.fMother0, "fMother0/I");
    fTree[kMcParticle]->Branch("fMother1", &mcparticle.fMother1, "fMother1/I");
    fTree[kMcParticle]->Branch("fDaughter0", &mcparticle.fDaughter0, "fDaughter0/I");
    fTree[kMcParticle]->Branch("fDaughter1", &mcparticle.fDaughter1, "fDaughter1/I");
    fTree[kMcParticle]->Branch("fWeight", &mcparticle.fWeight, "fWeight/F");

    fTree[kMcParticle]->Branch("fPx", &mcparticle.fPx, "fPx/F");
    fTree[kMcParticle]->Branch("fPy", &mcparticle.fPy, "fPy/F");
    fTree[kMcParticle]->Branch("fPz", &mcparticle.fPz, "fPz/F");
    fTree[kMcParticle]->Branch("fE", &mcparticle.fE, "fE/F");

    fTree[kMcParticle]->Branch("fVx", &mcparticle.fVx, "fVx/F");
    fTree[kMcParticle]->Branch("fVy", &mcparticle.fVy, "fVy/F");
    fTree[kMcParticle]->Branch("fVz", &mcparticle.fVz, "fVz/F");
    fTree[kMcParticle]->Branch("fVt", &mcparticle.fVt, "fVt/F");
    fTree[kMcParticle]->SetBasketSize("*", fBasketSizeTracks);
  }

  if (gSaveTree[kMcCollisionLabel]) {
    fTree[kMcCollisionLabel]->Branch("fLabel", &mccollisionlabel.fLabel, "fLabel/i");
    fTree[kMcCollisionLabel]->Branch("fLabelMask", &mccollisionlabel.fLabelMask, "fLabelMask/s");
    fTree[kMcCollisionLabel]->SetBasketSize("*", fBasketSizeEvents);
  }

  if (gSaveTree[kBC]) {
    fTree[kBC]->Branch("fRunNumber", &bc.fRunNumber, "fRunNumber/I");
    fTree[kBC]->Branch("fGlobalBC", &bc.fGlobalBC, "fGlobalBC/l");
    fTree[kBC]->Branch("fTriggerMask", &bc.fTriggerMask, "fTriggerMask/l");
    fTree[kBC]->SetBasketSize("*", fBasketSizeEvents);
  }

  if (gSaveTree[kFDD]) {
    fTree[kFDD]->Branch("fBCsID", &fdd.fBCsID, "fBCsID/I");
    fTree[kFDD]->Branch("fAmplitudeA", fdd.fAmplitudeA, "fAmplitudeA[4]/F");
    fTree[kFDD]->Branch("fAmplitudeC", fdd.fAmplitudeC, "fAmplitudeC[4]/F");
    fTree[kFDD]->Branch("fTimeA", &fdd.fTimeA, "fTimeA/F");
    fTree[kFDD]->Branch("fTimeC", &fdd.fTimeC, "fTimeC/F");
    fTree[kFDD]->Branch("fTriggerMask", &fdd.fTriggerMask, "fTriggerMask/b");
    fTree[kFDD]->SetBasketSize("*", fBasketSizeEvents);
  }

  // Associate branches for V0A
  if (gSaveTree[kFV0A]) {
    fTree[kFV0A]->Branch("fBCsID", &fv0a.fBCsID, "fBCsID/I");
    fTree[kFV0A]->Branch("fAmplitude", fv0a.fAmplitude, "fAmplitude[48]/F");
    fTree[kFV0A]->Branch("fTime", &fv0a.fTime, "fTime/F");
    fTree[kFV0A]->Branch("fTriggerMask", &fv0a.fTriggerMask, "fTriggerMask/b");
    fTree[kFV0A]->SetBasketSize("*", fBasketSizeEvents);
  }

  // Associate branches for V0C
  if (gSaveTree[kFV0C]) {
    fTree[kFV0C]->Branch("fBCsID", &fv0c.fBCsID, "fBCsID/I");
    fTree[kFV0C]->Branch("fAmplitude", fv0c.fAmplitude, "fAmplitude[32]/F");
    fTree[kFV0C]->Branch("fTime", &fv0c.fTime, "fTime/F");
    fTree[kFV0C]->SetBasketSize("*", fBasketSizeEvents);
  }

  // Associate branches for FT0
  if (gSaveTree[kFT0]) {
    fTree[kFT0]->Branch("fBCsID", &ft0.fBCsID, "fBCsID/I");
    fTree[kFT0]->Branch("fAmplitudeA", ft0.fAmplitudeA, "fAmplitudeA[96]/F");
    fTree[kFT0]->Branch("fAmplitudeC", ft0.fAmplitudeC, "fAmplitudeC[112]/F");
    fTree[kFT0]->Branch("fTimeA", &ft0.fTimeA, "fTimeA/F");
    fTree[kFT0]->Branch("fTimeC", &ft0.fTimeC, "fTimeC/F");
    fTree[kFT0]->Branch("fTriggerMask", &ft0.fTriggerMask, "fTriggerMask/b");
    fTree[kFT0]->SetBasketSize("*", fBasketSizeEvents);
  }

  if (gSaveTree[kZdc]) {
    fTree[kZdc]->Branch("fBCsID", &zdc.fBCsID, "fBCsID/I");
    fTree[kZdc]->Branch("fEnergyZEM1", &zdc.fEnergyZEM1, "fEnergyZEM1/F");
    fTree[kZdc]->Branch("fEnergyZEM2", &zdc.fEnergyZEM2, "fEnergyZEM2/F");
    fTree[kZdc]->Branch("fEnergyCommonZNA", &zdc.fEnergyCommonZNA, "fEnergyCommonZNA/F");
    fTree[kZdc]->Branch("fEnergyCommonZNC", &zdc.fEnergyCommonZNC, "fEnergyCommonZNC/F");
    fTree[kZdc]->Branch("fEnergyCommonZPA", &zdc.fEnergyCommonZPA, "fEnergyCommonZPA/F");
    fTree[kZdc]->Branch("fEnergyCommonZPC", &zdc.fEnergyCommonZPC, "fEnergyCommonZPC/F");
    fTree[kZdc]->Branch("fEnergySectorZNA", &zdc.fEnergySectorZNA, "fEnergySectorZNA[4]/F");
    fTree[kZdc]->Branch("fEnergySectorZNC", &zdc.fEnergySectorZNC, "fEnergySectorZNC[4]/F");
    fTree[kZdc]->Branch("fEnergySectorZPA", &zdc.fEnergySectorZPA, "fEnergySectorZPA[4]/F");
    fTree[kZdc]->Branch("fEnergySectorZPC", &zdc.fEnergySectorZPC, "fEnergySectorZPC[4]/F");
    fTree[kZdc]->Branch("fTimeZEM1", &zdc.fTimeZEM1, "fTimeZEM1/F");
    fTree[kZdc]->Branch("fTimeZEM2", &zdc.fTimeZEM2, "fTimeZEM2/F");
    fTree[kZdc]->Branch("fTimeZNA", &zdc.fTimeZNA, "fTimeZNA/F");
    fTree[kZdc]->Branch("fTimeZNC", &zdc.fTimeZNC, "fTimeZNC/F");
    fTree[kZdc]->Branch("fTimeZPA", &zdc.fTimeZPA, "fTimeZPA/F");
    fTree[kZdc]->Branch("fTimeZPC", &zdc.fTimeZPC, "fTimeZPC/F");
    fTree[kZdc]->SetBasketSize("*", fBasketSizeEvents);
  }

  //+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
  Long_t lGoodEvents = 0;
  Long_t fOffsetLabel = 0;
  std::cout << "*- Number of events detected: " << itsHits.GetEntries() << std::endl;
  for (int iEvent{0}; iEvent < itsHits.GetEntriesFast(); ++iEvent) {
    itsHits.GetEntry(iEvent);
    mcTree.GetEvent(iEvent);
    o2::its::ROframe event{iEvent, 12};
    std::cout << "*- Processing event " << iEvent << "..." << std::endl;

    int id{0};

    for (auto& hit : *hits) {
      int layer{hit.GetDetectorID()};
      float xyz[3]{hit.GetX(), hit.GetY(), hit.GetZ()};
      float r{std::hypot(xyz[0], xyz[1])};
      float phi{std::atan2(-xyz[1], -xyz[0]) + o2::its::constants::math::Pi};

      if (kUseSmearing) {
        phi = gRandom->Gaus(phi, std::asin(0.0005f / r));
        xyz[0] = r * std::cos(phi);
        xyz[1] = r * std::sin(phi);
        xyz[2] = gRandom->Gaus(xyz[2], 0.0005f);
      }

      //if you see radius + epsilon, it's still the N-th layer... likely a bug
      if (r > 99.0 && r < 101) {
        //std::cout << "*- Exception caught at a radius of "<< r << std::endl;
        layer = 11;
      }
      event.addTrackingFrameInfoToLayer(layer, xyz[0], xyz[1], xyz[2], r, phi, std::array<float, 2>{0.f, xyz[2]},
                                        std::array<float, 3>{0.0005f * 0.0005f, 0.f, 0.0005f * 0.0005f});
      event.addClusterToLayer(layer, xyz[0], xyz[1], xyz[2], event.getClustersOnLayer(layer).size());
      event.addClusterLabelToLayer(layer, o2::MCCompLabel(hit.GetTrackID(), iEvent, iEvent, false));
      event.addClusterExternalIndexToLayer(layer, id++);
    }
    roFrame = iEvent;
    std::cout << "*- Event " << iEvent << " finished adding hits." << std::endl;

    vertexer.clustersToVertices(event);

    std::vector<Vertex> vertices = vertexer.exportVertices();
    std::cout << "*- Number of vertices found: " << vertices.size() << endl;
    hNVertices->Fill(vertices.size());

    tf.addPrimaryVertices(vertices);
  }

  tracker.clustersToTracks();

  for (int iROF{0}; iROF < tf.getNrof(); ++iROF) {
    auto vertices{tf.getPrimaryVertices(iROF)};
    if (vertices.size() == 0) {
      std::cout << "*- No primary vertex found, skipping event" << std::endl;
    }

    o2::math_utils::Point3D<float> pos{vertices[0].getX(), vertices[0].getY(), vertices[0].getZ()};
    std::array<float, 6> cov;
    for (Int_t jj = 0; jj < 6; jj++)
      cov[jj] = vertices[0].getCov()[jj];
    o2::dataformats::VertexBase vtx(pos, cov);
    o2::dataformats::DCA dca;

    //+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    std::cout << "*- Acquiring collision information..." << std::endl;
    //---> Collision data
    Long_t lEventNumber = lGoodEvents;
    collision.fBCsID = fTree[kEvents]->GetEntries();
    fdd.fBCsID = collision.fBCsID;
    ft0.fBCsID = collision.fBCsID;
    fv0a.fBCsID = collision.fBCsID;
    fv0c.fBCsID = collision.fBCsID;
    zdc.fBCsID = collision.fBCsID;
    collision.fPosX = vertices[0].getX();
    collision.fPosY = vertices[0].getY();
    collision.fPosZ = vertices[0].getZ();
    collision.fCovXX = cov[0];
    collision.fCovXY = cov[1];
    collision.fCovXZ = cov[2];
    collision.fCovYY = cov[3];
    collision.fCovYZ = cov[4];
    collision.fCovZZ = cov[5];
    collision.fChi2 = vertices[0].getChi2();
    collision.fN = vertices[0].getNContributors();
    collision.fCollisionTime = 10;
    collision.fCollisionTimeRes = 1e-6;
    ft0.fTimeA = 10;
    ft0.fTimeC = 10;

    //---> MC collision data
    mccollision.fBCsID = lGoodEvents;
    if (!mcHead) {
      std::cout << "*- Problem with MC header! " << std::endl;
      return;
    }
    mccollision.fPosX = mcHead->GetX();
    mccollision.fPosY = mcHead->GetY();
    mccollision.fPosZ = mcHead->GetZ();
    mccollision.fT = mcHead->GetT();
    mccollision.fWeight = 1;
    mccollision.fImpactParameter = mcHead->GetB();

    mccollisionlabel.fLabel = lGoodEvents;
    mccollisionlabel.fLabelMask = 0;
    //---> Save fake hits on i-th layer for track

    //---> Dummy trigger mask to ensure nobody rejects this
    bc.fTriggerMask = 0;
    for (Int_t iii = 0; iii < 60; iii++)
      bc.fTriggerMask |= 1ull << iii;
    bc.fRunNumber = 246087; //ah, the good old days

    auto& lTracks = tf.getTracks(iROF);
    auto& lTracksLabels = tf.getTracksLabel(iROF);
    hNTracks->Fill(lTracks.size());

    //+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    //---> Track data
    Long_t lNTracks = lTracks.size();
    for (Int_t i = 0; i < lNTracks; i++) {
      //get result from tracker
      auto& lab = lTracksLabels[i];
      auto& track = lTracks[i];
      int trackID = std::abs(lab.getTrackID());

      //Propagate to primary vertex as usual
      o2::dataformats::DCA dca1;
      if (!track.propagateToDCA(vtx, tracker.getBz(), &dca1)) {
        std::cout << "Track propagation to primary vertex failed." << std::endl;
      }

      //Fill QA histograms
      hPtSpectra->Fill(track.getPt());
      if (lab.isFake())
        hPtSpectraFake->Fill(track.getPt());

      tracks.fCollisionsID = lEventNumber;
      tracks.fTrackType = o2::aod::track::TrackTypeEnum::Run2Track; //Make track selection happy, please
      tracks.fFlags = 0x0;
      //Assume it all worked, fool regular selections
      tracks.fFlags |= o2::aod::track::TrackFlagsRun2Enum::ITSrefit;
      tracks.fFlags |= o2::aod::track::TrackFlagsRun2Enum::TPCrefit;
      tracks.fFlags |= o2::aod::track::TrackFlagsRun2Enum::GoldenChi2;

      //Main: X, alpha, track params
      tracks.fX = track.getX();
      tracks.fY = track.getY();
      tracks.fZ = track.getZ();
      tracks.fAlpha = track.getAlpha();
      tracks.fSnp = track.getSnp();
      tracks.fTgl = track.getTgl();
      tracks.fSigned1Pt = track.getQ2Pt();

      // diagonal elements of covariance matrix
      tracks.fSigmaY = TMath::Sqrt(track.getSigmaY2());
      tracks.fSigmaZ = TMath::Sqrt(track.getSigmaZ2());
      tracks.fSigmaSnp = TMath::Sqrt(track.getSigmaSnp2());
      tracks.fSigmaTgl = TMath::Sqrt(track.getSigmaTgl2());
      tracks.fSigma1Pt = TMath::Sqrt(track.getSigma1Pt2());
      // off-diagonal elements of covariance matrix
      tracks.fRhoZY = (Char_t)(128. * track.getSigmaZY() / tracks.fSigmaZ / tracks.fSigmaY);
      tracks.fRhoSnpY = (Char_t)(128. * track.getSigmaSnpY() / tracks.fSigmaSnp / tracks.fSigmaY);
      tracks.fRhoSnpZ = (Char_t)(128. * track.getSigmaSnpZ() / tracks.fSigmaSnp / tracks.fSigmaZ);
      tracks.fRhoTglY = (Char_t)(128. * track.getSigmaTglY() / tracks.fSigmaTgl / tracks.fSigmaY);
      tracks.fRhoTglZ = (Char_t)(128. * track.getSigmaTglZ() / tracks.fSigmaTgl / tracks.fSigmaZ);
      tracks.fRhoTglSnp = (Char_t)(128. * track.getSigmaTglSnp() / tracks.fSigmaTgl / tracks.fSigmaSnp);
      tracks.fRho1PtY = (Char_t)(128. * track.getSigma1PtY() / tracks.fSigma1Pt / tracks.fSigmaY);
      tracks.fRho1PtZ = (Char_t)(128. * track.getSigma1PtZ() / tracks.fSigma1Pt / tracks.fSigmaZ);
      tracks.fRho1PtSnp = (Char_t)(128. * track.getSigma1PtSnp() / tracks.fSigma1Pt / tracks.fSigmaSnp);
      tracks.fRho1PtTgl = (Char_t)(128. * track.getSigma1PtTgl() / tracks.fSigma1Pt / tracks.fSigmaTgl);

      //insist it's good
      tracks.fITSChi2NCl = 1.0;
      tracks.fTPCChi2NCl = 1.0;
      tracks.fTPCNClsFindable = (UChar_t)(120);
      tracks.fTPCNClsFindableMinusFound = (Char_t)(0);
      tracks.fTPCNClsFindableMinusCrossedRows = (Char_t)(0);
      UChar_t fITSClusterMap = 0u;
      fITSClusterMap |= 0x1 << 0; // flag manually
      fITSClusterMap |= 0x1 << 1; // flag manually
      tracks.fITSClusterMap = fITSClusterMap;

      //MC labels for MC use - negative, yes, but negative with offset
      mctracklabel.fLabel = TMath::Abs(lab.getTrackID()) + fOffsetLabel;
      mctracklabel.fLabelMask = 0;
      //Tag as fake. Note: used first bit only.
      if (lab.isFake())
        mctracklabel.fLabelMask = 1;

      fTree[kTracks]->Fill();
      fTree[kMcTrackLabel]->Fill();
    }
    //+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    //---> MC stack information for de-referencing
    for (Long_t iii = 0; iii < (Long_t)mcArr->size(); iii++) {
      auto part = mcArr->at(iii);

      mcparticle.fMcCollisionsID = lGoodEvents;

      //Get the kinematic values of the particles
      mcparticle.fPdgCode = part.GetPdgCode();
      mcparticle.fStatusCode = part.isPrimary();

      mcparticle.fFlags = 0;
      if (part.isSecondary())
        mcparticle.fFlags |= MCParticleFlags::ProducedInTransport;

      mcparticle.fMother0 = part.getMotherTrackId();
      if (mcparticle.fMother0 > -1)
        mcparticle.fMother0 += fOffsetLabel;
      mcparticle.fMother1 = -1;
      mcparticle.fDaughter0 = part.getFirstDaughterTrackId();
      if (mcparticle.fDaughter0 > -1)
        mcparticle.fDaughter0 += fOffsetLabel;
      mcparticle.fDaughter1 = part.getLastDaughterTrackId();
      if (mcparticle.fDaughter1 > -1)
        mcparticle.fDaughter1 += fOffsetLabel;
      mcparticle.fWeight = 1;

      mcparticle.fPx = part.Px();
      mcparticle.fPy = part.Py();
      mcparticle.fPz = part.Pz();
      mcparticle.fE = part.GetEnergy();

      mcparticle.fVx = part.Vx();
      mcparticle.fVy = part.Vy();
      mcparticle.fVz = part.Vz();
      mcparticle.fVt = part.T();

      fTree[kMcParticle]->Fill();
    }
    //+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    // Go for conversion: save info
    fTree[kEvents]->Fill();
    fTree[kMcCollision]->Fill();
    fTree[kMcCollisionLabel]->Fill();
    fTree[kBC]->Fill();
    fTree[kFDD]->Fill();
    fTree[kFV0A]->Fill();
    fTree[kFV0C]->Fill();
    fTree[kFT0]->Fill();
    fTree[kZdc]->Fill();
    //+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    fOffsetLabel = fTree[kMcParticle]->GetEntries(); //processed total
    lGoodEvents++;
  }

  TFile output("conversion-output.root", "recreate");

  //QA of conversion process: the basics
  hNVertices->Write();
  hNTracks->Write();
  hPtSpectra->Write();
  hPtSpectraFake->Write();

  fOutputDir->cd();
  fTree[kEvents]->Write();
  fTree[kBC]->Write();
  fTree[kFDD]->Write();
  fTree[kFV0A]->Write();
  fTree[kFV0C]->Write();
  fTree[kFT0]->Write();
  fTree[kZdc]->Write();
  fTree[kTracks]->Write();
  fTree[kMcTrackLabel]->Write();
  fTree[kMcParticle]->Write();
  fTree[kMcCollision]->Write();
  fTree[kMcCollisionLabel]->Write();

  std::cout << "*- Saved " << lGoodEvents << " events with a PV (total processed: " << itsHits.GetEntries() << "). Enjoy! \U0001F596" << std::endl;
}
} // namespace o2::upgrades_utils
