#if !defined(__CLING__) || defined(__ROOTCLING__)
// ROOT header
#include <TChain.h>
// GPU header
#include "GPUO2Interface.h"
#include "GPUReconstruction.h"
#include "GPUChainTracking.h"
#include "GPUSettings.h"
#include "GPUDataTypes.h"
#include "GPUTRDDef.h"
#include "GPUTRDTrack.h"
#include "GPUTRDTracker.h"
#include "GPUTRDTrackletWord.h"
#include "GPUTRDInterfaces.h"
#include "GPUTRDGeometry.h"

// O2 header
#include "DetectorsBase/GeometryManager.h"
#include "DetectorsBase/Propagator.h"
#include "ReconstructionDataFormats/TrackTPCITS.h"
#include "TRDBase/Tracklet.h"

#endif

using namespace GPUCA_NAMESPACE::gpu;

void run_trd_tracker(std::string path = "./",
                     std::string inputGRP = "o2sim_grp.root",
                     std::string inputTracks = "o2match_itstpc.root",
                     std::string inputTracklets = "trdtracklets.root")
{

  //-------- init geometry and field --------//
  o2::base::GeometryManager::loadGeometry(path);
  o2::base::Propagator::initFieldFromGRP(path + inputGRP);

  auto geo = o2::trd::TRDGeometry::instance();
  geo->createPadPlaneArray();
  geo->createClusterMatrixArray();
  const o2::trd::TRDGeometryFlat geoFlat(*geo);

  //-------- init GPU reconstruction --------//
  GPUSettingsEvent cfgEvent;                       // defaults should be ok
  GPUSettingsRec cfgRec;                           // don't care for now, NWaysOuter is set in here for instance
  GPUSettingsDeviceProcessing cfgDeviceProcessing; // also keep defaults here, or adjust debug level
  cfgDeviceProcessing.debugLevel = 10;
  GPURecoStepConfiguration cfgRecoStep;
  cfgRecoStep.steps = GPUDataTypes::RecoStep::NoRecoStep;
  auto rec = GPUReconstruction::CreateInstance("CPU", true);
  rec->SetSettings(&cfgEvent, &cfgRec, &cfgDeviceProcessing, &cfgRecoStep);

  auto chainTracking = rec->AddChain<GPUChainTracking>();

  auto tracker = new GPUTRDTracker();
  tracker->SetNCandidates(1); // must be set before initialization

  rec->RegisterGPUProcessor(tracker, false);
  chainTracking->SetTRDGeometry(&geoFlat);
  tracker->SetTrackingChain(chainTracking);
  rec->Init();
  rec->AllocateRegisteredMemory(nullptr);

  // configure the tracker
  //tracker->EnableDebugOutput();
  //tracker->StartDebugging();
  tracker->SetPtThreshold(0.5);
  tracker->SetChi2Threshold(15);
  tracker->SetChi2Penalty(12);
  tracker->SetMaxMissingLayers(6);
  tracker->PrintSettings();

  // load input tracks
  TChain tracksItsTpc("matchTPCITS");
  tracksItsTpc.AddFile((path + inputTracks).c_str());

  std::vector<o2::dataformats::TrackTPCITS>* tracksInArrayPtr = nullptr;
  tracksItsTpc.SetBranchAddress("TPCITS", &tracksInArrayPtr);
  printf("Attached ITS-TPC tracks branch with %lli entries\n", (tracksItsTpc.GetBranch("TPCITS"))->GetEntries());

  tracksItsTpc.GetEntry(0);
  int nTracks = tracksInArrayPtr->size();
  printf("There are %i tracks in total\n", nTracks);

  // and load input tracklets
  TChain trdTracklets("o2sim");
  trdTracklets.AddFile((path + inputTracklets).c_str());

  std::vector<o2::trd::Tracklet>* trackletsInArrayPtr = nullptr;
  trdTracklets.SetBranchAddress("Tracklet", &trackletsInArrayPtr);
  trdTracklets.GetEntry(0);
  int nTracklets = trackletsInArrayPtr->size();
  printf("There are %i tracklets in total\n", nTracklets);

  auto pp = geo->getPadPlane(0, 0);
  printf("Tilt=%f\n", pp->getTiltingAngle());

  tracker->Reset(true);

  chainTracking->mIOPtrs.nMergedTracks = nTracks;
  chainTracking->mIOPtrs.nTRDTracklets = nTracklets;
  chainTracking->AllocateIOMemory();
  rec->PrepareEvent();
  rec->AllocateRegisteredMemory(tracker->MemoryTracks());

  // load everything into the tracker
  for (int iTrk = 0; iTrk < nTracks; ++iTrk) {
    auto trk = tracksInArrayPtr->at(iTrk);
    GPUTRDTrack trkLoad;
    trkLoad.setX(trk.getX());
    trkLoad.setAlpha(trk.getAlpha());
    for (int i = 0; i < 5; ++i) {
      trkLoad.setParam(trk.getParam(i), i);
    }
    for (int i = 0; i < 15; ++i) {
      trkLoad.setCov(trk.getCov()[i], i);
    }
    tracker->LoadTrack(trkLoad);
  }
  for (int iTrklt = 0; iTrklt < nTracklets; ++iTrklt) {
    auto trklt = trackletsInArrayPtr->at(iTrklt);
    unsigned int trkltWord = trklt.getTrackletWord();
    GPUTRDTrackletWord trkltLoad;
    trkltLoad.SetId(iTrklt);
    trkltLoad.SetHCId(trklt.getHCId());
    trkltLoad.SetTrackletWord(trkltWord);
    if (tracker->LoadTracklet(trkltLoad) > 0) {
      printf("Could not load tracklet %i\n", iTrklt);
    }
  }
  tracker->DumpTracks();
  tracker->DoTracking();
  tracker->DumpTracks();

  printf("Done\n");
}
