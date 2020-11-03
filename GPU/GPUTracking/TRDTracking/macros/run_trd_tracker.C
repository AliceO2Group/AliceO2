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
#include "DetectorsCommonDataFormats/NameConf.h"
#include "CommonConstants/LHCConstants.h"
#include "DetectorsBase/GeometryManager.h"
#include "DetectorsBase/Propagator.h"
#include "TRDBase/Geometry.h"
#include "ReconstructionDataFormats/TrackTPCITS.h"
#include "DataFormatsTRD/Tracklet64.h"
#include "DataFormatsTRD/TriggerRecord.h"

#endif

using namespace GPUCA_NAMESPACE::gpu;

unsigned int convertTrkltWordToRun2Format(uint64_t trkltWordRun3)
{
  // FIXME: this is currently a dummy function
  // need proper functionality to convert the new tracklet data format to
  // something compatible with the TRD tracker, but this macro is probably
  // not the right place for this
  unsigned int trkltWord = 0;
  return trkltWord;
}

void run_trd_tracker(std::string path = "./",
                     std::string inputTracks = "o2match_itstpc.root",
                     std::string inputTracklets = "trdtracklets.root")
{
  //-------- debug time information from tracks and tracklets
  std::vector<float> trdTriggerTimes;
  std::vector<int> trdTriggerIndices;

  //-------- init geometry and field --------//
  o2::base::GeometryManager::loadGeometry();
  o2::base::Propagator::initFieldFromGRP(o2::base::NameConf::getGRPFileName());

  auto geo = o2::trd::Geometry::instance();
  geo->createPadPlaneArray();
  geo->createClusterMatrixArray();
  const o2::trd::GeometryFlat geoFlat(*geo);

  //-------- init GPU reconstruction --------//
  GPUSettingsEvent cfgEvent;                       // defaults should be ok
  GPUSettingsRec cfgRec;                           // don't care for now, NWaysOuter is set in here for instance
  GPUSettingsProcessing cfgDeviceProcessing;       // also keep defaults here, or adjust debug level
  cfgDeviceProcessing.debugLevel = 5;
  GPURecoStepConfiguration cfgRecoStep;
  cfgRecoStep.steps = GPUDataTypes::RecoStep::NoRecoStep;
  cfgRecoStep.inputs.clear();
  cfgRecoStep.outputs.clear();
  auto rec = GPUReconstruction::CreateInstance("CPU", true);
  rec->SetSettings(&cfgEvent, &cfgRec, &cfgDeviceProcessing, &cfgRecoStep);

  auto chainTracking = rec->AddChain<GPUChainTracking>();

  auto tracker = new GPUTRDTracker();
  tracker->SetNCandidates(1); // must be set before initialization
  tracker->SetProcessPerTimeFrame();
  tracker->SetNMaxCollisions(100);

  rec->RegisterGPUProcessor(tracker, false);
  chainTracking->SetTRDGeometry(&geoFlat);
  if (rec->Init()) {
    printf("ERROR: GPUReconstruction not initialized\n");
  }

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

  std::vector<o2::trd::TriggerRecord>* triggerRecordsInArrayPtr = nullptr;
  trdTracklets.SetBranchAddress("TrackTrg", &triggerRecordsInArrayPtr);
  std::vector<o2::trd::Tracklet64>* trackletsInArrayPtr = nullptr;
  trdTracklets.SetBranchAddress("Tracklet", &trackletsInArrayPtr);
  trdTracklets.GetEntry(0);
  int nCollisions = triggerRecordsInArrayPtr->size();
  int nTracklets = trackletsInArrayPtr->size();
  printf("There are %i tracklets in total from %i trigger records\n", nTracklets, nCollisions);

  for (int iEv = 0; iEv < nCollisions; ++iEv) {
    o2::trd::TriggerRecord& trg = triggerRecordsInArrayPtr->at(iEv);
    int nTrackletsCurrent = trg.getNumberOfObjects();
    int iFirstTracklet = trg.getFirstEntry();
    int64_t evTime = trg.getBCData().toLong() * o2::constants::lhc::LHCBunchSpacingNS; // event time in ns
    trdTriggerTimes.push_back(evTime / 1000.);
    trdTriggerIndices.push_back(iFirstTracklet);
    printf("Event %i: Occured at %li us after SOR, contains %i tracklets, index of first tracklet is %i\n", iEv, evTime / 1000, nTrackletsCurrent, iFirstTracklet);
  }

  tracker->Reset();

  chainTracking->mIOPtrs.nMergedTracks = nTracks;
  chainTracking->mIOPtrs.nTRDTracklets = nTracklets;
  chainTracking->AllocateIOMemory();
  rec->PrepareEvent();
  rec->SetupGPUProcessor(tracker, true);

  printf("Start loading input into TRD tracker\n");
  // load everything into the tracker
  for (int iTrk = 0; iTrk < nTracks; ++iTrk) {
    const auto& match = (*tracksInArrayPtr)[iTrk];
    const auto& trk = match.getParamOut();
    GPUTRDTrack trkLoad;
    trkLoad.setX(trk.getX());
    trkLoad.setAlpha(trk.getAlpha());
    for (int i = 0; i < 5; ++i) {
      trkLoad.setParam(trk.getParam(i), i);
    }
    for (int i = 0; i < 15; ++i) {
      trkLoad.setCov(trk.getCov()[i], i);
    }
    trkLoad.setTime(match.getTimeMUS().getTimeStamp());
    tracker->LoadTrack(trkLoad);
    printf("Loaded track %i with time %f\n", iTrk, trkLoad.getTime());
  }

  for (int iTrklt = 0; iTrklt < nTracklets; ++iTrklt) {
    auto trklt = trackletsInArrayPtr->at(iTrklt);
    unsigned int trkltWord = convertTrkltWordToRun2Format(trklt.getTrackletWord());
    GPUTRDTrackletWord trkltLoad;
    trkltLoad.SetId(iTrklt);
    trkltLoad.SetHCId(trklt.getHCID());
    trkltLoad.SetTrackletWord(trkltWord);
    if (tracker->LoadTracklet(trkltLoad) > 0) {
      printf("Could not load tracklet %i\n", iTrklt);
    }
  }
  tracker->SetTriggerRecordTimes(&(trdTriggerTimes[0]));
  tracker->SetTriggerRecordIndices(&(trdTriggerIndices[0]));
  tracker->SetNCollisions(nCollisions);
  tracker->DumpTracks();
  tracker->DoTracking(chainTracking);
  tracker->DumpTracks();
  printf("Done\n");
}
