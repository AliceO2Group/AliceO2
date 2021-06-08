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
#include "TPCBase/ParameterElectronics.h"
#include "TPCBase/ParameterGas.h"

#endif

using namespace GPUCA_NAMESPACE::gpu;


void run_trd_tracker(std::string path = "./",
                     std::string inputTracks = "o2match_itstpc.root",
                     std::string inputTracklets = "trdtracklets.root")
{
  //-------- debug time information from tracks and tracklets
  std::vector<float> trdTriggerTimes;
  std::vector<int> trdTriggerIndices;

  //-------- init geometry and field --------//
  o2::base::GeometryManager::loadGeometry();
  o2::base::Propagator::initFieldFromGRP();

  auto geo = o2::trd::Geometry::instance();
  geo->createPadPlaneArray();
  geo->createClusterMatrixArray();
  const o2::trd::GeometryFlat geoFlat(*geo);

  //-------- init GPU reconstruction --------//
  // different settings are defined in GPUSettingsList.h
  GPUSettingsGRP cfgGRP;                     // defaults should be ok
  GPUSettingsRec cfgRec;                     // settings concerning reconstruction
  cfgRec.trdMinTrackPt = .5f;
  cfgRec.trdMaxChi2 = 15.f;
  cfgRec.trdPenaltyChi2 = 12.f;
  cfgRec.trdStopTrkAfterNMissLy = 6;
  GPUSettingsProcessing cfgDeviceProcessing; // also keep defaults here, or adjust debug level
  cfgDeviceProcessing.debugLevel = 5;
  GPURecoStepConfiguration cfgRecoStep;
  cfgRecoStep.steps = GPUDataTypes::RecoStep::NoRecoStep;
  cfgRecoStep.inputs.clear();
  cfgRecoStep.outputs.clear();
  auto rec = GPUReconstruction::CreateInstance("CPU", true);
  rec->SetSettings(&cfgGRP, &cfgRec, &cfgDeviceProcessing, &cfgRecoStep);

  auto chainTracking = rec->AddChain<GPUChainTracking>();

  auto tracker = new GPUTRDTracker();
  tracker->SetNCandidates(1); // must be set before initialization
  tracker->SetProcessPerTimeFrame(true);
  tracker->SetGenerateSpacePoints(true);

  rec->RegisterGPUProcessor(tracker, false);
  chainTracking->SetTRDGeometry(&geoFlat);
  if (rec->Init()) {
    printf("ERROR: GPUReconstruction not initialized\n");
  }

  auto& elParam = o2::tpc::ParameterElectronics::Instance();
  auto& gasParam = o2::tpc::ParameterGas::Instance();
  auto tpcTBinMUS = elParam.ZbinWidth;
  auto tpcVdrift = gasParam.DriftV;
  tracker->SetTPCVdrift(tpcVdrift);

  // configure the tracker
  //tracker->EnableDebugOutput();
  //tracker->StartDebugging();
  tracker->PrintSettings();

  // load input tracks
  TChain tracksItsTpc("matchTPCITS");
  tracksItsTpc.AddFile((path + inputTracks).c_str());

  std::vector<o2::dataformats::TrackTPCITS>* tracksInArrayPtr = nullptr;
  tracksItsTpc.SetBranchAddress("TPCITS", &tracksInArrayPtr);
  printf("Attached ITS-TPC tracks branch with %lli entries\n", (tracksItsTpc.GetBranch("TPCITS"))->GetEntries());

  tracksItsTpc.GetEntry(0);
  unsigned int nTracks = tracksInArrayPtr->size();
  printf("There are %u tracks in total\n", nTracks);

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
    int nTrackletsCurrent = trg.getNumberOfTracklets();
    int iFirstTracklet = trg.getFirstTracklet();
    int64_t evTime = trg.getBCData().toLong() * o2::constants::lhc::LHCBunchSpacingNS; // event time in ns
    trdTriggerTimes.push_back(evTime / 1000.);
    trdTriggerIndices.push_back(iFirstTracklet);
    printf("Event %i: Occured at %li us after SOR, contains %i tracklets, index of first tracklet is %i\n", iEv, evTime / 1000, nTrackletsCurrent, iFirstTracklet);
  }

  tracker->Reset();

  chainTracking->mIOPtrs.nMergedTracks = nTracks;
  chainTracking->mIOPtrs.nTRDTracklets = nTracklets;
  chainTracking->mIOPtrs.trdTriggerTimes = &(trdTriggerTimes[0]);
  chainTracking->mIOPtrs.trdTrackletIdxFirst = &(trdTriggerIndices[0]);
  chainTracking->mIOPtrs.nTRDTriggerRecords = nCollisions;
  chainTracking->mIOPtrs.trdTracklets = reinterpret_cast<const o2::gpu::GPUTRDTrackletWord*>(trackletsInArrayPtr->data());

  rec->PrepareEvent();
  rec->SetupGPUProcessor(tracker, true);

  printf("Start loading input tracks into TRD tracker\n");
  // load everything into the tracker
  for (unsigned int iTrk = 0; iTrk < nTracks; ++iTrk) {
    const auto& trkITSTPC = tracksInArrayPtr->at(iTrk);
    GPUTRDTrack trkLoad(trkITSTPC, tpcVdrift);
    tracker->LoadTrack(trkLoad, iTrk);
  }

  tracker->DumpTracks();
  tracker->DoTracking(chainTracking);
  tracker->DumpTracks();
  printf("Done\n");
}
