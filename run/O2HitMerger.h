// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @author Sandro Wenzel

#ifndef ALICEO2_DEVICES_HITMERGER_H_
#define ALICEO2_DEVICES_HITMERGER_H_

#include <memory>
#include <string>
#include <type_traits>
#include "FairMQMessage.h"
#include <FairMQDevice.h>
#include <FairLogger.h>
#include <SimulationDataFormat/MCEventHeader.h>
#include <SimulationDataFormat/Stack.h>
#include <SimulationDataFormat/PrimaryChunk.h>
#include <DetectorsCommonDataFormats/DetID.h>
#include <DetectorsCommonDataFormats/NameConf.h>
#include <gsl/gsl>
#include "TFile.h"
#include "TMemFile.h"
#include "TTree.h"
#include "TROOT.h"
#include <memory>
#include <TMessage.h>
#include <FairMQParts.h>
#include <ctime>
#include <TStopwatch.h>
#include <sstream>
#include <cassert>
#include "FairSystemInfo.h"
#include "Steer/InteractionSampler.h"

#include "O2HitMerger.h"
#include "O2SimDevice.h"
#include <TPCSimulation/Detector.h>
#include <ITSSimulation/Detector.h>
#include <MFTSimulation/Detector.h>
#include <EMCALSimulation/Detector.h>
#include <TOFSimulation/Detector.h>
#include <TRDSimulation/Detector.h>
#include <FT0Simulation/Detector.h>
#include <FV0Simulation/Detector.h>
#include <FDDSimulation/Detector.h>
#include <HMPIDSimulation/Detector.h>
#include <PHOSSimulation/Detector.h>
#include <CPVSimulation/Detector.h>
#include <MCHSimulation/Detector.h>
#include <MIDSimulation/Detector.h>
#include <ZDCSimulation/Detector.h>

#include "CommonUtils/ShmManager.h"
#include <map>
#include <vector>
#include <csignal>
#include <mutex>

#ifdef ENABLE_UPGRADES
#include <ITS3Simulation/Detector.h>
#include <TRKSimulation/Detector.h>
#endif

namespace o2
{
namespace devices
{

// signal handler
void sighandler(int signal)
{
  if (signal == SIGSEGV) {
    LOG(WARN) << "segmentation violation ... just exit without coredump in order not to hang";
    raise(SIGKILL);
  }
}

class O2HitMerger : public FairMQDevice
{

  class TMessageWrapper : public TMessage
  {
   public:
    TMessageWrapper(void* buf, Int_t len) : TMessage(buf, len) { ResetBit(kIsOwner); }
    ~TMessageWrapper() override = default;
  };

 public:
  /// Default constructor
  O2HitMerger()
  {
    mTimer.Start();
  }

  /// Default destructor
  ~O2HitMerger() override
  {
    FairSystemInfo sysinfo;
    LOG(INFO) << "TIME-STAMP " << mTimer.RealTime() << "\t";
    mTimer.Continue();
    LOG(INFO) << "MEM-STAMP " << sysinfo.GetCurrentMemory() / (1024. * 1024) << " "
              << sysinfo.GetMaxMemory() << " MB\n";
  }

 private:
  /// Overloads the InitTask() method of FairMQDevice
  void InitTask() final
  {
    signal(SIGSEGV, sighandler);
    ROOT::EnableThreadSafety();

    std::string outfilename("o2sim_merged_hits.root"); // default name
    // query the sim config ... which is used to extract the filenames
    if (o2::devices::O2SimDevice::querySimConfig(fChannels.at("primary-get").at(0))) {
      outfilename = o2::base::NameConf::getMCKinematicsFileName(o2::conf::SimConfig::Instance().getOutPrefix().c_str());
      mNExpectedEvents = o2::conf::SimConfig::Instance().getNEvents();
    }
    mOutFileName = outfilename.c_str();
    mOutFile = new TFile(outfilename.c_str(), "RECREATE");
    mOutTree = new TTree("o2sim", "o2sim");
    mOutTree->SetDirectory(mOutFile);

    initDetInstances();
    // has to be after init of Detectors
    o2::utils::ShmManager::Instance().attachToGlobalSegment();

    // init pipe
    auto pipeenv = getenv("ALICE_O2SIMMERGERTODRIVER_PIPE");
    if (pipeenv) {
      mPipeToDriver = atoi(pipeenv);
      LOG(INFO) << "ASSIGNED PIPE HANDLE " << mPipeToDriver;
    } else {
      LOG(WARNING) << "DID NOT FIND ENVIRONMENT VARIABLE TO INIT PIPE";
    }

    // if no data to expect we shut down the device NOW since it would otherwise hang
    // (because we use OnData and would never receive anything)
    if (mNExpectedEvents == 0) {
      LOG(INFO) << "NOT EXPECTING ANY DATA; SHUTTING DOWN";
      raise(SIGINT);
    }
  }

  template <typename T, typename V>
  V insertAdd(std::map<T, V>& m, T const& key, V value)
  {
    const auto iter = m.find(key);
    V accum{0};
    if (iter != m.end()) {
      iter->second += value;
      accum = iter->second;
    } else {
      m.insert(std::make_pair(key, value));
      accum = value;
    }
    return accum;
  }

  template <typename T>
  bool isDataComplete(T checksum, T nparts)
  {
    return checksum == nparts * (nparts + 1) / 2;
  }

  void consumeHits(int eventID, FairMQParts& data, int& index)
  {
    auto detIDmessage = std::move(data.At(index++));
    // this should be a detector ID
    if (detIDmessage->GetSize() == 4) {
      auto ptr = (int*)detIDmessage->GetData();
      o2::detectors::DetID id(ptr[0]);
      LOG(DEBUG2) << "I1 " << ptr[0] << " NAME " << id.getName() << " MB "
                  << data.At(index)->GetSize() / 1024. / 1024.;

      TTree* tree = mEventToTTreeMap[eventID];

      // get the detector that can interpret it
      auto detector = mDetectorInstances[id].get();
      if (detector) {
        detector->fillHitBranch(*tree, data, index);
      }
    }
  }

  template <typename T>
  void fillBranch(int eventID, std::string const& name, T* ptr)
  {
    // fetch tree into which to fill
    const std::lock_guard<std::mutex> lock(mMapsMtx);
    auto iter = mEventToTTreeMap.find(eventID);
    if (iter == mEventToTTreeMap.end()) {
      {
        std::stringstream str;
        str << "memfile" << eventID;
        mEventToTMemFileMap[eventID] = new TMemFile(str.str().c_str(), "RECREATE");
      }
      {
        std::stringstream str;
        str << "o2sim" << eventID;
        mEventToTTreeMap[eventID] = new TTree(str.str().c_str(), str.str().c_str());
        mEventToTTreeMap[eventID]->SetDirectory(mEventToTMemFileMap[eventID]);
      }
    }
    TTree* tree = mEventToTTreeMap[eventID];

    auto br = o2::base::getOrMakeBranch(*tree, name.c_str(), &ptr);
    br->SetAddress(&ptr);
    br->Fill();
    br->ResetAddress();
  }

  template <typename T>
  void consumeData(int eventID, std::string name, FairMQParts& data, int& index)
  {
    auto decodeddata = o2::base::decodeTMessage<T*>(data, index);
    fillBranch(eventID, name, decodeddata);
    delete decodeddata;
    index++;
  }

  // fills a special branch of SubEventInfos in order to keep
  // track of which entry corresponds to which event etc.
  // also creates the MCEventHeader branch expected for physics analysis
  void fillSubEventInfoEntry(o2::data::SubEventInfo& info)
  {
    auto infoptr = &info;
    fillBranch(info.eventID, "SubEventInfo", infoptr);
    // a separate branch for MCEventHeader to be backward compatible
    auto headerptr = &info.mMCEventHeader;
    fillBranch(info.eventID, "MCEventHeader.", headerptr);
  }

  bool ConditionalRun() override
  {
    auto& channel = fChannels.at("simdata").at(0);
    FairMQParts request;
    auto bytes = channel.Receive(request);
    if (bytes < 0) {
      LOG(ERROR) << "Some error occurred on socket during receive on sim data";
      return true; // keep going
    }
    return handleSimData(request, 0);
  }

  bool handleSimData(FairMQParts& data, int /*index*/)
  {
    bool expectmore = true;
    int index = 0;
    auto infoptr = o2::base::decodeTMessage<o2::data::SubEventInfo*>(data, index++);
    o2::data::SubEventInfo& info = *infoptr;
    auto accum = insertAdd<uint32_t, uint32_t>(mPartsCheckSum, info.eventID, (uint32_t)info.part);

    LOG(INFO) << "SIMDATA channel got " << data.Size() << " parts for event " << info.eventID << " part " << info.part << " out of " << info.nparts;

    fillSubEventInfoEntry(info);
    consumeData<std::vector<o2::MCTrack>>(info.eventID, "MCTrack", data, index);
    consumeData<std::vector<o2::TrackReference>>(info.eventID, "TrackRefs", data, index);
    while (index < data.Size()) {
      consumeHits(info.eventID, data, index);
    }
    // set the number of entries in the tree
    auto tree = mEventToTTreeMap[info.eventID];
    auto memfile = mEventToTMemFileMap[info.eventID];
    tree->SetEntries(tree->GetEntries() + 1);
    LOG(INFO) << "tree has file " << tree->GetDirectory()->GetFile()->GetName();
    mEntries++;

    if (isDataComplete<uint32_t>(accum, info.nparts)) {
      LOG(INFO) << "EVERYTHING IS HERE FOR EVENT " << info.eventID << "\n";

      // check if previous flush finished
      if (mMergerIOThread.joinable()) {
        mMergerIOThread.join();
      }

      // start hit merging and flushing in a separate thread in order not to block
      mMergerIOThread = std::thread([info, this]() { mergeAndFlushData(info.eventID); });

      mEventChecksum += info.eventID;
      // we also need to check if we have all events
      if (isDataComplete<uint32_t>(mEventChecksum, info.maxEvents)) {
        LOG(INFO) << "ALL EVENTS HERE; CHECKSUM " << mEventChecksum;

        // flush remaining data and close file
        if (mMergerIOThread.joinable()) {
          mMergerIOThread.join();
        }

        expectmore = false;
      }

      if (mPipeToDriver != -1) {
        if (write(mPipeToDriver, &info.eventID, sizeof(info.eventID)) == -1) {
          LOG(ERROR) << "FAILED WRITING TO PIPE";
        };
      }
    }
    if (!expectmore) {
      // somehow FairMQ has difficulties shutting down; helping manually
      raise(SIGINT);
    }
    return expectmore;
  }

  void cleanEvent(int eventID)
  {
    // remove tree for that eventID
    const std::lock_guard<std::mutex> lock(mMapsMtx);
    delete mEventToTTreeMap[eventID];
    delete mEventToTMemFileMap[eventID];
    mEventToTTreeMap.erase(eventID);
    mEventToTMemFileMap.erase(eventID); // remove memfile
  }

  template <typename T>
  void backInsert(T const& from, T& to)
  {
    std::copy(from.begin(), from.end(), std::back_inserter(to));
  }

  void reorderAndMergeMCTRacks(TTree& origin, TTree& target, const std::vector<int>& nprimaries, const std::vector<int>& nsubevents)
  {
    std::vector<MCTrack>* incomingdata = nullptr;
    auto targetdata = new std::vector<MCTrack>;
    auto originbr = origin.GetBranch("MCTrack");
    originbr->SetAddress(&incomingdata);
    const auto entries = origin.GetEntries();
    //
    // loop over subevents to store the primary events
    //
    Int_t nprimTot = 0;
    for (auto entry = entries - 1; entry >= 0; --entry) {
      int index = nsubevents[entry];
      nprimTot += nprimaries[index];
      printf("merge %lld %5d %5d %5d \n", entry, index, nsubevents[entry], nsubevents[index]);
      originbr->GetEntry(index);
      for (Int_t i = 0; i < nprimaries[index]; i++) {
        auto& track = incomingdata->at(i);
        if (track.isTransported()) { // reset daughters only if track was transported, it will be fixed below
          track.SetFirstDaughterTrackId(-1);
          track.SetLastDaughterTrackId(-1);
        }
        targetdata->push_back(track);
      }
      incomingdata->clear();
      delete incomingdata;
      incomingdata = nullptr;
    }
    //
    // loop a second time to store the secondaries and fix the mother track IDs
    //
    Int_t idelta1 = nprimTot;
    Int_t idelta0 = 0;
    for (auto entry = entries - 1; entry >= 0; --entry) {
      int index = nsubevents[entry];

      originbr->GetEntry(index);

      Int_t npart = (int)(incomingdata->size());
      Int_t nprim = nprimaries[index];
      idelta1 -= nprim;

      for (Int_t i = nprim; i < npart; i++) {
        auto& track = incomingdata->at(i);
        Int_t cId = track.getMotherTrackId();
        if (cId >= nprim) {
          cId += idelta1;
        } else {
          cId += idelta0;
        }
        track.SetMotherTrackId(cId);
        track.SetFirstDaughterTrackId(-1);

        Int_t hwm = (int)(targetdata->size());
        auto& mother = targetdata->at(cId);
        if (mother.getFirstDaughterTrackId() == -1) {
          mother.SetFirstDaughterTrackId(hwm);
        }
        mother.SetLastDaughterTrackId(hwm);

        targetdata->push_back(track);
      }
      idelta0 += nprim;
      idelta1 += npart;
      incomingdata->clear();
      delete incomingdata;
      incomingdata = nullptr;
    }

    //
    // write to output
    auto targetbr = o2::base::getOrMakeBranch(target, "MCTrack", &targetdata);
    targetbr->SetAddress(&targetdata);
    targetbr->Fill();
    targetbr->ResetAddress();
    targetdata->clear();
  }

  template <typename T>
  void remapTrackIdsAndMerge(std::string brname, TTree& origin, TTree& target,
                             const std::vector<int>& trackoffsets, const std::vector<int>& nprimaries, const std::vector<int>& subevOrdered)
  {
    //
    // Remap the mother track IDs by adding an offset.
    // The offset calculated as the sum of the number of entries in the particle list of the previous subevents.
    // This method is called by O2HitMerger::mergeAndFlushData(int)
    //
    T* incomingdata = nullptr;
    auto targetdata = new T;
    auto originbr = origin.GetBranch(brname.c_str());
    originbr->SetAddress(&incomingdata);
    const auto entries = origin.GetEntries();

    if (entries == 1) {
      // nothing to do in case there is only one entry
      originbr->GetEntry(0);
      targetdata = incomingdata;
    } else {
      // loop over subevents
      Int_t nprimTot = 0;
      for (auto entry = 0; entry < entries; entry++) {
        nprimTot += nprimaries[entry];
      }
      Int_t idelta0 = 0;
      Int_t idelta1 = nprimTot;
      for (auto entry = entries - 1; entry >= 0; --entry) {
        Int_t index = subevOrdered[entry];
        Int_t nprim = nprimaries[index];
        originbr->GetEntry(index);
        idelta1 -= nprim;
        for (auto& data : *incomingdata) {
          updateTrackIdWithOffset(data, nprim, idelta0, idelta1);
          targetdata->push_back(data);
        }
        incomingdata->clear();
        idelta0 += nprim;
        idelta1 += trackoffsets[index];
        delete incomingdata;
        incomingdata = nullptr;
      }
    }
    auto targetbr = o2::base::getOrMakeBranch(target, brname.c_str(), &targetdata);
    targetbr->SetAddress(&targetdata);
    targetbr->Fill();
    targetbr->ResetAddress();
    targetdata->clear();
  }

  void updateTrackIdWithOffset(MCTrack& track, Int_t nprim, Int_t idelta0, Int_t idelta1)
  {
    Int_t cId = track.getMotherTrackId();
    Int_t ioffset = (cId < nprim) ? idelta0 : idelta1;
    if (cId != -1) {
      track.SetMotherTrackId(cId + ioffset);
    }
  }

  void updateTrackIdWithOffset(TrackReference& ref, Int_t nprim, Int_t idelta0, Int_t idelta1)
  {
    Int_t cId = ref.getTrackID();
    Int_t ioffset = (cId < nprim) ? idelta0 : idelta1;
    ref.setTrackID(cId + ioffset);
  }

  // this merges all entries from the TBranch brname from the origin TTree (containing one event only)
  // into a single entry in a target TTree / same branch
  // (assuming T is typically a vector; merging is simply done by appending)
  template <typename T>
  void merge(std::string brname, TTree& origin, TTree& target)
  {
    auto originbr = origin.GetBranch(brname.c_str());
    auto targetdata = new T;
    T* incomingdata = nullptr;
    originbr->SetAddress(&incomingdata);

    const auto entries = origin.GetEntries();

    T* filladdress = nullptr;
    if (entries == 1) {
      // this avoids useless copy in case there was no sub-event splitting; we just use the original data
      originbr->GetEntry(0);
      filladdress = incomingdata;
    } else {
      filladdress = targetdata;
      for (auto entry = 0; entry < entries; ++entry) {
        originbr->GetEntry(entry);
        backInsert(*incomingdata, *targetdata);
        delete incomingdata;
        incomingdata = nullptr;
      }
    }

    // fill target for this event
    auto targetbr = o2::base::getOrMakeBranch(target, brname.c_str(), &filladdress);
    targetbr->SetAddress(&filladdress);
    targetbr->Fill();
    targetbr->ResetAddress();
    targetdata->clear();
    if (incomingdata) {
      delete incomingdata;
      incomingdata = nullptr;
    }

    delete targetdata;
  }

  void initHitTreeAndOutFile(int detID)
  {
    using o2::detectors::DetID;
    if (mDetectorOutFiles[detID]) {
      LOG(WARN) << "Hit outfile for detID " << DetID::getName(detID) << " already initialized";
      return;
    }
    std::string name(o2::base::NameConf::getHitsFileName(detID, o2::conf::SimConfig::Instance().getOutPrefix().c_str()));
    mDetectorOutFiles[detID] = new TFile(name.c_str(), "RECREATE");
    mDetectorToTTreeMap[detID] = new TTree("o2sim", "o2sim");
    mDetectorToTTreeMap[detID]->SetDirectory(mDetectorOutFiles[detID]);
  }

  // This method goes over the tree containing data for a given event; potentially merges
  // it and flushes it into the actual output file.
  // The method can be called asynchronously to data collection
  bool mergeAndFlushData(int eventID)
  {
    LOG(INFO) << "ENTERING MERGING/FLUSHING HITS STAGE FOR EVENT " << eventID;

    auto tree = mEventToTTreeMap[eventID];
    if (!tree) {
      LOG(INFO) << "NO TTREE FOUND FOR EVENT " << eventID;
      return false;
    }

    if (tree->GetEntries() == 0 || mNExpectedEvents == 0) {
      LOG(INFO) << "NO ENTRY IN TTREE FOUND FOR EVENT " << eventID;
      return false;
    }

    TStopwatch timer;
    timer.Start();

    // calculate trackoffsets
    auto infobr = tree->GetBranch("SubEventInfo");

    auto& confref = o2::conf::SimConfig::Instance();

    std::vector<int> trackoffsets; // collecting trackoffsets to be applied to correct
    std::vector<int> nprimaries;   // collecting primary particles in each subevent
    std::vector<int> nsubevents;   // collecting of subevent numbers

    o2::dataformats::MCEventHeader* eventheader = nullptr; // The event header

    // the MC labels (trackID) for hits
    o2::data::SubEventInfo* info = nullptr;
    infobr->SetAddress(&info);
    for (int i = 0; i < infobr->GetEntries(); ++i) {
      infobr->GetEntry(i);
      assert(info->npersistenttracks >= 0);
      trackoffsets.emplace_back(info->npersistenttracks);
      nprimaries.emplace_back(info->nprimarytracks);
      nsubevents.emplace_back(info->part);
      info->mMCEventHeader.printInfo();
      if (eventheader == nullptr) {
        eventheader = &info->mMCEventHeader;
      } else {
        eventheader->getMCEventStats().add(info->mMCEventHeader.getMCEventStats());
      }
    }

    // now see which events can be discarded in any case due to no hits
    if (confref.isFilterOutNoHitEvents()) {
      if (eventheader && eventheader->getMCEventStats().getNHits() == 0) {
        LOG(INFO) << " Taking out event " << eventID << " due to no hits ";
        cleanEvent(eventID);
        return false;
      }
    }

    // put the event headers into the new TTree
    eventheader->printInfo();
    auto headerbr = o2::base::getOrMakeBranch(*mOutTree, "MCEventHeader.", &eventheader);
    headerbr->SetAddress(&eventheader);
    headerbr->Fill();
    headerbr->ResetAddress();

    // attention: We need to make sure that we write everything in the same event order
    // but iteration over keys of a standard map in C++ is ordered

    // b) merge the general data
    //
    // for MCTrack remap the motherIds and merge at the same go
    const auto entries = tree->GetEntries();
    std::vector<int> subevOrdered((int)(nsubevents.size()));
    for (auto entry = entries - 1; entry >= 0; --entry) {
      subevOrdered[nsubevents[entry] - 1] = entry;
      printf("HitMerger entry: %lld nprimry: %5d trackoffset: %5d \n", entry, nprimaries[entry], trackoffsets[entry]);
    }

    reorderAndMergeMCTRacks(*tree, *mOutTree, nprimaries, subevOrdered);
    Int_t ioffset = 0;
    remapTrackIdsAndMerge<std::vector<o2::TrackReference>>("TrackRefs", *tree, *mOutTree, trackoffsets, nprimaries, subevOrdered);

    // c) do the merge procedure for all hits ... delegate this to detector specific functions
    // since they know about types; number of branches; etc.
    // this will also fix the trackIDs inside the hits
    for (int id = 0; id < mDetectorInstances.size(); ++id) {
      auto& det = mDetectorInstances[id];
      if (det) {
        auto hittree = mDetectorToTTreeMap[id];
        det->mergeHitEntries(*tree, *hittree, trackoffsets, nprimaries, subevOrdered);
        hittree->SetEntries(hittree->GetEntries() + 1);
        LOG(INFO) << "flushing tree to file " << hittree->GetDirectory()->GetFile()->GetName();
        mDetectorOutFiles[id]->Write("", TObject::kOverwrite);
      }
    }

    // increase the entry count in the tree
    mOutTree->SetEntries(mOutTree->GetEntries() + 1);
    LOG(INFO) << "outtree has file " << mOutTree->GetDirectory()->GetFile()->GetName();
    mOutFile->Write("", TObject::kOverwrite);

    cleanEvent(eventID);

    LOG(INFO) << "MERGING HITS TOOK " << timer.RealTime();
    return true;
  }

  std::map<uint32_t, uint32_t> mPartsCheckSum; //! mapping event id -> part checksum used to detect when all info

  std::string mOutFileName; //!

  // structures for the final flush
  TFile* mOutFile;                                     //! outfile for kinematics
  TTree* mOutTree;                                     //! tree (kinematics) associated to mOutFile
  std::unordered_map<int, TFile*> mDetectorOutFiles;   //! outfiles per detector for hits
  std::unordered_map<int, TTree*> mDetectorToTTreeMap; //! the trees

  // intermediate structures to collect data per event
  std::unordered_map<int, TTree*> mEventToTTreeMap;       //! in memory trees to collect / presort incoming data per event
  std::unordered_map<int, TMemFile*> mEventToTMemFileMap; //! files associated to the TTrees
  std::thread mMergerIOThread;                            //! a thread used to do hit merging and IO flushing asynchronously
  std::mutex mMapsMtx;                                    //!
  int mEntries = 0;         //! counts the number of entries in the branches
  int mEventChecksum = 0;   //! checksum for events
  int mNExpectedEvents = 0; //! number of events that we expect to receive
  TStopwatch mTimer;

  int mPipeToDriver = -1;

  std::vector<std::unique_ptr<o2::base::Detector>> mDetectorInstances;

  // init detector instances
  void initDetInstances();
};

// init detector instances used to write hit data to a TTree
void O2HitMerger::initDetInstances()
{
  using o2::detectors::DetID;

  // a little helper lambda
  auto isActivated = [](std::string s) -> bool {
    // access user configuration for list of wanted modules
    auto& modulelist = o2::conf::SimConfig::Instance().getActiveDetectors();
    auto active = std::find(modulelist.begin(), modulelist.end(), s) != modulelist.end();
    return active; };

  mDetectorInstances.resize(DetID::nDetectors);
  // like a factory of detector objects

  int counter = 0;
  for (int i = DetID::First; i <= DetID::Last; ++i) {
    if (!isActivated(DetID::getName(i))) {
      continue;
    }

    if (i == DetID::TPC) {
      mDetectorInstances[i] = std::move(std::make_unique<o2::tpc::Detector>(true));
      counter++;
    }
    if (i == DetID::ITS) {
      mDetectorInstances[i] = std::move(std::make_unique<o2::its::Detector>(true));
      counter++;
    }
    if (i == DetID::MFT) {
      mDetectorInstances[i] = std::move(std::make_unique<o2::mft::Detector>());
      counter++;
    }
    if (i == DetID::TRD) {
      mDetectorInstances[i] = std::move(std::make_unique<o2::trd::Detector>(true));
      counter++;
    }
    if (i == DetID::PHS) {
      mDetectorInstances[i] = std::move(std::make_unique<o2::phos::Detector>(true));
      counter++;
    }
    if (i == DetID::CPV) {
      mDetectorInstances[i] = std::move(std::make_unique<o2::cpv::Detector>(true));
      counter++;
    }
    if (i == DetID::EMC) {
      mDetectorInstances[i] = std::move(std::make_unique<o2::emcal::Detector>(true));
      counter++;
    }
    if (i == DetID::HMP) {
      mDetectorInstances[i] = std::move(std::make_unique<o2::hmpid::Detector>(true));
      counter++;
    }
    if (i == DetID::TOF) {
      mDetectorInstances[i] = std::move(std::make_unique<o2::tof::Detector>(true));
      counter++;
    }
    if (i == DetID::FT0) {
      mDetectorInstances[i] = std::move(std::make_unique<o2::ft0::Detector>(true));
      counter++;
    }
    if (i == DetID::FV0) {
      mDetectorInstances[i] = std::move(std::make_unique<o2::fv0::Detector>(true));
      counter++;
    }
    if (i == DetID::FDD) {
      mDetectorInstances[i] = std::move(std::make_unique<o2::fdd::Detector>(true));
      counter++;
    }
    if (i == DetID::MCH) {
      mDetectorInstances[i] = std::move(std::make_unique<o2::mch::Detector>(true));
      counter++;
    }
    if (i == DetID::MID) {
      mDetectorInstances[i] = std::move(std::make_unique<o2::mid::Detector>(true));
      counter++;
    }
    if (i == DetID::ZDC) {
      mDetectorInstances[i] = std::move(std::make_unique<o2::zdc::Detector>(true));
      counter++;
    }
#ifdef ENABLE_UPGRADES
    if (i == DetID::IT3) {
      mDetectorInstances[i] = std::move(std::make_unique<o2::its3::Detector>(true));
      counter++;
    }
    if (i == DetID::TRK) {
      mDetectorInstances[i] = std::move(std::make_unique<o2::trk::Detector>(true));
      counter++;
    }
#endif
    // init the detector specific output files
    initHitTreeAndOutFile(i);
  }
  if (counter != DetID::nDetectors) {
    LOG(WARNING) << " O2HitMerger: Some Detectors are potentially missing in this initialization ";
  }
}

} // namespace devices
} // namespace o2

#endif
