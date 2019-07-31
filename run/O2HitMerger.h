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
#include "FairMQMessage.h"
#include <FairMQDevice.h>
#include <FairLogger.h>
#include <SimulationDataFormat/MCEventHeader.h>
#include <SimulationDataFormat/Stack.h>
#include <SimulationDataFormat/PrimaryChunk.h>
#include <DetectorsCommonDataFormats/DetID.h>
#include <gsl/gsl>
#include "TFile.h"
#include "TTree.h"
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
#include <DetectorsCommonDataFormats/DetID.h>
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

namespace o2
{
namespace devices
{

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
    initDetInstances();

    // has to be after init of Detectors
    o2::utils::ShmManager::Instance().attachToGlobalSegment();

    mTimer.Start();
  }

  /// Default destructor
  ~O2HitMerger() override
  {
    FairSystemInfo sysinfo;
    mOutTree->SetEntries(mEntries);
    mOutTree->Write();

    // TODO: instead of doing this at the end; one
    // should investigate more asynchronous ways
    auto merged = mergeEntries();

    mTmpOutFile->Close();
    if (merged) {
      std::remove(mTmpOutFileName.c_str());
    } else {
      // if no merge was necessary, we simply rename the file
      std::rename(mTmpOutFileName.c_str(), mOutFileName.c_str());
    }

    LOG(INFO) << "TIME-STAMP " << mTimer.RealTime() << "\t";
    mTimer.Continue();
    LOG(INFO) << "MEM-STAMP " << sysinfo.GetCurrentMemory() / (1024. * 1024) << " "
              << sysinfo.GetMaxMemory() << " MB\n";
  }

 private:
  /// Overloads the InitTask() method of FairMQDevice
  void InitTask() final
  {
    std::string outfilename("o2sim_merged_hits.root"); // default name
    // query the sim config ... which is used to extract the filenames
    if (o2::devices::O2SimDevice::querySimConfig(fChannels.at("primary-get").at(0))) {
      outfilename = o2::conf::SimConfig::Instance().getOutPrefix() + ".root";
      mNExpectedEvents = o2::conf::SimConfig::Instance().getNEvents();
    }
    mOutFileName = outfilename.c_str();
    mTmpOutFileName = "o2sim_tmp.root";
    mTmpOutFile = new TFile(mTmpOutFileName.c_str(), "RECREATE");
    mOutTree = new TTree("o2sim", "o2sim");

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
    V accum{ 0 };
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

  void consumeHits(FairMQParts& data, int& index)
  {
    auto detIDmessage = std::move(data.At(index++));
    // this should be a detector ID
    if (detIDmessage->GetSize() == 4) {
      auto ptr = (int*)detIDmessage->GetData();
      o2::detectors::DetID id(ptr[0]);
      LOG(DEBUG2) << "I1 " << ptr[0] << " NAME " << id.getName() << " MB "
                  << data.At(index)->GetSize() / 1024. / 1024.;

      // get the detector than can interpret it
      auto detector = mDetectorInstances[id].get();
      if (detector) {
        detector->fillHitBranch(*mOutTree, data, index);
      }
    }
  }

  template <typename T>
  void fillBranch(std::string const& name, T* ptr)
  {
    auto br = o2::base::getOrMakeBranch(*mOutTree, name.c_str(), &ptr);
    br->SetAddress(&ptr);
    br->Fill();
    br->ResetAddress();
  }

  template <typename T>
  void consumeData(std::string name, FairMQParts& data, int& index)
  {
    auto decodeddata = o2::base::decodeTMessage<T*>(data, index);
    fillBranch(name, decodeddata);
    delete decodeddata;
    index++;
  }

  // fills a special branch of SubEventInfos in order to keep
  // track of which entry corresponds to which event etc.
  // also creates the MCEventHeader branch expected for physics analysis
  void fillSubEventInfoEntry(o2::data::SubEventInfo& info)
  {
    auto infoptr = &info;
    fillBranch("SubEventInfo", infoptr);
    // a separate branch for MCEventHeader to be backward compatible
    auto headerptr = &info.mMCEventHeader;
    fillBranch("MCEventHeader.", headerptr);
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
    LOG(INFO) << "SIMDATA channel got " << data.Size() << " parts\n";

    int index = 0;
    auto infoptr = o2::base::decodeTMessage<o2::data::SubEventInfo*>(data, index++);
    o2::data::SubEventInfo& info = *infoptr;
    auto accum = insertAdd<uint32_t, uint32_t>(mPartsCheckSum, info.eventID, (uint32_t)info.part);

    fillSubEventInfoEntry(info);
    consumeData<std::vector<o2::MCTrack>>("MCTrack", data, index);
    consumeData<std::vector<o2::TrackReference>>("TrackRefs", data, index);
    consumeData<o2::dataformats::MCTruthContainer<o2::TrackReference>>("IndexedTrackRefs", data, index);
    while (index < data.Size()) {
      consumeHits(data, index);
    }

    mEntries++;

    if (isDataComplete<uint32_t>(accum, info.nparts)) {
      LOG(INFO) << "EVERYTHING IS HERE FOR EVENT " << info.eventID << "\n";

      if (mPipeToDriver != -1) {
        if (write(mPipeToDriver, &info.eventID, sizeof(info.eventID) == -1)) {
          LOG(ERROR) << "FAILED WRITING TO PIPE";
        };
      }

      mEventChecksum += info.eventID;
      // we also need to check if we have all events
      if (isDataComplete<uint32_t>(mEventChecksum, info.maxEvents)) {
        LOG(INFO) << "ALL EVENTS HERE; CHECKSUM " << mEventChecksum;
        return false;
      }
    }
    return true;
  }

  template <typename T>
  void backInsert(T const& from, T& to)
  {
    std::copy(from.begin(), from.end(), std::back_inserter(to));
  }
  // specialization for o2::MCTruthContainer<S>
  template <typename S>
  void backInsert(o2::dataformats::MCTruthContainer<S> const& from,
                  o2::dataformats::MCTruthContainer<S>& to)
  {
    to.mergeAtBack(from);
  }

  // this merges several entries from the TBranch brname from the origin TTree
  // into a single entry in a target TTree / same branch
  // (assuming T is typically a vector; merging is simply done by appending)
  template <typename T>
  void merge(std::string brname, TTree& origin, TTree& target, std::map<int, std::vector<int>> const& entrygroups)
  {
    auto originbr = origin.GetBranch(brname.c_str());
    auto targetdata = new T;
    T* incomingdata = nullptr;
    originbr->SetAddress(&incomingdata);
    for (auto& event_entries_pair : entrygroups) {
      const auto& entries = event_entries_pair.second;
      auto currentevent = event_entries_pair.first;
      LOG(DEBUG) << "MERGING EVENT " << currentevent;

      T* filladdress;
      if (entries.size() == 1) {
        // this avoids useless copy in case there was no sub-event splitting; we just use the original data
        originbr->GetEntry(entries[0]);
        filladdress = incomingdata;
      } else {
        filladdress = targetdata;
        for (auto& e : entries) {
          originbr->GetEntry(e);
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
    }
    delete targetdata;
  }

  // This method goes over the final tree containing the hits
  // and merges entries (of subevents) into entries corresponding to one event.
  // (Preference would be not to do this at all and to keep references to sub-events in the MC labels instead;
  //  ... but method is provided in order to have backward compatibility first)
  // returns false if no merge is necessary; returns true if otherwise
  bool mergeEntries()
  {
    if (mEntries == 0 || mNExpectedEvents == 0) {
      return false;
    }

    LOG(INFO) << "ENTERING MERGING HITS STAGE";
    TStopwatch timer;
    timer.Start();
    // a) find out which entries to merge together
    // we produce a vector<vector<int>>
    auto infobr = mOutTree->GetBranch("SubEventInfo");

    auto& confref = o2::conf::SimConfig::Instance();
    if (!confref.isFilterOutNoHitEvents() && (infobr->GetEntries() == mNExpectedEvents)) {
      LOG(INFO) << "NO MERGING NECESSARY";
      return false;
    }

    std::map<int, std::vector<int>> entrygroups;  // collecting all entries belonging to an event
    std::map<int, std::vector<int>> trackoffsets; // collecting trackoffsets to be applied to correct

    std::vector<std::unique_ptr<o2::dataformats::MCEventHeader>> eventheaders; // collecting the event headers

    eventheaders.resize(mNExpectedEvents);

    // the MC labels (trackID) for hits
    o2::data::SubEventInfo* info = nullptr;
    infobr->SetAddress(&info);
    for (int i = 0; i < infobr->GetEntries(); ++i) {
      infobr->GetEntry(i);
      assert(info->npersistenttracks >= 0);
      auto event = info->eventID;
      entrygroups[event].emplace_back(i);
      trackoffsets[event].emplace_back(info->npersistenttracks);
      assert(event <= mNExpectedEvents && event >= 1);
      LOG(INFO) << event << " " << mNExpectedEvents;
      if (eventheaders[event - 1] == nullptr) {
        eventheaders[event - 1] = std::unique_ptr<dataformats::MCEventHeader>(
          new dataformats::MCEventHeader(info->mMCEventHeader));
      } else {
        eventheaders[event - 1]->getMCEventStats().add(info->mMCEventHeader.getMCEventStats());
      }
    }

    // now see which events can be discarded in any case due to no hits
    if (confref.isFilterOutNoHitEvents()) {
      for (int i = 0; i < info->maxEvents; i++) {
        if (eventheaders[i] && eventheaders[i]->getMCEventStats().getNHits() == 0) {
          LOG(INFO) << "Taking out event " << i << " due to no hits";
          entrygroups.erase(i + 1); // +1 since "eventID"
          trackoffsets.erase(i + 1);
          eventheaders[i].reset();
        }
      }
    }

    // create the final output
    auto mergedOutFile = new TFile(mOutFileName.c_str(), "RECREATE");
    gFile = mergedOutFile;
    auto mergedOutTree = new TTree("o2sim", "o2sim");

    //    for (auto& e : entrygroups) {
    //      LOG(INFO) << "EVENT " << e.first << " HAS " << e.second.size() << " ENTRIES ";
    //      std::stringstream indices;
    //      indices << "{";
    //      for (int index : e.second) {
    //        indices << index << " , ";
    //      }
    //      indices << "}";
    //      LOG(INFO) << "# " << indices.str();
    //    }

    mergedOutTree->SetEntries(entrygroups.size());

    // put the event headers into the new TTree
    o2::dataformats::MCEventHeader header;
    auto headerbr = o2::base::getOrMakeBranch(*mergedOutTree, "MCEventHeader.", &header);
    for (int i = 0; i < info->maxEvents; i++) {
      if (eventheaders[i]) {
        header = *(eventheaders[i]);
        headerbr->Fill();
      }
    }
    // attention: We need to make sure that we write everything in the same event order
    // but iteration over keys of a standard map in C++ is ordered

    // b) merge the general data
    merge<std::vector<o2::MCTrack>>("MCTrack", *mOutTree, *mergedOutTree, entrygroups);
    // TODO: fix track numbers in TrackRefs
    merge<std::vector<o2::TrackReference>>("TrackRefs", *mOutTree, *mergedOutTree, entrygroups);
    merge<o2::dataformats::MCTruthContainer<o2::TrackReference>>("IndexedTrackRefs",
                                                                 *mOutTree, *mergedOutTree, entrygroups);

    // c) do the merge procedure for all hits ... delegate this to detector specific functions
    // since they know about types; number of branches; etc.
    // this will also fix the trackIDs inside the hits
    for (auto& det : mDetectorInstances) {
      if (det) {
        det->mergeHitEntries(*mOutTree, *mergedOutTree, entrygroups, trackoffsets);
      }
    }

    mergedOutTree->Write();
    mergedOutFile->Close();
    gFile = mOutFile;
    LOG(INFO) << "MERGING HITS TOOK " << timer.RealTime();
    return true;
  }

  std::map<uint32_t, uint32_t> mPartsCheckSum; //! mapping event id -> part checksum used to detect when all info

  std::string mOutFileName;    //!
  std::string mTmpOutFileName; //!

  TFile* mOutFile;    //!
  TFile* mTmpOutFile; //! temporary IO
  TTree* mOutTree;    //!

  int mEntries = 0; //! counts the number of entries in the branches
  int mEventChecksum = 0; //! checksum for events
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

  mDetectorInstances.resize(DetID::Last);
  // like a factory of detector objects

  int counter = 0;
  for (int i = DetID::First; i <= DetID::Last; ++i) {
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
  }
  if (counter != DetID::Last) {
    LOG(WARNING) << " O2HitMerger: Some Detectors are potentially missing in this initialization ";
  }
}

} // namespace devices
} // namespace o2

#endif
