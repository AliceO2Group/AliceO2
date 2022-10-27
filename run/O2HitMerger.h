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

/// @author Sandro Wenzel

#ifndef ALICEO2_DEVICES_HITMERGER_H_
#define ALICEO2_DEVICES_HITMERGER_H_

#include <memory>
#include <string>
#include <type_traits>
#include <fairmq/Message.h>
#include <fairmq/Device.h>
#include <fairlogger/Logger.h>
#include <SimulationDataFormat/MCEventHeader.h>
#include <SimulationDataFormat/Stack.h>
#include <SimulationDataFormat/PrimaryChunk.h>
#include <DetectorsCommonDataFormats/DetID.h>
#include <DetectorsCommonDataFormats/DetectorNameConf.h>
#include <gsl/gsl>
#include "TFile.h"
#include "TMemFile.h"
#include "TTree.h"
#include "TROOT.h"
#include <memory>
#include <TMessage.h>
#include <fairmq/Parts.h>
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
#include <list>
#include <csignal>
#include <mutex>
#include <filesystem>
#include <functional>

#include "SimPublishChannelHelper.h"

#ifdef ENABLE_UPGRADES
#include <ITS3Simulation/Detector.h>
#include <TRKSimulation/Detector.h>
#include <FT3Simulation/Detector.h>
#include <FCTSimulation/Detector.h>
#endif

#include <tbb/concurrent_unordered_map.h>

namespace o2
{
namespace devices
{

// signal handler
void sighandler(int signal)
{
  if (signal == SIGSEGV) {
    LOG(warn) << "segmentation violation ... just exit without coredump in order not to hang";
    raise(SIGKILL);
  }
}

class O2HitMerger : public fair::mq::Device
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
    mInitialOutputDir = std::filesystem::current_path().string();
    mCurrentOutputDir = mInitialOutputDir;
  }

  /// Default destructor
  ~O2HitMerger() override
  {
    FairSystemInfo sysinfo;
    LOG(info) << "TIME-STAMP " << mTimer.RealTime() << "\t";
    mTimer.Continue();
    LOG(info) << "MEM-STAMP " << sysinfo.GetCurrentMemory() / (1024. * 1024) << " "
              << sysinfo.GetMaxMemory() << " MB\n";
  }

 private:
  /// Overloads the InitTask() method of fair::mq::Device
  void InitTask() final
  {
    LOG(info) << "INIT HIT MERGER";
    // signal(SIGSEGV, sighandler);
    ROOT::EnableThreadSafety();

    std::string outfilename("o2sim_merged_hits.root"); // default name
    // query the sim config ... which is used to extract the filenames
    if (o2::devices::O2SimDevice::querySimConfig(fChannels.at("o2sim-primserv-info").at(0))) {
      outfilename = o2::base::NameConf::getMCKinematicsFileName(o2::conf::SimConfig::Instance().getOutPrefix().c_str());
      mNExpectedEvents = o2::conf::SimConfig::Instance().getNEvents();
    }
    mAsService = o2::conf::SimConfig::Instance().asService();

    mOutFileName = outfilename.c_str();
    mOutFile = new TFile(outfilename.c_str(), "RECREATE");
    mOutTree = new TTree("o2sim", "o2sim");
    mOutTree->SetDirectory(mOutFile);

    // detectors init only once
    if (mDetectorInstances.size() == 0) {
      initDetInstances();
      // has to be after init of Detectors
      o2::utils::ShmManager::Instance().attachToGlobalSegment();
      initHitFiles(o2::conf::SimConfig::Instance().getOutPrefix());
    }

    // init pipe
    auto pipeenv = getenv("ALICE_O2SIMMERGERTODRIVER_PIPE");
    if (pipeenv) {
      mPipeToDriver = atoi(pipeenv);
      LOG(info) << "ASSIGNED PIPE HANDLE " << mPipeToDriver;
    } else {
      LOG(warning) << "DID NOT FIND ENVIRONMENT VARIABLE TO INIT PIPE";
    }

    // if no data to expect we shut down the device NOW since it would otherwise hang
    if (mNExpectedEvents == 0) {
      if (mAsService) {
        waitForControlInput();
      } else {
        LOG(info) << "NOT EXPECTING ANY DATA; SHUTTING DOWN";
        raise(SIGINT);
      }
    }
  }

  bool setWorkingDirectory(std::string const& dir)
  {
    namespace fs = std::filesystem;

    // sets the output directory where simulation files are produced
    // and creates it when it doesn't exist already

    // 2 possibilities:
    // a) dir is relative dir. Then we interpret it as relative to the initial
    //    base directory
    // b) or dir is itself absolut.
    try {
      fs::current_path(fs::path(mInitialOutputDir)); // <--- to make sure relative start is always the same
      if (!dir.empty()) {
        auto absolutePath = fs::absolute(fs::path(dir));
        if (!fs::exists(absolutePath)) {
          if (!fs::create_directory(absolutePath)) {
            LOG(error) << "Could not create directory " << absolutePath.string();
            return false;
          }
        }
        // set the current path
        fs::current_path(absolutePath.string().c_str());
        mCurrentOutputDir = fs::current_path().string();
      }
      LOG(info) << "FINAL PATH " << mCurrentOutputDir;
    } catch (std::exception e) {
      LOG(error) << " could not change path to " << dir;
    }
    return true;
  }

  // function for intermediate/on-the-fly reinitializations
  bool ReInit(o2::conf::SimReconfigData const& reconfig)
  {
    if (reconfig.stop) {
      return false;
    }
    if (!setWorkingDirectory(reconfig.outputDir)) {
      return false;
    }

    std::string outfilename("o2sim_merged_hits.root"); // default name
    outfilename = o2::base::NameConf::getMCKinematicsFileName(reconfig.outputPrefix);
    mNExpectedEvents = reconfig.nEvents;
    mOutFileName = outfilename.c_str();
    mOutFile = new TFile(outfilename.c_str(), "RECREATE");
    mOutTree = new TTree("o2sim", "o2sim");
    mOutTree->SetDirectory(mOutFile);

    // reinit detectorInstance files (also make sure they are closed before continuing)
    initHitFiles(reconfig.outputPrefix);

    // clear "counter" datastructures
    mPartsCheckSum.clear();
    mEventChecksum = 0;

    // clear collector datastructures
    mMCTrackBuffer.clear();
    mTrackRefBuffer.clear();
    mSubEventInfoBuffer.clear();
    mFlushableEvents.clear();

    return true;
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

  void consumeHits(int eventID, fair::mq::Parts& data, int& index)
  {
    auto detIDmessage = std::move(data.At(index++));
    // this should be a detector ID
    if (detIDmessage->GetSize() == 4) {
      auto ptr = (int*)detIDmessage->GetData();
      o2::detectors::DetID id(ptr[0]);
      LOG(debug2) << "I1 " << ptr[0] << " NAME " << id.getName() << " MB "
                  << data.At(index)->GetSize() / 1024. / 1024.;

      // get the detector that can interpret it
      auto detector = mDetectorInstances[id].get();
      if (detector) {
        detector->collectHits(eventID, data, index);
      }
    }
  }

  template <typename T, typename BT>
  void consumeData(int eventID, fair::mq::Parts& data, int& index, BT& buffer)
  {
    auto decodeddata = o2::base::decodeTMessage<T*>(data, index);
    if (buffer.find(eventID) == buffer.end()) {
      buffer[eventID] = typename BT::mapped_type();
    }
    buffer[eventID].push_back(decodeddata);
    // delete decodeddata; --> we store the pointers
    index++;
  }

  // fills a special branch of SubEventInfos in order to keep
  // track of which entry corresponds to which event etc.
  // also creates the MCEventHeader branch expected for physics analysis
  void fillSubEventInfoEntry(o2::data::SubEventInfo& info)
  {
    if (mSubEventInfoBuffer.find(info.eventID) == mSubEventInfoBuffer.end()) {
      mSubEventInfoBuffer[info.eventID] = std::list<o2::data::SubEventInfo*>();
    }
    mSubEventInfoBuffer[info.eventID].push_back(&info);
  }

  bool waitForControlInput()
  {
    o2::simpubsub::publishMessage(fChannels["merger-notifications"].at(0), o2::simpubsub::simStatusString("MERGER", "STATUS", "AWAITING INPUT"));

    auto factory = fair::mq::TransportFactory::CreateTransportFactory("zeromq");
    auto channel = fair::mq::Channel{"o2sim-control", "sub", factory};
    auto controlsocketname = getenv("ALICE_O2SIMCONTROL");
    LOG(info) << "SOCKETNAME " << controlsocketname;
    channel.Connect(std::string(controlsocketname));
    channel.Validate();
    std::unique_ptr<fair::mq::Message> reply(channel.NewMessage());

    LOG(info) << "WAITING FOR INPUT";
    if (channel.Receive(reply) > 0) {
      auto data = reply->GetData();
      auto size = reply->GetSize();

      std::string command(reinterpret_cast<char const*>(data), size);
      LOG(info) << "message: " << command;

      o2::conf::SimReconfigData reconfig;
      o2::conf::parseSimReconfigFromString(command, reconfig);
      return ReInit(reconfig);
    } else {
      LOG(info) << "NOTHING RECEIVED";
    }
    return true;
  }

  bool ConditionalRun() override
  {
    auto& channel = fChannels.at("simdata").at(0);
    fair::mq::Parts request;
    auto bytes = channel.Receive(request);
    if (bytes < 0) {
      LOG(error) << "Some error occurred on socket during receive on sim data";
      return true; // keep going
    }
    TStopwatch timer;
    timer.Start();
    auto more = handleSimData(request, 0);
    LOG(info) << "HitMerger processing took " << timer.RealTime();
    if (!more && mAsService) {
      LOG(info) << " CONTROL ";
      // if we are done treating data we may go back to init phase
      // for the next batch
      return waitForControlInput();
    }
    return more;
  }

  bool handleSimData(fair::mq::Parts& data, int /*index*/)
  {
    bool expectmore = true;
    int index = 0;
    auto infoptr = o2::base::decodeTMessage<o2::data::SubEventInfo*>(data, index++);
    o2::data::SubEventInfo& info = *infoptr;
    auto accum = insertAdd<uint32_t, uint32_t>(mPartsCheckSum, info.eventID, (uint32_t)info.part);

    LOG(info) << "SIMDATA channel got " << data.Size() << " parts for event " << info.eventID << " part " << info.part << " out of " << info.nparts;

    fillSubEventInfoEntry(info);
    consumeData<std::vector<o2::MCTrack>>(info.eventID, data, index, mMCTrackBuffer);
    consumeData<std::vector<o2::TrackReference>>(info.eventID, data, index, mTrackRefBuffer);
    while (index < data.Size()) {
      consumeHits(info.eventID, data, index);
    }

    if (isDataComplete<uint32_t>(accum, info.nparts)) {
      LOG(info) << "Event " << info.eventID << " complete. Marking as flushable";
      mFlushableEvents[info.eventID] = true;

      // check if previous flush finished
      // start merging only when no merging currently happening
      // Like this we don't have to join/wait on the thread here and do not block the outer ConditionalRun handling
      // TODO: Let this run fully asynchronously (not even triggered by ConditionalRun)
      if (!mergingInProgress) {
        if (mMergerIOThread.joinable()) {
          mMergerIOThread.join();
        }
        // start hit merging and flushing in a separate thread in order not to block
        mMergerIOThread = std::thread([info, this]() { mergingInProgress = true; mergeAndFlushData(); mergingInProgress = false; });
      }

      mEventChecksum += info.eventID;
      // we also need to check if we have all events
      if (isDataComplete<uint32_t>(mEventChecksum, info.maxEvents)) {
        LOG(info) << "ALL EVENTS HERE; CHECKSUM " << mEventChecksum;

        // flush remaining data and close file
        if (mMergerIOThread.joinable()) {
          mMergerIOThread.join();
        }
        mMergerIOThread = std::thread([info, this]() { mergingInProgress = true; mergeAndFlushData(); mergingInProgress = false; });
        if (mMergerIOThread.joinable()) {
          mMergerIOThread.join();
        }

        expectmore = false;
      }

      if (mPipeToDriver != -1) {
        if (write(mPipeToDriver, &info.eventID, sizeof(info.eventID)) == -1) {
          LOG(error) << "FAILED WRITING TO PIPE";
        };
      }
    }
    if (!expectmore) {
      // somehow FairMQ has difficulties shutting down; helping manually
      // raise(SIGINT);
    }
    return expectmore;
  }

  void cleanEvent(int eventID)
  {
    // cleanup intermediate per-Event buffers
  }

  template <typename T>
  void backInsert(T const& from, T& to)
  {
    std::copy(from.begin(), from.end(), std::back_inserter(to));
  }

  void reorderAndMergeMCTracks(int eventID, TTree& target, const std::vector<int>& nprimaries, const std::vector<int>& nsubevents, std::function<void(std::vector<MCTrack> const&)> tracks_analysis_hook)
  {
    // avoid doing this for trivial cases
    std::vector<MCTrack>* mcTracksPerSubEvent = nullptr;
    auto targetdata = std::make_unique<std::vector<MCTrack>>();

    auto& vectorOfSubEventMCTracks = mMCTrackBuffer[eventID];
    const auto entries = vectorOfSubEventMCTracks.size();

    if (entries > 1) {
      //
      // loop over subevents to store the primary events
      //
      int nprimTot = 0;
      for (int entry = entries - 1; entry >= 0; --entry) {
        int index = nsubevents[entry];
        nprimTot += nprimaries[index];
        printf("merge %d %5d %5d %5d \n", entry, index, nsubevents[entry], nsubevents[index]);
        for (int i = 0; i < nprimaries[index]; i++) {
          auto& track = (*vectorOfSubEventMCTracks[index])[i];
          if (track.isTransported()) { // reset daughters only if track was transported, it will be fixed below
            track.SetFirstDaughterTrackId(-1);
            track.SetLastDaughterTrackId(-1);
          }
          targetdata->push_back(track);
        }
      }
      //
      // loop a second time to store the secondaries and fix the mother track IDs
      //
      Int_t idelta1 = nprimTot;
      Int_t idelta0 = 0;
      for (int entry = entries - 1; entry >= 0; --entry) {
        int index = nsubevents[entry];

        auto& subEventTracks = *(vectorOfSubEventMCTracks[index]);
        // we need to fetch the right mctracks here!!
        Int_t npart = (int)(subEventTracks.size());
        Int_t nprim = nprimaries[index];
        idelta1 -= nprim;

        for (Int_t i = nprim; i < npart; i++) {
          auto& track = subEventTracks[i];
          Int_t cId = track.getMotherTrackId();
          if (cId >= nprim) {
            cId += idelta1;
          } else {
            cId += idelta0;
          }
          track.SetMotherTrackId(cId);
          track.SetFirstDaughterTrackId(-1);

          Int_t hwm = (int)(targetdata->size());
          auto& mother = (*targetdata)[cId];
          if (mother.getFirstDaughterTrackId() == -1) {
            mother.SetFirstDaughterTrackId(hwm);
          }
          mother.SetLastDaughterTrackId(hwm);

          targetdata->push_back(track);
        }
        idelta0 += nprim;
        idelta1 += npart;
      }
    }
    //
    // write to output
    auto filladdr = (entries > 1) ? targetdata.get() : vectorOfSubEventMCTracks[0];

    // we give the possibility to produce some MC track statistics
    // to be saved as part of the MCHeader structure
    tracks_analysis_hook(*filladdr);

    auto targetbr = o2::base::getOrMakeBranch(target, "MCTrack", &filladdr);
    targetbr->SetAddress(&filladdr);
    targetbr->Fill();
    targetbr->ResetAddress();

    // cleanup buffered data
    for (auto ptr : vectorOfSubEventMCTracks) {
      delete ptr; // avoid this by using unique ptr
    }
  }

  template <typename T, typename M>
  void remapTrackIdsAndMerge(std::string brname, int eventID, TTree& target,
                             const std::vector<int>& trackoffsets, const std::vector<int>& nprimaries, const std::vector<int>& subevOrdered, M& mapOfVectorOfTs)
  {
    //
    // Remap the mother track IDs by adding an offset.
    // The offset calculated as the sum of the number of entries in the particle list of the previous subevents.
    // This method is called by O2HitMerger::mergeAndFlushData(int)
    //
    T* incomingdata = nullptr;
    std::unique_ptr<T> targetdata(nullptr);
    auto& vectorOfT = mapOfVectorOfTs[eventID];
    const auto entries = vectorOfT.size();

    if (entries == 1) {
      // nothing to do in case there is only one entry
      incomingdata = vectorOfT[0];
    } else {
      targetdata = std::make_unique<T>();
      // loop over subevents
      Int_t nprimTot = 0;
      for (int entry = 0; entry < entries; entry++) {
        nprimTot += nprimaries[entry];
      }
      Int_t idelta0 = 0;
      Int_t idelta1 = nprimTot;
      for (int entry = entries - 1; entry >= 0; --entry) {
        Int_t index = subevOrdered[entry];
        Int_t nprim = nprimaries[index];
        incomingdata = vectorOfT[index];
        idelta1 -= nprim;
        for (auto& data : *incomingdata) {
          updateTrackIdWithOffset(data, nprim, idelta0, idelta1);
          targetdata->push_back(data);
        }
        idelta0 += nprim;
        idelta1 += trackoffsets[index];
      }
    }
    auto dataaddr = (entries == 1) ? incomingdata : targetdata.get();
    auto targetbr = o2::base::getOrMakeBranch(target, brname.c_str(), &dataaddr);
    targetbr->SetAddress(&dataaddr);
    targetbr->Fill();
    targetbr->ResetAddress();

    // cleanup mem
    for (auto ptr : vectorOfT) {
      delete ptr; // avoid this by using unique ptr
    }
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

  void initHitTreeAndOutFile(std::string prefix, int detID)
  {
    using o2::detectors::DetID;
    if (mDetectorOutFiles[detID]) {
      LOG(warn) << "Hit outfile for detID " << DetID::getName(detID) << " already initialized --> Reopening";
      mDetectorOutFiles[detID]->Close();
      delete mDetectorOutFiles[detID];
    }
    std::string name(o2::base::DetectorNameConf::getHitsFileName(detID, prefix));
    mDetectorOutFiles[detID] = new TFile(name.c_str(), "RECREATE");
    mDetectorToTTreeMap[detID] = new TTree("o2sim", "o2sim");
    mDetectorToTTreeMap[detID]->SetDirectory(mDetectorOutFiles[detID]);
  }

  // This method goes over the buffers containing data for a given event; potentially merges
  // them and flushes into the actual output file.
  // The method can be called asynchronously to data collection
  bool mergeAndFlushData()
  {
    auto checkIfNextFlushable = [this]() -> bool {
      mNextFlushID++;
      return mFlushableEvents.find(mNextFlushID) != mFlushableEvents.end() && mFlushableEvents[mNextFlushID] == true;
    };

    LOG(info) << "Launching merge kernel ";
    bool canflush = mFlushableEvents.find(mNextFlushID) != mFlushableEvents.end() && mFlushableEvents[mNextFlushID] == true;
    if (!canflush) {
      return false;
    }
    while (canflush == true) {
      auto flusheventID = mNextFlushID;
      LOG(info) << "Merge and flush event " << flusheventID;
      auto iter = mSubEventInfoBuffer.find(flusheventID);
      if (iter == mSubEventInfoBuffer.end()) {
        LOG(error) << "No info/data found for event " << flusheventID;
        if (!checkIfNextFlushable()) {
          return false;
        }
      }

      auto& subEventInfoList = (*iter).second;
      if (subEventInfoList.size() == 0 || mNExpectedEvents == 0) {
        LOG(error) << "No data entries found for event " << flusheventID;
        if (!checkIfNextFlushable()) {
          return false;
        }
      }

      TStopwatch timer;
      timer.Start();

      // calculate trackoffsets
      auto& confref = o2::conf::SimConfig::Instance();

      // collecting trackoffsets (per data arrival id) to be used for global track-ID correction pass
      std::vector<int> trackoffsets;
      // collecting primary particles in each subevent (data arrival id)
      std::vector<int> nprimaries;
      // mapping of id to actual sub-event id (or part)
      std::vector<int> nsubevents;

      o2::dataformats::MCEventHeader* eventheader = nullptr; // The event header

      // the MC labels (trackID) for hits
      for (auto info : subEventInfoList) {
        assert(info->npersistenttracks >= 0);
        trackoffsets.emplace_back(info->npersistenttracks);
        nprimaries.emplace_back(info->nprimarytracks);
        nsubevents.emplace_back(info->part);
        if (eventheader == nullptr) {
          eventheader = &info->mMCEventHeader;
        } else {
          eventheader->getMCEventStats().add(info->mMCEventHeader.getMCEventStats());
        }
      }

      // now see which events can be discarded in any case due to no hits
      if (confref.isFilterOutNoHitEvents()) {
        if (eventheader && eventheader->getMCEventStats().getNHits() == 0) {
          LOG(info) << " Taking out event " << flusheventID << " due to no hits ";
          cleanEvent(flusheventID);
          if (!checkIfNextFlushable()) {
            return true;
          }
        }
      }

      // put the event headers into the new TTree
      auto headerbr = o2::base::getOrMakeBranch(*mOutTree, "MCEventHeader.", &eventheader);

      // attention: We need to make sure that we write everything in the same event order
      // but iteration over keys of a standard map in C++ is ordered

      // b) merge the general data
      //
      // for MCTrack remap the motherIds and merge at the same go
      const auto entries = subEventInfoList.size();
      std::vector<int> subevOrdered((int)(nsubevents.size()));
      for (int entry = entries - 1; entry >= 0; --entry) {
        subevOrdered[nsubevents[entry] - 1] = entry;
        printf("HitMerger entry: %d nprimry: %5d trackoffset: %5d \n", entry, nprimaries[entry], trackoffsets[entry]);
      }

      // This is a hook that collects some useful statistics/properties on the event
      // for use by other components;
      // Properties are attached making use of the extensible "Info" feature which is already
      // part of MCEventHeader. In such a way, one can also do this pass outside and attach arbitrary
      // metadata to MCEventHeader without needing to change the data layout or API of the class itself.
      // NOTE: This function might also be called directly in the primary server!?
      auto mcheaderhook = [eventheader](std::vector<MCTrack> const& tracks) {
        int eta1Point2Counter = 0;
        int eta1Point0Counter = 0;
        int eta0Point8Counter = 0;
        int eta1Point2CounterPi = 0;
        int eta1Point0CounterPi = 0;
        int eta0Point8CounterPi = 0;
        int prims = 0;
        for (auto& tr : tracks) {
          if (tr.isPrimary()) {
            prims++;
            const auto eta = tr.GetEta();
            if (eta < 1.2) {
              eta1Point2Counter++;
              if (std::abs(tr.GetPdgCode()) == 211) {
                eta1Point2CounterPi++;
              }
            }
            if (eta < 1.0) {
              eta1Point0Counter++;
              if (std::abs(tr.GetPdgCode()) == 211) {
                eta1Point0CounterPi++;
              }
            }
            if (eta < 0.8) {
              eta0Point8Counter++;
              if (std::abs(tr.GetPdgCode()) == 211) {
                eta0Point8CounterPi++;
              }
            }
          } else {
            break; // track layout is such that all prims are first anyway
          }
        }
        // attach these properties to eventheader
        // we only need to make the names standard
        eventheader->putInfo("prims_eta_1.2", eta1Point2Counter);
        eventheader->putInfo("prims_eta_1.0", eta1Point0Counter);
        eventheader->putInfo("prims_eta_0.8", eta0Point8Counter);
        eventheader->putInfo("prims_eta_1.2_pi", eta1Point2CounterPi);
        eventheader->putInfo("prims_eta_1.0_pi", eta1Point0CounterPi);
        eventheader->putInfo("prims_eta_0.8_pi", eta0Point8CounterPi);
        eventheader->putInfo("prims_total", prims);
      };

      reorderAndMergeMCTracks(flusheventID, *mOutTree, nprimaries, subevOrdered, mcheaderhook);
      remapTrackIdsAndMerge<std::vector<o2::TrackReference>>("TrackRefs", flusheventID, *mOutTree, trackoffsets, nprimaries, subevOrdered, mTrackRefBuffer);

      // header can be written
      headerbr->SetAddress(&eventheader);
      headerbr->Fill();
      headerbr->ResetAddress();

      // c) do the merge procedure for all hits ... delegate this to detector specific functions
      // since they know about types; number of branches; etc.
      // this will also fix the trackIDs inside the hits
      for (int id = 0; id < mDetectorInstances.size(); ++id) {
        auto& det = mDetectorInstances[id];
        if (det) {
          auto hittree = mDetectorToTTreeMap[id];
          // det->mergeHitEntries(*tree, *hittree, trackoffsets, nprimaries, subevOrdered);
          det->mergeHitEntriesAndFlush(flusheventID, *hittree, trackoffsets, nprimaries, subevOrdered);
          hittree->SetEntries(hittree->GetEntries() + 1);
          LOG(info) << "flushing tree to file " << hittree->GetDirectory()->GetFile()->GetName();
        }
      }

      // increase the entry count in the tree
      mOutTree->SetEntries(mOutTree->GetEntries() + 1);
      LOG(info) << "outtree has file " << mOutTree->GetDirectory()->GetFile()->GetName();

      cleanEvent(flusheventID);
      LOG(info) << "Merge/flush for event " << flusheventID << " took " << timer.RealTime();
      if (!checkIfNextFlushable()) {
        break;
      }
    } // end while
    LOG(info) << "Writing TTrees";
    mOutFile->Write("", TObject::kOverwrite);
    for (int id = 0; id < mDetectorInstances.size(); ++id) {
      auto& det = mDetectorInstances[id];
      if (det) {
        mDetectorOutFiles[id]->Write("", TObject::kOverwrite);
      }
    }

    return true;
  }

  std::map<uint32_t, uint32_t> mPartsCheckSum; //! mapping event id -> part checksum used to detect when all info
  std::string mOutFileName;                    //!

  // structures for the final flush
  TFile* mOutFile; //! outfile for kinematics
  TTree* mOutTree; //! tree (kinematics) associated to mOutFile

  template <class K, class V>
  using Hashtable = tbb::concurrent_unordered_map<K, V>;
  Hashtable<int, TFile*> mDetectorOutFiles;   //! outfiles per detector for hits
  Hashtable<int, TTree*> mDetectorToTTreeMap; //! the trees

  // intermediate structures to collect data per event
  std::thread mMergerIOThread; //! a thread used to do hit merging and IO flushing asynchronously
  bool mergingInProgress = false;

  Hashtable<int, std::vector<std::vector<o2::MCTrack>*>> mMCTrackBuffer;         //! vector of sub-event track vectors; one per event
  Hashtable<int, std::vector<std::vector<o2::TrackReference>*>> mTrackRefBuffer; //!
  Hashtable<int, std::list<o2::data::SubEventInfo*>> mSubEventInfoBuffer;
  Hashtable<int, bool> mFlushableEvents; //! collection of events which have completely arrived

  int mEventChecksum = 0;   //! checksum for events
  int mNExpectedEvents = 0; //! number of events that we expect to receive
  int mNextFlushID = 1;     //! EventID to be flushed next
  TStopwatch mTimer;

  bool mAsService = false; //! if run in deamonized mode

  int mPipeToDriver = -1;

  std::vector<std::unique_ptr<o2::base::Detector>> mDetectorInstances; //!

  // output folder configuration
  std::string mInitialOutputDir; // initial output folder of the process (initialized during construction)
  std::string mCurrentOutputDir; // current output folder asked

  // channel to PUB status messages to outside subscribers
  fair::mq::Channel mPubChannel;

  // init detector instances
  void initDetInstances();
  void initHitFiles(std::string prefix);
};

void O2HitMerger::initHitFiles(std::string prefix)
{
  using o2::detectors::DetID;

  // a little helper lambda
  auto isActivated = [](std::string s) -> bool {
    // access user configuration for list of wanted modules
    auto& modulelist = o2::conf::SimConfig::Instance().getReadoutDetectors();
    auto active = std::find(modulelist.begin(), modulelist.end(), s) != modulelist.end();
    return active; };

  for (int i = DetID::First; i <= DetID::Last; ++i) {
    if (!isActivated(DetID::getName(i))) {
      continue;
    }
    // init the detector specific output files
    initHitTreeAndOutFile(prefix, i);
  }
}

// init detector instances used to write hit data to a TTree
void O2HitMerger::initDetInstances()
{
  using o2::detectors::DetID;

  // a little helper lambda
  auto isActivated = [](std::string s) -> bool {
    // access user configuration for list of wanted modules
    auto& modulelist = o2::conf::SimConfig::Instance().getReadoutDetectors();
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
      mDetectorInstances[i] = std::move(std::make_unique<o2::mft::Detector>(true));
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
    if (i == DetID::FT3) {
      mDetectorInstances[i] = std::move(std::make_unique<o2::ft3::Detector>(true));
      counter++;
    }
    if (i == DetID::FCT) {
      mDetectorInstances[i] = std::move(std::make_unique<o2::fct::Detector>(true));
      counter++;
    }
#endif
  }
  if (counter != DetID::nDetectors) {
    LOG(warning) << " O2HitMerger: Some Detectors are potentially missing in this initialization ";
  }
}

} // namespace devices
} // namespace o2

#endif
