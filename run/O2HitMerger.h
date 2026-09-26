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

#include <atomic>
#include <functional>
#include <list>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include <fairmq/Device.h>
#include <fairmq/Parts.h>
#include <TStopwatch.h>
#include <tbb/concurrent_unordered_map.h>
#include <DetectorsBase/Detector.h>
#include <SimConfig/SimConfig.h>
#include <SimulationDataFormat/MCEventHeader.h>
#include <SimulationDataFormat/MCTrack.h>
#include <SimulationDataFormat/PrimaryChunk.h>
#include <SimulationDataFormat/TrackReference.h>

class TFile;
class TTree;

namespace o2
{
namespace devices
{

class O2HitMerger : public fair::mq::Device
{
 public:
  /// Default constructor
  O2HitMerger();

  /// Default destructor
  ~O2HitMerger() override;

 private:
  /// Overloads the InitTask() method of fair::mq::Device
  void InitTask() final;

  bool setWorkingDirectory(std::string const& dir);

  // function for intermediate/on-the-fly reinitializations
  bool ReInit(o2::conf::SimReconfigData const& reconfig);

  template <typename T, typename V>
  V insertAdd(std::map<T, V>& m, T const& key, V value);

  template <typename T>
  bool isDataComplete(T checksum, T nparts);

  void consumeHits(int eventID, fair::mq::Parts& data, int& index);

  template <typename T, typename BT>
  void consumeData(int eventID, fair::mq::Parts& data, int& index, BT& buffer);

  // fills a special branch of SubEventInfos in order to keep
  // track of which entry corresponds to which event etc.
  // also creates the MCEventHeader branch expected for physics analysis
  void fillSubEventInfoEntry(o2::data::SubEventInfo& info);

  bool waitForControlInput();

  bool ConditionalRun() override;

  bool handleSimData(fair::mq::Parts& data, int /*index*/);

  // releases the buffered data of an event once it is flushed or discarded
  void cleanEvent(int eventID);

  template <typename T>
  void backInsert(T const& from, T& to);

  void reorderAndMergeMCTracks(int eventID, TTree* target, const std::vector<int>& nprimaries, const std::vector<int>& nsubevents, std::function<void(std::vector<MCTrack> const&)> tracks_analysis_hook, o2::dataformats::MCEventHeader const* mceventheader);

  template <typename T, typename M>
  void remapTrackIdsAndMerge(std::string brname, int eventID, TTree& target,
                             const std::vector<int>& trackoffsets, const std::vector<int>& nprimaries, const std::vector<int>& subevOrdered, M& mapOfVectorOfTs);

  void updateTrackIdWithOffset(MCTrack& track, Int_t nprim, Int_t idelta0, Int_t idelta1);

  void updateTrackIdWithOffset(TrackReference& ref, Int_t nprim, Int_t idelta0, Int_t idelta1);

  void initHitTreeAndOutFile(std::string prefix, int detID);

  // This method goes over the buffers containing data for a given event; potentially merges
  // them and flushes into the actual output file.
  // The method can be called asynchronously to data collection
  bool mergeAndFlushData();

  std::map<uint32_t, uint32_t> mPartsCheckSum; //! mapping event id -> part checksum used to detect when all info
  std::string mOutFileName;                    //!

  // structures for the final flush
  TFile* mOutFile = nullptr;             //! outfile for kinematics
  TTree* mOutTree = nullptr;             //! tree (kinematics) associated to mOutFile
  TFile* mMCHeaderOnlyOutFile = nullptr; //! outfile for header only information
  TTree* mMCHeaderTree = nullptr;        //! tree to hold MCHeader branch in mMCHeaderOnlyOutFile;

  template <class K, class V>
  using Hashtable = tbb::concurrent_unordered_map<K, V>;
  Hashtable<int, TFile*> mDetectorOutFiles;   //! outfiles per detector for hits
  Hashtable<int, TTree*> mDetectorToTTreeMap; //! the trees

  // intermediate structures to collect data per event
  std::thread mMergerIOThread; //! a thread used to do hit merging and IO flushing asynchronously
  std::atomic<bool> mergingInProgress{false};

  Hashtable<int, std::vector<std::vector<o2::MCTrack>*>> mMCTrackBuffer;         //! vector of sub-event track vectors; one per event
  Hashtable<int, std::vector<std::vector<o2::TrackReference>*>> mTrackRefBuffer; //!
  Hashtable<int, std::list<o2::data::SubEventInfo*>> mSubEventInfoBuffer;
  Hashtable<int, bool> mFlushableEvents; //! collection of events which have completely arrived

  int mEventChecksum = 0;   //! checksum for events
  int mNExpectedEvents = 0; //! number of events that we expect to receive
  int mNextFlushID = 1;     //! EventID to be flushed next
  TStopwatch mTimer;

  bool mAsService = false;  //! if run in deamonized mode
  bool mForwardKine = true; //! if we forward kinematics (tracks, eventheaders) on some output channel
  bool mWriteToDisc = true; //! if we want to write simulation products to disc

  int mPipeToDriver = -1;

  std::vector<std::unique_ptr<o2::base::Detector>> mDetectorInstances; //!
  std::vector<int> mExternalDetIDs;                                    //! DetID slots occupied by external (CAD) detectors

  // output folder configuration
  std::string mInitialOutputDir; // initial output folder of the process (initialized during construction)
  std::string mCurrentOutputDir; // current output folder asked

  // channel to PUB status messages to outside subscribers
  fair::mq::Channel mPubChannel;

  // init detector instances
  void initDetInstances();
  void initExternalDetInstances();
  void initHitFiles(std::string prefix);
};

} // namespace devices
} // namespace o2

#endif
