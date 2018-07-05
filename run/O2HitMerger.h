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
#include <FITSimulation/Detector.h>
#include <HMPIDSimulation/Detector.h>
#include <PHOSSimulation/Detector.h>
#include <MCHSimulation/Detector.h>

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
    OnData("simdata", &O2HitMerger::handleSimData);
    mTimer.Start();
  }

  /// Default destructor
  ~O2HitMerger() override
  {
    FairSystemInfo sysinfo;
    mOutTree->SetEntries(mEntries);
    mOutTree->Write();
    mOutFile->Close();
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
    }

    mOutFile = new TFile(outfilename.c_str(), "RECREATE");
    mOutTree = new TTree("o2sim", "o2sim");
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
    LOG(INFO) << detIDmessage->GetSize();
    if (detIDmessage->GetSize() == 4) {
      auto ptr = (int*)detIDmessage->GetData();
      o2::detectors::DetID id(ptr[0]);
      LOG(INFO) << "I1 " << ptr[0] << " NAME " << id.getName();

      // get the detector than can interpret it
      auto detector = mDetectorInstances[id].get();
      detector->fillHitBranch(*mOutTree, data, index);
    }
  }

  template <typename T>
  void consumeData(std::string name, FairMQParts& data, int& index)
  {
    auto decodeddata = o2::Base::decodeTMessage<T*>(data, index);
    auto br = o2::Base::getOrMakeBranch(*mOutTree, name.c_str(), &decodeddata);
    br->SetAddress(&decodeddata);
    br->Fill();
    br->ResetAddress();
    index++;
  }

  bool handleSimData(FairMQParts& data, int /*index*/)
  {
    LOG(INFO) << "SIMDATA channel got " << data.Size() << " parts\n";

    // extract the event info
    auto infomessage = std::move(data.At(0));
    o2::Data::SubEventInfo info;
    // could actually avoid the copy
    memcpy((void*)&info, infomessage->GetData(), infomessage->GetSize());

    auto accum = insertAdd<uint32_t, uint32_t>(mPartsCheckSum, info.eventID, (uint32_t)info.part);
    // auto totalsize = insertAdd<uint32_t, size_t>(mITSTotalSize, info.eventID, itshits.size());

    int index = 1;
    consumeData<std::vector<o2::MCTrack>>("MCTrack", data, index);
    consumeData<std::vector<o2::TrackReference>>("TrackRefs", data, index);
    consumeData<o2::dataformats::MCTruthContainer<o2::TrackReference>>("IndexedTrackRefs", data, index);
    while (index < data.Size()) {
      consumeHits(data, index);
    }
    mEntries++;

    if (isDataComplete<uint32_t>(accum, info.nparts)) {
      LOG(INFO) << "EVERYTHING IS HERE FOR EVENT " << info.eventID << "\n";
      mEventChecksum += info.eventID;
      // we also need to check if we have all events
      if (isDataComplete<uint32_t>(mEventChecksum, info.maxEvents)) {
        return false;
      }
    }
    return true;
  }

  std::map<uint32_t, uint32_t> mPartsCheckSum; //! mapping event id -> part checksum used to detect when all info
  std::map<uint32_t, size_t> mITSTotalSize;    //! total number of tracks for a given event

  TFile* mOutFile;  //!
  TTree* mOutTree;  //!
  int mEntries = 0; //! counts the number of entries in the branches
  int mEventChecksum = 0; //! checksum for events
  TStopwatch mTimer;

  std::vector<std::unique_ptr<o2::Base::Detector>> mDetectorInstances;

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
      mDetectorInstances[i] = std::move(std::make_unique<o2::TPC::Detector>(true));
      counter++;
    }
    if (i == DetID::ITS) {
      mDetectorInstances[i] = std::move(std::make_unique<o2::ITS::Detector>(true));
      counter++;
    }
    if (i == DetID::MFT) {
      mDetectorInstances[i] = std::move(std::make_unique<o2::MFT::Detector>());
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
    if (i == DetID::EMC) {
      mDetectorInstances[i] = std::move(std::make_unique<o2::EMCAL::Detector>(true));
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
    if (i == DetID::FIT) {
      mDetectorInstances[i] = std::move(std::make_unique<o2::fit::Detector>(true));
      counter++;
    }
    if (i == DetID::MCH) {
      mDetectorInstances[i] = std::move(std::make_unique<o2::mch::Detector>(true));
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
