// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <Steer/O2MCApplication.h>
#include <FairMQChannel.h>
#include <FairMQMessage.h>
#include <FairMQDevice.h>
#include <FairMQParts.h>
#include <ITSMFTSimulation/Hit.h>
#include <TPCSimulation/Point.h>
#include <SimulationDataFormat/PrimaryChunk.h>
#include <TMessage.h>
#include <sstream>
#include <SimConfig/SimConfig.h>
#include <DetectorsBase/Detector.h>
#include <CommonUtils/ShmManager.h>
#include <cassert>
#include <SimulationDataFormat/MCEventHeader.h>
#include <TGeoManager.h>
#include <fstream>
#include <FairVolume.h>

namespace o2
{
namespace steer
{
// helper function to send trivial data
template <typename T>
void TypedVectorAttach(const char* name, FairMQChannel& channel, FairMQParts& parts)
{
  static auto mgr = FairRootManager::Instance();
  auto vector = mgr->InitObjectAs<const std::vector<T>*>(name);
  if (vector) {
    auto buffer = (char*)&(*vector)[0];
    auto buffersize = vector->size() * sizeof(T);
    FairMQMessagePtr message(channel.NewMessage(
      buffer, buffersize,
      [](void* data, void* hint) {}, buffer));
    parts.AddPart(std::move(message));
  }
}

void O2MCApplicationBase::Stepping()
{
  mStepCounter++;
  if (mCutParams.stepFiltering) {
    // we can kill tracks here based on our
    // custom detector specificities

    float x, y, z;
    fMC->TrackPosition(x, y, z);

    if (z > mCutParams.ZmaxA) {
      fMC->StopTrack();
      return;
    }
    if (-z > mCutParams.ZmaxC) {
      fMC->StopTrack();
      return;
    }
  }

  // dispatch first to stepping function in FairRoot
  FairMCApplication::Stepping();
}

void O2MCApplicationBase::PreTrack()
{
  // dispatch first to function in FairRoot
  FairMCApplication::PreTrack();
}

void O2MCApplicationBase::ConstructGeometry()
{
  // fill the mapping
  mModIdToName.clear();
  for (int i = 0; i < fModules->GetEntries(); ++i) {
    auto mod = static_cast<FairModule*>(fModules->At(i));
    if (mod) {
      mModIdToName[mod->GetModId()] = mod->GetName();
    }
  }

  FairMCApplication::ConstructGeometry();

  std::ofstream voltomodulefile("MCStepLoggerVolMap.dat");
  // construct the volume name to module name mapping useful for StepAnalysis
  auto vollist = gGeoManager->GetListOfVolumes();
  for (int i = 0; i < vollist->GetEntries(); ++i) {
    auto vol = static_cast<TGeoVolume*>(vollist->At(i));
    auto iter = fModVolMap.find(vol->GetNumber());
    voltomodulefile << vol->GetName() << ":" << mModIdToName[iter->second] << "\n";
  }
}

void O2MCApplicationBase::InitGeometry()
{
  FairMCApplication::InitGeometry();
  // now the sensitive volumes are set up in fVolMap and we can query them
  for (auto e : fVolMap) {
    // since fVolMap contains multiple entries (if multiple copies), this may
    // write to the same entry multiple times
    mSensitiveVolumes[e.first] = e.second->GetName();
  }
  std::ofstream sensvolfile("MCStepLoggerSenVol.dat");
  for (auto e : mSensitiveVolumes) {
    sensvolfile << e.first << ":" << e.second << "\n";
  }
}

void O2MCApplicationBase::finishEventCommon()
{
  LOG(INFO) << "This event/chunk did " << mStepCounter << " steps";

  auto header = static_cast<o2::dataformats::MCEventHeader*>(fMCEventHeader);
  header->getMCEventStats().setNSteps(mStepCounter);

  static_cast<o2::data::Stack*>(GetStack())->updateEventStats();
}

void O2MCApplicationBase::FinishEvent()
{
  finishEventCommon();

  auto header = static_cast<o2::dataformats::MCEventHeader*>(fMCEventHeader);
  auto& confref = o2::conf::SimConfig::Instance();

  if (confref.isFilterOutNoHitEvents() && header->getMCEventStats().getNHits() == 0) {
    LOG(INFO) << "Discarding current event due to no hits";
    SetSaveCurrentEvent(false);
  }

  // dispatch to function in FairRoot
  FairMCApplication::FinishEvent();
}

void O2MCApplicationBase::BeginEvent()
{
  // dispatch first to function in FairRoot
  FairMCApplication::BeginEvent();

  // register event header with our stack
  auto header = static_cast<o2::dataformats::MCEventHeader*>(fMCEventHeader);
  static_cast<o2::data::Stack*>(GetStack())->setMCEventStats(&header->getMCEventStats());

  mStepCounter = 0;
}

void O2MCApplication::initLate()
{
  o2::utils::ShmManager::Instance().occupySegment();
  for (auto det : listActiveDetectors) {
    if (dynamic_cast<o2::base::Detector*>(det)) {
      ((o2::base::Detector*)det)->initializeLate();
    }
  }
}

void O2MCApplication::attachSubEventInfo(FairMQParts& parts, o2::data::SubEventInfo const& info) const
{
  // parts.AddPart(std::move(mSimDataChannel->NewSimpleMessage(info)));
  o2::base::attachTMessage(info, *mSimDataChannel, parts);
}

// helper function to fetch data from FairRootManager branch and serialize it
// returns handle to container
template <typename T>
const T* attachBranch(std::string const& name, FairMQChannel& channel, FairMQParts& parts)
{
  auto mgr = FairRootManager::Instance();
  // check if branch is present
  if (mgr->GetBranchId(name) == -1) {
    LOG(ERROR) << "Branch " << name << " not found";
    return nullptr;
  }
  auto data = mgr->InitObjectAs<const T*>(name.c_str());
  if (data) {
    o2::base::attachTMessage(*data, channel, parts);
  }
  return data;
}

void O2MCApplication::setSubEventInfo(o2::data::SubEventInfo* i)
{
  mSubEventInfo = i;
  // being communicated a SubEventInfo also means we get a FairMCEventHeader
  fMCEventHeader = &mSubEventInfo->mMCEventHeader;
}

void O2MCApplication::SendData()
{
  FairMQParts simdataparts;

  // fill these parts ... the receiver has to unpack similary
  // TODO: actually we could just loop over branches in FairRootManager at this moment?
  mSubEventInfo->npersistenttracks = static_cast<o2::data::Stack*>(GetStack())->getMCTracks()->size();
  attachSubEventInfo(simdataparts, *mSubEventInfo);
  auto tracks = attachBranch<std::vector<o2::MCTrack>>("MCTrack", *mSimDataChannel, simdataparts);
  attachBranch<std::vector<o2::TrackReference>>("TrackRefs", *mSimDataChannel, simdataparts);
  attachBranch<o2::dataformats::MCTruthContainer<o2::TrackReference>>("IndexedTrackRefs", *mSimDataChannel, simdataparts);
  assert(tracks->size() == mSubEventInfo->npersistenttracks);

  for (auto det : listActiveDetectors) {
    if (dynamic_cast<o2::base::Detector*>(det)) {
      ((o2::base::Detector*)det)->attachHits(*mSimDataChannel, simdataparts);
    }
  }
  LOG(INFO) << "sending message with " << simdataparts.Size() << " parts";
  mSimDataChannel->Send(simdataparts);
}
} // namespace steer
} // namespace o2
