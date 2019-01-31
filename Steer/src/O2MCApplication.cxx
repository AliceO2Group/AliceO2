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
#include <FairMCEventHeader.h>

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
    FairMQMessagePtr message(channel.NewMessage(buffer, buffersize,
                                                [](void* data, void* hint) {}, buffer));
    parts.AddPart(std::move(message));
  }
}

void O2MCApplication::initLate()
{
  o2::utils::ShmManager::Instance().occupySegment();
  for (auto det : listActiveDetectors) {
    if (dynamic_cast<o2::Base::Detector*>(det)) {
      ((o2::Base::Detector*)det)->initializeLate();
    }
  }
}

void O2MCApplication::attachSubEventInfo(FairMQParts& parts, o2::Data::SubEventInfo const& info) const
{
  // parts.AddPart(std::move(mSimDataChannel->NewSimpleMessage(info)));
  o2::Base::attachTMessage(info, *mSimDataChannel, parts);
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
    o2::Base::attachTMessage(*data, channel, parts);
  }
  return data;
}

void O2MCApplication::SendData()
{
  FairMQParts simdataparts;

  // fill these parts ... the receiver has to unpack similary
  // TODO: actually we could just loop over branches in FairRootManager at this moment?
  mSubEventInfo.npersistenttracks = static_cast<o2::Data::Stack*>(GetStack())->getMCTracks()->size();
  attachSubEventInfo(simdataparts, mSubEventInfo);
  auto tracks = attachBranch<std::vector<o2::MCTrack>>("MCTrack", *mSimDataChannel, simdataparts);
  attachBranch<std::vector<o2::TrackReference>>("TrackRefs", *mSimDataChannel, simdataparts);
  attachBranch<o2::dataformats::MCTruthContainer<o2::TrackReference>>("IndexedTrackRefs", *mSimDataChannel, simdataparts);
  assert(tracks->size() == mSubEventInfo.npersistenttracks);

  for (auto det : listActiveDetectors) {
    if (dynamic_cast<o2::Base::Detector*>(det)) {
      ((o2::Base::Detector*)det)->attachHits(*mSimDataChannel, simdataparts);
    }
  }
  LOG(INFO) << "sending message with " << simdataparts.Size() << " parts";
  mSimDataChannel->Send(simdataparts);
}
}
}
