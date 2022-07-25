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

#include <Steer/O2MCApplication.h>
#include <fairmq/Channel.h>
#include <fairmq/Message.h>
#include <fairmq/Device.h>
#include <fairmq/Parts.h>
#include <SimulationDataFormat/PrimaryChunk.h>
#include <TMessage.h>
#include <sstream>
#include <SimConfig/SimConfig.h>
#include <DetectorsBase/Detector.h>
#include "DetectorsBase/Aligner.h"
#include "DetectorsBase/MaterialManager.h"
#include <CommonUtils/ShmManager.h>
#include <cassert>
#include <SimulationDataFormat/MCEventHeader.h>
#include <TGeoManager.h>
#include <fstream>
#include <FairVolume.h>
#include <CommonUtils/NameConf.h>
#include "SimConfig/SimUserDecay.h"

namespace o2
{
namespace steer
{
// helper function to send trivial data
template <typename T>
void TypedVectorAttach(const char* name, fair::mq::Channel& channel, fair::mq::Parts& parts)
{
  static auto mgr = FairRootManager::Instance();
  auto vector = mgr->InitObjectAs<const std::vector<T>*>(name);
  if (vector) {
    auto buffer = (char*)&(*vector)[0];
    auto buffersize = vector->size() * sizeof(T);
    fair::mq::MessagePtr message(channel.NewMessage(
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

    // Note that this is done in addition to the generic
    // R + Z-cut mechanism at VMC level.

    float x, y, z;
    fMC->TrackPosition(x, y, z);

    // this function is implementing a basic z-dependent R cut
    // can be generalized later on
    auto outOfR = [x, y, this](float z) {
      // for the moment for cases when we have ZDC enabled
      if (std::abs(z) > mCutParams.tunnelZ) {
        if ((x * x + y * y) > mCutParams.maxRTrackingZDC * mCutParams.maxRTrackingZDC) {
          return true;
        }
      }
      return false;
    };

    if (z > mCutParams.ZmaxA ||
        -z > mCutParams.ZmaxC ||
        outOfR(z)) {
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
  o2::detectors::DetID::mask_t dmask{};
  for (int i = 0; i < fModules->GetEntries(); ++i) {
    auto mod = static_cast<FairModule*>(fModules->At(i));
    if (mod) {
      mModIdToName[mod->GetModId()] = mod->GetName();
      int did = o2::detectors::DetID::nameToID(mod->GetName());
      if (did >= 0) {
        dmask |= o2::detectors::DetID::getMask(did);
      }
    }
  }
  gGeoManager->SetUniqueID(dmask.to_ulong());
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
  // load special cuts which might be given from the outside first.
  auto& matMgr = o2::base::MaterialManager::Instance();
  matMgr.loadCutsAndProcessesFromJSON(o2::base::MaterialManager::ESpecial::kTRUE);
  // During the following, FairModule::SetSpecialPhysicsCuts will be called for each module
  FairMCApplication::InitGeometry();
  matMgr.writeCutsAndProcessesToJSON();
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

bool O2MCApplicationBase::MisalignGeometry()
{
  for (auto det : listActiveDetectors) {
    if (dynamic_cast<o2::base::Detector*>(det)) {
      ((o2::base::Detector*)det)->addAlignableVolumes();
    }
  }

  // we stream out both unaligned geometry (to allow for
  // dynamic post-alignment) as well as the aligned version
  // which can be used by digitization etc. immediately
  auto& confref = o2::conf::SimConfig::Instance();
  auto geomfile = o2::base::NameConf::getGeomFileName(confref.getOutPrefix());
  // since in general the geometry is a CCDB object, it must be exported under the standard name
  gGeoManager->SetName(std::string(o2::base::NameConf::CCDBOBJECT).c_str());
  gGeoManager->Export(geomfile.c_str());

  // apply alignment for included detectors AFTER exporting ideal geometry
  auto& aligner = o2::base::Aligner::Instance();
  aligner.applyAlignment(confref.getTimestamp());

  // export aligned geometry into different file
  auto alignedgeomfile = o2::base::NameConf::getAlignedGeomFileName(confref.getOutPrefix());
  gGeoManager->Export(alignedgeomfile.c_str());

  // return original return value of misalignment procedure
  return true;
}

void O2MCApplicationBase::finishEventCommon()
{
  LOG(info) << "This event/chunk did " << mStepCounter << " steps";

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
    LOG(info) << "Discarding current event due to no hits";
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

void O2MCApplicationBase::AddParticles()
{
  // dispatch first to function in FairRoot
  FairMCApplication::AddParticles();

  auto& param = o2::conf::SimUserDecay::Instance();
  LOG(info) << "Printing \'SimUserDecay\' parameters";
  LOG(info) << param;

  // check if there are PDG codes requested for user decay
  if (param.pdglist.empty()) {
    return;
  }

  // loop over PDG codes in the string
  std::stringstream ss(param.pdglist);
  int pdg;
  while (ss >> pdg) {
    LOG(info) << "Setting user decay for PDG " << pdg;
    TVirtualMC::GetMC()->SetUserDecay(pdg);
  }
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

void O2MCApplication::attachSubEventInfo(fair::mq::Parts& parts, o2::data::SubEventInfo const& info) const
{
  // parts.AddPart(std::move(mSimDataChannel->NewSimpleMessage(info)));
  o2::base::attachTMessage(info, *mSimDataChannel, parts);
}

// helper function to fetch data from FairRootManager branch and serialize it
// returns handle to container
template <typename T>
const T* attachBranch(std::string const& name, fair::mq::Channel& channel, fair::mq::Parts& parts)
{
  auto mgr = FairRootManager::Instance();
  // check if branch is present
  if (mgr->GetBranchId(name) == -1) {
    LOG(error) << "Branch " << name << " not found";
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
  fair::mq::Parts simdataparts;

  // fill these parts ... the receiver has to unpack similary
  // TODO: actually we could just loop over branches in FairRootManager at this moment?
  mSubEventInfo->npersistenttracks = static_cast<o2::data::Stack*>(GetStack())->getMCTracks()->size();
  mSubEventInfo->nprimarytracks = static_cast<o2::data::Stack*>(GetStack())->GetNprimary();
  attachSubEventInfo(simdataparts, *mSubEventInfo);
  auto tracks = attachBranch<std::vector<o2::MCTrack>>("MCTrack", *mSimDataChannel, simdataparts);
  attachBranch<std::vector<o2::TrackReference>>("TrackRefs", *mSimDataChannel, simdataparts);
  assert(tracks->size() == mSubEventInfo->npersistenttracks);

  for (auto det : listActiveDetectors) {
    if (dynamic_cast<o2::base::Detector*>(det)) {
      ((o2::base::Detector*)det)->attachHits(*mSimDataChannel, simdataparts);
    }
  }
  LOG(info) << "sending message with " << simdataparts.Size() << " parts";
  mSimDataChannel->Send(simdataparts);
}
} // namespace steer
} // namespace o2
