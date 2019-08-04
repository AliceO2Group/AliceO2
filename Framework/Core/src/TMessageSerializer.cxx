// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include <Framework/TMessageSerializer.h>
#include <algorithm>
#include <memory>

using namespace o2::framework;

TMessageSerializer::StreamerList TMessageSerializer::sStreamers{};
std::mutex TMessageSerializer::sStreamersLock{};

void TMessageSerializer::loadSchema(gsl::span<o2::byte> buffer)
{
  std::unique_ptr<TObject> obj = deserialize(buffer);

  TObjArray* pSchemas = dynamic_cast<TObjArray*>(obj.get());
  if (!pSchemas) {
    return;
  }

  // TODO: this is a bit of a problem in general: non-owning ROOT containers should become
  // owners at deserialize, otherwise there is a leak. Switch to a better container.
  pSchemas->SetOwner(kTRUE);

  for (int i = 0; i < pSchemas->GetEntriesFast(); i++) {
    TStreamerInfo* pSchema = dynamic_cast<TStreamerInfo*>(pSchemas->At(i));
    if (!pSchema) {
      continue;
    }
    int version = pSchema->GetClassVersion();
    TClass* pClass = TClass::GetClass(pSchema->GetName());
    if (!pClass) {
      continue;
    }
    if (pClass->GetClassVersion() == version) {
      continue;
    }
    TObjArray* pInfos = const_cast<TObjArray*>(pClass->GetStreamerInfos());
    if (!pInfos) {
      continue;
    }
    TVirtualStreamerInfo* pInfo = dynamic_cast<TVirtualStreamerInfo*>(pInfos->At(version));
    if (pInfo) {
      continue;
    }
    pSchema->SetClass(pClass);
    pSchema->BuildOld();
    pInfos->AddAtAndExpand(pSchema, version);
    pSchemas->Remove(pSchema);
  }
}

void TMessageSerializer::fillSchema(FairTMessage& msg, const StreamerList& streamers)
{
  // TODO: this is a bit of a problem in general: non-owning ROOT containers should become
  // owners at deserialize, otherwise there is a leak. Switch to a better container.
  TObjArray infoArray{};
  for (const auto& info : streamers) {
    infoArray.Add(info);
  }
  serialize(msg, &infoArray);
}

void TMessageSerializer::loadSchema(const FairMQMessage& msg) { loadSchema(as_span(msg)); }
void TMessageSerializer::fillSchema(FairMQMessage& msg, const StreamerList& streamers)
{
  // TODO: this is a bit of a problem in general: non-owning ROOT containers should become
  // owners at deserialize, otherwise there is a leak. Switch to a better container.
  TObjArray infoArray{};
  for (const auto& info : streamers) {
    infoArray.Add(info);
  }
  Serialize(msg, &infoArray);
}

void TMessageSerializer::updateStreamers(const FairTMessage& message, StreamerList& streamers)
{
  std::lock_guard<std::mutex> lock{TMessageSerializer::sStreamersLock};

  TIter nextStreamer(message.GetStreamerInfos()); // unfortunately ROOT uses TList* here
  // this looks like we could use std::map here.
  while (TVirtualStreamerInfo* in = static_cast<TVirtualStreamerInfo*>(nextStreamer())) {
    auto found = std::find_if(streamers.begin(), streamers.end(), [&](const auto& old) {
      return (old->GetName() == in->GetName() && old->GetClassVersion() == in->GetClassVersion());
    });
    if (found == streamers.end()) {
      streamers.push_back(in);
    }
  }
}

void TMessageSerializer::updateStreamers(const TObject* object)
{
  FairTMessage msg(kMESS_OBJECT);
  serialize(msg, object, CacheStreamers::yes, CompressionLevel{0});
}
