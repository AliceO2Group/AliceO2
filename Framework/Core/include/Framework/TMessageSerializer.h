// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_TMESSAGESERIALIZER_H
#define FRAMEWORK_TMESSAGESERIALIZER_H

#include <fairmq/FairMQMessage.h>

#include <TList.h>
#include <TMessage.h>
#include <TObjArray.h>
#include <TStreamerInfo.h>
#include <gsl/gsl_util>
#include <gsl/span>
#include <memory>
#include <mutex>

namespace o2
{
namespace framework
{
// TODO: this should come from a global definitions header (or do we use gsl::byte?)
using byte = unsigned char;
class FairTMessage;

// utilities to produce a span over a byte buffer held by various message types
// this is to avoid littering code with casts and conversions (span has a signed index type(!))
gsl::span<byte> as_span(const FairTMessage& msg);
gsl::span<byte> as_span(const FairMQMessage& msg);

class FairTMessage : public TMessage
{
 public:
  using TMessage::TMessage;
  FairTMessage() : TMessage(kMESS_OBJECT) {}
  FairTMessage(void* buf, Int_t len) : TMessage(buf, len) { ResetBit(kIsOwner); }
  FairTMessage(gsl::span<byte> buf) : TMessage(buf.data(), buf.size()) { ResetBit(kIsOwner); }
  // helper function to clean up the object holding the data after it is transported.
  static void free(void* /*data*/, void* hint);
};

struct TMessageSerializer {
  using StreamerList = std::vector<TVirtualStreamerInfo*>;
  using CompressionLevel = int;
  enum class CacheStreamers { yes, no };

  static void Serialize(FairMQMessage& msg, const TObject* input, CacheStreamers streamers = CacheStreamers::no,
                 CompressionLevel compressionLevel = -1);
  static void Deserialize(const FairMQMessage& msg, std::unique_ptr<TObject>& output);

  static void serialize(FairTMessage& msg, const TObject* input, CacheStreamers streamers = CacheStreamers::no,
                 CompressionLevel compressionLevel = -1);
  static std::unique_ptr<TObject> deserialize(gsl::span<byte> buffer);

  // load the schema information from a message/buffer
  static void loadSchema(const FairMQMessage& msg);
  static void loadSchema(gsl::span<byte> buffer);

  // write the schema into an empty message/buffer
  static void fillSchema(FairMQMessage& msg, const StreamerList& streamers);
  static void fillSchema(FairTMessage& msg, const StreamerList& streamers);

  // get the streamers
  static StreamerList getStreamers();

  // update the streamer list with infos appropriate for this type
  static void updateStreamers(const TObject* object);

 private:
  // update the cache of streamer infos for serialized classes
  static void updateStreamers(const FairTMessage& message, StreamerList& streamers);

  // for now this is a static, maybe it would be better to move the storage somewhere else?
  static StreamerList sStreamers;
  static std::mutex sStreamersLock;
};

inline void TMessageSerializer::serialize(FairTMessage& tm, const TObject* input, CacheStreamers streamers,
                                          CompressionLevel compressionLevel)
{
  if (streamers == CacheStreamers::yes) {
    tm.EnableSchemaEvolution(true);
  }

  if (compressionLevel >= 0) {
    // if negative, skip to use ROOT default
    tm.SetCompressionLevel(compressionLevel);
  }

  tm.WriteObject(input);

  if (streamers == CacheStreamers::yes) {
    updateStreamers(tm, sStreamers);
  }
}

inline std::unique_ptr<TObject> TMessageSerializer::deserialize(gsl::span<byte> buffer)
{
  FairTMessage tm(buffer);
  return std::unique_ptr<TObject>(reinterpret_cast<TObject*>(tm.ReadObject(tm.GetClass())));
}

inline void FairTMessage::free(void* /*data*/, void* hint)
{
  std::default_delete<FairTMessage> deleter;
  deleter(static_cast<FairTMessage*>(hint));
}

inline void TMessageSerializer::Serialize(FairMQMessage& msg, const TObject* input,
                                          TMessageSerializer::CacheStreamers streamers,
                                          TMessageSerializer::CompressionLevel compressionLevel)
{
  std::unique_ptr<FairTMessage> tm = std::make_unique<FairTMessage>(kMESS_OBJECT);

  serialize(*tm, input, streamers, compressionLevel);

  msg.Rebuild(tm->Buffer(), tm->BufferSize(), FairTMessage::free, tm.get());
  tm.release();
}

inline void TMessageSerializer::Deserialize(const FairMQMessage& msg, std::unique_ptr<TObject>& output)
{
  // we know the message will not be modified by this,
  // so const_cast should be OK here(IMHO).
  output = deserialize(as_span(msg));
}

inline TMessageSerializer::StreamerList TMessageSerializer::getStreamers()
{
  std::lock_guard<std::mutex> lock{ TMessageSerializer::sStreamersLock };
  return sStreamers;
}

// gsl::narrow is used to do a runtime narrowing check, this might be a bit paranoid,
// we would probably be fine with e.g. gsl::narrow_cast (or just a static_cast)
inline gsl::span<byte> as_span(const FairMQMessage& msg)
{
  return gsl::span<byte>{ static_cast<byte*>(msg.GetData()), gsl::narrow<gsl::span<byte>::index_type>(msg.GetSize()) };
}

inline gsl::span<byte> as_span(const FairTMessage& msg)
{
  return gsl::span<byte>{ reinterpret_cast<byte*>(msg.Buffer()),
                          gsl::narrow<gsl::span<byte>::index_type>(msg.BufferSize()) };
}

} // namespace framework
} // namespace o2
#endif // FRAMEWORK_TMESSAGESERIALIZER_H
