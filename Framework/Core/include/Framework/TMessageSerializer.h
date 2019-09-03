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
#include <MemoryResources/MemoryResources.h>

namespace o2
{
namespace framework
{
class FairTMessage;

// utilities to produce a span over a byte buffer held by various message types
// this is to avoid littering code with casts and conversions (span has a signed index type(!))
gsl::span<o2::byte> as_span(const FairTMessage& msg);
gsl::span<o2::byte> as_span(const FairMQMessage& msg);

class FairTMessage : public TMessage
{
 public:
  using TMessage::TMessage;
  FairTMessage() : TMessage(kMESS_OBJECT) {}
  FairTMessage(void* buf, Int_t len) : TMessage(buf, len) { ResetBit(kIsOwner); }
  FairTMessage(gsl::span<o2::byte> buf) : TMessage(buf.data(), buf.size()) { ResetBit(kIsOwner); }
  // helper function to clean up the object holding the data after it is transported.
  static void free(void* /*data*/, void* hint);
};

struct TMessageSerializer {
  using StreamerList = std::vector<TVirtualStreamerInfo*>;
  using CompressionLevel = int;
  enum class CacheStreamers { yes,
                              no };

  static void Serialize(FairMQMessage& msg, const TObject* input,
                        CacheStreamers streamers = CacheStreamers::no,
                        CompressionLevel compressionLevel = -1);

  template <typename T>
  static void Serialize(FairMQMessage& msg, const T* input, const TClass* cl, //
                        CacheStreamers streamers = CacheStreamers::no,        //
                        CompressionLevel compressionLevel = -1);

  template <typename T = TObject>
  static void Deserialize(const FairMQMessage& msg, std::unique_ptr<T>& output);

  static void serialize(FairTMessage& msg, const TObject* input,
                        CacheStreamers streamers = CacheStreamers::no,
                        CompressionLevel compressionLevel = -1);

  template <typename T>
  static void serialize(FairTMessage& msg, const T* input,                               //
                        const TClass* cl, CacheStreamers streamers = CacheStreamers::no, //
                        CompressionLevel compressionLevel = -1);

  template <typename T = TObject>
  static std::unique_ptr<T> deserialize(gsl::span<o2::byte> buffer);
  template <typename T = TObject>
  static inline std::unique_ptr<T> deserialize(byte* buffer, size_t size);

  // load the schema information from a message/buffer
  static void loadSchema(const FairMQMessage& msg);
  static void loadSchema(gsl::span<o2::byte> buffer);

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

inline void TMessageSerializer::serialize(FairTMessage& tm, const TObject* input,
                                          CacheStreamers streamers,
                                          CompressionLevel compressionLevel)
{
  return serialize(tm, input, nullptr, streamers, compressionLevel);
}

template <typename T>
inline void TMessageSerializer::serialize(FairTMessage& tm, const T* input,           //
                                          const TClass* cl, CacheStreamers streamers, //
                                          CompressionLevel compressionLevel)
{
  if (streamers == CacheStreamers::yes) {
    tm.EnableSchemaEvolution(true);
  }

  if (compressionLevel >= 0) {
    // if negative, skip to use ROOT default
    tm.SetCompressionLevel(compressionLevel);
  }

  // TODO: check what WriateObject and WriteObjectAny are doing
  if (cl == nullptr) {
    tm.WriteObject(input);
  } else {
    tm.WriteObjectAny(input, cl);
  }

  if (streamers == CacheStreamers::yes) {
    updateStreamers(tm, sStreamers);
  }
}

template <typename T>
inline std::unique_ptr<T> TMessageSerializer::deserialize(gsl::span<o2::byte> buffer)
{
  TClass* tgtClass = TClass::GetClass(typeid(T));
  if (tgtClass == nullptr) {
    std::string error("class is not ROOT-serializable: ");
    error += typeid(T).name();
    throw std::runtime_error(error);
  }
  // FIXME: we need to add consistency check for buffer data to be serialized
  // at the moment, TMessage might simply crash if an invalid or inconsistent
  // buffer is provided
  FairTMessage tm(buffer);
  TClass* serializedClass = tm.GetClass();
  if (serializedClass == nullptr) {
    std::string error("can not read class info from buffer");
    throw std::runtime_error(error);
  }
  if (tgtClass != serializedClass && serializedClass->GetBaseClass(tgtClass) == nullptr) {
    std::string error("can not convert serialized class ");
    error += tm.GetClass()->GetName();
    error += " into target class ";
    error += tgtClass->GetName();
    throw std::runtime_error(error);
  }
  return std::unique_ptr<T>(reinterpret_cast<T*>(tm.ReadObjectAny(serializedClass)));
}

template <typename T>
inline std::unique_ptr<T> TMessageSerializer::deserialize(byte* buffer, size_t size)
{
  return deserialize<T>(gsl::span<o2::byte>(buffer, gsl::narrow<gsl::span<o2::byte>::index_type>(size)));
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

  serialize(*tm, input, input->Class(), streamers, compressionLevel);

  msg.Rebuild(tm->Buffer(), tm->BufferSize(), FairTMessage::free, tm.get());
  tm.release();
}

template <typename T>
inline void TMessageSerializer::Serialize(FairMQMessage& msg, const T* input,           //
                                          const TClass* cl,                             //
                                          TMessageSerializer::CacheStreamers streamers, //
                                          TMessageSerializer::CompressionLevel compressionLevel)
{
  std::unique_ptr<FairTMessage> tm = std::make_unique<FairTMessage>(kMESS_OBJECT);

  serialize(*tm, input, cl, streamers, compressionLevel);

  msg.Rebuild(tm->Buffer(), tm->BufferSize(), FairTMessage::free, tm.get());
  tm.release();
}

template <typename T>
inline void TMessageSerializer::Deserialize(const FairMQMessage& msg, std::unique_ptr<T>& output)
{
  // we know the message will not be modified by this,
  // so const_cast should be OK here(IMHO).
  output = deserialize(as_span(msg));
}

inline TMessageSerializer::StreamerList TMessageSerializer::getStreamers()
{
  std::lock_guard<std::mutex> lock{TMessageSerializer::sStreamersLock};
  return sStreamers;
}

// gsl::narrow is used to do a runtime narrowing check, this might be a bit paranoid,
// we would probably be fine with e.g. gsl::narrow_cast (or just a static_cast)
inline gsl::span<o2::byte> as_span(const FairMQMessage& msg)
{
  return gsl::span<o2::byte>{static_cast<o2::byte*>(msg.GetData()), gsl::narrow<gsl::span<o2::byte>::index_type>(msg.GetSize())};
}

inline gsl::span<o2::byte> as_span(const FairTMessage& msg)
{
  return gsl::span<o2::byte>{reinterpret_cast<o2::byte*>(msg.Buffer()),
                             gsl::narrow<gsl::span<o2::byte>::index_type>(msg.BufferSize())};
}

} // namespace framework
} // namespace o2
#endif // FRAMEWORK_TMESSAGESERIALIZER_H
