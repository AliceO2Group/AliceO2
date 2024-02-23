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
#ifndef FRAMEWORK_TMESSAGESERIALIZER_H
#define FRAMEWORK_TMESSAGESERIALIZER_H

#include <fairmq/Message.h>

#include "Framework/RuntimeError.h"

#include <TList.h>
#include <TMessage.h>
#include <TObjArray.h>
#include <TStreamerInfo.h>
#include <gsl/util>
#include <gsl/span>
#include <gsl/narrow>
#include <memory>
#include <mutex>
#include <cstddef>

namespace o2::framework
{
class FairTMessage;

// utilities to produce a span over a byte buffer held by various message types
// this is to avoid littering code with casts and conversions (span has a signed index type(!))
gsl::span<std::byte> as_span(const FairTMessage& msg);
gsl::span<std::byte> as_span(const fair::mq::Message& msg);

class FairTMessage : public TMessage
{
 public:
  using TMessage::TMessage;
  FairTMessage() : TMessage(kMESS_OBJECT) {}
  FairTMessage(void* buf, Int_t len) : TMessage(buf, len) { ResetBit(kIsOwner); }
  FairTMessage(gsl::span<std::byte> buf) : TMessage(buf.data(), buf.size()) { ResetBit(kIsOwner); }
  // helper function to clean up the object holding the data after it is transported.
  static void free(void* /*data*/, void* hint);
};

struct TMessageSerializer {
  using StreamerList = std::vector<TVirtualStreamerInfo*>;
  using CompressionLevel = int;

  static void Serialize(fair::mq::Message& msg, const TObject* input,
                        CompressionLevel compressionLevel = -1);

  template <typename T>
  static void Serialize(fair::mq::Message& msg, const T* input, const TClass* cl, //
                        CompressionLevel compressionLevel = -1);

  template <typename T = TObject>
  static void Deserialize(const fair::mq::Message& msg, std::unique_ptr<T>& output);

  static void serialize(FairTMessage& msg, const TObject* input,
                        CompressionLevel compressionLevel = -1);

  template <typename T>
  static void serialize(FairTMessage& msg, const T* input, //
                        const TClass* cl,
                        CompressionLevel compressionLevel = -1);

  template <typename T = TObject>
  static std::unique_ptr<T> deserialize(gsl::span<std::byte> buffer);
  template <typename T = TObject>
  static inline std::unique_ptr<T> deserialize(std::byte* buffer, size_t size);
};

inline void TMessageSerializer::serialize(FairTMessage& tm, const TObject* input,
                                          CompressionLevel compressionLevel)
{
  return serialize(tm, input, nullptr, compressionLevel);
}

template <typename T>
inline void TMessageSerializer::serialize(FairTMessage& tm, const T* input, //
                                          const TClass* cl, CompressionLevel compressionLevel)
{
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
}

template <typename T>
inline std::unique_ptr<T> TMessageSerializer::deserialize(gsl::span<std::byte> buffer)
{
  TClass* tgtClass = TClass::GetClass(typeid(T));
  if (tgtClass == nullptr) {
    throw runtime_error_f("class is not ROOT-serializable: %s", typeid(T).name());
  }
  // FIXME: we need to add consistency check for buffer data to be serialized
  // at the moment, TMessage might simply crash if an invalid or inconsistent
  // buffer is provided
  FairTMessage tm(buffer);
  TClass* serializedClass = tm.GetClass();
  if (serializedClass == nullptr) {
    throw runtime_error_f("can not read class info from buffer");
  }
  if (tgtClass != serializedClass && serializedClass->GetBaseClass(tgtClass) == nullptr) {
    throw runtime_error_f("can not convert serialized class %s into target class %s",
                          tm.GetClass()->GetName(),
                          tgtClass->GetName());
  }
  return std::unique_ptr<T>(reinterpret_cast<T*>(tm.ReadObjectAny(serializedClass)));
}

template <typename T>
inline std::unique_ptr<T> TMessageSerializer::deserialize(std::byte* buffer, size_t size)
{
  return deserialize<T>(gsl::span<std::byte>(buffer, gsl::narrow<gsl::span<std::byte>::size_type>(size)));
}

inline void FairTMessage::free(void* /*data*/, void* hint)
{
  std::default_delete<FairTMessage> deleter;
  deleter(static_cast<FairTMessage*>(hint));
}

inline void TMessageSerializer::Serialize(fair::mq::Message& msg, const TObject* input,
                                          TMessageSerializer::CompressionLevel compressionLevel)
{
  std::unique_ptr<FairTMessage> tm = std::make_unique<FairTMessage>(kMESS_OBJECT);

  serialize(*tm, input, input->Class(), compressionLevel);

  msg.Rebuild(tm->Buffer(), tm->BufferSize(), FairTMessage::free, tm.get());
  tm.release();
}

template <typename T>
inline void TMessageSerializer::Serialize(fair::mq::Message& msg, const T* input, //
                                          const TClass* cl,                       //
                                          TMessageSerializer::CompressionLevel compressionLevel)
{
  std::unique_ptr<FairTMessage> tm = std::make_unique<FairTMessage>(kMESS_OBJECT);

  serialize(*tm, input, cl, compressionLevel);

  msg.Rebuild(tm->Buffer(), tm->BufferSize(), FairTMessage::free, tm.get());
  tm.release();
}

template <typename T>
inline void TMessageSerializer::Deserialize(const fair::mq::Message& msg, std::unique_ptr<T>& output)
{
  // we know the message will not be modified by this,
  // so const_cast should be OK here(IMHO).
  output = deserialize(as_span(msg));
}

// gsl::narrow is used to do a runtime narrowing check, this might be a bit paranoid,
// we would probably be fine with e.g. gsl::narrow_cast (or just a static_cast)
inline gsl::span<std::byte> as_span(const fair::mq::Message& msg)
{
  return gsl::span<std::byte>{static_cast<std::byte*>(msg.GetData()), gsl::narrow<gsl::span<std::byte>::size_type>(msg.GetSize())};
}

inline gsl::span<std::byte> as_span(const FairTMessage& msg)
{
  return gsl::span<std::byte>{reinterpret_cast<std::byte*>(msg.Buffer()),
                              gsl::narrow<gsl::span<std::byte>::size_type>(msg.BufferSize())};
}

} // namespace o2::framework
#endif // FRAMEWORK_TMESSAGESERIALIZER_H
