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
#include <TBufferFile.h>
#include <TObjArray.h>
#include <memory>
#include <mutex>
#include <cstddef>

namespace o2::framework
{
class FairOutputTBuffer;
class FairInputTBuffer;

// A TBufferFile which we can use to serialise data to a FairMQ message.
class FairOutputTBuffer : public TBufferFile
{
 public:
  // This is to serialise data to FairMQ. We embed the pointer to the message
  // in the data itself, so that we can use it to reallocate the message if needed.
  // The FairMQ message retains ownership of the data.
  // When deserialising the root object, keep in mind one needs to skip the 8 bytes
  // for the pointer.
  FairOutputTBuffer(fair::mq::Message& msg)
    : TBufferFile(TBuffer::kWrite, msg.GetSize() - sizeof(char*), embedInItself(msg), false, fairMQrealloc)
  {
  }
  // Helper function to keep track of the FairMQ message that holds the data
  // in the data itself. We can use this to make sure the message can be reallocated
  // even if we simply have a pointer to the data. Hopefully ROOT will not play dirty
  // with us.
  void* embedInItself(fair::mq::Message& msg);
  // helper function to clean up the object holding the data after it is transported.
  static char* fairMQrealloc(char* oldData, size_t newSize, size_t oldSize);
};

class FairInputTBuffer : public TBufferFile
{
 public:
  // This is to serialise data to FairMQ. The provided message is expeted to have 8 bytes
  // of overhead, where the source embedded the pointer for the reallocation.
  // Notice this will break if the sender and receiver are not using the same
  // size for a pointer.
  FairInputTBuffer(char* data, size_t size)
    : TBufferFile(TBuffer::kRead, size - sizeof(char*), data + sizeof(char*), false, nullptr)
  {
  }
};

struct TMessageSerializer {
  static void Serialize(fair::mq::Message& msg, const TObject* input);

  template <typename T>
  static void Serialize(fair::mq::Message& msg, const T* input, const TClass* cl);

  template <typename T = TObject>
  static void Deserialize(const fair::mq::Message& msg, std::unique_ptr<T>& output);

  static void serialize(o2::framework::FairOutputTBuffer& msg, const TObject* input);

  template <typename T>
  static void serialize(o2::framework::FairOutputTBuffer& msg, const T* input, const TClass* cl);

  template <typename T = TObject>
  static inline std::unique_ptr<T> deserialize(FairInputTBuffer& buffer);
};

inline void TMessageSerializer::serialize(FairOutputTBuffer& tm, const TObject* input)
{
  return serialize(tm, input, nullptr);
}

template <typename T>
inline void TMessageSerializer::serialize(FairOutputTBuffer& tm, const T* input, const TClass* cl)
{
  // TODO: check what WriateObject and WriteObjectAny are doing
  if (cl == nullptr) {
    tm.WriteObject(input);
  } else {
    tm.WriteObjectAny(input, cl);
  }
}

template <typename T>
inline std::unique_ptr<T> TMessageSerializer::deserialize(FairInputTBuffer& buffer)
{
  TClass* tgtClass = TClass::GetClass(typeid(T));
  if (tgtClass == nullptr) {
    throw runtime_error_f("class is not ROOT-serializable: %s", typeid(T).name());
  }
  // FIXME: we need to add consistency check for buffer data to be serialized
  // at the moment, TMessage might simply crash if an invalid or inconsistent
  // buffer is provided
  buffer.SetBufferOffset(0);
  buffer.InitMap();
  TClass* serializedClass = buffer.ReadClass();
  buffer.SetBufferOffset(0);
  buffer.ResetMap();
  if (serializedClass == nullptr) {
    throw runtime_error_f("can not read class info from buffer");
  }
  if (tgtClass != serializedClass && serializedClass->GetBaseClass(tgtClass) == nullptr) {
    throw runtime_error_f("can not convert serialized class %s into target class %s",
                          serializedClass->GetName(),
                          tgtClass->GetName());
  }
  return std::unique_ptr<T>(reinterpret_cast<T*>(buffer.ReadObjectAny(serializedClass)));
}

inline void TMessageSerializer::Serialize(fair::mq::Message& msg, const TObject* input)
{
  FairOutputTBuffer output(msg);
  serialize(output, input, input->Class());
}

template <typename T>
inline void TMessageSerializer::Serialize(fair::mq::Message& msg, const T* input, const TClass* cl)
{
  FairOutputTBuffer output(msg);
  serialize(output, input, cl);
}

template <typename T>
inline void TMessageSerializer::Deserialize(const fair::mq::Message& msg, std::unique_ptr<T>& output)
{
  // we know the message will not be modified by this,
  // so const_cast should be OK here(IMHO).
  FairInputTBuffer input(static_cast<char*>(msg.GetData()), static_cast<int>(msg.GetSize()));
  output = deserialize(input);
}

} // namespace o2::framework
#endif // FRAMEWORK_TMESSAGESERIALIZER_H
