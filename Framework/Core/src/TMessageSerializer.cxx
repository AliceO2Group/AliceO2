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
#include <Framework/TMessageSerializer.h>
#include <FairMQTransportFactory.h>
#include <algorithm>
#include <memory>

using namespace o2::framework;

void* FairOutputTBuffer::embedInItself(fair::mq::Message& msg)
{
  // The first bytes of the message are used to store the pointer to the message itself
  // so that we can reallocate it if needed.
  if (sizeof(char*) > msg.GetSize()) {
    throw std::runtime_error("Message size too small to embed pointer");
  }
  char* data = reinterpret_cast<char*>(msg.GetData());
  char* ptr = reinterpret_cast<char*>(&msg);
  std::memcpy(data, &ptr, sizeof(char*));
  return data + sizeof(char*);
}

// Reallocation function. Get the message pointer from the data and call Rebuild.
char* FairOutputTBuffer::fairMQrealloc(char* oldData, size_t newSize, size_t oldSize)
{
  // Old data is the pointer at the beginning of the message, so the pointer
  // to the message is **stored** in the 8 bytes before it.
  auto* msg = *(fair::mq::Message**)(oldData - sizeof(char*));
  if (newSize <= msg->GetSize()) {
    // no need to reallocate, the message is already big enough
    return oldData;
  }
  // Create a shallow copy of the message
  fair::mq::MessagePtr oldMsg = msg->GetTransport()->CreateMessage();
  oldMsg->Copy(*msg);
  // Copy the old data while rebuilding. Reference counting should make
  // sure the old message is not deleted until the new one is ready.
  // We need 8 extra bytes for the pointer to the message itself (realloc does not know about it)
  // and we need to copy 8 bytes more than the old size (again, the extra pointer).
  msg->Rebuild(newSize + 8, fair::mq::Alignment{64});
  memcpy(msg->GetData(), oldMsg->GetData(), oldSize + 8);

  return reinterpret_cast<char*>(msg->GetData()) + sizeof(char*);
}
