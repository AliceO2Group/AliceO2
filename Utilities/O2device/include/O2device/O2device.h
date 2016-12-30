/// @copyright
/// © Copyright 2014 Copyright Holders of the ALICE O2 collaboration.
/// See https://aliceinfo.cern.ch/AliceO2 for details on the Copyright holders.
/// This software is distributed under the terms of the
/// GNU General Public License version 3 (GPL Version 3).
///
/// License text in a separate file.
///
/// In applying this license, CERN does not waive the privileges and immunities
/// granted to it by virtue of its status as an Intergovernmental Organization
/// or submit itself to any jurisdiction.

/// @headerfile O2device.h
///
/// @since 2014-12-10
/// @author M. Krzewicki <mkrzewic@cern.ch>

#ifndef O2DEVICE_H_
#define O2DEVICE_H_

#include "FairMQDevice.h"
#include "FairMQLogger.h"
#include "Headers/DataHeader.h"
#include <stdexcept>
#include "O2device/SharedMemory.h"

namespace AliceO2 {
namespace Base {

/// just a typedef to express the fact that it is not just a FairMQParts vector,
/// it has to follow the O2 convention of header-payload-header-payload
using O2message = FairMQParts;

// class O2message : public FairMQParts {
// public:
//  using FairMQParts::FairMQParts;
//  void setHasShared(bool has) {mHasShared = has;}
//  virtual ~O2message() {
//    //some logic to clean up shared memory on send failure
//    //this would best be handled in FairMQMessage,
//    //probably a second callback?
//  }
// private:
//  bool mHasShared;
//};

class O2device : public FairMQDevice {
public:
  // way to select how many consumers we expect for the message in shared memory:
  // None: message is not in shared memory, went over some-mq directly.
  // One: expect exactly one consumer
  // Many: expect many consumers (e.g. PUB/SUB)
  enum class SharedMemoryConsumers {
    None,
    One,
    Many
  };

  using FairMQDevice::FairMQDevice;
  O2device() : mShmManager{}, mSharedMemorySize{ 2000000000 }
  {
  }
  virtual ~O2device()
  {
  }

  void setSharedMemorySize(size_t siz)
  {
    mSharedMemorySize = siz;
  }

  /// Add a message part to a message intendended for a local channel;
  /// this forces tha creation of shared memory metadata.
  /// Payload is empty so the user has to make sure this is only used when
  /// shared mem makes sense.
  /// TODO: automate this, the user should not have to think about this
  bool AddMessage(O2message& parts, AliceO2::Header::Block headerBlockOrig, AliceO2::SharedMemory::BlockPtr& shmBlock,
                  SharedMemoryConsumers consumers = SharedMemoryConsumers::One)
  {
    using namespace AliceO2;

    // if we target meny consumers, register with garbage collector
    if (consumers == SharedMemoryConsumers::Many) {
      mShmManager->registerBlock(shmBlock);
    }

    // only add the shmem information if we actually send via shared memory
    if (consumers != SharedMemoryConsumers::None) {
      auto handle = shmBlock->getHandle(*mShmManager);
      Header::Block headerBlock{ std::move(headerBlockOrig), Header::BoostShmHeader{ handle, shmBlock->getID() } };

      parts.AddPart(
        NewMessage(headerBlock.buffer.get(), headerBlock.bufferSize, &Header::Block::freefn, headerBlock.buffer.get()));
      headerBlock.buffer.release();
      parts.AddPart(NewMessage());
    } else {
      // here we just sent the data "normally" over the network and need to decrease the use count
      // in the callback
      parts.AddPart(NewMessage(headerBlockOrig.buffer.get(), headerBlockOrig.bufferSize, &Header::Block::freefn,
                               headerBlockOrig.buffer.get()));
      headerBlockOrig.buffer.release();
      parts.AddPart(NewMessage(shmBlock->data(), shmBlock->size(),
                               [](void*, void* ptr) { intrusive_ptr_release(static_cast<SharedMemory::Block*>(ptr)); },
                               &*shmBlock));
    }
    return true;
  }

  void CloseSharedMemoryMessage(O2message&& tmpmessage)
  {
    O2message parts{ std::move(tmpmessage) };
    for (auto it = parts.fParts.begin(); it != parts.fParts.end(); ++it) {
      byte* headerBuffer = nullptr;
      size_t headerBufferSize = 0;
      if (*it != nullptr) {
        headerBuffer = reinterpret_cast<byte*>((*it)->GetData());
        headerBufferSize = (*it)->GetSize();
        // handle possible use of shared memory
        using namespace AliceO2::Header;
        const BoostShmHeader* shmheader = get<BoostShmHeader>(headerBuffer, headerBufferSize);
        if (shmheader) {
          if (!mShmManager) {
            mShmManager.reset(new AliceO2::SharedMemory::Manager(mSharedMemorySize));
          }
          AliceO2::SharedMemory::BlockPtr shmBlock{ mShmManager->getBlock(shmheader->handle, shmheader->id) };
        }
        ++it;

        byte* dataBuffer = nullptr;
        size_t dataBufferSize = 0;
        if (*it != nullptr) {
          // do nothing here, payloads are either empty
          // or will be cleaned by FairMQMessage dtor
        }
      }
    }
  }

  /// Here is how to add an annotated data part (with header);
  /// @param[in,out] parts is a reference to the message;
  /// @param[] incomingBlock header block must be MOVED in (movable only)
  /// @param[] dataMessage the data message must be MOVED in (unique_ptr by value)
  bool AddMessage(O2message& parts, AliceO2::Header::Block headerBlock, FairMQMessagePtr dataMessage)
  {
    FairMQMessagePtr headerMessage = NewMessage(headerBlock.buffer.get(), headerBlock.bufferSize,
                                                &AliceO2::Header::Block::freefn, headerBlock.buffer.get());
    headerBlock.buffer.release();

    parts.AddPart(std::move(headerMessage));
    parts.AddPart(std::move(dataMessage));
    return true;
  }

  /// The user needs to define a member function with correct signature
  /// currently this is old school: buf,len pairs;
  /// In the end I'd like to move to array_view
  /// when this becomes available (either with C++17 or via GSL)
  template <typename T>
  bool ForEach(O2message& parts, bool (T::*memberFunction)(const byte* headerBuffer, size_t headerBufferSize,
                                                           const byte* dataBuffer, size_t dataBufferSize))
  {
    if ((parts.Size() % 2) != 0)
      throw std::invalid_argument("number of parts in message not even (n%2 != 0)");

    for (auto it = parts.fParts.begin(); it != parts.fParts.end(); ++it) {

      byte* headerBuffer = nullptr;
      size_t headerBufferSize = 0;
      if (*it != nullptr) {
        headerBuffer = reinterpret_cast<byte*>((*it)->GetData());
        headerBufferSize = (*it)->GetSize();
      }
      ++it;

      byte* dataBuffer = nullptr;
      size_t dataBufferSize = 0;

      if (*it != nullptr) {
        // handle possible use of shared memory
        using namespace AliceO2::Header;
        const BoostShmHeader* shmheader = get<BoostShmHeader>(headerBuffer, headerBufferSize);
        if (shmheader) {
          // here we expect the data to be in shmem
          if (!mShmManager) {
            mShmManager.reset(new AliceO2::SharedMemory::Manager(mSharedMemorySize));
          }

          auto shmBlock = mShmManager->getBlock(shmheader->handle, shmheader->id);
          if (!shmBlock) {
            LOG(WARN) << "id mismatch or invalid block";
            continue;
          }

          dataBuffer = shmBlock->data();
          dataBufferSize = shmBlock->size();
          (static_cast<T*>(this)->*memberFunction)(headerBuffer, headerBufferSize, dataBuffer, dataBufferSize);
        } else {
          // here we expect to have the data in the fairmq message directly
          dataBuffer = reinterpret_cast<byte*>((*it)->GetData());
          dataBufferSize = (*it)->GetSize();
          (static_cast<T*>(this)->*memberFunction)(headerBuffer, headerBufferSize, dataBuffer, dataBufferSize);
        }
      }
    }
    return true;
  }

protected:
  std::unique_ptr<AliceO2::SharedMemory::Manager> mShmManager;
  size_t mSharedMemorySize;

private:
};
}
}
#endif /* O2DEVICE_H_ */
