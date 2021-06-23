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
#ifndef FRAMEWORK_FAIRMQDEVICEPROXY_H
#define FRAMEWORK_FAIRMQDEVICEPROXY_H

#include <memory>

class FairMQDevice;
class FairMQMessage;
class FairMQTransportFactory;

namespace o2
{
namespace framework
{
/// Helper class to hide FairMQDevice headers in the DataAllocator header.
/// This is done because FairMQDevice brings in a bunch of boost.mpl /
/// boost.fusion stuff, slowing down compilation times enourmously.
class FairMQDeviceProxy
{
 public:
  FairMQDeviceProxy(FairMQDevice* device)
    : mDevice{device}
  {
  }

  /// To be used in DataAllocator.cxx to avoid reimplenting any device
  /// API.
  FairMQDevice* getDevice()
  {
    return mDevice;
  }

  /// Looks like what we really need in the headers is just the transport.
  FairMQTransportFactory* getTransport();
  FairMQTransportFactory* getTransport(const std::string& channel, int index = 0);
  std::unique_ptr<FairMQMessage> createMessage() const;
  std::unique_ptr<FairMQMessage> createMessage(const size_t size) const;

 private:
  FairMQDevice* mDevice;
};

} // namespace framework
} // namespace o2

#endif // FRAMEWORK_FAIRMQDEVICEPROXY_H
