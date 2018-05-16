// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

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

/// @headerfile O2Device.h
///
/// @since 2014-12-10
/// @author M. Krzewicki <mkrzewic@cern.ch>

#ifndef O2DEVICE_H_
#define O2DEVICE_H_

#include <FairMQDevice.h>
#include <options/FairMQProgOptions.h>
#include "Headers/DataHeader.h"
#include "Monitoring/MonitoringFactory.h"
#include <stdexcept>
#include <gsl/gsl>

namespace o2
{
namespace Base
{

/// just a typedef to express the fact that it is not just a FairMQParts vector,
/// it has to follow the O2 convention of header-payload-header-payload
using O2Message = FairMQParts;

class O2Device : public FairMQDevice
{
 public:
  using FairMQDevice::FairMQDevice;
  ~O2Device() override = default;

  /// Monitoring instance
  std::unique_ptr<o2::monitoring::Monitoring> monitoring;

  /// Provides monitoring instance
  auto GetMonitoring() { return monitoring.get(); }

  /// Connects to a monitoring backend
  void Init() override
  {
    FairMQDevice::Init();
    static constexpr const char* MonitoringUrlKey = "monitoring-url";
    std::string monitoringUrl = GetConfig()->GetValue<std::string>(MonitoringUrlKey);
    if (!monitoringUrl.empty()) {
      monitoring->addBackend(o2::monitoring::MonitoringFactory::GetBackend(monitoringUrl));
    }
  }

  /// Here is how to add an annotated data part (with header);
  /// @param[in,out] parts is a reference to the message;
  /// @param[] inputHeaderStack header block must be MOVED in (rvalue ref)
  /// @param[] inputDataMessage the data message must be MOVED in (unique_ptr by value)
  bool AddMessage(O2Message& parts, o2::header::Stack&& inputHeaderStack, FairMQMessagePtr inputDataMessage)
  {

    // we have to move the incoming data
    o2::header::Stack headerStack{ std::move(inputHeaderStack) };
    FairMQMessagePtr dataMessage{ std::move(inputDataMessage) };

    FairMQMessagePtr headerMessage =
      NewMessage(headerStack.data(), headerStack.size(), &o2::header::Stack::freefn, headerStack.data());
    headerStack.release();

    parts.AddPart(std::move(headerMessage));
    parts.AddPart(std::move(dataMessage));
    return true;
  }

  // this executes user code (e.g. a lambda) on each data block (header-payload pair)
  template <typename F>
  bool ForEach(O2Message& parts, F function)
  {
    if ((parts.Size() % 2) != 0) {
      throw std::invalid_argument(
        "number of parts in message not even (n%2 != 0), cannot be considered an O2 compliant message");
    }

    return ForEach(parts.begin(), parts.end(), function);
  }

  // this executes user code (a member function) on a data block (header-payload pair)
  // at some point should de DEPRECATED in favor of the lambda version
  template <typename T, typename std::enable_if<std::is_base_of<O2Device, T>::value, int>::type = 0>
  bool ForEach(O2Message& parts, bool (T::*memberFunction)(const byte* headerBuffer, size_t headerBufferSize,
                                                           const byte* dataBuffer, size_t dataBufferSize))
  {
    if ((parts.Size() % 2) != 0) {
      throw std::invalid_argument(
        "number of parts in message not even (n%2 != 0), cannot be considered an O2 compliant message");
    }

    return ForEach(parts.fParts.begin(), parts.fParts.end(),
                   [&](gsl::span<const byte> headerBuffer, gsl::span<const byte> dataBuffer) {
                     (static_cast<T*>(this)->*memberFunction)(headerBuffer.data(), headerBuffer.size(),
                                                              dataBuffer.data(), dataBuffer.size());
                   });
  }

 private:
  template <typename I, typename F>
  bool ForEach(I begin, I end, F function)
  {
    using span = gsl::span<const byte>;
    using gsl::narrow_cast;
    for (auto it = begin; it != end; ++it) {
      byte* headerBuffer{ nullptr };
      span::index_type headerBufferSize{ 0 };
      if (*it != nullptr) {
        headerBuffer = reinterpret_cast<byte*>((*it)->GetData());
        headerBufferSize = narrow_cast<span::index_type>((*it)->GetSize());
      }
      ++it;
      byte* dataBuffer{ nullptr };
      span::index_type dataBufferSize{ 0 };
      if (*it != nullptr) {
        dataBuffer = reinterpret_cast<byte*>((*it)->GetData());
        dataBufferSize = narrow_cast<span::index_type>((*it)->GetSize());
      }

      // call the user provided function
      function(span{ headerBuffer, headerBufferSize }, span{ dataBuffer, dataBufferSize });
    }
    return true;
  }
};
}
}
#endif /* O2DEVICE_H_ */
