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
#ifndef O2_FRAMEWORK_CONTROLWEBSOCKETHANDLER_H_
#define O2_FRAMEWORK_CONTROLWEBSOCKETHANDLER_H_
#include "HTTPParser.h"
#include "ControlServiceHelpers.h"
#include <map>
#include <string>

namespace o2::framework
{
struct DriverServerContext;

/// An handler for a websocket message stream.
struct ControlWebSocketHandler : public WebSocketHandler {
  ControlWebSocketHandler(DriverServerContext& context);
  ~ControlWebSocketHandler() override = default;

  /// Invoked at the end of the headers.
  /// as a special header we have "x-dpl-pid" which devices can use
  /// to identify themselves.
  /// FIXME: No effort is done to guarantee their identity. Maybe each device
  ///        should be started with a unique secret if we wanted to provide
  ///        some secutity.
  void headers(std::map<std::string, std::string> const& headers) override;

  /// FIXME: not implemented by the backend.
  void beginFragmentation() override {}

  /// Invoked when a frame it's parsed. Notice you do not own the data and you must
  /// not free the memory.
  void frame(char const* frame, size_t s) override;

  /// FIXME: not implemented
  void endFragmentation() override{};
  /// FIXME: not implemented
  void control(char const* frame, size_t s) override{};

  /// Invoked at the beginning of some incoming data. We simply
  /// reset actions which need to happen on a per chunk basis.
  void beginChunk() override
  {
    didProcessMetric = false;
    didHaveNewMetric = false;
  }

  /// Invoked after we have processed all the available incoming data.
  /// In this particular case we must handle the metric callbacks, if
  /// needed.
  void endChunk() override;

  /// The driver context were we want to accumulate changes
  /// which we got from the websocket.
  DriverServerContext& mContext;
  /// The pid of the remote process actually associated to this
  /// handler. Notice that this information comes as part of
  /// the HTTP headers via x-dpl-pid.
  pid_t mPid = 0;
  /// The index of the remote process associated to this handler.
  size_t mIndex = (size_t)-1;
  /// Wether any frame operation between beginChunk and endChunk
  /// actually processed some metric.
  bool didProcessMetric = false;
  bool didHaveNewMetric = false;
};

} // namespace o2::framework
#endif // O2_FRAMEWORK_CONTROLWEBSOCKETHANDLER_H_
