// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Framework/DeviceController.h"
#include "DPLWebSocket.h"
#include "HTTPParser.h"
#include "Framework/Logger.h"
#include <uv.h>
#include <vector>

namespace o2::framework
{

DeviceController::DeviceController(WSDPLHandler* handler)
  : mHandler{handler}
{
}

void DeviceController::hello()
{
  LOG(debug) << "Saying hello";
  std::vector<uv_buf_t> outputs;
  encode_websocket_frames(outputs, "hello", strlen("hello"), WebSocketOpCode::Binary, 0);
  mHandler->write(outputs);
}

void DeviceController::write(char const* message, size_t s)
{
  LOGP(debug, "Saying {} to device", message);
  std::vector<uv_buf_t> outputs;
  encode_websocket_frames(outputs, message, s, WebSocketOpCode::Binary, 0);
  mHandler->write(outputs);
}

} // namespace o2::framework
