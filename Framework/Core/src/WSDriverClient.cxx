// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "WSDriverClient.h"
#include "Framework/DeviceState.h"
#include "Framework/Logger.h"
#include "Framework/ServiceRegistry.h"
#include "Framework/DeviceSpec.h"
#include "DPLWebSocket.h"
#include <uv.h>

namespace o2::framework
{

void on_connect(uv_connect_t* connection, int status)
{
  if (status < 0) {
    LOG(ERROR) << "Unable to connect to driver.";
    return;
  }
  WSDriverClient* client = (WSDriverClient*)connection->data;
  client->setDPLClient(std::make_unique<WSDPLClient>(connection->handle, client->spec()));
  client->sendHandshake();
}

/// Helper to connect to a
void connectToDriver(WSDriverClient* driver, uv_loop_t* loop, char const* address, short port)
{
  uv_tcp_t* socket = (uv_tcp_t*)malloc(sizeof(uv_tcp_t));
  uv_tcp_init(loop, socket);
  uv_connect_t* connection = (uv_connect_t*)malloc(sizeof(uv_connect_t));
  connection->data = driver;

  struct sockaddr_in dest;
  uv_ip4_addr(strdup(address), port, &dest);

  uv_tcp_connect(connection, socket, (const struct sockaddr*)&dest, on_connect);
}

WSDriverClient::WSDriverClient(ServiceRegistry& registry, DeviceState& state, char const* ip, unsigned short port)
  : mSpec{registry.get<const DeviceSpec>()}
{
  // Must connect the device to the server and send a websocket request.
  // On successful connection we can then start to send commands to the driver.
  // We keep a backlog to make sure we do not lose messages.
  connectToDriver(this, state.loop, ip, port);
}

void sendMessageToDriver(std::unique_ptr<o2::framework::WSDPLClient>& client, char const* message)
{
  std::vector<uv_buf_t> outputs;
  encode_websocket_frames(outputs, message, strlen(message), WebSocketOpCode::Binary, 0);
  client->write(outputs);
}

void WSDriverClient::setDPLClient(std::unique_ptr<WSDPLClient> client)
{
  mClient = std::move(client);
}

void WSDriverClient::sendHandshake()
{
  mClient->sendHandshake();
  /// FIXME: nonce should be random
}

void WSDriverClient::observe(const char*, std::function<void(char const*)>)
{
}

void WSDriverClient::tell(const char* msg)
{
  static bool printed1 = false;
  static bool printed2 = false;
  if (!mClient) {
    mBacklog.push_back(msg);
    if (mBacklog.size() > 100) {
      if (!printed1) {
        LOG(WARNING) << "Unable to communicate with driver because client does not exist";
        printed1 = true;
      }
    }
    return;
  }
  if (!(mClient->isConnected())) {
    mBacklog.push_back(msg);
    if (mBacklog.size() > 100) {
      if (!printed2) {
        LOG(WARNING) << "Unable to communicate with driver because client is not connected";
        printed2 = true;
      }
    }
    return;
  }
  if (printed1 || printed2) {
    LOGP(warning, "DriverClient connected successfully. Flushing message backlog of {} messages", mBacklog.size());
    printed1 = false;
    printed2 = false;
  }
  for (auto& old : mBacklog) {
    sendMessageToDriver(mClient, old.c_str());
  }
  mBacklog.resize(0);
  sendMessageToDriver(mClient, msg);
}

void WSDriverClient::flushPending()
{
  if (!mClient || !(mClient->isConnected())) {
    return;
  }
  for (auto& old : mBacklog) {
    sendMessageToDriver(mClient, old.c_str());
  }
  mBacklog.resize(0);
}

} // namespace o2::framework
