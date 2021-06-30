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
#include "WSDriverClient.h"
#include "Framework/DeviceState.h"
#include "Framework/DeviceSpec.h"
#include "Framework/Logger.h"
#include "Framework/ServiceRegistry.h"
#include "Framework/DeviceSpec.h"
#include "DriverClientContext.h"
#include "DPLWebSocket.h"
#include <uv.h>
#include <string_view>
#include <charconv>

namespace o2::framework
{

struct ClientWebSocketHandler : public WebSocketHandler {
  ClientWebSocketHandler(WSDriverClient& client)
    : mClient{client}
  {
  }

  void headers(std::map<std::string, std::string> const& headers) override
  {
  }
  /// FIXME: not implemented by the backend.
  void beginFragmentation() override {}

  /// Invoked when a frame it's parsed. Notice you do not own the data and you must
  /// not free the memory.
  void frame(char const* frame, size_t s) override
  {
    mClient.dispatch(std::string_view(frame, s));
  }

  void endFragmentation() override{};
  void control(char const* frame, size_t s) override{};

  /// Invoked at the beginning of some incoming data. We simply
  /// reset actions which need to happen on a per chunk basis.
  void beginChunk() override
  {
  }

  /// Invoked after we have processed all the available incoming data.
  /// In this particular case we must handle the metric callbacks, if
  /// needed.
  void endChunk() override
  {
  }

  /// The driver context were we want to accumulate changes
  /// which we got from the websocket.
  WSDriverClient& mClient;
};

struct ConnectionContext {
  WSDriverClient* client;
  DeviceState* state;
};

void on_connect(uv_connect_t* connection, int status)
{
  if (status < 0) {
    LOG(ERROR) << "Unable to connect to driver.";
    return;
  }
  ConnectionContext* context = (ConnectionContext*)connection->data;
  WSDriverClient* client = context->client;
  context->state->loopReason |= DeviceState::WS_CONNECTED;
  auto onHandshake = [client]() {
    client->flushPending();
  };
  std::lock_guard<std::mutex> lock(client->mutex());
  auto handler = std::make_unique<ClientWebSocketHandler>(*client);
  client->observe("/ping", [](std::string_view) {
    LOG(INFO) << "ping";
  });
  /// FIXME: for now we simply take any offer as 1GB of SHM available
  client->observe("/shm-offer", [state = context->state](std::string_view cmd) {
    static constexpr int prefixSize = std::string_view{"/shm-offer "}.size();
    if (prefixSize > cmd.size()) {
      LOG(ERROR) << "Malformed shared memory offer";
      return;
    }
    cmd.remove_prefix(prefixSize);
    size_t offerSize;
    auto offerSizeError = std::from_chars(cmd.data(), cmd.data() + cmd.size(), offerSize);
    if (offerSizeError.ec != std::errc()) {
      LOG(ERROR) << "Malformed shared memory offer";
      return;
    }
    LOGP(info, "Received {}MB shared memory offer", offerSize);
    ComputingQuotaOffer offer;
    offer.cpu = 0;
    offer.memory = 0;
    offer.sharedMemory = offerSize * 1000000;
    offer.runtime = -1;
    offer.user = -1;
    offer.valid = true;

    state->pendingOffers.push_back(offer);
  });

  client->observe("/quit", [state = context->state](std::string_view offer) {
    state->quitRequested = true;
  });
  auto clientContext = std::make_unique<o2::framework::DriverClientContext>(DriverClientContext{client->spec(), context->state});
  client->setDPLClient(std::make_unique<WSDPLClient>(connection->handle, std::move(clientContext), onHandshake, std::move(handler)));
  client->sendHandshake();
}

/// Helper to connect to a
void connectToDriver(WSDriverClient* driver, DeviceState* state, char const* address, short port)
{
  uv_tcp_t* socket = (uv_tcp_t*)malloc(sizeof(uv_tcp_t));
  uv_tcp_init(state->loop, socket);
  uv_connect_t* connection = (uv_connect_t*)malloc(sizeof(uv_connect_t));
  ConnectionContext* context = new ConnectionContext;
  context->client = driver;
  context->state = state;
  connection->data = context;

  struct sockaddr_in dest;
  uv_ip4_addr(strdup(address), port, &dest);

  uv_tcp_connect(connection, socket, (const struct sockaddr*)&dest, on_connect);
}

void on_awake_main_thread(uv_async_t* handle)
{
  DeviceState* state = (DeviceState*)handle->data;
  state->loopReason |= DeviceState::ASYNC_NOTIFICATION;
}

WSDriverClient::WSDriverClient(ServiceRegistry& registry, DeviceState& state, char const* ip, unsigned short port)
  : mSpec{registry.get<const DeviceSpec>()}
{
  // Must connect the device to the server and send a websocket request.
  // On successful connection we can then start to send commands to the driver.
  // We keep a backlog to make sure we do not lose messages.
  connectToDriver(this, &state, ip, port);
  this->mAwakeMainThread = (uv_async_t*)malloc(sizeof(uv_async_t));
  this->mAwakeMainThread->data = &state;
  uv_async_init(state.loop, this->mAwakeMainThread, on_awake_main_thread);
}

WSDriverClient::~WSDriverClient()
{
  free(this->mAwakeMainThread);
}

void sendMessageToDriver(std::unique_ptr<o2::framework::WSDPLClient>& client, char const* message, size_t s)
{
}

void WSDriverClient::setDPLClient(std::unique_ptr<WSDPLClient> client)
{
  mClient = std::move(client);
  mConnected = true;
}

void WSDriverClient::sendHandshake()
{
  mClient->sendHandshake();
  /// FIXME: nonce should be random
}

void WSDriverClient::tell(const char* msg, size_t s, bool flush)
{
  // Tell will always accumulate and we signal the main thread we
  // have metrics to push
  std::lock_guard<std::mutex> lock(mClientMutex);
  encode_websocket_frames(mBacklog, msg, s, WebSocketOpCode::Binary, 0);
  if (flush) {
    this->awake();
  }
}

void WSDriverClient::awake()
{
  uv_async_send(mAwakeMainThread);
}

void WSDriverClient::flushPending()
{
  std::lock_guard<std::mutex> lock(mClientMutex);
  static bool printed1 = false;
  static bool printed2 = false;
  if (!mClient) {
    if (mBacklog.size() > 2000) {
      if (!printed1) {
        LOG(WARNING) << "Unable to communicate with driver because client does not exist. Continuing connection attempts.";
        printed1 = true;
      }
    }
    return;
  }
  if (!(mClient->isHandshaken())) {
    if (mBacklog.size() > 2000) {
      if (!printed2) {
        LOG(WARNING) << "Unable to communicate with driver because client is not connected. Continuing connection attempts.";
        printed2 = true;
      }
    }
    return;
  }
  if (printed1 || printed2) {
    LOGP(warning, "DriverClient connected successfully. Flushing message backlog of {} messages. All is good.", mBacklog.size());
    printed1 = false;
    printed2 = false;
  }
  mClient->write(mBacklog);
  mBacklog.resize(0);
}

} // namespace o2::framework
