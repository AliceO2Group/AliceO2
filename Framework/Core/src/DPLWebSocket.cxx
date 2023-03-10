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
#include "Framework/Logger.h"
#include "DPLWebSocket.h"
#include "Framework/GuiCallbackContext.h"
#include "Framework/RuntimeError.h"
#include "Framework/DeviceSpec.h"
#include "Framework/DeviceController.h"
#include "Framework/DevicesManager.h"
#include "DriverServerContext.h"
#include "DriverClientContext.h"
#include "ControlWebSocketHandler.h"
#include "HTTPParser.h"
#include <algorithm>
#include <atomic>
#include <uv.h>
#include <sys/types.h>
#include <unistd.h>
#include "ControlWebSocketHandler.h"

namespace o2::framework
{

static void my_alloc_cb(uv_handle_t* handle, size_t suggested_size, uv_buf_t* buf)
{
  buf->base = (char*)malloc(suggested_size);
  buf->len = suggested_size;
}

/// Free any resource associated with the device - driver channel
void websocket_server_close_callback(uv_handle_t* handle)
{
  LOG(debug) << "socket closed";
  delete (WSDPLHandler*)handle->data;
  free(handle);
}

void ws_error_write_callback(uv_write_t* h, int status)
{
  LOG(error) << "Error in write callback: " << uv_strerror(status);
  if (h->data) {
    free(h->data);
  }
  uv_close((uv_handle_t*)h->handle, websocket_server_close_callback);
  free(h);
}

/// Actually replies to any incoming websocket stuff.
void websocket_server_callback(uv_stream_t* stream, ssize_t nread, const uv_buf_t* buf)
{
  WSDPLHandler* server = (WSDPLHandler*)stream->data;
  assert(server);
  if (nread == 0) {
    return;
  }
  if (nread == UV_EOF) {
    LOG(detail) << "websocket_server_callback: communication with driver closed upon EOF";
    uv_close((uv_handle_t*)stream, websocket_server_close_callback);
    return;
  }
  if (nread < 0) {
    LOG(error) << "websocket_server_callback: Error while reading from websocket" << uv_strerror((int)nread);
    uv_close((uv_handle_t*)stream, websocket_server_close_callback);
    return;
  }
  try {
    parse_http_request(buf->base, nread, server);
    free(buf->base);
  } catch (RuntimeErrorRef& ref) {
    auto& err = o2::framework::error_from_ref(ref);
    LOG(error) << "Error while parsing request: " << err.what;
  }
}

/// Whenever we have handshaken correctly, we can wait for the
/// actual frames until we get an error.
void ws_handshake_done_callback(uv_write_t* h, int status)
{
  if (status) {
    LOG(error) << "uv_write error: " << uv_err_name(status);
    free(h);
    return;
  }
  uv_read_start((uv_stream_t*)h->handle, (uv_alloc_cb)my_alloc_cb, websocket_server_callback);
}

enum struct GUIOpcodes : uint8_t {
  Mousepos = 1,
  Mouseclick = 2,
  Mousewheel = 3,
  Window = 4,
  Latency = 5,
  Keydown = 6,
  Keyup = 7,
  Charin = 8
};

/// An handler for a websocket message stream.
struct GUIWebSocketHandler : public WebSocketHandler {
  GUIWebSocketHandler(DriverServerContext& context, GuiRenderer* renderer)
    : mContext{context}, mRenderer{renderer}
  {
  }
  ~GUIWebSocketHandler() override
  {
    mContext.gui->renderers.erase(mRenderer);
    uv_timer_stop(&(mRenderer->drawTimer));
    delete mRenderer;
    LOGP(info, "RemoteGUI disconnected, {} left", mContext.gui->renderers.size());
  }

  void headers(std::map<std::string, std::string> const& headers) override {}
  void beginFragmentation() override {}
  void frame(char const* frame, size_t s) override
  {
    GUIOpcodes opcode = (GUIOpcodes) * (frame++);
    switch (opcode) {
      case GUIOpcodes::Mousepos: {
        float* positions = (float*)frame;
        mContext.gui->plugin->updateMousePos(positions[0], positions[1]);
        break;
      }
      case GUIOpcodes::Mouseclick: {
        char isClicked = *frame;
        mContext.gui->plugin->updateMouseButton(isClicked == 1);
        break;
      }
      case GUIOpcodes::Mousewheel: {
        int movement = *frame;
        mContext.gui->plugin->updateMouseWheel(movement);
        break;
      }
      case GUIOpcodes::Window: {
        int* size = (int*)frame;
        mContext.gui->plugin->updateWindowSize(size[0], size[1]);
        break;
      }
      case GUIOpcodes::Latency: {
        int lat = *((int*)frame);
        lat = lat < 20 ? 20 : lat;
        uv_timer_set_repeat(&(mRenderer->drawTimer), lat);
        break;
      }
      case GUIOpcodes::Keydown: {
        char key = *frame;
        mContext.gui->plugin->keyDown(key);
        break;
      }
      case GUIOpcodes::Keyup: {
        char key = *frame;
        mContext.gui->plugin->keyUp(key);
        break;
      }
      case GUIOpcodes::Charin: {
        char key = *frame;
        mContext.gui->plugin->charIn(key);
        break;
      }
    }
  }
  void endFragmentation() override{};
  void control(char const* frame, size_t s) override{};
  void beginChunk() override{};
  void endChunk() override{};

  /// The driver context were we want to accumulate changes
  /// which we got from the websocket.
  DriverServerContext& mContext;
  GuiRenderer* mRenderer;
};

WSDPLHandler::WSDPLHandler(uv_stream_t* s, DriverServerContext* context)
  : mStream{s},
    mServerContext{context}
{
}

void WSDPLHandler::method(std::string_view const& s)
{
  if (s != "GET") {
    throw WSError{400, "Bad Request"};
  }
}

void WSDPLHandler::target(std::string_view const& s)
{
  if (s != "/") {
    throw WSError{404, "Unknown"};
  }
}

void populateHeader(std::map<std::string, std::string>& headers, std::string_view const& k, std::string_view const& v)
{
  std::string kk{k};
  std::string vv{v};
  std::transform(kk.begin(), kk.end(), kk.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  if (kk != "sec-websocket-accept" && kk != "sec-websocket-key") {
    std::transform(vv.begin(), vv.end(), vv.begin(),
                   [](unsigned char c) { return std::tolower(c); });
  }
  headers.insert(std::make_pair(kk, vv));
}

void remoteGuiCallback(uv_timer_s* ctx)
{
  GuiRenderer* renderer = reinterpret_cast<GuiRenderer*>(ctx->data);
  assert(renderer);

  void* frame = nullptr;
  void* draw_data = nullptr;
  int size;
  uint64_t frameStart = uv_hrtime();
  uint64_t frameLatency = frameStart - renderer->gui->frameLast;

  // if less than 15ms have passed reuse old frame
  if (renderer->gui->lastFrame == nullptr || frameLatency / 1000000 > 15) {
    renderer->gui->plugin->pollGUIPreRender(renderer->gui->window, (float)frameLatency / 1000000000.0f);
    draw_data = renderer->gui->plugin->pollGUIRender(renderer->gui->callback);
    renderer->gui->plugin->pollGUIPostRender(renderer->gui->window, draw_data);
  } else {
    draw_data = renderer->gui->lastFrame;
  }

  renderer->gui->plugin->getFrameRaw(draw_data, &frame, &size, renderer->updateTextures);
  // For now we only sent the text atlas once
  renderer->updateTextures = false;
  std::vector<uv_buf_t> outputs;
  encode_websocket_frames(outputs, (const char*)frame, size, WebSocketOpCode::Binary, 0);
  renderer->handler->write(outputs);
  free(frame);

  renderer->guiConnected = true;

  if (frameLatency / 1000000 > 15) {
    uint64_t frameEnd = uv_hrtime();
    *(renderer->gui->frameCost) = (frameEnd - frameStart) / 1000000.f;
    *(renderer->gui->frameLatency) = frameLatency / 1000000.f;
    renderer->gui->frameLast = frameStart;
    renderer->gui->lastFrame = draw_data;
  }
}

void WSDPLHandler::header(std::string_view const& k, std::string_view const& v)
{
  populateHeader(mHeaders, k, v);
}

void WSDPLHandler::endHeaders()
{
  /// Make sure this is a websocket upgrade request.
  if (mHeaders["upgrade"] != "websocket") {
    throw WSError{400, "Bad Request: not a websocket upgrade"};
  }

  if (mHeaders["connection"].find("upgrade") == std::string::npos) {
    throw WSError{400, "Bad Request: connection not for upgrade"};
  }
  if (mHeaders["sec-websocket-protocol"] != "dpl") {
    throw WSError{400, "Bad Request: websocket protocol not \"dpl\"."};
  }
  if (mHeaders.count("sec-websocket-key") == 0) {
    throw WSError{400, "Bad Request: sec-websocket-key missing"};
  }
  if (mHeaders["sec-websocket-version"] != "13") {
    throw WSError{400, "Bad Request: wrong protocol version"};
  }
  /// Create an appropriate reply
  LOG(debug) << "Got upgrade request with nonce " << mHeaders["sec-websocket-key"].c_str();
  std::string reply = encode_websocket_handshake_reply(mHeaders["sec-websocket-key"].c_str());
  mHandshaken = true;

  uv_buf_t bfr = uv_buf_init(strdup(reply.data()), reply.size());
  uv_write_t* info_req = (uv_write_t*)malloc(sizeof(uv_write_t));
  uv_write(info_req, (uv_stream_t*)mStream, &bfr, 1, ws_handshake_done_callback);
  auto header = mHeaders.find("x-dpl-pid");
  if (header != mHeaders.end()) {
    LOG(debug) << "Driver connected to PID : " << header->second;
    mHandler = std::make_unique<ControlWebSocketHandler>(*mServerContext);
    mHandler->headers(mHeaders);

    for (size_t i = 0; i < mServerContext->infos->size(); ++i) {
      if (std::to_string((*mServerContext->infos)[i].pid) == header->second) {
        (*mServerContext->controls)[i].controller = new DeviceController{this};
        break;
      }
    }
  } else {
    if ((mServerContext->isDriver && getenv("DPL_DRIVER_REMOTE_GUI")) || ((mServerContext->isDriver == false) && getenv("DPL_DEVICE_REMOTE_GUI"))) {
      LOG(info) << "Connection not bound to a PID";
      GuiRenderer* renderer = new GuiRenderer;
      renderer->gui = mServerContext->gui;
      renderer->handler = this;
      uv_timer_init(mServerContext->loop, &(renderer->drawTimer));
      renderer->drawTimer.data = renderer;
      uv_timer_start(&(renderer->drawTimer), remoteGuiCallback, 0, 200);
      mHandler = std::make_unique<GUIWebSocketHandler>(*mServerContext, renderer);
      mHandler->headers(mHeaders);
      mServerContext->gui->renderers.insert(renderer);

      LOGP(info, "RemoteGUI connected, {} running", mServerContext->gui->renderers.size());
    } else {
      LOGP(warning, "Connection not bound to a PID however {} is not set. Skipping.",
           mServerContext->isDriver ? "DPL_DRIVER_REMOTE_GUI" : "DPL_DEVICE_REMOTE_GUI");
      throw WSError{418, "Remote GUI not enabled"};
    }
  }
}

/// Actual handling of WS frames happens inside here.
void WSDPLHandler::body(char* data, size_t s)
{
  decode_websocket(data, s, *mHandler.get());
}

void ws_server_write_callback(uv_write_t* h, int status)
{
  if (status) {
    LOG(error) << "uv_write error: " << uv_err_name(status);
    free(h);
    return;
  }
  if (h->data) {
    free(h->data);
  }
  free(h);
}

void ws_server_bulk_write_callback(uv_write_t* h, int status)
{
  if (status) {
    LOG(error) << "uv_write error: " << uv_err_name(status);
    free(h);
    return;
  }
  std::vector<uv_buf_t>* buffers = (std::vector<uv_buf_t>*)h->data;
  if (buffers) {
    for (auto& b : *buffers) {
      free(b.base);
    }
  }
  delete buffers;
  free(h);
}

void WSDPLHandler::write(char const* message, size_t s)
{
  uv_buf_t bfr = uv_buf_init(strdup(message), s);
  uv_write_t* write_req = (uv_write_t*)malloc(sizeof(uv_write_t));
  write_req->data = bfr.base;
  uv_write(write_req, (uv_stream_t*)mStream, &bfr, 1, ws_server_write_callback);
}

void WSDPLHandler::write(std::vector<uv_buf_t>& outputs)
{
  if (outputs.empty()) {
    return;
  }
  auto* write_req = (uv_write_t*)malloc(sizeof(uv_write_t));
  auto* buffers = new std::vector<uv_buf_t>;
  buffers->swap(outputs);
  write_req->data = buffers;
  uv_write(write_req, (uv_stream_t*)mStream, &buffers->at(0), buffers->size(), ws_server_bulk_write_callback);
}

/// Helper to return an error
void WSDPLHandler::error(int code, char const* message)
{
  static char const* errorFMT = "HTTP/1.1 {} {}\r\ncontent-type: text/plain\r\n\r\n{}: {}\r\n";
  std::string error = fmt::format(errorFMT, code, message, code, message);
  char* reply = strdup(error.data());
  uv_buf_t bfr = uv_buf_init(reply, error.size());
  auto* error_rep = (uv_write_t*)malloc(sizeof(uv_write_t));
  error_rep->data = reply;
  uv_write(error_rep, (uv_stream_t*)mStream, &bfr, 1, ws_error_write_callback);
}

void close_client_websocket(uv_handle_t* stream)
{
  LOG(debug) << "Closing websocket connection to server";
}

void websocket_client_callback(uv_stream_t* stream, ssize_t nread, const uv_buf_t* buf)
{
  auto* context = (DriverClientContext*)stream->data;
  context->state->loopReason |= DeviceState::WS_COMMUNICATION;
  assert(context->client);
  if (nread == 0) {
    return;
  }
  if (nread == UV_EOF) {
    LOG(debug) << "EOF received from server, closing.";
    uv_read_stop(stream);
    uv_close((uv_handle_t*)stream, close_client_websocket);
    return;
  }
  if (nread < 0) {
    // FIXME: improve error message
    // FIXME: should I close?
    LOG(error) << "Error while reading from websocket";
    uv_read_stop(stream);
    uv_close((uv_handle_t*)stream, close_client_websocket);
    return;
  }
  try {
    LOG(debug) << "Data received from server";
    parse_http_request(buf->base, nread, context->client);
  } catch (RuntimeErrorRef& ref) {
    auto& err = o2::framework::error_from_ref(ref);
    LOG(error) << "Error while parsing request: " << err.what;
  }
}

// FIXME: mNonce should be random
WSDPLClient::WSDPLClient(uv_stream_t* s, std::unique_ptr<DriverClientContext> context, std::function<void()> handshake, std::unique_ptr<WebSocketHandler> handler)
  : mStream{s},
    mNonce{"dGhlIHNhbXBsZSBub25jZQ=="},
    mContext{std::move(context)},
    mHandshake{handshake},
    mHandler{std::move(handler)}
{
  mContext->client = this;
  s->data = mContext.get();
  uv_read_start((uv_stream_t*)s, (uv_alloc_cb)my_alloc_cb, websocket_client_callback);
}

void WSDPLClient::sendHandshake()
{
  std::vector<std::pair<std::string, std::string>> headers = {
    {{"x-dpl-pid"}, std::to_string(getpid())},
    {{"x-dpl-id"}, mContext->spec.id},
    {{"x-dpl-name"}, mContext->spec.name}};
  std::string handShakeString = encode_websocket_handshake_request("/", "dpl", 13, mNonce.c_str(), headers);
  this->write(handShakeString.c_str(), handShakeString.size());
}

void WSDPLClient::replyVersion(std::string_view const& s)
{
  if (s != "HTTP/1.1") {
    throw runtime_error("Not an HTTP reply");
  }
}

void WSDPLClient::replyCode(std::string_view const& s)
{
  if (s != "101") {
    throw runtime_error("Upgrade denied");
  }
}

void WSDPLClient::header(std::string_view const& k, std::string_view const& v)
{
  populateHeader(mHeaders, k, v);
}

void WSDPLClient::dumpHeaders()
{
  for (auto [k, v] : mHeaders) {
    LOG(info) << k << ": " << v;
  }
}

void WSDPLClient::endHeaders()
{
  /// Make sure this is a websocket upgrade request.
  if (mHeaders["upgrade"] != "websocket") {
    throw runtime_error_f("No websocket upgrade");
  }
  // find is used to account for multiple options
  if (mHeaders["connection"].find("upgrade") == std::string::npos) {
    throw runtime_error_f("No connection upgrade");
  }
  if (mHeaders.count("sec-websocket-accept") == 0) {
    throw runtime_error("sec-websocket-accept not found");
  }

  std::string expectedAccept = HTTPParserHelpers::calculateAccept(mNonce.c_str());
  if (mHeaders["sec-websocket-accept"] != expectedAccept) {
    throw runtime_error_f(R"(Invalid accept received: "%s", expected "%s")", mHeaders["sec-websocket-accept"].c_str(), expectedAccept.c_str());
  }

  LOG(info) << "Correctly handshaken websocket connection.";
  /// Create an appropriate reply
  mHandshaken = true;
  mHandshake();
}

struct WriteRequestContext {
  uv_buf_t buf;
  DeviceState* state;
};

struct BulkWriteRequestContext {
  std::vector<uv_buf_t> buffers;
  DeviceState* state;
};

void ws_client_write_callback(uv_write_t* h, int status)
{
  WriteRequestContext* context = (WriteRequestContext*)h->data;
  if (status) {
    LOG(error) << "uv_write error: " << uv_err_name(status);
    free(h);
    return;
  }
  context->state->loopReason |= (DeviceState::WS_COMMUNICATION | DeviceState::WS_READING);
  if (context->buf.base) {
    free(context->buf.base);
  }
  delete context;
  free(h);
}

void ws_client_bulk_write_callback(uv_write_t* h, int status)
{
  BulkWriteRequestContext* context = (BulkWriteRequestContext*)h->data;
  context->state->loopReason |= (DeviceState::WS_COMMUNICATION | DeviceState::WS_WRITING);
  if (status < 0) {
    LOG(error) << "uv_write error: " << uv_err_name(status);
    free(h);
    return;
  }
  if (context->buffers.size()) {
    for (auto& b : context->buffers) {
      free(b.base);
    }
  }
  delete context;
  free(h);
}

/// Actual handling of WS frames happens inside here.
void WSDPLClient::body(char* data, size_t s)
{
  decode_websocket(data, s, *mHandler.get());
}

/// Helper to return an error
void WSDPLClient::write(char const* message, size_t s)
{
  WriteRequestContext* context = new WriteRequestContext;
  context->buf = uv_buf_init(strdup(message), s);
  context->state = mContext->state;
  uv_write_t* write_req = (uv_write_t*)malloc(sizeof(uv_write_t));
  write_req->data = context;
  uv_write(write_req, (uv_stream_t*)mStream, &context->buf, 1, ws_client_write_callback);
}

void WSDPLClient::write(std::vector<uv_buf_t>& outputs)
{
  if (outputs.empty()) {
    return;
  }
  uv_write_t* write_req = (uv_write_t*)malloc(sizeof(uv_write_t));
  BulkWriteRequestContext* context = new BulkWriteRequestContext;
  context->buffers.swap(outputs);
  context->state = mContext->state;
  write_req->data = context;
  uv_write(write_req, (uv_stream_t*)mStream, &context->buffers.at(0),
           context->buffers.size(), ws_client_bulk_write_callback);
}

} // namespace o2::framework
