// Copyright 2019-2023 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <CCDB/CCDBDownloader.h>
#include "CommonUtils/StringUtils.h"
#include "CCDB/CCDBTimeStampUtils.h"

#include <curl/curl.h>
#include <unordered_map>
#include <cstdio>
#include <cstdlib>
#include <uv.h>
#include <string>
#include <thread>
#include <vector>
#include <condition_variable>
#include <mutex>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <fairlogger/Logger.h>
#include <boost/asio/ip/host_name.hpp>

namespace o2::ccdb
{

void uvErrorCheck(int code)
{
  if (code != 0) {
    char buf[1000];
    uv_strerror_r(code, buf, 1000);
    LOG(error) << "CCDBDownloader: UV error - " << buf;
  }
}

void curlEasyErrorCheck(CURLcode code)
{
  if (code != CURLE_OK) {
    LOG(error) << "CCDBDownloader: CURL error - " << curl_easy_strerror(code);
  }
}

void curlMultiErrorCheck(CURLMcode code)
{
  if (code != CURLM_OK) {
    LOG(error) << "CCDBDownloader: CURL error - " << curl_multi_strerror(code);
  }
}
namespace
{
std::string uniqueAgentID()
{
  std::string host = boost::asio::ip::host_name();
  char const* jobID = getenv("ALIEN_PROC_ID");
  if (jobID) {
    return fmt::format("{}-{}-{}-{}", host, getCurrentTimestamp() / 1000, o2::utils::Str::getRandomString(6), jobID);
  } else {
    return fmt::format("{}-{}-{}", host, getCurrentTimestamp() / 1000, o2::utils::Str::getRandomString(6));
  }
}
} // namespace

CCDBDownloader::CCDBDownloader()
  : mUserAgentId(uniqueAgentID())
{
  mConstructorCV = new std::condition_variable();
  std::mutex cv_m;
  std::unique_lock<std::mutex> lk(cv_m);

  mLoopThread = new std::thread(&CCDBDownloader::runLoop, this);

  // Don't allow constructor to return unless the uv_loop started running.
  // This protects the loop from receiving handles before it was initialized.
  mConstructorCV->wait(lk);
  mLoopRunning = true;
  delete mConstructorCV;
}

void CCDBDownloader::initializeMultiHandle()
{
  mCurlMultiHandle = curl_multi_init();
  curlMultiErrorCheck(curl_multi_setopt(mCurlMultiHandle, CURLMOPT_SOCKETFUNCTION, handleSocket));
  auto socketData = &mSocketData;
  socketData->curlm = mCurlMultiHandle;
  socketData->CD = this;
  curlMultiErrorCheck(curl_multi_setopt(mCurlMultiHandle, CURLMOPT_SOCKETDATA, socketData));
  curlMultiErrorCheck(curl_multi_setopt(mCurlMultiHandle, CURLMOPT_TIMERFUNCTION, startTimeout));
  curlMultiErrorCheck(curl_multi_setopt(mCurlMultiHandle, CURLMOPT_TIMERDATA, mTimeoutTimer));
  curlMultiErrorCheck(curl_multi_setopt(mCurlMultiHandle, CURLMOPT_MAX_TOTAL_CONNECTIONS, mMaxHandlesInUse));
}

CCDBDownloader::~CCDBDownloader()
{
  // Flag used for debug. Indicates that the uv_loop is supposed to close.
  mIsClosing = true;

  // Send asynchronous signal to close the loop
  signalToClose();
  mLoopThread->join();

  delete mLoopThread;
  delete mUVLoop;
}

void CCDBDownloader::signalToClose()
{
  auto asyncHandle = new uv_async_t();
  asyncHandle->data = this;
  uvErrorCheck(uv_async_init(mUVLoop, asyncHandle, uvCloseCallback));
  uvErrorCheck(uv_async_send(asyncHandle));
}

void CCDBDownloader::uvCloseCallback(uv_async_t* handle)
{
  auto CD = (CCDBDownloader*)handle->data;
  uv_close((uv_handle_t*)handle, onUVClose);
  uv_stop(CD->mUVLoop);
}

void closeHandles(uv_handle_t* handle, void* arg)
{
  auto CD = (CCDBDownloader*)arg;
  if (!uv_is_closing(handle)) {
    uv_close(handle, onUVClose);
  }
}

void onUVClose(uv_handle_t* handle)
{
  if (handle != nullptr) {
    delete handle;
  }
}

void CCDBDownloader::upkeepTimerFunction(uv_timer_t* handle)
{
  auto CD = (CCDBDownloader*)handle->data;
  if (!CD->mLoopRunning) {
    // The fact that this callback is executed means the uv_loop is running.
    // Notify the constructor if it hasn't been already.
    CD->mConstructorCV->notify_all();
  }
  CD->checkForThreadsToJoin();
}

void CCDBDownloader::closesocketCallback(void* clientp, curl_socket_t item)
{
  auto CD = (CCDBDownloader*)clientp;
  if (CD->mSocketTimerMap.find(item) != CD->mSocketTimerMap.end()) {
    auto timer = CD->mSocketTimerMap[item];
    uvErrorCheck(uv_timer_stop(timer));
    // we are getting rid of the uv_timer_t pointer ... so we need
    // to free possibly attached user data pointers as well. Counteracts action of opensocketCallback
    if (timer->data) {
      delete (DataForClosingSocket*)timer->data;
    }
    CD->mSocketTimerMap.erase(item);
    if (close(item) == -1) {
      LOG(error) << "CCDBDownloader: Socket failed to close";
    }
  }
}

curl_socket_t opensocketCallback(void* clientp, curlsocktype purpose, struct curl_sockaddr* address)
{
  auto CD = (CCDBDownloader*)clientp;
  auto sock = socket(address->family, address->socktype, address->protocol);
  if (sock == -1) {
    LOG(error) << "CCDBDownloader: Socket failed to open";
  }

  CD->mSocketTimerMap[sock] = new uv_timer_t();
  uvErrorCheck(uv_timer_init(CD->mUVLoop, CD->mSocketTimerMap[sock]));
  CD->mHandleMap[(uv_handle_t*)CD->mSocketTimerMap[sock]] = true;

  auto data = new DataForClosingSocket();
  data->CD = CD;
  data->socket = sock;
  CD->mSocketTimerMap[sock]->data = data;

  return sock;
}

void CCDBDownloader::asyncUVHandleCheckQueue(uv_async_t* handle)
{
  auto CD = (CCDBDownloader*)handle->data;
  uv_close((uv_handle_t*)handle, onUVClose);
  CD->checkHandleQueue();
}

void CCDBDownloader::closeSocketByTimer(uv_timer_t* handle)
{
  auto data = (DataForClosingSocket*)handle->data;
  auto CD = data->CD;
  auto sock = data->socket;

  if (CD->mSocketTimerMap.find(sock) != CD->mSocketTimerMap.end()) {
    uvErrorCheck(uv_timer_stop(CD->mSocketTimerMap[sock]));
    CD->mSocketTimerMap.erase(sock);
    if (close(sock) == -1) {
      LOG(error) << "CCDBDownloader: Socket failed to close";
    }

    delete data;
  }
}

void CCDBDownloader::curlTimeout(uv_timer_t* handle)
{
  auto CD = (CCDBDownloader*)handle->data;
  int running_handles;
  curl_multi_socket_action(CD->mCurlMultiHandle, CURL_SOCKET_TIMEOUT, 0, &running_handles);
  CD->checkMultiInfo();
}

void CCDBDownloader::curlPerform(uv_poll_t* handle, int status, int events)
{
  uvErrorCheck(status);
  int running_handles;
  int flags = 0;
  if (events & UV_READABLE) {
    flags |= CURL_CSELECT_IN;
  }
  if (events & UV_WRITABLE) {
    flags |= CURL_CSELECT_OUT;
  }

  auto context = (CCDBDownloader::curl_context_t*)handle->data;

  curlMultiErrorCheck(curl_multi_socket_action(context->CD->mCurlMultiHandle, context->sockfd, flags, &running_handles));
  context->CD->checkMultiInfo();
}

int CCDBDownloader::handleSocket(CURL* easy, curl_socket_t s, int action, void* userp, void* socketp)
{
  auto socketData = (CCDBDownloader::DataForSocket*)userp;
  auto CD = (CCDBDownloader*)socketData->CD;
  CCDBDownloader::curl_context_t* curl_context;
  int events = 0;

  switch (action) {
    case CURL_POLL_IN:
    case CURL_POLL_OUT:
    case CURL_POLL_INOUT:

      curl_context = socketp ? (CCDBDownloader::curl_context_t*)socketp : CD->createCurlContext(s);
      curlMultiErrorCheck(curl_multi_assign(socketData->curlm, s, (void*)curl_context));

      if (action != CURL_POLL_IN) {
        events |= UV_WRITABLE;
      }
      if (action != CURL_POLL_OUT) {
        events |= UV_READABLE;
      }

      if (CD->mSocketTimerMap.find(s) != CD->mSocketTimerMap.end()) {
        uvErrorCheck(uv_timer_stop(CD->mSocketTimerMap[s]));
      }

      uvErrorCheck(uv_poll_start(curl_context->poll_handle, events, curlPerform));
      break;
    case CURL_POLL_REMOVE:
      if (socketp) {
        if (CD->mSocketTimerMap.find(s) != CD->mSocketTimerMap.end()) {
          uvErrorCheck(uv_timer_start(CD->mSocketTimerMap[s], closeSocketByTimer, CD->mKeepaliveTimeoutMS, 0));
        }
        uvErrorCheck(uv_poll_stop(((CCDBDownloader::curl_context_t*)socketp)->poll_handle));
        CD->destroyCurlContext((CCDBDownloader::curl_context_t*)socketp);
        curlMultiErrorCheck(curl_multi_assign(socketData->curlm, s, nullptr));
      }
      break;
    default:
      abort();
  }

  return 0;
}

void CCDBDownloader::setMaxParallelConnections(int limit)
{
  mMaxHandlesInUse = limit;
}

void CCDBDownloader::setKeepaliveTimeoutTime(int timeoutMS)
{
  mKeepaliveTimeoutMS = timeoutMS;
}

void CCDBDownloader::setConnectionTimeoutTime(int timeoutMS)
{
  mConnectionTimeoutMS = timeoutMS;
}

void CCDBDownloader::setRequestTimeoutTime(int timeoutMS)
{
  mRequestTimeoutMS = timeoutMS;
}

void CCDBDownloader::setHappyEyeballsHeadstartTime(int headstartMS)
{
  mHappyEyeballsHeadstartMS = headstartMS;
}

void CCDBDownloader::setOfflineTimeoutSettings()
{
  setConnectionTimeoutTime(60000);
  setRequestTimeoutTime(300000);
  setHappyEyeballsHeadstartTime(500);
}

void CCDBDownloader::setOnlineTimeoutSettings()
{
  setConnectionTimeoutTime(5000);
  setRequestTimeoutTime(30000);
  setHappyEyeballsHeadstartTime(500);
}

void CCDBDownloader::checkForThreadsToJoin()
{
  for (int i = 0; i < mThreadFlagPairVector.size(); i++) {
    if (*(mThreadFlagPairVector[i].second)) {
      mThreadFlagPairVector[i].first->join();
      delete (mThreadFlagPairVector[i].first);
      delete (mThreadFlagPairVector[i].second);
      mThreadFlagPairVector.erase(mThreadFlagPairVector.begin() + i);
    }
  }
}

CCDBDownloader::curl_context_t* CCDBDownloader::createCurlContext(curl_socket_t sockfd)
{
  curl_context_t* context;

  context = (curl_context_t*)malloc(sizeof(*context));
  context->CD = this;
  context->sockfd = sockfd;
  context->poll_handle = new uv_poll_t();

  uvErrorCheck(uv_poll_init_socket(mUVLoop, context->poll_handle, sockfd));
  mHandleMap[(uv_handle_t*)(context->poll_handle)] = true;
  context->poll_handle->data = context;

  return context;
}

void CCDBDownloader::curlCloseCB(uv_handle_t* handle)
{
  auto* context = (curl_context_t*)handle->data;
  delete context->poll_handle;
  free(context);
}

void CCDBDownloader::destroyCurlContext(curl_context_t* context)
{
  uv_close((uv_handle_t*)context->poll_handle, curlCloseCB);
}

void callbackWrappingFunction(void (*cbFun)(void*), void* data, bool* completionFlag)
{
  cbFun(data);
  *completionFlag = true;
}

void CCDBDownloader::transferFinished(CURL* easy_handle, CURLcode curlCode)
{
  mHandlesInUse--;
  PerformData* data;
  curlEasyErrorCheck(curl_easy_getinfo(easy_handle, CURLINFO_PRIVATE, &data));

  curlMultiErrorCheck(curl_multi_remove_handle(mCurlMultiHandle, easy_handle));
  *data->codeDestination = curlCode;

  // If no requests left then signal finished based on type of operation
  if (--(*data->requestsLeft) == 0) {
    switch (data->type) {
      case BLOCKING:
        data->cv->notify_all();
        break;
      case ASYNCHRONOUS:
        *data->completionFlag = true;
        break;
      case ASYNCHRONOUS_WITH_CALLBACK:
        *data->completionFlag = true;
        bool* cbFlag = (bool*)malloc(sizeof(bool));
        *cbFlag = false;
        auto cbThread = new std::thread(&callbackWrappingFunction, data->cbFun, data->cbData, cbFlag);
        mThreadFlagPairVector.emplace_back(cbThread, cbFlag);
        break;
    }
  }
  delete data;

  checkHandleQueue();

  // Calling timeout starts a new download if a new easy_handle was added.
  int running_handles;
  curlMultiErrorCheck(curl_multi_socket_action(mCurlMultiHandle, CURL_SOCKET_TIMEOUT, 0, &running_handles));
  checkMultiInfo();
}

void CCDBDownloader::checkMultiInfo()
{
  CURLMsg* message;
  int pending;

  while ((message = curl_multi_info_read(mCurlMultiHandle, &pending))) {
    switch (message->msg) {
      case CURLMSG_DONE: {
        CURLcode code = message->data.result;
        transferFinished(message->easy_handle, code);
      } break;

      default:
        fprintf(stderr, "CURLMSG default\n");
        break;
    }
  }
}

int CCDBDownloader::startTimeout(CURLM* multi, long timeout_ms, void* userp)
{
  auto timeout = (uv_timer_t*)userp;

  if (timeout_ms < 0) {
    uvErrorCheck(uv_timer_stop(timeout));
  } else {
    if (timeout_ms == 0) {
      timeout_ms = 1; // Calling curlTimeout when timeout = 0 could create an infinite loop
    }
    uvErrorCheck(uv_timer_start(timeout, curlTimeout, timeout_ms, 0));
  }
  return 0;
}

void CCDBDownloader::setHandleOptions(CURL* handle, PerformData* data)
{
  curlEasyErrorCheck(curl_easy_setopt(handle, CURLOPT_PRIVATE, data));
  curlEasyErrorCheck(curl_easy_setopt(handle, CURLOPT_CLOSESOCKETFUNCTION, closesocketCallback));
  curlEasyErrorCheck(curl_easy_setopt(handle, CURLOPT_CLOSESOCKETDATA, this));
  curlEasyErrorCheck(curl_easy_setopt(handle, CURLOPT_OPENSOCKETFUNCTION, opensocketCallback));
  curlEasyErrorCheck(curl_easy_setopt(handle, CURLOPT_OPENSOCKETDATA, this));

  curlEasyErrorCheck(curl_easy_setopt(handle, CURLOPT_TIMEOUT_MS, mRequestTimeoutMS));
  curlEasyErrorCheck(curl_easy_setopt(handle, CURLOPT_CONNECTTIMEOUT_MS, mConnectionTimeoutMS));
  curlEasyErrorCheck(curl_easy_setopt(handle, CURLOPT_HAPPY_EYEBALLS_TIMEOUT_MS, mHappyEyeballsHeadstartMS));
  curlEasyErrorCheck(curl_easy_setopt(handle, CURLOPT_USERAGENT, mUserAgentId.c_str()));
}

void CCDBDownloader::checkHandleQueue()
{
  // Lock access to handle queue
  mHandlesQueueLock.lock();
  if (mHandlesToBeAdded.size() > 0) {
    // Add handles without going over the limit
    while (mHandlesToBeAdded.size() > 0 && mHandlesInUse < mMaxHandlesInUse) {
      curlMultiErrorCheck(curl_multi_add_handle(mCurlMultiHandle, mHandlesToBeAdded.front()));
      mHandlesInUse++;
      mHandlesToBeAdded.erase(mHandlesToBeAdded.begin());
    }
  }
  mHandlesQueueLock.unlock();
}

void CCDBDownloader::runLoop()
{
  // Only runLoop() function and functions called by it are allowed to interact with the uv_loop via non asynchronous handles.
  mUVLoop = new uv_loop_t();
  uvErrorCheck(uv_loop_init(mUVLoop));

  // Preparing timer to be used by curl
  mTimeoutTimer = new uv_timer_t();
  mTimeoutTimer->data = this;
  uvErrorCheck(uv_timer_init(mUVLoop, mTimeoutTimer));
  mHandleMap[(uv_handle_t*)mTimeoutTimer] = true;

  // Preparing curl handle
  initializeMultiHandle();

  // Global timer
  // uv_loop runs only when there are active handles, this handle guarantees the loop won't close immedietly after starting
  auto timerCheckQueueHandle = new uv_timer_t();
  timerCheckQueueHandle->data = this;
  uvErrorCheck(uv_timer_init(mUVLoop, timerCheckQueueHandle));
  mHandleMap[(uv_handle_t*)timerCheckQueueHandle] = true;
  uvErrorCheck(uv_timer_start(timerCheckQueueHandle, upkeepTimerFunction, 1, 100));

  // Start the loop
  uv_run(mUVLoop, UV_RUN_DEFAULT);

  // Loop has been ordered to stop via signalToClose()
  curlMultiErrorCheck(curl_multi_cleanup(mCurlMultiHandle));

  // Schedule all handles to close. Execute loop once to allow them to execute their destructors.
  while (uv_loop_alive(mUVLoop) && uv_loop_close(mUVLoop) == UV_EBUSY) {
    uv_walk(mUVLoop, closeHandles, this);
    uv_run(mUVLoop, UV_RUN_ONCE);
  }

  if (!mIsClosing) {
    LOG(error) << "CCDBDownloader: uvloop closed prematurely";
  }
}

CURLcode CCDBDownloader::perform(CURL* handle)
{
  std::vector<CURL*> handleVector;
  handleVector.push_back(handle);
  return batchBlockingPerform(handleVector).back();
}

std::vector<CURLcode> CCDBDownloader::batchAsynchPerform(std::vector<CURL*> const& handleVector, bool* completionFlag)
{
  std::vector<CURLcode> codeVector(handleVector.size());
  size_t* requestsLeft = new size_t();
  *requestsLeft = handleVector.size();

  mHandlesQueueLock.lock();
  for (int i = 0; i < handleVector.size(); i++) {
    auto* data = new CCDBDownloader::PerformData();

    data->codeDestination = &(codeVector)[i];
    (codeVector)[i] = CURLE_FAILED_INIT;

    data->requestsLeft = requestsLeft;
    data->completionFlag = completionFlag;
    data->type = ASYNCHRONOUS;

    setHandleOptions(handleVector[i], data);
    mHandlesToBeAdded.push_back(handleVector[i]);
  }
  mHandlesQueueLock.unlock();
  makeLoopCheckQueueAsync();
  return codeVector;
}

std::vector<CURLcode> CCDBDownloader::batchBlockingPerform(std::vector<CURL*> const& handleVector)
{
  std::condition_variable cv;
  std::mutex cv_m;
  std::unique_lock<std::mutex> lk(cv_m);

  std::vector<CURLcode> codeVector(handleVector.size());
  size_t requestsLeft = handleVector.size();

  mHandlesQueueLock.lock();
  for (int i = 0; i < handleVector.size(); i++) {
    auto* data = new CCDBDownloader::PerformData();
    data->codeDestination = &codeVector[i];
    codeVector[i] = CURLE_FAILED_INIT;

    data->cv = &cv;
    data->type = BLOCKING;
    data->requestsLeft = &requestsLeft;

    setHandleOptions(handleVector[i], data);
    mHandlesToBeAdded.push_back(handleVector[i]);
  }
  mHandlesQueueLock.unlock();
  makeLoopCheckQueueAsync();
  cv.wait(lk);
  return codeVector;
}

std::vector<CURLcode> CCDBDownloader::asynchBatchPerformWithCallback(std::vector<CURL*> const& handleVector, bool* completionFlag, void (*cbFun)(void*), void* cbData)
{
  std::vector<CURLcode> codeVector(handleVector.size());
  size_t* requestsLeft = new size_t();
  *requestsLeft = handleVector.size();

  mHandlesQueueLock.lock();
  for (int i = 0; i < handleVector.size(); i++) {
    auto* data = new CCDBDownloader::PerformData();

    data->codeDestination = &(codeVector)[i];
    (codeVector)[i] = CURLE_FAILED_INIT;

    data->requestsLeft = requestsLeft;
    data->completionFlag = completionFlag;
    data->type = ASYNCHRONOUS_WITH_CALLBACK;
    data->cbFun = cbFun;
    data->cbData = cbData;

    setHandleOptions(handleVector[i], data);
    mHandlesToBeAdded.push_back(handleVector[i]);
  }
  mHandlesQueueLock.unlock();
  makeLoopCheckQueueAsync();
  return codeVector;
}

void CCDBDownloader::makeLoopCheckQueueAsync()
{
  auto asyncHandle = new uv_async_t();
  asyncHandle->data = this;
  uvErrorCheck(uv_async_init(mUVLoop, asyncHandle, asyncUVHandleCheckQueue));
  uvErrorCheck(uv_async_send(asyncHandle));
}

} // namespace o2
