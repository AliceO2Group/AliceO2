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

CCDBDownloader::CCDBDownloader(uv_loop_t* uv_loop)
  : mUserAgentId(uniqueAgentID())
{
  if (uv_loop) {
    mExternalLoop = true;
    mUVLoop = uv_loop;
  } else {
    mExternalLoop = false;
    setupInternalUVLoop();
  }

  // Preparing timer to be used by curl
  mTimeoutTimer = new uv_timer_t();
  mTimeoutTimer->data = this;
  uvErrorCheck(uv_timer_init(mUVLoop, mTimeoutTimer));
  mHandleMap[(uv_handle_t*)mTimeoutTimer] = true;

  initializeMultiHandle();
}

void CCDBDownloader::setupInternalUVLoop()
{
  mUVLoop = new uv_loop_t();
  uvErrorCheck(uv_loop_init(mUVLoop));
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
  // Loop has been ordered to stop via signalToClose()
  curlMultiErrorCheck(curl_multi_cleanup(mCurlMultiHandle));

  if (!mExternalLoop) {
    // Schedule all handles to close. Execute loop to allow them to execute their destructors.
    while (uv_loop_alive(mUVLoop) && uv_loop_close(mUVLoop) == UV_EBUSY) {
      uv_walk(mUVLoop, closeHandles, this);
      uv_run(mUVLoop, UV_RUN_ONCE);
    }
    delete mUVLoop;
  }
}

void closeHandles(uv_handle_t* handle, void* arg)
{
  auto CD = (CCDBDownloader*)arg;
  // Close only handles belonging to the Downloader
  if (CD->mHandleMap.find(handle) != CD->mHandleMap.end()) {
    if (!uv_is_closing(handle)) {
      uv_close(handle, onUVClose);
    }
  }
}

void onUVClose(uv_handle_t* handle)
{
  if (handle != nullptr) {
    delete handle;
  }
}

void CCDBDownloader::closesocketCallback(void* clientp, curl_socket_t item)
{
  auto CD = (CCDBDownloader*)clientp;
  if (CD->mExternalLoop) {
    // If external uv loop is used then the keepalive mechanism is active.
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
  } else {
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

  if (CD->mExternalLoop) {
    CD->mSocketTimerMap[sock] = new uv_timer_t();
    uvErrorCheck(uv_timer_init(CD->mUVLoop, CD->mSocketTimerMap[sock]));
    CD->mHandleMap[(uv_handle_t*)CD->mSocketTimerMap[sock]] = true;

    auto data = new DataForClosingSocket();
    data->CD = CD;
    data->socket = sock;
    CD->mSocketTimerMap[sock]->data = data;
  }

  return sock;
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

      if (CD->mExternalLoop && CD->mSocketTimerMap.find(s) != CD->mSocketTimerMap.end()) {
        uvErrorCheck(uv_timer_stop(CD->mSocketTimerMap[s]));
      }

      uvErrorCheck(uv_poll_start(curl_context->poll_handle, events, curlPerform));
      break;
    case CURL_POLL_REMOVE:
      if (socketp) {
        if (CD->mExternalLoop) {
          // If external loop is used then start the keepalive timeout.
          if (CD->mSocketTimerMap.find(s) != CD->mSocketTimerMap.end()) {
            uvErrorCheck(uv_timer_start(CD->mSocketTimerMap[s], closeSocketByTimer, CD->mKeepaliveTimeoutMS, 0));
          }
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

void CCDBDownloader::tryNewHost(PerformData* performData, CURL* easy_handle)
{
  auto requestData = performData->requestData;
  std::string newUrl = requestData->hosts.at(performData->hostInd) + "/" + requestData->path + "/" + std::to_string(requestData->timestamp);
  LOG(debug) << "Connecting to another host " << newUrl;
  requestData->hoPair.header.clear();
  curl_easy_setopt(easy_handle, CURLOPT_URL, newUrl.c_str());
  mHandlesToBeAdded.push_back(easy_handle);
}

void CCDBDownloader::getLocalContent(PerformData* performData, std::string& newLocation, bool& contentRetrieved, std::vector<std::string>& locations)
{
  auto requestData = performData->requestData;
  LOG(debug) << "Redirecting to local content " << newLocation << "\n";
  if (requestData->localContentCallback(newLocation)) {
    contentRetrieved = true;
  } else {
    // Prepare next redirect url
    newLocation = getNewLocation(performData, locations);
  }
}

std::string CCDBDownloader::getNewLocation(PerformData* performData, std::vector<std::string>& locations) const
{
  auto requestData = performData->requestData;
  if (performData->locInd < locations.size()) {
    std::string newLocation = locations.at(performData->locInd++);
    std::string hostUrl = requestData->hosts.at(performData->hostInd);
    std::string newUrl = prepareRedirectedURL(newLocation, hostUrl);
    return newUrl;
  } else {
    return "";
  }
}

void CCDBDownloader::httpRedirect(PerformData* performData, std::string& newLocation, CURL* easy_handle)
{
  auto requestData = performData->requestData;
  LOG(debug) << "Trying content location " << newLocation;
  curl_easy_setopt(easy_handle, CURLOPT_URL, newLocation.c_str());
  mHandlesToBeAdded.push_back(easy_handle);
}

void CCDBDownloader::followRedirect(PerformData* performData, CURL* easy_handle, std::vector<std::string>& locations, bool& rescheduled, bool& contentRetrieved)
{
  std::string newLocation = getNewLocation(performData, locations);
  if (newLocation.find("alien:/", 0) != std::string::npos || newLocation.find("file:/", 0) != std::string::npos) {
    getLocalContent(performData, newLocation, contentRetrieved, locations);
  }
  if (!contentRetrieved && newLocation != "") {
    httpRedirect(performData, newLocation, easy_handle);
    rescheduled = true;
  }
}

std::string CCDBDownloader::trimHostUrl(std::string full_host_url) const
{
  CURLU* host_url = curl_url();
  curl_url_set(host_url, CURLUPART_URL, full_host_url.c_str(), 0);

  // Get host part (the only critical part)
  char* host;
  CURLUcode host_result = curl_url_get(host_url, CURLUPART_HOST, &host, 0);
  if (host_result != CURLUE_OK) {
    LOG(error) << "CCDBDownloader: Malformed url detected when processing redirect, could not identify the host part: " << host;
    curl_url_cleanup(host_url);
    return "";
  }
  // Get scheme (protocol) part
  char* scheme;
  CURLUcode scheme_result = curl_url_get(host_url, CURLUPART_SCHEME, &scheme, 0);
  // Get port
  char* port;
  CURLUcode port_result = curl_url_get(host_url, CURLUPART_PORT, &port, 0);

  curl_url_cleanup(host_url);

  // Assemble parts
  std::string trimmed_url = "";
  if (scheme_result == CURLUE_OK) {
    trimmed_url += scheme + std::string("://");
    free(scheme);
  }
  trimmed_url += host;
  free(host);
  if (port_result == CURLUE_OK) {
    trimmed_url += std::string(":") + port;
    free(port);
  }
  return trimmed_url;
}

std::string CCDBDownloader::prepareRedirectedURL(std::string address, std::string potentialHost) const
{
  // If it is an alien or local address it does not need preparation
  if (address.find("alien:/") != std::string::npos || address.find("file:/") != std::string::npos) {
    return address;
  }
  // Check if URL contains a scheme (protocol)
  CURLU* redirected_url = curl_url();
  curl_url_set(redirected_url, CURLUPART_URL, address.c_str(), 0);
  char* scheme;
  CURLUcode scheme_result = curl_url_get(redirected_url, CURLUPART_SCHEME, &scheme, 0);
  curl_free(scheme);
  curl_url_cleanup(redirected_url);
  if (scheme_result == CURLUE_OK) {
    // The redirected_url contains a scheme (protocol) so there is no need for preparation
    return address;
  }
  // If the address doesn't contain a scheme it means it is a relative url. We need to append it to the trimmed host url
  // The host url must be trimmed from it's path (if it ends in one) as otherwise the redirection url would be appended after said path
  return trimHostUrl(potentialHost) + address;
}

void CCDBDownloader::transferFinished(CURL* easy_handle, CURLcode curlCode)
{
  mHandlesInUse--;
  PerformData* performData;
  curlEasyErrorCheck(curl_easy_getinfo(easy_handle, CURLINFO_PRIVATE, &performData));

  curlMultiErrorCheck(curl_multi_remove_handle(mCurlMultiHandle, easy_handle));
  *performData->codeDestination = curlCode;

  bool rescheduled = false;
  bool contentRetrieved = false;

  switch (performData->type) {
    case BLOCKING: {
      --(*performData->requestsLeft);
    } break;
    case ASYNCHRONOUS: {
      DownloaderRequestData* requestData = performData->requestData;

      if (requestData->headers) {
        for (auto& p : requestData->hoPair.header) {
          (*requestData->headers)[p.first] = p.second;
        }
      }
      if (requestData->errorflag && requestData->headers) {
        (*requestData->headers)["Error"] = "An error occurred during retrieval";
      }

      // Log that transfer finished
      long httpCode;
      curl_easy_getinfo(easy_handle, CURLINFO_RESPONSE_CODE, &httpCode);
      char* url;
      curl_easy_getinfo(easy_handle, CURLINFO_EFFECTIVE_URL, &url);
      LOG(debug) << "Transfer for " << url << " finished with code " << httpCode << "\n";

      // Get alternative locations for the same host
      auto locations = getLocations(&(requestData->hoPair.header));

      // React to received http code
      if (404 == httpCode) {
        LOG(error) << "Requested resource does not exist: " << url;
      } else if (304 == httpCode) {
        LOGP(debug, "Object exists but I am not serving it since it's already in your possession");
        contentRetrieved = true;
      } else if (300 <= httpCode && httpCode < 400 && performData->locInd < locations.size()) {
        followRedirect(performData, easy_handle, locations, rescheduled, contentRetrieved);
      } else if (200 <= httpCode && httpCode < 300) {
        contentRetrieved = true;
      } else {
        LOG(error) << "Error in fetching object " << url << ", curl response code:" << httpCode;
      }

      // Check if content was retrieved, or scheduled to be retrieved
      if (!rescheduled && !contentRetrieved && performData->locInd == locations.size()) {
        // Ran out of locations to redirect, try new host
        if (++performData->hostInd < requestData->hosts.size()) {
          tryNewHost(performData, easy_handle);
          rescheduled = true;
        } else {
          LOG(error) << "File " << requestData->path << " could not be retrieved. No more hosts to try.";
        }
      }

      if (!rescheduled) {
        // No more transfers will be done for this request, do cleanup specific for ASYNCHRONOUS calls
        --(*performData->requestsLeft);
        delete requestData;
        delete performData->codeDestination;
        if (!contentRetrieved) {
          LOGP(alarm, "Curl request to {}, response code: {}", url, httpCode);
        }
      }
    } break;
  }
  if (!rescheduled) {
    // No more transfers will be done for this request, do general cleanup
    delete performData;
  }

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
  if (mHandlesToBeAdded.size() > 0) {
    // Add handles without going over the limit
    while (mHandlesToBeAdded.size() > 0 && mHandlesInUse < mMaxHandlesInUse) {
      curlMultiErrorCheck(curl_multi_add_handle(mCurlMultiHandle, mHandlesToBeAdded.front()));
      mHandlesInUse++;
      mHandlesToBeAdded.erase(mHandlesToBeAdded.begin());
    }
  }
}

void CCDBDownloader::runLoop(bool noWait)
{
  uv_run(mUVLoop, noWait ? UV_RUN_NOWAIT : UV_RUN_ONCE);
}

CURLcode CCDBDownloader::perform(CURL* handle)
{
  std::vector<CURL*> handleVector;
  handleVector.push_back(handle);
  return batchBlockingPerform(handleVector).back();
}

std::vector<std::string> CCDBDownloader::getLocations(std::multimap<std::string, std::string>* headerMap) const
{
  std::vector<std::string> locs;
  auto iter = headerMap->find("Location");
  if (iter != headerMap->end()) {
    locs.push_back(iter->second);
  }
  // add alternative locations (not yet included)
  auto iter2 = headerMap->find("Content-Location");
  if (iter2 != headerMap->end()) {
    auto range = headerMap->equal_range("Content-Location");
    for (auto it = range.first; it != range.second; ++it) {
      if (std::find(locs.begin(), locs.end(), it->second) == locs.end()) {
        locs.push_back(it->second);
      }
    }
  }
  return locs;
}

std::vector<CURLcode> CCDBDownloader::batchBlockingPerform(std::vector<CURL*> const& handleVector)
{
  std::vector<CURLcode> codeVector(handleVector.size());
  size_t requestsLeft = handleVector.size();

  for (int i = 0; i < handleVector.size(); i++) {
    auto* data = new CCDBDownloader::PerformData();
    data->codeDestination = &codeVector[i];
    codeVector[i] = CURLE_FAILED_INIT;

    data->type = BLOCKING;
    data->requestsLeft = &requestsLeft;
    setHandleOptions(handleVector[i], data);
    mHandlesToBeAdded.push_back(handleVector[i]);
  }
  checkHandleQueue();
  while (requestsLeft > 0) {
    uv_run(mUVLoop, UV_RUN_ONCE);
  }

  return codeVector;
}

void CCDBDownloader::asynchSchedule(CURL* handle, size_t* requestCounter)
{
  (*requestCounter)++;

  CURLcode* codeVector = new CURLcode();

  // Get data about request
  DownloaderRequestData* requestData;
  std::multimap<std::string, std::string>* headerMap;
  std::vector<std::string>* hostsPool;
  curl_easy_getinfo(handle, CURLINFO_PRIVATE, &requestData);
  headerMap = &(requestData->hoPair.header);
  hostsPool = &(requestData->hosts);

  // Prepare temporary data about transfer
  auto* data = new CCDBDownloader::PerformData(); // Freed in transferFinished
  data->codeDestination = codeVector;
  *codeVector = CURLE_FAILED_INIT;

  data->type = ASYNCHRONOUS;
  data->requestsLeft = requestCounter;
  data->hostInd = 0;
  data->locInd = 0;
  data->requestData = requestData;

  // Prepare handle and schedule download
  setHandleOptions(handle, data);
  mHandlesToBeAdded.push_back(handle);

  checkHandleQueue();

  // return codeVector;
}

} // namespace o2
