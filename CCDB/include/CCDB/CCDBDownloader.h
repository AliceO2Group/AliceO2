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
#ifndef O2_CCDBDOWNLOADER_H_
#define O2_CCDBDOWNLOADER_H_

#if !defined(__CINT__) && !defined(__MAKECINT__) && !defined(__ROOTCLING__) && !defined(__CLING__)
#include "MemoryResources/MemoryResources.h"
#endif

#include <cstdio>
#include <cstdlib>
#include <curl/curl.h>
#include <string>
#include <vector>
#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <unordered_map>
#include <map>
#include <functional>

typedef struct uv_loop_s uv_loop_t;
typedef struct uv_timer_s uv_timer_t;
typedef struct uv_poll_s uv_poll_t;
typedef struct uv_signal_s uv_signal_t;
typedef struct uv_async_s uv_async_t;
typedef struct uv_handle_s uv_handle_t;

namespace o2::ccdb
{

#if !defined(__CINT__) && !defined(__MAKECINT__) && !defined(__ROOTCLING__) && !defined(__CLING__)
struct HeaderObjectPair_t {
  std::multimap<std::string, std::string> header;
  o2::pmr::vector<char>* object = nullptr;
  int counter = 0;
};

typedef struct DownloaderRequestData {
  std::vector<std::string> hosts;
  std::string path;
  long timestamp;
  HeaderObjectPair_t hoPair;
  std::map<std::string, std::string>* headers;

  std::function<bool(std::string)> localContentCallback;
  bool errorflag = false;
} DownloaderRequestData;
#endif

/*
 Some functions below aren't member functions of CCDBDownloader because both curl and libuv require callback functions which have to be either static or non-member.
 Because non-static functions are used in the functions below, they must be non-member.
*/

/**
 * uv_walk callback which is used to close passed handle.
 *
 * @param handle Handle to be closed.
 * @param arg Argument required by callback template. Is not used in this implementation.
 */
void closeHandles(uv_handle_t* handle, void* arg);

/**
 * Called by CURL in order to open a new socket. Newly opened sockets are assigned a timeout timer and added to socketTimerMap.
 *
 * @param clientp Pointer to the CCDBDownloader instance which controls the socket.
 * @param purpose Purpose of opened socket. This parameter is unused but required by the callback template.
 * @param address Structure containing information about family, type and protocol for the socket.
 */
curl_socket_t opensocketCallback(void* clientp, curlsocktype purpose, struct curl_sockaddr* address);

/**
 * Delete the handle.
 *
 * @param handle Handle assigned to this callback.
 */
void onUVClose(uv_handle_t* handle);

/// A class encapsulating and performing simple CURL requests in terms of a so-called CURL multi-handle.
/// A multi-handle allows to use a connection pool (connection cache) in the CURL layer even
/// with short-lived CURL easy-handles. Thereby the overhead of connection to servers can be
/// significantly reduced. For more info, see for instance https://everything.curl.dev/libcurl/connectionreuse.
///
/// Further, this class adds functionality on top
/// of simple CURL (aysync requests, timeout handling, event loop, etc).
class CCDBDownloader
{
 public:
  /**
   * Timer starts for each socket when its respective transfer finishes, and is stopped when another transfer starts for that handle.
   * When the timer runs out it closes the socket. The period for which socket stays open is defined by socketTimeoutMS.
   */
  std::unordered_map<curl_socket_t, uv_timer_t*> mSocketTimerMap;

  /**
   * The UV loop which handles transfers. Can be created internally or provided through a constructor.
   */
  uv_loop_t* mUVLoop;

  /**
   * Map used to store active uv_handles belonging to the CcdbDownloader. If internal uv_loop is used, then all uv_handles should be marked in this map.
   */
  std::unordered_map<uv_handle_t*, bool> mHandleMap;

  /**
   * Time for which sockets will stay open after last download finishes
   */
  int mKeepaliveTimeoutMS = 100;

  /**
   * Time for connection to start before it times out.
   */
  int mConnectionTimeoutMS = 60000;

  /**
   * Time for request to finish before it times out.
   */
  int mRequestTimeoutMS = 300000;

  /**
   * Head start of IPv6 in regards to IPv4.
   */
  int mHappyEyeballsHeadstartMS = 500;

  /**
   * Max number of handles that can be used at the same time
   */
  int mMaxHandlesInUse = 3;

  /**
   * Variable denoting whether an external or internal uv_loop is being used.
   */
  bool mExternalLoop;

  CCDBDownloader(uv_loop_t* uv_loop = nullptr);
  ~CCDBDownloader();

  /**
   * Perform on a single handle in a blocking manner. Has the same effect as curl_easy_perform().
   *
   * @param handle Handle to be performed on. It can be reused or cleaned after perform finishes.
   */
  CURLcode perform(CURL* handle);

  /**
   * Perform on a batch of handles in a blocking manner. Has the same effect as calling curl_easy_perform() on all handles in the vector.
   * @param handleVector Handles to be performed on.
   */
  std::vector<CURLcode> batchBlockingPerform(std::vector<CURL*> const& handleVector);

  /**
   * Schedules an asynchronous transfer but doesn't perform it.
   *
   * @param handle Handle to be performed on.
   * @param requestCounter Counter shared by a batch of CURL handles.
   */
  void asynchSchedule(CURL* handle, size_t* requestCounter);

  /**
   * Limits the number of parallel connections. Should be used only if no transfers are happening.
   */
  void setMaxParallelConnections(int limit);

  /**
   * Limits the time a socket and its connection will be opened after transfer finishes.
   */
  void setKeepaliveTimeoutTime(int timeoutMS);

  /**
   * Setter for the connection timeout.
   */
  void setConnectionTimeoutTime(int timeoutMS);

  /**
   * Setter for the request timeout.
   */
  void setRequestTimeoutTime(int timeoutMS);

  /**
   * Setter for the happy eyeballs headstart.
   */
  void setHappyEyeballsHeadstartTime(int headstartMS);

  /**
   * Sets the timeout values selected for the offline environment.
   */
  void setOfflineTimeoutSettings();

  /**
   * Sets the timeout values selected for the online environment.
   */
  void setOnlineTimeoutSettings();

  /**
   * Run the uvLoop once.
   *
   * @param noWait Using this flag will cause the loop to run only if sockets have pendind data.
   */
  void runLoop(bool noWait);

 private:
  /**
   * Leaves only the protocol and host part of the url, discrading path and metadata.
   */
  std::string trimHostUrl(std::string full_host_url) const;

  /**
   * Recognizes whether the address is a full url, or a partial one (like for example "/Task/Detector/1") and combines it with potentialHost if needed.
   */
  std::string prepareRedirectedURL(std::string address, std::string potentialHost) const;

  /**
   * Returns a vector of possible content locations based on the redirect headers.
   *
   * @param baseUrl Content path.
   * @param headerMap Map containing response headers.
   */
  std::vector<std::string> getLocations(std::multimap<std::string, std::string>* headerMap) const;

  std::string mUserAgentId = "CCDBDownloader";
  /**
   * Sets up internal UV loop.
   */
  void setupInternalUVLoop();

  /**
   * Current amount of handles which are performed on.
   */
  int mHandlesInUse = 0;

  /**
   * Multi handle which controlls all network flow.
   */
  CURLM* mCurlMultiHandle = nullptr;

  /**
   * The timeout clock that is be used by CURL.
   */
  uv_timer_t* mTimeoutTimer;

  /**
   * Queue of handles awaiting their transfers to start.
   */
  std::vector<CURL*> mHandlesToBeAdded;

  /**
   * Types of requests.
   */
  enum RequestType {
    BLOCKING,
    ASYNCHRONOUS
  };

  /**
   * Information about a socket.
   */
  typedef struct curl_context_s {
    uv_poll_t* poll_handle;
    curl_socket_t sockfd = -1;
    CCDBDownloader* CD = nullptr;
  } curl_context_t;

  /**
   * Structure used for CURLMOPT_SOCKETDATA, which gives context for handleSocket
   */
  typedef struct DataForSocket {
    CCDBDownloader* CD;
    CURLM* curlm;
  } DataForSocket;

  DataForSocket mSocketData;

#if !defined(__CINT__) && !defined(__MAKECINT__) && !defined(__ROOTCLING__) && !defined(__CLING__)
  /**
   * Structure which is stored in a easy_handle. It carries information about the request which the easy_handle is part of.
   */
  typedef struct PerformData {
    CURLcode* codeDestination;
    size_t* requestsLeft;
    RequestType type;
    int hostInd;
    int locInd;
    DownloaderRequestData* requestData;
  } PerformData;
#endif

  /**
   * Called by CURL in order to close a socket. It will be called by CURL even if a timeout timer closed the socket beforehand.
   *
   * @param clientp Pointer to the CCDBDownloader instance which controls the socket.
   * @param item File descriptor of the socket.
   */
  static void closesocketCallback(void* clientp, curl_socket_t item);

#if !defined(__CINT__) && !defined(__MAKECINT__) && !defined(__ROOTCLING__) && !defined(__CLING__)
  // Returns a new location string or an empty string if all locations under current host have been accessedd
  std::string getNewLocation(PerformData* performData, std::vector<std::string>& locations) const;

  // Reschedules the transfer to be performed with a different host.
  void tryNewHost(PerformData* performData, CURL* easy_handle);

  // Retrieves content from either alien, cvmfs or local storage using a callback to CCDBApi.
  void getLocalContent(PerformData* performData, std::string& newLocation, bool& contentRetrieved, std::vector<std::string>& locations);

  // Continues a transfer via a http redirect.
  void httpRedirect(PerformData* performData, std::string& newLocation, CURL* easy_handle);

  // Continues a transfer via a redirect. The redirect can point to a local file, alien file or a http address.
  void followRedirect(PerformData* performData, CURL* easy_handle, std::vector<std::string>& locations, bool& rescheduled, bool& contentRetrieved);
#endif

  /**
   *  Is used to react to polling file descriptors in poll_handle.
   *
   * @param handle Handle assigned to this callback.
   * @param status Used to signal errors.
   * @param events Bitmask used to describe events on the socket.
   */
  static void curlPerform(uv_poll_t* handle, int status, int events);

  /**
   * Used by CURL to react to action happening on a socket.
   */
  static int handleSocket(CURL* easy, curl_socket_t s, int action, void* userp, void* socketp);

  /**
   * Close socket assigned to the timer handle.
   *
   * @param handle Handle which is assigned to this callback.
   */
  static void closeSocketByTimer(uv_timer_t* handle);

  /**
   * Start new transfers, terminate expired transfers.
   *
   * @param req Handle which is assigned to this callback.
   */
  static void curlTimeout(uv_timer_t* req);

  /**
   * Free curl context assigned to the handle.
   *
   * @param handle Handle assigned to this callback.
   */
  static void curlCloseCB(uv_handle_t* handle);

  /**
   * Close poll handle assigned to the socket contained in the context and free data within the handle.
   *
   * @param context Structure containing information about socket and handle to be closed.
   */
  static void destroyCurlContext(curl_context_t* context);

  /**
   * Connect curl timer with uv timer.
   *
   * @param multi Multi handle for which the timeout will be set
   * @param timeout_ms Time until timeout
   * @param userp Pointer to the uv_timer_t handle that is used for timeout.
   */
  static int startTimeout(CURLM* multi, long timeout_ms, void* userp);

  /**
   * Create a new multi_handle for the downloader
   */
  void initializeMultiHandle();

  /**
   * Release resources reserver for the transfer, mark transfer as complete, passe the CURLcode to the destination and launche callbacks if it is specified in PerformData.
   *
   * @param handle The easy_handle for which the transfer completed
   * @param curlCode The code produced for the handle by the transfer
   */
  void transferFinished(CURL* handle, CURLcode curlCode);

  /**
   * Check message queue inside curl multi handle.
   */
  void checkMultiInfo();

#if !defined(__CINT__) && !defined(__MAKECINT__) && !defined(__ROOTCLING__) && !defined(__CLING__)
  /**
   * Set openSocketCallback and closeSocketCallback with appropriate arguments. Stores data inside the CURL handle.
   */
  void setHandleOptions(CURL* handle, PerformData* data);
#endif

  /**
   * Create structure holding information about a socket including a poll handle assigned to it
   *
   * @param socketfd File descriptor of socket for which the structure will be created
   */
  curl_context_t* createCurlContext(curl_socket_t sockfd);

  /**
   * If multi_handles uses less then maximum number of handles then add handles from the queue.
   */
  void checkHandleQueue();
};

/**
 * Structure assigned  to a uv_timer_t before adding it to socketTimerMap. It stores the information about the socket connected to the timer.
 */
typedef struct DataForClosingSocket {
  CCDBDownloader* CD;
  curl_socket_t socket;
} DataForClosingSocket;

} // namespace o2

#endif // O2_CCDB_CCDBDOWNLOADER_H
