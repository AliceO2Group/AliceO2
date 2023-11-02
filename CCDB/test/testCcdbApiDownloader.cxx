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

#define BOOST_TEST_MODULE CCDB
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "CommonUtils/StringUtils.h"
#include "CCDB/CCDBTimeStampUtils.h"
#include <CCDB/CCDBDownloader.h>
#include <curl/curl.h>
#include <chrono>
#include <iostream>
#include <unistd.h> // Sleep function to wait for asynch results
#include <fairlogger/Logger.h>

#include <boost/test/unit_test.hpp>
#include <boost/optional/optional.hpp>
#include <boost/asio/ip/host_name.hpp>
#include <uv.h>

#include <boost/algorithm/string.hpp>
#include "MemoryResources/MemoryResources.h"

using namespace std;

namespace o2
{
namespace ccdb
{

size_t CurlWrite_CallbackFunc_StdString2(void* contents, size_t size, size_t nmemb, std::string* s)
{
  size_t newLength = size * nmemb;
  size_t oldLength = s->size();
  try {
    s->resize(oldLength + newLength);
  } catch (std::bad_alloc& e) {
    LOG(error) << "memory error when getting data from CCDB";
    return 0;
  }

  std::copy((char*)contents, (char*)contents + newLength, s->begin() + oldLength);
  return size * nmemb;
}

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

CURL* createTestHandle(std::string* dst)
{
  CURL* handle = curl_easy_init();
  curl_easy_setopt(handle, CURLOPT_WRITEFUNCTION, CurlWrite_CallbackFunc_StdString2);
  curl_easy_setopt(handle, CURLOPT_WRITEDATA, dst);
  curl_easy_setopt(handle, CURLOPT_URL, "http://ccdb-test.cern.ch:8080/");
  auto userAgent = uniqueAgentID();
  curl_easy_setopt(handle, CURLOPT_USERAGENT, userAgent.c_str());
  return handle;
}

namespace
{
template <typename MapType = std::map<std::string, std::string>>
size_t header_map_callback(char* buffer, size_t size, size_t nitems, void* userdata)
{
  auto* headers = static_cast<MapType*>(userdata);
  auto header = std::string(buffer, size * nitems);
  std::string::size_type index = header.find(':', 0);
  if (index != std::string::npos) {
    const auto key = boost::algorithm::trim_copy(header.substr(0, index));
    const auto value = boost::algorithm::trim_copy(header.substr(index + 1));
    headers->insert(std::make_pair(key, value));
  }
  return size * nitems;
}
} // namespace

size_t writeCallbackNoLambda(void* contents, size_t size, size_t nmemb, void* chunkptr)
{
  auto& ho = *static_cast<HeaderObjectPair_t*>(chunkptr);
  auto& chunk = *ho.object;
  size_t realsize = size * nmemb, sz = 0;
  ho.counter++;
  try {
    if (chunk.capacity() < chunk.size() + realsize) {
      auto cl = ho.header.find("Content-Length");
      if (cl != ho.header.end()) {
        sz = std::max(chunk.size() + realsize, (size_t)std::stol(cl->second));
      } else {
        sz = chunk.size() + realsize;
        // LOGP(debug, "SIZE IS NOT IN HEADER, allocate {}", sz);
      }
      chunk.reserve(sz);
    }
    char* contC = (char*)contents;
    chunk.insert(chunk.end(), contC, contC + realsize);
  } catch (std::exception e) {
    // LOGP(alarm, "failed to reserve {} bytes in CURL write callback (realsize = {}): {}", sz, realsize, e.what());
    realsize = 0;
  }
  return realsize;
}

std::vector<CURL*> prepareAsyncHandles(size_t num, std::vector<o2::pmr::vector<char>*>& dests)
{
  std::vector<CURL*> handles;

  for (int i = 0; i < num; i++) {
    auto dest = new o2::pmr::vector<char>();
    dests.push_back(dest);
    CURL* curl_handle = curl_easy_init();
    handles.push_back(curl_handle);

    auto data = new DownloaderRequestData();
    data->hoPair.object = dest;
    data->hosts.push_back("http://ccdb-test.cern.ch:8080");
    data->path = "Analysis/ALICE3/Centrality";
    data->timestamp = 1646729604010;
    data->localContentCallback = nullptr;

    curl_easy_setopt(curl_handle, CURLOPT_URL, "http://ccdb-test.cern.ch:8080/Analysis/ALICE3/Centrality/1646729604010");
    curl_easy_setopt(curl_handle, CURLOPT_WRITEFUNCTION, writeCallbackNoLambda);
    curl_easy_setopt(curl_handle, CURLOPT_WRITEDATA, (void*)&(data->hoPair));

    curl_easy_setopt(curl_handle, CURLOPT_HEADERFUNCTION, header_map_callback<decltype(data->hoPair.header)>);
    curl_easy_setopt(curl_handle, CURLOPT_HEADERDATA, (void*)&(data->hoPair.header));
    curl_easy_setopt(curl_handle, CURLOPT_PRIVATE, (void*)data);
  }
  return handles;
}

BOOST_AUTO_TEST_CASE(asynch_schedule_test)
{
  int TRANSFERS = 5;

  if (curl_global_init(CURL_GLOBAL_ALL)) {
    fprintf(stderr, "Could not init curl\n");
    return;
  }

  CCDBDownloader downloader;
  std::vector<o2::pmr::vector<char>*> dests;
  auto handles = prepareAsyncHandles(TRANSFERS, dests);
  size_t transfersLeft = 0;

  for (auto handle : handles) {
    downloader.asynchSchedule(handle, &transfersLeft);
  }

  while (transfersLeft > 0) {
    downloader.runLoop(0);
  }

  for (int i = 0; i < TRANSFERS; i++) {
    long httpCode;
    curl_easy_getinfo(handles[i], CURLINFO_HTTP_CODE, &httpCode);
    BOOST_CHECK(httpCode == 200);
    BOOST_CHECK(dests[i]->size() != 0);
    curl_easy_cleanup(handles[i]);
    delete dests[i];
  }
  curl_global_cleanup();
}

BOOST_AUTO_TEST_CASE(perform_test)
{
  if (curl_global_init(CURL_GLOBAL_ALL)) {
    fprintf(stderr, "Could not init curl\n");
    return;
  }

  CCDBDownloader downloader;
  std::string dst = "";
  CURL* handle = createTestHandle(&dst);

  CURLcode curlCode = downloader.perform(handle);

  BOOST_CHECK(curlCode == CURLE_OK);
  std::cout << "CURL code: " << curlCode << "\n";

  long httpCode;
  curl_easy_getinfo(handle, CURLINFO_HTTP_CODE, &httpCode);
  BOOST_CHECK(httpCode == 200);
  std::cout << "HTTP code: " << httpCode << "\n";

  curl_easy_cleanup(handle);
  curl_global_cleanup();
}

BOOST_AUTO_TEST_CASE(blocking_batch_test)
{
  if (curl_global_init(CURL_GLOBAL_ALL)) {
    fprintf(stderr, "Could not init curl\n");
    return;
  }

  CCDBDownloader downloader;
  std::vector<CURL*> handleVector;
  std::vector<std::string*> destinations;
  for (int i = 0; i < 100; i++) {
    destinations.push_back(new std::string());
    handleVector.push_back(createTestHandle(destinations.back()));
  }

  auto curlCodes = downloader.batchBlockingPerform(handleVector);
  for (CURLcode code : curlCodes) {
    BOOST_CHECK(code == CURLE_OK);
    if (code != CURLE_OK) {
      std::cout << "CURL Code: " << code << "\n";
    }
  }

  for (CURL* handle : handleVector) {
    long httpCode;
    curl_easy_getinfo(handle, CURLINFO_HTTP_CODE, &httpCode);
    BOOST_CHECK(httpCode == 200);
    if (httpCode != 200) {
      std::cout << "HTTP Code: " << httpCode << "\n";
    }
    curl_easy_cleanup(handle);
  }

  for (std::string* dst : destinations) {
    delete dst;
  }

  curl_global_cleanup();
}

BOOST_AUTO_TEST_CASE(test_with_break)
{
  if (curl_global_init(CURL_GLOBAL_ALL)) {
    fprintf(stderr, "Could not init curl\n");
    return;
  }

  CCDBDownloader downloader;
  std::vector<CURL*> handleVector;
  std::vector<std::string*> destinations;
  for (int i = 0; i < 100; i++) {
    destinations.push_back(new std::string());
    handleVector.push_back(createTestHandle(destinations.back()));
  }

  auto curlCodes = downloader.batchBlockingPerform(handleVector);

  for (CURLcode code : curlCodes) {
    BOOST_CHECK(code == CURLE_OK);
    if (code != CURLE_OK) {
      std::cout << "CURL Code: " << code << "\n";
    }
  }

  for (CURL* handle : handleVector) {
    long httpCode;
    curl_easy_getinfo(handle, CURLINFO_HTTP_CODE, &httpCode);
    BOOST_CHECK(httpCode == 200);
    if (httpCode != 200) {
      std::cout << "HTTP Code: " << httpCode << "\n";
    }
    curl_easy_cleanup(handle);
  }

  for (std::string* dst : destinations) {
    delete dst;
  }

  sleep(10);

  std::vector<CURL*> handleVector2;
  std::vector<std::string*> destinations2;
  for (int i = 0; i < 100; i++) {
    destinations2.push_back(new std::string());
    handleVector2.push_back(createTestHandle(destinations2.back()));
  }

  auto curlCodes2 = downloader.batchBlockingPerform(handleVector2);
  for (CURLcode code : curlCodes2) {
    BOOST_CHECK(code == CURLE_OK);
    if (code != CURLE_OK) {
      std::cout << "CURL Code: " << code << "\n";
    }
  }

  for (CURL* handle : handleVector2) {
    long httpCode;
    curl_easy_getinfo(handle, CURLINFO_HTTP_CODE, &httpCode);
    BOOST_CHECK(httpCode == 200);
    if (httpCode != 200) {
      std::cout << "HTTP Code: " << httpCode << "\n";
    }
    curl_easy_cleanup(handle);
  }

  for (std::string* dst : destinations2) {
    delete dst;
  }

  curl_global_cleanup();
}

void onUVClose(uv_handle_t* handle)
{
  if (handle != nullptr) {
    delete handle;
  }
}

void closeAllHandles(uv_handle_t* handle, void* arg)
{
  if (!uv_is_closing(handle)) {
    uv_close(handle, onUVClose);
  }
}

void testTimerCB(uv_timer_t* handle)
{
  // Mock function to be used by tested timer
}

BOOST_AUTO_TEST_CASE(external_loop_test)
{
  // Prepare uv_loop to be provided to the downloader
  auto uvLoop = new uv_loop_t();
  uv_loop_init(uvLoop);

  // Prepare test timer. It will be used to check whether the downloader affects external handles.
  auto testTimer = new uv_timer_t();
  uv_timer_init(uvLoop, testTimer);
  uv_timer_start(testTimer, testTimerCB, 10, 10);

  if (curl_global_init(CURL_GLOBAL_ALL)) {
    fprintf(stderr, "Could not init curl\n");
    return;
  }

  // Regular downloader test
  auto downloader = new o2::ccdb::CCDBDownloader(uvLoop);
  std::string dst = "";
  CURL* handle = createTestHandle(&dst);

  CURLcode curlCode = downloader->perform(handle);

  BOOST_CHECK(curlCode == CURLE_OK);

  long httpCode;
  curl_easy_getinfo(handle, CURLINFO_HTTP_CODE, &httpCode);
  BOOST_CHECK(httpCode == 200);

  curl_easy_cleanup(handle);
  curl_global_cleanup();

  // Check if test timer and external loop are still alive
  BOOST_CHECK(uv_is_active((uv_handle_t*)testTimer) != 0);
  BOOST_CHECK(uv_loop_alive(uvLoop) != 0);

  // Downloader must be closed before uv_loop.
  // The reason for that are the uv_poll handles attached to the curl multi handle.
  // The multi handle must be cleaned (via destuctor) before poll handles attached to them are removed (via walking and closing).
  delete downloader;
  while (uv_loop_alive(uvLoop) && uv_loop_close(uvLoop) == UV_EBUSY) {
    uv_walk(uvLoop, closeAllHandles, nullptr);
    uv_run(uvLoop, UV_RUN_ONCE);
  }
  delete uvLoop;
}

} // namespace ccdb
} // namespace o2
