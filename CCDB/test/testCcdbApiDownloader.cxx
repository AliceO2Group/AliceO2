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

#include <CCDB/CCDBDownloader.h>
#include <curl/curl.h>
#include <chrono>
#include <iostream>
#include <unistd.h> // Sleep function to wait for asynch results
#include <fairlogger/Logger.h>

#include <boost/test/unit_test.hpp>
#include <boost/optional/optional.hpp>

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

CURL* createTestHandle(std::string* dst)
{
  CURL* handle = curl_easy_init();
  curl_easy_setopt(handle, CURLOPT_WRITEFUNCTION, CurlWrite_CallbackFunc_StdString2);
  curl_easy_setopt(handle, CURLOPT_WRITEDATA, dst);
  curl_easy_setopt(handle, CURLOPT_URL, "http://ccdb-test.cern.ch:8080/latest/");
  return handle;
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

BOOST_AUTO_TEST_CASE(asynch_batch_test)
{
  if (curl_global_init(CURL_GLOBAL_ALL)) {
    fprintf(stderr, "Could not init curl\n");
    return;
  }

  CCDBDownloader downloader;
  std::vector<CURL*> handleVector;
  std::vector<std::string*> destinations;
  for (int i = 0; i < 10; i++) {
    destinations.push_back(new std::string());
    handleVector.push_back(createTestHandle(destinations.back()));
  }

  bool flag = false;
  auto curlCodes = downloader.batchAsynchPerform(handleVector, &flag);
  while (!flag) {
    sleep(1);
  }

  for (CURLcode code : (*curlCodes)) {
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

void testCallback(void* ptr)
{
  int* intPtr = (int*)ptr;
  *intPtr = 46;
}

BOOST_AUTO_TEST_CASE(asynch_batch_callback)
{
  if (curl_global_init(CURL_GLOBAL_ALL)) {
    fprintf(stderr, "Could not init curl\n");
    return;
  }

  CCDBDownloader downloader;
  std::vector<CURL*> handleVector;
  std::vector<std::string*> destinations;
  for (int i = 0; i < 10; i++) {
    destinations.push_back(new std::string());
    handleVector.push_back(createTestHandle(destinations.back()));
  }

  int testValue = 0;

  bool flag = false;
  auto curlCodes = downloader.asynchBatchPerformWithCallback(handleVector, &flag, testCallback, &testValue);
  while (!flag) {
    sleep(1);
  }

  BOOST_CHECK(testValue == 46);

  for (CURLcode code : (*curlCodes)) {
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

BOOST_AUTO_TEST_CASE(external_loop_test)
{
  if (curl_global_init(CURL_GLOBAL_ALL)) {
    fprintf(stderr, "Could not init curl\n");
    return;
  }

  uv_loop_t loop;

  CCDBDownloader downloader(&loop);
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

} // namespace ccdb
} // namespace o2
