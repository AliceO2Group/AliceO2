// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

///
/// \file   CcdbApi.cxx
/// \author Barthelemy von Haller
///

#include "CCDB/CcdbApi.h"
#include <regex>
#include <chrono>
#include <TMessage.h>
#include <sstream>
#include <CommonUtils/StringUtils.h>

namespace o2
{
namespace ccdb
{

using namespace std;

CcdbApi::CcdbApi() : mUrl("")
{
}

CcdbApi::~CcdbApi()
{
  curl_global_cleanup();
}

void CcdbApi::curlInit()
{
  // todo : are there other things to initialize globally for curl ?
  curl_global_init(CURL_GLOBAL_DEFAULT);
}

void CcdbApi::init(std::string host)
{
  mUrl = host;
  curlInit();
}

/**
 * Keep only the alphanumeric characters plus '_' plus '/' from the string passed in argument.
 * @param objectName
 * @return a new string following the rule enounced above.
 */
std::string sanitizeObjectName(const std::string& objectName)
{
  string tmpObjectName = objectName;
  tmpObjectName.erase(std::remove_if(tmpObjectName.begin(), tmpObjectName.end(),
                                     [](auto const& c) -> bool { return (!std::isalnum(c) && c != '_' && c != '/'); }),
                      tmpObjectName.end());
  return tmpObjectName;
}

void CcdbApi::store(TObject* rootObject, std::string path, std::map<std::string, std::string> metadata,
                    long startValidityTimestamp, long endValidityTimestamp)
{
  // Serialize the object
  TMessage message(kMESS_OBJECT);
  message.Reset();
  message.WriteObjectAny(rootObject, rootObject->IsA());

  // Prepare
  long sanitizedStartValidityTimestamp = startValidityTimestamp;
  if (startValidityTimestamp == -1) {
    cout << "Start of Validity not set, current timestamp used." << endl;
    sanitizedStartValidityTimestamp = getCurrentTimestamp();
  }
  long sanitizedEndValidityTimestamp = endValidityTimestamp;
  if (endValidityTimestamp == -1) {
    cout << "End of Validity not set, start of validity plus 1 year used." << endl;
    sanitizedEndValidityTimestamp = getFutureTimestamp(60 * 60 * 24 * 365);
  }
  string sanitizedPath = sanitizeObjectName(path);
  string fullUrl = getFullUrlForStorage(sanitizedPath, metadata, sanitizedStartValidityTimestamp, sanitizedEndValidityTimestamp);

  // Curl preparation
  CURL* curl;
  struct curl_httppost* formpost = nullptr;
  struct curl_httppost* lastptr = nullptr;
  struct curl_slist* headerlist = nullptr;
  static const char buf[] = "Expect:";
  string objectName = string(rootObject->GetName());
  string tmpFileName = sanitizeObjectName(objectName) + "_" + getTimestampString(getCurrentTimestamp()) + ".root";
  curl_formadd(&formpost,
               &lastptr,
               CURLFORM_COPYNAME, "send",
               CURLFORM_BUFFER, tmpFileName.c_str(),
               CURLFORM_BUFFERPTR, message.Buffer(),
               CURLFORM_BUFFERLENGTH, message.Length(),
               CURLFORM_END);

  curl = curl_easy_init();
  headerlist = curl_slist_append(headerlist, buf);
  if (curl != nullptr) {
    /* what URL that receives this POST */
    curl_easy_setopt(curl, CURLOPT_URL, fullUrl.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headerlist);
    curl_easy_setopt(curl, CURLOPT_HTTPPOST, formpost);

    /* Perform the request, res will get the return code */
    CURLcode res = curl_easy_perform(curl);
    /* Check for errors */
    if (res != CURLE_OK) {
      fprintf(stderr, "curl_easy_perform() failed: %s\n",
              curl_easy_strerror(res));
    }

    /* always cleanup */
    curl_easy_cleanup(curl);

    /* then cleanup the formpost chain */
    curl_formfree(formpost);
    /* free slist */
    curl_slist_free_all(headerlist);
  }
}

string CcdbApi::getFullUrlForStorage(const string& path, const map<string, string>& metadata,
                                     long startValidityTimestamp, long endValidityTimestamp)
{
  // Prepare timestamps
  string startValidityString = getTimestampString(startValidityTimestamp < 0 ? getCurrentTimestamp() : startValidityTimestamp);
  string endValidityString = getTimestampString(endValidityTimestamp < 0 ? getFutureTimestamp(60 * 60 * 24 * 365) : endValidityTimestamp);
  // Build URL
  string fullUrl = mUrl + "/" + path + "/" + startValidityString + "/" + endValidityString + "/";
  // Add metadata
  for (auto& kv : metadata) {
    fullUrl += kv.first + "=" + kv.second + "/";
  }
  return fullUrl;
}

// todo make a single method of the one above and below
string CcdbApi::getFullUrlForRetrieval(const string& path, const map<string, string>& metadata, long timestamp)
{
  // Prepare timestamps
  string validityString = getTimestampString(timestamp < 0 ? getCurrentTimestamp() : timestamp);
  // Build URL
  string fullUrl = mUrl + "/" + path + "/" + validityString + "/";
  // Add metadata
  for (auto& kv : metadata) {
    fullUrl += kv.first + "=" + kv.second + "/";
  }
  return fullUrl;
}

/**
 * Struct to store the data we will receive from the CCDB with CURL.
 */
struct MemoryStruct {
  char* memory;
  unsigned int size;
};

/**
 * Callback used by CURL to store the data received from the CCDB.
 * See https://curl.haxx.se/libcurl/c/getinmemory.html
 * @param contents
 * @param size
 * @param nmemb
 * @param userp a MemoryStruct where data is stored.
 * @return the size of the data we received and stored at userp.
 */
static size_t WriteMemoryCallback(void* contents, size_t size, size_t nmemb, void* userp)
{
  size_t realsize = size * nmemb;
  auto* mem = (struct MemoryStruct*)userp;

  mem->memory = (char*)realloc(mem->memory, mem->size + realsize + 1);
  if (mem->memory == nullptr) {
    printf("not enough memory (realloc returned NULL)\n");
    return 0;
  }

  memcpy(&(mem->memory[mem->size]), contents, realsize);
  mem->size += realsize;
  mem->memory[mem->size] = 0;

  return realsize;
}

TObject* CcdbApi::retrieve(std::string path, std::map<std::string, std::string> metadata,
                           long timestamp)
{
  // Note : based on https://curl.haxx.se/libcurl/c/getinmemory.html
  // Thus it does not comply to our coding guidelines as it is a copy paste.

  string fullUrl = getFullUrlForRetrieval(path, metadata, timestamp);

  // Prepare CURL
  CURL* curl_handle;
  CURLcode res;
  struct MemoryStruct chunk {
    (char*)malloc(1) /*memory*/, 0 /*size*/
  };
  TObject* result = nullptr;

  /* init the curl session */
  curl_handle = curl_easy_init();

  /* specify URL to get */
  curl_easy_setopt(curl_handle, CURLOPT_URL, fullUrl.c_str());

  /* send all data to this function  */
  curl_easy_setopt(curl_handle, CURLOPT_WRITEFUNCTION, WriteMemoryCallback);

  /* we pass our 'chunk' struct to the callback function */
  curl_easy_setopt(curl_handle, CURLOPT_WRITEDATA, (void*)&chunk);

  /* some servers don't like requests that are made without a user-agent
     field, so we provide one */
  curl_easy_setopt(curl_handle, CURLOPT_USERAGENT, "libcurl-agent/1.0");

  /* if redirected , we tell libcurl to follow redirection */
  curl_easy_setopt(curl_handle, CURLOPT_FOLLOWLOCATION, 1L);

  /* get it! */
  res = curl_easy_perform(curl_handle);

  /* check for errors */
  if (res != CURLE_OK) {
    fprintf(stderr, "curl_easy_perform() failed: %s\n",
            curl_easy_strerror(res));
  } else {
    /*
     * Now, our chunk.memory points to a memory block that is chunk.size
     * bytes big and contains the remote file.
     */

    //    printf("%lu bytes retrieved\n", (long) chunk.size);

    long response_code;
    res = curl_easy_getinfo(curl_handle, CURLINFO_RESPONSE_CODE, &response_code);
    if ((res == CURLE_OK) && (response_code != 404)) {
      TMessage mess(kMESS_OBJECT);
      mess.SetBuffer(chunk.memory, chunk.size, kFALSE);
      mess.SetReadMode();
      mess.Reset();
      result = (TObject*)(mess.ReadObjectAny(mess.GetClass()));
      if (result == nullptr) {
        cerr << "couldn't retrieve the object " << path << endl;
      }
    } else {
      cerr << "invalid URL : " << fullUrl << endl;
    }

    // Print data
    //    cout << "size : " << chunk.size << endl;
    //    cout << "data : " << endl;
    //    char* mem = (char*)chunk.memory;
    //    for (int i = 0 ; i < chunk.size/4 ; i++)  {
    //      cout << mem;
    //      mem += 4;
    //    }
  }

  /* cleanup curl stuff */
  curl_easy_cleanup(curl_handle);

  free(chunk.memory);

  return result;
}

size_t CurlWrite_CallbackFunc_StdString2(void* contents, size_t size, size_t nmemb, std::string* s)
{
  size_t newLength = size * nmemb;
  size_t oldLength = s->size();
  try {
    s->resize(oldLength + newLength);
  } catch (std::bad_alloc& e) {
    cerr << "memory error when getting data from CCDB" << endl;
    return 0;
  }

  std::copy((char*)contents, (char*)contents + newLength, s->begin() + oldLength);
  return size * nmemb;
}

std::string CcdbApi::list(std::string path, bool latestOnly, std::string returnFormat)
{
  CURL* curl;
  CURLcode res;
  string fullUrl = mUrl;
  fullUrl += latestOnly ? "/latest/" : "/browse/";
  fullUrl += path;
  std::string result;

  curl = curl_easy_init();
  if (curl != nullptr) {

    curl_easy_setopt(curl, CURLOPT_URL, fullUrl.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, CurlWrite_CallbackFunc_StdString2);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &result);

    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, (string("Accept: ") + returnFormat).c_str());
    headers = curl_slist_append(headers, (string("Content-Type: ") + returnFormat).c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

    // Perform the request, res will get the return code
    res = curl_easy_perform(curl);
    if (res != CURLE_OK) {
      fprintf(stderr, "curl_easy_perform() failed: %s\n", curl_easy_strerror(res));
    }
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
  }

  return result;
}

long CcdbApi::getFutureTimestamp(int secondsInFuture)
{
  std::chrono::seconds sec(secondsInFuture);
  auto future = std::chrono::system_clock::now() + sec;
  auto future_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(future);
  auto epoch = future_ms.time_since_epoch();
  auto value = std::chrono::duration_cast<std::chrono::milliseconds>(epoch);
  return value.count();
}

long CcdbApi::getCurrentTimestamp()
{
  auto now = std::chrono::system_clock::now();
  auto now_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(now);
  auto epoch = now_ms.time_since_epoch();
  auto value = std::chrono::duration_cast<std::chrono::milliseconds>(epoch);
  return value.count();
}

std::string CcdbApi::getTimestampString(long timestamp)
{
  stringstream ss;
  ss << timestamp;
  return ss.str();
}

void CcdbApi::deleteObject(std::string path, long timestamp)
{
  CURL* curl;
  CURLcode res;
  stringstream fullUrl;
  long timestampLocal = timestamp == -1 ? getCurrentTimestamp() : timestamp;

  fullUrl << mUrl << "/" << path << "/" << timestampLocal;

  curl = curl_easy_init();
  if (curl != nullptr) {
    curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, "DELETE");
    curl_easy_setopt(curl, CURLOPT_URL, fullUrl.str().c_str());

    // Perform the request, res will get the return code
    res = curl_easy_perform(curl);
    if (res != CURLE_OK) {
      fprintf(stderr, "curl_easy_perform() failed: %s\n", curl_easy_strerror(res));
    }
    curl_easy_cleanup(curl);
  }
}

void CcdbApi::truncate(std::string path)
{
  CURL* curl;
  CURLcode res;
  stringstream fullUrl;
  fullUrl << mUrl << "/truncate/" << path;

  curl = curl_easy_init();
  if (curl != nullptr) {
    curl_easy_setopt(curl, CURLOPT_URL, fullUrl.str().c_str());

    // Perform the request, res will get the return code
    res = curl_easy_perform(curl);
    if (res != CURLE_OK) {
      fprintf(stderr, "curl_easy_perform() failed: %s\n", curl_easy_strerror(res));
    }
    curl_easy_cleanup(curl);
  }
}

size_t write_data(void* buffer, size_t size, size_t nmemb, void* userp)
{
  return size * nmemb;
}

bool CcdbApi::isHostReachable()
{
  CURL* curl;
  CURLcode res;
  bool result = false;

  curl = curl_easy_init();
  if (curl) {
    curl_easy_setopt(curl, CURLOPT_URL, mUrl.data());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_data);
    res = curl_easy_perform(curl);
    result = (res == CURLE_OK);

    /* always cleanup */
    curl_easy_cleanup(curl);
  }
  return result;
}

} // namespace ccdb
} // namespace o2
