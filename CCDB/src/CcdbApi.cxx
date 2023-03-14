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

///
/// \file   CcdbApi.cxx
/// \author Barthelemy von Haller, Sandro Wenzel
///

#include "CCDB/CcdbApi.h"
#include "CCDB/CCDBQuery.h"

#include "CommonUtils/StringUtils.h"
#include "CommonUtils/FileSystemUtils.h"
#include "CommonUtils/MemFileHelper.h"
#include "MemoryResources/MemoryResources.h"
#include <chrono>
#include <memory>
#include <sstream>
#include <TFile.h>
#include <TGrid.h>
#include <TSystem.h>
#include <TStreamerInfo.h>
#include <TMemFile.h>
#include <TH1F.h>
#include <TTree.h>
#include <fairlogger/Logger.h>
#include <TError.h>
#include <TClass.h>
#include <CCDB/CCDBTimeStampUtils.h>
#include <algorithm>
#include <filesystem>
#include <boost/algorithm/string.hpp>
#include <boost/asio/ip/host_name.hpp>
#include <iostream>
#include <mutex>
#include <boost/interprocess/sync/named_semaphore.hpp>
#include <regex>
#include <cstdio>

namespace o2::ccdb
{

using namespace std;

std::mutex gIOMutex; // to protect TMemFile IO operations
unique_ptr<TJAlienCredentials> CcdbApi::mJAlienCredentials = nullptr;

CcdbApi::CcdbApi()
{
  std::string host = boost::asio::ip::host_name();
  mUniqueAgentID = fmt::format("{}-{}-{}", host, getCurrentTimestamp() / 1000, o2::utils::Str::getRandomString(6));

  mIsCCDBDownloaderEnabled = getenv("ALICEO2_ENABLE_MULTIHANDLE_CCDBAPI") && atoi(getenv("ALICEO2_ENABLE_MULTIHANDLE_CCDBAPI"));
  if (mIsCCDBDownloaderEnabled) {
    mDownloader = new CCDBDownloader();
  }
}

CcdbApi::~CcdbApi()
{
  curl_global_cleanup();
  if (mDownloader) {
    delete mDownloader;
  }
}

bool CcdbApi::checkAlienToken()
{
#ifdef __APPLE__
  LOG(debug) << "On macOS we simply rely on TGrid::Connect(\"alien\").";
  return true;
#endif
  if (getenv("ALICEO2_CCDB_NOTOKENCHECK") && atoi(getenv("ALICEO2_CCDB_NOTOKENCHECK"))) {
    return true;
  }
  if (getenv("JALIEN_TOKEN_CERT")) {
    return true;
  }
  auto returncode = system("LD_PRELOAD= alien-token-info &> /dev/null");
  if (returncode == -1) {
    LOG(error) << "...";
  }
  return returncode == 0;
}

void CcdbApi::curlInit()
{
  // todo : are there other things to initialize globally for curl ?
  curl_global_init(CURL_GLOBAL_DEFAULT);
  CcdbApi::mJAlienCredentials = std::make_unique<TJAlienCredentials>();
  CcdbApi::mJAlienCredentials->loadCredentials();
  CcdbApi::mJAlienCredentials->selectPreferedCredentials();

  // allow to configure the socket timeout of CCDBDownloader, if activated (for some tuning studies)
  if (mIsCCDBDownloaderEnabled) {
    if (getenv("ALICEO2_CCDB_SOCKET_TIMEOUT")) {
      auto timeoutMS = atoi(getenv("ALICEO2_CCDB_SOCKET_TIMEOUT"));
      if (timeoutMS >= 0) {
        LOG(info) << "Setting socket timeout to " << timeoutMS << " milliseconds";
        mDownloader->setSocketTimoutTime(timeoutMS);
      }
    }
  }
}

void CcdbApi::init(std::string const& host)
{
  // if host is prefixed with "file://" this is a local snapshot
  // in this case we init the API in snapshot (readonly) mode
  constexpr const char* SNAPSHOTPREFIX = "file://";
  mUrl = host;

  if (host.substr(0, 7).compare(SNAPSHOTPREFIX) == 0) {
    auto path = host.substr(7);
    initInSnapshotMode(path);
  } else {
    initHostsPool(host);
    curlInit();
  }
  // The environment option ALICEO2_CCDB_LOCALCACHE allows
  // to reduce the number of queries to the server, by collecting the objects in a local
  // cache folder, and serving from this folder for repeated queries.
  // This is useful for instance for MC GRID productions in which we spawn
  // many isolated processes, all querying the CCDB (for potentially the same objects and same timestamp).
  // In addition, we can monitor exactly which objects are fetched and what is their content.
  // One can also distribute so obtained caches to sites without network access.
  //
  // When used with the DPL CCDB fetcher (i.e. loadFileToMemory is called), in order to prefer the available snapshot w/o its validity
  // check an extra variable IGNORE_VALIDITYCHECK_OF_CCDB_LOCALCACHE must be defined, otherwhise the object will be fetched from the
  // server after the validity check and new snapshot will be created if needed

  std::string snapshotReport{};
  const char* cachedir = getenv("ALICEO2_CCDB_LOCALCACHE");
  if (cachedir) {
    if (cachedir[0] == 0) {
      mSnapshotCachePath = ".";
    } else {
      mSnapshotCachePath = cachedir;
    }
    snapshotReport = fmt::format("(cache snapshots to dir={}", mSnapshotCachePath);
  }
  if (getenv("IGNORE_VALIDITYCHECK_OF_CCDB_LOCALCACHE")) {
    mPreferSnapshotCache = true;
    if (mSnapshotCachePath.empty()) {
      LOGP(fatal, "IGNORE_VALIDITYCHECK_OF_CCDB_LOCALCACHE is defined but the ALICEO2_CCDB_LOCALCACHE is not");
    }
    snapshotReport += ", prefer if available";
  }
  if (!snapshotReport.empty()) {
    snapshotReport += ')';
  }

  mNeedAlienToken = (host.find("https://") != std::string::npos) || (host.find("alice-ccdb.cern.ch") != std::string::npos);

  LOGP(info, "Init CcdApi with UserAgentID: {}, Host: {}{}", mUniqueAgentID, host,
       mInSnapshotMode ? "(snapshot readonly mode)" : snapshotReport.c_str());
}

/**
 * Keep only the alphanumeric characters plus '_' plus '/' plus '.' from the string passed in argument.
 * @param objectName
 * @return a new string following the rule enounced above.
 */
std::string sanitizeObjectName(const std::string& objectName)
{
  string tmpObjectName = objectName;
  tmpObjectName.erase(std::remove_if(tmpObjectName.begin(), tmpObjectName.end(),
                                     [](auto const& c) -> bool { return (!std::isalnum(c) && c != '_' && c != '/' && c != '.'); }),
                      tmpObjectName.end());
  return tmpObjectName;
}

std::unique_ptr<std::vector<char>> CcdbApi::createObjectImage(const void* obj, std::type_info const& tinfo, CcdbObjectInfo* info)
{
  // Create a binary image of the object, if CcdbObjectInfo pointer is provided, register there
  // the assigned object class name and the filename
  std::lock_guard<std::mutex> guard(gIOMutex);
  std::string className = o2::utils::MemFileHelper::getClassName(tinfo);
  std::string tmpFileName = generateFileName(className);
  if (info) {
    info->setFileName(tmpFileName);
    info->setObjectType(className);
  }
  return o2::utils::MemFileHelper::createFileImage(obj, tinfo, tmpFileName, CCDBOBJECT_ENTRY);
}

std::unique_ptr<std::vector<char>> CcdbApi::createObjectImage(const TObject* rootObject, CcdbObjectInfo* info)
{
  // Create a binary image of the object, if CcdbObjectInfo pointer is provided, register there
  // the assigned object class name and the filename
  std::string className = rootObject->GetName();
  std::string tmpFileName = generateFileName(className);
  if (info) {
    info->setFileName(tmpFileName);
    info->setObjectType("TObject"); // why TObject and not the actual name?
  }
  std::lock_guard<std::mutex> guard(gIOMutex);
  return o2::utils::MemFileHelper::createFileImage(*rootObject, tmpFileName, CCDBOBJECT_ENTRY);
}

int CcdbApi::storeAsTFile_impl(const void* obj, std::type_info const& tinfo, std::string const& path,
                               std::map<std::string, std::string> const& metadata,
                               long startValidityTimestamp, long endValidityTimestamp,
                               std::vector<char>::size_type maxSize) const
{
  // We need the TClass for this type; will verify if dictionary exists
  CcdbObjectInfo info;
  auto img = createObjectImage(obj, tinfo, &info);
  return storeAsBinaryFile(img->data(), img->size(), info.getFileName(), info.getObjectType(),
                           path, metadata, startValidityTimestamp, endValidityTimestamp, maxSize);
}

int CcdbApi::storeAsBinaryFile(const char* buffer, size_t size, const std::string& filename, const std::string& objectType,
                               const std::string& path, const std::map<std::string, std::string>& metadata,
                               long startValidityTimestamp, long endValidityTimestamp, std::vector<char>::size_type maxSize) const
{
  if (maxSize > 0 && size > maxSize) {
    LOGP(alarm, "Object will not be uploaded to {} since its size {} exceeds max allowed {}", path, size, maxSize);
    return -1;
  }
  int returnValue = 0;

  // Prepare URL
  long sanitizedStartValidityTimestamp = startValidityTimestamp;
  if (startValidityTimestamp == -1) {
    LOGP(info, "Start of Validity not set, current timestamp used.");
    sanitizedStartValidityTimestamp = getCurrentTimestamp();
  }
  long sanitizedEndValidityTimestamp = endValidityTimestamp;
  if (endValidityTimestamp == -1) {
    LOGP(info, "End of Validity not set, start of validity plus 1 day used.");
    sanitizedEndValidityTimestamp = getFutureTimestamp(60 * 60 * 24 * 1);
  }
  if (mInSnapshotMode) { // write local file
    auto pthLoc = getSnapshotDir(mSnapshotTopPath, path);
    o2::utils::createDirectoriesIfAbsent(pthLoc);
    auto flLoc = getSnapshotFile(mSnapshotTopPath, path, filename);
    // add the timestamps to the end
    auto pent = flLoc.find_last_of('.');
    if (pent == std::string::npos) {
      pent = flLoc.size();
    }
    flLoc.insert(pent, fmt::format("_{}_{}", startValidityTimestamp, endValidityTimestamp));
    ofstream outf(flLoc.c_str(), ios::out | ios::binary);
    outf.write(buffer, size);
    outf.close();
    if (!outf.good()) {
      throw std::runtime_error(fmt::format("Failed to write local CCDB file {}", flLoc));
    } else {
      std::string metaStr{};
      for (const auto& mentry : metadata) {
        metaStr += fmt::format("{}={};", mentry.first, mentry.second);
      }
      metaStr += "$USER_META;";
      LOGP(info, "Created local snapshot {}", flLoc);
      LOGP(info, R"(Upload with: o2-ccdb-upload --host "$ccdbhost" -p {} -f {} -k {} --starttimestamp {} --endtimestamp {} -m "{}")",
           path, flLoc, CCDBOBJECT_ENTRY, startValidityTimestamp, endValidityTimestamp, metaStr);
    }
    return returnValue;
  }

  // Curl preparation
  CURL* curl = nullptr;
  curl = curl_easy_init();

  // checking that all metadata keys do not contain invalid characters
  checkMetadataKeys(metadata);

  if (curl != nullptr) {
    struct curl_httppost* formpost = nullptr;
    struct curl_httppost* lastptr = nullptr;
    curl_formadd(&formpost,
                 &lastptr,
                 CURLFORM_COPYNAME, "send",
                 CURLFORM_BUFFER, filename.c_str(),
                 CURLFORM_BUFFERPTR, buffer, //.Buffer(),
                 CURLFORM_BUFFERLENGTH, size,
                 CURLFORM_END);

    struct curl_slist* headerlist = nullptr;
    static const char buf[] = "Expect:";
    headerlist = curl_slist_append(headerlist, buf);

    curlSetSSLOptions(curl);

    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headerlist);
    curl_easy_setopt(curl, CURLOPT_HTTPPOST, formpost);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, mUniqueAgentID.c_str());

    CURLcode res = CURL_LAST;

    for (size_t hostIndex = 0; hostIndex < hostsPool.size() && res > 0; hostIndex++) {
      string fullUrl = getFullUrlForStorage(curl, path, objectType, metadata, sanitizedStartValidityTimestamp, sanitizedEndValidityTimestamp, hostIndex);
      LOG(debug3) << "Full URL Encoded: " << fullUrl;
      /* what URL that receives this POST */
      curl_easy_setopt(curl, CURLOPT_URL, fullUrl.c_str());

      /* Perform the request, res will get the return code */
      res = CURL_perform(curl);
      /* Check for errors */
      if (res != CURLE_OK) {
        LOGP(alarm, "curl_easy_perform() failed: {}", curl_easy_strerror(res));
        returnValue = res;
      }
    }

    /* always cleanup */
    curl_easy_cleanup(curl);

    /* then cleanup the formpost chain */
    curl_formfree(formpost);
    /* free slist */
    curl_slist_free_all(headerlist);
  } else {
    LOGP(alarm, "curl initialization failure");
    returnValue = -2;
  }
  return returnValue;
}

int CcdbApi::storeAsTFile(const TObject* rootObject, std::string const& path, std::map<std::string, std::string> const& metadata,
                          long startValidityTimestamp, long endValidityTimestamp, std::vector<char>::size_type maxSize) const
{
  // Prepare file
  CcdbObjectInfo info;
  auto img = createObjectImage(rootObject, &info);
  return storeAsBinaryFile(img->data(), img->size(), info.getFileName(), info.getObjectType(), path, metadata, startValidityTimestamp, endValidityTimestamp, maxSize);
}

string CcdbApi::getFullUrlForStorage(CURL* curl, const string& path, const string& objtype,
                                     const map<string, string>& metadata,
                                     long startValidityTimestamp, long endValidityTimestamp, int hostIndex) const
{
  // Prepare timestamps
  string startValidityString = getTimestampString(startValidityTimestamp < 0 ? getCurrentTimestamp() : startValidityTimestamp);
  string endValidityString = getTimestampString(endValidityTimestamp < 0 ? getFutureTimestamp(60 * 60 * 24 * 1) : endValidityTimestamp);
  // Get url
  string url = getHostUrl(hostIndex);
  // Build URL
  string fullUrl = url + "/" + path + "/" + startValidityString + "/" + endValidityString + "/";
  // Add type as part of metadata
  // we need to URL encode the object type, since in case it has special characters (like the "<", ">" for templated classes) it won't work otherwise
  char* objtypeEncoded = curl_easy_escape(curl, objtype.c_str(), objtype.size());
  fullUrl += "ObjectType=" + string(objtypeEncoded) + "/";
  curl_free(objtypeEncoded);
  // Add general metadata
  for (auto& kv : metadata) {
    string mfirst = kv.first;
    string msecond = kv.second;
    // same trick for the metadata as for the object type
    char* mfirstEncoded = curl_easy_escape(curl, mfirst.c_str(), mfirst.size());
    char* msecondEncoded = curl_easy_escape(curl, msecond.c_str(), msecond.size());
    fullUrl += string(mfirstEncoded) + "=" + string(msecondEncoded) + "/";
    curl_free(mfirstEncoded);
    curl_free(msecondEncoded);
  }
  return fullUrl;
}

// todo make a single method of the one above and below
string CcdbApi::getFullUrlForRetrieval(CURL* curl, const string& path, const map<string, string>& metadata, long timestamp, int hostIndex) const
{
  if (mInSnapshotMode) {
    return getSnapshotFile(mSnapshotTopPath, path);
  }

  // Prepare timestamps
  string validityString = getTimestampString(timestamp < 0 ? getCurrentTimestamp() : timestamp);
  // Get host url
  string hostUrl = getHostUrl(hostIndex);
  // Build URL
  string fullUrl = hostUrl + "/" + path + "/" + validityString + "/";
  // Add metadata
  for (auto& kv : metadata) {
    string mfirst = kv.first;
    string msecond = kv.second;
    // trick for the metadata in case it contains special characters
    char* mfirstEncoded = curl_easy_escape(curl, mfirst.c_str(), mfirst.size());
    char* msecondEncoded = curl_easy_escape(curl, msecond.c_str(), msecond.size());
    fullUrl += string(mfirstEncoded) + "=" + string(msecondEncoded) + "/";
    curl_free(mfirstEncoded);
    curl_free(msecondEncoded);
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

/**
 * Callback used by CURL to store the data received from the CCDB
 * directly into a binary file
 * @param contents
 * @param size
 * @param nmemb
 * @param userp a MemoryStruct where data is stored.
 * @return the size of the data we received and stored at userp.
 * If an error is returned no attempt to establish a connection is made
 * and the perform operation will return the callback's error code
 */
static size_t WriteToFileCallback(void* ptr, size_t size, size_t nmemb, FILE* stream)
{
  size_t written = fwrite(ptr, size, nmemb, stream);
  return written;
}

/**
 * Callback to load credentials and CA's
 * @param curl curl handler
 * @param ssl_ctx SSL context that will be modified
 * @param parm
 * @return
 */
static CURLcode ssl_ctx_callback(CURL*, void*, void* parm)
{
  std::string msg((const char*)parm);
  int start = 0, end = msg.find('\n');

  if (msg.length() > 0 && end == -1) {
    LOG(warn) << msg;
  } else if (end > 0) {
    while (end > 0) {
      LOG(warn) << msg.substr(start, end - start);
      start = end + 1;
      end = msg.find('\n', start);
    }
  }
  return CURLE_OK;
}

void CcdbApi::curlSetSSLOptions(CURL* curl_handle)
{
  CredentialsKind cmk = mJAlienCredentials->getPreferedCredentials();

  /* NOTE: return early, the warning should be printed on SSL callback if needed */
  if (cmk == cNOT_FOUND) {
    return;
  }

  TJAlienCredentialsObject cmo = mJAlienCredentials->get(cmk);

  char* CAPath = getenv("X509_CERT_DIR");
  if (CAPath) {
    curl_easy_setopt(curl_handle, CURLOPT_CAPATH, CAPath);
  }
  curl_easy_setopt(curl_handle, CURLOPT_CAINFO, nullptr);
  curl_easy_setopt(curl_handle, CURLOPT_SSLCERT, cmo.certpath.c_str());
  curl_easy_setopt(curl_handle, CURLOPT_SSLKEY, cmo.keypath.c_str());

  // NOTE: for lazy logging only
  curl_easy_setopt(curl_handle, CURLOPT_SSL_CTX_FUNCTION, ssl_ctx_callback);
  curl_easy_setopt(curl_handle, CURLOPT_SSL_CTX_DATA, mJAlienCredentials->getMessages().c_str());

  // CURLcode ret = curl_easy_setopt(curl_handle, CURLOPT_SSL_CTX_FUNCTION, *ssl_ctx_callback);
}

using CurlWriteCallback = size_t (*)(void*, size_t, size_t, void*);

void CcdbApi::initCurlOptionsForRetrieve(CURL* curlHandle, void* chunk, CurlWriteCallback writeCallback, bool followRedirect) const
{
  curl_easy_setopt(curlHandle, CURLOPT_WRITEFUNCTION, writeCallback);
  curl_easy_setopt(curlHandle, CURLOPT_WRITEDATA, chunk);
  curl_easy_setopt(curlHandle, CURLOPT_FOLLOWLOCATION, followRedirect ? 1L : 0L);
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

void CcdbApi::initHeadersForRetrieve(CURL* curlHandle, long timestamp, std::map<std::string, std::string>* headers, std::string const& etag,
                                     const std::string& createdNotAfter, const std::string& createdNotBefore) const
{
  struct curl_slist* list = nullptr;
  if (!etag.empty()) {
    list = curl_slist_append(list, ("If-None-Match: " + etag).c_str());
  }

  if (!createdNotAfter.empty()) {
    list = curl_slist_append(list, ("If-Not-After: " + createdNotAfter).c_str());
  }

  if (!createdNotBefore.empty()) {
    list = curl_slist_append(list, ("If-Not-Before: " + createdNotBefore).c_str());
  }

  if (headers != nullptr) {
    list = curl_slist_append(list, ("If-None-Match: " + to_string(timestamp)).c_str());
    curl_easy_setopt(curlHandle, CURLOPT_HEADERFUNCTION, header_map_callback<>);
    curl_easy_setopt(curlHandle, CURLOPT_HEADERDATA, headers);
  }

  if (list) {
    curl_easy_setopt(curlHandle, CURLOPT_HTTPHEADER, list);
  }

  curl_easy_setopt(curlHandle, CURLOPT_USERAGENT, mUniqueAgentID.c_str());
}

bool CcdbApi::receiveToFile(FILE* fileHandle, std::string const& path, std::map<std::string, std::string> const& metadata,
                            long timestamp, std::map<std::string, std::string>* headers, std::string const& etag,
                            const std::string& createdNotAfter, const std::string& createdNotBefore, bool followRedirect) const
{
  return receiveObject((void*)fileHandle, path, metadata, timestamp, headers, etag, createdNotAfter, createdNotBefore, followRedirect, (CurlWriteCallback)&WriteToFileCallback);
}

bool CcdbApi::receiveToMemory(void* chunk, std::string const& path, std::map<std::string, std::string> const& metadata,
                              long timestamp, std::map<std::string, std::string>* headers, std::string const& etag,
                              const std::string& createdNotAfter, const std::string& createdNotBefore, bool followRedirect) const
{
  return receiveObject((void*)chunk, path, metadata, timestamp, headers, etag, createdNotAfter, createdNotBefore, followRedirect, (CurlWriteCallback)&WriteMemoryCallback);
}

bool CcdbApi::receiveObject(void* dataHolder, std::string const& path, std::map<std::string, std::string> const& metadata,
                            long timestamp, std::map<std::string, std::string>* headers, std::string const& etag,
                            const std::string& createdNotAfter, const std::string& createdNotBefore, bool followRedirect, CurlWriteCallback writeCallback) const
{
  CURL* curlHandle;

  curlHandle = curl_easy_init();

  if (curlHandle != nullptr) {

    curlSetSSLOptions(curlHandle);
    initCurlOptionsForRetrieve(curlHandle, dataHolder, writeCallback, followRedirect);
    initHeadersForRetrieve(curlHandle, timestamp, headers, etag, createdNotAfter, createdNotBefore);

    long responseCode = 0;
    CURLcode curlResultCode = CURL_LAST;

    for (size_t hostIndex = 0; hostIndex < hostsPool.size() && (responseCode >= 400 || curlResultCode > 0); hostIndex++) {
      string fullUrl = getFullUrlForRetrieval(curlHandle, path, metadata, timestamp, hostIndex);
      curl_easy_setopt(curlHandle, CURLOPT_URL, fullUrl.c_str());

      curlResultCode = CURL_perform(curlHandle);

      if (curlResultCode != CURLE_OK) {
        LOGP(alarm, "curl_easy_perform() failed: {}", curl_easy_strerror(curlResultCode));
      } else {
        curlResultCode = curl_easy_getinfo(curlHandle, CURLINFO_RESPONSE_CODE, &responseCode);
        if ((curlResultCode == CURLE_OK) && (responseCode < 300)) {
          curl_easy_cleanup(curlHandle);
          return true;
        } else {
          if (curlResultCode != CURLE_OK) {
            LOGP(alarm, "invalid URL {}", fullUrl);
          } else {
            LOGP(alarm, "not found under link {}", fullUrl);
          }
        }
      }
    }

    curl_easy_cleanup(curlHandle);
  }
  return false;
}

TObject* CcdbApi::retrieve(std::string const& path, std::map<std::string, std::string> const& metadata,
                           long timestamp) const
{
  struct MemoryStruct chunk {
    (char*)malloc(1) /*memory*/, 0 /*size*/
  };

  TObject* result = nullptr;

  bool res = receiveToMemory((void*)&chunk, path, metadata, timestamp);

  if (res) {
    std::lock_guard<std::mutex> guard(gIOMutex);
    TMessage mess(kMESS_OBJECT);
    mess.SetBuffer(chunk.memory, chunk.size, kFALSE);
    mess.SetReadMode();
    mess.Reset();
    result = (TObject*)(mess.ReadObjectAny(mess.GetClass()));
    if (result == nullptr) {
      LOGP(info, "couldn't retrieve the object {}", path);
    }
  }

  free(chunk.memory);

  return result;
}

std::string CcdbApi::generateFileName(const std::string& inp)
{
  // generate file name for the CCDB object  (for now augment the input string by the timestamp)
  std::string str = inp;
  str.erase(std::remove_if(str.begin(), str.end(), ::isspace), str.end());
  str = std::regex_replace(str, std::regex("::"), "-");
  str += "_" + std::to_string(o2::ccdb::getCurrentTimestamp()) + ".root";
  return str;
}

TObject* CcdbApi::retrieveFromTFile(std::string const& path, std::map<std::string, std::string> const& metadata,
                                    long timestamp, std::map<std::string, std::string>* headers, std::string const& etag,
                                    const std::string& createdNotAfter, const std::string& createdNotBefore) const
{
  return (TObject*)retrieveFromTFile(typeid(TObject), path, metadata, timestamp, headers, etag, createdNotAfter, createdNotBefore);
}

bool CcdbApi::retrieveBlob(std::string const& path, std::string const& targetdir, std::map<std::string, std::string> const& metadata,
                           long timestamp, bool preservePath, std::string const& localFileName, std::string const& createdNotAfter, std::string const& createdNotBefore) const
{

  // we setup the target path for this blob
  std::string fulltargetdir = targetdir + (preservePath ? ('/' + path) : "");

  try {
    o2::utils::createDirectoriesIfAbsent(fulltargetdir);
  } catch (std::exception e) {
    LOGP(error, fmt::format("Could not create local snapshot cache directory {}, reason: {}", fulltargetdir, e.what()));
    return false;
  }

  o2::pmr::vector<char> buff;
  std::map<std::string, std::string> headers;
  // avoid creating snapshot via loadFileToMemory itself
  loadFileToMemory(buff, path, metadata, timestamp, &headers, "", createdNotAfter, createdNotBefore, false);
  if ((headers.count("Error") != 0) || (buff.empty())) {
    LOGP(error, "Unable to find object {}/{}, Aborting", path, timestamp);
    return false;
  }
  // determine local filename --> use user given one / default -- or if empty string determine from content
  auto getFileName = [&headers]() {
    auto& s = headers["Content-Disposition"];
    if (s != "") {
      std::regex re("(.*;)filename=\"(.*)\"");
      std::cmatch m;
      if (std::regex_match(s.c_str(), m, re)) {
        return m[2].str();
      }
    }
    std::string backupname("ccdb-blob.bin");
    LOG(error) << "Cannot determine original filename from Content-Disposition ... falling back to " << backupname;
    return backupname;
  };
  auto filename = localFileName.size() > 0 ? localFileName : getFileName();
  std::string targetpath = fulltargetdir + "/" + filename;
  {
    std::ofstream objFile(targetpath, std::ios::out | std::ofstream::binary);
    std::copy(buff.begin(), buff.end(), std::ostreambuf_iterator<char>(objFile));
    if (!objFile.good()) {
      LOGP(error, "Unable to open local file {}, Aborting", targetpath);
      return false;
    }
  }
  CCDBQuery querysummary(path, metadata, timestamp);
  {
    std::lock_guard<std::mutex> guard(gIOMutex);
    auto oldlevel = gErrorIgnoreLevel;
    gErrorIgnoreLevel = 6001; // ignoring error messages here (since we catch with IsZombie)
    TFile snapshotfile(targetpath.c_str(), "UPDATE");
    // The assumption is that the blob is a ROOT file
    if (!snapshotfile.IsZombie()) {
      snapshotfile.WriteObjectAny(&querysummary, TClass::GetClass(typeid(querysummary)), CCDBQUERY_ENTRY);
      snapshotfile.WriteObjectAny(&headers, TClass::GetClass(typeid(metadata)), CCDBMETA_ENTRY);
      snapshotfile.Close();
    }
    gErrorIgnoreLevel = oldlevel;
  }
  return true;
}

void CcdbApi::snapshot(std::string const& ccdbrootpath, std::string const& localDir, long timestamp) const
{
  // query all subpaths to ccdbrootpath
  const auto allfolders = getAllFolders(ccdbrootpath);
  std::map<string, string> metadata;
  for (auto& folder : allfolders) {
    retrieveBlob(folder, localDir, metadata, timestamp);
  }
}

void* CcdbApi::extractFromTFile(TFile& file, TClass const* cl, const char* what)
{
  if (!cl) {
    return nullptr;
  }
  auto object = file.GetObjectChecked(what, cl);
  if (!object) {
    // it could be that object was stored with previous convention
    // where the classname was taken as key
    std::string objectName(cl->GetName());
    o2::utils::Str::trim(objectName);
    object = file.GetObjectChecked(objectName.c_str(), cl);
    LOG(warn) << "Did not find object under expected name " << what;
    if (!object) {
      return nullptr;
    }
    LOG(warn) << "Found object under deprecated name " << cl->GetName();
  }
  auto result = object;
  // We need to handle some specific cases as ROOT ties them deeply
  // to the file they are contained in
  if (cl->InheritsFrom("TObject")) {
    // make a clone
    // detach from the file
    auto tree = dynamic_cast<TTree*>((TObject*)object);
    if (tree) {
      tree->LoadBaskets(0x1L << 32); // make tree memory based
      tree->SetDirectory(nullptr);
      result = tree;
    } else {
      auto h = dynamic_cast<TH1*>((TObject*)object);
      if (h) {
        h->SetDirectory(nullptr);
        result = h;
      }
    }
  }
  return result;
}

void* CcdbApi::extractFromLocalFile(std::string const& filename, std::type_info const& tinfo, std::map<std::string, std::string>* headers) const
{
  if (!std::filesystem::exists(filename)) {
    LOG(error) << "Local snapshot " << filename << " not found \n";
    return nullptr;
  }
  std::lock_guard<std::mutex> guard(gIOMutex);
  auto tcl = tinfo2TClass(tinfo);
  TFile f(filename.c_str(), "READ");
  if (headers) {
    auto storedmeta = retrieveMetaInfo(f);
    if (storedmeta) {
      *headers = *storedmeta; // do a simple deep copy
      delete storedmeta;
    }
    if ((isSnapshotMode() || mPreferSnapshotCache) && headers->find("ETag") == headers->end()) { // generate dummy ETag to profit from the caching
      (*headers)["ETag"] = filename;
    }
  }
  return extractFromTFile(f, tcl);
}

bool CcdbApi::initTGrid() const
{
  if (mNeedAlienToken && !mAlienInstance) {
    static bool allowNoToken = getenv("ALICEO2_CCDB_NOTOKENCHECK") && atoi(getenv("ALICEO2_CCDB_NOTOKENCHECK"));
    if (!allowNoToken && !checkAlienToken()) {
      LOG(fatal) << "Alien Token Check failed - Please get an alien token before running with https CCDB endpoint, or alice-ccdb.cern.ch!";
    }
    mAlienInstance = TGrid::Connect("alien");
    static bool errorShown = false;
    if (!mAlienInstance && errorShown == false) {
      if (allowNoToken) {
        LOG(error) << "TGrid::Connect returned nullptr. May be due to missing alien token";
      } else {
        LOG(fatal) << "TGrid::Connect returned nullptr. May be due to missing alien token";
      }
      errorShown = true;
    }
  }
  return mAlienInstance != nullptr;
}

void* CcdbApi::downloadAlienContent(std::string const& url, std::type_info const& tinfo) const
{
  if (!initTGrid()) {
    return nullptr;
  }
  std::lock_guard<std::mutex> guard(gIOMutex);
  auto memfile = TMemFile::Open(url.c_str(), "OPEN");
  if (memfile) {
    auto cl = tinfo2TClass(tinfo);
    auto content = extractFromTFile(*memfile, cl);
    delete memfile;
    return content;
  }
  return nullptr;
}

void* CcdbApi::interpretAsTMemFileAndExtract(char* contentptr, size_t contentsize, std::type_info const& tinfo)
{
  void* result = nullptr;
  Int_t previousErrorLevel = gErrorIgnoreLevel;
  gErrorIgnoreLevel = kFatal;
  std::lock_guard<std::mutex> guard(gIOMutex);
  TMemFile memFile("name", contentptr, contentsize, "READ");
  gErrorIgnoreLevel = previousErrorLevel;
  if (!memFile.IsZombie()) {
    auto tcl = tinfo2TClass(tinfo);
    result = extractFromTFile(memFile, tcl);
    if (!result) {
      LOG(error) << o2::utils::Str::concat_string("Couldn't retrieve object corresponding to ", tcl->GetName(), " from TFile");
    }
    memFile.Close();
  }
  return result;
}

// navigate sequence of URLs until TFile content is found; object is extracted and returned
void* CcdbApi::navigateURLsAndRetrieveContent(CURL* curl_handle, std::string const& url, std::type_info const& tinfo, std::map<string, string>* headers) const
{
  // a global internal data structure that can be filled with HTTP header information
  // static --> to avoid frequent alloc/dealloc as optimization
  // not sure if thread_local takes away that benefit
  static thread_local std::multimap<std::string, std::string> headerData;

  // let's see first of all if the url is something specific that curl cannot handle
  if (url.find("alien:/", 0) != std::string::npos) {
    return downloadAlienContent(url, tinfo);
  }
  // add other final cases here
  // example root://

  // otherwise make an HTTP/CURL request
  // specify URL to get
  curl_easy_setopt(curl_handle, CURLOPT_URL, url.c_str());

  MemoryStruct chunk{(char*)malloc(1), 0};
  initCurlOptionsForRetrieve(curl_handle, (void*)&chunk, WriteMemoryCallback, false);

  curl_easy_setopt(curl_handle, CURLOPT_HEADERFUNCTION, header_map_callback<decltype(headerData)>);
  headerData.clear();
  curl_easy_setopt(curl_handle, CURLOPT_HEADERDATA, (void*)&headerData);

  curlSetSSLOptions(curl_handle);

  auto res = CURL_perform(curl_handle);
  long response_code = -1;
  void* content = nullptr;
  bool errorflag = false;
  if (res == CURLE_OK && curl_easy_getinfo(curl_handle, CURLINFO_RESPONSE_CODE, &response_code) == CURLE_OK) {
    if (headers) {
      for (auto& p : headerData) {
        (*headers)[p.first] = p.second;
      }
    }
    if (200 <= response_code && response_code < 300) {
      // good response and the content is directly provided and should have been dumped into "chunk"
      content = interpretAsTMemFileAndExtract(chunk.memory, chunk.size, tinfo);
    } else if (response_code == 304) {
      // this means the object exist but I am not serving
      // it since it's already in your possession

      // there is nothing to be done here
      LOGP(debug, "Object exists but I am not serving it since it's already in your possession");
    }
    // this is a more general redirection
    else if (300 <= response_code && response_code < 400) {
      // we try content locations in order of appearance until one succeeds
      // 1st: The "Location" field
      // 2nd: Possible "Content-Location" fields - Location field

      // some locations are relative to the main server so we need to fix/complement them
      auto complement_Location = [this](std::string const& loc) {
        if (loc[0] == '/') {
          // if it's just a path (noticed by trailing '/' we prepend the server url
          return getURL() + loc;
        }
        return loc;
      };

      std::vector<std::string> locs;
      auto iter = headerData.find("Location");
      if (iter != headerData.end()) {
        locs.push_back(complement_Location(iter->second));
      }
      // add alternative locations (not yet included)
      auto iter2 = headerData.find("Content-Location");
      if (iter2 != headerData.end()) {
        auto range = headerData.equal_range("Content-Location");
        for (auto it = range.first; it != range.second; ++it) {
          if (std::find(locs.begin(), locs.end(), it->second) == locs.end()) {
            locs.push_back(complement_Location(it->second));
          }
        }
      }
      for (auto& l : locs) {
        if (l.size() > 0) {
          LOG(debug) << "Trying content location " << l;
          content = navigateURLsAndRetrieveContent(curl_handle, l, tinfo, headers);
          if (content /* or other success marker in future */) {
            break;
          }
        }
      }
    } else if (response_code == 404) {
      LOG(error) << "Requested resource does not exist: " << url;
      errorflag = true;
    } else {
      LOG(error) << "Error in fetching object " << url << ", curl response code:" << response_code;
      errorflag = true;
    }
    // cleanup
    if (chunk.memory != nullptr) {
      free(chunk.memory);
    }
  } else {
    LOGP(alarm, "Curl request to {} failed with result {}, response code: {}", url, int(res), response_code);
    errorflag = true;
  }
  // indicate that an error occurred ---> used by caching layers (such as CCDBManager)
  if (errorflag && headers) {
    (*headers)["Error"] = "An error occurred during retrieval";
  }
  return content;
}

void* CcdbApi::retrieveFromTFile(std::type_info const& tinfo, std::string const& path,
                                 std::map<std::string, std::string> const& metadata, long timestamp,
                                 std::map<std::string, std::string>* headers, std::string const& etag,
                                 const std::string& createdNotAfter, const std::string& createdNotBefore) const
{
  if (!mSnapshotCachePath.empty()) {
    // protect this sensitive section by a multi-process named semaphore
    boost::interprocess::named_semaphore* sem = nullptr;
    std::hash<std::string> hasher;
    const auto semhashedstring = "aliceccdb" + std::to_string(hasher(mSnapshotCachePath + path)).substr(0, 16);
    try {
      sem = new boost::interprocess::named_semaphore(boost::interprocess::open_or_create_t{}, semhashedstring.c_str(), 1);
    } catch (std::exception e) {
      LOG(warn) << "Exception occurred during CCDB (cache) semaphore setup; Continuing without";
      sem = nullptr;
    }
    if (sem) {
      sem->wait(); // wait until we can enter (no one else there)
    }
    std::string logfile = mSnapshotCachePath + "/log";
    std::fstream out(logfile, ios_base::out | ios_base::app);
    if (out.is_open()) {
      out << "CCDB-access[" << getpid() << "] of " << mUniqueAgentID << " to " << path << " timestamp " << timestamp << "\n";
    }
    auto snapshotfile = getSnapshotFile(mSnapshotCachePath, path);
    bool snapshoting = false;
    if (!std::filesystem::exists(snapshotfile)) {
      snapshoting = true;
      out << "CCDB-access[" << getpid() << "] ... " << mUniqueAgentID << " downloading to snapshot " << snapshotfile << "\n";
      // if file not already here and valid --> snapshot it
      if (!retrieveBlob(path, mSnapshotCachePath, metadata, timestamp)) {
        out << "CCDB-access[" << getpid() << "] ... " << mUniqueAgentID << " failed to create directory for " << snapshotfile << "\n";
      }
    } else {
      out << "CCDB-access[" << getpid() << "]  ... " << mUniqueAgentID << "serving from local snapshot " << snapshotfile << "\n";
    }
    if (sem) {
      sem->post();
      if (sem->try_wait()) {
        // if nobody else is waiting remove the semaphore resource
        sem->post();
        boost::interprocess::named_semaphore::remove(semhashedstring.c_str());
      }
    }
    auto res = extractFromLocalFile(snapshotfile, tinfo, headers);
    if (!snapshoting) { // if snapshot was created at this call, the log was already done
      logReading(path, timestamp, headers, "retrieve from snapshot");
    }
    return res;
  }

  // normal mode follows

  CURL* curl_handle = curl_easy_init();
  string fullUrl = getFullUrlForRetrieval(curl_handle, path, metadata, timestamp); // todo check if function still works correctly in case mInSnapshotMode
  // if we are in snapshot mode we can simply open the file; extract the object and return
  if (mInSnapshotMode) {
    return extractFromLocalFile(fullUrl, tinfo, headers);
    logReading(path, timestamp, headers, "retrieve from snapshot");
  }

  initHeadersForRetrieve(curl_handle, timestamp, headers, etag, createdNotAfter, createdNotBefore);
  auto content = navigateURLsAndRetrieveContent(curl_handle, fullUrl, tinfo, headers);

  for (size_t hostIndex = 1; hostIndex < hostsPool.size() && !(content); hostIndex++) {
    fullUrl = getFullUrlForRetrieval(curl_handle, path, metadata, timestamp, hostIndex);
    content = navigateURLsAndRetrieveContent(curl_handle, fullUrl, tinfo, headers);
  }
  if (content) {
    logReading(path, timestamp, headers, "retrieve");
  }
  curl_easy_cleanup(curl_handle);
  return content;
}

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

std::string CcdbApi::list(std::string const& path, bool latestOnly, std::string const& returnFormat) const
{
  CURL* curl;
  CURLcode res = CURL_LAST;
  std::string result;

  curl = curl_easy_init();
  if (curl != nullptr) {
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, CurlWrite_CallbackFunc_StdString2);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &result);

    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, (string("Accept: ") + returnFormat).c_str());
    headers = curl_slist_append(headers, (string("Content-Type: ") + returnFormat).c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

    curlSetSSLOptions(curl);

    string fullUrl;
    // Perform the request, res will get the return code
    for (size_t hostIndex = 0; hostIndex < hostsPool.size() && res != CURLE_OK; hostIndex++) {
      fullUrl = getHostUrl(hostIndex);
      fullUrl += latestOnly ? "/latest/" : "/browse/";
      fullUrl += path;
      curl_easy_setopt(curl, CURLOPT_URL, fullUrl.c_str());

      res = CURL_perform(curl);
      if (res != CURLE_OK) {
        LOGP(alarm, "CURL_perform() failed: {}", curl_easy_strerror(res));
      }
    }
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
  }

  return result;
}

std::string CcdbApi::getTimestampString(long timestamp) const
{
  stringstream ss;
  ss << timestamp;
  return ss.str();
}

void CcdbApi::deleteObject(std::string const& path, long timestamp) const
{
  CURL* curl;
  CURLcode res;
  stringstream fullUrl;
  long timestampLocal = timestamp == -1 ? getCurrentTimestamp() : timestamp;

  curl = curl_easy_init();
  if (curl != nullptr) {
    curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, "DELETE");
    curlSetSSLOptions(curl);

    for (size_t hostIndex = 0; hostIndex < hostsPool.size(); hostIndex++) {
      fullUrl << getHostUrl(hostIndex) << "/" << path << "/" << timestampLocal;
      curl_easy_setopt(curl, CURLOPT_URL, fullUrl.str().c_str());

      // Perform the request, res will get the return code
      res = CURL_perform(curl);
      if (res != CURLE_OK) {
        LOGP(alarm, "CURL_perform() failed: {}", curl_easy_strerror(res));
      }
      curl_easy_cleanup(curl);
    }
  }
}

void CcdbApi::truncate(std::string const& path) const
{
  CURL* curl;
  CURLcode res;
  stringstream fullUrl;
  for (size_t i = 0; i < hostsPool.size(); i++) {
    string url = getHostUrl(i);
    fullUrl << url << "/truncate/" << path;

    curl = curl_easy_init();
    if (curl != nullptr) {
      curl_easy_setopt(curl, CURLOPT_URL, fullUrl.str().c_str());

      curlSetSSLOptions(curl);

      // Perform the request, res will get the return code
      res = CURL_perform(curl);
      if (res != CURLE_OK) {
        LOGP(alarm, "CURL_perform() failed: {}", curl_easy_strerror(res));
      }
      curl_easy_cleanup(curl);
    }
  }
}

size_t write_data(void*, size_t size, size_t nmemb, void*)
{
  return size * nmemb;
}

bool CcdbApi::isHostReachable() const
{
  CURL* curl;
  CURLcode res = CURL_LAST;
  bool result = false;

  curl = curl_easy_init();
  if (curl) {
    for (size_t hostIndex = 0; hostIndex < hostsPool.size() && res != CURLE_OK; hostIndex++) {
      curl_easy_setopt(curl, CURLOPT_URL, mUrl.data());
      curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_data);
      curlSetSSLOptions(curl);
      res = CURL_perform(curl);
      result = (res == CURLE_OK);
    }

    /* always cleanup */
    curl_easy_cleanup(curl);
  }
  return result;
}

std::vector<std::string> CcdbApi::parseSubFolders(std::string const& reply) const
{
  // this needs some text filtering
  // go through reply line by line until we see "SubFolders:"
  std::stringstream ss(reply.c_str());
  std::string line;
  std::vector<std::string> folders;

  size_t numberoflines = std::count(reply.begin(), reply.end(), '\n');
  bool inSubFolderSection = false;

  for (size_t linenumber = 0; linenumber < numberoflines; ++linenumber) {
    std::getline(ss, line);
    if (inSubFolderSection && line.size() > 0) {
      // remove all white space
      folders.push_back(sanitizeObjectName(line));
    }

    if (line.compare("Subfolders:") == 0) {
      inSubFolderSection = true;
    }
  }
  return folders;
}

namespace
{
size_t header_callback(char* buffer, size_t size, size_t nitems, void* userdata)
{
  auto* headers = static_cast<std::vector<std::string>*>(userdata);
  auto header = std::string(buffer, size * nitems);
  headers->emplace_back(std::string(header.data()));
  return size * nitems;
}
} // namespace

std::map<std::string, std::string> CcdbApi::retrieveHeaders(std::string const& path, std::map<std::string, std::string> const& metadata, long timestamp) const
{
  CURL* curl = curl_easy_init();
  CURLcode res = CURL_LAST;
  string fullUrl = getFullUrlForRetrieval(curl, path, metadata, timestamp);
  std::map<std::string, std::string> headers;

  if (curl != nullptr) {
    struct curl_slist* list = nullptr;
    list = curl_slist_append(list, ("If-None-Match: " + std::to_string(timestamp)).c_str());

    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, list);

    /* get us the resource without a body! */
    curl_easy_setopt(curl, CURLOPT_NOBODY, 1L);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_HEADERFUNCTION, header_map_callback<>);
    curl_easy_setopt(curl, CURLOPT_HEADERDATA, &headers);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, mUniqueAgentID.c_str());

    curlSetSSLOptions(curl);

    // Perform the request, res will get the return code

    long httpCode = 404;
    CURLcode getCodeRes = CURL_LAST;
    for (size_t hostIndex = 0; hostIndex < hostsPool.size() && (httpCode >= 400 || res > 0 || getCodeRes > 0); hostIndex++) {
      curl_easy_setopt(curl, CURLOPT_URL, fullUrl.c_str());
      res = CURL_perform(curl);
      if (res != CURLE_OK && res != CURLE_UNSUPPORTED_PROTOCOL) {
        // We take out the unsupported protocol error because we are only querying
        // header info which is returned in any case. Unsupported protocol error
        // occurs sometimes because of redirection to alien for blobs.
        LOG(error) << "CURL_perform() failed: " << curl_easy_strerror(res);
      }

      getCodeRes = curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &httpCode);
    }

    if (httpCode == 404) {
      headers.clear();
    }

    curl_easy_cleanup(curl);
  }

  return headers;
}

bool CcdbApi::getCCDBEntryHeaders(std::string const& url, std::string const& etag, std::vector<std::string>& headers, const std::string& agentID)
{
  auto curl = curl_easy_init();
  headers.clear();
  if (!curl) {
    return true;
  }

  struct curl_slist* list = nullptr;
  list = curl_slist_append(list, ("If-None-Match: " + etag).c_str());

  curl_easy_setopt(curl, CURLOPT_HTTPHEADER, list);

  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  /* get us the resource without a body! */
  curl_easy_setopt(curl, CURLOPT_NOBODY, 1L);
  curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
  curl_easy_setopt(curl, CURLOPT_HEADERFUNCTION, header_callback);
  curl_easy_setopt(curl, CURLOPT_HEADERDATA, &headers);
  if (!agentID.empty()) {
    curl_easy_setopt(curl, CURLOPT_USERAGENT, agentID.c_str());
  }

  curlSetSSLOptions(curl);

  /* Perform the request */
  curl_easy_perform(curl);
  long http_code = 404;
  curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
  if (http_code == 304) {
    return false;
  }
  return true;
}

void CcdbApi::parseCCDBHeaders(std::vector<std::string> const& headers, std::vector<std::string>& pfns, std::string& etag)
{
  static std::string etagHeader = "ETag: ";
  static std::string locationHeader = "Content-Location: ";
  for (auto h : headers) {
    if (h.find(etagHeader) == 0) {
      etag = std::string(h.data() + etagHeader.size());
    } else if (h.find(locationHeader) == 0) {
      pfns.emplace_back(std::string(h.data() + locationHeader.size(), h.size() - locationHeader.size()));
    }
  }
}

CCDBQuery* CcdbApi::retrieveQueryInfo(TFile& file)
{
  auto object = file.GetObjectChecked(CCDBQUERY_ENTRY, TClass::GetClass(typeid(o2::ccdb::CCDBQuery)));
  if (object) {
    return static_cast<CCDBQuery*>(object);
  }
  return nullptr;
}

std::map<std::string, std::string>* CcdbApi::retrieveMetaInfo(TFile& file)
{
  auto object = file.GetObjectChecked(CCDBMETA_ENTRY, TClass::GetClass(typeid(std::map<std::string, std::string>)));
  if (object) {
    return static_cast<std::map<std::string, std::string>*>(object);
  }
  return nullptr;
}

namespace
{
void traverseAndFillFolders(CcdbApi const& api, std::string const& top, std::vector<std::string>& folders)
{
  // LOG(info) << "Querying " << top;
  auto reply = api.list(top);
  folders.emplace_back(top);
  // LOG(info) << reply;
  auto subfolders = api.parseSubFolders(reply);
  if (subfolders.size() > 0) {
    // LOG(info) << subfolders.size() << " folders in " << top;
    for (auto& sub : subfolders) {
      traverseAndFillFolders(api, sub, folders);
    }
  } else {
    // LOG(info) << "NO subfolders in " << top;
  }
}
} // namespace

std::vector<std::string> CcdbApi::getAllFolders(std::string const& top) const
{
  std::vector<std::string> folders;
  traverseAndFillFolders(*this, top, folders);
  return folders;
}

TClass* CcdbApi::tinfo2TClass(std::type_info const& tinfo)
{
  TClass* cl = TClass::GetClass(tinfo);
  if (!cl) {
    throw std::runtime_error(fmt::format("Could not retrieve ROOT dictionary for type {}, aborting", tinfo.name()));
    return nullptr;
  }
  return cl;
}

void CcdbApi::updateMetadata(std::string const& path, std::map<std::string, std::string> const& metadata, long timestamp, std::string const& id, long newEOV)
{
  CURL* curl = curl_easy_init();
  if (curl != nullptr) {
    CURLcode res;
    stringstream fullUrl;
    for (size_t hostIndex = 0; hostIndex < hostsPool.size(); hostIndex++) {
      fullUrl << getHostUrl(hostIndex) << "/" << path << "/" << timestamp;
      if (newEOV > 0) {
        fullUrl << "/" << newEOV;
      }
      if (!id.empty()) {
        fullUrl << "/" << id;
      }
      fullUrl << "?";

      for (auto& kv : metadata) {
        string mfirst = kv.first;
        string msecond = kv.second;
        // same trick for the metadata as for the object type
        char* mfirstEncoded = curl_easy_escape(curl, mfirst.c_str(), mfirst.size());
        char* msecondEncoded = curl_easy_escape(curl, msecond.c_str(), msecond.size());
        fullUrl << string(mfirstEncoded) + "=" + string(msecondEncoded) + "&";
        curl_free(mfirstEncoded);
        curl_free(msecondEncoded);
      }

      if (curl != nullptr) {
        LOG(debug) << "passing to curl: " << fullUrl.str();
        curl_easy_setopt(curl, CURLOPT_URL, fullUrl.str().c_str());
        curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, "PUT"); // make sure we use PUT
        curl_easy_setopt(curl, CURLOPT_USERAGENT, mUniqueAgentID.c_str());
        curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
        curlSetSSLOptions(curl);

        // Perform the request, res will get the return code
        res = CURL_perform(curl);
        if (res != CURLE_OK) {
          LOGP(alarm, "CURL_perform() failed: {}", curl_easy_strerror(res));
        }
        curl_easy_cleanup(curl);
      }
    }
  }
}

std::vector<std::string> CcdbApi::splitString(const std::string& str, const char* delimiters)
{
  std::vector<std::string> tokens;
  char stringForStrTok[str.length() + 1];
  strcpy(stringForStrTok, str.c_str());
  char* token = strtok(stringForStrTok, delimiters);
  while (token != nullptr) {
    tokens.emplace_back(token);
    token = strtok(nullptr, delimiters);
  }
  return tokens;
}

void CcdbApi::initHostsPool(std::string hosts)
{
  hostsPool = splitString(hosts, ",;");
}

std::string CcdbApi::getHostUrl(int hostIndex) const
{
  return hostsPool.at(hostIndex);
}

void CcdbApi::loadFileToMemory(o2::pmr::vector<char>& dest, std::string const& path,
                               std::map<std::string, std::string> const& metadata, long timestamp,
                               std::map<std::string, std::string>* headers, std::string const& etag,
                               const std::string& createdNotAfter, const std::string& createdNotBefore, bool considerSnapshot) const
{
  LOGP(debug, "loadFileToMemory {} ETag=[{}]", path, etag);

  // if we are in snapshot mode we can simply open the file, unless the etag is non-empty:
  // this would mean that the object was is already fetched and in this mode we don't to validity checks!
  bool createSnapshot = considerSnapshot && !mSnapshotCachePath.empty(); // create snaphot if absent
  int fromSnapshot = 0;
  boost::interprocess::named_semaphore* sem = nullptr;
  std::string semhashedstring{}, snapshotpath{}, logfile{};
  std::unique_ptr<std::fstream> logStream;
  auto sem_release = [&sem, &semhashedstring, path, this]() {
    if (sem) {
      sem->post();
      if (sem->try_wait()) { // if nobody else is waiting remove the semaphore resource
        sem->post();
        boost::interprocess::named_semaphore::remove(semhashedstring.c_str());
      }
    }
  };

  if (createSnapshot) { // create named semaphore
    std::hash<std::string> hasher;
    semhashedstring = "aliceccdb" + std::to_string(hasher(mSnapshotCachePath + path)).substr(0, 16);
    try {
      sem = new boost::interprocess::named_semaphore(boost::interprocess::open_or_create_t{}, semhashedstring.c_str(), 1);
    } catch (std::exception e) {
      LOG(warn) << "Exception occurred during CCDB (cache) semaphore setup; Continuing without";
      sem = nullptr;
    }
    if (sem) {
      sem->wait(); // wait until we can enter (no one else there)
    }
    logfile = mSnapshotCachePath + "/log";
    logStream = std::make_unique<std::fstream>(logfile, ios_base::out | ios_base::app);
    if (logStream->is_open()) {
      *logStream.get() << "CCDB-access[" << getpid() << "] of " << mUniqueAgentID << " to " << path << " timestamp " << timestamp << " for load to memory\n";
    }
  }

  if (mInSnapshotMode) { // file must be there, otherwise a fatal will be produced
    loadFileToMemory(dest, getSnapshotFile(mSnapshotTopPath, path), headers);
    fromSnapshot = 1;
  } else if (mPreferSnapshotCache && std::filesystem::exists(snapshotpath = getSnapshotFile(mSnapshotCachePath, path))) {
    // if file is available, use it, otherwise cache it below from the server. Do this only when etag is empty since otherwise the object was already fetched and cached
    if (etag.empty()) {
      loadFileToMemory(dest, snapshotpath, headers);
    }
    fromSnapshot = 2;
  } else { // look on the server
    CURL* curl_handle = curl_easy_init();
    string fullUrl = getFullUrlForRetrieval(curl_handle, path, metadata, timestamp);

    initHeadersForRetrieve(curl_handle, timestamp, headers, etag, createdNotAfter, createdNotBefore);

    navigateURLsAndLoadFileToMemory(dest, curl_handle, fullUrl, headers);

    for (size_t hostIndex = 1; hostIndex < hostsPool.size() && isMemoryFileInvalid(dest); hostIndex++) {
      fullUrl = getFullUrlForRetrieval(curl_handle, path, metadata, timestamp, hostIndex);
      loadFileToMemory(dest, fullUrl, headers); // headers loaded from the file in case of the snapshot reading only
    }
    curl_easy_cleanup(curl_handle);
  }

  if (dest.empty()) {
    sem_release();
    return; // nothing was fetched: either cached value is good or error was produced
  }
  // !considerSnapshot means that the call was made by retrieve for snapshoting reasons
  logReading(path, timestamp, headers, fmt::format("{}{}", considerSnapshot ? "load to memory" : "retrieve", fromSnapshot ? " from snapshot" : ""));

  // are we asked to create a snapshot ?
  if (createSnapshot && fromSnapshot != 2 && !(mInSnapshotMode && mSnapshotTopPath == mSnapshotCachePath)) { // store in the snapshot only if the object was not read from the snapshot
    auto snapshotdir = getSnapshotDir(mSnapshotCachePath, path);
    snapshotpath = getSnapshotFile(mSnapshotCachePath, path);
    o2::utils::createDirectoriesIfAbsent(snapshotdir);
    if (logStream->is_open()) {
      *logStream.get() << "CCDB-access[" << getpid() << "] ... " << mUniqueAgentID << " downloading to snapshot " << snapshotpath << " from memory\n";
    }
    { // dump image to a file
      LOGP(debug, "creating snapshot {} -> {}", path, snapshotpath);
      CCDBQuery querysummary(path, metadata, timestamp);
      {
        std::ofstream objFile(snapshotpath, std::ios::out | std::ofstream::binary);
        std::copy(dest.begin(), dest.end(), std::ostreambuf_iterator<char>(objFile));
      }
      // now open the same file as root file and store metadata
      std::lock_guard<std::mutex> guard(gIOMutex);
      auto oldlevel = gErrorIgnoreLevel;
      gErrorIgnoreLevel = 6001;                       // ignoring error messages here (since we catch with IsZombie)
      TFile snapshot(snapshotpath.c_str(), "UPDATE"); // the assumption is that the blob is a ROOT file
      if (!snapshot.IsZombie()) {
        snapshot.WriteObjectAny(&querysummary, TClass::GetClass(typeid(querysummary)), CCDBQUERY_ENTRY);
        if (headers) {
          snapshot.WriteObjectAny(headers, TClass::GetClass(typeid(metadata)), CCDBMETA_ENTRY);
        }
      }
      snapshot.Close();
      gErrorIgnoreLevel = oldlevel;
    }
  }
  sem_release();
}

// navigate sequence of URLs until TFile content is found; object is extracted and returned
void CcdbApi::navigateURLsAndLoadFileToMemory(o2::pmr::vector<char>& dest, CURL* curl_handle, std::string const& url, std::map<string, string>* headers) const
{
  // a global internal data structure that can be filled with HTTP header information
  // static --> to avoid frequent alloc/dealloc as optimization
  // not sure if thread_local takes away that benefit
  static thread_local std::multimap<std::string, std::string> headerData;

  // let's see first of all if the url is something specific that curl cannot handle
  if (url.find("alien:/", 0) != std::string::npos) {
    return loadFileToMemory(dest, url, nullptr); // headers loaded from the file in case of the snapshot reading only
  }
  // otherwise make an HTTP/CURL request
  bool errorflag = false;
  auto signalError = [&chunk = dest, &errorflag]() {
    chunk.clear();
    chunk.reserve(1);
    errorflag = true;
  };
  auto writeCallBack = [](void* contents, size_t size, size_t nmemb, void* chunkptr) {
    o2::pmr::vector<char>& chunk = *static_cast<o2::pmr::vector<char>*>(chunkptr);
    size_t realsize = size * nmemb;
    try {
      chunk.reserve(chunk.size() + realsize);
      char* contC = (char*)contents;
      chunk.insert(chunk.end(), contC, contC + realsize);
    } catch (std::exception e) {
      LOGP(info, "failed to expand by {} bytes chunk provided to CURL: {}", realsize, e.what());
      realsize = 0;
    }
    return realsize;
  };

  // specify URL to get
  curl_easy_setopt(curl_handle, CURLOPT_URL, url.c_str());
  initCurlOptionsForRetrieve(curl_handle, (void*)&dest, writeCallBack, false);
  curl_easy_setopt(curl_handle, CURLOPT_HEADERFUNCTION, header_map_callback<decltype(headerData)>);
  headerData.clear();
  curl_easy_setopt(curl_handle, CURLOPT_HEADERDATA, (void*)&headerData);
  curlSetSSLOptions(curl_handle);

  auto res = CURL_perform(curl_handle);
  long response_code = -1;
  if (res == CURLE_OK && curl_easy_getinfo(curl_handle, CURLINFO_RESPONSE_CODE, &response_code) == CURLE_OK) {
    if (headers) {
      for (auto& p : headerData) {
        (*headers)[p.first] = p.second;
      }
    }
    if (200 <= response_code && response_code < 300) {
      // good response and the content is directly provided and should have been dumped into "chunk"
    } else if (response_code == 304) {
      LOGP(debug, "Object exists but I am not serving it since it's already in your possession");
    }
    // this is a more general redirection
    else if (300 <= response_code && response_code < 400) {
      // we try content locations in order of appearance until one succeeds
      // 1st: The "Location" field
      // 2nd: Possible "Content-Location" fields - Location field
      // some locations are relative to the main server so we need to fix/complement them
      auto complement_Location = [this](std::string const& loc) {
        if (loc[0] == '/') {
          // if it's just a path (noticed by trailing '/' we prepend the server url
          return getURL() + loc;
        }
        return loc;
      };

      std::vector<std::string> locs;
      auto iter = headerData.find("Location");
      if (iter != headerData.end()) {
        locs.push_back(complement_Location(iter->second));
      }
      // add alternative locations (not yet included)
      auto iter2 = headerData.find("Content-Location");
      if (iter2 != headerData.end()) {
        auto range = headerData.equal_range("Content-Location");
        for (auto it = range.first; it != range.second; ++it) {
          if (std::find(locs.begin(), locs.end(), it->second) == locs.end()) {
            locs.push_back(complement_Location(it->second));
          }
        }
      }
      for (auto& l : locs) {
        if (l.size() > 0) {
          LOG(debug) << "Trying content location " << l;
          navigateURLsAndLoadFileToMemory(dest, curl_handle, l, headers);
          if (dest.size()) { /* or other success marker in future */
            break;
          }
        }
      }
    } else if (response_code == 404) {
      LOG(error) << "Requested resource does not exist: " << url;
      signalError();
    } else {
      LOG(error) << "Error in fetching object " << url << ", curl response code:" << response_code;
      signalError();
    }
  } else {
    LOGP(alarm, "Curl request to {} failed with result {}, response code: {}", url, int(res), response_code);
    signalError();
  }
  // indicate that an error occurred ---> used by caching layers (such as CCDBManager)
  if (errorflag && headers) {
    (*headers)["Error"] = "An error occurred during retrieval";
  }
  return;
}

void CcdbApi::loadFileToMemory(o2::pmr::vector<char>& dest, const std::string& path, std::map<std::string, std::string>* localHeaders) const
{
  // Read file to memory as vector. For special case of the locally cached file retriev metadata stored directly in the file
  constexpr size_t MaxCopySize = 0x1L << 25;
  auto signalError = [&dest, localHeaders]() {
    dest.clear();
    dest.reserve(1);
    if (localHeaders) { // indicate that an error occurred ---> used by caching layers (such as CCDBManager)
      (*localHeaders)["Error"] = "An error occurred during retrieval";
    }
  };
  if (path.find("alien:/") == 0 && !initTGrid()) {
    signalError();
    return;
  }
  std::string fname(path);
  if (fname.find("?filetype=raw") == std::string::npos) {
    fname += "?filetype=raw";
  }
  std::unique_ptr<TFile> sfile{TFile::Open(fname.c_str())};
  if (!sfile || sfile->IsZombie()) {
    LOG(error) << "Failed to open file " << fname;
    signalError();
    return;
  }
  size_t totalread = 0, fsize = sfile->GetSize(), b00 = sfile->GetBytesRead();
  dest.resize(fsize);
  char* dptr = dest.data();
  sfile->Seek(0);
  long nread = 0;
  do {
    size_t b0 = sfile->GetBytesRead(), b1 = b0 - b00;
    size_t readsize = fsize - b1 > MaxCopySize ? MaxCopySize : fsize - b1;
    if (readsize == 0) {
      break;
    }
    sfile->Seek(totalread, TFile::kBeg);
    bool failed = sfile->ReadBuffer(dptr, (Int_t)readsize);
    nread = sfile->GetBytesRead() - b0;
    if (failed || nread < 0) {
      LOG(error) << "failed to copy file " << fname << " to memory buffer";
      signalError();
      return;
    }
    dptr += nread;
    totalread += nread;
  } while (nread == (long)MaxCopySize);

  if (localHeaders) {
    TMemFile memFile("name", const_cast<char*>(dest.data()), dest.size(), "READ");
    auto storedmeta = (std::map<std::string, std::string>*)extractFromTFile(memFile, TClass::GetClass("std::map<std::string, std::string>"), CCDBMETA_ENTRY);
    if (storedmeta) {
      *localHeaders = *storedmeta; // do a simple deep copy
      delete storedmeta;
    }
    if ((isSnapshotMode() || mPreferSnapshotCache) && localHeaders->find("ETag") == localHeaders->end()) { // generate dummy ETag to profit from the caching
      (*localHeaders)["ETag"] = path;
    }
  }
  return;
}

void CcdbApi::checkMetadataKeys(std::map<std::string, std::string> const& metadata) const
{

  // function to check if any key contains invalid characters
  // if so, a fatal will be issued

  const std::regex regexPatternSearch(R"([ :;.,\\/'?!\(\)\{\}\[\]@<>=+*#$&`|~^%])");
  bool isInvalid = false;

  for (auto& el : metadata) {
    auto keyMd = el.first;
    auto tmp = keyMd;
    std::smatch searchRes;
    while (std::regex_search(keyMd, searchRes, regexPatternSearch)) {
      isInvalid = true;
      LOG(error) << "Invalid character found in metadata key '" << tmp << "\': '" << searchRes.str() << "\'";
      keyMd = searchRes.suffix();
    }
  }
  if (isInvalid) {
    LOG(fatal) << "Some metadata keys have invalid characters, please fix!";
  }
  return;
}

void CcdbApi::logReading(const std::string& path, long ts, const std::map<std::string, std::string>* headers, const std::string& comment) const
{
  std::string upath{path};
  if (headers) {
    auto ent = headers->find("Valid-From");
    if (ent != headers->end()) {
      upath += "/" + ent->second;
    }
    ent = headers->find("ETag");
    if (ent != headers->end()) {
      upath += "/" + ent->second;
    }
  }
  upath.erase(remove(upath.begin(), upath.end(), '\"'), upath.end());
  LOGP(info, "ccdb reads {}{}{} for {} ({}, agent_id: {}), ", mUrl, mUrl.back() == '/' ? "" : "/", upath, ts < 0 ? getCurrentTimestamp() : ts, comment, mUniqueAgentID);
}

CURLcode CcdbApi::CURL_perform(CURL* handle) const
{
  if (mIsCCDBDownloaderEnabled) {
    return mDownloader->perform(handle);
  }
  return curl_easy_perform(handle);
}

} // namespace o2
