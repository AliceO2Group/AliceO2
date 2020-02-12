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
/// \author Barthelemy von Haller, Sandro Wenzel
///

#include "CCDB/CcdbApi.h"
#include "CCDB/CCDBQuery.h"
#include <regex>
#include <chrono>
#include <TMessage.h>
#include <sstream>
#include <CommonUtils/StringUtils.h>
#include <TFile.h>
#include <TSystem.h>
#include <TStreamerInfo.h>
#include <TMemFile.h>
#include <TBufferFile.h>
#include <TWebFile.h>
#include <TH1F.h>
#include <TTree.h>
#include <FairLogger.h>
#include <TError.h>
#include <TClass.h>
#include <CCDB/CCDBTimeStampUtils.h>
#include <algorithm>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <iostream>

namespace o2
{
namespace ccdb
{

using namespace std;

CcdbApi::~CcdbApi()
{
  curl_global_cleanup();
}

void CcdbApi::curlInit()
{
  // todo : are there other things to initialize globally for curl ?
  curl_global_init(CURL_GLOBAL_DEFAULT);
}

void CcdbApi::init(std::string const& host)
{
  // if host is prefixed with "file://" this is a local snapshot
  // in this case we init the API in snapshot (readonly) mode
  constexpr const char* SNAPSHOTPREFIX = "file://";
  mUrl = host;

  if (host.substr(0, 7).compare(SNAPSHOTPREFIX) == 0) {
    auto path = host.substr(7);
    LOG(INFO) << "Initializing CcdbApi in snapshot readonly mode ... reading snapshot from path " << path;
    initInSnapshotMode(path);
  } else {
    curlInit();
  }
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

void CcdbApi::storeAsTFile_impl(void* obj, std::type_info const& tinfo, std::string const& path, std::map<std::string, std::string> const& metadata,
                                long startValidityTimestamp, long endValidityTimestamp) const
{
  // We need the TClass for this type; will verify if dictionary exists
  auto tcl = TClass::GetClass(tinfo);
  if (!tcl) {
    std::cerr << "Could not retrieve ROOT dictionary for type " << tinfo.name() << " aborting to write to CCDB";
    return;
  }

  // Prepare file name (for now the name corresponds to the name stored by TClass)
  string className = string(tcl->GetName());
  utils::trim(className);
  string tmpFileName = className + "_" + getTimestampString(getCurrentTimestamp()) + ".root";
#if ROOT_VERSION_CODE < ROOT_VERSION(6, 18, 0)
  TMemFile memFile(tmpFileName.c_str(), "RECREATE");
#else
  size_t memFileBlockSize = 1024; // 1KB
  TMemFile memFile(tmpFileName.c_str(), "RECREATE", "", ROOT::RCompressionSetting::EDefaults::kUseGeneralPurpose,
                   memFileBlockSize);
#endif
  if (memFile.IsZombie()) {
    cerr << "Error opening file " << tmpFileName << ", we can't store object " << className << endl;
    memFile.Close();
    return;
  }
  memFile.WriteObjectAny(obj, tcl, className.c_str());
  memFile.Close();

  // Prepare Buffer
  void* buffer = malloc(memFile.GetSize());
  memFile.CopyTo(buffer, memFile.GetSize());

  // Prepare URL
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
  string fullUrl = getFullUrlForStorage(path, metadata, sanitizedStartValidityTimestamp, sanitizedEndValidityTimestamp);
  LOG(DEBUG) << "Full URL " << fullUrl;

  // Curl preparation
  CURL* curl = nullptr;
  struct curl_httppost* formpost = nullptr;
  struct curl_httppost* lastptr = nullptr;
  struct curl_slist* headerlist = nullptr;
  static const char buf[] = "Expect:";
  curl_formadd(&formpost,
               &lastptr,
               CURLFORM_COPYNAME, "send",
               CURLFORM_BUFFER, tmpFileName.c_str(),
               CURLFORM_BUFFERPTR, buffer, //.Buffer(),
               CURLFORM_BUFFERLENGTH, memFile.GetSize(),
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
  } else {
    cerr << "curl initialization failure" << endl;
  }
}

void CcdbApi::storeAsTFile(TObject* rootObject, std::string const& path, std::map<std::string, std::string> const& metadata,
                           long startValidityTimestamp, long endValidityTimestamp) const
{
  // Prepare file
  string objectName = string(rootObject->GetName());
  utils::trim(objectName);
  string tmpFileName = objectName + "_" + getTimestampString(getCurrentTimestamp()) + ".root";
#if ROOT_VERSION_CODE < ROOT_VERSION(6, 18, 0)
  TMemFile memFile(tmpFileName.c_str(), "RECREATE");
#else
  size_t memFileBlockSize = 1024; // 1KB
  TMemFile memFile(tmpFileName.c_str(), "RECREATE", "", ROOT::RCompressionSetting::EDefaults::kUseGeneralPurpose,
                   memFileBlockSize);
#endif
  if (memFile.IsZombie()) {
    cerr << "Error opening file " << tmpFileName << ", we can't store object " << rootObject->GetName() << endl;
    memFile.Close();
    return;
  }
  rootObject->Write(CCDBOBJECT_ENTRY);
  memFile.Close();

  // Prepare Buffer
  void* buffer = malloc(memFile.GetSize());
  memFile.CopyTo(buffer, memFile.GetSize());

  // Prepare URL
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
  string fullUrl = getFullUrlForStorage(path, metadata, sanitizedStartValidityTimestamp, sanitizedEndValidityTimestamp);

  // Curl preparation
  CURL* curl;
  struct curl_httppost* formpost = nullptr;
  struct curl_httppost* lastptr = nullptr;
  struct curl_slist* headerlist = nullptr;
  static const char buf[] = "Expect:";
  curl_formadd(&formpost,
               &lastptr,
               CURLFORM_COPYNAME, "send",
               CURLFORM_BUFFER, tmpFileName.c_str(),
               CURLFORM_BUFFERPTR, buffer, //.Buffer(),
               CURLFORM_BUFFERLENGTH, memFile.GetSize(),
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
  } else {
    cerr << "curl initialization failure" << endl;
  }
}

string CcdbApi::getFullUrlForStorage(const string& path, const map<string, string>& metadata,
                                     long startValidityTimestamp, long endValidityTimestamp) const
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
string CcdbApi::getFullUrlForRetrieval(const string& path, const map<string, string>& metadata, long timestamp) const
{
  if (mInSnapshotMode) {
    string snapshotPath = mSnapshotTopPath + "/" + path + "/snapshot.root";
    return snapshotPath;
  }

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

/**
 * Callback used by CURL to store the data received from the CCDB
 * directly into a binary file
 * @param contents
 * @param size
 * @param nmemb
 * @param userp a MemoryStruct where data is stored.
 * @return the size of the data we received and stored at userp.
 */
static size_t WriteToFileCallback(void* ptr, size_t size, size_t nmemb, FILE* stream)
{
  size_t written = fwrite(ptr, size, nmemb, stream);
  return written;
}

TObject* CcdbApi::retrieve(std::string const& path, std::map<std::string, std::string> const& metadata,
                           long timestamp) const
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

namespace
{
size_t header_map_callback(char* buffer, size_t size, size_t nitems, void* userdata)
{
  auto* headers = static_cast<std::map<std::string, std::string>*>(userdata);
  auto header = std::string(buffer, size * nitems);
  std::string::size_type index = header.find(':', 0);
  if (index != std::string::npos) {
    headers->insert(std::make_pair(
      boost::algorithm::trim_copy(header.substr(0, index)),
      boost::algorithm::trim_copy(header.substr(index + 1))));
  }
  return size * nitems;
}
} // namespace

TObject* CcdbApi::retrieveFromTFile(std::string const& path, std::map<std::string, std::string> const& metadata,
                                    long timestamp, std::map<std::string, std::string>* headers) const
{
  // Note : based on https://curl.haxx.se/libcurl/c/getinmemory.html
  // Thus it does not comply to our coding guidelines as it is a copy paste.

  string fullUrl = getFullUrlForRetrieval(path, metadata, timestamp);
  //  std::map<std::string, std::string> headers2;

  // Prepare CURL
  CURL* curl_handle;
  CURLcode res;
  struct MemoryStruct chunk {
    (char*)malloc(1) /*memory*/, 0 /*size*/
  };

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

  // setup curl for headers handling
  if (headers != nullptr) {
    struct curl_slist* list = nullptr;
    list = curl_slist_append(list, ("If-None-Match: " + to_string(timestamp)).c_str());
    curl_easy_setopt(curl_handle, CURLOPT_HTTPHEADER, list);
    curl_easy_setopt(curl_handle, CURLOPT_HEADERFUNCTION, header_map_callback);
    curl_easy_setopt(curl_handle, CURLOPT_HEADERDATA, headers);
  }

  /* get it! */
  res = curl_easy_perform(curl_handle);

  TObject* result = nullptr;
  if (res == CURLE_OK) {
    long response_code;
    res = curl_easy_getinfo(curl_handle, CURLINFO_RESPONSE_CODE, &response_code);
    if ((res == CURLE_OK) && (response_code != 404)) {
      Int_t previousErrorLevel = gErrorIgnoreLevel;
      gErrorIgnoreLevel = kFatal;
      TMemFile memFile("name", chunk.memory, chunk.size, "READ");
      gErrorIgnoreLevel = previousErrorLevel;
      if (!memFile.IsZombie()) {
        result = (TObject*)extractFromTFile(memFile, CCDBOBJECT_ENTRY, TClass::GetClass("TObject"));
        if (result == nullptr) {
          LOG(ERROR) << "Couldn't retrieve the object " << path;
        }
        memFile.Close();
      } else {
        LOG(DEBUG) << "Object " << path << " is not stored in a TMemFile";
      }
    } else {
      LOG(ERROR) << "Invalid URL : " << fullUrl;
    }
  } else {
    fprintf(stderr, "curl_easy_perform() failed: %s\n", curl_easy_strerror(res));
  }

  curl_easy_cleanup(curl_handle);
  free(chunk.memory);
  return result;
}

void CcdbApi::retrieveBlob(std::string const& path, std::string const& targetdir, std::map<std::string, std::string> const& metadata, long timestamp) const
{
  string fullUrl = getFullUrlForRetrieval(path, metadata, timestamp);

  // we setup the target path for this blob
  std::string fulltargetdir = targetdir + '/' + path;

  if (!boost::filesystem::exists(fulltargetdir)) {
    if (!boost::filesystem::create_directories(fulltargetdir)) {
      std::cerr << "Could not create target directory " << fulltargetdir << "\n";
    }
  }

  std::string targetpath = fulltargetdir + "/snapshot.root";
  FILE* fp = fopen(targetpath.c_str(), "w");
  if (!fp) {
    std::cerr << " Could not open/create target file " << targetpath << "\n";
    return;
  }

  // Prepare CURL
  CURL* curl_handle;
  CURLcode res;

  /* init the curl session */
  curl_handle = curl_easy_init();

  /* specify URL to get */
  curl_easy_setopt(curl_handle, CURLOPT_URL, fullUrl.c_str());

  /* send all data to this function  */
  curl_easy_setopt(curl_handle, CURLOPT_WRITEFUNCTION, WriteToFileCallback);

  /* we pass our file handle to the callback function */
  curl_easy_setopt(curl_handle, CURLOPT_WRITEDATA, (void*)fp);

  /* some servers don't like requests that are made without a user-agent
         field, so we provide one */
  curl_easy_setopt(curl_handle, CURLOPT_USERAGENT, "libcurl-agent/1.0");

  /* if redirected , we tell libcurl to follow redirection */
  curl_easy_setopt(curl_handle, CURLOPT_FOLLOWLOCATION, 1L);

  /* get it! */
  res = curl_easy_perform(curl_handle);

  void* result = nullptr;
  if (res == CURLE_OK) {
    long response_code;
    res = curl_easy_getinfo(curl_handle, CURLINFO_RESPONSE_CODE, &response_code);
    if ((res == CURLE_OK) && (response_code != 404)) {
    } else {
      LOG(ERROR) << "Invalid URL : " << fullUrl;
    }
  } else {
    fprintf(stderr, "curl_easy_perform() failed: %s\n", curl_easy_strerror(res));
  }

  if (fp) {
    fclose(fp);
  }
  curl_easy_cleanup(curl_handle);

  // trying to append metadata to the file so that it can be inspected WHERE/HOW/WHAT IT corresponds to
  // Just a demonstrator for the moment
  TFile snapshotfile(targetpath.c_str(), "UPDATE");
  CCDBQuery querysummary(path, metadata, timestamp);
  snapshotfile.WriteObjectAny(&querysummary, TClass::GetClass(typeid(querysummary)), CCDBQUERY_ENTRY);

  // retrieveHeaders
  auto headers = retrieveHeaders(path, metadata, timestamp);
  snapshotfile.WriteObjectAny(&headers, TClass::GetClass(typeid(metadata)), CCDBMETA_ENTRY);

  snapshotfile.Close();
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

void* CcdbApi::extractFromTFile(TFile& file, std::string const& objname, TClass const* cl)
{
  if (!cl) {
    return nullptr;
  }
  auto object = file.GetObjectChecked(objname.c_str(), cl);
  if (!object) {
    return nullptr;
  }
  auto result = object;
  // We need to handle some specific cases as ROOT ties them deeply
  // to the file they are contained in
  if (cl->InheritsFrom("TObject")) {
    // make a clone
    auto obj = ((TObject*)object)->Clone();
    // detach from the file
    if (auto tree = dynamic_cast<TTree*>(obj)) {
      tree->SetDirectory(nullptr);
    } else if (auto h = dynamic_cast<TH1*>(obj)) {
      h->SetDirectory(nullptr);
    }
    result = obj;
  }
  return result;
}

void* CcdbApi::extractFromLocalFile(std::string const& filename, std::string const& objname, TClass const* tcl) const
{
  if (!boost::filesystem::exists(filename)) {
    LOG(INFO) << "Local snapshot " << filename << " not found \n";
    return nullptr;
  }
  TFile f(filename.c_str(), "READ");
  return extractFromTFile(f, objname, tcl);
}

void* CcdbApi::retrieveFromTFile(std::type_info const& tinfo, std::string const& path,
                                 std::map<std::string, std::string> const& metadata, long timestamp, std::map<std::string, std::string>* headers) const
{
  // We need the TClass for this type; will verify if dictionary exists
  auto tcl = TClass::GetClass(tinfo);
  if (!tcl) {
    std::cerr << "Could not retrieve ROOT dictionary for type " << tinfo.name() << " aborting to read from CCDB";
    return nullptr;
  }
  string objectName = string(tcl->GetName());
  utils::trim(objectName);

  // Note : based on https://curl.haxx.se/libcurl/c/getinmemory.html
  // Thus it does not comply to our coding guidelines as it is a copy paste.

  string fullUrl = getFullUrlForRetrieval(path, metadata, timestamp);

  // if we are in snapshot mode we can simply open the file; extract the object and return
  if (mInSnapshotMode) {
    return extractFromLocalFile(fullUrl, objectName, tcl);
  }

  // Prepare CURL
  CURL* curl_handle;
  CURLcode res;
  struct MemoryStruct chunk {
    (char*)malloc(1) /*memory*/, 0 /*size*/
  };

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

  // setup curl for headers handling
  if (headers != nullptr) {
    struct curl_slist* list = nullptr;
    list = curl_slist_append(list, ("If-None-Match: " + to_string(timestamp)).c_str());
    curl_easy_setopt(curl_handle, CURLOPT_HTTPHEADER, list);
    curl_easy_setopt(curl_handle, CURLOPT_HEADERFUNCTION, header_map_callback);
    curl_easy_setopt(curl_handle, CURLOPT_HEADERDATA, headers);
  }

  /* get it! */
  res = curl_easy_perform(curl_handle);

  void* result = nullptr;
  if (res == CURLE_OK) {
    long response_code;
    res = curl_easy_getinfo(curl_handle, CURLINFO_RESPONSE_CODE, &response_code);
    if ((res == CURLE_OK) && (response_code != 404)) {
      Int_t previousErrorLevel = gErrorIgnoreLevel;
      gErrorIgnoreLevel = kFatal;
      TMemFile memFile("name", chunk.memory, chunk.size, "READ");
      gErrorIgnoreLevel = previousErrorLevel;
      if (!memFile.IsZombie()) {
        result = extractFromTFile(memFile, objectName.c_str(), tcl);
        if (!result) {
          LOG(ERROR) << "Couldn't retrieve the object " << path;
        }
        memFile.Close();
      } else {
        LOG(DEBUG) << "Object " << path << " is not stored in a TMemFile";
      }
    } else {
      LOG(ERROR) << "Invalid URL : " << fullUrl;
    }
  } else {
    fprintf(stderr, "curl_easy_perform() failed: %s\n", curl_easy_strerror(res));
  }

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
    LOG(ERROR) << "memory error when getting data from CCDB";
    return 0;
  }

  std::copy((char*)contents, (char*)contents + newLength, s->begin() + oldLength);
  return size * nmemb;
}

std::string CcdbApi::list(std::string const& path, bool latestOnly, std::string const& returnFormat) const
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

void CcdbApi::truncate(std::string const& path) const
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

bool CcdbApi::isHostReachable() const
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

std::vector<std::string> CcdbApi::parseSubFolders(std::string const& reply) const
{
  // this needs some text filtering
  // go through reply line by line until we see "SubFolders:"
  std::stringstream ss(reply.c_str());
  std::string line;
  std::vector<std::string> folders;

  size_t numberoflines = std::count(reply.begin(), reply.end(), '\n');
  bool inSubFolderSection = false;

  int counter = 0;
  for (int linenumber = 0; linenumber < numberoflines; ++linenumber) {
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
  std::vector<std::string>* headers = static_cast<std::vector<std::string>*>(userdata);
  auto header = std::string(buffer, size * nitems);
  headers->emplace_back(std::string(header.data()));
  return size * nitems;
}
}

std::map<std::string, std::string> CcdbApi::retrieveHeaders(std::string const& path, std::map<std::string, std::string> const& metadata, long timestamp) const
{

  CURL* curl = curl_easy_init();
  CURLcode res;
  string fullUrl = getFullUrlForRetrieval(path, metadata, timestamp);
  std::map<std::string, std::string> headers;

  if (curl != nullptr) {
    struct curl_slist* list = nullptr;
    list = curl_slist_append(list, ("If-None-Match: " + std::to_string(timestamp)).c_str());

    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, list);
    curl_easy_setopt(curl, CURLOPT_URL, fullUrl.c_str());
    /* get us the resource without a body! */
    curl_easy_setopt(curl, CURLOPT_NOBODY, 1L);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_HEADERFUNCTION, header_map_callback);
    curl_easy_setopt(curl, CURLOPT_HEADERDATA, &headers);

    // Perform the request, res will get the return code
    res = curl_easy_perform(curl);
    if (res != CURLE_OK) {
      fprintf(stderr, "curl_easy_perform() failed: %s\n", curl_easy_strerror(res));
    }
    long http_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
    if (http_code == 404) {
      headers.clear();
    }

    curl_easy_cleanup(curl);
  }

  return headers;
}

bool CcdbApi::getCCDBEntryHeaders(std::string const& url, std::string const& etag, std::vector<std::string>& headers)
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

  /* Perform the request */
  curl_easy_perform(curl);
  long http_code = 0;
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
  // LOG(INFO) << "Querying " << top;
  auto reply = api.list(top);
  folders.emplace_back(top);
  // LOG(INFO) << reply;
  auto subfolders = api.parseSubFolders(reply);
  if (subfolders.size() > 0) {
    // LOG(INFO) << subfolders.size() << " folders in " << top;
    for (auto& sub : subfolders) {
      traverseAndFillFolders(api, sub, folders);
    }
  } else {
    // LOG(INFO) << "NO subfolders in " << top;
  }
}
} // namespace

std::vector<std::string> CcdbApi::getAllFolders(std::string const& top) const
{
  std::vector<std::string> folders;
  traverseAndFillFolders(*this, top, folders);
  return folders;
}

} // namespace ccdb
} // namespace o2
