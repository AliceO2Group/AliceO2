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
#include "CommonUtils/StringUtils.h"
#include "CommonUtils/MemFileHelper.h"
#include <regex>
#include <chrono>
#include <TMessage.h>
#include <sstream>
#include <TFile.h>
#include <TGrid.h>
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

  // find out if we can can in principle connect to Alien
  mHaveAlienToken = checkAlienToken();
  if (!mHaveAlienToken) {
    LOG(WARN) << "CCDB: Did not find an alien token; Cannot serve objects located on alien://";
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

std::unique_ptr<std::vector<char>> CcdbApi::createObjectImage(const void* obj, std::type_info const& tinfo, CcdbObjectInfo* info)
{
  // Create a binary image of the object, if CcdbObjectInfo pointer is provided, register there
  // the assigned object class name and the filename
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
  return o2::utils::MemFileHelper::createFileImage(*rootObject, tmpFileName, CCDBOBJECT_ENTRY);
}

void CcdbApi::storeAsTFile_impl(const void* obj, std::type_info const& tinfo, std::string const& path,
                                std::map<std::string, std::string> const& metadata,
                                long startValidityTimestamp, long endValidityTimestamp) const
{
  // We need the TClass for this type; will verify if dictionary exists
  CcdbObjectInfo info;
  auto img = createObjectImage(obj, tinfo, &info);
  storeAsBinaryFile(img->data(), img->size(), info.getFileName(), info.getObjectType(),
                    path, metadata, startValidityTimestamp, endValidityTimestamp);
}

void CcdbApi::storeAsBinaryFile(const char* buffer, size_t size, const std::string& filename, const std::string& objectType,
                                const std::string& path, const std::map<std::string, std::string>& metadata,
                                long startValidityTimestamp, long endValidityTimestamp) const
{
  // Store a binary file

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

  // Curl preparation
  CURL* curl = nullptr;
  struct curl_httppost* formpost = nullptr;
  struct curl_httppost* lastptr = nullptr;
  struct curl_slist* headerlist = nullptr;
  static const char buf[] = "Expect:";
  curl_formadd(&formpost,
               &lastptr,
               CURLFORM_COPYNAME, "send",
               CURLFORM_BUFFER, filename.c_str(),
               CURLFORM_BUFFERPTR, buffer, //.Buffer(),
               CURLFORM_BUFFERLENGTH, size,
               CURLFORM_END);

  curl = curl_easy_init();
  headerlist = curl_slist_append(headerlist, buf);
  if (curl != nullptr) {
    string fullUrl = getFullUrlForStorage(curl, path, objectType, metadata, sanitizedStartValidityTimestamp, sanitizedEndValidityTimestamp);
    LOG(debug3) << "Full URL Encoded: " << fullUrl;
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

void CcdbApi::storeAsTFile(const TObject* rootObject, std::string const& path, std::map<std::string, std::string> const& metadata,
                           long startValidityTimestamp, long endValidityTimestamp) const
{
  // Prepare file
  CcdbObjectInfo info;
  auto img = createObjectImage(rootObject, &info);
  storeAsBinaryFile(img->data(), img->size(), info.getFileName(), info.getObjectType(), path, metadata, startValidityTimestamp, endValidityTimestamp);
}

string CcdbApi::getFullUrlForStorage(CURL* curl, const string& path, const string& objtype,
                                     const map<string, string>& metadata,
                                     long startValidityTimestamp, long endValidityTimestamp) const
{
  // Prepare timestamps
  string startValidityString = getTimestampString(startValidityTimestamp < 0 ? getCurrentTimestamp() : startValidityTimestamp);
  string endValidityString = getTimestampString(endValidityTimestamp < 0 ? getFutureTimestamp(60 * 60 * 24 * 365) : endValidityTimestamp);
  // Build URL
  string fullUrl = mUrl + "/" + path + "/" + startValidityString + "/" + endValidityString + "/";
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
string CcdbApi::getFullUrlForRetrieval(CURL* curl, const string& path, const map<string, string>& metadata, long timestamp) const
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

  // Prepare CURL
  CURL* curl_handle;
  CURLcode res;
  struct MemoryStruct chunk {
    (char*)malloc(1) /*memory*/, 0 /*size*/
  };
  TObject* result = nullptr;

  /* init the curl session */
  curl_handle = curl_easy_init();

  string fullUrl = getFullUrlForRetrieval(curl_handle, path, metadata, timestamp);

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

std::string CcdbApi::generateFileName(const std::string& inp)
{
  // generate file name for the CCDB object  (for now augment the input string by the timestamp)
  std::string str = inp;
  str += "_" + std::to_string(o2::ccdb::getCurrentTimestamp()) + ".root";
  return str;
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

TObject* CcdbApi::retrieveFromTFile(std::string const& path, std::map<std::string, std::string> const& metadata,
                                    long timestamp, std::map<std::string, std::string>* headers, std::string const& etag,
                                    const std::string& createdNotAfter, const std::string& createdNotBefore) const
{
  // Note : based on https://curl.haxx.se/libcurl/c/getinmemory.html
  // Thus it does not comply to our coding guidelines as it is a copy paste.

  //  std::map<std::string, std::string> headers2;

  // Prepare CURL
  CURL* curl_handle;
  CURLcode res;
  struct MemoryStruct chunk {
    (char*)malloc(1) /*memory*/, 0 /*size*/
  };

  /* init the curl session */
  curl_handle = curl_easy_init();

  string fullUrl = getFullUrlForRetrieval(curl_handle, path, metadata, timestamp);

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

  // setup curl for headers handling
  if (headers != nullptr) {
    list = curl_slist_append(list, ("If-None-Match: " + to_string(timestamp)).c_str());
    curl_easy_setopt(curl_handle, CURLOPT_HEADERFUNCTION, header_map_callback<>);
    curl_easy_setopt(curl_handle, CURLOPT_HEADERDATA, headers);
  }

  if (list) {
    curl_easy_setopt(curl_handle, CURLOPT_HTTPHEADER, list);
  }

  /* get it! */
  res = curl_easy_perform(curl_handle);
  std::string errStr;
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
        result = (TObject*)extractFromTFile(memFile, TClass::GetClass("TObject"));
        if (result == nullptr) {
          errStr = o2::utils::concat_string("Couldn't retrieve the object ", path);
          LOG(ERROR) << errStr;
        }
        memFile.Close();
      } else {
        LOG(DEBUG) << "Object " << path << " is stored in a TMemFile";
      }
    } else {
      errStr = o2::utils::concat_string("Invalid URL : ", fullUrl);
      LOG(ERROR) << errStr;
    }
  } else {
    errStr = o2::utils::concat_string("curl_easy_perform() failed: ", curl_easy_strerror(res));
    fprintf(stderr, "%s", errStr.c_str());
  }

  if (!errStr.empty() && headers) {
    (*headers)["Error"] = errStr;
  }

  curl_easy_cleanup(curl_handle);
  free(chunk.memory);
  return result;
}

void CcdbApi::retrieveBlob(std::string const& path, std::string const& targetdir, std::map<std::string, std::string> const& metadata, long timestamp) const
{

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

  string fullUrl = getFullUrlForRetrieval(curl_handle, path, metadata, timestamp);

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
  bool success = true;
  if (res == CURLE_OK) {
    long response_code;
    res = curl_easy_getinfo(curl_handle, CURLINFO_RESPONSE_CODE, &response_code);
    if ((res == CURLE_OK) && (response_code != 404)) {
    } else {
      LOG(ERROR) << "Invalid URL : " << fullUrl;
      success = false;
    }
  } else {
    fprintf(stderr, "curl_easy_perform() failed: %s\n", curl_easy_strerror(res));
    success = false;
  }

  if (fp) {
    fclose(fp);
  }
  curl_easy_cleanup(curl_handle);

  if (success) {
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

void* CcdbApi::extractFromTFile(TFile& file, TClass const* cl)
{
  if (!cl) {
    return nullptr;
  }
  auto object = file.GetObjectChecked(CCDBOBJECT_ENTRY, cl);
  if (!object) {
    // it could be that object was stored with previous convention
    // where the classname was taken as key
    std::string objectName(cl->GetName());
    utils::trim(objectName);
    object = file.GetObjectChecked(objectName.c_str(), cl);
    LOG(WARN) << "Did not find object under expected name " << CCDBOBJECT_ENTRY;
    if (!object) {
      return nullptr;
    }
    LOG(WARN) << "Found object under deprecated name " << cl->GetName();
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

void* CcdbApi::extractFromLocalFile(std::string const& filename, TClass const* tcl) const
{
  if (!boost::filesystem::exists(filename)) {
    LOG(INFO) << "Local snapshot " << filename << " not found \n";
    return nullptr;
  }
  TFile f(filename.c_str(), "READ");
  return extractFromTFile(f, tcl);
}

bool CcdbApi::checkAlienToken() const
{
  // a somewhat weird construction to programmatically find out if we
  // have a GRID token; Can be replaced with something more elegant once
  // alien-token-info does not ask for passwords interactively
  if (getenv("JALIEN_TOKEN_CERT")) {
    return true;
  }
  auto returncode = system("alien-token-info > /dev/null 2> /dev/null");
  return returncode == 0;
}

bool CcdbApi::initTGrid() const
{
  if (!mAlienInstance) {
    if (mHaveAlienToken) {
      mAlienInstance = TGrid::Connect("alien");
    }
  }
  return mAlienInstance != nullptr;
}

void* CcdbApi::downloadAlienContent(std::string const& url, TClass* cl) const
{
  if (!initTGrid()) {
    return nullptr;
  }
  auto memfile = TMemFile::Open(url.c_str(), "OPEN");
  if (memfile) {
    auto content = extractFromTFile(*memfile, cl);
    delete memfile;
    return content;
  }
  return nullptr;
}

void* CcdbApi::interpretAsTMemFileAndExtract(char* contentptr, size_t contentsize, TClass* tcl) const
{
  void* result = nullptr;
  Int_t previousErrorLevel = gErrorIgnoreLevel;
  gErrorIgnoreLevel = kFatal;
  TMemFile memFile("name", contentptr, contentsize, "READ");
  gErrorIgnoreLevel = previousErrorLevel;
  if (!memFile.IsZombie()) {
    result = extractFromTFile(memFile, tcl);
    if (!result) {
      LOG(ERROR) << o2::utils::concat_string("Couldn't retrieve object corresponding to ", tcl->GetName(), " from TFile");
    }
    memFile.Close();
  }
  return result;
}

// navigate sequence of URLs until TFile content is found; object is extracted and returned
void* CcdbApi::navigateURLsAndRetrieveContent(CURL* curl_handle, std::string const& url, TClass* cl, std::map<string, string>* headers) const
{
  // a global internal data structure that can be filled with HTTP header information
  // static --> to avoid frequent alloc/dealloc as optimization
  // not sure if thread_local takes away that benefit
  static thread_local std::multimap<std::string, std::string> headerData;

  // let's see first of all if the url is something specific that curl cannot handle
  if (url.find("alien:/", 0) != std::string::npos) {
    return downloadAlienContent(url, cl);
  }
  // add other final cases here
  // example root://

  // otherwise make an HTTP/CURL request
  // specify URL to get
  curl_easy_setopt(curl_handle, CURLOPT_URL, url.c_str());
  // some servers don't like requests that are made without a user-agent
  // field, so we provide one
  curl_easy_setopt(curl_handle, CURLOPT_USERAGENT, "libcurl-agent/1.0");
  // if redirected , we tell libcurl NOT to follow redirection
  curl_easy_setopt(curl_handle, CURLOPT_FOLLOWLOCATION, 0L);
  curl_easy_setopt(curl_handle, CURLOPT_HEADERFUNCTION, header_map_callback<decltype(headerData)>);
  headerData.clear();
  curl_easy_setopt(curl_handle, CURLOPT_HEADERDATA, (void*)&headerData);

  MemoryStruct chunk{(char*)malloc(1), 0};

  // send all data to this function
  curl_easy_setopt(curl_handle, CURLOPT_WRITEFUNCTION, WriteMemoryCallback);
  curl_easy_setopt(curl_handle, CURLOPT_WRITEDATA, (void*)&chunk);

  auto res = curl_easy_perform(curl_handle);
  long response_code = -1;
  void* content = nullptr;
  bool errorflag = false;
  bool cachingflag = false;
  if (res == CURLE_OK && curl_easy_getinfo(curl_handle, CURLINFO_RESPONSE_CODE, &response_code) == CURLE_OK) {
    if (headers) {
      for (auto& p : headerData) {
        (*headers)[p.first] = p.second;
      }
    }
    if (200 <= response_code && response_code < 300) {
      // good response and the content is directly provided and should have been dumped into "chunk"
      content = interpretAsTMemFileAndExtract(chunk.memory, chunk.size, cl);
    } else if (response_code == 304) {
      // this means the object exist but I am not serving
      // it since it's already in your possession

      // there is nothing to be done here
      cachingflag = true;
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
          LOG(DEBUG) << "Trying content location " << l;
          content = navigateURLsAndRetrieveContent(curl_handle, l, cl, nullptr);
          if (content /* or other success marker in future */) {
            break;
          }
        }
      }
    } else if (response_code == 404) {
      LOG(ERROR) << "Requested resource does not exist";
      errorflag = true;
    } else {
      errorflag = true;
    }
    // cleanup
    if (chunk.memory != nullptr) {
      free(chunk.memory);
    }
  } else {
    LOG(ERROR) << "Curl request to " << url << " failed ";
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
  // We need the TClass for this type; will verify if dictionary exists
  auto tcl = TClass::GetClass(tinfo);
  if (!tcl) {
    std::cerr << "Could not retrieve ROOT dictionary for type " << tinfo.name() << " aborting to read from CCDB\n";
    return nullptr;
  }

  CURL* curl_handle = curl_easy_init();
  string fullUrl = getFullUrlForRetrieval(curl_handle, path, metadata, timestamp);
  // if we are in snapshot mode we can simply open the file; extract the object and return
  if (mInSnapshotMode) {
    return extractFromLocalFile(fullUrl, tcl);
  }

  // add some global options to the curl query
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
  if (headers) {
    list = curl_slist_append(list, ("If-None-Match: " + to_string(timestamp)).c_str());
  }
  curl_easy_setopt(curl_handle, CURLOPT_HTTPHEADER, list);

  auto content = navigateURLsAndRetrieveContent(curl_handle, fullUrl, tcl, headers);
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
} // namespace

std::map<std::string, std::string> CcdbApi::retrieveHeaders(std::string const& path, std::map<std::string, std::string> const& metadata, long timestamp) const
{

  CURL* curl = curl_easy_init();
  CURLcode res;
  string fullUrl = getFullUrlForRetrieval(curl, path, metadata, timestamp);
  std::map<std::string, std::string> headers;

  if (curl != nullptr) {
    struct curl_slist* list = nullptr;
    list = curl_slist_append(list, ("If-None-Match: " + std::to_string(timestamp)).c_str());

    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, list);
    curl_easy_setopt(curl, CURLOPT_URL, fullUrl.c_str());
    /* get us the resource without a body! */
    curl_easy_setopt(curl, CURLOPT_NOBODY, 1L);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_HEADERFUNCTION, header_map_callback<>);
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
