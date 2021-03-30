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
/// \file   CcdbApi.h
/// \author Barthelemy von Haller, Sandro Wenzel
///

#ifndef PROJECT_CCDBAPI_H
#define PROJECT_CCDBAPI_H

#include <string>
#include <memory>
#include <map>
#include <curl/curl.h>
#include <TObject.h>
#include <TMessage.h>
#include "CCDB/CcdbObjectInfo.h"

#if !defined(__CINT__) && !defined(__MAKECINT__) && !defined(__ROOTCLING__) && !defined(__CLING__)
#include <TJAlienCredentials.h>
#else
class TJAlienCredentials;
#endif

class TFile;
class TGrid;

namespace o2
{
namespace ccdb
{

class CCDBQuery;

/**
 * Interface to the CCDB.
 * It uses Curl to talk to the REST api.
 *
 * @todo use smart pointers ?
 * @todo handle errors and exceptions
 * @todo extend code coverage
 */
class CcdbApi //: public DatabaseInterface
{
 public:
  /// \brief Default constructor
  CcdbApi() = default;
  /// \brief Default destructor
  virtual ~CcdbApi();

  /**
   * Initialize connection to CCDB
   *
   * @param host The URL to the CCDB (e.g. "ccdb-test.cern.ch:8080" or to a local snapshot "file:///tmp/CCDBSnapshot")
   */
  void init(std::string const& host);

  /**
   * Query current URL
   *
   */
  std::string const& getURL() const { return mUrl; }

  /**
   * Create a binary image of the arbitrary type object, if CcdbObjectInfo pointer is provided, register there 
   *
   * the assigned object class name and the filename
   * @param obj: Raw pointer to the object to store.
   * @param info: optinal info where assigned object name and filename will be filled
   */
  template <typename T>
  inline static std::unique_ptr<std::vector<char>> createObjectImage(const T* obj, CcdbObjectInfo* info = nullptr)
  {
    return createObjectImage(reinterpret_cast<const void*>(obj), typeid(T), info);
  }

  /**
   * Create a binary image of the TObject, if CcdbObjectInfo pointer is provided, register there 
   *
   * the assigned object class name and the filename
   * @param obj: Raw pointer to the object to store.
   * @param info: optinal info where assigned object name and filename will be filled
   */
  static std::unique_ptr<std::vector<char>> createObjectImage(const TObject* obj, CcdbObjectInfo* info = nullptr);

  /**
   * Create a binary image of the object, if CcdbObjectInfo pointer is provided, register there 
   *
   * the assigned object class name and the filename
   * @param obj: Raw pointer to the object to store.
   * @param tinfo: object type info
   * @param info: optinal info where assigned object name and filename will be filled
   */
  static std::unique_ptr<std::vector<char>> createObjectImage(const void* obj, std::type_info const& tinfo, CcdbObjectInfo* info = nullptr);

  /**
     * Store into the CCDB a TFile containing the ROOT object.
     *
     * @param rootObject Raw pointer to the object to store.
     * @param path The path where the object is going to be stored.
     * @param metadata Key-values representing the metadata for this object.
     * @param startValidityTimestamp Start of validity. If omitted, current timestamp is used.
     * @param endValidityTimestamp End of validity. If omitted, current timestamp + 1 year is used.
     */
  void storeAsTFile(const TObject* rootObject, std::string const& path, std::map<std::string, std::string> const& metadata,
                    long startValidityTimestamp = -1, long endValidityTimestamp = -1) const;

  /**
     * Store into the CCDB a TFile containing an object of type T (which needs to have a ROOT dictionary)
     *
     * @param obj Raw pointer to the object to store.
     * @param path The path where the object is going to be stored.
     * @param metadata Key-values representing the metadata for this object.
     * @param startValidityTimestamp Start of validity. If omitted, current timestamp is used.
     * @param endValidityTimestamp End of validity. If omitted, current timestamp + 1 year is used.
     */
  template <typename T>
  void storeAsTFileAny(const T* obj, std::string const& path, std::map<std::string, std::string> const& metadata,
                       long startValidityTimestamp = -1, long endValidityTimestamp = -1) const
  {
    storeAsTFile_impl(reinterpret_cast<const void*>(obj), typeid(T), path, metadata, startValidityTimestamp, endValidityTimestamp);
  }

  // interface for storing TObject via storeAsTFileAny
  void storeAsTFileAny(const TObject* rootobj, std::string const& path, std::map<std::string, std::string> const& metadata,
                       long startValidityTimestamp = -1, long endValidityTimestamp = -1) const
  {
    storeAsTFile(rootobj, path, metadata, startValidityTimestamp, endValidityTimestamp);
  }

  /**
   * Retrieve object at the given path for the given timestamp.
   *
   * @param path The path where the object is to be found.
   * @param metadata Key-values representing the metadata to filter out objects.
   * @param timestamp Timestamp of the object to retrieve. If omitted, current timestamp is used.
   * @return the object, or nullptr if none were found.
   * @deprecated in favour of retrieveFromTFileAny
   */
  TObject* retrieve(std::string const& path, std::map<std::string, std::string> const& metadata,
                    long timestamp = -1) const;

  /**
   * Retrieve object at the given path for the given timestamp.
   *
   * @param path The path where the object is to be found.
   * @param metadata Key-values representing the metadata to filter out objects.
   * @param timestamp Timestamp of the object to retrieve. If omitted, current timestamp is used.
   * @param headers Map to be populated with the headers we received, if it is not null.
   * @param optional etag from previous call
   * @param optional createdNotAfter upper time limit for the object creation timestamp (TimeMachine mode)
   * @param optional createdNotBefore lower time limit for the object creation timestamp (TimeMachine mode)
   * @return the object, or nullptr if none were found.
   * @deprecated in favour of retrieveFromTFileAny as it is not limited to TObjects.
   */
  TObject* retrieveFromTFile(std::string const& path, std::map<std::string, std::string> const& metadata,
                             long timestamp = -1, std::map<std::string, std::string>* headers = nullptr, std::string const& etag = "",
                             const std::string& createdNotAfter = "", const std::string& createdNotBefore = "") const;

  /**
   * Retrieve object at the given path for the given timestamp.
   *
   * @param path The path where the object is to be found.
   * @param metadata Key-values representing the metadata to filter out objects.
   * @param timestamp Timestamp of the object to retrieve. If omitted, current timestamp is used.
   * @param headers Map to be populated with the headers we received, if it is not null.
   * @param optional etag from previous call
   * @param optional createdNotAfter upper time limit for the object creation timestamp (TimeMachine mode)
   * @param optional createdNotBefore lower time limit for the object creation timestamp (TimeMachine mode)
   * @return the object, or nullptr if none were found or type does not match serialized type.
   */
  template <typename T>
  T* retrieveFromTFileAny(std::string const& path, std::map<std::string, std::string> const& metadata,
                          long timestamp = -1, std::map<std::string, std::string>* headers = nullptr, std::string const& etag = "",
                          const std::string& createdNotAfter = "", const std::string& createdNotBefore = "") const;

  /**
   * Delete all versions of the object at this path.
   *
   * @todo Raise an exception if no such object exist.
   * @param path
   */
  void truncate(std::string const& path) const;

  /**
   * Delete the matching version of this object.
   *
   * @todo Raise an exception if no such object exist.
   * @param path Path to the object to delete
   * @param timestamp Timestamp of the object to delete.
   */
  void deleteObject(std::string const& path, long timestamp = -1) const;

  /**
   * Return the listing of objects, and in some cases subfolders, matching this path.
   * The path can contain sql patterns (correctly encoded) or regexps.
   *
   * In the case where there is no pattern, the list of subfolders is returned along with all the objects at
   * this path. It does not work recursively and objects from the subfolders are not returned.
   *
   * In the case where there is a pattern, subfolders are not returned and any object matching the pattern will
   * be returned, including those in subfolders if they match.
   *
   * Example : Task/Detector will return objects and subfolders in /Task/Detector but not the object(s) in
   * /Task/Detector/Sub.
   * Example : Task/Detector/.* will return any object below Detector recursively.
   * Example : Te*e* will return any object matching this pattern, including Test/detector and TestSecond/A/B.
   *
   * @todo accept should use an enum class.
   * @param path The path to the folder we want to list the children of (default : top dir).
   * @param latestOnly In case there are several versions of the same object, list only the latest one.
   * @param returnFormat The format of the returned string -> one of "text/plain (default)", "application/json", "text/xml"
   * @return The listing of folder and/or objects in the format requested
   */
  std::string list(std::string const& path = "", bool latestOnly = false, std::string const& returnFormat = "text/plain") const;

  /**
   * Make a local snapshot of all valid objects, given a timestamp, of the CCDB under a given local path.
   * This is doing a recursive list and fetching the files locally.
   */
  void snapshot(std::string const& ccdbrootpath, std::string const& localDir, long timestamp) const;

  /**
   * Check whether the url is reachable.
   * @param url The url to test.
   * @return a bool indicating whether the url is reachable or not.
   */
  bool isHostReachable() const;

  /**
  * Helper function to extract the list of sub-folders from a list reply into a vector container.
  * Can be used to achieve full recursive traversal/listing of the CCDB.
  *
  * @param reply The reply that we got from a GET/browse sort of request.
  * @return The vector of sub-folders.
  */
  std::vector<std::string> parseSubFolders(std::string const& reply) const;

  /**
   * Function returning the complete list of (recursive) paths below a given top path
   *
   * @param top The top folder from which to search
   * @return The vector of all possible CCDB folders
   */
  std::vector<std::string> getAllFolders(std::string const& top) const;

  /**
  *  Simple function to retrieve the blob corresponding to some path and timestamp.
  *  Saves the blob locally to a binary file. The final path (including filename) is given by targetdir.
  */
  void retrieveBlob(std::string const& path, std::string const& targetdir, std::map<std::string, std::string> const& metadata, long timestamp) const;

  /**
   * Retrieve the headers of a CCDB entry, if it exists.
   * @param path The path where the object is to be found.
   * @param metadata Key-values representing the metadata to filter out objects.
   * @param timestamp Timestamp of the object to retrieve. If omitted, current timestamp is used.
   * @return A map containing the headers. The map is empty if no CCDB entry can be found.
   */
  std::map<std::string, std::string> retrieveHeaders(std::string const& path, std::map<std::string, std::string> const& metadata, long timestamp = -1) const;

  /**
   * A helper function to extract an object from an existing in-memory TFile
   * @param file a TFile instance
   * @param cl The TClass object describing the serialized type
   * @return raw pointer to created object
   */
  static void* extractFromTFile(TFile& file, TClass const* cl);

  /** Get headers associated to a given CCDBEntry on the server. 
   * @param url the url which refers to the objects
   * @param etag of the previous reply
   * @param headers the headers found in the request. Will be emptied when we return false.
   * @return true if the headers where updated WRT last time, false if the previous results can still be used.
   */
  static bool getCCDBEntryHeaders(std::string const& url, std::string const& etag, std::vector<std::string>& headers);

  /**
   * Extract the possible locations for a file and check whether or not
   * the current cached object is still valid.
   * @param headers the headers to be parsed
   * @param pfns the vector of pfns to be filled.
   * @param etag the etag to be updated with the new value
   */
  static void parseCCDBHeaders(std::vector<std::string> const& headers, std::vector<std::string>& pfns, std::string& etag);

  /**
   * Extracts meta-information of the query from a TFile containing the CCDB blob.
   */
  static CCDBQuery* retrieveQueryInfo(TFile&);

  /**
   * Extracts meta-information associated to the CCDB blob sitting in given TFile.
   */
  static std::map<std::string, std::string>* retrieveMetaInfo(TFile&);

  /**
   * Generates a file-name where the object will be stored (usually, from the provided class name)
   */
  static std::string generateFileName(const std::string& inp);

  constexpr static const char* CCDBQUERY_ENTRY = "ccdb_query";
  constexpr static const char* CCDBMETA_ENTRY = "ccdb_meta";
  constexpr static const char* CCDBOBJECT_ENTRY = "ccdb_object";

  /**
   * Set curl SSL options. The client still will be able to connect to non-ssl endpoints
   * @param curl curl handler
   * @return
   */
  static void curlSetSSLOptions(CURL* curl);

 private:
  /**
   * Initialize in local mode; Objects will be retrieved from snapshot
   *
   * @param snapshotpath (e.g. "/path/CCDBSnapshot/")
   */
  void initInSnapshotMode(std::string const& snapshotpath)
  {
    mSnapshotTopPath = snapshotpath;
    mInSnapshotMode = true;
  }

  /**
   * Transform and return a string representation of the given timestamp.
   *
   * @param timestamp
   * @return a string representation of the given timestamp.
   */
  std::string getTimestampString(long timestamp) const;

  /**
   * Build the full url to store an object.
   *
   * @param path The path where the object is going to be stored.
   * @param metadata Key-values representing the metadata for this object.
   * @param startValidityTimestamp Start of validity. If omitted or negative, the current timestamp is used.
   * @param endValidityTimestamp End of validity. If omitted or negative, current timestamp + 1 year is used.
   * @return The full url to store an object (url / startValidity / endValidity / [metadata &]* )
   */
  std::string getFullUrlForStorage(CURL* curl, const std::string& path, const std::string& objtype,
                                   const std::map<std::string, std::string>& metadata,
                                   long startValidityTimestamp = -1, long endValidityTimestamp = -1) const;

  /**
   * Build the full url to store an object.
   * @param path The path where the object is going to be found.
   * @param metadata Key-values representing the metadata for this object.
   * @param timestamp When the object we retrieve must be valid. If omitted or negative, the current timestamp is used.
   * @return The full url to store an object (url / startValidity / endValidity / [metadata &]* )
   */
  std::string getFullUrlForRetrieval(CURL* curl, const std::string& path, const std::map<std::string, std::string>& metadata,
                                     long timestamp = -1) const;

 public:
  /**
   * A generic method to store a binary buffer (e.g. an image of the TMemFile)
   */
  void storeAsBinaryFile(const char* buffer, size_t size, const std::string& fileName, const std::string& objectType,
                         const std::string& path, const std::map<std::string, std::string>& metadata,
                         long startValidityTimestamp, long endValidityTimestamp) const;

  /**
   * A generic helper implementation to store an obj whose type is given by a std::type_info
   */
  void storeAsTFile_impl(const void* obj1, std::type_info const& info, std::string const& path, std::map<std::string, std::string> const& metadata,
                         long startValidityTimestamp = -1, long endValidityTimestamp = -1) const;

  /**
   * A generic helper implementation to query obj whose type is given by a std::type_info
   */
  void* retrieveFromTFile(std::type_info const&, std::string const& path, std::map<std::string, std::string> const& metadata,
                          long timestamp = -1, std::map<std::string, std::string>* headers = nullptr, std::string const& etag = "",
                          const std::string& createdNotAfter = "", const std::string& createdNotBefore = "") const;

 private:
  /**
   * A helper function to extract object from a local ROOT file
   * @param filename name of ROOT file
   * @param cl The TClass object describing the serialized type
   * @return raw pointer to created object
   */
  void* extractFromLocalFile(std::string const& filename, std::type_info const& tinfo) const;

  /**
   * Helper function to download binary content from alien:// storage
   * @param fullUrl The alien URL
   * @param tcl The TClass object describing the serialized type
   * @return raw pointer to created object
   */
  void* downloadAlienContent(std::string const& fullUrl, std::type_info const& tinfo) const;

  // initialize the TGrid (Alien connection)
  bool initTGrid() const;
  // checks if an alien token is available, required to make a TGrid connection
  bool checkAlienToken() const;

  /// Queries the CCDB server and navigates through possible redirects until binary content is found; Retrieves content as instance
  /// given by tinfo if that is possible. Returns nullptr if something fails...
  void* navigateURLsAndRetrieveContent(CURL*, std::string const& url, std::type_info const& tinfo, std::map<std::string, std::string>* headers) const;

  // helper that interprets a content chunk as TMemFile and extracts the object therefrom
  void* interpretAsTMemFileAndExtract(char* contentptr, size_t contentsize, std::type_info const& tinfo) const;

  /**
   * Initialization of CURL
   */
  void curlInit();

  // convert type_info to TClass, throw on failure
  static TClass* tinfo2TClass(std::type_info const& tinfo);

  /// Base URL of the CCDB (with port)
  std::string mUrl{};
  std::string mSnapshotTopPath{};
  bool mInSnapshotMode = false;
  mutable TGrid* mAlienInstance = nullptr;                       // a cached connection to TGrid (needed for Alien locations)
  bool mHaveAlienToken = false;                                  // stores if an alien token is available
  static std::unique_ptr<TJAlienCredentials> mJAlienCredentials; // access JAliEn credentials

  ClassDefNV(CcdbApi, 1);
};

template <typename T>
T* CcdbApi::retrieveFromTFileAny(std::string const& path, std::map<std::string, std::string> const& metadata,
                                 long timestamp, std::map<std::string, std::string>* headers, std::string const& etag,
                                 const std::string& createdNotAfter, const std::string& createdNotBefore) const
{
  return static_cast<T*>(retrieveFromTFile(typeid(T), path, metadata, timestamp, headers, etag, createdNotAfter, createdNotBefore));
}

} // namespace ccdb
} // namespace o2

#endif //PROJECT_CCDBAPI_H
