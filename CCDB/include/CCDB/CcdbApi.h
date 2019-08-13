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
/// \author Barthelemy von Haller
///

#ifndef PROJECT_CCDBAPI_H
#define PROJECT_CCDBAPI_H

#include <string>
#include <memory>
#include <iostream>
#include <map>
#include <curl/curl.h>
#include <TObject.h>
#include <TMessage.h>

namespace o2
{
namespace ccdb
{

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
   * @param host The URL to the CCDB (e.g. "ccdb-test.cern.ch:8080")
   */
  void init(std::string const& host);

  /**
   * Stores an object in the CCDB as a streamed object, not a TFile.
   *
   * @param rootObject Raw pointer to the object to store.
   * @param path The path where the object is going to be stored.
   * @param metadata Key-values representing the metadata for this object.
   * @param startValidityTimestamp Start of validity. If omitted, current timestamp is used.
   * @param endValidityTimestamp End of validity. If omitted, current timestamp + 1 year is used.
   */
  void store(TObject* rootObject, std::string const& path, std::map<std::string, std::string> const& metadata,
             long startValidityTimestamp = -1, long endValidityTimestamp = -1, bool storeStreamerInfo = false) const;

  /**
     * Store into the CCDB a TFile containing the ROOT object.
     *
     * @param rootObject Raw pointer to the object to store.
     * @param path The path where the object is going to be stored.
     * @param metadata Key-values representing the metadata for this object.
     * @param startValidityTimestamp Start of validity. If omitted, current timestamp is used.
     * @param endValidityTimestamp End of validity. If omitted, current timestamp + 1 year is used.
     */
  void storeAsTFile(TObject* rootObject, std::string const& path, std::map<std::string, std::string> const& metadata,
                    long startValidityTimestamp = -1, long endValidityTimestamp = -1) const;

  /**
     * Store into the CCDB a TFile containing the ROOT object.
     *
     * @param rootObject Raw pointer to the object to store.
     * @param path The path where the object is going to be stored.
     * @param metadata Key-values representing the metadata for this object.
     * @param startValidityTimestamp Start of validity. If omitted, current timestamp is used.
     * @param endValidityTimestamp End of validity. If omitted, current timestamp + 1 year is used.
     */
  template <typename T>
  void storeAsTFileAny(T* obj, std::string const& path, std::map<std::string, std::string> const& metadata,
                       long startValidityTimestamp = -1, long endValidityTimestamp = -1) const;

  /**
   * Retrieve object at the given path for the given timestamp.
   *
   * @param path The path where the object is to be found.
   * @param metadata Key-values representing the metadata to filter out objects.
   * @param timestamp Timestamp of the object to retrieve. If omitted, current timestamp is used.
   * @return the object, or nullptr if none were found.
   */
  TObject* retrieve(std::string const& path, std::map<std::string, std::string> const& metadata,
                    long timestamp = -1) const;

  /**
   * Retrieve object at the given path for the given timestamp.
   *
   * @param path The path where the object is to be found.
   * @param metadata Key-values representing the metadata to filter out objects.
   * @param timestamp Timestamp of the object to retrieve. If omitted, current timestamp is used.
   * @return the object, or nullptr if none were found.
   */
  TObject* retrieveFromTFile(std::string const& path, std::map<std::string, std::string> const& metadata,
                             long timestamp = -1) const;

  template <typename T>
  T* retrieveFromTFileAny(std::string const& path, std::map<std::string, std::string> const& metadata, long timestamp = -1) const;

  //    std::vector<std::string> getListOfTasksWithPublications();
  //    std::vector<std::string> getPublishedObjectNames(std::string taskName);

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
   * Check whether the url is reachable.
   * @param url The url to test.
   * @return a bool indicating whether the url is reachable or not.
   */
  bool isHostReachable() const;

 private:
  /**
   * Get the current timestamp.
   *
   * @return the current timestamp as a long
   */
  long getCurrentTimestamp() const;
  /**
   * Transform and return a string representation of the given timestamp.
   *
   * @param timestamp
   * @return a string representation of the given timestamp.
   */
  std::string getTimestampString(long timestamp) const;
  /**
   * Compute and return a timestamp X seconds in the future.
   *
   * @param secondsInFuture The number of seconds in the future.
   * @return the future timestamp
   */
  long getFutureTimestamp(int secondsInFuture) const;

  /**
   * Build the full url to store an object.
   *
   * @param path The path where the object is going to be stored.
   * @param metadata Key-values representing the metadata for this object.
   * @param startValidityTimestamp Start of validity. If omitted or negative, the current timestamp is used.
   * @param endValidityTimestamp End of validity. If omitted or negative, current timestamp + 1 year is used.
   * @return The full url to store an object (url / startValidity / endValidity / [metadata &]* )
   */
  std::string getFullUrlForStorage(const std::string& path, const std::map<std::string, std::string>& metadata,
                                   long startValidityTimestamp = -1, long endValidityTimestamp = -1) const;

  /**
   * Build the full url to store an object.
   * @param path The path where the object is going to be found.
   * @param metadata Key-values representing the metadata for this object.
   * @param timestamp When the object we retrieve must be valid. If omitted or negative, the current timestamp is used.
   * @return The full url to store an object (url / startValidity / endValidity / [metadata &]* )
   */
  std::string getFullUrlForRetrieval(const std::string& path, const std::map<std::string, std::string>& metadata,
                                     long timestamp = -1) const;

  /**
   * A generic helper implementation to store an obj whose type is given by a std::type_info
   */
  void storeAsTFile_impl(void* obj, std::type_info const& info, std::string const& path, std::map<std::string, std::string> const& metadata,
                         long startValidityTimestamp = -1, long endValidityTimestamp = -1) const;

  /**
   * A generic helper implementation to query obj whose type is given by a std::type_info
   */
  void* retrieveFromTFile(std::type_info const&, std::string const& path, std::map<std::string, std::string> const& metadata,
                          long timestamp = -1) const;

  /**
   * Initialization of CURL
   */
  void curlInit();

  /// Base URL of the CCDB (with port)
  std::string mUrl{};

  ClassDefNV(CcdbApi, 1);
};

template <typename T>
inline void CcdbApi::storeAsTFileAny(T* obj, std::string const& path, std::map<std::string, std::string> const& metadata,
                                     long startValidityTimestamp, long endValidityTimestamp) const
{
  // get the type_info and dispatch to generic (untyped) implementation
  storeAsTFile_impl(reinterpret_cast<void*>(obj), typeid(T), path, metadata,
                    startValidityTimestamp, endValidityTimestamp);
}

template <typename T>
T* CcdbApi::retrieveFromTFileAny(std::string const& path, std::map<std::string, std::string> const& metadata, long timestamp) const
{
  return static_cast<T*>(retrieveFromTFile(typeid(T), path, metadata, timestamp));
}

} // namespace ccdb
} // namespace o2

#endif //PROJECT_CCDBAPI_H
