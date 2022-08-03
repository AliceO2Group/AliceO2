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

#ifndef CCDB_RESPONSE_H_
#define CCDB_RESPONSE_H_

#include <Rtypes.h>
#include <map>
#include <vector>
#include <string>
#include <unordered_map>
#include "rapidjson/document.h"

namespace o2
{
namespace ccdb
{

class CCDBResponse
{
 public:
  CCDBResponse(const std::string& jsonString);
  ~CCDBResponse() = default;

  /**
   * Parses the response contained in the object to string.
   * @returns - string formatted as JSON
   */
  std::string toString();

  /**
   * The number of objects inside the document.
   */
  int objectNum;

  /**
   * Return attribute in string type.
   *
   * @param ind - index of object inside document
   * @param attributeName - name of attribute to be retrieved
   * @return - string value of the attribute
   */
  std::string getStringAttribute(int ind, std::string attributeName);

  /**
   * Merges objects with unique IDs and paths from another document into this one.
   *
   * @param other - Other CCDBResponse to be merged into this one.
   */
  void latestAndMerge(CCDBResponse* other);

  /**
   * Merges objects with unique IDs from another document into this one.
   *
   * @param other - Other CCDBResponse to be merged into this one.
   */
  void browseAndMerge(CCDBResponse* other);

  /**
   * Debug method. Print all attributes of all objects inside the document.
   */
  void printObjectAttributes(rapidjson::Document* document);

 private:
  /**
   * Rapidjson document used to store the server response.
   */
  rapidjson::Document document;

  /**
   * Debug method. Print all attributes of all objects inside the document.
   * @return - pointer to the document
   */
  rapidjson::Document* getDocument();

  /**
   * Parses a given rapidjson document into string.
   * @param document - document to be parsed into string
   * @return - string formatted as JSON
   */
  std::string JsonToString(rapidjson::Document* document);

  /**
   * Concatenates other response object into this object excluding duplicate ids.
   */
  void browse(CCDBResponse* other);

  /**
   * Updates the number of objects described by the document.
   */
  void refreshObjectNum();

  /**
   * Counts the number of objects described by the document.
   * @return - number of objectsinside the document.
   */
  size_t countObjects();

  /**
   * Removes as objects from given document using a vector of boolean values.
   * @param document - document which objects will be removed
   * @param toBeRemoved - Vector of the length of number of objects in document. Objects at indexes containing true will be removed from the document.
   */
  void removeObjects(rapidjson::Document* document, std::vector<bool> toBeRemoved);

  /**
   * Merges two rapidjson documents into one.
   * @param dstObject - destination object for the merge
   * @param srcObject - source object for the merge
   * @param allocator - allocator of the destination object
   */
  bool mergeObjects(rapidjson::Value& dstObject, rapidjson::Value& srcObject, rapidjson::Document::AllocatorType& allocator);

  /**
   * Unordered_map of <id, id> used to store the ids of all objects inside the document.
   */
  std::unordered_map<std::string, std::string> idHashmap;

  /**
   * Unordered_map of <path, path> used to store the path attribute of all objects in the document.
   */
  std::unordered_map<std::string, std::string> pathHashmap;

  /**
   * Refreshes the unordered_map of ids.
   */
  void refreshIdMap();

  /**
   * Refreshes the unordered_map of paths.
   */
  void refreshPathMap();

  ClassDefNV(CCDBResponse, 1);
};

} // namespace ccdb
} // namespace o2

#endif
