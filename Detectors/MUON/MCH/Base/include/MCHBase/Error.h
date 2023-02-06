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

/**
 * @file Error.h
 * @brief definition of the MCH processing errors
 * @author Philippe Pillot, Subatech
 */

#ifndef O2_MCH_BASE_ERROR_H
#define O2_MCH_BASE_ERROR_H

#include <cstdint>
#include <map>
#include <string>

namespace o2::mch
{

/** groups of MCH processing errors, each group corresponding to a processing units */
enum class ErrorGroup : uint8_t {
  Unassigned,
  Decoding,
  Filtering,
  TimeClustering,
  PreClustering,
  Clustering,
  Tracking
};

namespace internal
{
/**
 * @brief helper function to construct the error type GID (not supposed to be used outside of this header)
 * @details - the 8 most significant bits identify the error group
 *          - the other bits identify the error type within the group
 * @param group group to which this error belongs
 * @param id error UID within this group
 */
constexpr uint32_t buildTypeGID(ErrorGroup group, uint32_t id) { return (static_cast<uint32_t>(group) << 24) + id; }
} // namespace internal

/** types of MCH processing errors */
enum class ErrorType : uint32_t {
  PreClustering_MultipleDigitsInSamePad = internal::buildTypeGID(ErrorGroup::PreClustering, 0),
  PreClustering_LostDigit = internal::buildTypeGID(ErrorGroup::PreClustering, 1),
  Clustering_TooManyLocalMaxima = internal::buildTypeGID(ErrorGroup::Clustering, 0),
  Tracking_TooManyCandidates = internal::buildTypeGID(ErrorGroup::Tracking, 0),
  Tracking_TooLong = internal::buildTypeGID(ErrorGroup::Tracking, 1)
};

/**
 * returns the group to which this error type belongs
 * @param error error type
 */
constexpr ErrorGroup errorGroup(ErrorType error) { return static_cast<ErrorGroup>(static_cast<uint32_t>(error) >> 24); }

/** generic structure to handle MCH processing errors */
struct Error {
  static const std::map<ErrorGroup, std::string> groupNames;      ///< names of known error group
  static const std::map<ErrorType, std::string> typeNames;        ///< names of known error type
  static const std::map<ErrorType, std::string> typeDescriptions; ///< descriptions of known error type

  ErrorType type{0};  ///< type of processing error
  uint32_t id0 = 0;   ///< additional descriptor used for certain error types
  uint32_t id1 = 0;   ///< additional descriptor used for certain error types
  uint64_t count = 0; ///< number of occurences

  /**
   * returns the known error type names within the given group
   * @param group error group
   */
  static const std::map<ErrorType, std::string> getTypeNames(ErrorGroup group);

  /** returns the group to which this error belongs */
  ErrorGroup getGroup() const { return o2::mch::errorGroup(type); }
  /** returns the name of the group to which this error belongs */
  std::string getGroupName() const;
  /** returns the type name of this error */
  std::string getTypeName() const;
  /** returns the type description of this error */
  std::string getTypeDescription() const;
  /** returns the error message corresponding to this error */
  std::string asString() const;
};

} // namespace o2::mch

#endif // O2_MCH_BASE_ERROR_H
