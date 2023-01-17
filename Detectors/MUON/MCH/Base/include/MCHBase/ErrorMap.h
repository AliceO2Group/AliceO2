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

#ifndef O2_MCH_BASE_ERROR_MAP_H_H
#define O2_MCH_BASE_ERROR_MAP_H_H

#include <cstdint>
#include <functional>
#include <map>

#include <gsl/span>

#include "MCHBase/Error.h"

namespace o2::mch
{

/** @brief A container class to summarize errors encountered during processing.
 *
 * @details The main interface is :
 * add(errorType, id0, id1[, count])
 *
 * where errorType is the type of the error and id0 and id1 are additional
 * descriptors, whose meaning depends on the error type (see Error.h/cxx)
 *
 * additional interfaces are provided to add and access the errors,
 * or execute a function on all or some of them
 */
class ErrorMap
{
 public:
  using ErrorFunction = std::function<void(Error error)>;

  /** increment the count of the {errorType,id0,id1} triplet by n */
  void add(ErrorType errorType, uint32_t id0, uint32_t id1, uint64_t n = 1);
  /** add or increment this error */
  void add(Error error);
  /** add or increment these errors */
  void add(gsl::span<const Error> errors);
  /** add or increment these errors */
  void add(const ErrorMap& errors);

  /** erase all encountered errors */
  void clear() { mErrors.clear(); }

  /** return the number of encountered types of error */
  uint64_t getNumberOfErrorTypes() const { return mErrors.size(); }
  /** return the total number of encountered errors */
  uint64_t getNumberOfErrors() const;
  /** return the total number of encountered errors of a given type */
  uint64_t getNumberOfErrors(ErrorType type) const;
  /** return the total number of encountered errors of a given group */
  uint64_t getNumberOfErrors(ErrorGroup group) const;

  /** execute function f on all encountered errors */
  void forEach(ErrorFunction f) const;
  /** execute function f on all encountered errors of a given type */
  void forEach(ErrorType type, ErrorFunction f) const;
  /** execute function f on all encountered errors of a given group */
  void forEach(ErrorGroup group, ErrorFunction f) const;

 private:
  std::map<ErrorType, std::map<uint64_t, Error>> mErrors{}; ///< map of encountered errors
};

} // namespace o2::mch

#endif
