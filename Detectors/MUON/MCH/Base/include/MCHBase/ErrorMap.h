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

#include <map>
#include <set>
#include <cstdint>
#include <functional>

namespace o2::mch
{

/** A container class to summarize errors encountered during processing.
 *
 * The interface is :
 * add(errorType, id0, id1)
 *
 * where errorType, id0 and id1 are integers (unsigned, 32 bits wide)
 *
 * ErrorMap stores the number of times the add method has been
 * called for the {errorType,id0,id1} triplet.
 *
 * The exact meaning of the triplet members is left to the client of ErrorMap.
 *
 */
class ErrorMap
{
 public:
  /* ErrorFunction is a function that receive a triplet {errorType,id0,id1)
   * and the number of times (count) that triplet has been seen.
   */
  using ErrorFunction = std::function<void(uint32_t errorType,
                                           uint32_t id0,
                                           uint32_t id1,
                                           uint64_t count)>;

  /* increment the count of the {errorType,id0,id1} triplet by one.*/
  void add(uint32_t errorType, uint32_t id0, uint32_t id1);

  /* execute function f on all {errorType,id0,id1} triplets.
   *
   * The function is passed the triplet and the corresponding occurence count
   * of that triplet.
   */
  void forEach(ErrorFunction f) const;

 private:
  std::map<uint32_t, std::map<uint64_t, uint64_t>> mErrorCounts;
};

/* convenience function to get the number of error types */
uint64_t numberOfErrorTypes(const ErrorMap& em);

/* convenience function to get the total number of errors */
uint64_t totalNumberOfErrors(const ErrorMap& em);

}; // namespace o2::mch

#endif
