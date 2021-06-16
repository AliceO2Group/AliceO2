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

/* 
 * File:   GenericFunctions.h
 * Author: John Lång (john.larry.lang@cern.ch)
 *
 * Created on 9 June 2017, 17:29
 */

#ifndef O2_DCS_GENERIC_FUNCTIONS_H
#define O2_DCS_GENERIC_FUNCTIONS_H

#include <string>

namespace o2
{
namespace dcs
{
/**
     * This function template is used for converting ADAPRO enumerated values
     * into strings in a fashion similar to the function show in the Show type 
     * class in Haskell. The exact implementation depends on the specialization.
     * 
     * @param input A T value to be converted.
     * @return      A string representation of the given value.
     * @throws std::domain_error The specialized function may throw a domain
     * error if applied with an invalid input value.
     */
template <typename T>
std::string show(const T input);

/**
     * This function template is used for parsing strings as ADAPRO enumerated
     * values. The exact implementation depends on the specialization.
     * 
     * @param input A string to be interpreted as a T value.
     * @return      The T value corresponding with the input.
     * @throws std::domain_error The specialized function may throw a domain
     * error if the given string couldn't be converted into a T value.
     */
template <typename T>
T read(const std::string& input);
} // namespace dcs
} // namespace o2

#endif /* O2_DCS_GENERIC_FUNCTIONS_H */
