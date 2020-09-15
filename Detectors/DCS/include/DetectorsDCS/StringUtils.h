// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/*
 * File:   StringUtils.h
 * Author: John Lång (john.larry.lang@cern.ch)
 *
 * Created on 14 July 2015, 10:37
 *
 * This library contains miscellaneous functions used in this program for
 * handling (C-style) strings.
 */

#ifndef O2_DCS_STRING_UTILS_H
#define O2_DCS_STRING_UTILS_H

#include <string>
#include <vector>
#include <list>
#include <memory>

namespace o2
{
namespace dcs
{
/**
     * Returns an uniformly distributed string of exactly the given length. The
     * alphabet of the string is ASCII characters between <tt>32</tt>
     * (<tt>' '</tt>) and <tt>126</tt> (<tt>'~'</tt>), inclusive.
     *
     * @param length    Length for the random string.
     * @return          The random string.
     */
std::string random_string(const size_t length) noexcept;

/**
     * Returns a random string of exactly the given length. The alphabet for the
     * string contains upper case letters, digits and the symbols <tt>'_'</tt>,
     * <tt>'/'</tt>, and <tt>'?'</tt>. Their distribution is geometric and tries
     * to roughly approximate the symbol frequencies in ALICE DCS DIM service
     * aliases.
     *
     * @param length    Length for the random string.
     * @return          The random string.
     */
std::string random_string2(const size_t length) noexcept;

/**
     * Calculates a hash code for the given string. Up to 52 first characters of
     * the string contribute to the hash code. The hash code is case
     * insensitive, so for example <tt>"abc"</tt> and <tt>"ABC"</tt> will have
     * colliding hash codes.
     *
     * @param input A string.
     * @return      An unsigned integer.
     */
uint64_t hash_code(const std::string& input) noexcept;

/**
    * Converts the C-style strings of the command line parameters to a more
    * civilized data type.
    */
std::unique_ptr<std::vector<std::string>> convert_strings(const int argc,
                                                          char** argv) noexcept;

/**
     * Converts the given string into upper case. <em>This function will change
     * the state of the given string</em>, instead of returning a copy.
     *
     * @param str The string to be converted into upper case.
     */
inline void to_upper_case(std::string& str) noexcept
{
  for (char& c : str) {
    c = toupper(c);
  }
}

/**
     * Converts the given string into lower case. <em>This function will change
     * the state of the given string</em>, instead of returning a copy.
     *
     * @param str The string to be converted into lower case.
     */
inline void to_lower_case(std::string& str) noexcept
{
  for (char& c : str) {
    c = tolower(c);
  }
}

/**
     * Splits a string using the given separator.
     *
     * @param source    The string to be splitted.
     * @param separator The delimiting character.
     * @return          A vector containing the substrings.
     */
std::unique_ptr<std::vector<std::string>> split(const std::string& source,
                                                const char separator) noexcept;

/**
     * Splits a string using whitespace as separator. Only the non-whitespace
     * substrings will be returned.
     *
     * @param source    The string to be splitted.
     * @return          A vector containing the substrings.
     */
std::unique_ptr<std::vector<std::string>> split_by_whitespace(
  const std::string& source) noexcept;

/**
     * Returns a simple big endian hexadecimal presentation of the given
     * segment of memory.
     *
     * @param start     Address of a memory segment.
     * @param length    Length of the memory segment.
     * @return          A hexadecimal string representation of the segment in
     * bytes.
     */
std::string to_hex_big_endian(const char* const start, size_t length) noexcept;

/**
     * Returns a simple little endian hexadecimal presentation of the given
     * segment of memory.
     *
     * @param start     Address of a memory segment.
     * @param length    Length of the memory segment (in chars).
     * @return          A hexadecimal string representation of the segment in
     * bytes.
     */
std::string to_hex_little_endian(const char* const start, size_t length) noexcept;

/**
     * Prints the given list of key-value pairs.
     *
     * @param logger        The Logger instance used for output.
     * @param list_name     Name of the list to be printed as a heading.
     * @param parameters    The list to be printed
     */

void print_k_v_list(
  const std::string& list_name,
  const std::list<std::pair<std::string, std::string>>& parameters) noexcept;

} // namespace dcs
} // namespace o2

#endif /* O2_DCS_STRING_UTILS_H */
