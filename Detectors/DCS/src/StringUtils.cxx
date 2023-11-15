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

#include <cctype>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <string>
#include <random>
#include <vector>
#include <list>
#include <memory>
#include <fstream>
#include <regex>
#include <map>
#include <functional>
#include <utility>
#include <sstream>
#include <istream>
#include <iomanip>
#include <sstream>
#include <iterator>
#include "DetectorsDCS/StringUtils.h"

using namespace std;

random_device dev;
default_random_engine gen(dev());
uniform_int_distribution<long long> uniform_dist(32, 126);
geometric_distribution<long long> geom_dist(0.1);

constexpr char ALPHABET[39]{
  'E', 'T', 'A', 'O', 'I', '_', 'N', 'S', 'H', 'R',
  '/', 'D', 'L', '1', '2', 'C', 'U', 'M', 'W', '3',
  '4', '5', '6', '7', '8', '9', '0', 'F', 'G', 'Y',
  'P', 'B', 'V', 'K', 'J', 'X', 'Q', 'Z', '?'};

/**
 * The first 52 prime numbers used for calculating the hash code. Each symbol of
 * the alias will be multiplied (mod 256) with a prime of the same index and
 * adding the obtained number to the sum that is the hash code.
 */
constexpr uint64_t PRIMES[52]{
  2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
  31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
  73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
  127, 131, 137, 139, 149, 151, 157, 163, 167, 173,
  179, 181, 191, 193, 197, 199, 211, 223, 227, 229,
  233, 239};

/**
 * The modulo used in the hash function after calculating the sum. It is the
 * prime number closest to 1.5 * 2^63.
 */
constexpr uint64_t MODULO(13835058055282163729ULL);

/**
 * Transforms the given character into a 8-bit unsigned code symbol.
 *
 * @param input
 * @return
 */
uint8_t lookup(const char input) noexcept
{
  switch (input) {
    case 'e':
    case 'E':
      return 0;
    case 't':
    case 'T':
      return 1;
    case 'a':
    case 'A':
      return 2;
    case 'o':
    case 'O':
      return 3;
    case 'i':
    case 'I':
      return 4;
    case '_':
      return 5;
    case 'n':
    case 'N':
      return 6;
    case 's':
    case 'S':
      return 7;
    case 'h':
    case 'H':
      return 8;
    case 'r':
    case 'R':
      return 9;
    case '/':
      return 10;
    case 'd':
    case 'D':
      return 11;
    case 'l':
    case 'L':
      return 12;
    case '1':
      return 13;
    case '2':
      return 14;
    case 'c':
    case 'C':
      return 15;
    case 'u':
    case 'U':
      return 16;
    case 'm':
    case 'M':
      return 17;
    case 'w':
    case 'W':
      return 18;
    case '3':
      return 19;
    case '4':
      return 20;
    case '5':
      return 21;
    case '6':
      return 22;
    case '7':
      return 23;
    case '8':
      return 24;
    case '9':
      return 25;
    case '0':
      return 26;
    case 'f':
    case 'F':
      return 27;
    case 'g':
    case 'G':
      return 28;
    case 'y':
    case 'Y':
      return 29;
    case 'p':
    case 'P':
      return 30;
    case 'b':
    case 'B':
      return 31;
    case 'v':
    case 'V':
      return 32;
    case 'k':
    case 'K':
      return 33;
    case 'j':
    case 'J':
      return 34;
    case 'x':
    case 'X':
      return 35;
    case 'q':
    case 'Q':
      return 36;
    case 'z':
    case 'Z':
      return 37;
    default:
      return 38;
  }
}

/**
 * An all-integer exponent function. The orgiginal algorithm was submitted as an
 * answer at StackOverflow, by Elias Yarrkow. The code was obtained from
 * http://stackoverflow.com/a/101613 in 14 February 2017, 11:24, and modified to
 * work with the unsigned C integer types. It is well-known that these kind of
 * functions usually overflow, and overflows are ignored in this case.
 *
 * @param base  The base of the exponent function.
 * @param exp   The exponent.
 * @return      The result.
 */
uint64_t exp(uint64_t base, uint8_t exp) noexcept
{
  uint64_t result(1);
  while (exp) {
    if (exp & 1) {
      result *= base;
    }
    exp >>= 1;
    base *= base;
  }
  return result;
}

string o2::dcs::random_string(const size_t length) noexcept
{
  string s;
  s.reserve(length);
  for (size_t i = 0; i < length; ++i) {
    s.push_back(uniform_dist(gen));
  }
  return s;
}

string o2::dcs::random_string2(const size_t length) noexcept
{
  string s;
  s.reserve(length);
  for (size_t i = 0; i < length; ++i) {
    s.push_back(ALPHABET[geom_dist(gen) % 39]);
  }
  return s;
}

uint64_t o2::dcs::hash_code(const std::string& input) noexcept
{
  size_t result(0);
  const size_t length(min(input.length(), (string::size_type)52));
  for (size_t i = 0; i < length; ++i) {
    //        result += exp(PRIMES[i], (uint8_t) input[i]);
    result += exp(PRIMES[i], lookup(input[i]));
    //        result += exp(PRIMES[i], ((uint8_t) input[i]) & 0x1F);
  }
  return result;
}

unique_ptr<vector<string>> o2::dcs::convert_strings(const int argc,
                                                    char** argv) noexcept
{
  vector<string>* arguments = new vector<string>();

  for (int i = 0; i < argc; ++i) {
    arguments->push_back(std::move(string(argv[i])));
  }

  return unique_ptr<vector<string>>(arguments);
}

unique_ptr<vector<string>> o2::dcs::split(const string& source,
                                          const char separator) noexcept
{
  stringstream ss(source);
  string item;
  vector<string>* substrings = new vector<string>();
  while (getline(ss, item, separator)) {
    substrings->push_back(item);
  }
  return unique_ptr<vector<string>>(substrings);
}

unique_ptr<vector<string>> o2::dcs::split_by_whitespace(const string& source) noexcept
{
  istringstream buffer(source);
  vector<string>* ret = new vector<string>();

  std::copy(istream_iterator<string>(buffer),
            istream_iterator<string>(),
            back_inserter(*ret));
  return unique_ptr<vector<string>>(ret);
}

inline char to_alpha_numeric(const uint8_t c) noexcept
{
  switch (c) {
    case 0x00:
      return '0';
    case 0x01:
    case 0x10:
      return '1';
    case 0x02:
    case 0x20:
      return '2';
    case 0x03:
    case 0x30:
      return '3';
    case 0x04:
    case 0x40:
      return '4';
    case 0x05:
    case 0x50:
      return '5';
    case 0x06:
    case 0x60:
      return '6';
    case 0x07:
    case 0x70:
      return '7';
    case 0x08:
    case 0x80:
      return '8';
    case 0x09:
    case 0x90:
      return '9';
    case 0x0A:
    case 0xA0:
      return 'A';
    case 0x0B:
    case 0xB0:
      return 'B';
    case 0x0C:
    case 0xC0:
      return 'C';
    case 0x0D:
    case 0xD0:
      return 'D';
    case 0x0E:
    case 0xE0:
      return 'E';
    case 0x0F:
    case 0xF0:
      return 'F';
    default:
      return '?';
  }
}

string o2::dcs::to_hex_big_endian(const char* const start, const size_t length) noexcept
{
  string s;
  s.reserve(length * 3);
  // My machine is little endian:
  size_t i = length - 1;
  do {
    const char next = start[i];
    s.push_back(to_alpha_numeric(next & 0xF0));
    s.push_back(to_alpha_numeric(next & 0x0F));
    s.push_back(' ');
    --i;
  } while (i < length);
  s.pop_back();
  return s;
}

string o2::dcs::to_hex_little_endian(const char* const start, const size_t length) noexcept
{
  string s;
  s.reserve(length * 3);
  // My machine is little endian:
  size_t i = 0;
  do {
    const char next = start[i];
    s.push_back(to_alpha_numeric(next & 0xF0));
    s.push_back(to_alpha_numeric(next & 0x0F));
    s.push_back(' ');
    ++i;
  } while (i < length);
  s.pop_back();
  return s;
}

void o2::dcs::print_k_v_list(
  const string& list_name,
  const list<pair<string, string>>& parameters) noexcept
{
  stringstream ss;
  ss << list_name << endl
     << string(list_name.length() + 26, '-') << endl
     << endl;
  uint32_t key_length(0);
  for (auto k_v : parameters) {
    if (k_v.first.length() > key_length) {
      key_length = k_v.first.length();
    }
  }
  key_length += 4 - (key_length % 4);
  for (auto k_v : parameters) {
    ss << "    " << k_v.first << string(key_length - k_v.first.length(), ' ') << k_v.second << endl;
  }
  ss << endl
     << "=========================================================="
        "======================"
     << endl
     << endl;
}
