// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_FRAMEWORK_TYPEIDHELPERS_H_
#define O2_FRAMEWORK_TYPEIDHELPERS_H_

#include <cstddef>
#include <cstdint>

namespace o2::framework
{
/// helper to get constexpr unique string and typeid from a given type
/// Adapted from https://github.com/Manu343726/ctti
struct TypeIdHelpers {
  // From https://github.com/foonathan/string_id. As usually, thanks Jonathan.

  using hash_t = uint64_t;

  // See http://www.isthe.com/chongo/tech/comp/fnv/#FNV-param
  constexpr static hash_t fnv_basis = 14695981039346656037ull;
  constexpr static hash_t fnv_prime = 1099511628211ull;

  // FNV-1a 64 bit hash
  constexpr static hash_t fnv1a_hash(size_t n, const char* str, hash_t hash = fnv_basis)
  {
    return n > 0 ? fnv1a_hash(n - 1, str + 1, (hash ^ *str) * fnv_prime) : hash;
  }

  template <size_t N>
  constexpr static hash_t fnv1a_hash(const char (&array)[N])
  {
    return fnv1a_hash(N - 1, &array[0]);
  }

  struct cstring {
    template <size_t N>
    constexpr cstring(const char (&str)[N]) : str{&str[0]},
                                              length{N - 1}
    {
    }

    constexpr hash_t hash() const
    {
      return TypeIdHelpers::fnv1a_hash(length, str);
    }

    const char* str;
    size_t length;
  };
  template <typename T>
  constexpr static cstring typeName()
  {
    return {__PRETTY_FUNCTION__};
  }

  template <typename TYPE>
  constexpr static size_t uniqueId()
  {
    constexpr auto h = typeName<TYPE>().hash();
    return h;
  }
};

} // namespace o2::framework

#endif // O2_FRAMEWORK_TYPEIDHELPERS_H_
