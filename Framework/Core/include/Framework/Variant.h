// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_VARIANT_H
#define FRAMEWORK_VARIANT_H
#include <type_traits>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <iosfwd>
#include <initializer_list>
#include <string_view>
#include <string>

namespace o2
{
namespace framework
{

enum class VariantType : int { Int = 0,
                               Int64,
                               Float,
                               Double,
                               String,
                               Bool,
                               Empty,
                               Unknown };

template <typename T>
struct variant_trait : std::integral_constant<VariantType, VariantType::Unknown> {
};

template <>
struct variant_trait<int> : std::integral_constant<VariantType, VariantType::Int> {
};

template <>
struct variant_trait<long int> : std::integral_constant<VariantType, VariantType::Int64> {
};

template <>
struct variant_trait<long long int> : std::integral_constant<VariantType, VariantType::Int64> {
};

template <>
struct variant_trait<float> : std::integral_constant<VariantType, VariantType::Float> {
};

template <>
struct variant_trait<double> : std::integral_constant<VariantType, VariantType::Double> {
};

template <>
struct variant_trait<const char*> : std::integral_constant<VariantType, VariantType::String> {
};

template <>
struct variant_trait<char*> : std::integral_constant<VariantType, VariantType::String> {
};

template <>
struct variant_trait<char* const> : std::integral_constant<VariantType, VariantType::String> {
};

template <>
struct variant_trait<const char* const> : std::integral_constant<VariantType, VariantType::String> {
};

template <>
struct variant_trait<std::string_view> : std::integral_constant<VariantType, VariantType::String> {
};

template <>
struct variant_trait<std::string> : std::integral_constant<VariantType, VariantType::String> {
};

template <>
struct variant_trait<bool> : std::integral_constant<VariantType, VariantType::Bool> {
};

template <typename T>
inline constexpr VariantType variant_trait_v = variant_trait<T>::value;

template <VariantType type>
struct variant_type {
};

template <>
struct variant_type<VariantType::Int> {
  using type = int;
};

template <>
struct variant_type<VariantType::Int64> {
  using type = int64_t;
};

template <>
struct variant_type<VariantType::Float> {
  using type = float;
};

template <>
struct variant_type<VariantType::Double> {
  using type = double;
};

template <>
struct variant_type<VariantType::String> {
  using type = const char*;
};

template <>
struct variant_type<VariantType::Bool> {
  using type = bool;
};

template <typename S, typename T>
struct variant_helper {
  static void set(S* store, T value) { *(reinterpret_cast<T*>(store)) = value; }

  static T get(const S* store) { return *(reinterpret_cast<const T*>(store)); }
};

template <typename S>
struct variant_helper<S, const char*> {
  static const char* get(const S* store) { return *reinterpret_cast<const char* const*>(store); }

  static void set(S* store, const char* value) { *reinterpret_cast<char**>(store) = strdup(value); }
};

template <typename S>
struct variant_helper<S, std::string_view> {
  static std::string_view get(const S* store) { return std::string_view(*reinterpret_cast<const char* const*>(store)); }

  static void set(S* store, std::string_view value) { *reinterpret_cast<char**>(store) = strdup(value.data()); }
};

template <typename S>
struct variant_helper<S, std::string> {
  static std::string get(const S* store) { return std::string(strdup(*reinterpret_cast<const char* const*>(store))); }

  static void set(S* store, std::string value) { *reinterpret_cast<char**>(store) = strdup(value.data()); }
};

// Poor man variant class. Does not take ownership of anything passed to it.
// FIXME: we should really use C++17 std::variant when it
// comes about
class Variant
{
  using storage_t = std::aligned_union<8, int, int64_t, const char*, float, double, bool>::type;

 public:
  Variant(VariantType type = VariantType::Unknown) : mType{type} {}

  template <typename T>
  Variant(T value) : mType{variant_trait_v<T>}
  {
    variant_helper<storage_t, decltype(value)>::set(&mStore, value);
  }

  template <typename T>
  Variant(std::initializer_list<T>) : mType{VariantType::Unknown}
  {
    static_assert(sizeof(T) == 0,
                  "brace-enclosed initializer list forbidden for Variant"
                  "\n did you accidentally put braces around the default value?");
  }

  Variant(const Variant& other) : mType(other.mType)
  {
    // In case this is a string we need to duplicate it to avoid
    // double deletion.
    if (mType == variant_trait_v<const char*>) {
      variant_helper<storage_t, const char*>::set(&mStore, other.get<const char*>());
    } else {
      mStore = other.mStore;
    }
  }

  Variant(Variant&& other) : mType(other.mType)
  {
    // In case this is a string we need to duplicate it to avoid
    // double deletion.
    if (mType == variant_trait_v<const char*>) {
      mStore = other.mStore;
      *reinterpret_cast<char**>(&(other.mStore)) = nullptr;
    } else {
      mStore = other.mStore;
    }
  }

  ~Variant()
  {
    // In case we allocated a string out of bound, we
    // should delete it.
    if (mType == variant_trait_v<const char*> || mType == variant_trait_v<char*>) {
      free(*reinterpret_cast<void**>(&mStore));
    }
  }

  void operator=(const Variant& other)
  {
    if (mType == variant_trait_v<const char*>) {
      variant_helper<storage_t, const char*>::set(&mStore, other.get<const char*>());
    } else {
      mStore = other.mStore;
    }
  }

  template <typename T>
  T get() const
  {
    if (mType != variant_trait_v<T>) {
      throw std::runtime_error("Mismatch between types");
    }
    return variant_helper<storage_t, T>::get(&mStore);
  }

  template <typename T>
  void set(T value)
  {
    return variant_helper<storage_t, T>::set(mStore, value);
  }

  VariantType type() const { return mType; }

 private:
  friend std::ostream& operator<<(std::ostream& oss, Variant const& val);
  storage_t mStore;
  VariantType mType;
};

} // namespace framework
} // namespace o2

#endif
