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

#include "Framework/RuntimeError.h"
#include <type_traits>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <iosfwd>
#include <initializer_list>
#include <string_view>
#include <vector>
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
                               ArrayInt,
                               ArrayFloat,
                               ArrayDouble,
                               ArrayBool,
                               Empty,
                               Unknown };

template <typename T>
struct variant_trait : std::integral_constant<VariantType, VariantType::Unknown> {
};

#define DECLARE_VARIANT_TRAIT(_Type1_, _Type2_)                                               \
  template <>                                                                                 \
  struct variant_trait<_Type1_> : std::integral_constant<VariantType, VariantType::_Type2_> { \
  };

DECLARE_VARIANT_TRAIT(int, Int);
DECLARE_VARIANT_TRAIT(long int, Int64);
DECLARE_VARIANT_TRAIT(long long int, Int64);
DECLARE_VARIANT_TRAIT(float, Float);
DECLARE_VARIANT_TRAIT(double, Double);
DECLARE_VARIANT_TRAIT(bool, Bool);

DECLARE_VARIANT_TRAIT(const char*, String);
DECLARE_VARIANT_TRAIT(char*, String);
DECLARE_VARIANT_TRAIT(char* const, String);
DECLARE_VARIANT_TRAIT(const char* const, String);
DECLARE_VARIANT_TRAIT(std::string_view, String);
DECLARE_VARIANT_TRAIT(std::string, String);

DECLARE_VARIANT_TRAIT(int*, ArrayInt);
DECLARE_VARIANT_TRAIT(float*, ArrayFloat);
DECLARE_VARIANT_TRAIT(double*, ArrayDouble);
DECLARE_VARIANT_TRAIT(bool*, ArrayBool);

DECLARE_VARIANT_TRAIT(std::vector<int>, ArrayInt);
DECLARE_VARIANT_TRAIT(std::vector<float>, ArrayFloat);
DECLARE_VARIANT_TRAIT(std::vector<double>, ArrayDouble);
DECLARE_VARIANT_TRAIT(std::vector<bool>, ArrayBool);

template <typename T>
struct variant_array_symbol {
  constexpr static char symbol = 'u';
};

template <>
struct variant_array_symbol<int> {
  constexpr static char symbol = 'i';
};

template <>
struct variant_array_symbol<float> {
  constexpr static char symbol = 'f';
};

template <>
struct variant_array_symbol<double> {
  constexpr static char symbol = 'd';
};

template <>
struct variant_array_symbol<bool> {
  constexpr static char symbol = 'b';
};

template <typename T>
inline constexpr VariantType variant_trait_v = variant_trait<T>::value;

template <VariantType type>
struct variant_type {
};

#define DECLARE_VARIANT_TYPE(_Type1_, _Type2_) \
  template <>                                  \
  struct variant_type<VariantType::_Type2_> {  \
    using type = _Type1_;                      \
  };

DECLARE_VARIANT_TYPE(int, Int);
DECLARE_VARIANT_TYPE(int64_t, Int64);
DECLARE_VARIANT_TYPE(float, Float);
DECLARE_VARIANT_TYPE(double, Double);
DECLARE_VARIANT_TYPE(const char*, String);
DECLARE_VARIANT_TYPE(bool, Bool);

DECLARE_VARIANT_TYPE(int*, ArrayInt);
DECLARE_VARIANT_TYPE(float*, ArrayFloat);
DECLARE_VARIANT_TYPE(double*, ArrayDouble);
DECLARE_VARIANT_TYPE(bool*, ArrayBool);

template <typename S, typename T>
struct variant_helper {
  static void set(S* store, T value) { *(reinterpret_cast<T*>(store)) = value; }
  static void set(S* store, T values, size_t size)
  {
    *reinterpret_cast<T*>(store) = reinterpret_cast<T>(std::memcpy(std::malloc(size * sizeof(std::remove_pointer_t<T>)), reinterpret_cast<void*>(values), size * sizeof(std::remove_pointer_t<T>)));
  }
  static void reset(S* store, T values, size_t)
  {
    *reinterpret_cast<T*>(store) = values;
  }

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

/// Variant for configuration parameter storage. Owns stored data.
class Variant
{
  using storage_t = std::aligned_union<8, int, int64_t, const char*, float, double, bool, int*, float*, double*, bool*>::type;

 public:
  Variant(VariantType type = VariantType::Unknown) : mType{type}, mSize{1} {}

  template <typename T>
  Variant(T value) : mType{variant_trait_v<T>}, mSize{1}
  {
    variant_helper<storage_t, decltype(value)>::set(&mStore, value);
  }

  template <typename T>
  Variant(T values, size_t size) : mType{variant_trait_v<T>}, mSize{size}
  {
    variant_helper<storage_t, T>::set(&mStore, values, mSize);
  }

  template <typename T>
  Variant(std::vector<T>& values) : mType{variant_trait_v<T*>}, mSize{values.size()}
  {
    variant_helper<storage_t, T*>::set(&mStore, values.data(), mSize);
  }

  template <typename T>
  Variant(std::initializer_list<T>) : mType{VariantType::Unknown}, mSize{1}
  {
    static_assert(sizeof(T) == 0,
                  "brace-enclosed initializer list forbidden for Variant"
                  "\n did you accidentally put braces around the default value?");
  }

  Variant(const Variant& other) : mType(other.mType)
  {
    // In case this is an array we need to duplicate it to avoid
    // double deletion.
    switch (mType) {
      case variant_trait_v<const char*>:
        mSize = other.mSize;
        variant_helper<storage_t, const char*>::set(&mStore, other.get<const char*>());
        return;
      case variant_trait_v<int*>:
        mSize = other.mSize;
        variant_helper<storage_t, int*>::set(&mStore, other.get<int*>(), mSize);
        return;
      case variant_trait_v<float*>:
        mSize = other.mSize;
        variant_helper<storage_t, float*>::set(&mStore, other.get<float*>(), mSize);
        return;
      case variant_trait_v<double*>:
        mSize = other.mSize;
        variant_helper<storage_t, double*>::set(&mStore, other.get<double*>(), mSize);
        return;
      case variant_trait_v<bool*>:
        mSize = other.mSize;
        variant_helper<storage_t, bool*>::set(&mStore, other.get<bool*>(), mSize);
        return;
      default:
        mStore = other.mStore;
        mSize = other.mSize;
    }
  }

  Variant(Variant&& other) : mType(other.mType)
  {
    mStore = other.mStore;
    mSize = other.mSize;
    switch (mType) {
      case variant_trait_v<const char*>:
        *reinterpret_cast<char**>(&(other.mStore)) = nullptr;
        return;
      case variant_trait_v<int*>:
        *reinterpret_cast<int**>(&(other.mStore)) = nullptr;
        return;
      case variant_trait_v<float*>:
        *reinterpret_cast<float**>(&(other.mStore)) = nullptr;
        return;
      case variant_trait_v<double*>:
        *reinterpret_cast<double**>(&(other.mStore)) = nullptr;
        return;
      case variant_trait_v<bool*>:
        *reinterpret_cast<bool**>(&(other.mStore)) = nullptr;
        return;
      default:
        return;
    }
  }

  ~Variant()
  {
    // In case we allocated an array, we
    // should delete it.
    switch (mType) {
      case variant_trait_v<const char*>:
      case variant_trait_v<int*>:
      case variant_trait_v<float*>:
      case variant_trait_v<double*>:
      case variant_trait_v<bool*>:
        if (reinterpret_cast<void**>(&mStore) != nullptr) {
          free(*reinterpret_cast<void**>(&mStore));
        }
        return;
      default:
        return;
    }
  }

  void operator=(const Variant& other)
  {
    switch (mType) {
      case variant_trait_v<const char*>:
        mSize = other.mSize;
        variant_helper<storage_t, const char*>::set(&mStore, other.get<const char*>());
        return;
      case variant_trait_v<int*>:
        mSize = other.mSize;
        variant_helper<storage_t, int*>::set(&mStore, other.get<int*>(), mSize);
        return;
      case variant_trait_v<float*>:
        mSize = other.mSize;
        variant_helper<storage_t, float*>::set(&mStore, other.get<float*>(), mSize);
        return;
      case variant_trait_v<double*>:
        mSize = other.mSize;
        variant_helper<storage_t, double*>::set(&mStore, other.get<double*>(), mSize);
        return;
      case variant_trait_v<bool*>:
        mSize = other.mSize;
        variant_helper<storage_t, bool*>::set(&mStore, other.get<bool*>(), mSize);
        return;
      default:
        mStore = other.mStore;
        mSize = other.mSize;
    }
  }

  template <typename T>
  T get() const
  {
    if (mType != variant_trait_v<T>) {
      throw runtime_error("Mismatch between types");
    }
    return variant_helper<storage_t, T>::get(&mStore);
  }

  template <typename T>
  void set(T value)
  {
    return variant_helper<storage_t, T>::set(&mStore, value);
  }

  template <typename T>
  void set(T value, size_t size)
  {
    mSize = size;
    return variant_helper<storage_t, T>::set(&mStore, value, mSize);
  }

  template <typename T>
  void set(std::vector<T>& values)
  {
    return variant_helper<storage_t, T*>::set(&mStore, values.data(), values.size());
  }

  VariantType type() const { return mType; }
  size_t size() const { return mSize; }
  std::string asString() const;

 private:
  friend std::ostream& operator<<(std::ostream& oss, Variant const& val);
  storage_t mStore;
  VariantType mType;
  size_t mSize = 1;
};

} // namespace framework
} // namespace o2

#endif
