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
#ifndef FRAMEWORK_VARIANT_H
#define FRAMEWORK_VARIANT_H

#include "Framework/RuntimeError.h"
#include "Framework/Array2D.h"
#include <type_traits>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <iosfwd>
#include <initializer_list>
#include <string_view>
#include <vector>
#include <string>

namespace o2::framework
{

// Do NOT insert entries in this enum, only append at the end (before "Empty"). Hyperloop depends on the order.
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
                               ArrayString,
                               Array2DInt,
                               Array2DFloat,
                               Array2DDouble,
                               LabeledArrayInt,
                               LabeledArrayFloat,
                               LabeledArrayDouble,
                               UInt8,
                               UInt16,
                               UInt32,
                               UInt64,
                               Int8,
                               Int16,
                               Empty,
                               Dict,
                               Unknown };

template <VariantType V>
constexpr auto isArray()
{
  return (V == VariantType::ArrayBool ||
          V == VariantType::ArrayDouble ||
          V == VariantType::ArrayFloat ||
          V == VariantType::ArrayInt ||
          V == VariantType::ArrayString);
}

template <VariantType V>
constexpr auto isArray2D()
{
  return (V == VariantType::Array2DInt ||
          V == VariantType::Array2DFloat ||
          V == VariantType::Array2DDouble);
}

template <VariantType V>
constexpr auto isLabeledArray()
{
  return (V == VariantType::LabeledArrayInt ||
          V == VariantType::LabeledArrayFloat ||
          V == VariantType::LabeledArrayDouble);
}

template <VariantType V>
constexpr auto isSimpleVariant()
{
  return (V == VariantType::Int) ||
         (V == VariantType::Int8) ||
         (V == VariantType::Int16) ||
         (V == VariantType::Int64) ||
         (V == VariantType::UInt8) ||
         (V == VariantType::UInt16) ||
         (V == VariantType::UInt32) ||
         (V == VariantType::UInt64) ||
         (V == VariantType::Float) ||
         (V == VariantType::Double) ||
         (V == VariantType::Bool);
}

template <typename T>
struct variant_trait : std::integral_constant<VariantType, VariantType::Unknown> {
};

#define DECLARE_VARIANT_TRAIT(_Type1_, _Type2_)                                               \
  template <>                                                                                 \
  struct variant_trait<_Type1_> : std::integral_constant<VariantType, VariantType::_Type2_> { \
  };

DECLARE_VARIANT_TRAIT(int, Int);
DECLARE_VARIANT_TRAIT(int8_t, Int8);
DECLARE_VARIANT_TRAIT(int16_t, Int16);
DECLARE_VARIANT_TRAIT(long int, Int64);
DECLARE_VARIANT_TRAIT(long long int, Int64);
DECLARE_VARIANT_TRAIT(uint8_t, UInt8);
DECLARE_VARIANT_TRAIT(uint16_t, UInt16);
DECLARE_VARIANT_TRAIT(uint32_t, UInt32);
DECLARE_VARIANT_TRAIT(uint64_t, UInt64);

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
DECLARE_VARIANT_TRAIT(std::string*, ArrayString);

DECLARE_VARIANT_TRAIT(std::vector<int>, ArrayInt);
DECLARE_VARIANT_TRAIT(std::vector<float>, ArrayFloat);
DECLARE_VARIANT_TRAIT(std::vector<double>, ArrayDouble);
DECLARE_VARIANT_TRAIT(std::vector<bool>, ArrayBool);
DECLARE_VARIANT_TRAIT(std::vector<std::string>, ArrayString);

DECLARE_VARIANT_TRAIT(Array2D<int>, Array2DInt);
DECLARE_VARIANT_TRAIT(Array2D<float>, Array2DFloat);
DECLARE_VARIANT_TRAIT(Array2D<double>, Array2DDouble);

DECLARE_VARIANT_TRAIT(LabeledArray<int>, LabeledArrayInt);
DECLARE_VARIANT_TRAIT(LabeledArray<float>, LabeledArrayFloat);
DECLARE_VARIANT_TRAIT(LabeledArray<double>, LabeledArrayDouble);

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

template <>
struct variant_array_symbol<std::string> {
  constexpr static char symbol = 's';
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
DECLARE_VARIANT_TYPE(int8_t, Int8);
DECLARE_VARIANT_TYPE(int16_t, Int16);
DECLARE_VARIANT_TYPE(int64_t, Int64);
DECLARE_VARIANT_TYPE(uint8_t, UInt8);
DECLARE_VARIANT_TYPE(uint16_t, UInt16);
DECLARE_VARIANT_TYPE(uint32_t, UInt32);
DECLARE_VARIANT_TYPE(uint64_t, UInt64);
DECLARE_VARIANT_TYPE(float, Float);
DECLARE_VARIANT_TYPE(double, Double);
DECLARE_VARIANT_TYPE(const char*, String);
DECLARE_VARIANT_TYPE(bool, Bool);

DECLARE_VARIANT_TYPE(int*, ArrayInt);
DECLARE_VARIANT_TYPE(float*, ArrayFloat);
DECLARE_VARIANT_TYPE(double*, ArrayDouble);
DECLARE_VARIANT_TYPE(bool*, ArrayBool);
DECLARE_VARIANT_TYPE(std::string*, ArrayString);

DECLARE_VARIANT_TYPE(Array2D<int>, Array2DInt);
DECLARE_VARIANT_TYPE(Array2D<float>, Array2DFloat);
DECLARE_VARIANT_TYPE(Array2D<double>, Array2DDouble);

DECLARE_VARIANT_TYPE(LabeledArray<int>, LabeledArrayInt);
DECLARE_VARIANT_TYPE(LabeledArray<float>, LabeledArrayFloat);
DECLARE_VARIANT_TYPE(LabeledArray<double>, LabeledArrayDouble);

template <VariantType type>
struct variant_array_element_type {
};

#define DECLARE_VARIANT_ARRAY_ELEMENT_TYPE(_Type1_, _Type2_) \
  template <>                                                \
  struct variant_array_element_type<VariantType::_Type2_> {  \
    using type = _Type1_;                                    \
  };

DECLARE_VARIANT_ARRAY_ELEMENT_TYPE(int, ArrayInt);
DECLARE_VARIANT_ARRAY_ELEMENT_TYPE(int, Array2DInt);
DECLARE_VARIANT_ARRAY_ELEMENT_TYPE(float, ArrayFloat);
DECLARE_VARIANT_ARRAY_ELEMENT_TYPE(float, Array2DFloat);
DECLARE_VARIANT_ARRAY_ELEMENT_TYPE(double, ArrayDouble);
DECLARE_VARIANT_ARRAY_ELEMENT_TYPE(double, Array2DDouble);
DECLARE_VARIANT_ARRAY_ELEMENT_TYPE(bool, ArrayBool);
DECLARE_VARIANT_ARRAY_ELEMENT_TYPE(std::string, ArrayString);

DECLARE_VARIANT_ARRAY_ELEMENT_TYPE(int, LabeledArrayInt);
DECLARE_VARIANT_ARRAY_ELEMENT_TYPE(float, LabeledArrayFloat);
DECLARE_VARIANT_ARRAY_ELEMENT_TYPE(double, LabeledArrayDouble);

template <VariantType V>
using variant_array_element_type_t = typename variant_array_element_type<V>::type;

template <typename S, typename T>
struct variant_helper {
  static void set(S* store, T value)
  {
    new (reinterpret_cast<T*>(store)) T{};
    *(reinterpret_cast<T*>(store)) = value;
  }
  static void set(S* store, T values, size_t size)
  {
    *reinterpret_cast<T*>(store) = reinterpret_cast<T>(std::memcpy(std::malloc(size * sizeof(std::remove_pointer_t<T>)), reinterpret_cast<void*>(values), size * sizeof(std::remove_pointer_t<T>)));
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
  using storage_t = std::aligned_union<8, int, int8_t, int16_t, int64_t,
                                       uint8_t, uint16_t, uint32_t, uint64_t,
                                       const char*, float, double, bool,
                                       int*, float*, double*, bool*,
                                       Array2D<int>, Array2D<float>, Array2D<double>,
                                       LabeledArray<int>, LabeledArray<float>, LabeledArray<double>>::type;

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

  Variant(const Variant& other);
  Variant(Variant&& other) noexcept;
  ~Variant();
  Variant& operator=(const Variant& other);
  Variant& operator=(Variant&& other) noexcept;

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

inline Variant emptyDict() { return Variant(VariantType::Dict); }

} // namespace o2::framework

#endif
