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
                               Empty,
                               Unknown };

template <VariantType V>
constexpr auto isArray()
{
  return (V == VariantType::ArrayBool || V == VariantType::ArrayDouble || V == VariantType::ArrayFloat || V == VariantType::ArrayInt || V == VariantType::ArrayString);
}

template <VariantType V>
constexpr auto isArray2D()
{
  return (V == VariantType::Array2DInt || V == VariantType::Array2DFloat || V == VariantType::Array2DDouble);
}

template <VariantType V>
constexpr auto isLabeledArray()
{
  return (V == VariantType::LabeledArrayInt || V == VariantType::LabeledArrayFloat || V == VariantType::LabeledArrayDouble);
}

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
DECLARE_VARIANT_TYPE(int64_t, Int64);
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
  using storage_t = std::aligned_union<8, int, int64_t, const char*, float, double, bool,
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
      case variant_trait_v<std::string*>:
        mSize = other.mSize;
        variant_helper<storage_t, std::string*>::set(&mStore, other.get<std::string*>(), mSize);
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
      case variant_trait_v<std::string*>:
        *reinterpret_cast<std::string**>(&(other.mStore)) = nullptr;
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
      case variant_trait_v<std::string*>:
        if (reinterpret_cast<void**>(&mStore) != nullptr) {
          free(*reinterpret_cast<void**>(&mStore));
        }
        return;
      default:
        return;
    }
  }

  Variant& operator=(const Variant& other)
  {
    mSize = other.mSize;
    mType = other.mType;
    switch (mType) {
      case variant_trait_v<const char*>:
        variant_helper<storage_t, const char*>::set(&mStore, other.get<const char*>());
        return *this;
      case variant_trait_v<int*>:
        variant_helper<storage_t, int*>::set(&mStore, other.get<int*>(), mSize);
        return *this;
      case variant_trait_v<float*>:
        variant_helper<storage_t, float*>::set(&mStore, other.get<float*>(), mSize);
        return *this;
      case variant_trait_v<double*>:
        variant_helper<storage_t, double*>::set(&mStore, other.get<double*>(), mSize);
        return *this;
      case variant_trait_v<bool*>:
        variant_helper<storage_t, bool*>::set(&mStore, other.get<bool*>(), mSize);
        return *this;
      case variant_trait_v<std::string*>:
        variant_helper<storage_t, std::string*>::set(&mStore, other.get<std::string*>(), mSize);
        return *this;
      default:
        mStore = other.mStore;
        return *this;
    }
  }

  Variant& operator=(Variant&& other)
  {
    mSize = other.mSize;
    mType = other.mType;
    switch (mType) {
      case variant_trait_v<const char*>:
        variant_helper<storage_t, const char*>::set(&mStore, other.get<const char*>());
        *reinterpret_cast<char**>(&(other.mStore)) = nullptr;
        return *this;
      case variant_trait_v<int*>:
        variant_helper<storage_t, int*>::set(&mStore, other.get<int*>(), mSize);
        *reinterpret_cast<int**>(&(other.mStore)) = nullptr;
        return *this;
      case variant_trait_v<float*>:
        variant_helper<storage_t, float*>::set(&mStore, other.get<float*>(), mSize);
        *reinterpret_cast<float**>(&(other.mStore)) = nullptr;
        return *this;
      case variant_trait_v<double*>:
        variant_helper<storage_t, double*>::set(&mStore, other.get<double*>(), mSize);
        *reinterpret_cast<double**>(&(other.mStore)) = nullptr;
        return *this;
      case variant_trait_v<bool*>:
        variant_helper<storage_t, bool*>::set(&mStore, other.get<bool*>(), mSize);
        *reinterpret_cast<bool**>(&(other.mStore)) = nullptr;
        return *this;
      case variant_trait_v<std::string*>:
        variant_helper<storage_t, std::string*>::set(&mStore, other.get<std::string*>(), mSize);
        *reinterpret_cast<std::string**>(&(other.mStore)) = nullptr;
        return *this;
      default:
        mStore = other.mStore;
        return *this;
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

} // namespace o2::framework

#endif
