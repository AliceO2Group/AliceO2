// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
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
#include <iostream>

namespace o2 {
namespace framework {

enum class VariantType : int {
  Int = 0,
  Int64,
  Float,
  Double,
  String,
  Bool,
  Unknown
};

template <typename T>
struct variant_trait {
  static VariantType type() { return VariantType::Unknown; }
};

template <> struct variant_trait<int> {
  static VariantType type() { return VariantType::Int; }
};
template <> struct variant_trait<int64_t> {
  static VariantType type() { return VariantType::Int64; }
};
template <> struct variant_trait<float> {
  static VariantType type() { return VariantType::Float; }
};
template <> struct variant_trait<double> {
  static VariantType type() { return VariantType::Double; }
};
template <> struct variant_trait<const char *> {
  static VariantType type() { return VariantType::String; }
};
template <> struct variant_trait<char *> {
  static VariantType type() { return VariantType::String; }
};
template <> struct variant_trait<char *const> {
  static VariantType type() { return VariantType::String; }
};
template <> struct variant_trait<const char *const> {
  static VariantType type() { return VariantType::String; }
};
template <> struct variant_trait<bool> {
  static VariantType type() { return VariantType::Bool; }
};

template <typename S, typename T>
struct variant_helper {
  static void set(S*store, T value)
  {
    *(reinterpret_cast<T*>(store)) = value;
  }

  static T get(const S*store)
  {
    return *(reinterpret_cast<const T*>(store));
  }
};

template<typename S>
struct variant_helper<S, const char *> {
  static const char *get(const S* store)
  {
    return *reinterpret_cast<const char *const*>(store);
  }

  static void set(S*store, const char *value)
  {
    *reinterpret_cast<char **>(store) = strdup(value);
  }
};

// Poor man variant class. Does not take ownership of anything passed to it.
// FIXME: we should really use C++17 std::variant when it
// comes about
class Variant {
  using storage_t = std::aligned_union<8, int, int64_t, const char *, float, double, bool>::type;
public:
  template <typename T> Variant(T value)
  : mType{variant_trait<T>::type()} {
    variant_helper<storage_t, decltype(value)>::set(&mStore, value);
  }

  ~Variant() {
    // In case we allocated a string out of bound, we
    // should delete it.
    if (mType == variant_trait<const char *>::type()) {
      free(*reinterpret_cast<void **>(&mStore));
    }
  }

  template<typename T>
  T get() const {
    if (mType != variant_trait<T>::type()) {
      throw std::runtime_error("Unknown type");
    }
    return variant_helper<storage_t, T>::get(&mStore);
  }

  template<typename T>
  void set(T value) {
    return variant_helper<storage_t, T>::set(mStore, value);
  }

  friend std::ostream& operator<< (std::ostream& oss, Variant const & val) {
    switch(val.mType) {
      case VariantType::Int:
        oss << val.get<int>();
        break;
      case VariantType::Int64:
        oss << val.get<int64_t>();
        break;
      case VariantType::Float:
        oss << val.get<float>();
        break;
      case VariantType::Double:
        oss << val.get<double>();
        break;
      case VariantType::String:
        oss << val.get<const char*>();
        break;
      case VariantType::Bool:
        oss << val.get<bool>();
        break;
      default:
        oss << "undefined";
        break;
    };
    return oss;
  }

private:
  storage_t mStore;
  VariantType mType;
};

}
}

#endif
