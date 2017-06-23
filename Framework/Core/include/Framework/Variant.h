#ifndef FRAMEWORK_VARIANT_H
#define FRAMEWORK_VARIANT_H
#include <type_traits>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>

namespace o2 {
namespace framework {

enum Type {
  Int,
  Int64,
  Float,
  Double,
  String,
  Bool,
  Unknown
};

template <typename T>
struct variant_trait {
  static int type() { return Unknown; }
};

template <> struct variant_trait<int> {
  static int type() { return Type::Int; }
};
template <> struct variant_trait<int64_t> {
  static int type() { return Int64; }
};
template <> struct variant_trait<float> {
  static int type() { return Float; }
};
template <> struct variant_trait<double> {
  static int type() { return Double; }
};
template <> struct variant_trait<const char *> {
  static int type() { return String; }
};
template <> struct variant_trait<bool> {
  static int type() { return Bool; }
};

template <typename S, typename T>
struct variant_helper {
  static void set(S*store, T value)
  {
    *(reinterpret_cast<T*>(store)) = value;
  }

  static T get(S*store)
  {
    return *(reinterpret_cast<T*>(store));
  }
};

template<typename S>
struct variant_helper<S, const char *> {
  static const char *get(S* store)
  {
    return *(reinterpret_cast<const char **>(store));
  }

  static void set(S*store, const char *value)
  {
    *store = strdup(value);
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
      free(*(void **)(&mStore));
    }
  }

  template<typename T>
  T get() {
    if (mType != variant_trait<T>::type())
    {
      throw std::runtime_error("Unknown type");
    }
    return variant_helper<storage_t, T>::get(&mStore);
  }

  template<typename T>
  void set(T value) {
    return variant_helper<storage_t, T>::set(mStore, value);
  }

private:
  storage_t mStore;
  int mType;
};

}
}

#endif
