// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_ROOTSERKEYVALUESTORE_H
#define ALICEO2_ROOTSERKEYVALUESTORE_H

#include <map>
#include <string>
#include <TClass.h>
#include <Rtypes.h>
#include <typeinfo>
#include <typeindex>
#include <TBufferFile.h>
#include <type_traits>

namespace o2
{
namespace utils
{

/// A ROOT serializable container mapping a key (string) to an object of arbitrary type.
///
/// The container allows to group objects in heterogeneous type in a key-value container.
/// The container can be ROOT serialized which adds on-top of existing solutions such as boost::property_tree.
/// This may be useful in various circumenstances, for
/// instance to assemble various CCDB objects into a single aggregate to reduce the number of CCDB files/entries.
class RootSerializableKeyValueStore
{
 public:
  /// Structure encapsulating the stored information: raw buffers and attached type information (combination of type_index_hash and TClass information)
  struct SerializedInfo {
    void* objptr = nullptr; //! pointer to existing object in memory
    Int_t N = 0;
    char* bufferptr = nullptr; //[N]  pointer to serialized buffer

    // we use the TClass and/or the type_index_hash for type idendification
    TClass* cl = nullptr;
    std::string typeinfo_name; // typeinfo name that can be used to store type if TClass not available (for PODs!)
    ClassDefNV(SerializedInfo, 1);
  };

  enum class GetState {
    kOK = 0,
    kNOSUCHKEY = 1,
    kWRONGTYPE = 2,
    kNOTCLASS = 3
  };

 private:
  static constexpr const char* GetStateString[4] = {
    "ok",
    "no such key",
    "wrong type",
    "no TClass"};

 public:
  static const char* getStateString(GetState state)
  {
    return (int)state < 4 ? GetStateString[(int)state] : nullptr;
  };

  RootSerializableKeyValueStore() = default;

  /// Putting a value (and overrides previous entries)
  /// T needs to be trivial non-pointer type or a type having a ROOT TClass instance
  template <typename T>
  void put(std::string const& key, T const& value)
  {
    remove_entry(key);
    GetState s;
    return put_impl(key, value, s, int{});
  }

  /// returns object pointer for this key or nullptr if error or does not exist.
  template <typename T>
  const T* get(std::string const& key) const
  {
    GetState s;
    return get_impl<T>(key, s, int{});
  }

  /// returns object pointer for this key or nullptr if error or does not exist; state is set with meaningful error code
  template <typename T>
  const T* get(std::string const& key, GetState& state) const
  {
    return get_impl<T>(key, state, int{});
  }

  /// get interface returning a const reference instead of pointer; Error handling/detection is done via the state argument;
  /// Beware: In case of errors, a default object is returned.
  template <typename T>
  const T& getRef(std::string const& key, GetState& state) const
  {
    auto ptr = get_impl<T>(key, state, int{});
    if (ptr) {
      return *ptr;
    } else {
      static T t = T();
      // in case of error we return a default object
      return t;
    }
  }

  /// checks if a key exists
  bool has(std::string const& key) const
  {
    return mStore.find(key) != mStore.end();
  }

  /// clear the store
  void clear()
  {
    mStore.clear();
  }

  /// print list of keys and type information
  void print() const;

 private:
  std::map<std::string, SerializedInfo*> mStore;

  // generic implementation for put relying on TClass
  template <typename T>
  void put_impl(std::string const& key, T const& value, GetState& state, ...)
  {
    // make sure we have a TClass for this
    // if there is a TClass, we'll use ROOT serialization to encode into the buffer
    auto ptr = new T(value);
    auto cl = TClass::GetClass(typeid(value));
    if (!cl) {
      state = GetState::kNOTCLASS;
      return;
    }
    char* bufferptr = nullptr;
    TBufferFile buff(TBuffer::kWrite);
    buff.WriteObjectAny(ptr, cl);
    int N = buff.Length();
    bufferptr = new char[N];
    memcpy(bufferptr, buff.Buffer(), sizeof(char) * N);

    auto name = std::type_index(typeid(value)).name();
    mStore.insert(std::pair<std::string, SerializedInfo*>(key, new SerializedInfo{(void*)ptr, N, (char*)bufferptr, cl, name}));
  }

  // implementation for put for trivial types
  template <typename T, typename std::enable_if<std::is_trivial<T>::value, T>::type* = nullptr>
  void put_impl(std::string const& key, T const& value, GetState& state, int)
  {
    // we forbid pointers
    static_assert(!std::is_pointer<T>::value);
    // serialization of trivial types is easy (not based on ROOT)
    auto ptr = new T(value);
    int N = sizeof(T);
    auto bufferptr = new char[N];
    memcpy(bufferptr, (char*)ptr, sizeof(char) * N);

    auto name = std::type_index(typeid(value)).name();
    mStore.insert(std::pair<std::string, SerializedInfo*>(key, new SerializedInfo{(void*)ptr, N, (char*)bufferptr, nullptr, name}));
  }

  // generic implementation for get relying on TClass
  template <typename T>
  const T* get_impl(std::string const& key, GetState& state, ...) const
  {
    state = GetState::kOK;
    auto iter = mStore.find(key);
    if (iter != mStore.end()) {
      auto value = iter->second;
      auto cl = TClass::GetClass(typeid(T));
      if (!cl) {
        state = GetState::kNOTCLASS;
        return nullptr;
      }
      if (value->cl && strcmp(cl->GetName(), value->cl->GetName()) == 0) {
        // if there is a (cached) object pointer ... we return it
        if (value->objptr) {
          return (T*)value->objptr;
        }
        // do this only once and cache instance into value.objptr
        TBufferFile buff(TBuffer::kRead, value->N, value->bufferptr, false, nullptr);
        buff.Reset();
        auto instance = (T*)buff.ReadObjectAny(cl);
        value->objptr = instance;
        return (T*)value->objptr;
      } else {
        state = GetState::kWRONGTYPE;
        return nullptr;
      }
    }
    state = GetState::kNOSUCHKEY;
    return nullptr;
  }

  // implementation for standard POD types
  template <typename T, typename std::enable_if<std::is_trivial<T>::value, T>::type* = nullptr>
  const T* get_impl(std::string const& key, GetState& state, int) const
  {
    state = GetState::kOK;
    auto iter = mStore.find(key);
    if (iter != mStore.end()) {
      auto value = iter->second;
      if (strcmp(std::type_index(typeid(T)).name(), value->typeinfo_name.c_str()) == 0) {
        // if there is a (cached) object pointer ... we return it
        if (value->objptr) {
          return (T*)value->objptr;
        }
        value->objptr = (T*)value->bufferptr;
        return (T*)value->objptr;
      } else {
        state = GetState::kWRONGTYPE;
        return nullptr;
      }
    }
    state = GetState::kNOSUCHKEY;
    return nullptr;
  }

  // removes a previous entry
  void remove_entry(std::string const& key)
  {
    auto iter = mStore.find(key);
    if (iter != mStore.end()) {
      delete iter->second;
      mStore.erase(iter);
    }
  }

  ClassDefNV(RootSerializableKeyValueStore, 1);
};

} // namespace utils
} // namespace o2

#endif
