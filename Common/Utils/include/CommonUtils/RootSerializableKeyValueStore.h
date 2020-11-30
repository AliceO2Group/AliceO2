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
#include <iostream>
#include <TBufferFile.h>

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
    size_t type_index_hash = 0;
    ClassDefNV(SerializedInfo, 1);
  };

  RootSerializableKeyValueStore() = default;

  /// putting a value
  /// TODO: list the constraints on T
  template <typename T>
  void put(std::string const& key, T const& value);

  /// returns object for this key or nullptr if error or does not exist
  /// TODO: error handling/info via some enum?
  template <typename T>
  const T* get(std::string const& key) const;

 private:
  std::map<std::string, SerializedInfo*> mStore;

  ClassDefNV(RootSerializableKeyValueStore, 1);
};

template <typename T>
inline void RootSerializableKeyValueStore::put(std::string const& key, T const& value)
{
  auto ptr = new T(value);
  auto cl = TClass::GetClass(typeid(value));

  // if there is a TClass, we'll use ROOT serialization to encode into the buffer
  // TODO: do this only upon streaming (in a custom streamer)
  int N = 0;
  char* bufferptr = nullptr;
  if (cl) {
    TBufferFile buff(TBuffer::kWrite);
    buff.WriteObjectAny(ptr, cl);
    N = buff.Length();
    bufferptr = new char[N];
    memcpy(bufferptr, buff.Buffer(), sizeof(char) * N);
  } else {
    // TODO: static assert that this is a pod
    N = sizeof(T);
    bufferptr = new char[N];
    memcpy(bufferptr, (char*)ptr, sizeof(char) * N);
  }

  auto hash = std::type_index(typeid(value)).hash_code();
  mStore.insert(std::pair<std::string, SerializedInfo*>(key, new SerializedInfo{(void*)ptr, N, (char*)bufferptr, cl, hash}));
}

template <typename T>
inline const T* RootSerializableKeyValueStore::get(std::string const& key) const
{
  auto iter = mStore.find(key);
  if (iter != mStore.end()) {
    auto value = iter->second;
    std::cerr << "{" << value->objptr << " , " << value->N << " , " << (void*)value->bufferptr << " , " << value->cl << " , " << value->type_index_hash << "}\n";
    auto cl = TClass::GetClass(typeid(T));
    if (std::type_index(typeid(T)).hash_code() == value->type_index_hash) {
      // if there is a (cached) object pointer ... we return it
      if (value->objptr) {
        return (T*)value->objptr;
      }

      // if we have TClass ... unpack first of all
      if (value->cl) {
        // do this only once and cache instance into value.objptr
        TBufferFile buff(TBuffer::kRead, value->N, value->bufferptr, false, nullptr);
        buff.Reset();
        auto instance = (T*)buff.ReadObjectAny(cl);
        value->objptr = instance;
        return (T*)value->objptr;
      } else {
        value->objptr = (T*)value->bufferptr;
        return (T*)value->objptr;
      }
    } else {
      std::cerr << "type does not match\n";
      return nullptr;
    }
  }
  std::cerr << "no such key " << key << "\n";
  return nullptr;
}

} // namespace utils
} // namespace o2

#endif
