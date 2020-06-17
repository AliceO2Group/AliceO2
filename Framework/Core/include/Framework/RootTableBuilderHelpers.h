// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef o2_framework_RootTableBuilderHelpers_H_INCLUDED
#define o2_framework_RootTableBuilderHelpers_H_INCLUDED

#include "Framework/TableBuilder.h"
#include "Framework/Logger.h"

#include <arrow/stl.h>
#include <arrow/type_traits.h>
#include <arrow/table.h>
#include <arrow/builder.h>

#include <TTreeReader.h>
#include <TTreeReaderValue.h>
#include <TTreeReaderArray.h>

#include <vector>
#include <string>
#include <memory>
#include <tuple>

namespace o2
{
namespace framework
{

template <typename T>
struct TreeReaderValueTraits {
};

/// Trait class to go from a set of TTreeReaderValues to
/// arrow types.
template <typename T>
struct TreeReaderValueTraits<TTreeReaderValue<T>> {
  using Type = typename TTreeReaderValue<T>::NonConstT_t;
  using ArrowType = typename o2::framework::detail::ConversionTraits<Type>::ArrowType;
  using BuilderType = typename arrow::TypeTraits<ArrowType>::BuilderType;
};

template <typename VALUE>
struct TreeReaderValueTraits<TTreeReaderArray<VALUE>> {
  using Iterator = typename TTreeReaderArray<VALUE>::iterator;
  using Type = std::pair<Iterator, Iterator>;
  using ArrowType = arrow::ListType;
};

template <typename T>
struct ReaderHolder {
  using Reader = TTreeReaderValue<T>;
  using Type = T;
  std::unique_ptr<Reader> reader;
};

template <typename T, int N>
struct ReaderHolder<T[N]> {
  using Reader = TTreeReaderArray<T>;
  using Type = T (&)[N];
  std::unique_ptr<Reader> reader;
};

struct ValueExtractor {
  template <typename T>
  static T deref(TTreeReaderValue<T>& rv)
  {
    return *rv;
  }

  template <typename T>
  static std::pair<typename TTreeReaderArray<T>::iterator, typename TTreeReaderArray<T>::iterator> deref(TTreeReaderArray<T>& rv)
  {
    return std::make_pair(rv.begin(), rv.end());
  }

  template <typename T>
  static T deref(ReaderHolder<T>& holder)
  {
    return **holder.reader;
  }

  template <typename T, int N>
  static T* deref(ReaderHolder<T[N]>& holder)
  {
    return &((*holder.reader)[0]);
  }
};

// When reading from a ROOT file special care must happen
// because uint64_t is platform specific while ULong64_t is
// always long long unsigned int (same for the signed version).
// By using this traits we make sure that any 64 bit quantity
// read from a root file uses the ROOT datatype, not the platform one.
template <typename T>
struct Remap64Bit {
  using type = T;
};

template <>
struct Remap64Bit<int64_t> {
  using type = Long64_t;
};

template <>
struct Remap64Bit<uint64_t> {
  using type = ULong64_t;
};

template <int N>
struct Remap64Bit<int64_t[N]> {
  using type = Long64_t[N];
};

template <int N>
struct Remap64Bit<uint64_t[N]> {
  using type = ULong64_t[N];
};

template <typename T>
using Remap64Bit_t = typename Remap64Bit<T>::type;

template <typename T>
struct HolderMaker {
  static auto make(TTreeReader& reader, char const* branchName)
  {
    using Reader = TTreeReaderValue<T>;
    return std::move(ReaderHolder<T>{std::move(std::make_unique<Reader>(reader, branchName))});
  }
};

template <typename T, int N>
struct HolderMaker<T[N]> {
  static auto make(TTreeReader& reader, char const* branchName)
  {
    using Reader = TTreeReaderArray<T>;
    return std::move(ReaderHolder<T[N]>{std::move(std::make_unique<Reader>(reader, branchName))});
  }
};

template <typename C>
struct ColumnReaderTrait {
  static auto createReader(TTreeReader& reader)
  {
    return std::move(HolderMaker<Remap64Bit_t<typename C::type>>::make(reader, C::base::columnLabel()));
  }
};

struct RootTableBuilderHelpers {
  template <typename... TTREEREADERVALUE>
  static void convertTTree(TableBuilder& builder,
                           TTreeReader& reader,
                           ReaderHolder<TTREEREADERVALUE>... holders)
  {
    std::vector<std::string> branchNames = {holders.reader->GetBranchName()...};

    auto filler = builder.preallocatedPersist<typename std::decay_t<decltype(holders)>::Type...>(branchNames, reader.GetEntries(true));
    reader.Restart();
    while (reader.Next()) {
      filler(0, ValueExtractor::deref(holders)...);
    }
  }

  template <typename... C>
  static void convertASoAColumns(TableBuilder& builder, TTreeReader& reader, pack<C...>)
  {
    return convertTTree(builder, reader, ColumnReaderTrait<C>::createReader(reader)...);
  }

  template <typename T>
  static void convertASoA(TableBuilder& builder, TTreeReader& reader)
  {
    return convertASoAColumns(builder, reader, typename T::persistent_columns_t{});
  }
};

} // namespace framework
} // namespace o2
#endif // FRAMEWORK_ROOTTABLEBUILDERHELPERS_H
