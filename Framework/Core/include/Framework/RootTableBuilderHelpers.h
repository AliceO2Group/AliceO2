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

template <typename C>
struct ColumnReaderTrait {
  using Reader = TTreeReaderValue<typename Remap64Bit<typename C::type>::type>;
  static std::unique_ptr<Reader> createReader(TTreeReader& reader)
  {
    return std::make_unique<Reader>(reader, C::base::label());
  };
};

struct RootTableBuilderHelpers {
  template <typename... TTREEREADERVALUE>
  static void convertTTree(TableBuilder& builder,
                           TTreeReader& reader,
                           TTREEREADERVALUE&... values)
  {
    std::vector<std::string> branchNames = {values.GetBranchName()...};
    auto filler = builder.preallocatedPersist<typename TreeReaderValueTraits<std::decay_t<TTREEREADERVALUE>>::Type...>(branchNames, reader.GetEntries(true));
    reader.Restart();
    while (reader.Next()) {
      filler(0, ValueExtractor::deref(values)...);
    }
  }

  template <typename... C>
  static void convertASoAColumns(TableBuilder& builder, TTreeReader& reader, pack<C...>)
  {
    return convertTTree(builder, reader, *ColumnReaderTrait<C>::createReader(reader)...);
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
