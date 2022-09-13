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

#ifndef o2_framework_RootTableBuilderHelpers_H_INCLUDED
#define o2_framework_RootTableBuilderHelpers_H_INCLUDED

#include "Framework/TableBuilder.h"
#include "Framework/Logger.h"

#include <Rtypes.h>
#include <arrow/stl.h>
#include <arrow/type_traits.h>
#include <arrow/table.h>
#include <arrow/builder.h>

#include <TTreeReader.h>
#include <TTreeReaderValue.h>
#include <TTreeReaderArray.h>
#include <TBuffer.h>
#include <TBufferFile.h>

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

static constexpr int PREBUFFER_SIZE = 32 * 1024;

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
struct ReaderHolder {
  using Reader = TTreeReaderValue<T>;
  using Type = T;

  ReaderHolder(TBranch* branch, std::unique_ptr<Reader> reader_)
    : reader{std::move(reader_)}
  {
  }

  ReaderHolder(ReaderHolder&& other)
    : reader{std::move(other.reader)},
      pos{other.pos}
  {
  }

  ReaderHolder& operator=(ReaderHolder&& other) = delete;

  std::unique_ptr<Reader> reader;
  int pos = 0;
  Remap64Bit_t<T> buffer[PREBUFFER_SIZE];
  int itemSize = sizeof(T);
};

template <typename T, int N>
struct ReaderHolder<T[N]> {
  using Reader = TTreeReaderArray<T>;
  using Type = T (&)[N];

  ReaderHolder(TBranch* branch, std::unique_ptr<Reader> reader_)
    : reader{std::move(reader_)}
  {
  }

  ReaderHolder(ReaderHolder&& other)
    : reader{std::move(other.reader)},
      pos{other.pos}
  {
  }

  ReaderHolder& operator=(ReaderHolder&& other) = delete;

  std::unique_ptr<Reader> reader;
  int pos = 0;
  Remap64Bit_t<T> buffer[PREBUFFER_SIZE * N];
  int itemSize = sizeof(T) * N;
};

struct BulkExtractor {
  template <typename T>
  static auto deref(ReaderHolder<T>& holder, size_t maxSize)
  {
    holder.buffer[holder.pos % PREBUFFER_SIZE] = **holder.reader;
    holder.pos++;
    if (holder.pos == maxSize) {
      return BulkInfo<Remap64Bit_t<T> const*>{holder.buffer, maxSize % PREBUFFER_SIZE};
    }
    // We flush only after PREBUFFER_SIZE items have been inserted
    if ((holder.pos % PREBUFFER_SIZE) != 0) {
      return BulkInfo<Remap64Bit_t<T> const*>{nullptr, 0};
    }
    return BulkInfo<Remap64Bit_t<T> const*>{holder.buffer, PREBUFFER_SIZE};
  }

  template <typename T, int N>
  static auto deref(ReaderHolder<T[N]>& holder, size_t maxSize)
  {
    memcpy(&holder.buffer[(holder.pos % PREBUFFER_SIZE) * N], &((*holder.reader)[0]), N * sizeof(T));
    holder.pos++;
    if (holder.pos == maxSize) {
      return BulkInfo<Remap64Bit_t<T> const*>{holder.buffer, maxSize % PREBUFFER_SIZE};
    }
    // We flush only after PREBUFFER_SIZE items have been inserted
    if ((holder.pos % PREBUFFER_SIZE) != 0) {
      return BulkInfo<Remap64Bit_t<T> const*>{nullptr, 0};
    }
    return BulkInfo<Remap64Bit_t<T> const*>{reinterpret_cast<T const*>(holder.buffer), PREBUFFER_SIZE};
  }
};

template <typename T>
struct HolderMaker {
  static auto make(TTreeReader& reader, char const* branchName)
  {
    using Reader = TTreeReaderValue<T>;
    return ReaderHolder<T>{reader.GetTree()->GetBranch(branchName), std::move(std::make_unique<Reader>(reader, branchName))};
  }
};

template <typename T, int N>
struct HolderMaker<T[N]> {
  static auto make(TTreeReader& reader, char const* branchName)
  {
    using Reader = TTreeReaderArray<T>;
    return ReaderHolder<T[N]>{reader.GetTree()->GetBranch(branchName), std::move(std::make_unique<Reader>(reader, branchName))};
  }
};

template <typename C>
struct ColumnReaderTrait {
  static auto createReader(TTreeReader& reader)
  {
    return HolderMaker<Remap64Bit_t<typename C::type>>::make(reader, C::base::columnLabel());
  }
};

struct RootTableBuilderHelpers {
  /// Use bulk insertion when TTreeReaderValue everywhere
  template <typename... T>
  static void convertTTree(TableBuilder& builder,
                           TTreeReader& reader,
                           ReaderHolder<T>... holders)
  {
    std::array<char const*, sizeof...(T)> branchNames = {holders.reader->GetBranchName()...};
    TTree* tree = reader.GetTree();
    size_t maxExtries = reader.GetEntries(true);
    tree->SetCacheSize(maxExtries * (holders.itemSize + ...));
    (tree->AddBranchToCache(tree->GetBranch(holders.reader->GetBranchName()), true), ...);
    tree->StopCacheLearningPhase();

    auto filler = builder.bulkPersistChunked<Remap64Bit_t<typename std::decay_t<decltype(holders)>::Type>...>(branchNames, maxExtries);
    while (reader.Next()) {
      filler(0, BulkExtractor::deref(holders, maxExtries)...);
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
