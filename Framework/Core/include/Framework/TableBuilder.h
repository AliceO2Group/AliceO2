// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef FRAMEWORK_TABLEBUILDER_H
#define FRAMEWORK_TABLEBUILDER_H

#include <arrow/stl.h>
#include <arrow/type_traits.h>
#include <arrow/table.h>
#include <arrow/builder.h>
#include <vector>
#include <string>
#include <memory>
#include <tuple>

namespace arrow
{
class ArrayBuilder;
class Table;
class Array;
} // namespace arrow

namespace o2
{
namespace framework
{

template <typename T>
struct BuilderMaker {
  using ArrowType = typename arrow::stl::ConversionTraits<T>::ArrowType;
  using BuilderType = typename arrow::TypeTraits<ArrowType>::BuilderType;

  static BuilderType&& make()
  {
    BuilderType builder;
    return std::move(builder);
  }
};

template <typename... ARGS>
auto make_builders()
{
  return std::make_tuple(std::make_unique<ARGS>()...);
}

template <typename T>
struct BuilderTraits {
  using ArrowType = typename arrow::stl::ConversionTraits<T>::ArrowType;
  using BuilderType = typename arrow::TypeTraits<ArrowType>::BuilderType;
};

struct TableBuilderHelpers {
  template <typename... ARGS>
  static auto makeFields(std::vector<std::string> const& names)
  {
    std::vector<std::shared_ptr<arrow::DataType>> types{ arrow::TypeTraits<typename BuilderMaker<ARGS>::ArrowType>::type_singleton()... };
    std::vector<std::shared_ptr<arrow::Field>> result;
    for (size_t i = 0; i < names.size(); ++i) {
      result.emplace_back(std::make_shared<arrow::Field>(names[i], types[i], true, nullptr));
    }
    return std::move(result);
  }

  /// Invokes the append method for each entry in the tuple
  template <std::size_t... Is, typename BUILDERS, typename VALUES>
  static bool append(BUILDERS& builders, std::index_sequence<Is...>, VALUES&& values)
  {
    return (std::get<Is>(builders)->Append(std::get<Is>(values)).ok() && ...);
  }

  /// Invokes the append method for each entry in the tuple
  template <typename BUILDERS, std::size_t... Is>
  static bool finalize(std::vector<std::shared_ptr<arrow::Array>>& arrays, BUILDERS& builders, std::index_sequence<Is...> seq)
  {
    return (std::get<Is>(builders)->Finish(&arrays[Is]).ok() && ...);
  }

  template <typename BUILDERS, std::size_t... Is>
  static bool reserveAll(BUILDERS& builders, size_t s, std::index_sequence<Is...>)
  {
    return (std::get<Is>(builders)->Reserve(s).ok() && ...);
  }
};

/// Helper class which creates a lambda suitable for building
/// an arrow table from a tuple. This can be used, for example
/// to build an arrow::Table from a TDataFrame.
class TableBuilder
{
 public:
  TableBuilder(arrow::MemoryPool* pool = arrow::default_memory_pool())
    : mBuilders{ nullptr },
      mMemoryPool{ pool }
  {
  }
  /// Creates a lambda which is suitable to persist things
  /// in an arrow::Table
  template <typename... ARGS>
  auto persist(std::vector<std::string> const& columnNames)
  {
    constexpr int nColumns = sizeof...(ARGS);
    if (nColumns != columnNames.size()) {
      throw std::runtime_error("Mismatching number of column types and names");
    }
    if (mBuilders != nullptr) {
      throw std::runtime_error("TableBuilder::persist can only be invoked once per instance");
    }
    mArrays.resize(nColumns);

    using BuildersTuple = typename std::tuple<std::unique_ptr<typename BuilderTraits<ARGS>::BuilderType>...>;
    mSchema = std::make_shared<arrow::Schema>(TableBuilderHelpers::makeFields<ARGS...>(columnNames));

    BuildersTuple* builders = new BuildersTuple(std::make_unique<typename BuilderTraits<ARGS>::BuilderType>(mMemoryPool)...);
    auto seq = std::make_index_sequence<sizeof...(ARGS)>{};
    TableBuilderHelpers::reserveAll(*builders, 1000, seq);
    mBuilders = builders; // We store the builders
    /// Callback used to finalize the table.
    mFinalizer = [ schema = mSchema, &arrays = mArrays, builders = builders]()->std::shared_ptr<arrow::Table>
    {
      auto status = TableBuilderHelpers::finalize(arrays, *builders, std::make_index_sequence<sizeof...(ARGS)>{});
      if (status == false) {
        throw std::runtime_error("Unable to finalize");
      }
      return arrow::Table::Make(schema, arrays);
    };
    // Callback used to fill the builders
    return [builders = builders](unsigned int slot, ARGS... args) -> void {
      auto status = TableBuilderHelpers::append(*builders, std::index_sequence_for<ARGS...>{}, std::forward_as_tuple(args...));
      if (status == false) {
        throw std::runtime_error("Unable to append");
      }
    };
  }

  /// Actually creates the arrow::Table from the builders
  std::shared_ptr<arrow::Table> finalize();

 private:
  std::function<void(void)> mFinalizer;
  void* mBuilders;
  arrow::MemoryPool* mMemoryPool;
  std::shared_ptr<arrow::Schema> mSchema;
  std::vector<std::shared_ptr<arrow::Array>> mArrays;
};

} // namespace framework
} // namespace o2
#endif // FRAMEWORK_TABLEBUILDER_H
