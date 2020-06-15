// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef FRAMEWORK_ANALYSIS_TASK_H_
#define FRAMEWORK_ANALYSIS_TASK_H_

#include "Framework/Kernels.h"
#include "Framework/AlgorithmSpec.h"
#include "Framework/AnalysisDataModel.h"
#include "Framework/CallbackService.h"
#include "Framework/ControlService.h"
#include "Framework/ConfigParamSpec.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Expressions.h"
#include "Framework/EndOfStreamContext.h"
#include "Framework/Logger.h"
#include "Framework/HistogramRegistry.h"
#include "Framework/StructToTuple.h"
#include "Framework/FunctionalHelpers.h"
#include "Framework/Traits.h"
#include "Framework/VariantHelpers.h"
#include "Framework/OutputObjHeader.h"
#include "Framework/RootConfigParamHelpers.h"

#include <arrow/compute/context.h>
#include <arrow/compute/kernel.h>
#include <arrow/table.h>
#include <gandiva/node.h>
#include <type_traits>
#include <utility>
#include <memory>
#include <sstream>
#include <iomanip>
namespace o2::framework
{

/// A more familiar task API for the DPL analysis framework.
/// This allows you to define your own tasks as subclasses
/// of o2::framework::AnalysisTask and to pass them in the specification
/// using:
///
/// adaptAnalysisTask<YourDerivedTask>(constructor args, ...);
///
// FIXME: for the moment this needs to stay outside AnalysisTask
//        because we cannot inherit from it due to a C++17 bug
//        in GCC 7.3. We need to move to 7.4+
template <typename T>
struct WritingCursor {
  static_assert(always_static_assert_v<T>, "Type must be a o2::soa::Table");
};

template <typename T>
struct Produces {
  static_assert(always_static_assert_v<T>, "Type must be a o2::soa::Table");
};

/// Helper class actually implementing the cursor which can write to
/// a table. The provided template arguments are if type Column and
/// therefore refer only to the persisted columns.
template <typename... PC>
struct WritingCursor<soa::Table<PC...>> {
  using persistent_table_t = soa::Table<PC...>;
  using cursor_t = decltype(std::declval<TableBuilder>().cursor<persistent_table_t>());

  template <typename... T>
  void operator()(T... args)
  {
    static_assert(sizeof...(PC) == sizeof...(T), "Argument number mismatch");
    ++mCount;
    cursor(0, extract(args)...);
  }

  /// Last index inserted in the table
  int64_t lastIndex()
  {
    return mCount;
  }

  bool resetCursor(TableBuilder& builder)
  {
    mBuilder = &builder;
    cursor = std::move(FFL(builder.cursor<persistent_table_t>()));
    mCount = -1;
    return true;
  }

  /// reserve @a size rows when filling, so that we do not
  /// spend time reallocating the buffers.
  void reserve(int64_t size)
  {
    mBuilder->reserve(typename persistent_table_t::columns{}, size);
  }

  decltype(FFL(std::declval<cursor_t>())) cursor;

 private:
  template <typename T>
  static decltype(auto) extract(T const& arg)
  {
    if constexpr (soa::is_soa_iterator_t<T>::value) {
      return arg.globalIndex();
    } else {
      static_assert(!framework::has_type_v<T, framework::pack<PC...>>, "Argument type mismatch");
      return arg;
    }
  }

  /// The table builder which actually performs the
  /// construction of the table. We keep it around to be
  /// able to do all-columns methods like reserve.
  TableBuilder* mBuilder = nullptr;
  int64_t mCount = -1;
};

/// This helper class allow you to declare things which will be crated by a
/// give analysis task. Notice how the actual cursor is implemented by the
/// means of the WritingCursor helper class, from which produces actually
/// derives.
template <typename... C>
struct Produces<soa::Table<C...>> : WritingCursor<typename soa::FilterPersistentColumns<soa::Table<C...>>::persistent_table_t> {
  using table_t = soa::Table<C...>;
  using metadata = typename aod::MetadataTrait<table_t>::metadata;

  // @return the associated OutputSpec
  OutputSpec const spec()
  {
    return OutputSpec{OutputLabel{metadata::tableLabel()}, metadata::origin(), metadata::description()};
  }

  OutputRef ref()
  {
    return OutputRef{metadata::tableLabel(), 0};
  }
};

/// This helper class allows you to declare things which will be created by a
/// given analysis task. Currently wrapped objects are limited to be TNamed
/// descendants. Objects will be written to a ROOT file at the end of the
/// workflow, in directories, corresponding to the task they were declared in.
/// Each object has associated handling policy, which is used by the framework
/// to determine the target file, e.g. analysis result, QA or control histogram,
/// etc.
template <typename T>
struct OutputObj {
  using obj_t = T;

  OutputObj(T&& t, OutputObjHandlingPolicy policy_ = OutputObjHandlingPolicy::AnalysisObject)
    : object(std::make_shared<T>(t)),
      label(t.GetName()),
      policy{policy_},
      mTaskHash{0}
  {
  }

  OutputObj(std::string const& label_, OutputObjHandlingPolicy policy_ = OutputObjHandlingPolicy::AnalysisObject)
    : object(nullptr),
      label(label_),
      policy{policy_},
      mTaskHash{0}
  {
  }

  void setObject(T const& t)
  {
    object = std::make_shared<T>(t);
    object->SetName(label.c_str());
  }

  void setObject(T&& t)
  {
    object = std::make_shared<T>(t);
    object->SetName(label.c_str());
  }

  void setObject(T* t)
  {
    object.reset(t);
    object->SetName(label.c_str());
  }

  void setHash(uint32_t hash)
  {
    mTaskHash = hash;
  }

  /// @return the associated OutputSpec
  OutputSpec const spec()
  {
    header::DataDescription desc{};
    auto lhash = compile_time_hash(label.c_str());
    std::memset(desc.str, '_', 16);
    std::stringstream s;
    s << std::hex << lhash;
    s << std::hex << mTaskHash;
    s << std::hex << reinterpret_cast<uint64_t>(this);
    std::memcpy(desc.str, s.str().c_str(), 12);
    return OutputSpec{OutputLabel{label}, "ATSK", desc, 0};
  }

  T* operator->()
  {
    return object.get();
  }

  T& operator*()
  {
    return *object.get();
  }

  OutputRef ref()
  {
    return OutputRef{std::string{label}, 0,
                     o2::header::Stack{OutputObjHeader{policy, mTaskHash}}};
  }

  std::shared_ptr<T> object;
  std::string label;
  OutputObjHandlingPolicy policy;
  uint32_t mTaskHash;
};

/// This helper allows you to create a configurable option associated to a task.
/// Internally it will be bound to a ConfigParamSpec.
template <typename T>
struct Configurable {
  Configurable(std::string const& name, T defaultValue, std::string const& help)
    : name(name), value(defaultValue), help(help)
  {
  }
  using type = T;
  std::string name;
  T value;
  std::string help;
  operator T()
  {
    return value;
  }
  T const* operator->() const
  {
    return &value;
  }
};

struct AnalysisTask {
};

// Helper struct which builds a DataProcessorSpec from
// the contents of an AnalysisTask...

struct AnalysisDataProcessorBuilder {
  template <typename Arg>
  static void doAppendInputWithMetadata(std::vector<InputSpec>& inputs)
  {
    using metadata = typename aod::MetadataTrait<std::decay_t<Arg>>::metadata;
    static_assert(std::is_same_v<metadata, void> == false,
                  "Could not find metadata. Did you register your type?");
    inputs.push_back({metadata::tableLabel(), metadata::origin(), metadata::description()});
  }

  template <typename... Args>
  static void doAppendInputWithMetadata(framework::pack<Args...>, std::vector<InputSpec>& inputs)
  {
    (doAppendInputWithMetadata<Args>(inputs), ...);
  }

  template <typename T, size_t At>
  static void appendSomethingWithMetadata(std::vector<InputSpec>& inputs, std::vector<ExpressionInfo>& eInfos)
  {
    using dT = std::decay_t<T>;
    if constexpr (framework::is_specialization<dT, soa::Filtered>::value) {
      eInfos.push_back({At, o2::soa::createSchemaFromColumns(typename dT::table_t::persistent_columns_t{}), nullptr});
    } else if constexpr (soa::is_soa_iterator_t<dT>::value) {
      if (std::is_same_v<typename dT::policy_t, soa::FilteredIndexPolicy>) {
        eInfos.push_back({At, o2::soa::createSchemaFromColumns(typename dT::table_t::persistent_columns_t{}), nullptr});
      }
    }
    doAppendInputWithMetadata(soa::make_originals_from_type<dT>(), inputs);
  }

  template <typename R, typename C, typename... Args>
  static void inputsFromArgs(R (C::*)(Args...), std::vector<InputSpec>& inputs, std::vector<ExpressionInfo>& eInfos)
  {
    (appendSomethingWithMetadata<Args, has_type_at<Args>(pack<Args...>{})>(inputs, eInfos), ...);
  }

  template <typename R, typename C, typename Grouping, typename... Args>
  static auto signatures(InputRecord&, R (C::*)(Grouping, Args...))
  {
    return std::declval<std::tuple<Grouping, Args...>>();
  }

  template <typename R, typename C, typename Grouping, typename... Args>
  static auto bindGroupingTable(InputRecord& record, R (C::*)(Grouping, Args...), std::vector<ExpressionInfo> const& infos)
  {
    return extractSomethingFromRecord<Grouping, 0>(record, infos);
  }

  template <typename R, typename C>
  static auto bindGroupingTable(InputRecord&, R (C::*)(), std::vector<ExpressionInfo> const&)
  {
    static_assert(always_static_assert_v<C>, "Your task process method needs at least one argument");
    return o2::soa::Table<>{nullptr};
  }

  template <typename T>
  static auto extractTableFromRecord(InputRecord& record)
  {
    if constexpr (soa::is_type_with_metadata_v<aod::MetadataTrait<T>>) {
      return record.get<TableConsumer>(aod::MetadataTrait<T>::metadata::tableLabel())->asArrowTable();
    } else if constexpr (soa::is_type_with_originals_v<T>) {
      return extractFromRecord<T>(record, typename T::originals{});
    }
    O2_BUILTIN_UNREACHABLE();
  }

  template <typename T, typename... Os>
  static auto extractFromRecord(InputRecord& record, pack<Os...> const&)
  {
    if constexpr (soa::is_soa_iterator_t<T>::value) {
      return typename T::parent_t{{extractTableFromRecord<Os>(record)...}};
    } else {
      return T{{extractTableFromRecord<Os>(record)...}};
    }
  }

  template <typename T, typename... Os>
  static auto extractFilteredFromRecord(InputRecord& record, ExpressionInfo const& info, pack<Os...> const&)
  {
    if constexpr (soa::is_soa_iterator_t<T>::value) {
      return typename T::parent_t(std::vector<std::shared_ptr<arrow::Table>>{extractTableFromRecord<Os>(record)...}, info.tree);
    } else {
      return T(std::vector<std::shared_ptr<arrow::Table>>{extractTableFromRecord<Os>(record)...}, info.tree);
    }
  }

  template <typename T, size_t At>
  static auto extractSomethingFromRecord(InputRecord& record, std::vector<ExpressionInfo> const infos)
  {
    using decayed = std::decay_t<T>;

    if constexpr (soa::is_soa_filtered_t<decayed>::value) {
      for (auto& info : infos) {
        if (info.index == At)
          return extractFilteredFromRecord<decayed>(record, info, soa::make_originals_from_type<decayed>());
      }
    } else if constexpr (soa::is_soa_iterator_t<decayed>::value) {
      if constexpr (std::is_same_v<typename decayed::policy_t, soa::FilteredIndexPolicy>) {
        for (auto& info : infos) {
          if (info.index == At)
            return extractFilteredFromRecord<decayed>(record, info, soa::make_originals_from_type<decayed>());
        }
      } else {
        return extractFromRecord<decayed>(record, soa::make_originals_from_type<decayed>());
      }
    } else {
      return extractFromRecord<decayed>(record, soa::make_originals_from_type<decayed>());
    }
    O2_BUILTIN_UNREACHABLE();
  }

  template <typename R, typename C, typename Grouping, typename... Args>
  static auto bindAssociatedTables(InputRecord& record, R (C::*)(Grouping, Args...), std::vector<ExpressionInfo> const infos)
  {
    return std::make_tuple(extractSomethingFromRecord<Args, has_type_at<Args>(pack<Args...>{}) + 1u>(record, infos)...);
  }

  template <typename R, typename C>
  static auto bindAssociatedTables(InputRecord&, R (C::*)(), std::vector<ExpressionInfo> const)
  {
    static_assert(always_static_assert_v<C>, "Your task process method needs at least one argument");
    return std::tuple<>{};
  }

  template <typename T, typename C>
  using is_external_index_to_t = std::is_same<typename C::binding_t, T>;

  template <typename G, typename... A>
  struct GroupSlicer {
    using grouping_t = std::decay_t<G>;
    GroupSlicer(G& gt, std::tuple<A...>& at)
      : max{gt.size()},
        mBegin{GroupSlicerIterator(gt, at)}
    {
    }

    struct GroupSlicerSentinel {
      int64_t position;
    };

    struct GroupSlicerIterator {
      using associated_pack_t = framework::pack<A...>;

      GroupSlicerIterator() = default;
      GroupSlicerIterator(GroupSlicerIterator const&) = default;
      GroupSlicerIterator(GroupSlicerIterator&&) = default;
      GroupSlicerIterator& operator=(GroupSlicerIterator const&) = default;
      GroupSlicerIterator& operator=(GroupSlicerIterator&&) = default;

      auto getLabelFromType()
      {
        if constexpr (soa::is_type_with_originals_v<std::decay_t<G>>) {
          using T = typename framework::pack_element_t<0, typename std::decay_t<G>::originals>;
          using groupingMetadata = typename aod::MetadataTrait<T>::metadata;
          return std::string("f") + groupingMetadata::tableLabel() + "ID";
        } else {
          using groupingMetadata = typename aod::MetadataTrait<std::decay_t<G>>::metadata;
          return std::string("f") + groupingMetadata::tableLabel() + "ID";
        }
      }

      GroupSlicerIterator(G& gt, std::tuple<A...>& at)
        : mAt{&at},
          mGroupingElement{gt.begin()},
          position{0}
      {
        auto indexColumnName = getLabelFromType();
        arrow::compute::FunctionContext ctx;
        /// prepare slices and offsets for all associated tables that have index
        /// to grouping table
        ///
        auto splitter = [&](auto&& x) {
          using xt = std::decay_t<decltype(x)>;
          constexpr auto index = framework::has_type_at<std::decay_t<decltype(x)>>(associated_pack_t{});
          if (hasIndexTo<std::decay_t<G>>(typename xt::persistent_columns_t{})) {
            auto result = o2::framework::sliceByColumn(&ctx, indexColumnName,
                                                       static_cast<int32_t>(gt.size()),
                                                       x.asArrowTable(),
                                                       &groups[index],
                                                       &offsets[index]);
            if (result.ok() == false) {
              throw std::runtime_error("Cannot split collection");
            }
            if (groups[index].size() != gt.size()) {
              throw std::runtime_error("Splitting collection resulted in different group number than there is rows in the grouping table.");
            };
          }
        };

        std::apply(
          [&](auto&&... x) -> void {
            (splitter(x), ...);
          },
          at);
        /// extract selections from filtered associated tables
        auto extractor = [&](auto&& x) {
          using xt = std::decay_t<decltype(x)>;
          if constexpr (soa::is_soa_filtered_t<xt>::value) {
            constexpr auto index = framework::has_type_at<std::decay_t<decltype(x)>>(associated_pack_t{});
            selections[index] = &x.getSelectedRows();
            starts[index] = selections[index]->begin();
            offsets[index].push_back(std::get<xt>(at).tableSize());
          }
        };
        std::apply(
          [&](auto&&... x) -> void {
            (extractor(x), ...);
          },
          at);
      }

      template <typename B, typename... C>
      constexpr bool hasIndexTo(framework::pack<C...>&&)
      {
        return (isIndexTo<B, C>() || ...);
      }

      template <typename B, typename C>
      constexpr bool isIndexTo()
      {
        if constexpr (soa::is_type_with_binding_v<C>) {
          if constexpr (soa::is_type_with_originals_v<std::decay_t<B>>) {
            using TT = typename framework::pack_element_t<0, typename std::decay_t<B>::originals>;
            return std::is_same_v<typename C::binding_t, TT>;
          } else {
            using TT = std::decay_t<B>;
            return std::is_same_v<typename C::binding_t, TT>;
          }
        }
        return false;
      }

      GroupSlicerIterator operator++()
      {
        ++position;
        ++mGroupingElement;
        return *this;
      }

      bool operator==(GroupSlicerSentinel const& other)
      {
        return O2_BUILTIN_UNLIKELY(position == other.position);
      }

      bool operator!=(GroupSlicerSentinel const& other)
      {
        return O2_BUILTIN_LIKELY(position != other.position);
      }

      auto& groupingElement()
      {
        return mGroupingElement;
      }

      GroupSlicerIterator& operator*()
      {
        return *this;
      }

      auto associatedTables()
      {
        return std::make_tuple(prepareArgument<A>()...);
      }

      template <typename A1>
      auto prepareArgument()
      {
        constexpr auto index = framework::has_type_at<A1>(associated_pack_t{});
        if (hasIndexTo<G>(typename std::decay_t<A1>::persistent_columns_t{})) {
          if constexpr (soa::is_soa_filtered_t<std::decay_t<A1>>::value) {
            auto groupedElementsTable = arrow::util::get<std::shared_ptr<arrow::Table>>(((groups[index])[position]).value);

            // for each grouping element we need to slice the selection vector
            auto start_iterator = std::lower_bound(starts[index], selections[index]->end(), (offsets[index])[position]);
            auto stop_iterator = std::lower_bound(start_iterator, selections[index]->end(), (offsets[index])[position + 1]);
            starts[index] = stop_iterator;
            soa::SelectionVector slicedSelection{start_iterator, stop_iterator};
            std::transform(slicedSelection.begin(), slicedSelection.end(), slicedSelection.begin(),
                           [&](int64_t idx) {
                             return idx - static_cast<int64_t>((offsets[index])[position]);
                           });

            std::decay_t<A1> typedTable{{groupedElementsTable}, std::move(slicedSelection), (offsets[index])[position]};
            return typedTable;
          } else {
            auto groupedElementsTable = arrow::util::get<std::shared_ptr<arrow::Table>>(((groups[index])[position]).value);
            std::decay_t<A1> typedTable{{groupedElementsTable}, (offsets[index])[position]};
            return typedTable;
          }
        } else {
          return std::get<A1>(*mAt);
        }
        O2_BUILTIN_UNREACHABLE();
      }

      std::tuple<A...>* mAt;
      typename grouping_t::iterator mGroupingElement;
      uint64_t position = 0;

      std::array<std::vector<arrow::compute::Datum>, sizeof...(A)> groups;
      std::array<std::vector<uint64_t>, sizeof...(A)> offsets;
      std::array<soa::SelectionVector const*, sizeof...(A)> selections;
      std::array<soa::SelectionVector::const_iterator, sizeof...(A)> starts;
    };

    GroupSlicerIterator& begin()
    {
      return mBegin;
    }

    GroupSlicerSentinel end()
    {
      return GroupSlicerSentinel{max};
    }
    int64_t max;
    GroupSlicerIterator mBegin;
  };

  template <typename Task, typename R, typename C, typename Grouping, typename... Associated>
  static void invokeProcess(Task& task, InputRecord& inputs, R (C::*)(Grouping, Associated...), std::vector<ExpressionInfo> const& infos)
  {
    using G = std::decay_t<Grouping>;
    auto groupingTable = AnalysisDataProcessorBuilder::bindGroupingTable(inputs, &C::process, infos);
    if constexpr (sizeof...(Associated) == 0) {
      // single argument to process
      if constexpr (soa::is_soa_iterator_t<G>::value) {
        for (auto& element : groupingTable) {
          task.process(*element);
        }
      } else {
        static_assert(soa::is_soa_table_like_t<G>::value,
                      "Single argument of process() should be a table-like or an iterator");
        task.process(groupingTable);
      }
    } else {
      // multiple arguments to process
      static_assert(((soa::is_soa_iterator_t<std::decay_t<Associated>>::value == false) && ...),
                    "Associated arguments of process() should not be iterators");
      auto associatedTables = AnalysisDataProcessorBuilder::bindAssociatedTables(inputs, &C::process, infos);
      auto binder = [&](auto&& x) {
        x.bindExternalIndices(&groupingTable, &std::get<std::decay_t<Associated>>(associatedTables)...);
      };
      groupingTable.bindExternalIndices(&std::get<std::decay_t<Associated>>(associatedTables)...);

      if constexpr (soa::is_soa_iterator_t<std::decay_t<G>>::value) {
        // grouping case
        auto slicer = GroupSlicer(groupingTable, associatedTables);
        for (auto& slice : slicer) {
          auto associatedSlices = slice.associatedTables();
          std::apply(
            [&](auto&&... x) {
              (binder(x), ...);
            },
            associatedSlices);

          invokeProcessWithArgs(task, slice.groupingElement(), associatedSlices);
        }
      } else {
        // non-grouping case
        std::apply(
          [&](auto&&... x) {
            (binder(x), ...);
          },
          associatedTables);

        invokeProcessWithArgs(task, groupingTable, associatedTables);
      }
    }
  }

  template <typename T, typename G, typename... A>
  static void invokeProcessWithArgs(T& task, G g, std::tuple<A...>& at)
  {
    task.process(g, std::get<A>(at)...);
  }
};

template <typename T>
struct FilterManager {
  template <typename ANY>
  static bool createExpressionTrees(ANY&, std::vector<ExpressionInfo>&)
  {
    return false;
  }
};

template <>
struct FilterManager<expressions::Filter> {
  static bool createExpressionTrees(expressions::Filter const& filter, std::vector<ExpressionInfo>& expressionInfos)
  {

    updateExpressionInfos(filter, expressionInfos);
    return true;
  }
};

template <typename T>
struct OutputManager {
  template <typename ANY>
  static bool appendOutput(std::vector<OutputSpec>&, ANY&, uint32_t)
  {
    return false;
  }

  template <typename ANY>
  static bool prepare(ProcessingContext&, ANY&)
  {
    return false;
  }

  template <typename ANY>
  static bool postRun(EndOfStreamContext&, ANY&)
  {
    return true;
  }

  template <typename ANY>
  static bool finalize(ProcessingContext&, ANY&)
  {
    return true;
  }
};

template <typename TABLE>
struct OutputManager<Produces<TABLE>> {
  static bool appendOutput(std::vector<OutputSpec>& outputs, Produces<TABLE>& what, uint32_t)
  {
    outputs.emplace_back(what.spec());
    return true;
  }
  static bool prepare(ProcessingContext& context, Produces<TABLE>& what)
  {
    what.resetCursor(context.outputs().make<TableBuilder>(what.ref()));
    return true;
  }
  static bool finalize(ProcessingContext&, Produces<TABLE>&)
  {
    return true;
  }
  static bool postRun(EndOfStreamContext&, Produces<TABLE>&)
  {
    return true;
  }
};

template <>
struct OutputManager<HistogramRegistry> {
  static bool appendOutput(std::vector<OutputSpec>& outputs, HistogramRegistry& what, uint32_t)
  {
    outputs.emplace_back(what.spec());
    return true;
  }
  static bool prepare(ProcessingContext&, HistogramRegistry&)
  {
    return true;
  }

  static bool finalize(ProcessingContext&, HistogramRegistry&)
  {
    return true;
  }

  static bool postRun(EndOfStreamContext&, HistogramRegistry&)
  {
    return true;
  }
};

template <typename T>
struct OutputManager<OutputObj<T>> {
  static bool appendOutput(std::vector<OutputSpec>& outputs, OutputObj<T>& what, uint32_t hash)
  {
    what.setHash(hash);
    outputs.emplace_back(what.spec());
    return true;
  }
  static bool prepare(ProcessingContext&, OutputObj<T>&)
  {
    return true;
  }

  static bool finalize(ProcessingContext&, OutputObj<T>&)
  {
    return true;
  }

  static bool postRun(EndOfStreamContext& context, OutputObj<T>& what)
  {
    context.outputs().snapshot(what.ref(), *what);
    return true;
  }
};

template <typename T>
struct OptionManager {
  template <typename ANY>
  static bool appendOption(std::vector<ConfigParamSpec>&, ANY&)
  {
    return false;
  }

  template <typename ANY>
  static bool prepare(InitContext&, ANY&)
  {
    return false;
  }
};

template <typename T>
struct OptionManager<Configurable<T>> {
  static bool appendOption(std::vector<ConfigParamSpec>& options, Configurable<T>& what)
  {
    if constexpr (variant_trait_v<typename std::decay<T>::type> != VariantType::Unknown) {
      options.emplace_back(ConfigParamSpec{what.name, variant_trait_v<typename std::decay<T>::type>, what.value, {what.help}});
    } else {
      auto specs = RootConfigParamHelpers::asConfigParamSpecs<T>(what.name, what.value);
      options.insert(options.end(), specs.begin(), specs.end());
    }
    return true;
  }

  static bool prepare(InitContext& context, Configurable<T>& what)
  {
    if constexpr (variant_trait_v<typename std::decay<T>::type> != VariantType::Unknown) {
      what.value = context.options().get<T>(what.name.c_str());
    } else {
      auto pt = context.options().get<boost::property_tree::ptree>(what.name.c_str());
      what.value = RootConfigParamHelpers::as<T>(pt);
    }
    return true;
  }
};

// SFINAE test
template <typename T>
class has_process
{
  typedef char one;
  struct two {
    char x[2];
  };

  template <typename C>
  static one test(decltype(&C::process));
  template <typename C>
  static two test(...);

 public:
  enum { value = sizeof(test<T>(nullptr)) == sizeof(char) };
};

template <typename T>
class has_run
{
  typedef char one;
  struct two {
    char x[2];
  };

  template <typename C>
  static one test(decltype(&C::run));
  template <typename C>
  static two test(...);

 public:
  enum { value = sizeof(test<T>(nullptr)) == sizeof(char) };
};

template <typename T>
class has_init
{
  typedef char one;
  struct two {
    char x[2];
  };

  template <typename C>
  static one test(decltype(&C::init));
  template <typename C>
  static two test(...);

 public:
  enum { value = sizeof(test<T>(nullptr)) == sizeof(char) };
};

/// Adaptor to make an AlgorithmSpec from a o2::framework::Task
///
template <typename T, typename... Args>
DataProcessorSpec adaptAnalysisTask(char const* name, Args&&... args)
{
  TH1::AddDirectory(false);
  auto task = std::make_shared<T>(std::forward<Args>(args)...);
  auto hash = compile_time_hash(name);

  std::vector<OutputSpec> outputs;
  std::vector<ConfigParamSpec> options;

  auto tupledTask = o2::framework::to_tuple_refs(*task.get());
  static_assert(has_process<T>::value || has_run<T>::value || has_init<T>::value,
                "At least one of process(...), T::run(...), init(...) must be defined");

  std::vector<InputSpec> inputs;
  std::vector<ExpressionInfo> expressionInfos;

  if constexpr (has_process<T>::value) {
    // this pushes (I,schemaPtr,nullptr) into expressionInfos for arguments that are Filtered/filtered_iterators
    AnalysisDataProcessorBuilder::inputsFromArgs(&T::process, inputs, expressionInfos);
    // here the FilterManager will prepare the gandiva trees matched to schemas and put the pointers into expressionInfos
    std::apply([&expressionInfos](auto&... x) {
      return (FilterManager<std::decay_t<decltype(x)>>::createExpressionTrees(x, expressionInfos), ...);
    },
               tupledTask);
  }

  std::apply([&outputs, &hash](auto&... x) { return (OutputManager<std::decay_t<decltype(x)>>::appendOutput(outputs, x, hash), ...); }, tupledTask);
  std::apply([&options, &hash](auto&... x) { return (OptionManager<std::decay_t<decltype(x)>>::appendOption(options, x), ...); }, tupledTask);

  auto algo = AlgorithmSpec::InitCallback{[task, expressionInfos](InitContext& ic) {
    auto tupledTask = o2::framework::to_tuple_refs(*task.get());
    std::apply([&ic](auto&&... x) { return (OptionManager<std::decay_t<decltype(x)>>::prepare(ic, x), ...); }, tupledTask);

    auto& callbacks = ic.services().get<CallbackService>();
    auto endofdatacb = [task](EndOfStreamContext& eosContext) {
      auto tupledTask = o2::framework::to_tuple_refs(*task.get());
      std::apply([&eosContext](auto&&... x) { return (OutputManager<std::decay_t<decltype(x)>>::postRun(eosContext, x), ...); }, tupledTask);
      eosContext.services().get<ControlService>().readyToQuit(QuitRequest::Me);
    };
    callbacks.set(CallbackService::Id::EndOfStream, endofdatacb);

    if constexpr (has_init<T>::value) {
      task->init(ic);
    }
    return [task, expressionInfos](ProcessingContext& pc) {
      auto tupledTask = o2::framework::to_tuple_refs(*task.get());
      std::apply([&pc](auto&&... x) { return (OutputManager<std::decay_t<decltype(x)>>::prepare(pc, x), ...); }, tupledTask);
      if constexpr (has_run<T>::value) {
        task->run(pc);
      }
      if constexpr (has_process<T>::value) {
        AnalysisDataProcessorBuilder::invokeProcess(*(task.get()), pc.inputs(), &T::process, expressionInfos);
      }
      std::apply([&pc](auto&&... x) { return (OutputManager<std::decay_t<decltype(x)>>::finalize(pc, x), ...); }, tupledTask);
    };
  }};

  DataProcessorSpec spec{
    name,
    // FIXME: For the moment we hardcode this. We could build
    // this list from the list of methods actually implemented in the
    // task itself.
    inputs,
    outputs,
    algo,
    options};
  return spec;
}

} // namespace o2::framework
#endif // FRAMEWORK_ANALYSISTASK_H_
