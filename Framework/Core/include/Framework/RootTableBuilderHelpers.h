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

#include <arrow/stl.h>
#include <arrow/type_traits.h>
#include <arrow/table.h>
#include <arrow/builder.h>

#include <TTreeReader.h>
#include <TTreeReaderValue.h>

#include <vector>
#include <string>
#include <memory>
#include <tuple>

namespace o2
{
namespace framework
{
/// Trait class to go from a set of TTreeReaderValues to
/// arrow types.
template <typename TTREEREADERVALUE>
struct TreeReaderValueTraits {
  using Type = typename TTREEREADERVALUE::NonConstT_t;
  using ArrowType = typename arrow::stl::ConversionTraits<Type>::ArrowType;
  using BuilderType = typename arrow::TypeTraits<ArrowType>::BuilderType;
};

struct RootTableBuilderHelpers {
  template <typename... TTREEREADERVALUE>
  static void convertTTree(TableBuilder& builder,
                           TTreeReader& reader,
                           TTREEREADERVALUE&... values)
  {
    std::vector<std::string> branchNames = { values.GetBranchName()... };
    auto filler = builder.persist<typename TreeReaderValueTraits<TTREEREADERVALUE>::Type...>(branchNames);
    reader.Restart();
    while (reader.Next()) {
      filler(0, *values...);
    }
  }
};

} // namespace framework
} // namespace o2
#endif // FRAMEWORK_ROOTTABLEBUILDERHELPERS_H
