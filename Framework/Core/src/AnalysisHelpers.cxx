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
#include "Framework/ExpressionHelpers.h"

namespace o2::framework
{
void initializePartitionCaches(std::set<uint32_t> const& hashes, std::shared_ptr<arrow::Schema> const& schema, expressions::Filter const& filter, gandiva::NodePtr& tree, gandiva::FilterPtr& gfilter)
{
  if (tree == nullptr) {
    expressions::Operations ops = createOperations(filter);
    if (isTableCompatible(hashes, ops)) {
      tree = createExpressionTree(ops, schema);
    } else {
      throw std::runtime_error("Partition filter does not match declared table type");
    }
  }
  if (gfilter == nullptr) {
    gfilter = framework::expressions::createFilter(schema, framework::expressions::makeCondition(tree));
  }
}
} // namespace o2::framework
