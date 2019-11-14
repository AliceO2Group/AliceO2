// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef O2_FRAMEWORK_EXPRESSIONS_HELPERS_H_
#define O2_FRAMEWORK_EXPRESSIONS_HELPERS_H_
#include "Framework/Expressions.h"
#include <vector>
#include <iosfwd>

namespace o2::framework::expressions
{
struct DatumSpec {
  // datum spec either contains an index, a value of a literal or a binding label
  std::variant<std::monostate, size_t, LiteralNode::var_t, std::string> datum;
  explicit DatumSpec(size_t index) : datum{index} {}
  explicit DatumSpec(LiteralNode::var_t literal) : datum{literal} {}
  explicit DatumSpec(std::string binding) : datum{binding} {}
  DatumSpec() : datum{std::monostate{}} {}
};

bool operator==(DatumSpec const& lhs, DatumSpec const& rhs);

std::ostream& operator<<(std::ostream& os, DatumSpec const& spec);

struct ColumnOperationSpec {
  BasicOp op;
  DatumSpec left;
  DatumSpec right;
  DatumSpec result;
  ColumnOperationSpec() = default;
  explicit ColumnOperationSpec(BasicOp op_) : op{op_},
                                              left{},
                                              right{},
                                              result{} {}
};

std::vector<ColumnOperationSpec> createKernelsFromFilter(Filter const& filter);
} // namespace o2::framework::expressions

#endif // O2_FRAMEWORK_EXPRESSIONS_HELPERS_H_
