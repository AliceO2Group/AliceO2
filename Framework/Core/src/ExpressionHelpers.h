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

namespace o2::framework::expressions
{
struct ArrowDatumSpec {
  // datum spec either contains an index, a value of a literal or a binding label
  std::variant<std::monostate, size_t, LiteralNode::var_t, std::string> datum;
  explicit ArrowDatumSpec(size_t index) : datum{index} {}
  explicit ArrowDatumSpec(LiteralNode::var_t literal) : datum{literal} {}
  explicit ArrowDatumSpec(std::string binding) : datum{binding} {}
  ArrowDatumSpec() : datum{std::monostate{}} {}
};

bool operator==(ArrowDatumSpec const& lhs, ArrowDatumSpec const& rhs);

std::ostream& operator<<(std::ostream& os, ArrowDatumSpec const& spec);

struct ArrowKernelSpec {
  std::unique_ptr<arrow::compute::OpKernel> kernel = nullptr;
  ArrowDatumSpec left;
  ArrowDatumSpec right;
  ArrowDatumSpec result;
};

std::vector<ArrowKernelSpec> createKernelsFromFilter(Filter const& filter);
} // namespace o2::framework::expressions

#endif // O2_FRAMEWORK_EXPRESSIONS_HELPERS_H_
