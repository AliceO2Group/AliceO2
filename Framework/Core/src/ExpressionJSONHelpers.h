// Copyright 2019-2025 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_EXPRESSIONJSONHELPERS_H
#define FRAMEWORK_EXPRESSIONJSONHELPERS_H

#include "Framework/Expressions.h"

namespace o2::framework
{
struct ExpressionJSONHelpers {
  static std::vector<expressions::Projector> read(std::istream& s);
  static void write(std::ostream& o, std::vector<expressions::Projector>& projectors);
};

struct ArrowJSONHelpers {
  static std::shared_ptr<arrow::Schema> read(std::istream& s);
  static void write(std::ostream& o, std::shared_ptr<arrow::Schema>& schema);
};
} // namespace o2::framework

#endif // FRAMEWORK_EXPRESSIONJSONHELPERS_H
