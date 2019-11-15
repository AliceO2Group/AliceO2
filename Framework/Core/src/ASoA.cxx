// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Framework/ASoA.h"

namespace o2::soa
{
std::shared_ptr<arrow::Table> ArrowHelpers::joinTables(std::vector<std::shared_ptr<arrow::Table>>&& tables)
{
  std::vector<std::shared_ptr<arrow::Column>> columns;
  std::vector<std::shared_ptr<arrow::Field>> fields;

  for (auto& t : tables) {
    for (size_t i = 0; i < t->num_columns(); ++i) {
      columns.push_back(t->column(i));
      fields.push_back(t->column(i)->field());
    }
  }
  return arrow::Table::Make(std::make_shared<arrow::Schema>(fields), columns);
}
} // namespace o2::soa
