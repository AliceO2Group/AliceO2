// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef FRAMEWORK_TABLECONSUMER_H
#define FRAMEWORK_TABLECONSUMER_H

#include <memory>

namespace arrow
{
class Table;
class Buffer;
} // namespace arrow

namespace o2
{
namespace framework
{
/// Helper class which creates a lambda suitable for building
/// an arrow table from a tuple. This can be used, for example
/// to build an arrow::Table from a TDataFrame.
class TableConsumer
{
 public:
  TableConsumer(const uint8_t* data, int64_t size);
  /// Return the table in the message as a arrow::Table instance.
  std::shared_ptr<arrow::Table> asArrowTable();

 private:
  std::shared_ptr<arrow::Buffer> mBuffer;
};

} // namespace framework
} // namespace o2

#endif // FRAMEWORK_TABLECONSUMER_H
