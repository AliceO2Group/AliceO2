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

/// @file   RawPageTestData.h
/// @author Matthias Richter
/// @since  2021-06-21
/// @brief  Raw page test data generator

#ifndef FRAMEWORK_UTILS_RAWPAGETESTDATA_H
#define FRAMEWORK_UTILS_RAWPAGETESTDATA_H

#include "Framework/InputRecord.h"
#include "Framework/InputSpan.h"
#include "Headers/RAWDataHeader.h"
#include "Headers/DataHeader.h"
#include <vector>
#include <memory>

using DataHeader = o2::header::DataHeader;
using Stack = o2::header::Stack;
using RAWDataHeaderV6 = o2::header::RAWDataHeaderV6;

namespace o2::framework::test
{
using RAWDataHeader = RAWDataHeaderV6;
static const size_t PAGESIZE = 8192;

/// @class DataSet
/// @brief Simple helper struct to keep the InputRecord and ownership of messages
///        together with some test data.
struct DataSet {
  // not nice with the double vector but for quick unit test ok
  using Messages = std::vector<std::vector<std::unique_ptr<std::vector<char>>>>;
  DataSet(std::vector<InputRoute>&& s, Messages&& m, std::vector<int>&& v, ServiceRegistryRef registry)
    : schema{std::move(s)},
      messages{std::move(m)},
      span{[this](size_t i, size_t part) {
             auto header = static_cast<char const*>(this->messages[i].at(2 * part)->data());
             auto payload = static_cast<char const*>(this->messages[i].at(2 * part + 1)->data());
             return DataRef{nullptr, header, payload};
           },
           [this](size_t i) { return i < this->messages.size() ? messages[i].size() / 2 : 0; }, this->messages.size()},
      record{schema, span, registry},
      values{std::move(v)}
  {
  }

  std::vector<InputRoute> schema;
  Messages messages;
  InputSpan span;
  InputRecord record;
  std::vector<int> values;
};

using AmendRawDataHeader = std::function<void(RAWDataHeader&)>;
DataSet createData(std::vector<InputSpec> const& inputspecs, std::vector<DataHeader> const& dataheaders, AmendRawDataHeader amendRdh = nullptr);

} // namespace o2::framework
#endif // FRAMEWORK_UTILS_RAWPAGETESTDATA_H
