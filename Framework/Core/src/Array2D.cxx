// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <Framework/Array2D.h>
#include <Framework/RuntimeError.h>
#include <Framework/CompilerBuiltins.h>

template class std::unordered_map<std::string, u_int32_t>;

namespace o2::framework
{

labelmap_t LabelMap::populate(uint32_t size, std::vector<std::string> labels)
{
  labelmap_t map;
  if (labels.empty() == false) {
    if (labels.size() != size) {
      throw runtime_error("labels array size != array dimension");
    }
    for (auto i = 0u; i < size; ++i) {
      map.emplace(labels[i], (uint32_t)i);
    }
  }
  return map;
}

void LabelMap::replaceLabelsRows(uint32_t size, std::vector<std::string> const& labels)
{
  if (O2_BUILTIN_UNLIKELY(size != labels.size())) {
    throw runtime_error_f("Row labels array has different size (%d) than number of rows (%d)", labels.size(), size);
  };
  labels_rows = labels;
  rowmap.clear();
  rowmap = populate(labels.size(), labels);
}

void LabelMap::replaceLabelsCols(uint32_t size, std::vector<std::string> const& labels)
{
  if (O2_BUILTIN_UNLIKELY(size != labels.size())) {
    throw runtime_error_f("Column labels array has different size (%d) than number of columns (%d)", labels.size(), size);
  };
  labels_cols = labels;
  colmap.clear();
  colmap = populate(labels.size(), labels);
}
LabelMap::LabelMap()
  : rowmap{},
    colmap{},
    labels_rows{},
    labels_cols{}
{
}

LabelMap::LabelMap(uint32_t rows, uint32_t cols, std::vector<std::string> labels_rows_, std::vector<std::string> labels_cols_)
  : rowmap{populate(rows, labels_rows_)},
    colmap{populate(cols, labels_cols_)},
    labels_rows{labels_rows_},
    labels_cols{labels_cols_}
{
}

LabelMap::LabelMap(uint32_t size, std::vector<std::string> labels)
  : rowmap{},
    colmap{populate(size, labels)},
    labels_rows{},
    labels_cols{labels}
{
}

LabelMap::LabelMap(LabelMap const& other) = default;
LabelMap::LabelMap(LabelMap&& other) = default;
LabelMap& LabelMap::operator=(LabelMap const& other) = default;
LabelMap& LabelMap::operator=(LabelMap&& other) = default;

} // namespace o2::framework
