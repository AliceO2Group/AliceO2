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

} // namespace o2::framework
