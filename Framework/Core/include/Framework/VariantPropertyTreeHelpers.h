// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_VARIANTPTREEHELPERS_H
#define FRAMEWORK_VARIANTPTREEHELPERS_H

#include "Framework/Variant.h"
#include <boost/property_tree/ptree.hpp>

namespace o2::framework
{
namespace
{
template <typename T>
auto basicVectorToBranch(T* values, size_t size)
{
  boost::property_tree::ptree branch;
  for (auto i = 0u; i < size; ++i) {
    boost::property_tree::ptree leaf;
    leaf.put("", values[i]);
    branch.push_back(std::make_pair("", leaf));
  }
  return branch;
}

template <typename T>
auto basicVectorToBranch(std::vector<T>&& values)
{
  return basicVectorToBranch(values.data(), values.size());
}

template <typename T>
auto vectorToBranch(T* values, size_t size)
{
  boost::property_tree::ptree branch;
  branch.put_child("values", basicVectorToBranch(values, size));
  return branch;
}
} // namespace

template <typename T>
auto vectorToBranch(std::vector<T>&& values)
{
  return vectorToBranch(values.data(), values.size());
}

namespace
{
template <typename T>
auto basicArray2DToBranch(Array2D<T>&& array)
{
  boost::property_tree::ptree subtree;
  for (auto i = 0u; i < array.rows; ++i) {
    boost::property_tree::ptree branch;
    for (auto j = 0u; j < array.cols; ++j) {
      boost::property_tree::ptree leaf;
      leaf.put("", array(i, j));
      branch.push_back(std::make_pair("", leaf));
    }
    subtree.push_back(std::make_pair("", branch));
  }
  return subtree;
}
} // namespace

template <typename T>
auto array2DToBranch(Array2D<T>&& array)
{
  boost::property_tree::ptree subtree;
  subtree.put_child("values", basicArray2DToBranch(std::forward<Array2D<T>>(array)));
  return subtree;
}

namespace
{
template <typename T>
auto basicVectorFromBranch(boost::property_tree::ptree const& branch)
{
  std::vector<T> result(branch.size());
  auto count = 0U;
  for (auto const& entry : branch) {
    result[count++] = entry.second.get_value<T>();
  }
  return result;
}
} // namespace

template <typename T>
auto vectorFromBranch(boost::property_tree::ptree const& branch)
{
  return basicVectorFromBranch<T>(branch.get_child("values"));
}

namespace
{
template <typename T>
auto basicArray2DFromBranch(boost::property_tree::ptree const& branch)
{
  std::vector<T> cache;
  uint32_t nrows = branch.size();
  uint32_t ncols = 0;
  bool first = true;
  auto irow = 0u;
  for (auto const& row : branch) {
    if (first) {
      ncols = row.second.size();
      first = false;
    }
    auto icol = 0u;
    for (auto const& entry : row.second) {
      cache.push_back(entry.second.get_value<T>());
      ++icol;
    }
    ++irow;
  }
  return Array2D<T>{cache, nrows, ncols};
}
} // namespace

template <typename T>
auto array2DFromBranch(boost::property_tree::ptree const& ptree)
{
  return basicArray2DFromBranch<T>(ptree.get_child("values"));
}

std::pair<std::vector<std::string>, std::vector<std::string>> extractLabels(boost::property_tree::ptree const& tree);

template <typename T>
auto labeledArrayFromBranch(boost::property_tree::ptree const& tree)
{
  auto [labels_rows, labels_cols] = extractLabels(tree);
  auto values = basicArray2DFromBranch<T>(tree.get_child("values"));

  return LabeledArray<T>{values, labels_rows, labels_cols};
}

template <typename T>
auto labeledArrayToBranch(LabeledArray<T>&& array)
{
  boost::property_tree::ptree subtree;
  subtree.put_child(labels_rows_str, basicVectorToBranch(array.getLabelsRows()));
  subtree.put_child(labels_cols_str, basicVectorToBranch(array.getLabelsCols()));
  subtree.put_child("values", basicArray2DToBranch(array.getData()));

  return subtree;
}
} // namespace o2::framework

#endif // FRAMEWORK_VARIANTPTREEHELPERS_H
