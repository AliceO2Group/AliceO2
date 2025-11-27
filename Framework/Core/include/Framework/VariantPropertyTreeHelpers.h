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
#ifndef FRAMEWORK_VARIANTPTREEHELPERS_H
#define FRAMEWORK_VARIANTPTREEHELPERS_H

#include "Array2D.h"
#include "Framework/Variant.h"
#include <boost/property_tree/ptree.hpp>

namespace o2::framework
{
template <typename T>
boost::property_tree::ptree basicVectorToBranch(T* values, size_t size)
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
boost::property_tree::ptree basicVectorToBranch(std::vector<T>&& values)
{
  return basicVectorToBranch(values.data(), values.size());
}

template <typename T>
boost::property_tree::ptree vectorToBranch(T const* values, size_t size)
{
  boost::property_tree::ptree branch;
  branch.put_child("values", basicVectorToBranch(values, size));
  return branch;
}

template <typename T>
boost::property_tree::ptree vectorToBranch(std::vector<T>&& values)
{
  return vectorToBranch(values.data(), values.size());
}

template <typename T>
boost::property_tree::ptree basicArray2DToBranch(Array2D<T>&& array)
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

template <typename T>
boost::property_tree::ptree array2DToBranch(Array2D<T>&& array)
{
  boost::property_tree::ptree subtree;
  subtree.put_child("values", basicArray2DToBranch(std::forward<Array2D<T>>(array)));
  return subtree;
}

template <typename T>
std::vector<T> basicVectorFromBranch(boost::property_tree::ptree const& branch)
{
  std::vector<T> result(branch.size());
  auto count = 0U;
  for (auto const& entry : branch) {
    result[count++] = entry.second.get_value<T>();
  }
  return result;
}

template <typename T>
std::vector<T> vectorFromBranch(boost::property_tree::ptree const& branch)
{
  return basicVectorFromBranch<T>(branch.get_child("values"));
}

template <typename T>
Array2D<T> basicArray2DFromBranch(boost::property_tree::ptree const& branch)
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

template <typename T>
Array2D<T> array2DFromBranch(boost::property_tree::ptree const& ptree)
{
  return basicArray2DFromBranch<T>(ptree.get_child("values"));
}

std::pair<std::vector<std::string>, std::vector<std::string>> extractLabels(boost::property_tree::ptree const& tree);

template <typename T>
LabeledArray<T> labeledArrayFromBranch(boost::property_tree::ptree const& tree)
{
  auto [labels_rows, labels_cols] = extractLabels(tree);
  auto values = basicArray2DFromBranch<T>(tree.get_child("values"));

  return LabeledArray<T>{values, labels_rows, labels_cols};
}

template <typename T>
boost::property_tree::ptree labeledArrayToBranch(LabeledArray<T>&& array)
{
  boost::property_tree::ptree subtree;
  subtree.put_child(labels_rows_str, basicVectorToBranch(array.getLabelsRows()));
  subtree.put_child(labels_cols_str, basicVectorToBranch(array.getLabelsCols()));
  subtree.put_child("values", basicArray2DToBranch(array.getData()));

  return subtree;
}
} // namespace o2::framework

extern template boost::property_tree::ptree o2::framework::basicVectorToBranch(std::vector<float>&& values);
extern template boost::property_tree::ptree o2::framework::basicVectorToBranch(std::vector<int>&& values);
extern template boost::property_tree::ptree o2::framework::basicVectorToBranch(std::vector<double>&& values);
extern template boost::property_tree::ptree o2::framework::basicVectorToBranch(std::vector<std::string>&& values);
extern template boost::property_tree::ptree o2::framework::basicVectorToBranch(float*, size_t);
extern template boost::property_tree::ptree o2::framework::basicVectorToBranch(int*, size_t);
extern template boost::property_tree::ptree o2::framework::basicVectorToBranch(double*, size_t);
extern template boost::property_tree::ptree o2::framework::basicVectorToBranch(bool*, size_t);
extern template boost::property_tree::ptree o2::framework::basicVectorToBranch(std::basic_string<char> const*, size_t);

extern template boost::property_tree::ptree o2::framework::vectorToBranch(std::vector<float>&& values);
extern template boost::property_tree::ptree o2::framework::vectorToBranch(std::vector<int>&& values);
extern template boost::property_tree::ptree o2::framework::vectorToBranch(std::vector<double>&& values);
extern template boost::property_tree::ptree o2::framework::vectorToBranch(std::vector<std::string>&& values);
extern template boost::property_tree::ptree o2::framework::vectorToBranch(float const*, size_t);
extern template boost::property_tree::ptree o2::framework::vectorToBranch(int const*, size_t);
extern template boost::property_tree::ptree o2::framework::vectorToBranch(double const*, size_t);
extern template boost::property_tree::ptree o2::framework::vectorToBranch(bool const*, size_t);
extern template boost::property_tree::ptree o2::framework::vectorToBranch(std::basic_string<char> const*, size_t);

extern template boost::property_tree::ptree o2::framework::labeledArrayToBranch(o2::framework::LabeledArray<float>&& array);
extern template boost::property_tree::ptree o2::framework::labeledArrayToBranch(o2::framework::LabeledArray<int>&& array);
extern template boost::property_tree::ptree o2::framework::labeledArrayToBranch(o2::framework::LabeledArray<double>&& array);
extern template boost::property_tree::ptree o2::framework::labeledArrayToBranch(o2::framework::LabeledArray<std::string>&& array);

extern template std::vector<float> o2::framework::basicVectorFromBranch<float>(boost::property_tree::ptree const& tree);
extern template std::vector<int> o2::framework::basicVectorFromBranch<int>(boost::property_tree::ptree const& tree);
extern template std::vector<std::basic_string<char>> o2::framework::basicVectorFromBranch<std::basic_string<char>>(boost::property_tree::ptree const& tree);
extern template std::vector<double> o2::framework::basicVectorFromBranch<double>(boost::property_tree::ptree const& tree);

extern template o2::framework::LabeledArray<float> o2::framework::labeledArrayFromBranch<float>(boost::property_tree::ptree const& tree);
extern template o2::framework::LabeledArray<int> o2::framework::labeledArrayFromBranch<int>(boost::property_tree::ptree const& tree);
extern template o2::framework::LabeledArray<std::string> o2::framework::labeledArrayFromBranch<std::string>(boost::property_tree::ptree const& tree);
extern template o2::framework::LabeledArray<double> o2::framework::labeledArrayFromBranch<double>(boost::property_tree::ptree const& tree);

extern template o2::framework::Array2D<float> o2::framework::array2DFromBranch<float>(boost::property_tree::ptree const& tree);
extern template o2::framework::Array2D<int> o2::framework::array2DFromBranch<int>(boost::property_tree::ptree const& tree);
extern template o2::framework::Array2D<std::string> o2::framework::array2DFromBranch<std::string>(boost::property_tree::ptree const& tree);
extern template o2::framework::Array2D<double> o2::framework::array2DFromBranch<double>(boost::property_tree::ptree const& tree);

extern template boost::property_tree::ptree o2::framework::array2DToBranch(o2::framework::Array2D<float>&& array);
extern template boost::property_tree::ptree o2::framework::array2DToBranch(o2::framework::Array2D<int>&& array);
extern template boost::property_tree::ptree o2::framework::array2DToBranch(o2::framework::Array2D<double>&& array);
extern template boost::property_tree::ptree o2::framework::array2DToBranch(o2::framework::Array2D<std::string>&& array);
#endif // FRAMEWORK_VARIANTPTREEHELPERS_H
