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

#include "Framework/VariantPropertyTreeHelpers.h"

template boost::property_tree::ptree o2::framework::basicVectorToBranch(std::vector<float>&& values);
template boost::property_tree::ptree o2::framework::basicVectorToBranch(std::vector<int>&& values);
template boost::property_tree::ptree o2::framework::basicVectorToBranch(std::vector<double>&& values);
template boost::property_tree::ptree o2::framework::basicVectorToBranch(std::vector<std::string>&& values);
template boost::property_tree::ptree o2::framework::basicVectorToBranch(float*, size_t);
template boost::property_tree::ptree o2::framework::basicVectorToBranch(int*, size_t);
template boost::property_tree::ptree o2::framework::basicVectorToBranch(double*, size_t);
template boost::property_tree::ptree o2::framework::basicVectorToBranch(bool*, size_t);
template boost::property_tree::ptree o2::framework::basicVectorToBranch(std::basic_string<char> const*, size_t);

template boost::property_tree::ptree o2::framework::vectorToBranch(std::vector<float>&& values);
template boost::property_tree::ptree o2::framework::vectorToBranch(std::vector<int>&& values);
template boost::property_tree::ptree o2::framework::vectorToBranch(std::vector<double>&& values);
template boost::property_tree::ptree o2::framework::vectorToBranch(std::vector<std::string>&& values);
template boost::property_tree::ptree o2::framework::vectorToBranch(float const*, size_t);
template boost::property_tree::ptree o2::framework::vectorToBranch(int const*, size_t);
template boost::property_tree::ptree o2::framework::vectorToBranch(double const*, size_t);
template boost::property_tree::ptree o2::framework::vectorToBranch(bool const*, size_t);
template boost::property_tree::ptree o2::framework::vectorToBranch(std::basic_string<char> const*, size_t);

template boost::property_tree::ptree o2::framework::labeledArrayToBranch(o2::framework::LabeledArray<float>&& array);
template boost::property_tree::ptree o2::framework::labeledArrayToBranch(o2::framework::LabeledArray<int>&& array);
template boost::property_tree::ptree o2::framework::labeledArrayToBranch(o2::framework::LabeledArray<double>&& array);
template boost::property_tree::ptree o2::framework::labeledArrayToBranch(o2::framework::LabeledArray<std::string>&& array);

template std::vector<float> o2::framework::basicVectorFromBranch<float>(boost::property_tree::ptree const& tree);
template std::vector<int> o2::framework::basicVectorFromBranch<int>(boost::property_tree::ptree const& tree);
template std::vector<std::basic_string<char>> o2::framework::basicVectorFromBranch<std::basic_string<char>>(boost::property_tree::ptree const& tree);
template std::vector<double> o2::framework::basicVectorFromBranch<double>(boost::property_tree::ptree const& tree);

template o2::framework::LabeledArray<float> o2::framework::labeledArrayFromBranch<float>(boost::property_tree::ptree const& tree);
template o2::framework::LabeledArray<int> o2::framework::labeledArrayFromBranch<int>(boost::property_tree::ptree const& tree);
template o2::framework::LabeledArray<std::string> o2::framework::labeledArrayFromBranch<std::string>(boost::property_tree::ptree const& tree);
template o2::framework::LabeledArray<double> o2::framework::labeledArrayFromBranch<double>(boost::property_tree::ptree const& tree);

template o2::framework::Array2D<float> o2::framework::array2DFromBranch<float>(boost::property_tree::ptree const& tree);
template o2::framework::Array2D<int> o2::framework::array2DFromBranch<int>(boost::property_tree::ptree const& tree);
template o2::framework::Array2D<std::string> o2::framework::array2DFromBranch<std::string>(boost::property_tree::ptree const& tree);
template o2::framework::Array2D<double> o2::framework::array2DFromBranch<double>(boost::property_tree::ptree const& tree);

template boost::property_tree::ptree o2::framework::array2DToBranch(o2::framework::Array2D<float>&& array);
template boost::property_tree::ptree o2::framework::array2DToBranch(o2::framework::Array2D<int>&& array);
template boost::property_tree::ptree o2::framework::array2DToBranch(o2::framework::Array2D<double>&& array);
template boost::property_tree::ptree o2::framework::array2DToBranch(o2::framework::Array2D<std::string>&& array);
