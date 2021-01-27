// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "Framework/Variant.h"
#include <iostream>
#include <sstream>

namespace o2::framework
{

namespace
{
template <typename T>
void printArray(std::ostream& oss, T* array, size_t size)
{
  oss << variant_array_symbol<T>::symbol << "[";
  oss << array[0];
  for (auto i = 1U; i < size; ++i) {
    oss << ", " << array[i];
  }
  oss << "]";
}

template <typename T>
void printMatrix(std::ostream& oss, Array2D<T> const& m)
{
  oss << variant_array_symbol<T>::symbol << "[[";
  oss << m(0, 0);
  for (auto j = 1U; j < m.cols; ++j) {
    oss << ", " << m(0, j);
  }
  oss << "]";
  for (auto i = 1U; i < m.rows; ++i) {
    oss << ", [";
    oss << m(i, 0);
    for (auto j = 1U; j < m.cols; ++j) {
      oss << ", " << m(i, j);
    }
    oss << "]";
  }
  oss << "]";
}
} // namespace

std::ostream& operator<<(std::ostream& oss, Variant const& val)
{
  switch (val.type()) {
    case VariantType::Int:
      oss << val.get<int>();
      break;
    case VariantType::Int64:
      oss << val.get<int64_t>();
      break;
    case VariantType::Float:
      oss << val.get<float>();
      break;
    case VariantType::Double:
      oss << val.get<double>();
      break;
    case VariantType::String:
      oss << val.get<const char*>();
      break;
    case VariantType::Bool:
      oss << val.get<bool>();
      break;
    case VariantType::ArrayInt:
      printArray<int>(oss, val.get<int*>(), val.size());
      break;
    case VariantType::ArrayFloat:
      printArray<float>(oss, val.get<float*>(), val.size());
      break;
    case VariantType::ArrayDouble:
      printArray<double>(oss, val.get<double*>(), val.size());
      break;
    case VariantType::ArrayBool:
      printArray<bool>(oss, val.get<bool*>(), val.size());
      break;
    case VariantType::ArrayString:
      printArray<std::string>(oss, val.get<std::string*>(), val.size());
      break;
    case VariantType::Array2DInt:
      printMatrix<int>(oss, val.get<Array2D<int>>());
      break;
    case VariantType::Array2DFloat:
      printMatrix<float>(oss, val.get<Array2D<float>>());
      break;
    case VariantType::Array2DDouble:
      printMatrix<double>(oss, val.get<Array2D<double>>());
      break;
    case VariantType::Empty:
      break;
    default:
      oss << "undefined";
      break;
  };
  return oss;
}

std::string Variant::asString() const
{
  std::stringstream ss;
  ss << *this;
  return ss.str();
}

} // namespace o2::framework
