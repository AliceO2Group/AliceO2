// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_MATRIX_H
#define FRAMEWORK_MATRIX_H
#include <cstdint>
#include <vector>

namespace o2::framework
{
// matrix-like wrapper for std::vector
// has no range checks
template <typename T>
struct matrix {
  using element_t = T;

  matrix(T const* data_, uint32_t r, uint32_t c)
    : rows{r}, cols{c}
  {
    data = new T[rows * cols];
    for (auto i = 0U; i < rows; ++i) {
      for (auto j = 0U; j < cols; ++j) {
        data[i * cols + j] = *(data_ + (i * cols + j));
      }
    }
  }
  matrix(std::vector<T> data_, uint32_t r, uint32_t c)
    : rows{r}, cols{c}
  {
    data = new T[rows * cols];
    for (auto i = 0U; i < rows; ++i) {
      for (auto j = 0U; j < cols; ++j) {
        data[i * cols + j] = data_[i * cols + j];
      }
    }
  }

  matrix(matrix<T> const& other)
    : rows{other.rows},
      cols{other.cols}
  {
    data = new T[rows * cols];
    for (auto i = 0U; i < rows; ++i) {
      for (auto j = 0U; j < cols; ++j) {
        data[i * cols + j] = *(other.data + (i * cols + j));
      }
    }
  }

  matrix(matrix<T>&& other)
    : rows{other.rows},
      cols{other.cols}
  {
    data = other.data;
    other.data = nullptr;
    other.rows = 0;
    other.cols = 0;
  }

  matrix& operator=(matrix<T> const& other)
  {
    this->rows = other.rows;
    this->cols = other.cols;
    data = new T[rows * cols];
    for (auto i = 0U; i < rows; ++i) {
      for (auto j = 0U; j < cols; ++j) {
        data[i * cols + j] = *(other.data + (i * cols + j));
      }
    }
    return *this;
  }

  matrix& operator=(matrix<T>&& other)
  {
    this->rows = other.rows;
    this->cols = other.cols;
    data = other.data;
    other.data = nullptr;
    other.rows = 0;
    other.cols = 0;
    return *this;
  }

  ~matrix()
  {
    if (data != nullptr) {
      delete[] data;
    }
  }

  T operator()(uint32_t y, uint32_t x) const
  {
    return data[y * cols + x];
  }
  T* operator[](uint32_t y) const
  {
    return data + y * cols;
  }

  T* data = nullptr;
  uint32_t rows = 0;
  uint32_t cols = 0;
};
} // namespace o2::framework

#endif // FRAMEWORK_MATRIX_H
