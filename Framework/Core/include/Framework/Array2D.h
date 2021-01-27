// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_ARRAY2D_H
#define FRAMEWORK_ARRAY2D_H
#include <cstdint>
#include <vector>
#include <unordered_map>
#include <memory>
#include <string>
#include <cassert>

namespace o2::framework
{
// matrix-like wrapper for std::vector
// has no range checks
template <typename T>
struct Array2D {
  using element_t = T;

  Array2D()
    : data{nullptr},
      rows{0},
      cols{0}
  {
  }

  Array2D(T const* data_, uint32_t r, uint32_t c)
    : rows{r}, cols{c}
  {
    data = new T[rows * cols];
    for (auto i = 0U; i < rows; ++i) {
      for (auto j = 0U; j < cols; ++j) {
        data[i * cols + j] = data_[i * cols + j];
      }
    }
  }
  Array2D(std::vector<T> data_, uint32_t r, uint32_t c)
    : rows{r}, cols{c}
  {
    data = new T[rows * cols];
    for (auto i = 0U; i < rows; ++i) {
      for (auto j = 0U; j < cols; ++j) {
        data[i * cols + j] = data_[i * cols + j];
      }
    }
  }

  Array2D(Array2D<T> const& other)
    : rows{other.rows},
      cols{other.cols}
  {
    data = new T[rows * cols];
    for (auto i = 0U; i < rows; ++i) {
      for (auto j = 0U; j < cols; ++j) {
        data[i * cols + j] = other.data[i * cols + j];
      }
    }
  }

  Array2D(Array2D<T>&& other)
    : rows{other.rows},
      cols{other.cols}
  {
    data = other.data;
    other.data = nullptr;
    other.rows = 0;
    other.cols = 0;
  }

  Array2D& operator=(Array2D<T> const& other)
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

  Array2D& operator=(Array2D<T>&& other)
  {
    this->rows = other.rows;
    this->cols = other.cols;
    data = other.data;
    other.data = nullptr;
    other.rows = 0;
    other.cols = 0;
    return *this;
  }

  ~Array2D()
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

  T* data;
  uint32_t rows;
  uint32_t cols;
};

static constexpr const char* const labels_rows_str = "labels_rows";
static constexpr const char* const labels_cols_str = "labels_cols";

template <typename T>
class LabeledArray
{
 public:
  using element_t = T;

  LabeledArray()
    : values{},
      labels_rows{},
      labels_cols{},
      rowmap{},
      colmap{}
  {
  }

  LabeledArray(T const* data, uint32_t rows, uint32_t cols, std::vector<std::string> labels_rows_ = {}, std::vector<std::string> labels_cols_ = {})
    : values{data, rows, cols},
      labels_rows{labels_rows_},
      labels_cols{labels_cols_},
      rowmap{},
      colmap{}
  {
    if (labels_rows.empty() == false) {
      assert(labels_rows.size() == rows);
      for (auto i = 0u; i < labels_rows.size(); ++i) {
        rowmap.emplace(labels_rows[i], (uint32_t)i);
      }
    }
    if (labels_cols.empty() == false) {
      assert(labels_cols.size() == cols);
      for (auto i = 0u; i < labels_cols.size(); ++i) {
        colmap.emplace(labels_cols[i], (uint32_t)i);
      }
    }
  }

  LabeledArray(T const* data, uint32_t size, std::vector<std::string> labels_ = {})
    : values{data, 1, size},
      labels_rows{},
      labels_cols{labels_},
      rowmap{},
      colmap{}
  {
    if (labels_cols.empty() == false) {
      assert(labels_cols.size() == size);
      for (auto i = 0u; i < labels_cols.size(); ++i) {
        colmap.emplace(labels_cols[i], (uint32_t)i);
      }
    }
  }

  LabeledArray(Array2D<T> const& data, std::vector<std::string> labels_rows_ = {}, std::vector<std::string> labels_cols_ = {})
    : values{data},
      labels_rows{labels_rows_},
      labels_cols{labels_cols_},
      rowmap{},
      colmap{}
  {
    if (labels_rows.empty() == false) {
      assert(labels_rows.size() == values.rows);
      for (auto i = 0u; i < labels_rows.size(); ++i) {
        rowmap.emplace(labels_rows[i], (uint32_t)i);
      }
    }
    if (labels_cols.empty() == false) {
      assert(labels_cols.size() == values.cols);
      for (auto i = 0u; i < labels_cols.size(); ++i) {
        colmap.emplace(labels_cols[i], (uint32_t)i);
      }
    }
  }

  LabeledArray(LabeledArray<T> const& other) = default;
  LabeledArray(LabeledArray<T>&& other) = default;
  LabeledArray& operator=(LabeledArray<T> const& other) = default;
  LabeledArray& operator=(LabeledArray<T>&& other) = default;

  ~LabeledArray() = default;

  T get(uint32_t y, uint32_t x) const
  {
    return values(y, x);
  }

  T get(std::string y, std::string x) const
  {
    return values(rowmap.find(y)->second, colmap.find(x)->second);
  }

  T get(uint32_t x) const
  {
    return values[0][x];
  }

  T getRow(u_int32_t y) const
  {
    return values[y];
  }

  T* operator[](uint32_t y) const
  {
    return values[y];
  }

  auto getLabelsRows()
  {
    return labels_rows;
  }

  auto getLabelsCols()
  {
    return labels_cols;
  }

  auto getData()
  {
    return values;
  }

  auto rows()
  {
    return values.rows;
  }

  auto cols()
  {
    return values.cols;
  }

 private:
  Array2D<T> values;
  std::vector<std::string> labels_rows;
  std::vector<std::string> labels_cols;
  std::unordered_map<std::string, uint32_t> rowmap;
  std::unordered_map<std::string, uint32_t> colmap;
};
} // namespace o2::framework

#endif // FRAMEWORK_ARRAY2D_H
