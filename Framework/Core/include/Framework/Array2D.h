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

namespace o2::framework
{
// matrix-like wrapper for C-array
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

using labelmap_t = std::unordered_map<std::string, uint32_t>;
namespace
{
struct LabelMap {
  LabelMap()
    : rowmap{},
      colmap{},
      labels_rows{},
      labels_cols{}
  {
  }
  LabelMap(uint32_t rows, uint32_t cols, std::vector<std::string> labels_rows_, std::vector<std::string> labels_cols_)
    : rowmap{populate(rows, labels_rows_)},
      colmap{populate(cols, labels_cols_)},
      labels_rows{labels_rows_},
      labels_cols{labels_cols_}
  {
  }

  LabelMap(uint32_t size, std::vector<std::string> labels)
    : rowmap{},
      colmap{populate(size, labels)},
      labels_rows{},
      labels_cols{labels}
  {
  }

  LabelMap(LabelMap const& other) = default;
  LabelMap(LabelMap&& other) = default;
  LabelMap& operator=(LabelMap const& other) = default;
  LabelMap& operator=(LabelMap&& other) = default;

  labelmap_t rowmap;
  labelmap_t colmap;

  std::vector<std::string> labels_rows;
  std::vector<std::string> labels_cols;

  labelmap_t populate(uint32_t size, std::vector<std::string> labels)
  {
    assert(labels.size() == size);
    labelmap_t map;
    for (auto i = 0u; i < size; ++i) {
      map.emplace(labels[i], (uint32_t)i);
    }
    return map;
  }

  auto getLabelsRows()
  {
    return labels_rows;
  }

  auto getLabelsCols()
  {
    return labels_cols;
  }
};
} // namespace

template <typename T>
class LabeledArray : public LabelMap
{
 public:
  using element_t = T;

  LabeledArray()
    : values{},
      LabelMap{}
  {
  }

  LabeledArray(T const* data, uint32_t rows_, uint32_t cols_, std::vector<std::string> labels_rows_ = {}, std::vector<std::string> labels_cols_ = {})
    : values{data, rows_, cols_},
      LabelMap{rows_, cols_, labels_rows_, labels_cols_}
  {
  }

  LabeledArray(T const* data, uint32_t size, std::vector<std::string> labels_ = {})
    : values{data, 1, size},
      LabelMap{size, labels_}
  {
  }

  LabeledArray(Array2D<T> const& data, std::vector<std::string> labels_rows_ = {}, std::vector<std::string> labels_cols_ = {})
    : values{data},
      LabelMap{data.rows, data.cols, labels_rows_, labels_cols_}
  {
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

  T get(const char* y, const char* x) const
  {
    return values(rowmap.find(y)->second, colmap.find(x)->second);
  }

  T get(const char* y, uint32_t x) const
  {
    return values(rowmap.find(y)->second, x);
  }

  T get(uint32_t y, const char* x) const
  {
    return values(y, colmap.find(x)->second);
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
};
} // namespace o2::framework

#endif // FRAMEWORK_ARRAY2D_H
