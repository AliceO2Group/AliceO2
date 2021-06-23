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

/// \file  matrix.h
/// \brief Definition of Vector and Matrix class
///
/// \author  Matthias Kleiner <mkleiner@ikf.uni-frankfurt.de>

#ifndef ALICEO2_TPC_VECTOR_H_
#define ALICEO2_TPC_VECTOR_H_

#include <Vc/Vc>

namespace o2
{
namespace tpc
{
template <typename DataT = float, size_t N = 64>
class Matrix
{
  using VDataT = Vc::Vector<DataT>;

 public:
  /// constructor
  /// \param dataMatrix pointer to the data
  Matrix(const Vc::Memory<VDataT, N>* dataMatrix) : mDataMatrix(dataMatrix) {}

  const Vc::Memory<VDataT, N>& operator[](size_t i) const { return mDataMatrix[i]; }

 private:
  const Vc::Memory<VDataT, N>* mDataMatrix{};
};

template <typename DataT = float, size_t N = 64>
class Vector
{
  using VDataT = Vc::Vector<DataT>;

 public:
  /// default constructor
  Vector() = default;

  /// constructor
  /// \param dataVector data which is assigned to the vector
  Vector(const Vc::Memory<VDataT, N>& dataVector) : mDataVector(dataVector) {}

  /// operator access
  const DataT operator[](size_t i) const { return mDataVector.scalar(i); }
  DataT& operator[](size_t i) { return mDataVector.scalar(i); }

  /// sets the vector with index j
  void setVector(const size_t j, const VDataT& vector) { mDataVector.vector(j) = vector; }

  /// \return returns the vector with index j
  const VDataT getVector(const size_t j) const { return mDataVector.vector(j); }

  /// \return returns the number of Vc::Vector<DataT> stored in the Vector
  size_t getvectorsCount() const { return mDataVector.vectorsCount(); }

  /// \return returns the number of entries stored in the Vector
  size_t getentriesCount() const { return mDataVector.entriesCount(); }

 private:
  // storage for the data
  Vc::Memory<VDataT, N> mDataVector{};
};

template <typename DataT, size_t N>
inline Vector<DataT, N> operator*(const Matrix<DataT, N>& a, const Vector<DataT, N>& b)
{
  using V = Vc::Vector<DataT>;
  // resulting vector c
  Vector<DataT, N> c;
  for (size_t i = 0; i < N; ++i) {
    V c_ij{};
    for (size_t j = 0; j < a[i].vectorsCount(); ++j) {
      c_ij += a[i].vector(j) * b.getVector(j);
    }
    c[i] = c_ij.sum();
  }
  return c;
}

template <typename DataT, size_t N>
inline Vector<DataT, N> floor(const Vector<DataT, N>& a)
{
  Vector<DataT, N> c;
  for (size_t j = 0; j < a.getvectorsCount(); ++j) {
    c.setVector(j, Vc::floor(a.getVector(j)));
  }
  return c;
}

template <typename DataT, size_t N>
inline Vector<DataT, N> operator-(const Vector<DataT, N>& a, const Vector<DataT, N>& b)
{
  // resulting matrix c
  Vector<DataT, N> c;
  for (size_t j = 0; j < a.getvectorsCount(); ++j) {
    c.setVector(j, a.getVector(j) - b.getVector(j));
  }
  return c;
}

template <typename DataT, size_t N>
inline Vector<DataT, N> operator+(const Vector<DataT, N>& a, const Vector<DataT, N>& b)
{
  // resulting matrix c
  Vector<DataT, N> c;
  for (size_t j = 0; j < a.getvectorsCount(); ++j) {
    c.setVector(j, a.getVector(j) + b.getVector(j));
  }
  return c;
}

template <typename DataT, size_t N>
inline Vector<DataT, N> operator*(const DataT a, const Vector<DataT, N>& b)
{
  // resulting matrix c
  Vector<DataT, N> c;
  for (size_t j = 0; j < b.getvectorsCount(); ++j) {
    c.setVector(j, a * b.getVector(j));
  }
  return c;
}

// compute the sum of one Vector
template <typename DataT, size_t N>
inline DataT sum(const Vector<DataT, N>& a)
{
  // resulting matrix c
  Vc::Vector<DataT> b = a.getVector(0);
  for (size_t j = 1; j < a.getvectorsCount(); ++j) {
    b += a.getVector(j);
  }
  return b.sum();
}

// multiply each row from a vector with the row from a second vector
template <typename DataT, size_t N>
inline Vector<DataT, N> operator*(const Vector<DataT, N>& a, const Vector<DataT, N>& b)
{
  // resulting matrix c
  Vector<DataT, N> c{};
  for (size_t j = 0; j < a.getvectorsCount(); ++j) {
    c.setVector(j, a.getVector(j) * b.getVector(j));
  }
  return c;
}

// check if all elements are equal
template <typename DataT, size_t N>
inline bool operator==(const Vector<DataT, N>& a, const Vector<DataT, N>& b)
{
  for (size_t j = 0; j < a.getvectorsCount(); ++j) {
    if (any_of(a.getVector(j) != b.getVector(j))) {
      return false;
    }
  }
  return true;
}

} // namespace tpc
} // namespace o2

#endif
