// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Vector3D.h
/// \brief this is a simple 3D-matrix class with the possibility to resize the dimensions
///
/// \author  Matthias Kleiner <mkleiner@ikf.uni-frankfurt.de>
/// \date Oct 23, 2020

#ifndef ALICEO2_TPC_VECTOR3D_H_
#define ALICEO2_TPC_VECTOR3D_H_

namespace o2
{
namespace tpc
{

/// this is a simple vector class which is used in the poisson solver class

/// \tparam DataT the data type of the mStorage which is used during the calculations
template <typename DataT = double>
class Vector3D
{
 public:
  /// Constructor for a tricubic interpolator
  /// \param nr number of data points in r directions
  /// \param nz number of data points in r directions
  /// \param nphi number of data points in r directions
  Vector3D(const unsigned int nr, const unsigned int nz, const unsigned int nphi) : mNr{nr}, mNz{nz}, mNphi{nphi}, mStorage{nr * nz * nphi} {};

  /// default constructor
  Vector3D() = default;

  /// operator to set the values
  DataT& operator()(const unsigned int iR, const unsigned int iZ, const unsigned int iPhi)
  {
    return mStorage[getIndex(iR, iZ, iPhi)];
  }

  /// operator to read the values
  const DataT& operator()(const unsigned int iR, const unsigned int iZ, const unsigned int iPhi) const
  {
    return mStorage[getIndex(iR, iZ, iPhi)];
  }

  /// operator to directly access the values
  DataT& operator[](const unsigned int index)
  {
    return mStorage[index];
  }

  const DataT& operator[](const unsigned int index) const
  {
    return mStorage[index];
  }

  /// \param iR index in r direction
  /// \param iZ index in z direction
  /// \param iPhi index in phi direction
  /// \return returns the index for given indices
  int getIndex(const unsigned int iR, const unsigned int iZ, const unsigned int iPhi) const
  {
    return iR + mNr * (iZ + mNz * iPhi);
  }

  /// resize the vector
  /// \param nr number of data points in r directions
  /// \param nz number of data points in r directions
  /// \param nphi number of data points in r directions
  void resize(const unsigned int nr, const unsigned int nz, const unsigned int nphi)
  {
    mNr = nr;
    mNz = nz;
    mNphi = nphi;
    mStorage.resize(nr * nz * nphi);
  }

  const auto& data() const { return mStorage; }
  auto& data() { return mStorage; }

  unsigned int getNr() const { return mNr; }          ///< get number of data points in r direction
  unsigned int getNz() const { return mNz; }          ///< get number of data points in z direction
  unsigned int getNphi() const { return mNphi; }      ///< get number of data points in phi direction
  unsigned int size() const { return mStorage.size; } ///< get number of data points

  auto begin() const { return mStorage.begin(); }
  auto begin() { return mStorage.begin(); }

  auto end() const { return mStorage.end(); }
  auto end() { return mStorage.end(); }

 private:
  unsigned int mNr{};            ///< number of data points in r direction
  unsigned int mNz{};            ///< number of data points in z direction
  unsigned int mNphi{};          ///< number of data points in phi direction
  std::vector<DataT> mStorage{}; ///< vector containing the data
};

} // namespace tpc
} // namespace o2

#endif
