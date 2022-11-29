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

/// \file DataContainer3D.h
/// \brief This class provides a simple method to store values on a large 3-Dim grid with ROOT io functionality
///
/// \author  Matthias Kleiner <mkleiner@ikf.uni-frankfurt.de>
/// \date Aug 21, 2020

#ifndef ALICEO2_TPC_DATACONTAINER3D_H_
#define ALICEO2_TPC_DATACONTAINER3D_H_

#include <memory>
#include "TFile.h"
#include "Rtypes.h"
#include "Framework/Logger.h"
#include <iomanip>
#include <vector>

namespace o2
{
namespace tpc
{

/// \class DataContainer3D
/// The DataContainer3D class represents a simple method to store values on a large 3-Dim grid with ROOT io functionality.

/// \tparam DataT the type of data which is used during the calculations
template <typename DataT = double>
struct DataContainer3D {

  ///< constructor
  /// \param nZ number of vertices in z direction
  /// \param nR number of vertices in r direction
  /// \param nPhi number of vertices in phi direction
  DataContainer3D(unsigned short nZ, unsigned short nR, unsigned short nPhi) : mZVertices{nZ}, mRVertices{nR}, mPhiVertices{nPhi}, mData(nZ * nR * nPhi){};

  ///< default constructor for Root I/O
  DataContainer3D() = default;

  /// operator to directly access the values
  const DataT& operator[](size_t i) const { return mData[i]; }
  DataT& operator[](size_t i) { return mData[i]; }

  const auto& getData() const { return mData; }
  auto& getData() { return mData; }

  /// \param ix index in x dimension
  /// \param iy index in y dimension
  /// \param iz index in z dimension
  /// \return returns the stored value
  const DataT& operator()(size_t ix, size_t iy, size_t iz) const
  {
    const size_t ind = getDataIndex(ix, iy, iz);
    return mData[ind];
  }

  /// \param ix index in x dimension
  /// \param iy index in y dimension
  /// \param iz index in z dimension
  /// \return returns the stored value
  DataT& operator()(size_t ix, size_t iy, size_t iz)
  {
    const size_t ind = getDataIndex(ix, iy, iz);
    return mData[ind];
  }

  /// \param ix index in x dimension
  /// \param iy index in y dimension
  /// \param iz index in z dimension
  /// \return returns the index to the data
  size_t getDataIndex(const size_t ix, const size_t iy, const size_t iz) const
  {
    const size_t index = ix + mZVertices * (iy + iz * mRVertices);
    return index;
  }

  /// \return returns the number of values stored
  size_t getNDataPoints() const { return mData.size(); }

  /// \return returns the number of x vertices
  unsigned short getNZ() const { return mZVertices; }

  /// \return returns the number of y vertices
  unsigned short getNR() const { return mRVertices; }

  /// \return returns the number of z vertices
  unsigned short getNPhi() const { return mPhiVertices; }

  /// write this object to a file
  /// \param outf object is written to this file
  /// \param name object is saved with this name
  /// \tparam DataTOut format of the output container (can be used to store the container with a different precission than the current object)
  template <typename DataTOut = DataT>
  int writeToFile(TFile& outf, const char* name = "data") const;

  /// set values from file
  /// \tparam DataTOut format of the input container (can be used to load the container with a different precission than the current object)
  template <typename DataTIn = DataT>
  bool initFromFile(TFile& inpf, const char* name = "data");

  /// get pointer to object from file
  inline static DataContainer3D<DataT>* loadFromFile(TFile& inpf, const char* name = "data");

  /// print the matrix
  void print() const;

  /// operator overload
  DataContainer3D<DataT>& operator*=(const DataT value);

 private:
  unsigned short mZVertices{};   ///< number of z vertices
  unsigned short mRVertices{};   ///< number of r vertices
  unsigned short mPhiVertices{}; ///< number of phi vertices
  std::vector<DataT> mData{};    ///< storage for the data

  ClassDefNV(DataContainer3D, 1)
};

template <typename DataT>
DataContainer3D<DataT>& DataContainer3D<DataT>::operator*=(const DataT value)
{
  std::transform(mData.begin(), mData.end(), mData.begin(), [value = value](auto& val) { return val * value; });
  return *this;
}

///
/// ========================================================================================================
///                                Inline implementations of some methods
/// ========================================================================================================
///

template <typename DataT>
template <typename DataTOut>
int DataContainer3D<DataT>::writeToFile(TFile& outf, const char* name) const
{
  if (outf.IsZombie()) {
    LOGP(error, "Failed to write to file: {}", outf.GetName());
    return -1;
  }

  DataContainer3D<DataTOut> containerTmp(mZVertices, mRVertices, mPhiVertices);
  containerTmp.getData() = std::vector<DataTOut>(mData.begin(), mData.end());

  outf.WriteObjectAny(&containerTmp, DataContainer3D<DataTOut>::Class(), name);
  return 0;
}

/// set values from file
template <typename DataT>
template <typename DataTIn>
bool DataContainer3D<DataT>::initFromFile(TFile& inpf, const char* name)
{
  if (inpf.IsZombie()) {
    LOGP(error, "Failed to read from file: {}", inpf.GetName());
    return false;
  }
  DataContainer3D<DataTIn>* dataCont{nullptr};
  dataCont = reinterpret_cast<DataContainer3D<DataTIn>*>(inpf.GetObjectChecked(name, DataContainer3D<DataTIn>::Class()));

  if (!dataCont) {
    LOGP(error, "Failed to load {} from {}", name, inpf.GetName());
    return false;
  }

  if (mZVertices != dataCont->getNZ() || mRVertices != dataCont->getNR() || mPhiVertices != dataCont->getNPhi()) {
    LOGP(error, "Data from input file has different definition of vertices!");
    LOGP(error, "set vertices before creating the sc object to: SpaceCharge<>::setGrid({}, {}, {})", dataCont->getNZ(), dataCont->getNR(), dataCont->getNPhi());
    delete dataCont;
    return false;
  }

  mData = std::vector<DataT>(dataCont->getData().begin(), dataCont->getData().end());
  delete dataCont;
  return true;
}

template <typename DataT>
DataContainer3D<DataT>* DataContainer3D<DataT>::loadFromFile(TFile& inpf, const char* name)
{
  if (inpf.IsZombie()) {
    LOGP(error, "Failed to read from file {}", inpf.GetName());
    return nullptr;
  }
  DataContainer3D<DataT>* dataCont{nullptr};

  dataCont = reinterpret_cast<DataContainer3D<DataT>*>(inpf.GetObjectChecked(name, DataContainer3D<DataT>::Class()));
  if (!dataCont) {
    LOGP(error, "Failed to load {} from {}", name, inpf.GetName());
    return nullptr;
  }
  return dataCont;
}

template <typename DataT>
void DataContainer3D<DataT>::print() const
{
  std::stringstream stream;
  stream.precision(3);
  auto&& w = std::setw(9);
  stream << std::endl;

  for (unsigned int iz = 0; iz < mPhiVertices; ++iz) {
    stream << "z layer: " << iz << "\n";
    // print top x row
    stream << "⎡" << w << (*this)(0, 0, iz);
    for (unsigned int ix = 1; ix < mZVertices; ++ix) {
      stream << ", " << w << (*this)(ix, 0, iz);
    }
    stream << " ⎤ \n";

    for (unsigned int iy = 1; iy < mRVertices - 1; ++iy) {
      stream << "⎢" << w << (*this)(0, iy, iz);
      for (unsigned int ix = 1; ix < mZVertices; ++ix) {
        stream << ", " << w << (*this)(ix, iy, iz);
      }
      stream << " ⎥ \n";
    }

    stream << "⎣" << w << (*this)(0, mRVertices - 1, iz);
    for (unsigned int ix = 1; ix < mZVertices; ++ix) {
      stream << ", " << w << (*this)(ix, mRVertices - 1, iz);
    }
    stream << " ⎦ \n \n";
  }
  LOGP(info, "{} \n \n", stream.str());
}

} // namespace tpc
} // namespace o2

#endif
