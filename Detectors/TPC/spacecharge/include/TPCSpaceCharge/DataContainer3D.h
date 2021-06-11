// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

namespace o2
{
namespace tpc
{

/// \class DataContainer3D
/// The DataContainer3D class represents a simple method to store values on a large 3-Dim grid with ROOT io functionality.

/// \tparam DataT the type of data which is used during the calculations
/// \tparam Nx number of values in x direction
/// \tparam Ny number of values in y direction
/// \tparam Nz number of values in z direction
template <typename DataT = double, unsigned int Nx = 129, unsigned int Ny = 129, unsigned int Nz = 180>
struct DataContainer3D {

  ///< default constructor
  DataContainer3D() : mData(FN){};

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
  static constexpr size_t getDataIndex(const size_t ix, const size_t iy, const size_t iz)
  {
    const size_t index = ix + Nx * (iy + iz * Ny);
    return index;
  }

  /// \return returns the number of values stored
  static constexpr size_t getNDataPoints() { return FN; }

  /// \return returns the number of x vertices
  static constexpr size_t getNX() { return Nx; }

  /// \return returns the number of y vertices
  static constexpr size_t getNY() { return Ny; }

  /// \return returns the number of z vertices
  static constexpr size_t getNZ() { return Nz; }

  /// write this object to a file
  /// \param outf object is written to this file
  /// \param name object is saved with this name
  int writeToFile(TFile& outf, const char* name = "data") const;

  /// set values from file
  bool initFromFile(TFile& inpf, const char* name = "data");

  /// get pointer to object from file
  inline static DataContainer3D<DataT, Nx, Ny, Nz>* loadFromFile(TFile& inpf, const char* name = "data");

  /// print the matrix
  void print() const;

 private:
  static constexpr size_t FN{Nx * Ny * Nz}; ///< number of values stored in the container
  std::vector<DataT> mData;                 ///< storage for the data

  ClassDefNV(DataContainer3D, 1)
};

///
/// ========================================================================================================
///                                Inline implementations of some methods
/// ========================================================================================================
///

template <typename DataT, unsigned int Nx, unsigned int Ny, unsigned int Nz>
int DataContainer3D<DataT, Nx, Ny, Nz>::writeToFile(TFile& outf, const char* name) const
{
  if (outf.IsZombie()) {
    LOGP(ERROR, "Failed to write to file: {}", outf.GetName());
    return -1;
  }
  outf.WriteObjectAny(this, DataContainer3D<DataT, Nx, Ny, Nz>::Class(), name);
  return 0;
}

/// set values from file
template <typename DataT, unsigned int Nx, unsigned int Ny, unsigned int Nz>
bool DataContainer3D<DataT, Nx, Ny, Nz>::initFromFile(TFile& inpf, const char* name)
{
  if (inpf.IsZombie()) {
    LOGP(ERROR, "Failed to read from file: {}", inpf.GetName());
    return false;
  }
  DataContainer3D<DataT, Nx, Ny, Nz>* dataCont{nullptr};

  dataCont = reinterpret_cast<DataContainer3D<DataT, Nx, Ny, Nz>*>(inpf.GetObjectChecked(name, DataContainer3D<DataT, Nx, Ny, Nz>::Class()));
  if (!dataCont) {
    LOGP(ERROR, "Failed to load {} from {}", name, inpf.GetName());
    return false;
  }
  mData = dataCont->mData;
  delete dataCont;
  return true;
}

template <typename DataT, unsigned int Nx, unsigned int Ny, unsigned int Nz>
DataContainer3D<DataT, Nx, Ny, Nz>* DataContainer3D<DataT, Nx, Ny, Nz>::loadFromFile(TFile& inpf, const char* name)
{
  if (inpf.IsZombie()) {
    LOGP(ERROR, "Failed to read from file {}", inpf.GetName());
    return nullptr;
  }
  DataContainer3D<DataT, Nx, Ny, Nz>* dataCont{nullptr};

  dataCont = reinterpret_cast<DataContainer3D<DataT, Nx, Ny, Nz>*>(inpf.GetObjectChecked(name, DataContainer3D<DataT, Nx, Ny, Nz>::Class()));
  if (!dataCont) {
    LOGP(ERROR, "Failed to load {} from {}", name, inpf.GetName());
    return nullptr;
  }
  return dataCont;
}

template <typename DataT, unsigned int Nx, unsigned int Ny, unsigned int Nz>
void DataContainer3D<DataT, Nx, Ny, Nz>::print() const
{
  std::stringstream stream;
  stream.precision(3);
  auto&& w = std::setw(9);
  stream << std::endl;

  for (unsigned int iz = 0; iz < Nz; ++iz) {
    stream << "z layer: " << iz << "\n";
    // print top x row
    stream << "⎡" << w << (*this)(0, 0, iz);
    for (unsigned int ix = 1; ix < Nx; ++ix) {
      stream << ", " << w << (*this)(ix, 0, iz);
    }
    stream << " ⎤ \n";

    for (unsigned int iy = 1; iy < Ny - 1; ++iy) {
      stream << "⎢" << w << (*this)(0, iy, iz);
      for (unsigned int ix = 1; ix < Nx; ++ix) {
        stream << ", " << w << (*this)(ix, iy, iz);
      }
      stream << " ⎥ \n";
    }

    stream << "⎣" << w << (*this)(0, Ny - 1, iz);
    for (unsigned int ix = 1; ix < Nx; ++ix) {
      stream << ", " << w << (*this)(ix, Ny - 1, iz);
    }
    stream << " ⎦ \n \n";
  }
  LOGP(info, "{} \n \n", stream.str());
}

} // namespace tpc
} // namespace o2

#endif
