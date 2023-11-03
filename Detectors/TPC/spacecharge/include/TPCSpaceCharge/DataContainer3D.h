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

#include "Rtypes.h"
#include <vector>

class TFile;
class TTree;

namespace o2
{
namespace tpc
{

template <class T>
class RegularGrid3D;

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

  /// \param iz index in z dimension
  /// \param ir index in r dimension
  /// \param iphi index in phi dimension
  /// \return returns the stored value
  const DataT& operator()(size_t iz, size_t ir, size_t iphi) const { return mData[getDataIndex(iz, ir, iphi)]; }

  /// \param iz index in z dimension
  /// \param ir index in r dimension
  /// \param iphi index in phi dimension
  /// \return returns the stored value
  DataT& operator()(size_t iz, size_t ir, size_t iphi) { return mData[getDataIndex(iz, ir, iphi)]; }

  /// \return reutrns interpolated value at arbitrary coordinate
  DataT interpolate(const DataT z, const DataT r, const DataT phi, const o2::tpc::RegularGrid3D<DataT>& grid) const;

  /// \param iz index in z dimension
  /// \param ir index in r dimension
  /// \param iphi index in phi dimension
  /// \return returns the index to the data
  size_t getDataIndex(const size_t iz, const size_t ir, const size_t iphi) const { return (iz + mZVertices * (ir + iphi * mRVertices)); }

  /// \return returns the number of values stored
  size_t getNDataPoints() const { return mData.size(); }

  /// \return returns the number of x vertices
  unsigned short getNZ() const { return mZVertices; }

  /// \return returns the number of y vertices
  unsigned short getNR() const { return mRVertices; }

  /// \return returns the number of z vertices
  unsigned short getNPhi() const { return mPhiVertices; }

  /// define aliases for TTree for drawing
  /// \param input TTree for which the aliases will be defined
  static void setAliases(TTree* tree);

  /// define aliases for TTree for drawing which was created by dumpSlice() or dumpInterpolation()
  /// \param tree TTree for which the aliases will be defined
  static void setAliasesForDump(TTree* tree);

  /// \return returns index iz for given global index (can be used for TTree::Draw("second : o2::tpc::DataContainer3D<float>::getIndexZ(first+Iteration$, 129, 129, 360) ","","colz"))
  /// \param index global index
  /// \param nz number of vertices in z
  /// \param nr number of vertices in r
  /// \param nphi number of vertices in phi
  static size_t getIndexZ(size_t index, const int nz, const int nr, const int nphi);
  size_t getIndexZ(size_t index) const { return getIndexZ(index, getNZ(), getNR(), getNPhi()); }

  /// \return returns index ir for given global index (can be used for TTree::Draw("second : o2::tpc::DataContainer3D<float>::getIndexR(first+Iteration$, 129, 129, 360) ","","colz"))
  /// \param index global index
  /// \param nz number of vertices in z
  /// \param nr number of vertices in r
  /// \param nphi number of vertices in phi
  static size_t getIndexR(size_t index, const int nz, const int nr, const int nphi);
  size_t getIndexR(size_t index) const { return getIndexR(index, getNZ(), getNR(), getNPhi()); }

  /// \return returns index iphi for given global index (can be used for TTree::Draw("second : o2::tpc::DataContainer3D<float>::getIndexPhi(first+Iteration$, 129, 129, 360) ","","colz"))
  /// \param index global index
  /// \param nz number of vertices in z
  /// \param nr number of vertices in r
  /// \param nphi number of vertices in phi
  static size_t getIndexPhi(size_t index, const int nz, const int nr, const int nphi);
  size_t getIndexPhi(size_t index) const { return getIndexPhi(index, getNZ(), getNR(), getNPhi()); }

  /// set the grid points
  void setGrid(unsigned short nZ, unsigned short nR, unsigned short nPhi, const bool resize);

  /// write this object to a file (deprecated!)
  /// \param outf object is written to this file
  /// \param name object is saved with this name
  /// \tparam DataTOut format of the output container (can be used to store the container with a different precission than the current object)
  template <typename DataTOut = DataT>
  int writeToFile(TFile& outf, const char* name = "data") const;

  /// write this object to a file using RDataFrame
  /// \param file object is written to this file
  /// \param option "RECREATE" or "UPDATE"
  /// \param name object is saved with this name
  /// \param nthreads number of threads to use
  int writeToFile(std::string_view file, std::string_view option, std::string_view name = "data", const int nthreads = 1) const;

  /// set values from file (deprecated!)
  /// \tparam DataTOut format of the input container (can be used to load the container with a different precission than the current object)
  template <typename DataTIn = DataT>
  bool initFromFile(TFile& inpf, const char* name = "data");

  /// set values from file using RDataFrame
  /// \param file object is written to this file
  /// \param name object is saved with this name
  /// \param nthreads number of threads to use
  bool initFromFile(std::string_view file, std::string_view name = "data", const int nthreads = 1);

  /// get pointer to object from file (deprecated!)
  inline static DataContainer3D<DataT>* loadFromFile(TFile& inpf, const char* name = "data");

  /// dump slices to TTree including indices for Drawing
  /// \param treename name of the TTree in the input file for which the slices will be dumped
  /// \param fileIn input file
  /// \param fileOut output file
  /// \param option "RECREATE" or "UPDATE" the output file
  /// \param rangeiR indices range in radial direction
  /// \param rangeiZ indices range in z direction
  /// \param rangeiPhi indices range in phi direction
  static void dumpSlice(std::string_view treename, std::string_view fileIn, std::string_view fileOut, std::string_view option, std::pair<unsigned short, unsigned short> rangeiR, std::pair<unsigned short, unsigned short> rangeiZ, std::pair<unsigned short, unsigned short> rangeiPhi, const int nthreads = 1);

  /// dump interpolations of stored values to TTree including indices for Drawing
  /// \param treename name of the TTree in the input file for which the slices will be dumped
  /// \param fileIn input file
  /// \param fileOut output file
  /// \param option "RECREATE" or "UPDATE" the output file
  /// \param rangeR range in radial direction
  /// \param rangeZ range in z direction
  /// \param rangePhi range in phi direction
  /// \param nR number of points in radial direction which will be used to interpolate
  /// \param nZ number of points in z direction which will be used to interpolate
  /// \param nPhi number of points in phi direction which will be used to interpolate
  static void dumpInterpolation(std::string_view treename, std::string_view fileIn, std::string_view fileOut, std::string_view option, std::pair<float, float> rangeR, std::pair<float, float> rangeZ, std::pair<float, float> rangePhi, const int nR, const int nZ, const int nPhi, const int nthreads = 1);

  /// sets vertices definition stored from object in file
  /// \return true if vertices could be read from the input file
  /// \param treename name of the TTree in the input file for which the slices will be dumped
  /// \param fileIn input file
  /// \param nR vertices in radial direction will be stored in this variable
  /// \param nZ vertices in z direction will be stored in this variable
  /// \param nPhi vertices in phi direction will be stored in this variable
  static bool getVertices(std::string_view treename, std::string_view fileIn, unsigned short& nR, unsigned short& nZ, unsigned short& nPhi);

  /// print the matrix
  void print() const;

  /// operator overload
  DataContainer3D<DataT>& operator*=(const DataT value);
  DataContainer3D<DataT>& operator+=(const DataContainer3D<DataT>& other);
  DataContainer3D<DataT>& operator*=(const DataContainer3D<DataT>& other);
  DataContainer3D<DataT>& operator-=(const DataContainer3D<DataT>& other);

 private:
  unsigned short mZVertices{};   ///< number of z vertices
  unsigned short mRVertices{};   ///< number of r vertices
  unsigned short mPhiVertices{}; ///< number of phi vertices
  std::vector<DataT> mData{};    ///< storage for the data

  static auto getDataSlice(const std::vector<DataT>& data, size_t entries, const size_t values_per_entry, ULong64_t entry);

  ClassDefNV(DataContainer3D, 1)
};

} // namespace tpc
} // namespace o2

#endif
