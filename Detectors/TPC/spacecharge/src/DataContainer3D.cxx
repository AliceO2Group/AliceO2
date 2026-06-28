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

#include "TPCSpaceCharge/DataContainer3D.h"
#include "TPCSpaceCharge/PoissonSolverHelpers.h"
#include "TPCSpaceCharge/TriCubic.h"
#include "TPCSpaceCharge/RegularGrid3D.h"
#include "TPCBase/Mapper.h"
#include "Framework/Logger.h"
#include "TFile.h"
#include "TStopwatch.h"
#include "TTree.h"

#include <memory>
#include <iomanip>
#include <algorithm>

using namespace o2::tpc;

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


template <typename DataT>
DataContainer3D<DataT>& DataContainer3D<DataT>::operator*=(const DataT value)
{
  std::transform(mData.begin(), mData.end(), mData.begin(), [value = value](auto& val) { return val * value; });
  return *this;
}

template <typename DataT>
DataContainer3D<DataT>& DataContainer3D<DataT>::operator+=(const DataContainer3D<DataT>& other)
{
  std::transform(mData.begin(), mData.end(), other.mData.begin(), mData.begin(), std::plus<>());
  return *this;
}

template <typename DataT>
DataContainer3D<DataT>& DataContainer3D<DataT>::operator-=(const DataContainer3D<DataT>& other)
{
  std::transform(mData.begin(), mData.end(), other.mData.begin(), mData.begin(), std::minus<>());
  return *this;
}

template <typename DataT>
DataContainer3D<DataT>& DataContainer3D<DataT>::operator*=(const DataContainer3D<DataT>& other)
{
  std::transform(mData.begin(), mData.end(), other.mData.begin(), mData.begin(), std::multiplies<>());
  return *this;
}

template <typename DataT>
size_t DataContainer3D<DataT>::getIndexZ(size_t index, const int nz, const int nr, const int nphi)
{
  const size_t iphi = index / (nz * nr);
  index -= (iphi * nz * nr);
  const size_t iz = index % nz;
  return iz;
}

template <typename DataT>
size_t DataContainer3D<DataT>::getIndexR(size_t index, const int nz, const int nr, const int nphi)
{
  const size_t iphi = index / (nz * nr);
  index -= (iphi * nz * nr);
  const size_t ir = index / nz;
  return ir;
}

template <typename DataT>
size_t DataContainer3D<DataT>::getIndexPhi(size_t index, const int nz, const int nr, const int nphi)
{
  return index / (nz * nr);
}

template <typename DataT>
void DataContainer3D<DataT>::setAliases(TTree* tree)
{
  tree->SetAlias("ir", "o2::tpc::DataContainer3D<float>::getIndexR(first + Iteration$, nz, nr, nphi)");
  tree->SetAlias("iz", "o2::tpc::DataContainer3D<float>::getIndexZ(first + Iteration$, nz, nr, nphi)");
  tree->SetAlias("iphi", "o2::tpc::DataContainer3D<float>::getIndexPhi(first + Iteration$, nz, nr, nphi)");
  tree->SetAlias("r", "o2::tpc::GridProperties<float>::getRMin() + o2::tpc::GridProperties<float>::getGridSpacingR(nr) * ir");
  tree->SetAlias("z", "o2::tpc::GridProperties<float>::getZMin() + o2::tpc::GridProperties<float>::getGridSpacingZ(nz) * iz");
  tree->SetAlias("phi", "o2::tpc::GridProperties<float>::getPhiMin() + o2::tpc::GridProperties<float>::getGridSpacingPhi(nphi) * iphi");
}

template <typename DataT>
void DataContainer3D<DataT>::setAliasesForDump(TTree* tree)
{
  // actuall stored value
  tree->SetAlias("val", "_0");

  // some meta data
  tree->SetAlias("iz", "_1");
  tree->SetAlias("ir", "_2");
  tree->SetAlias("iphi", "_3");
  tree->SetAlias("z", "_4");
  tree->SetAlias("r", "_5");
  tree->SetAlias("phi", "_6");
  tree->SetAlias("lpos", "_7");
  tree->SetAlias("lx", "lpos.fCoordinates.fX");
  tree->SetAlias("ly", "lpos.fCoordinates.fY");
  tree->SetAlias("index", "_8");
}

template <typename DataT>
void DataContainer3D<DataT>::setGrid(unsigned short nZ, unsigned short nR, unsigned short nPhi, const bool resize)
{
  mZVertices = nZ;
  mRVertices = nR;
  mPhiVertices = nPhi;
  if (resize) {
    mData.resize(nZ * nR * static_cast<size_t>(nPhi));
  }
}


template <typename DataT>
DataT DataContainer3D<DataT>::interpolate(const DataT z, const DataT r, const DataT phi, const o2::tpc::RegularGrid3D<DataT>& grid) const
{
  TriCubicInterpolator<DataT> interpolator(*this, grid);
  return interpolator(z, r, phi);
}


template <typename DataT>
bool DataContainer3D<DataT>::getVertices(std::string_view treename, std::string_view fileIn, unsigned short& nR, unsigned short& nZ, unsigned short& nPhi)
{
  TFile fTmp(fileIn.data(), "READ");
  TTree* tree = (TTree*)fTmp.Get(treename.data());
  if (!tree) {
    LOGP(warning, "Tree {} not found in input file {}", treename, fileIn);
    return false;
  }
  tree->SetBranchAddress("nz", &nZ);
  tree->SetBranchAddress("nr", &nR);
  tree->SetBranchAddress("nphi", &nPhi);
  tree->GetEntry(0);
  delete tree;
  return true;
}

template <typename DataT>
DataContainer3D<DataT> DataContainer3D<DataT>::convert(const o2::tpc::RegularGrid3D<DataT>& gridNew, const o2::tpc::RegularGrid3D<DataT>& gridRef, const int threads) const
{
  const int nZNew = gridNew.getNZ();
  const int nRNew = gridNew.getNR();
  const int nPhiNew = gridNew.getNPhi();
  DataContainer3D<DataT> contCont(nZNew, nRNew, nPhiNew);
#pragma omp parallel for num_threads(threads)
  for (size_t iPhi = 0; iPhi < nPhiNew; ++iPhi) {
    const DataT phi = gridNew.getPhiVertex(iPhi);
    for (size_t iR = 0; iR < nRNew; ++iR) {
      const DataT radius = gridNew.getRVertex(iR);
      for (size_t iZ = 0; iZ < nZNew; ++iZ) {
        const DataT z = gridNew.getZVertex(iZ);
        contCont(iZ, iR, iPhi) = interpolate(z, radius, phi, gridRef);
      }
    }
  }
  return contCont;
}

template class o2::tpc::DataContainer3D<float>;
template class o2::tpc::DataContainer3D<double>;

// deprecated functions (to be removed...)
template int o2::tpc::DataContainer3D<float>::writeToFile<float>(TFile&, const char*) const;
template int o2::tpc::DataContainer3D<float>::writeToFile<double>(TFile&, const char*) const;
template int o2::tpc::DataContainer3D<double>::writeToFile<float>(TFile&, const char*) const;
template int o2::tpc::DataContainer3D<double>::writeToFile<double>(TFile&, const char*) const;
template bool o2::tpc::DataContainer3D<float>::initFromFile<float>(TFile&, const char*);
template bool o2::tpc::DataContainer3D<float>::initFromFile<double>(TFile&, const char*);
template bool o2::tpc::DataContainer3D<double>::initFromFile<float>(TFile&, const char*);
template bool o2::tpc::DataContainer3D<double>::initFromFile<double>(TFile&, const char*);
