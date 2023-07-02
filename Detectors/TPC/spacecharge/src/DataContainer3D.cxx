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
#include "ROOT/RDataFrame.hxx"

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

template <typename DataT>
int DataContainer3D<DataT>::writeToFile(std::string_view file, std::string_view option, std::string_view name, const int nthreads) const
{
  // max number of floats per Entry
  const size_t maxvalues = sizeof(float) * 1024 * 1024;

  // total number of values to be stored
  const size_t nsize = getNDataPoints();

  // calculate number of entries in the tree and restrict if the number of values per threads exceeds max size
  size_t entries = ((nsize / nthreads) > maxvalues) ? (nsize / maxvalues) : nthreads;

  if (entries > nsize) {
    entries = nsize;
  }

  // calculate numbers to store per entry
  const size_t values_per_entry = nsize / entries;

  // in case of remainder add additonal entry
  const size_t values_lastEntry = nsize % entries;
  if (values_lastEntry) {
    entries += 1;
  }

  // in case EnableImplicitMT was already called with different number of threads, perform reset
  if (ROOT::IsImplicitMTEnabled() && (ROOT::GetThreadPoolSize() != nthreads)) {
    ROOT::DisableImplicitMT();
  }
  ROOT::EnableImplicitMT(nthreads);

  // define dataframe which will be stored in the TTree
  ROOT::RDataFrame dFrame(entries);

  // define function which is used to fill the data frame
  auto dfStore = dFrame.DefineSlotEntry(name, [&data = std::as_const(mData), entries, values_per_entry](unsigned int, ULong64_t entry) { return DataContainer3D<DataT>::getDataSlice(data, entries, values_per_entry, entry); });
  dfStore = dfStore.Define("nz", [mZVertices = mZVertices]() { return mZVertices; });
  dfStore = dfStore.Define("nr", [mRVertices = mRVertices]() { return mRVertices; });
  dfStore = dfStore.Define("nphi", [mPhiVertices = mPhiVertices]() { return mPhiVertices; });

  // define options of TFile
  ROOT::RDF::RSnapshotOptions opt;
  opt.fMode = option;
  opt.fOverwriteIfExists = true; // overwrite if already exists

  TStopwatch timer;
  // note: first call has some overhead (~2s)
  dfStore.Snapshot(name, file, {name.data(), "nz", "nr", "nphi"}, opt);
  timer.Print("u");
  return 0;
}

template <typename DataT>
bool DataContainer3D<DataT>::initFromFile(std::string_view file, std::string_view name, const int nthreads)
{
  // in case EnableImplicitMT was already called with different number of threads, perform reset
  if (ROOT::IsImplicitMTEnabled() && (ROOT::GetThreadPoolSize() != nthreads)) {
    ROOT::DisableImplicitMT();
  }
  ROOT::EnableImplicitMT(nthreads);

  // compare first the meta data (is the number of vertices the same)
  // define data frame from imput file
  ROOT::RDataFrame dFrame(name, file);

  // compare vertices
  auto comp = [mZVertices = mZVertices, mRVertices = mRVertices, mPhiVertices = mPhiVertices](const unsigned short nz, const unsigned short nr, const unsigned short nphi) {
    if ((nz == mZVertices) && (nr == mRVertices) && (nphi == mPhiVertices)) {
      return false;
    }
    return true;
  };

  auto count = dFrame.Filter(comp, {"nz", "nr", "nphi"}).Count();
  if (*count != 0) {
    LOGP(error, "Data from input file has different number of vertices! Found {} same vertices", *count);
    return false;
  }

  // define lambda function which is used to copy the data
  auto readData = [&mData = mData](const std::pair<long, std::vector<float>>& data) {
    std::copy(data.second.begin(), data.second.end(), mData.begin() + data.first);
  };

  LOGP(info, "Reading {} from file {}", name, file);

  // fill data from RDataFrame
  TStopwatch timer;
  dFrame.Foreach(readData, {name.data()});
  timer.Print("u");
  return true;
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
auto DataContainer3D<DataT>::getDataSlice(const std::vector<DataT>& data, size_t entries, const size_t values_per_entry, ULong64_t entry)
{
  const long indStart = entry * values_per_entry;
  if (entry < (entries - 1)) {
    return std::pair(indStart, std::vector<float>(data.begin() + indStart, data.begin() + indStart + values_per_entry));
  } else if (entry == (entries - 1)) {
    // last entry might have different number of values. just copy the rest...
    return std::pair(indStart, std::vector<float>(data.begin() + indStart, data.end()));
  }
  return std::pair(indStart, std::vector<float>());
};

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
void DataContainer3D<DataT>::dumpSlice(std::string_view treename, std::string_view fileIn, std::string_view fileOut, std::string_view option, std::pair<unsigned short, unsigned short> rangeiR, std::pair<unsigned short, unsigned short> rangeiZ, std::pair<unsigned short, unsigned short> rangeiPhi, const int nthreads)
{
  if (ROOT::IsImplicitMTEnabled() && (ROOT::GetThreadPoolSize() != nthreads)) {
    ROOT::DisableImplicitMT();
  }
  ROOT::EnableImplicitMT(nthreads);
  ROOT::RDataFrame dFrame(treename, fileIn);

  auto df = dFrame.Define("slice", [rangeiZ, rangeiR, rangeiPhi](const std::pair<long, std::vector<float>>& values, unsigned short nz, unsigned short nr, unsigned short nphi) {
    const bool simOneSectorOnly = MGParameters::normalizeGridToOneSector;
    std::vector<size_t> ir;
    std::vector<size_t> iphi;
    std::vector<size_t> iz;
    std::vector<float> r;
    std::vector<float> phi;
    std::vector<float> z;
    std::vector<float> vals;
    std::vector<size_t> globalIdx;
    std::vector<LocalPosition3D> lPos;
    const auto nvalues = values.second.size();
    ir.reserve(nvalues);
    iphi.reserve(nvalues);
    iz.reserve(nvalues);
    r.reserve(nvalues);
    phi.reserve(nvalues);
    z.reserve(nvalues);
    vals.reserve(nvalues);
    lPos.reserve(nvalues);
    globalIdx.reserve(nvalues);
    for (size_t i = 0; i < nvalues; ++i) {
      const size_t idx = values.first + i;
      const auto iZTmp = o2::tpc::DataContainer3D<float>::getIndexZ(idx, nz, nr, nphi);
      if ((rangeiZ.first < rangeiZ.second) && ((iZTmp < rangeiZ.first) || (iZTmp > rangeiZ.second))) {
        continue;
      }

      const auto iRTmp = o2::tpc::DataContainer3D<float>::getIndexR(idx, nz, nr, nphi);
      if ((rangeiR.first < rangeiR.second) && ((iRTmp < rangeiR.first) || (iRTmp > rangeiR.second))) {
        continue;
      }

      const auto iPhiTmp = o2::tpc::DataContainer3D<float>::getIndexPhi(idx, nz, nr, nphi);
      if ((rangeiPhi.first < rangeiPhi.second) && ((iPhiTmp < rangeiPhi.first) || (iPhiTmp > rangeiPhi.second))) {
        continue;
      }

      const float rTmp = o2::tpc::GridProperties<float>::getRMin() + o2::tpc::GridProperties<float>::getGridSpacingR(nr) * iRTmp;
      const float zTmp = o2::tpc::GridProperties<float>::getZMin() + o2::tpc::GridProperties<float>::getGridSpacingZ(nz) * iZTmp;
      const float phiTmp = o2::tpc::GridProperties<float>::getPhiMin() + o2::tpc::GridProperties<float>::getGridSpacingPhi(nphi) / (simOneSectorOnly ? SECTORSPERSIDE : 1) * iPhiTmp;

      const float x = rTmp * std::cos(phiTmp);
      const float y = rTmp * std::sin(phiTmp);
      const LocalPosition3D pos(x, y, zTmp);
      unsigned char secNum = simOneSectorOnly ? 0 : std::floor(phiTmp / SECPHIWIDTH);
      Sector sector(secNum + (pos.Z() < 0) * SECTORSPERSIDE);
      LocalPosition3D lPosTmp = Mapper::GlobalToLocal(pos, sector);

      lPos.emplace_back(lPosTmp);
      ir.emplace_back(iRTmp);
      iphi.emplace_back(iPhiTmp);
      iz.emplace_back(iZTmp);
      r.emplace_back(rTmp);
      phi.emplace_back(phiTmp);
      z.emplace_back(zTmp);
      vals.emplace_back(values.second[i]);
      globalIdx.emplace_back(idx);
    }
    return std::make_tuple(vals, iz, ir, iphi, z, r, phi, lPos, globalIdx);
  },
                          {treename.data(), "nz", "nr", "nphi"});

  // define options of TFile
  ROOT::RDF::RSnapshotOptions opt;
  opt.fMode = option;
  df.Snapshot(treename, fileOut, {"slice"}, opt);
}

template <typename DataT>
DataT DataContainer3D<DataT>::interpolate(const DataT z, const DataT r, const DataT phi, const o2::tpc::RegularGrid3D<DataT>& grid) const
{
  TriCubicInterpolator<DataT> interpolator(*this, grid);
  return interpolator(z, r, phi);
}

template <typename DataT>
void DataContainer3D<DataT>::dumpInterpolation(std::string_view treename, std::string_view fileIn, std::string_view fileOut, std::string_view option, std::pair<float, float> rangeR, std::pair<float, float> rangeZ, std::pair<float, float> rangePhi, const int nR, const int nZ, const int nPhi, const int nthreads)
{
  if (ROOT::IsImplicitMTEnabled() && (ROOT::GetThreadPoolSize() != nthreads)) {
    ROOT::DisableImplicitMT();
  }
  ROOT::EnableImplicitMT(nthreads);
  ROOT::RDataFrame dFrame(nPhi);

  // get vertices for input TTree which is needed to define the grid for interpolation
  unsigned short nr, nz, nphi;
  if (!getVertices(treename, fileIn, nr, nz, nphi)) {
    return;
  }

  // load data from input TTree
  DataContainer3D<DataT> data;
  data.setGrid(nz, nr, nphi, true);
  data.initFromFile(fileIn, treename, nthreads);

  // define grid for interpolation
  using GridProp = GridProperties<DataT>;
  const RegularGrid3D<DataT> mGrid3D(GridProp::ZMIN, GridProp::RMIN, GridProp::PHIMIN, GridProp::getGridSpacingZ(nz), GridProp::getGridSpacingR(nr), o2::tpc::GridProperties<float>::getGridSpacingPhi(nphi) / (MGParameters::normalizeGridToOneSector ? SECTORSPERSIDE : 1), ParamSpaceCharge{nr, nz, nphi});

  auto interpolate = [&mGrid3D = std::as_const(mGrid3D), &data = std::as_const(data), rangeR, rangeZ, rangePhi, nR, nZ, nPhi](unsigned int, ULong64_t iPhi) {
    const bool simOneSectorOnly = MGParameters::normalizeGridToOneSector;
    std::vector<size_t> ir;
    std::vector<size_t> iphi;
    std::vector<size_t> iz;
    std::vector<float> r;
    std::vector<float> phi;
    std::vector<float> z;
    std::vector<float> vals;
    std::vector<size_t> globalIdx;
    std::vector<LocalPosition3D> lPos;
    const auto nvalues = nR * nZ;
    ir.reserve(nvalues);
    iphi.reserve(nvalues);
    iz.reserve(nvalues);
    r.reserve(nvalues);
    phi.reserve(nvalues);
    z.reserve(nvalues);
    vals.reserve(nvalues);
    lPos.reserve(nvalues);
    globalIdx.reserve(nvalues);

    const float rSpacing = (rangeR.second - rangeR.first) / (nR - 1);
    const float zSpacing = (rangeZ.second - rangeZ.first) / (nZ - 1);
    const float phiSpacing = (rangePhi.second - rangePhi.first) / (nPhi - 1);
    const DataT phiPos = rangePhi.first + iPhi * phiSpacing;
    // loop over grid and interpolate values
    for (int iR = 0; iR < nR; ++iR) {
      const DataT rPos = rangeR.first + iR * rSpacing;
      for (int iZ = 0; iZ < nZ; ++iZ) {
        const size_t idx = (iZ + nZ * (iR + iPhi * nR)); // unique index to Build index with other friend TTrees
        const DataT zPos = rangeZ.first + iZ * zSpacing;
        ir.emplace_back(iR);
        iphi.emplace_back(iPhi);
        iz.emplace_back(iZ);
        r.emplace_back(rPos);
        phi.emplace_back(phiPos);
        z.emplace_back(zPos);
        vals.emplace_back(data.interpolate(zPos, rPos, phiPos, mGrid3D)); // interpolated values
        globalIdx.emplace_back(idx);
        const float x = rPos * std::cos(phiPos);
        const float y = rPos * std::sin(phiPos);
        const LocalPosition3D pos(x, y, zPos);
        unsigned char secNum = simOneSectorOnly ? 0 : std::floor(phiPos / SECPHIWIDTH);
        Sector sector(secNum + (pos.Z() < 0) * SECTORSPERSIDE);
        LocalPosition3D lPosTmp = Mapper::GlobalToLocal(pos, sector);
        lPos.emplace_back(lPosTmp);
      }
    }
    return std::make_tuple(vals, iz, ir, iphi, z, r, phi, lPos, globalIdx);
  };

  // define RDataFrame entry
  auto dfStore = dFrame.DefineSlotEntry(treename, interpolate);

  // define options of TFile
  ROOT::RDF::RSnapshotOptions opt;
  opt.fMode = option;

  TStopwatch timer;
  // note: first call has some overhead (~2s)
  dfStore.Snapshot(treename, fileOut, {treename.data()}, opt);
  timer.Print("u");
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
