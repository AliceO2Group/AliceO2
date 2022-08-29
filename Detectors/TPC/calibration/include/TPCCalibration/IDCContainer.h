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

/// \file IDCContainer.h
/// \brief This file provides the structs for storing the factorized IDC values and fourier coefficients to be stored in the CCDB
///
/// \author  Matthias Kleiner <mkleiner@ikf.uni-frankfurt.de>
/// \date Apr 30, 2021

#ifndef ALICEO2_TPC_IDCCONTAINER_H_
#define ALICEO2_TPC_IDCCONTAINER_H_

#include <array>
#include <vector>
#include <limits>
#include "DataFormatsTPC/Defs.h"
#include "TPCCalibration/IDCGroupingParameter.h"
#include "TPCCalibration/SACParameter.h"

namespace o2
{
namespace tpc
{

/// IDC types
enum class IDCType { IDC = 0,     ///< integrated and grouped IDCs
                     IDCZero = 1, ///< IDC0: I_0(r,\phi) = <I(r,\phi,t)>_t
                     IDCOne = 2,  ///< IDC1: I_1(t) = <I(r,\phi,t) / I_0(r,\phi)>_{r,\phi}
                     IDCDelta = 3 ///< IDCDelta: \Delta I(r,\phi,t) = I(r,\phi,t) / ( I_0(r,\phi) * I_1(t) )
};

/// define alias for SACs
using SACType = IDCType;

/// IDC Delta IDC Compression types
enum class IDCDeltaCompression { NO = 0,     ///< no compression using floats
                                 MEDIUM = 1, ///< medium compression using short (data compression ratio 2 when stored in CCDB)
                                 HIGH = 2    ///< high compression using char (data compression ratio ~5.5 when stored in CCDB)
};

/// struct containing the IDC delta values
template <typename DataT>
struct IDCDeltaContainer {
  std::vector<DataT> mIDCDeltaCont{}; ///< \Delta I(r,\phi,t) = I(r,\phi,t) / ( I_0(r,\phi) * I_1(t) )
};

/// storage for the factor used to compress IDCDelta.
/// This factor is separated from the IDCDelta struct to able to store those values independently in the CCDB
struct IDCDeltaCompressionFactors {
  std::pair<float, float> mFactors{std::pair{1.f, 1.f}}; ///< compression factors
};

/// struct to access and set Delta IDCs
template <typename DataT>
struct IDCDelta {

  /// set idcDelta for given index
  /// \param idcDelta Delta IDC value which will be set
  /// \param index index in the storage
  void setValue(const float idcDelta, const unsigned int index) { mIDCDelta.mIDCDeltaCont[index] = compressValue(idcDelta); }

  /// set idcDelta ath the end of storage
  /// \param idcDelta Delta IDC value which will be set
  void emplace_back(const float idcDelta) { mIDCDelta.mIDCDeltaCont.emplace_back(compressValue(idcDelta)); }

  /// \return returns converted IDC value from float to new data type specified by template parameter of the object
  /// \param idcDelta Delta IDC value which will be set
  DataT compressValue(float idcDelta) const
  {
    idcDelta = (std::clamp(idcDelta, mCompressionFactor.mFactors.first, mCompressionFactor.mFactors.second) - mCompressionFactor.mFactors.first) * std::numeric_limits<DataT>::max() / (mCompressionFactor.mFactors.second - mCompressionFactor.mFactors.first);
    return std::nearbyint(idcDelta);
  }

  /// \return returns stored Delta IDC value
  /// \param index index in the storage (see: getIndexUngrouped int IDCGroupHelperSector)
  float getValue(const unsigned int index) const { return mCompressionFactor.mFactors.first + (mCompressionFactor.mFactors.second - mCompressionFactor.mFactors.first) * static_cast<float>(mIDCDelta.mIDCDeltaCont[index]) / std::numeric_limits<DataT>::max(); }

  /// set compression factor
  /// \param factor factor which will be used for the compression
  void setCompressionFactor(const float factorMin, const float factorMax) { mCompressionFactor.mFactors = std::pair{factorMin, factorMax}; }

  /// \return returns vector of Delta IDCs
  const auto& getIDCDelta() const { return mIDCDelta.mIDCDeltaCont; }

  /// \return returns vector of Delta IDCs
  auto& getIDCDelta() { return mIDCDelta.mIDCDeltaCont; }

  /// set IDCDelta value
  /// \param index index in the storage (see: getIndexUngrouped int IDCGroupHelperSector)
  /// \param val value of IDCDelta which will be stored
  void setIDCDelta(const unsigned int index, const DataT val) { mIDCDelta.mIDCDeltaCont[index] = val; }

  /// \return returns compression factors to uncompress Delta IDC
  const auto& getCompressionFactors() const { return mCompressionFactor; }

  /// \return returns compression factors to uncompress Delta IDC
  auto getCompressionFactor() const { return mCompressionFactor.mFactors; }

  /// \return returns number of stored IDCs
  auto getNIDCs() const { return getIDCDelta().size(); }

  IDCDeltaContainer<DataT> mIDCDelta{};            ///< storage for Delta IDCs
  IDCDeltaCompressionFactors mCompressionFactor{}; ///< compression factor for Delta IDCs
  ClassDefNV(IDCDelta, 2)
};

// template specialization for float as no compression factor is needed in this case
template <>
struct IDCDelta<float> {
  /// set idcDelta for given index
  /// \param idcDelta Delta IDC value which will be set
  /// \param index index in the storage
  void setValue(const float idcDelta, const unsigned int index) { mIDCDelta.mIDCDeltaCont[index] = idcDelta; }

  /// \return returns stored Delta IDC value
  /// \param index index in the storage
  float getValue(const unsigned int index) const { return mIDCDelta.mIDCDeltaCont[index]; }

  /// \return returns vector of Delta IDCs
  const auto& getIDCDelta() const { return mIDCDelta.mIDCDeltaCont; }

  /// \return returns vector of Delta IDCs
  auto& getIDCDelta() { return mIDCDelta.mIDCDeltaCont; }

  /// \return returns vector of Delta IDCs
  auto getIDCDeltaContainer() && { return std::move(mIDCDelta); }

  /// set IDCDelta value
  /// \param index index in the storage (see: getIndexUngrouped int IDCGroupHelperSector)
  /// \param val value of IDCDelta which will be stored
  void setIDCDelta(const unsigned int index, const float val) { mIDCDelta.mIDCDeltaCont[index] = val; }

  /// resize the container
  /// \param size new size of the container
  void resize(const unsigned int size) { mIDCDelta.mIDCDeltaCont.resize(size); }

  /// \return returns number of stored IDCs
  auto getNIDCs() const { return getIDCDelta().size(); }

  IDCDeltaContainer<float> mIDCDelta{}; ///< storage for uncompressed Delta IDCs
  ClassDefNV(IDCDelta, 2)
};

///< struct containing the IDC0 values
struct IDCZero {

  /// set IDC zero for given index
  /// \param idcZero Delta IDC value which will be set
  /// \param index index in the storage
  void setValueIDCZero(const float idcZero, const unsigned int index) { mIDCZero[index] = idcZero; }

  /// increase IDC zero for given index
  /// \param idcZero Delta IDC value which will be set
  /// \param index index in the storage
  void fillValueIDCZero(const float idcZero, const unsigned int index) { mIDCZero[index] += idcZero; }

  /// \return returns stored IDC zero value
  /// \param index index in the storage
  float getValueIDCZero(const unsigned int index) const { return mIDCZero[index]; }

  /// \param clearing IDC0 values
  void clear() { mIDCZero.clear(); }

  /// returns check if the IDC0 vector is empty
  bool empty() const { return mIDCZero.empty(); }

  /// resize vector
  void resize(const unsigned int size) { mIDCZero.resize(size); }

  /// get number of IDC0 values
  auto getNIDC0() const { return mIDCZero.size(); }

  std::vector<float> mIDCZero{}; ///< I_0(r,\phi) = <I(r,\phi,t)>_t
  ClassDefNV(IDCZero, 2)
};

///< struct containing the IDC1
struct IDCOne {

  /// default constructor
  IDCOne() = default;

  /// constructor for initializing member with default value (this is used in the IDCFourierTransform class to perform calculation of the fourier coefficients for the first aggregation interval)
  /// \param nIDC number of IDCs which will be initialized
  IDCOne(const unsigned int nIDC) : mIDCOne{std::vector<float>(nIDC)} {};

  /// set IDC one for given index
  /// \param idcOne Delta IDC value which will be set
  /// \param index index in the storage
  void setValueIDCOne(const float idcOne, const unsigned int index) { mIDCOne[index] = idcOne; }

  /// \return returns stored IDC one value
  /// \param index index in the storage
  float getValueIDCOne(const unsigned int index) const { return mIDCOne[index]; }

  /// \return returns number of stored IDCs
  auto getNIDCs() const { return mIDCOne.size(); }

  /// clearing the IDCOne values
  void clear() { mIDCOne.clear(); }

  /// resize vector
  void resize(const unsigned int size) { mIDCOne.resize(size); }

  std::vector<float> mIDCOne{}; ///< I_1(t) = <I(r,\phi,t) / I_0(r,\phi)>_{r,\phi}
  ClassDefNV(IDCOne, 2)
};

/// Helper class for aggregation of 1D-IDCs from different CRUs
class IDCOneAggregator
{
 public:
  /// aggregate 1D-IDCs
  /// \param side side of the tpcCRUHeader
  /// \param idc vector containing the 1D-IDCs
  void aggregate1DIDCs(const o2::tpc::Side side, const std::vector<float>& idc)
  {
    if (mIDCOneAgg[side].mIDCOne.empty()) {
      mIDCOneAgg[side].mIDCOne = idc;
    } else {
      std::transform(mIDCOneAgg[side].mIDCOne.begin(), mIDCOneAgg[side].mIDCOne.end(), idc.begin(), mIDCOneAgg[side].mIDCOne.begin(), std::plus<>());
    }
  }

  void aggregate1DIDCsWeights(const o2::tpc::Side side, const std::vector<unsigned int>& idcCount)
  {
    if (mWeight[side].empty()) {
      mWeight[side] = idcCount;
    } else {
      std::transform(mWeight[side].begin(), mWeight[side].end(), idcCount.begin(), mWeight[side].begin(), std::plus<>());
    }
  }

  void normalizeIDCOne()
  {
    normalizeIDCOne(Side::A);
    normalizeIDCOne(Side::C);
  }

  /// \return normalize aggregated IDC1 to number of channels
  void normalizeIDCOne(const o2::tpc::Side side)
  {
    std::transform(mIDCOneAgg[side].mIDCOne.begin(), mIDCOneAgg[side].mIDCOne.end(), mWeight[side].begin(), mIDCOneAgg[side].mIDCOne.begin(), std::divides<>());
  }

  /// \return returns IDC1 data by move and clears weights
  auto get(const Side side) &&
  {
    mWeight[side].clear();
    return std::move(mIDCOneAgg[side]);
  }

  /// \return returns weights for the stored values
  const auto& getWeight() const { return mWeight; }

 private:
  std::array<IDCOne, SIDES> mIDCOneAgg{};                          ///< 1D-IDCs = <I(r,\phi,t)>_{r,\phi}
  std::array<std::vector<unsigned int>, o2::tpc::SIDES> mWeight{}; ///< Number of channels used for IDC1 calculation used for normalization
};

/// struct containing the fourier coefficients calculated from IDC0 for n timeframes
struct FourierCoeff {
  /// constructor
  /// \param nTimeFrames number of time frames
  /// \param nCoeff number of real/imag fourier coefficients which will be stored
  FourierCoeff(const unsigned int nTimeFrames, const unsigned int nCoeff)
    : mFourierCoefficients{std::vector<float>(nTimeFrames * nCoeff)}, mCoeffPerTF{nCoeff} {};

  /// default constructor for ROOT I/O
  FourierCoeff() = default;

  /// \return returns total number of stored coefficients for given side and real/complex type
  unsigned long getNCoefficients() const { return mFourierCoefficients.size(); }

  /// \return returns number of stored coefficients for TF
  unsigned int getNCoefficientsPerTF() const { return mCoeffPerTF; }

  /// \return returns all stored coefficients for given side and real/complex type
  const auto& getFourierCoefficients() const { return mFourierCoefficients; }

  /// \return returns index to fourier coefficient
  /// \param interval index of interval
  /// \param coefficient index of coefficient
  unsigned int getIndex(const unsigned int interval, const unsigned int coefficient) const { return interval * mCoeffPerTF + coefficient; }

  /// \return returns the stored value
  /// \param index index of the data
  float operator()(unsigned int index) const { return mFourierCoefficients[index]; }

  /// \return returns the stored value
  /// \param index index of the data
  float& operator()(unsigned int index) { return mFourierCoefficients[index]; }

  void reset() { std::fill(mFourierCoefficients.begin(), mFourierCoefficients.end(), 0.f); };

  void resize(const unsigned int nTimeFrames) { mFourierCoefficients.resize(nTimeFrames * mCoeffPerTF); }

  std::vector<float> mFourierCoefficients{}; ///< fourier coefficients. coefficient real and complex parameters are stored alternating
  unsigned int mCoeffPerTF{};                ///< number of real+imag coefficients per TF

  ClassDefNV(FourierCoeff, 2)
};

struct SACZero {
  float getValueIDCZero(const Side side, const int stackInSector) const { return mSACZero[side].getValueIDCZero(stackInSector); }
  void clear()
  {
    mSACZero[Side::A].clear();
    mSACZero[Side::C].clear();
  }

  void resize(const unsigned int size)
  {
    mSACZero[Side::A].resize(size);
    mSACZero[Side::C].resize(size);
  }

  void setValueSACZero(const float sacZero, const Side side, const unsigned int index) { mSACZero[side].setValueIDCZero(sacZero, index); }

  std::array<IDCZero, SIDES> mSACZero{};
};

struct SACOne {
  float getValue(const Side side, const int interval) const { return mSACOne[side].mIDCOne[interval]; }
  void setValueIDCOne(const float sacOne, const Side side, const unsigned int index) { mSACOne[side].setValueIDCOne(sacOne, index); }

  void clear()
  {
    mSACOne[Side::A].clear();
    mSACOne[Side::C].clear();
  }
  void resize(const unsigned int size)
  {
    mSACOne[Side::A].resize(size);
    mSACOne[Side::C].resize(size);
  }
  std::array<IDCOne, SIDES> mSACOne{};
};

template <typename DataT>
struct SACDelta {
  float getValue(const Side side, const int index) const { return mSACDelta[side].getValue(index); }
  void setSACDelta(const Side side, const unsigned int index, const float sacDelta) { mSACDelta[side].setIDCDelta(index, sacDelta); }

  void resize(const unsigned int size)
  {
    mSACDelta[Side::A].resize(size);
    mSACDelta[Side::C].resize(size);
  }
  std::array<IDCDelta<DataT>, SIDES> mSACDelta{};
};

struct FourierCoeffSAC {
  std::array<FourierCoeff, SIDES> mCoeff{};
};

/// helper class to compress Delta IDC values
/// \tparam template parameter specifying the output format of the compressed IDCDelta
template <typename DataT>
class IDCDeltaCompressionHelper
{
 public:
  IDCDeltaCompressionHelper() = default;

  /// static method to get the compressed Delta IDCs from uncompressed Delta IDCs
  /// \return returns compressed Delta IDC values
  /// \param idcDeltaUncompressed uncompressed Delta IDC values
  static IDCDelta<DataT> getCompressedIDCs(const IDCDelta<float>& idcDeltaUncompressed)
  {
    const auto& paramIDCGroup = ParameterIDCCompression::Instance();

    IDCDelta<DataT> idcCompressed{};
    compress(idcDeltaUncompressed, idcCompressed, paramIDCGroup.minIDCDeltaValue, paramIDCGroup.maxIDCDeltaValue);
    return idcCompressed;
  }

  /// static method to get the compressed SACs from uncompressed Delta SACs
  /// \return returns compressed Delta SAC values
  /// \param sacDeltaUncompressed uncompressed Delta SAC values
  static SACDelta<DataT> getCompressedSACs(const SACDelta<float>& sacDeltaUncompressed)
  {
    const auto& paramSAC = ParameterSAC::Instance();
    SACDelta<DataT> sacCompressed{};
    compress(sacDeltaUncompressed.mSACDelta[Side::A], sacCompressed.mSACDelta[Side::A], paramSAC.minSACDeltaValue, paramSAC.maxSACDeltaValue);
    compress(sacDeltaUncompressed.mSACDelta[Side::C], sacCompressed.mSACDelta[Side::C], paramSAC.minSACDeltaValue, paramSAC.maxSACDeltaValue);
    return sacCompressed;
  }

 private:
  static void compress(const IDCDelta<float>& idcDeltaUncompressed, IDCDelta<DataT>& idcCompressed, const float minValue, const float maxValue)
  {
    idcCompressed.getIDCDelta().reserve(idcDeltaUncompressed.getIDCDelta().size());
    const auto minmaxIDC = std::minmax_element(std::begin(idcDeltaUncompressed.getIDCDelta()), std::end(idcDeltaUncompressed.getIDCDelta()));
    const auto& paramIDCGroup = ParameterIDCCompression::Instance();
    const float minIDCDelta = std::clamp(*minmaxIDC.first, minValue, maxValue);
    const float maxIDCDelta = std::clamp(*minmaxIDC.second, minValue, maxValue);
    idcCompressed.setCompressionFactor(minIDCDelta, maxIDCDelta);
    for (auto& idc : idcDeltaUncompressed.getIDCDelta()) {
      idcCompressed.emplace_back(idc);
    }
  }
};

enum class PadFlags : unsigned short {
  flagGoodPad = 1 << 0,      ///< flag for a good pad binary 0001
  flagDeadPad = 1 << 1,      ///< flag for a dead pad binary 0010
  flagUnknownPad = 1 << 2,   ///< flag for unknown status binary 0100
  flagSaturatedPad = 1 << 3, ///< flag for saturated status binary 0100
  flagHighPad = 1 << 4,      ///< flag for pad with extremly high IDC value
  flagLowPad = 1 << 5,       ///< flag for pad with extremly low IDC value
  flagSkip = 1 << 6          ///< flag for defining a pad which is just ignored during the calculation of I1 and IDCDelta
};

inline PadFlags operator&(PadFlags a, PadFlags b) { return static_cast<PadFlags>(static_cast<int>(a) & static_cast<int>(b)); }
inline PadFlags operator~(PadFlags a) { return static_cast<PadFlags>(~static_cast<int>(a)); }
inline PadFlags operator|(PadFlags a, PadFlags b) { return static_cast<PadFlags>(static_cast<int>(a) | static_cast<int>(b)); }

} // namespace tpc
} // namespace o2

#endif
