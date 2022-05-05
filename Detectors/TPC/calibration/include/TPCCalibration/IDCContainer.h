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

/// IDC Delta IDC Compression types
enum class IDCDeltaCompression { NO = 0,     ///< no compression using floats
                                 MEDIUM = 1, ///< medium compression using short (data compression ratio 2 when stored in CCDB)
                                 HIGH = 2    ///< high compression using char (data compression ratio ~5.5 when stored in CCDB)
};

/// struct containing the IDC delta values
template <typename DataT>
struct IDCDeltaContainer {
  std::array<std::vector<DataT>, o2::tpc::SIDES> mIDCDeltaCont{}; ///< \Delta I(r,\phi,t) = I(r,\phi,t) / ( I_0(r,\phi) * I_1(t) )
};

/// storage for the factor used to compress IDCDelta.
/// This factor is separated from the IDCDelta struct to able to store those values independently in the CCDB
struct IDCDeltaCompressionFactors {
  std::array<std::pair<float, float>, o2::tpc::SIDES> mFactors{std::pair{1.f, 1.f}, std::pair{1.f, 1.f}}; ///< compression factors for each TPC side
};

/// struct to access and set Delta IDCs
template <typename DataT>
struct IDCDelta {

  /// set idcDelta for given index
  /// \param idcDelta Delta IDC value which will be set
  /// \param side side of the TPC
  /// \param index index in the storage
  void setValue(const float idcDelta, const o2::tpc::Side side, const unsigned int index) { mIDCDelta.mIDCDeltaCont[side][index] = compressValue(idcDelta, side); }

  /// set idcDelta ath the end of storage
  /// \param idcDelta Delta IDC value which will be set
  /// \param side side of the TPC
  void emplace_back(const float idcDelta, const o2::tpc::Side side) { mIDCDelta.mIDCDeltaCont[side].emplace_back(compressValue(idcDelta, side)); }

  /// \return returns converted IDC value from float to new data type specified by template parameter of the object
  /// \param idcDelta Delta IDC value which will be set
  /// \param side side of the TPC
  DataT compressValue(float idcDelta, const o2::tpc::Side side) const
  {
    idcDelta = (std::clamp(idcDelta, mCompressionFactor.mFactors[side].first, mCompressionFactor.mFactors[side].second) - mCompressionFactor.mFactors[side].first) * std::numeric_limits<DataT>::max() / (mCompressionFactor.mFactors[side].second - mCompressionFactor.mFactors[side].first);
    return std::nearbyint(idcDelta);
  }

  /// \return returns stored Delta IDC value
  /// \param side side of the TPC
  /// \param index index in the storage (see: getIndexUngrouped int IDCGroupHelperSector)
  float getValue(const o2::tpc::Side side, const unsigned int index) const { return mCompressionFactor.mFactors[side].first + (mCompressionFactor.mFactors[side].second - mCompressionFactor.mFactors[side].first) * static_cast<float>(mIDCDelta.mIDCDeltaCont[side][index]) / std::numeric_limits<DataT>::max(); }

  /// set compression factor
  /// \param side side of the TPC
  /// \param factor factor which will be used for the compression
  void setCompressionFactor(const o2::tpc::Side side, const float factorMin, const float factorMax) { mCompressionFactor.mFactors[side] = std::pair{factorMin, factorMax}; }

  /// \return returns vector of Delta IDCs for given side
  /// \param side side of the TPC
  const auto& getIDCDelta(const o2::tpc::Side side) const { return mIDCDelta.mIDCDeltaCont[side]; }

  /// \return returns vector of Delta IDCs for given side
  /// \param side side of the TPC
  auto& getIDCDelta(const o2::tpc::Side side) { return mIDCDelta.mIDCDeltaCont[side]; }

  /// set IDCDelta value
  /// \param side side of the TPC
  /// \param index index in the storage (see: getIndexUngrouped int IDCGroupHelperSector)
  /// \param val value of IDCDelta which will be stored
  void setIDCDelta(const o2::tpc::Side side, const unsigned int index, const DataT val) { mIDCDelta.mIDCDeltaCont[side][index] = val; }

  /// \return returns compression factors to uncompress Delta IDC
  const auto& getCompressionFactors() const { return mCompressionFactor; }

  /// \return returns compression factors to uncompress Delta IDC
  /// \param side side of the TPC
  auto getCompressionFactor(const o2::tpc::Side side) const { return mCompressionFactor.mFactors[side]; }

  /// \return returns number of stored IDCs for given side
  /// \param side side of the TPC
  auto getNIDCs(const o2::tpc::Side side) const { return getIDCDelta(side).size(); }

  IDCDeltaContainer<DataT> mIDCDelta{};            ///< storage for Delta IDCs
  IDCDeltaCompressionFactors mCompressionFactor{}; ///< compression factor for Delta IDCs
  ClassDefNV(IDCDelta, 1)
};

// template specialization for float as no compression factor is needed in this case
template <>
struct IDCDelta<float> {
  /// set idcDelta for given index
  /// \param idcDelta Delta IDC value which will be set
  /// \param side side of the TPC
  /// \param index index in the storage
  void setValue(const float idcDelta, const o2::tpc::Side side, const unsigned int index) { mIDCDelta.mIDCDeltaCont[side][index] = idcDelta; }

  /// \return returns stored Delta IDC value
  /// \param side side of the TPC
  /// \param index index in the storage
  float getValue(const o2::tpc::Side side, const unsigned int index) const { return mIDCDelta.mIDCDeltaCont[side][index]; }

  /// \return returns vector of Delta IDCs for given side
  /// \param side side of the TPC
  const auto& getIDCDelta(const o2::tpc::Side side) const { return mIDCDelta.mIDCDeltaCont[side]; }

  /// \return returns vector of Delta IDCs for given side
  /// \param side side of the TPC
  auto& getIDCDelta(const o2::tpc::Side side) { return mIDCDelta.mIDCDeltaCont[side]; }

  /// \return returns vector of Delta IDCs for given side
  /// \param side side of the TPC
  auto getIDCDeltaContainer() && { return std::move(mIDCDelta); }

  /// set IDCDelta value
  /// \param side side of the TPC
  /// \param index index in the storage (see: getIndexUngrouped int IDCGroupHelperSector)
  /// \param val value of IDCDelta which will be stored
  void setIDCDelta(const o2::tpc::Side side, const unsigned int index, const float val) { mIDCDelta.mIDCDeltaCont[side][index] = val; }

  /// resize the container
  /// \param size new size of the container
  void resize(const o2::tpc::Side side, const unsigned int size) { mIDCDelta.mIDCDeltaCont[side].resize(size); }

  /// \return returns number of stored IDCs for given side
  /// \param side side of the TPC
  auto getNIDCs(const o2::tpc::Side side) const { return getIDCDelta(side).size(); }

  IDCDeltaContainer<float> mIDCDelta{}; ///< storage for uncompressed Delta IDCs
  ClassDefNV(IDCDelta, 1)
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
    IDCDelta<DataT> idcCompressed{};
    compress(idcDeltaUncompressed, idcCompressed, o2::tpc::Side::A);
    compress(idcDeltaUncompressed, idcCompressed, o2::tpc::Side::C);
    return idcCompressed;
  }

 private:
  static void compress(const IDCDelta<float>& idcDeltaUncompressed, IDCDelta<DataT>& idcCompressed, const o2::tpc::Side side)
  {
    idcCompressed.getIDCDelta(side).reserve(idcDeltaUncompressed.getIDCDelta(side).size());
    const auto minmaxIDC = std::minmax_element(std::begin(idcDeltaUncompressed.getIDCDelta(side)), std::end(idcDeltaUncompressed.getIDCDelta(side)));
    const auto& paramIDCGroup = ParameterIDCCompression::Instance();
    const float minIDCDelta = std::clamp(*minmaxIDC.first, paramIDCGroup.minIDCDeltaValue, paramIDCGroup.maxIDCDeltaValue);
    const float maxIDCDelta = std::clamp(*minmaxIDC.second, paramIDCGroup.minIDCDeltaValue, paramIDCGroup.maxIDCDeltaValue);
    idcCompressed.setCompressionFactor(side, minIDCDelta, maxIDCDelta);
    for (auto& idc : idcDeltaUncompressed.getIDCDelta(side)) {
      idcCompressed.emplace_back(idc, side);
    }
  }
};

///< struct containing the IDC0 values
struct IDCZero {

  /// set IDC zero for given index
  /// \param idcZero Delta IDC value which will be set
  /// \param side side of the TPC
  /// \param index index in the storage
  void setValueIDCZero(const float idcZero, const o2::tpc::Side side, const unsigned int index) { mIDCZero[side][index] = idcZero; }

  /// increase IDC zero for given index
  /// \param idcZero Delta IDC value which will be set
  /// \param side side of the TPC
  /// \param index index in the storage
  void fillValueIDCZero(const float idcZero, const o2::tpc::Side side, const unsigned int index) { mIDCZero[side][index] += idcZero; }

  /// \return returns stored IDC zero value
  /// \param side side of the TPC
  /// \param index index in the storage
  float getValueIDCZero(const o2::tpc::Side side, const unsigned int index) const { return mIDCZero[side][index]; }

  /// clear values
  void clear()
  {
    clear(Side::A);
    clear(Side::C);
  }

  /// \param side side of the TPC
  void clear(const o2::tpc::Side side) { mIDCZero[side].clear(); }

  /// resize vector
  void resize(const unsigned int size)
  {
    resize(Side::A, size);
    resize(Side::C, size);
  }

  /// returns false if both sides containes values. returns true if one side is empty
  bool empty() const { return mIDCZero[Side::A].empty() + mIDCZero[Side::C].empty(); }

  /// resize vector
  /// \param side side of the TPC
  void resize(const o2::tpc::Side side, const unsigned int size) { mIDCZero[side].resize(size); }

  /// get number of IDC0 values
  /// \param side side of the TPC
  auto getNIDC0(const o2::tpc::Side side) const { return mIDCZero[side].size(); }

  std::array<std::vector<float>, o2::tpc::SIDES> mIDCZero{}; ///< I_0(r,\phi) = <I(r,\phi,t)>_t
  ClassDefNV(IDCZero, 1)
};

///<struct containing the IDC1
struct IDCOne {

  /// default constructor
  IDCOne() = default;

  /// constructor for initializing member with default value (this is used in the IDCFourierTransform class to perform calculation of the fourier coefficients for the first aggregation interval)
  /// \param nIDC number of IDCs which will be initialized
  IDCOne(const unsigned int nIDC) : mIDCOne{std::vector<float>(nIDC), std::vector<float>(nIDC)} {};

  /// set IDC one for given index
  /// \param idcOne Delta IDC value which will be set
  /// \param side side of the TPC
  /// \param index index in the storage
  void setValueIDCOne(const float idcOne, const o2::tpc::Side side, const unsigned int index) { mIDCOne[side][index] = idcOne; }

  /// \return returns stored IDC one value
  /// \param side side of the TPC
  /// \param index index in the storage
  float getValueIDCOne(const o2::tpc::Side side, const unsigned int index) const { return mIDCOne[side][index]; }

  /// \return returns number of stored IDCs for given side
  /// \param side side of the TPC
  auto getNIDCs(const o2::tpc::Side side) const { return mIDCOne[side].size(); }

  /// clear values
  void clear()
  {
    clear(Side::A);
    clear(Side::C);
  }

  /// \param side side of the TPC
  void clear(const o2::tpc::Side side) { mIDCOne[side].clear(); }

  /// resize vector
  void resize(const unsigned int size)
  {
    resize(Side::A, size);
    resize(Side::C, size);
  }

  /// resize vector
  /// \param side side of the TPC
  void resize(const o2::tpc::Side side, const unsigned int size) { mIDCOne[side].resize(size); }

  std::array<std::vector<float>, o2::tpc::SIDES> mIDCOne{}; ///< I_1(t) = <I(r,\phi,t) / I_0(r,\phi)>_{r,\phi}
  ClassDefNV(IDCOne, 1)
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
    if (mIDCOneAgg.mIDCOne[side].empty()) {
      mIDCOneAgg.mIDCOne[side] = idc;
    } else {
      std::transform(mIDCOneAgg.mIDCOne[side].begin(), mIDCOneAgg.mIDCOne[side].end(), idc.begin(), mIDCOneAgg.mIDCOne[side].begin(), std::plus<>());
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
    std::transform(mIDCOneAgg.mIDCOne[side].begin(), mIDCOneAgg.mIDCOne[side].end(), mWeight[side].begin(), mIDCOneAgg.mIDCOne[side].begin(), std::divides<>());
  }

  /// \return returns IDC1 data by move and clears weights
  auto get() &&
  {
    mWeight[Side::A].clear();
    mWeight[Side::C].clear();
    return std::move(mIDCOneAgg);
  }

  /// \return returns weights for the stored values
  const auto& getWeight() const { return mWeight; }

 private:
  IDCOne mIDCOneAgg{};                                             ///< 1D-IDCs = <I(r,\phi,t)>_{r,\phi}
  std::array<std::vector<unsigned int>, o2::tpc::SIDES> mWeight{}; ///< Number of channels used for IDC1 calculation used for normalization
};

/// struct containing the fourier coefficients calculated from IDC0 for n timeframes
struct FourierCoeff {
  /// constructor
  /// \param nTimeFrames number of time frames
  /// \param nCoeff number of real/imag fourier coefficients which will be stored
  FourierCoeff(const unsigned int nTimeFrames, const unsigned int nCoeff)
    : mFourierCoefficients{{std::vector<float>(nTimeFrames * nCoeff), std::vector<float>(nTimeFrames * nCoeff)}}, mCoeffPerTF{nCoeff} {};

  /// default constructor for ROOT I/O
  FourierCoeff() = default;

  /// \return returns total number of stored coefficients for given side and real/complex type
  /// \param side side
  unsigned long getNCoefficients(const o2::tpc::Side side) const { return mFourierCoefficients[side].size(); }

  /// \return returns number of stored coefficients for TF
  unsigned int getNCoefficientsPerTF() const { return mCoeffPerTF; }

  /// \return returns all stored coefficients for given side and real/complex type
  /// \param side side
  const auto& getFourierCoefficients(const o2::tpc::Side side) const { return mFourierCoefficients[side]; }

  /// \return returns index to fourier coefficient
  /// \param interval index of interval
  /// \param coefficient index of coefficient
  unsigned int getIndex(const unsigned int interval, const unsigned int coefficient) const { return interval * mCoeffPerTF + coefficient; }

  /// \return returns the stored value
  /// \param side side of the TPC
  /// \param index index of the data
  float operator()(const o2::tpc::Side side, unsigned int index) const { return mFourierCoefficients[side][index]; }

  /// \return returns the stored value
  /// \param side side of the TPC
  /// \param index index of the data
  float& operator()(const o2::tpc::Side side, unsigned int index) { return mFourierCoefficients[side][index]; }

  void reset(const o2::tpc::Side side) { std::fill(mFourierCoefficients[side].begin(), mFourierCoefficients[side].end(), 0.f); };

  std::array<std::vector<float>, o2::tpc::SIDES> mFourierCoefficients{}; ///< fourier coefficients. side -> coefficient real and complex parameters are stored alternating
  const unsigned int mCoeffPerTF{};                                      ///< number of real+imag coefficients per TF

  ClassDefNV(FourierCoeff, 1)
};

template <typename T>
struct Enable_enum_class_bitfield {
  static constexpr bool value = false;
};

// operator overload for allowing bitfiedls with enum
template <typename T>
typename std::enable_if<std::is_enum<T>::value && Enable_enum_class_bitfield<T>::value, T>::type
  operator&(T lhs, T rhs)
{
  typedef typename std::underlying_type<T>::type integer_type;
  return static_cast<T>(static_cast<integer_type>(lhs) & static_cast<integer_type>(rhs));
}

template <typename T>
typename std::enable_if<std::is_enum<T>::value && Enable_enum_class_bitfield<T>::value, T>::type
  operator|(T lhs, T rhs)
{
  typedef typename std::underlying_type<T>::type integer_type;
  return static_cast<T>(static_cast<integer_type>(lhs) | static_cast<integer_type>(rhs));
}

enum class PadFlags : unsigned short {
  flagGoodPad = 1 << 0,      ///< flag for a good pad binary 0001
  flagDeadPad = 1 << 1,      ///< flag for a dead pad binary 0010
  flagUnknownPad = 1 << 2,   ///< flag for unknown status binary 0100
  flagSaturatedPad = 1 << 3, ///< flag for saturated status binary 0100
  flagHighPad = 1 << 4,      ///< flag for pad with extremly high IDC value
  flagLowPad = 1 << 5,       ///< flag for pad with extremly low IDC value
  flagSkip = 1 << 6          ///< flag for defining a pad which is just ignored during the calculation of I1 and IDCDelta
};

template <>
struct Enable_enum_class_bitfield<PadFlags> {
  static constexpr bool value = true;
};

} // namespace tpc
} // namespace o2

#endif
