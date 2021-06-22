// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

/// struct containing the IDC delta values
template <typename DataT>
struct IDCDeltaContainer {
  std::array<std::vector<DataT>, o2::tpc::SIDES> mIDCDelta{}; ///< \Delta I(r,\phi,t) = I(r,\phi,t) / ( I_0(r,\phi) * I_1(t) )
};

/// storage for the factor used to compress IDCDelta.
/// This factor is separated from the IDCDelta struct to able to store those values independently in the CCDB
struct IDCDeltaCompressionFactors {
  std::array<float, o2::tpc::SIDES> mFactors{1.f, 1.f}; ///< compression factors for each TPC side
};

/// struct to access and set Delta IDCs
template <typename DataT>
struct IDCDelta {

  /// set idcDelta for given index
  /// \param idcDelta Delta IDC value which will be set
  /// \param side side of the TPC
  /// \param index index in the storage
  void setValue(const float idcDelta, const o2::tpc::Side side, const unsigned int index) { mIDCDelta.mIDCDelta[side][index] = compressValue(idcDelta, side); }

  /// set idcDelta ath the end of storage
  /// \param idcDelta Delta IDC value which will be set
  /// \param side side of the TPC
  void emplace_back(const float idcDelta, const o2::tpc::Side side) { mIDCDelta.mIDCDelta[side].emplace_back(compressValue(idcDelta, side)); }

  /// \return returns converted IDC value from float to new data type specified by template parameter of the object
  /// \param idcDelta Delta IDC value which will be set
  /// \param side side of the TPC
  DataT compressValue(const float idcDelta, const o2::tpc::Side side) const
  {
    const static auto& paramIDCGroup = ParameterIDCCompression::Instance();
    return (std::abs(idcDelta) >= paramIDCGroup.MaxIDCDeltaValue) ? static_cast<DataT>(std::copysign(paramIDCGroup.MaxIDCDeltaValue * mCompressionFactor.mFactors[side] + 0.5f, idcDelta)) : static_cast<DataT>(idcDelta * mCompressionFactor.mFactors[side] + std::copysign(0.5f, idcDelta));
  }

  /// \return returns stored Delta IDC value
  /// \param side side of the TPC
  /// \param index index in the storage (see: getIndexUngrouped int IDCGroupHelperSector)
  float getValue(const o2::tpc::Side side, const unsigned int index) const { return (static_cast<float>(mIDCDelta.mIDCDelta[side][index]) / mCompressionFactor.mFactors[side]); }

  /// set compression factor
  /// \param side side of the TPC
  /// \param factor factor which will be used for the compression
  void setCompressionFactor(const o2::tpc::Side side, const float factor) { mCompressionFactor.mFactors[side] = factor; }

  /// \return returns vector of Delta IDCs for given side
  /// \param side side of the TPC
  const auto& getIDCDelta(const o2::tpc::Side side) const { return mIDCDelta.mIDCDelta[side]; }

  /// \return returns vector of Delta IDCs for given side
  /// \param side side of the TPC
  auto& getIDCDelta(const o2::tpc::Side side) { return mIDCDelta.mIDCDelta[side]; }

  /// \return returns compression factors to uncompress Delta IDC
  const auto& getCompressionFactors() const { return mCompressionFactor; }

  /// \return returns compression factors to uncompress Delta IDC
  /// \param side side of the TPC
  auto getCompressionFactor(const o2::tpc::Side side) const { return mCompressionFactor.mFactors[side]; }

  IDCDeltaContainer<DataT> mIDCDelta{};            ///< storage for Delta IDCs
  IDCDeltaCompressionFactors mCompressionFactor{}; ///< compression factor for Delta IDCs
};

// template specialization for float as no compression factor is needed in this case
template <>
struct IDCDelta<float> {
  /// set idcDelta for given index
  /// \param idcDelta Delta IDC value which will be set
  /// \param side side of the TPC
  /// \param index index in the storage
  void setValue(const float idcDelta, const o2::tpc::Side side, const unsigned int index) { mIDCDelta.mIDCDelta[side][index] = idcDelta; }

  /// \return returns stored Delta IDC value
  /// \param side side of the TPC
  /// \param index index in the storage
  float getValue(const o2::tpc::Side side, const unsigned int index) const { return mIDCDelta.mIDCDelta[side][index]; }

  /// \return returns vector of Delta IDCs for given side
  /// \param side side of the TPC
  const auto& getIDCDelta(const o2::tpc::Side side) const { return mIDCDelta.mIDCDelta[side]; }

  /// \return returns vector of Delta IDCs for given side
  /// \param side side of the TPC
  auto& getIDCDelta(const o2::tpc::Side side) { return mIDCDelta.mIDCDelta[side]; }

  IDCDeltaContainer<float> mIDCDelta{}; ///< storage for uncompressed Delta IDCs
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
    return std::move(idcCompressed);
  }

 private:
  static void compress(const IDCDelta<float>& idcDeltaUncompressed, IDCDelta<DataT>& idcCompressed, const o2::tpc::Side side)
  {
    idcCompressed.getIDCDelta(side).reserve(idcDeltaUncompressed.getIDCDelta(side).size());
    const float factor = getCompressionFactor(idcDeltaUncompressed, side);
    idcCompressed.setCompressionFactor(side, factor);
    for (auto& idc : idcDeltaUncompressed.getIDCDelta(side)) {
      idcCompressed.emplace_back(idc, side);
    }
  }

  /// \return returns the factor which is used during the compression
  static float getCompressionFactor(const IDCDelta<float>& idcDeltaUncompressed, const o2::tpc::Side side)
  {
    const float maxAbsIDC = getMaxValue(idcDeltaUncompressed.getIDCDelta(side));
    const auto& paramIDCGroup = ParameterIDCCompression::Instance();
    const float maxIDC = paramIDCGroup.MaxIDCDeltaValue;
    return (maxAbsIDC > maxIDC || maxAbsIDC == 0) ? (std::numeric_limits<DataT>::max() / maxIDC) : (std::numeric_limits<DataT>::max() / maxAbsIDC);
  }

  /// \returns returns maximum abs value in vector
  static float getMaxValue(const std::vector<float>& idcs)
  {
    return std::abs(*std::max_element(idcs.begin(), idcs.end(), [](const int a, const int b) -> bool { return (std::abs(a) < std::abs(b)); }));
  };
};

///<struct containing the IDC1 values
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

  std::array<std::vector<float>, o2::tpc::SIDES> mIDCZero{}; ///< I_0(r,\phi) = <I(r,\phi,t)>_t
};

///<struct containing the IDC0
struct IDCOne {
  /// set IDC one for given index
  /// \param idcOne Delta IDC value which will be set
  /// \param side side of the TPC
  /// \param index index in the storage
  void setValueIDCOne(const float idcOne, const o2::tpc::Side side, const unsigned int index) { mIDCOne[side][index] = idcOne; }

  /// \return returns stored IDC one value
  /// \param side side of the TPC
  /// \param index index in the storage
  float getValueIDCOne(const o2::tpc::Side side, const unsigned int index) const { return mIDCOne[side][index]; }

  std::array<std::vector<float>, o2::tpc::SIDES> mIDCOne{}; ///< I_1(t) = <I(r,\phi,t) / I_0(r,\phi)>_{r,\phi}
};

/// struct containing 1D-IDCs
struct OneDIDC {

  /// default constructor
  OneDIDC() = default;

  /// constructor for initializing member with default value (this is used in the IDCFourierTransform class to perform calculation of the fourier coefficients for the first aggregation interval)
  /// \param nIDC number of IDCs which will be initialized
  OneDIDC(const unsigned int nIDC) : mOneDIDC{std::vector<float>(nIDC), std::vector<float>(nIDC)} {};

  /// \return returns total number of 1D-IDCs for given side
  /// \param side side of the TPC
  unsigned int getNIDCs(const o2::tpc::Side side) const { return mOneDIDC[side].size(); }

  std::array<std::vector<float>, o2::tpc::SIDES> mOneDIDC{}; ///< 1D-IDCs = <I(r,\phi,t)>_{r,\phi}
};

/// Helper class for aggregation of 1D-IDCs from different CRUs
class OneDIDCAggregator
{
 public:
  /// constructor
  /// nTimeFrames number of time frames which will be aggregated
  OneDIDCAggregator(const unsigned int nTimeFrames = 1) : mOneDIDCAgg(nTimeFrames), mWeight(nTimeFrames){};

  /// aggregate 1D-IDCs
  /// \param side side of the tpcCRUHeader
  /// \param idc vector containing the 1D-IDCs
  /// \param timeframe of the input 1D-IDC
  /// \param region region of the TPC
  void aggregate1DIDCs(const o2::tpc::Side side, const std::vector<float>& idc, const unsigned int timeframe, const unsigned int region)
  {
    if (mOneDIDCAgg[timeframe].mOneDIDC[side].empty()) {
      mOneDIDCAgg[timeframe].mOneDIDC[side] = idc;
      mWeight[timeframe][side] = Mapper::REGIONAREA[region];
    } else {
      std::transform(mOneDIDCAgg[timeframe].mOneDIDC[side].begin(), mOneDIDCAgg[timeframe].mOneDIDC[side].end(), idc.begin(), mOneDIDCAgg[timeframe].mOneDIDC[side].begin(), std::plus<>());
      mWeight[timeframe][side] += Mapper::REGIONAREA[region];
    }
  }

  /// \return returns struct containing aggregated 1D IDCs normalized to number of channels weighted with the relative length of each region
  OneDIDC getAggregated1DIDCs()
  {
    OneDIDC oneDIDCTmp;
    for (int iside = 0; iside < o2::tpc::SIDES; ++iside) {
      const o2::tpc::Side side = iside == 0 ? o2::tpc::Side::A : o2::tpc::Side::C;
      for (unsigned int i = 0; i < mOneDIDCAgg.size(); ++i) {
        std::transform(mOneDIDCAgg[i].mOneDIDC[side].begin(), mOneDIDCAgg[i].mOneDIDC[side].end(), mOneDIDCAgg[i].mOneDIDC[side].begin(), [norm = mWeight[i][side]](auto& val) { return val / norm; });
        oneDIDCTmp.mOneDIDC[side].insert(oneDIDCTmp.mOneDIDC[side].end(), mOneDIDCAgg[i].mOneDIDC[side].begin(), mOneDIDCAgg[i].mOneDIDC[side].end());
        mOneDIDCAgg[i].mOneDIDC[side].clear();
      }
    }
    return oneDIDCTmp;
  }

  auto get() { return mOneDIDCAgg; }

 private:
  std::vector<OneDIDC> mOneDIDCAgg{};                       ///< 1D-IDCs = <I(r,\phi,t)>_{r,\phi}
  std::vector<std::array<float, o2::tpc::SIDES>> mWeight{}; ///< integrated pad area of received data used for normalization
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

  std::array<std::vector<float>, o2::tpc::SIDES> mFourierCoefficients{}; ///< fourier coefficients. side -> coefficient real and complex parameters are stored alternating
  const unsigned int mCoeffPerTF{};                                      ///< number of real+imag coefficients per TF
};

} // namespace tpc
} // namespace o2

#endif
