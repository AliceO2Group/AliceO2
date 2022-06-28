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

/// \file IDCFourierTransform.h
/// \brief base class for holding members and functions for separating between EPN and Agregator
/// \author Matthias Kleiner <mkleiner@ikf.uni-frankfurt.de>
/// \date Jun 10, 2021

#ifndef ALICEO2_IDCFOURIERTRANSFORMBASE_H_
#define ALICEO2_IDCFOURIERTRANSFORMBASE_H_

#include <vector>
#include "Rtypes.h"
#include "DataFormatsTPC/Defs.h"
#include "TPCCalibration/IDCContainer.h"

namespace o2::tpc
{

template <class Type>
class IDCFourierTransformBase;

// do not use enum class as type to avoid problems with ROOT dictionary generation!
class IDCFourierTransformBaseEPN;        /// dummy class for templating IDCFourierTransformBase class
class IDCFourierTransformBaseAggregator; /// dummy class for templating IDCFourierTransformBase class
using IDCFourierTransformEPN = o2::tpc::IDCFourierTransformBase<o2::tpc::IDCFourierTransformBaseEPN>;
using IDCFourierTransformAggregator = o2::tpc::IDCFourierTransformBase<o2::tpc::IDCFourierTransformBaseAggregator>;

template <>
class IDCFourierTransformBase<IDCFourierTransformBaseEPN>
{
 public:
  /// constructor
  /// \param rangeIDC number of IDCs for each interval which will be used to calculate the fourier coefficients
  IDCFourierTransformBase(const unsigned int rangeIDC) : mRangeIDC{rangeIDC} {};

  /// set input 1D-IDCs which are used to calculate fourier coefficients
  /// \param oneDIDCs 1D-IDCs
  void setIDCs(IDCOne&& oneDIDCs) { mIDCOne = std::move(oneDIDCs); }

  /// set input 1D-IDCs which are used to calculate fourier coefficients
  /// \param oneDIDCs 1D-IDCs
  void setIDCs(const IDCOne& oneDIDCs) { mIDCOne = oneDIDCs; }

  /// \return returns number of time frames (only 1!) for which the coefficients are obtained
  static constexpr unsigned int getNIntervals() { return 1; }

  /// \return returns indices used for accessing correct IDCs for given TF
  std::vector<unsigned int> getLastIntervals(const o2::tpc::Side side) const { return std::vector<unsigned int>{static_cast<unsigned int>(mIDCOne.getNIDCs(side)) - mRangeIDC}; }

  /// copy over IDCs from buffer to current IDCOne vector for easier access
  /// \return returns expanded 1D-IDC vector
  /// \param side TPC side
  std::vector<float> getExpandedIDCOne(const o2::tpc::Side side) const { return mIDCOne.mIDCOne[side]; }

  /// \return returns struct of stored 1D-IDC
  const IDCOne& getIDCOne() const { return mIDCOne; }

  /// \return returns number of 1D-IDCs
  /// \param side TPC side
  unsigned long getNIDCs(const o2::tpc::Side side) const { return mIDCOne.mIDCOne[side].size(); }

 protected:
  const unsigned int mRangeIDC{}; ///< number of IDCs used for the calculation of fourier coefficients
  IDCOne mIDCOne{};               ///< all 1D-IDCs which are used to calculate the fourier coefficients.
  ClassDefNV(IDCFourierTransformBase, 1)
};

template <>
class IDCFourierTransformBase<IDCFourierTransformBaseAggregator>
{
 public:
  /// \param rangeIDC number of IDCs for each interval which will be used to calculate the fourier coefficients
  /// \param timeFrames number of time frames which will be stored
  IDCFourierTransformBase(const unsigned int rangeIDC, const unsigned int timeFrames) : mRangeIDC{rangeIDC}, mTimeFrames{timeFrames} {};

  /// set input 1D-IDCs which are used to calculate fourier coefficients
  /// \param oneDIDCs 1D-IDCs
  /// \param integrationIntervalsPerTF vector containg for each TF the number of IDCs
  void setIDCs(IDCOne&& oneDIDCs, std::vector<unsigned int>&& integrationIntervalsPerTF);

  /// set input 1D-IDCs which are used to calculate fourier coefficients
  /// \param oneDIDCs 1D-IDCs
  /// \param integrationIntervalsPerTF vector containg for each TF the number of IDCs
  void setIDCs(const IDCOne& oneDIDCs, const std::vector<unsigned int>& integrationIntervalsPerTF);

  /// \return returns number of 1D-IDCs
  /// \param side TPC side
  unsigned long getNIDCs(const o2::tpc::Side side) const { return mIDCOne[!mBufferIndex].mIDCOne[side].size(); }

  /// \return returns number of time frames for which the coefficients are obtained
  unsigned int getNIntervals() const { return mTimeFrames; }

  /// \return returns struct of stored 1D-IDC
  const IDCOne& getIDCOne() const { return mIDCOne[!mBufferIndex]; }

  /// \return returns indices used for accessing correct IDCs for given TF
  std::vector<unsigned int> getLastIntervals(o2::tpc::Side) const;

  /// copy over IDCs from buffer to current IDCOne vector for easier access
  /// \return returns expanded 1D-IDC vector
  /// \param side TPC side
  std::vector<float> getExpandedIDCOne(const o2::tpc::Side side) const;

  /// allocate memory for variable holding getrangeIDC() IDCs
  /// \param side TPC side
  float* allocMemFFTW(const o2::tpc::Side side) const;

 protected:
  const unsigned int mRangeIDC{};                                        ///< number of IDCs used for the calculation of fourier coefficients
  const unsigned int mTimeFrames{};                                      ///< number of timeframes which for which teh fourier coefficients are stored
  std::array<IDCOne, 2> mIDCOne{IDCOne(mRangeIDC), IDCOne(mRangeIDC)};   ///< all 1D-IDCs which are used to calculate the fourier coefficients. A buffer for the last aggregation interval is used to calculate the fourier coefficients for the first TFs
  std::array<std::vector<unsigned int>, 2> mIntegrationIntervalsPerTF{}; ///< number of integration intervals per TF used to set the correct range of IDCs. A buffer is needed for the last aggregation interval.
  bool mBufferIndex{true};                                               ///< index for the buffer

  /// returns whether the buffer has to be used
  bool useLastBuffer() const { return (mRangeIDC > mIntegrationIntervalsPerTF[!mBufferIndex][0]); }

  ClassDefNV(IDCFourierTransformBase, 1)
};

} // namespace o2::tpc

#endif
