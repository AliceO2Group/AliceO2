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

/// \file TPCScaler.h
/// \author Matthias Kleiner <mkleiner@ikf.uni-frankfurt.de>

#ifndef ALICEO2_TPC_TPCSCALER
#define ALICEO2_TPC_TPCSCALER

#include "DataFormatsTPC/Defs.h"
#include <vector>

class TTree;

namespace o2::tpc
{

/*
Class for storing the scalers which are used to calculate an estimate for the mean space-charge density for the last ion drift time
*/

struct TPCScalerWeights {
  float getWeight(float deltaTime) const;
  bool isValid() const { return (mSamplingTimeMS > 0 && !mWeights.empty()); }
  float getDurationMS() const { return mSamplingTimeMS * mWeights.size(); }
  float mSamplingTimeMS = -1;  ///< sampling of the stored weights
  float mFirstTimeStampMS = 0; ///< first timestamp
  std::vector<float> mWeights; ///< stored weights

  ClassDefNV(TPCScalerWeights, 1);
};

class TPCScaler
{
 public:
  /// default constructor
  TPCScaler() = default;

  /// default move assignment
  TPCScaler& operator=(TPCScaler&& other) = default;

  /// \return returns number of stored TPC scaler values
  int getNValues(o2::tpc::Side side) const { return (side == o2::tpc::Side::A ? mScalerA.size() : mScalerC.size()); }

  /// set the parameters for the coefficients of the polynomial
  /// \param params parameter for the coefficients
  void setScaler(const std::vector<float>& values, const o2::tpc::Side side) { (side == o2::tpc::Side::A ? (mScalerA = values) : (mScalerC = values)); }

  /// \return returns ion drift time in ms
  void setIonDriftTimeMS(float ionDriftTimeMS) { mIonDriftTimeMS = ionDriftTimeMS; }

  /// \return returns run number for which this object is valid
  void setRun(int run) { mRun = run; }

  /// \param firstTFOrbit first TF orbit of first data
  void setFirstTFOrbit(unsigned int firstTFOrbit) { mFirstTFOrbit = firstTFOrbit; }

  /// \param timeStampMS first time in ms
  void setStartTimeStampMS(double timeStampMS) { mTimeStampMS = timeStampMS; }

  /// \param integrationTimeMS integration time for each scaler value
  void setIntegrationTimeMS(float integrationTimeMS) { mIntegrationTimeMS = integrationTimeMS; }

  /// dump this object to a file
  /// \param file output file
  void dumpToFile(const char* file, const char* name);

  /// load parameters from input file (which were written using the writeToFile method)
  /// \param inpf input file
  void loadFromFile(const char* inpf, const char* name);

  /// set this object from input tree
  void setFromTree(TTree& tpcScalerTree);

  /// \return returns stored scalers for given side and data index
  float getScalers(unsigned int idx, o2::tpc::Side side) const { return (side == o2::tpc::Side::A) ? mScalerA[idx] : mScalerC[idx]; }

  /// \return returns stored scalers for given side
  const auto& getScalers(o2::tpc::Side side) const { return (side == o2::tpc::Side::A) ? mScalerA : mScalerC; }

  /// \return returns ion drift time in ms
  float getIonDriftTimeMS() const { return mIonDriftTimeMS; }

  /// \return returns run number for which this object is valid
  int getRun() const { return mRun; }

  /// \return returns first TF orbit of first data
  unsigned int getFirstTFOrbit() const { return mFirstTFOrbit; }

  /// \return return first time in ms
  double getStartTimeStampMS() const { return mTimeStampMS; }

  /// \return return end time in ms
  double getEndTimeStampMS(o2::tpc::Side side) const { return mTimeStampMS + getDurationMS(side); }

  /// \return returns integration time for each scaler value
  float getIntegrationTimeMS() const { return mIntegrationTimeMS; }

  /// \return returns numbers of scalers for one ion drift time
  int getNValuesIonDriftTime() const { return mIonDriftTimeMS / mIntegrationTimeMS + 0.5; }

  /// \return returns mean scaler value for last ion drift time
  /// \param timestamp timestamp for which the last values are used to calculate the mean
  float getMeanScaler(double timestamp, o2::tpc::Side side) const;

  /// \return returns duration in ms for which the scalers are defined
  float getDurationMS(o2::tpc::Side side) const { return mIntegrationTimeMS * getNValues(side); }

  /// setting the weights for the scalers
  void setScalerWeights(const TPCScalerWeights& weights) { mScalerWeights = weights; }

  /// \return returns stored weights for TPC scalers
  const auto& getScalerWeights() const { return mScalerWeights; }

  /// enable usage of weights
  void useWeights(bool useWeights) { mUseWeights = useWeights; }

  /// return if weights are used
  bool weightsUsed() const { return mUseWeights; }

 private:
  float mIonDriftTimeMS{200};        ///< ion drift time in ms
  int mRun{};                        ///< run for which this object is valid
  unsigned int mFirstTFOrbit{};      ///< first TF orbit of the stored scalers
  double mTimeStampMS{};             ///< time stamp of the first stored values
  float mIntegrationTimeMS{1.};      ///< integration time for each stored value in ms
  std::vector<float> mScalerA{};     ///< TPC scaler for A-side
  std::vector<float> mScalerC{};     ///< TPC scaler for C-side
  TPCScalerWeights mScalerWeights{}; ///< weights for the scalers for A-side
  bool mUseWeights{false};           ///< use weights when calculating the mean scaler

  // get index to data for given timestamp
  int getDataIdx(double timestamp) const { return (timestamp - mTimeStampMS) / mIntegrationTimeMS + 0.5; }

  ClassDefNV(TPCScaler, 2);
};

} // namespace o2::tpc
#endif
