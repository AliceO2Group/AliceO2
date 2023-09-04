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

/// \file IntegratedClusterCalibrator.h
/// \brief calibrator class for accumulating integrated clusters
/// \author Matthias Kleiner <mkleiner@ikf.uni-frankfurt.de>
/// \date Jan 21, 2023

#ifndef INTEGRATEDCLUSTERCALIBRATOR_H_
#define INTEGRATEDCLUSTERCALIBRATOR_H_

#include "DetectorsCalibration/TimeSlotCalibration.h"
#include "DetectorsCalibration/TimeSlot.h"

namespace o2
{

// see https://stackoverflow.com/questions/56483053/if-condition-checking-multiple-vector-sizes-are-equal
// comparing if all vector have the same size
template <typename T0, typename... Ts>
bool sameSize(T0 const& first, Ts const&... rest)
{
  return ((first.size() == rest.size()) && ...);
}

namespace tof
{

/// struct containing the integrated TOF currents
struct ITOFC {
  std::vector<float> mITOFCNCl;                                     ///< integrated 1D TOF cluster currents
  std::vector<float> mITOFCQ;                                       ///< integrated 1D TOF qTot currents
  bool areSameSize() const { return sameSize(mITOFCNCl, mITOFCQ); } ///< check if stored currents have same number of entries
  bool isEmpty() const { return mITOFCNCl.empty(); }                ///< check if values are empty
  size_t getEntries() const { return mITOFCNCl.size(); }            ///< \return returns number of values stored

  /// acummulate integrated currents at given index
  /// \param posIndex index where data will be copied to
  /// \param data integrated currents which will be copied
  void fill(const unsigned int posIndex, const ITOFC& data)
  {
    std::copy(data.mITOFCNCl.begin(), data.mITOFCNCl.end(), mITOFCNCl.begin() + posIndex);
    std::copy(data.mITOFCQ.begin(), data.mITOFCQ.end(), mITOFCQ.begin() + posIndex);
  }

  /// \param nDummyValues number of empty values which are inserted at the beginning of the accumulated integrated currents
  void insert(const unsigned int nDummyValues)
  {
    std::vector<float> vecTmp(nDummyValues, 0);
    mITOFCNCl.insert(mITOFCNCl.begin(), vecTmp.begin(), vecTmp.end());
    mITOFCQ.insert(mITOFCQ.begin(), vecTmp.begin(), vecTmp.end());
  }

  /// resize buffer for accumulated currents
  void resize(const unsigned int nTotal)
  {
    mITOFCNCl.resize(nTotal);
    mITOFCQ.resize(nTotal);
  }

  ClassDefNV(ITOFC, 1);
};
} // end namespace tof

namespace tpc
{

/// struct containing the integrated TPC currents
struct ITPCC {
  std::vector<float> mIQMaxA; ///< integrated 1D-currents for QMax A-side
  std::vector<float> mIQMaxC; ///< integrated 1D-currents for QMax C-side
  std::vector<float> mIQTotA; ///< integrated 1D-currents for QTot A-side
  std::vector<float> mIQTotC; ///< integrated 1D-currents for QTot A-side
  std::vector<float> mINClA;  ///< integrated 1D-currents for NCl A-side
  std::vector<float> mINClC;  ///< integrated 1D-currents for NCl A-side

  float compression(float value, const int nBits) const
  {
    const int shiftN = std::pow(2, nBits);
    int exp2;
    const auto mantissa = std::frexp(value, &exp2);
    const auto mantissaRounded = std::round(mantissa * shiftN) / shiftN;
    return std::ldexp(mantissaRounded, exp2);
  }

  void compress(const int nBits)
  {
    std::transform(mIQMaxA.begin(), mIQMaxA.end(), mIQMaxA.begin(), [this, nBits](float val) { return compression(val, nBits); });
    std::transform(mIQMaxC.begin(), mIQMaxC.end(), mIQMaxC.begin(), [this, nBits](float val) { return compression(val, nBits); });
    std::transform(mIQTotA.begin(), mIQTotA.end(), mIQTotA.begin(), [this, nBits](float val) { return compression(val, nBits); });
    std::transform(mIQTotC.begin(), mIQTotC.end(), mIQTotC.begin(), [this, nBits](float val) { return compression(val, nBits); });
    std::transform(mINClA.begin(), mINClA.end(), mINClA.begin(), [this, nBits](float val) { return compression(val, nBits); });
    std::transform(mINClC.begin(), mINClC.end(), mINClC.begin(), [this, nBits](float val) { return compression(val, nBits); });
  }

  bool areSameSize() const { return sameSize(mIQMaxA, mIQMaxC, mIQTotA, mIQTotC, mINClA, mINClC); } ///< check if stored currents have same number of entries
  bool isEmpty() const { return mIQMaxA.empty(); }                                                  ///< check if values are empty
  size_t getEntries() const { return mIQMaxA.size(); }                                              ///< \return returns number of values stored

  /// acummulate integrated currents at given index
  /// \param posIndex index where data will be copied to
  /// \param data integrated currents which will be copied
  void fill(const unsigned int posIndex, const ITPCC& data)
  {
    std::copy(data.mIQMaxA.begin(), data.mIQMaxA.end(), mIQMaxA.begin() + posIndex);
    std::copy(data.mIQMaxC.begin(), data.mIQMaxC.end(), mIQMaxC.begin() + posIndex);
    std::copy(data.mIQTotA.begin(), data.mIQTotA.end(), mIQTotA.begin() + posIndex);
    std::copy(data.mIQTotC.begin(), data.mIQTotC.end(), mIQTotC.begin() + posIndex);
    std::copy(data.mINClA.begin(), data.mINClA.end(), mINClA.begin() + posIndex);
    std::copy(data.mINClC.begin(), data.mINClC.end(), mINClC.begin() + posIndex);
  }

  /// \param nDummyValues number of empty values which are inserted at the beginning of the accumulated integrated currents
  void insert(const unsigned int nDummyValues)
  {
    std::vector<float> vecTmp(nDummyValues, 0);
    mIQMaxA.insert(mIQMaxA.begin(), vecTmp.begin(), vecTmp.end());
    mIQMaxC.insert(mIQMaxC.begin(), vecTmp.begin(), vecTmp.end());
    mIQTotA.insert(mIQTotA.begin(), vecTmp.begin(), vecTmp.end());
    mIQTotC.insert(mIQTotC.begin(), vecTmp.begin(), vecTmp.end());
    mINClA.insert(mINClA.begin(), vecTmp.begin(), vecTmp.end());
    mINClC.insert(mINClC.begin(), vecTmp.begin(), vecTmp.end());
  }

  /// resize buffer for accumulated currents
  void resize(const unsigned int nTotal)
  {
    mIQMaxA.resize(nTotal);
    mIQMaxC.resize(nTotal);
    mIQTotA.resize(nTotal);
    mIQTotC.resize(nTotal);
    mINClA.resize(nTotal);
    mINClC.resize(nTotal);
  }

  /// reset buffered currents
  void reset()
  {
    std::fill(mIQMaxA.begin(), mIQMaxA.end(), 0);
    std::fill(mIQMaxC.begin(), mIQMaxC.end(), 0);
    std::fill(mIQTotA.begin(), mIQTotA.end(), 0);
    std::fill(mIQTotC.begin(), mIQTotC.end(), 0);
    std::fill(mINClA.begin(), mINClA.end(), 0);
    std::fill(mINClC.begin(), mINClC.end(), 0);
  }

  /// normalize currents
  void normalize(const float factor)
  {
    std::transform(mIQMaxA.begin(), mIQMaxA.end(), mIQMaxA.begin(), [factor](const float val) { return val * factor; });
    std::transform(mIQMaxC.begin(), mIQMaxC.end(), mIQMaxC.begin(), [factor](const float val) { return val * factor; });
    std::transform(mIQTotA.begin(), mIQTotA.end(), mIQTotA.begin(), [factor](const float val) { return val * factor; });
    std::transform(mIQTotC.begin(), mIQTotC.end(), mIQTotC.begin(), [factor](const float val) { return val * factor; });
    std::transform(mINClA.begin(), mINClA.end(), mINClA.begin(), [factor](const float val) { return val * factor; });
    std::transform(mINClC.begin(), mINClC.end(), mINClC.begin(), [factor](const float val) { return val * factor; });
  }

  ClassDefNV(ITPCC, 1);
};

} // end namespace tpc

namespace fit
{

/// struct containing the integrated FT0 currents
struct IFT0C {
  std::vector<float> mINChanA;                                                        ///< integrated 1D FIT currents for NChan A
  std::vector<float> mINChanC;                                                        ///< integrated 1D FIT currents for NChan C
  std::vector<float> mIAmplA;                                                         ///< integrated 1D FIT currents for Ampl A
  std::vector<float> mIAmplC;                                                         ///< integrated 1D FIT currents for Ampl C
  bool areSameSize() const { return sameSize(mINChanA, mINChanC, mIAmplA, mIAmplC); } ///< check if stored currents have same number of entries
  bool isEmpty() const { return mINChanA.empty(); }                                   ///< check if values are empty
  size_t getEntries() const { return mINChanA.size(); }                               ///< \return returns number of values stored

  /// acummulate integrated currents at given index
  /// \param posIndex index where data will be copied to
  /// \param data integrated currents which will be copied
  void fill(const unsigned int posIndex, const IFT0C& data)
  {
    std::copy(data.mINChanA.begin(), data.mINChanA.end(), mINChanA.begin() + posIndex);
    std::copy(data.mINChanC.begin(), data.mINChanC.end(), mINChanC.begin() + posIndex);
    std::copy(data.mIAmplA.begin(), data.mIAmplA.end(), mIAmplA.begin() + posIndex);
    std::copy(data.mIAmplC.begin(), data.mIAmplC.end(), mIAmplC.begin() + posIndex);
  }

  /// \param nDummyValues number of empty values which are inserted at the beginning of the accumulated integrated currents
  void insert(const unsigned int nDummyValues)
  {
    std::vector<float> vecTmp(nDummyValues, 0);
    mINChanA.insert(mINChanA.begin(), vecTmp.begin(), vecTmp.end());
    mINChanC.insert(mINChanC.begin(), vecTmp.begin(), vecTmp.end());
    mIAmplA.insert(mIAmplA.begin(), vecTmp.begin(), vecTmp.end());
    mIAmplC.insert(mIAmplC.begin(), vecTmp.begin(), vecTmp.end());
  }

  /// resize buffer for accumulated currents
  void resize(const unsigned int nTotal)
  {
    mINChanA.resize(nTotal);
    mINChanC.resize(nTotal);
    mIAmplA.resize(nTotal);
    mIAmplC.resize(nTotal);
  }

  /// reset buffered currents
  void reset()
  {
    std::fill(mINChanA.begin(), mINChanA.end(), 0);
    std::fill(mINChanC.begin(), mINChanC.end(), 0);
    std::fill(mIAmplA.begin(), mIAmplA.end(), 0);
    std::fill(mIAmplC.begin(), mIAmplC.end(), 0);
  }

  /// normalize currents
  void normalize(const float factor)
  {
    std::transform(mINChanA.begin(), mINChanA.end(), mINChanA.begin(), [factor](const float val) { return val * factor; });
    std::transform(mINChanC.begin(), mINChanC.end(), mINChanC.begin(), [factor](const float val) { return val * factor; });
    std::transform(mIAmplA.begin(), mIAmplA.end(), mIAmplA.begin(), [factor](const float val) { return val * factor; });
    std::transform(mIAmplC.begin(), mIAmplC.end(), mIAmplC.begin(), [factor](const float val) { return val * factor; });
  }

  ClassDefNV(IFT0C, 1);
};

/// struct containing the integrated FV0 currents
struct IFV0C {
  std::vector<float> mINChanA;                                     ///< integrated 1D FIT currents for NChan A
  std::vector<float> mIAmplA;                                      ///< integrated 1D FIT currents for Ampl A
  bool areSameSize() const { return sameSize(mINChanA, mIAmplA); } ///< check if stored currents have same number of entries
  bool isEmpty() const { return mINChanA.empty(); }                ///< check if values are empty
  size_t getEntries() const { return mINChanA.size(); }            ///< \return returns number of values stored

  /// acummulate integrated currents at given index
  /// \param posIndex index where data will be copied to
  /// \param data integrated currents which will be copied
  void fill(const unsigned int posIndex, const IFV0C& data)
  {
    std::copy(data.mINChanA.begin(), data.mINChanA.end(), mINChanA.begin() + posIndex);
    std::copy(data.mIAmplA.begin(), data.mIAmplA.end(), mIAmplA.begin() + posIndex);
  }

  /// \param nDummyValues number of empty values which are inserted at the beginning of the accumulated integrated currents
  void insert(const unsigned int nDummyValues)
  {
    std::vector<float> vecTmp(nDummyValues, 0);
    mINChanA.insert(mINChanA.begin(), vecTmp.begin(), vecTmp.end());
    mIAmplA.insert(mIAmplA.begin(), vecTmp.begin(), vecTmp.end());
  }

  /// resize buffer for accumulated currents
  void resize(const unsigned int nTotal)
  {
    mINChanA.resize(nTotal);
    mIAmplA.resize(nTotal);
  }

  /// reset buffered currents
  void reset()
  {
    std::fill(mINChanA.begin(), mINChanA.end(), 0);
    std::fill(mIAmplA.begin(), mIAmplA.end(), 0);
  }

  /// normalize currents
  void normalize(const float factor)
  {
    std::transform(mINChanA.begin(), mINChanA.end(), mINChanA.begin(), [factor](const float val) { return val * factor; });
    std::transform(mIAmplA.begin(), mIAmplA.end(), mIAmplA.begin(), [factor](const float val) { return val * factor; });
  }

  ClassDefNV(IFV0C, 1);
};

/// struct containing the integrated FDD currents
struct IFDDC {
  std::vector<float> mINChanA;                                                        ///< integrated 1D FIT currents for NChan A
  std::vector<float> mINChanC;                                                        ///< integrated 1D FIT currents for NChan C
  std::vector<float> mIAmplA;                                                         ///< integrated 1D FIT currents for Ampl A
  std::vector<float> mIAmplC;                                                         ///< integrated 1D FIT currents for Ampl C
  bool areSameSize() const { return sameSize(mINChanA, mINChanC, mIAmplA, mIAmplC); } ///< check if stored currents have same number of entries
  bool isEmpty() const { return mINChanA.empty(); }                                   ///< check if values are empty
  size_t getEntries() const { return mINChanA.size(); }                               ///< \return returns number of values stored

  /// acummulate integrated currents at given index
  /// \param posIndex index where data will be copied to
  /// \param data integrated currents which will be copied
  void fill(const unsigned int posIndex, const IFDDC& data)
  {
    std::copy(data.mINChanA.begin(), data.mINChanA.end(), mINChanA.begin() + posIndex);
    std::copy(data.mINChanC.begin(), data.mINChanC.end(), mINChanC.begin() + posIndex);
    std::copy(data.mIAmplA.begin(), data.mIAmplA.end(), mIAmplA.begin() + posIndex);
    std::copy(data.mIAmplC.begin(), data.mIAmplC.end(), mIAmplC.begin() + posIndex);
  }

  /// \param nDummyValues number of empty values which are inserted at the beginning of the accumulated integrated currents
  void insert(const unsigned int nDummyValues)
  {
    std::vector<float> vecTmp(nDummyValues, 0);
    mINChanA.insert(mINChanA.begin(), vecTmp.begin(), vecTmp.end());
    mINChanC.insert(mINChanC.begin(), vecTmp.begin(), vecTmp.end());
    mIAmplA.insert(mIAmplA.begin(), vecTmp.begin(), vecTmp.end());
    mIAmplC.insert(mIAmplC.begin(), vecTmp.begin(), vecTmp.end());
  }

  /// resize buffer for accumulated currents
  void resize(const unsigned int nTotal)
  {
    mINChanA.resize(nTotal);
    mINChanC.resize(nTotal);
    mIAmplA.resize(nTotal);
    mIAmplC.resize(nTotal);
  }

  /// reset buffered currents
  void reset()
  {
    std::fill(mINChanA.begin(), mINChanA.end(), 0);
    std::fill(mINChanC.begin(), mINChanC.end(), 0);
    std::fill(mIAmplA.begin(), mIAmplA.end(), 0);
    std::fill(mIAmplC.begin(), mIAmplC.end(), 0);
  }

  /// normalize currents
  void normalize(const float factor)
  {
    std::transform(mINChanA.begin(), mINChanA.end(), mINChanA.begin(), [factor](const float val) { return val * factor; });
    std::transform(mINChanC.begin(), mINChanC.end(), mINChanC.begin(), [factor](const float val) { return val * factor; });
    std::transform(mIAmplA.begin(), mIAmplA.end(), mIAmplA.begin(), [factor](const float val) { return val * factor; });
    std::transform(mIAmplC.begin(), mIAmplC.end(), mIAmplC.begin(), [factor](const float val) { return val * factor; });
  }

  ClassDefNV(IFDDC, 1);
};

} // end namespace fit
} // end namespace o2

namespace o2
{
namespace calibration
{

/// class for accumulating integrated currents
template <typename DataT>
class IntegratedClusters
{
 public:
  /// \constructor
  /// \param tFirst first TF of the stored currents
  /// \param tLast last TF of the stored currents
  IntegratedClusters(o2::calibration::TFType tFirst, o2::calibration::TFType tLast) : mTFFirst{tFirst}, mTFLast{tLast} {};

  /// \default constructor for ROOT I/O
  IntegratedClusters() = default;

  /// print summary informations
  void print() const { LOGP(info, "TF Range from {} to {} with {} of remaining data", mTFFirst, mTFLast, mRemainingData); }

  /// accumulate currents for given TF
  /// \param tfID TF ID of incoming data
  /// \param currentsContainer container containing the currents for given detector for one TF for number of clusters
  void fill(const o2::calibration::TFType tfID, const DataT& currentsContainer);

  /// merging TOF currents with previous interval
  void merge(const IntegratedClusters* prev);

  /// \return always return true. To specify the number of time slot intervals to wait for one should use the --max-delay option
  bool hasEnoughData() const { return true; }

  /// \return returns accumulated currents
  const auto& getCurrents() const& { return mCurrents; }

  /// \return returns accumulated currents using move semantics
  auto getCurrents() && { return std::move(mCurrents); }

  /// \param currents currents for given detector which will be set
  void setCurrents(const DataT& currents) { mCurrents = currents; }

  /// dump object to disc
  /// \param outFileName name of the output file
  /// \param outName name of the object in the output file
  void dumpToFile(const char* outFileName = "IntegratedClusters.root", const char* outName = "IC") const;

  /// dump object to TTree for visualisation
  /// \param outFileName name of the output file
  void dumpToTree(const char* outFileName = "ICTree.root");

 private:
  DataT mCurrents;                          ///< buffer for integrated currents
  o2::calibration::TFType mTFFirst{};       ///< first TF of currents
  o2::calibration::TFType mTFLast{};        ///< last TF of currents
  o2::calibration::TFType mRemainingData{}; ///< counter for received data
  unsigned int mNValuesPerTF{};             ///< number of expected currents per TF (estimated from first received data)
  bool mInitialize{true};                   ///< flag if this object will be initialized when fill method is called

  /// init member when first data is received
  /// \param valuesPerTF number of expected values per TF
  void initData(const unsigned int valuesPerTF);

  ClassDefNV(IntegratedClusters, 1);
};

template <typename DataT>
class IntegratedClusterCalibrator : public o2::calibration::TimeSlotCalibration<IntegratedClusters<DataT>>
{
  using TFType = o2::calibration::TFType;
  using Slot = o2::calibration::TimeSlot<IntegratedClusters<DataT>>;
  using CalibVector = std::vector<DataT>;
  using TFinterval = std::vector<std::pair<TFType, TFType>>;
  using TimeInterval = std::vector<std::pair<long, long>>;

 public:
  /// default constructor
  IntegratedClusterCalibrator() = default;

  /// default destructor
  ~IntegratedClusterCalibrator() final = default;

  /// check if given slot has already enough data
  bool hasEnoughData(const Slot& slot) const final { return slot.getContainer()->hasEnoughData(); }

  /// clearing all calibration objects in the output buffer
  void initOutput() final;

  /// storing the integrated currents for given slot
  void finalizeSlot(Slot& slot) final;

  /// Creates new time slot
  Slot& emplaceNewSlot(bool front, TFType tstart, TFType tend) final;

  /// \return CCDB output informations
  const TFinterval& getTFinterval() const { return mIntervals; }

  /// \return Time frame time information
  const TimeInterval& getTimeIntervals() const { return mTimeIntervals; }

  /// \return returns calibration objects (pad-by-pad gain maps)
  auto getCalibs() && { return std::move(mCalibs); }

  /// check if calibration data is available
  bool hasCalibrationData() const { return mCalibs.size() > 0; }

  /// set if debug objects will be created
  void setDebug(const bool debug) { mDebug = debug; }

 private:
  TFinterval mIntervals;       ///< start and end time frames of each calibration time slots
  TimeInterval mTimeIntervals; ///< start and end times of each calibration time slots
  CalibVector mCalibs;         ///< Calibration object containing for each pad a histogram with normalized charge
  bool mDebug{false};          ///< write debug output objects

  ClassDefOverride(IntegratedClusterCalibrator, 1);
};

} // end namespace calibration
} // end namespace o2

#endif
