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

#ifndef ALICEO2_TPC_CALDET_H_
#define ALICEO2_TPC_CALDET_H_

#include <memory>
#include <numeric>
#include <vector>
#include <string>
#include <cassert>

#include "DataFormatsTPC/Defs.h"
#include "TPCBase/Mapper.h"
#include "TPCBase/ROC.h"
#include "TPCBase/Sector.h"
#include "TPCBase/CalArray.h"

#ifndef GPUCA_ALIGPUCODE
#include <Framework/Logger.h>
#include <fmt/format.h>
#include "Rtypes.h"
#endif

namespace o2
{
namespace tpc
{
/// Class to hold calibration data on a pad level
///
template <class T>
class CalDet
{
  using CalType = CalArray<T>;

 public:
  CalDet() { initData(); }
  CalDet(CalDet const&) = default;
  CalDet& operator=(CalDet const&) = default;
  ~CalDet() = default;

  CalDet(PadSubset padSusbset) : mName{"PadCalibrationObject"}, mData{}, mPadSubset{padSusbset} { initData(); }

  CalDet(const std::string_view name, const PadSubset padSusbset = PadSubset::ROC) : mName(name), mData(), mPadSubset(padSusbset) { initData(); }

  /// Return the pad subset type
  /// \return pad subset type
  PadSubset getPadSubset() const { return mPadSubset; }

  const std::vector<CalType>& getData() const { return mData; }
  std::vector<CalType>& getData() { return mData; }
  // void setValue(const unsigned int channel, const T value) { mData[channel] = value; }
  // const T& getValue(const unsigned int channel) const { return mData[channel]; }

  const CalType& getCalArray(size_t position) const { return mData[position]; }
  CalType& getCalArray(size_t position) { return mData[position]; }

  ///
  ///
  const T getValue(const int sec, const int globalPadInSector) const;
  void setValue(const int sec, const int globalPadInSector, const T& value);
  void setValue(const int sec, const int rowInSector, const int padInRow, const T& value);

  /// \todo return value of T& not possible if a default value should be returned, e.g. T{}:
  ///       warning: returning reference to temporary
  const T getValue(const ROC roc, const size_t row, const size_t pad) const;
  const T getValue(const CRU cru, const size_t row, const size_t pad) const;
  const T getValue(const Sector sec, const int rowInSector, const int padInRow) const;

  void setName(const std::string_view name, bool nameCalArrays = true)
  {
    mName = name.data();
    if (nameCalArrays) {
      initData();
    }
  }
  const std::string& getName() const { return mName; }

  const CalDet& multiply(const T& val) { return *this *= val; }
  const CalDet& operator+=(const CalDet& other);
  const CalDet& operator-=(const CalDet& other);
  const CalDet& operator*=(const CalDet& other);
  const CalDet& operator/=(const CalDet& other);
  bool operator==(const CalDet& other) const;

  const CalDet& operator+=(const T& val);
  const CalDet& operator-=(const T& val);
  const CalDet& operator*=(const T& val);
  const CalDet& operator/=(const T& val);

  const CalDet& operator=(const T& val);

  template <class U>
  friend CalDet<U> operator+(const CalDet<U>&, const CalDet<U>&);

  template <class U>
  friend CalDet<U> operator-(const CalDet<U>&, const CalDet<U>&);

  template <typename U = T>
  U getMean() const
  {
    if (mData.size() == 0) {
      return U{0};
    }

    U nVal = 0;
    U sum = 0;
    for (const auto& data : mData) {
      const auto& vals = data.getData();
      sum += std::accumulate(vals.begin(), vals.end(), U{0});
      nVal += static_cast<U>(vals.size());
    }

    return (nVal > 0) ? sum / nVal : U{0};
  }

  template <typename U = T>
  U getSum() const
  {
    if (mData.size() == 0) {
      return U{};
    }

    U sum{};
    for (const auto& data : mData) {
      const auto& vals = data.getData();
      sum += data.template getSum<U>();
    }

    return sum;
  }

 private:
  std::string mName;                     ///< name of the object
  std::vector<CalType> mData;            ///< internal CalArrays
  PadSubset mPadSubset = PadSubset::ROC; ///< Pad subset granularity

  /// initialize the data array depending on what is set as PadSubset
  void initData();

  ClassDefNV(CalDet, 1)
};

//______________________________________________________________________________
template <class T>
inline const T CalDet<T>::getValue(const int sector, const int globalPadInSector) const
{
  // This shold be a temporary speedup, a proper restructuring of Mapper and CalDet/CalArray is needed.
  // The default granularity for the moment should be ROC, for the assumptions below this should be assured
  assert(mPadSubset == PadSubset::ROC);
  int roc = sector;
  int padInROC = globalPadInSector;
  const int padsInIROC = Mapper::getPadsInIROC();
  if (globalPadInSector >= padsInIROC) {
    roc += Mapper::getNumberOfIROCs();
    padInROC -= padsInIROC;
  }
  return mData[roc].getValue(padInROC);
}

//______________________________________________________________________________
template <class T>
inline const T CalDet<T>::getValue(const ROC roc, const size_t row, const size_t pad) const
{
  // TODO: might need speedup and beautification
  const Mapper& mapper = Mapper::instance();

  // bind row and pad to the maximum rows and pads in the requested region
  const size_t nRows = mapper.getNumberOfRowsROC(roc);
  const size_t mappedRow = row % nRows;
  const size_t nPads = mapper.getNumberOfPadsInRowROC(roc, row);
  const size_t mappedPad = pad % nPads;

  // TODO: implement missing cases
  switch (mPadSubset) {
    case PadSubset::ROC: {
      return mData[roc].getValue(mappedRow, mappedPad);
      break;
    }
    case PadSubset::Partition: {
      return T{};
      break;
    }
    case PadSubset::Region: {
      const auto globalRow = roc.isOROC() ? mappedRow + mapper.getNumberOfRowsROC(ROC(0)) : mappedRow;
      return mData[Mapper::REGION[globalRow] + roc.getSector() * Mapper::NREGIONS].getValue(Mapper::OFFSETCRUGLOBAL[globalRow] + mappedPad);
      break;
    }
  }
  return T{};
}

//______________________________________________________________________________
template <class T>
inline const T CalDet<T>::getValue(const CRU cru, const size_t row, const size_t pad) const
{
  // TODO: might need speedup and beautification
  const Mapper& mapper = Mapper::instance();
  const auto& info = mapper.getPadRegionInfo(cru.region());

  // bind row and pad to the maximum rows and pads in the requested region
  const size_t nRows = info.getNumberOfPadRows();
  const size_t mappedRow = row % nRows;
  const size_t nPads = info.getPadsInRowRegion(mappedRow);
  const size_t mappedPad = pad % nPads;

  switch (mPadSubset) {
    case PadSubset::ROC: {
      const ROC roc = cru.roc();
      const size_t irocOffset = (cru.rocType() == RocType::IROC) ? 0 : mapper.getNumberOfRowsROC(0);
      const size_t rowROC = mappedRow + info.getGlobalRowOffset() - irocOffset;
      const size_t channel = mapper.getPadNumberInROC(PadROCPos(roc, rowROC, mappedPad));
      // printf("roc %2d, row %2zu, pad %3zu, channel: %3zu\n", roc.getRoc(), rowROC, mappedPad, channel);
      return mData[roc].getValue(channel);
      break;
    }
    case PadSubset::Partition: {
      break;
    }
    case PadSubset::Region: {
      return mData[cru].getValue(mappedRow, mappedPad);
      break;
    }
  }
  return T{};
}

template <class T>
inline void CalDet<T>::setValue(const int sec, const int globalPadInSector, const T& value)
{
  assert(mPadSubset == PadSubset::ROC);
  int roc = sec;
  int padInROC = globalPadInSector;
  const int padsInIROC = Mapper::getPadsInIROC();
  if (globalPadInSector >= padsInIROC) {
    roc += Mapper::getNumberOfIROCs();
    padInROC -= padsInIROC;
  }
  mData[roc].setValue(padInROC, value);
}

template <class T>
inline void CalDet<T>::setValue(const int sec, const int rowInSector, const int padInRow, const T& value)
{
  assert(mPadSubset == PadSubset::ROC);
  int roc = sec;
  int rowInROC = rowInSector;
  const int rowsInIROC = 63;
  if (rowInSector >= rowsInIROC) {
    roc += Mapper::getNumberOfIROCs();
    rowInROC -= rowsInIROC;
  }
  mData[roc].setValue(rowInROC, padInRow, value);
}

template <class T>
inline const T CalDet<T>::getValue(const Sector sec, const int rowInSector, const int padInRow) const
{
  assert(mPadSubset == PadSubset::ROC);
  int roc = sec;
  int rowInROC = rowInSector;
  const int rowsInIROC = 63;
  if (rowInSector >= rowsInIROC) {
    roc += Mapper::getNumberOfIROCs();
    rowInROC -= rowsInIROC;
  }
  return mData[roc].getValue(rowInROC, padInRow);
}

#ifndef GPUCA_ALIGPUCODE // hide from GPU standalone compilation

//______________________________________________________________________________
template <class T>
inline const CalDet<T>& CalDet<T>::operator+=(const CalDet& other)
{
  // make sure the calibration objects have the same substructure
  // TODO: perhaps make it independed of this
  if (mPadSubset != other.mPadSubset) {
    LOG(error) << "Pad subste type of the objects it not compatible";
    return *this;
  }

  for (size_t i = 0; i < mData.size(); ++i) {
    mData[i] += other.mData[i];
  }
  return *this;
}

//______________________________________________________________________________
template <class T>
inline const CalDet<T>& CalDet<T>::operator-=(const CalDet& other)
{
  // make sure the calibration objects have the same substructure
  // TODO: perhaps make it independed of this
  if (mPadSubset != other.mPadSubset) {
    LOG(error) << "Pad subste type of the objects it not compatible";
    return *this;
  }

  for (size_t i = 0; i < mData.size(); ++i) {
    mData[i] -= other.mData[i];
  }
  return *this;
}

//______________________________________________________________________________
template <class T>
inline const CalDet<T>& CalDet<T>::operator*=(const CalDet& other)
{
  // make sure the calibration objects have the same substructure
  // TODO: perhaps make it independed of this
  if (mPadSubset != other.mPadSubset) {
    LOG(error) << "Pad subste type of the objects it not compatible";
    return *this;
  }

  for (size_t i = 0; i < mData.size(); ++i) {
    mData[i] *= other.mData[i];
  }
  return *this;
}

//______________________________________________________________________________
template <class T>
inline const CalDet<T>& CalDet<T>::operator/=(const CalDet& other)
{
  // make sure the calibration objects have the same substructure
  // TODO: perhaps make it independed of this
  if (mPadSubset != other.mPadSubset) {
    LOG(error) << "Pad subste type of the objects it not compatible";
    return *this;
  }

  for (size_t i = 0; i < mData.size(); ++i) {
    mData[i] /= other.mData[i];
  }
  return *this;
}

//______________________________________________________________________________
template <class T>
inline const CalDet<T>& CalDet<T>::operator+=(const T& val)
{
  for (auto& cal : mData) {
    cal += val;
  }
  return *this;
}

//______________________________________________________________________________
template <class T>
inline const CalDet<T>& CalDet<T>::operator-=(const T& val)
{
  for (auto& cal : mData) {
    cal -= val;
  }
  return *this;
}

//______________________________________________________________________________
template <class T>
inline const CalDet<T>& CalDet<T>::operator*=(const T& val)
{
  for (auto& cal : mData) {
    cal *= val;
  }
  return *this;
}

//______________________________________________________________________________
template <class T>
inline const CalDet<T>& CalDet<T>::operator/=(const T& val)
{
  for (auto& cal : mData) {
    cal /= val;
  }
  return *this;
}

//______________________________________________________________________________
template <class T>
inline const CalDet<T>& CalDet<T>::operator=(const T& val)
{
  for (auto& cal : mData) {
    cal = val;
  }
  return *this;
}

//______________________________________________________________________________
template <class T>
inline bool CalDet<T>::operator==(const CalDet& other) const
{
  // make sure the calibration objects have the same substructure
  // TODO: perhaps make it independed of this
  if (mPadSubset != other.mPadSubset) {
    LOG(error) << "Pad subste type of the objects it not compatible";
    return false;
  }

  for (size_t i = 0; i < mData.size(); ++i) {
    if (!(mData[i] == other.mData[i])) {
      return false;
    }
  }
  return true;
}

//______________________________________________________________________________
template <class T>
CalDet<T> operator+(const CalDet<T>& c1, const CalDet<T>& c2)
{
  CalDet<T> ret(c1);
  ret += c2;
  return ret;
}

//______________________________________________________________________________
template <class T>
CalDet<T> operator-(const CalDet<T>& c1, const CalDet<T>& c2)
{
  CalDet<T> ret(c1);
  ret -= c2;
  return ret;
}
// ===| Full detector initialisation |==========================================
template <class T>
void CalDet<T>::initData()
{
  const auto& mapper = Mapper::instance();

  // ---| Define number of sub pad regions |------------------------------------
  size_t size = 0;
  bool hasData = mData.size() > 0;
  std::string frmt;
  switch (mPadSubset) {
    case PadSubset::ROC: {
      size = ROC::MaxROC;
      frmt = "{}_ROC_{:02d}";
      break;
    }
    case PadSubset::Partition: {
      size = Sector::MAXSECTOR * mapper.getNumberOfPartitions();
      frmt = "{}_Partition_{:02d}";
      break;
    }
    case PadSubset::Region: {
      size = Sector::MAXSECTOR * mapper.getNumberOfPadRegions();
      frmt = "{}_Region_{:02d}";
      break;
    }
  }

  for (size_t i = 0; i < size; ++i) {
    if (!hasData) {
      mData.push_back(CalType(mPadSubset, i));
    }
    mData[i].setName(fmt::format(fmt::runtime(frmt), mName, i));
  }
}

#endif // GPUCA_ALIGPUCODE

using CalPad = CalDet<float>;
} // namespace tpc
} // namespace o2

#endif
