// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_TPC_CALDET_H_
#define ALICEO2_TPC_CALDET_H_

#include <memory>
#include <vector>
#include <string>
#include <boost/format.hpp>
#include <boost/range/combine.hpp>

#include "FairLogger.h"

#include "TPCBase/Defs.h"
#include "TPCBase/Mapper.h"
#include "TPCBase/ROC.h"
#include "TPCBase/CalArray.h"

using boost::format;

namespace o2 {
namespace TPC {

/// Class to hold calibration data on a pad level
///
template <class T>
class CalDet {
  using CalType = CalArray<T>;
public:
  CalDet() = default;
  ~CalDet() = default;

  CalDet(PadSubset padSusbset)
    : mName{"PadCalibrationObject"},
      mData{},
      mPadSubset{padSusbset}
  {
    initData();
  }

//______________________________________________________________________________

  CalDet(const std::string name) : 
    mName(name),
    mData(),
    mPadSubset(PadSubset::ROC)
  {
    initData();
  }

  //CalDet(const CalDet& calDet) :
    //mName(calDet.mName),
    //mData(calDet.mData)
  //{}

  /// Return the pad subset type
  /// \return pad subset type
  PadSubset getPadSubset() const { return mPadSubset; }

  const std::vector<CalType>& getData() const { return mData; }
  std::vector<CalType>& getData() { return mData; }

  //void setValue(const unsigned int channel, const T value) { mData[channel] = value; }
  //const T& getValue(const unsigned int channel) const { return mData[channel]; }

  const CalType& getCalArray(size_t position) const { return mData[position]; }
  CalType& getCalArray(size_t position) { return mData[position]; }

  /// \todo return value of T& not possible if a default value should be returned, e.g. T{}:
  ///       warning: returning reference to temporary
  const T getValue(const ROC roc, const size_t row, const size_t pad) const;
  const T getValue(const CRU cru, const size_t row, const size_t pad) const;

  void setName(const std::string& name) { mName = name; }
  const std::string& getName() const { return mName; }

  const CalDet& multiply(const T& val) { return *this *= val; }

  const CalDet& operator+= (const CalDet& other);
  const CalDet& operator-= (const CalDet& other);
  const CalDet& operator*= (const CalDet& other);
  const CalDet& operator/= (const CalDet& other);

  const CalDet& operator+= (const T& val);
  const CalDet& operator-= (const T& val);
  const CalDet& operator*= (const T& val);
  const CalDet& operator/= (const T& val);
private:
  std::string mName;          ///< name of the object
  std::vector<CalType> mData; ///< internal CalArrays
  PadSubset mPadSubset;       ///< Pad subset granularity

  /// initialize the data array depending on what is set as PadSubset
  void initData();
};

//______________________________________________________________________________
template <class T>
inline const T CalDet<T>::getValue(const ROC roc, const size_t row, const size_t pad) const
{
  // TODO: might need speedup and beautification
  static const Mapper& mapper = Mapper::instance();

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
      return T{};
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
  static const Mapper& mapper = Mapper::instance();
  const auto& info = mapper.getPadRegionInfo(cru.region());

  // bind row and pad to the maximum rows and pads in the requested region
  const size_t nRows = info.getNumberOfPadRows();
  const size_t mappedRow = row % nRows;
  const size_t nPads = info.getPadsInRowRegion(mappedRow);
  const size_t mappedPad = pad % nPads;

  switch (mPadSubset) {
    case PadSubset::ROC: {
      const ROC roc = cru.roc();
      const size_t irocOffset = (cru.rocType()==RocType::IROC)?0: mapper.getNumberOfRowsROC(0);
      const size_t rowROC = mappedRow + info.getGlobalRowOffset() - irocOffset;
      const size_t channel = mapper.getPadNumberInROC(PadROCPos(roc, rowROC, mappedPad));
      //printf("roc %2d, row %2zu, pad %3zu, channel: %3zu\n", roc.getRoc(), rowROC, mappedPad, channel);
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

//______________________________________________________________________________
template <class T>
inline const CalDet<T>& CalDet<T>::operator+= (const CalDet& other)
{
  // make sure the calibration objects have the same substructure
  // TODO: perhaps make it independed of this
  if (mPadSubset != other.mPadSubset) {
    LOG(ERROR) << "Pad subste type of the objects it not compatible" 
               << FairLogger::endl;
    return *this;
  }
  // somehow rootcint does not like boost::combine for some reason :(
  //for (auto& val : boost::combine(mData, other.mData)) {
    //val.get<0>() += val.get<1>();
  for (size_t i=0; i<mData.size(); ++i) {
    mData[i] += other.mData[i];
  }
  return *this;
}

//______________________________________________________________________________
template <class T>
inline const CalDet<T>& CalDet<T>::operator-= (const CalDet& other)
{
  // make sure the calibration objects have the same substructure
  // TODO: perhaps make it independed of this
  if (mPadSubset != other.mPadSubset) {
    LOG(ERROR) << "Pad subste type of the objects it not compatible" 
               << FairLogger::endl;
    return *this;
  }
  // somehow rootcint does not like boost::combine for some reason :(
  //for (auto& val : boost::combine(mData, other.mData)) {
    //val.get<0>() -= val.get<1>();
  for (size_t i=0; i<mData.size(); ++i) {
    mData[i] -= other.mData[i];
  }
  return *this;
}

//______________________________________________________________________________
template <class T>
inline const CalDet<T>& CalDet<T>::operator*= (const CalDet& other)
{
  // make sure the calibration objects have the same substructure
  // TODO: perhaps make it independed of this
  if (mPadSubset != other.mPadSubset) {
    LOG(ERROR) << "Pad subste type of the objects it not compatible" 
               << FairLogger::endl;
    return *this;
  }
  // somehow rootcint does not like boost::combine for some reason :(
  //for (auto& val : boost::combine(mData, other.mData)) {
    //val.get<0>() *= val.get<1>();
  for (size_t i=0; i<mData.size(); ++i) {
    mData[i] *= other.mData[i];
  }
  return *this;
}

//______________________________________________________________________________
template <class T>
inline const CalDet<T>& CalDet<T>::operator/= (const CalDet& other)
{
  // make sure the calibration objects have the same substructure
  // TODO: perhaps make it independed of this
  if (mPadSubset != other.mPadSubset) {
    LOG(ERROR) << "Pad subste type of the objects it not compatible" 
               << FairLogger::endl;
    return *this;
  }
  // somehow rootcint does not like boost::combine for some reason :(
  //for (auto& val : boost::combine(mData, other.mData)) {
    //val.get<0>() /= val.get<1>();
  for (size_t i=0; i<mData.size(); ++i) {
    mData[i] /= other.mData[i];
  }
  return *this;
}

//______________________________________________________________________________
template <class T>
inline const CalDet<T>& CalDet<T>::operator+= (const T& val)
{
  for (auto& cal : mData) {
    cal += val;
  }
  return *this;
}

//______________________________________________________________________________
template <class T>
inline const CalDet<T>& CalDet<T>::operator-= (const T& val)
{
  for (auto& cal : mData) {
    cal -= val;
  }
  return *this;
}

//______________________________________________________________________________
template <class T>
inline const CalDet<T>& CalDet<T>::operator*= (const T& val)
{
  for (auto& cal : mData) {
    cal *= val;
  }
  return *this;
}

//______________________________________________________________________________
template <class T>
inline const CalDet<T>& CalDet<T>::operator/= (const T& val)
{
  for (auto& cal : mData) {
    cal /= val;
  }
  return *this;
}

// ===| Full detector initialisation |==========================================
template <class T>
void CalDet<T>::initData()
{
  const auto& mapper = Mapper::instance();

  // ---| Define number of sub pad regions |------------------------------------
  size_t size = 0;

  switch (mPadSubset) {
    case PadSubset::ROC: {
      size = ROC::MaxROC;
      break;
    }
    case PadSubset::Partition: {
      size = Sector::MAXSECTOR * mapper.getNumberOfPartitions();
      break;
    }
    case PadSubset::Region: {
      size = Sector::MAXSECTOR * mapper.getNumberOfPadRegions();
      break;
    }
  }

  for (size_t i=0; i<size; ++i) {
    mData.push_back(CalType(mPadSubset, i));
  }
}

using CalPad = CalDet<float>;

}
}

#endif
