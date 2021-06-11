// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \class EMCALChannelCalibrator
/// \brief  Perform the EMCAL bad channel calibration
/// \author Hannah Bossi, Yale University
/// \ingroup EMCALCalib
/// \since Feb 11, 2021

#ifndef EMCAL_CHANNEL_CALIBRATOR_H_
#define EMCAL_CHANNEL_CALIBRATOR_H_

#include "DetectorsCalibration/TimeSlotCalibration.h"
#include "DetectorsCalibration/TimeSlot.h"
#include "DataFormatsEMCAL/Cell.h"
#include "EMCALBase/Geometry.h"
#include "CCDB/CcdbObjectInfo.h"

#include <array>
#include <boost/histogram.hpp>

namespace o2
{
namespace emcal
{
class EMCALChannelData
{
  //using Slot = o2::calibration::TimeSlot<o2::emcal::EMCALChannelData>;
  using Cells = o2::emcal::Cell;
  using boostHisto = boost::histogram::histogram<std::tuple<boost::histogram::axis::regular<double, boost::use_default, boost::use_default, boost::use_default>, boost::histogram::axis::integer<>>, boost::histogram::unlimited_storage<std::allocator<char>>>;

 public:
  // NCELLS includes DCal, treat as one calibration
  o2::emcal::Geometry* mGeometry = o2::emcal::Geometry::GetInstanceFromRunNumber(300000);
  int NCELLS = mGeometry->GetNCells();

  EMCALChannelData(int nb, float r) : mNBins(nb), mRange(r)
  {
    // boost histogram with amplitude vs. cell ID, specify the range and binning of the amplitude axis
    mHisto = boost::histogram::make_histogram(boost::histogram::axis::regular<>(mNBins, 0, mRange, "t-texp"), boost::histogram::axis::integer<>(0, NCELLS, "CELL ID"));
  }

  ~EMCALChannelData() = default;

  /// \brief Print relevant info for EMCALChannelData on a given stream
  /// \param stream Stream on which the info is printed on
  /// The function is called in the operator<< providing direct access
  /// to protected members. Explicit calls by the users is normally not
  /// necessary.
  void PrintStream(std::ostream& stream) const;
  /// \brief Print a useful message about the container.
  void print();
  /// \brief Fill the container with the cell ID and amplitude.
  void fill(const gsl::span<const o2::emcal::Cell> data);
  /// \brief Merge the data of two slots.
  void merge(const EMCALChannelData* prev);
  int findBin(float v) const;
  bool hasEnoughData() const;
  boostHisto& getHisto() { return mHisto; }
  const boostHisto& getHisto() const { return mHisto; }

  float getRange() const { return mRange; }
  void setRange(float r) { mRange = r; }

  int getNbins() const { return mNBins; }
  void setNbins(int nb) { mNBins = nb; }

  int getNEvents() const { return mEvents; }
  void setNEvents(int ne) { mEvents = ne; }

 private:
  float mRange = 0.35; // looked at old QA plots where max was 0.35 GeV, might need to be changed
  int mNBins = 1000;
  boostHisto mHisto;
  int mEvents = 1;

  ClassDefNV(EMCALChannelData, 1);
};
/// \brief Printing EMCALChannelData on the stream
/// \param in Stream where the EMCALChannelData is printed on
/// \param emcdata EMCALChannelData to be printed
std::ostream& operator<<(std::ostream& in, const EMCALChannelData& emcdata);

class EMCALChannelCalibrator : public o2::calibration::TimeSlotCalibration<o2::emcal::Cell, o2::emcal::EMCALChannelData>
{
  using TFType = uint64_t;
  using Slot = o2::calibration::TimeSlot<o2::emcal::EMCALChannelData>;
  using Cell = o2::emcal::Cell;
  using CcdbObjectInfo = o2::ccdb::CcdbObjectInfo;
  using CcdbObjectInfoVector = std::vector<CcdbObjectInfo>;

 public:
  EMCALChannelCalibrator(int nb = 1000, float r = 0.35) : mNBins(nb), mRange(r){};

  ~EMCALChannelCalibrator() final = default;

  /// \brief Checking if all channels have enough data to do calibration.
  bool hasEnoughData(const Slot& slot) const final;
  /// \brief Initialize the vector of our output objects.
  void initOutput() final;
  void finalizeSlot(Slot& slot) final;
  Slot& emplaceNewSlot(bool front, TFType tstart, TFType tend) final;

  void setIsTest(bool isTest) { mTest = isTest; }
  bool isTest() const { return mTest; }

 private:
  int mNBins = 0;     ///< bins of the histogram for passing
  float mRange = 0.;  ///< range of the histogram for passing
  bool mTest = false; ///< flag to be used when running in test mode: it simplify the processing (e.g. does not go through all channels)

  // output
  CcdbObjectInfoVector mInfoVector; // vector of CCDB Infos , each element is filled with the CCDB description of the accompanying TimeSlewing object

  ClassDefOverride(EMCALChannelCalibrator, 1);
};

} // end namespace emcal
} // end namespace o2

#endif /*EMCAL_CHANNEL_CALIBRATOR_H_ */
