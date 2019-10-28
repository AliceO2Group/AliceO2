// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_TPC_CALIBPEDESTAL_H_
#define ALICEO2_TPC_CALIBPEDESTAL_H_

/// \file   CalibPedestal.h
/// \author Jens Wiechula, Jens.Wiechula@ikf.uni-frankfurt.de

#include <vector>
#include <memory>

#include "Rtypes.h"

#include "DataFormatsTPC/Defs.h"
#include "TPCBase/CalDet.h"
#include "TPCBase/CRU.h"
#include "TPCCalibration/CalibRawBase.h"

class TH2;

namespace o2
{
namespace tpc
{

/// \brief Pedestal calibration class
///
/// This class is used to produce pad wise pedestal and noise calibration data
///
/// origin: TPC
/// \author Jens Wiechula, Jens.Wiechula@ikf.uni-frankfurt.de

class CalibPedestal : public CalibRawBase
{
 public:
  using vectorType = std::vector<float>;

  enum class StatisticsType {
    GausFit,   ///< Use Gaus fit for pedestal and noise
    MeanStdDev ///< Use mean and standard deviation
  };

  /// default constructor
  CalibPedestal(PadSubset padSubset = PadSubset::ROC);

  /// default destructor
  ~CalibPedestal() override = default;

  /// update function called once per digit
  ///
  /// \param roc readout chamber
  /// \param row row in roc
  /// \param pad pad in row
  /// \param timeBin time bin
  /// \param signal ADC signal
  Int_t updateROC(const Int_t roc, const Int_t row, const Int_t pad,
                  const Int_t timeBin, const Float_t signal) final;

  /// not used
  Int_t updateCRU(const CRU& cru, const Int_t row, const Int_t pad,
                  const Int_t timeBin, const Float_t signal) final { return 0; }

  /// Reset pedestal data
  void resetData();

  /// set the adc range
  void setADCRange(int minADC, int maxADC)
  {
    mADCMin = minADC;
    mADCMax = maxADC;
    mNumberOfADCs = mADCMax - mADCMin + 1;
  }

  /// set the statistics type
  void setStatisticsType(StatisticsType statisticsType) { mStatisticsType = statisticsType; }

  /// set the time bin range to analyse
  void setTimeBinRange(int first, int last)
  {
    mFirstTimeBin = first;
    mLastTimeBin = last;
  }
  /// Analyse the buffered adc values and calculate noise and pedestal
  void analyse();

  /// Get the pedestal calibration object
  ///
  /// \return pedestal calibration object
  const CalPad& getPedestal() const { return mPedestal; }

  /// Get the noise calibration object
  ///
  /// \return noise calibration object
  const CalPad& getNoise() const { return mNoise; }

  /// Get the statistics type
  StatisticsType getStatisticsType() const { return mStatisticsType; }

  /// Dump the relevant data to file
  void dumpToFile(const std::string filename) final;

  /// Dummy end event
  void endEvent() final{};

  /// generate a control histogram
  TH2* createControlHistogram(ROC roc);

 private:
  int mFirstTimeBin;              ///< first time bin used in analysis
  int mLastTimeBin;               ///< first time bin used in analysis
  int mADCMin;                    ///< minimum adc value
  int mADCMax;                    ///< maximum adc value
  int mNumberOfADCs;              ///< number of adc values (mADCMax-mADCMin+1)
  StatisticsType mStatisticsType; ///< statistics type to be used for pedestal and noise evaluation
  CalPad mPedestal;               ///< CalDet object with pedestal information
  CalPad mNoise;                  ///< CalDet object with noise

  std::vector<std::unique_ptr<vectorType>> mADCdata; //!< ADC data to calculate noise and pedestal

  /// return the value vector for a readout chamber
  ///
  /// \param roc readout chamber
  /// \param create if to create the vector if it does not exist
  vectorType* getVector(ROC roc, bool create = kFALSE);

  /// dummy reset
  void resetEvent() final {}
};

} // namespace tpc

} // namespace o2
#endif
