// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_TPC_CALIBPULSER_H_
#define ALICEO2_TPC_CALIBPULSER_H_

/// \file   CalibPulser.h
/// \author Jens Wiechula, Jens.Wiechula@ikf.uni-frankfurt.de

#include <vector>
#include <memory>

#include "TH2S.h"

#include "DataFormatsTPC/Defs.h"
#include "TPCBase/CalDet.h"
#include "TPCBase/CRU.h"
#include "TPCBase/PadROCPos.h"
#include "TPCCalibration/CalibRawBase.h"

namespace o2
{
namespace tpc
{

/// \brief Pulser calibration class
///
/// This class is used to produce pad wise pulser calibration information
///
/// origin: TPC
/// \author Jens Wiechula, Jens.Wiechula@ikf.uni-frankfurt.de

class CalibPulser : public CalibRawBase
{
 public:
  using VectorType = std::vector<float>;
  using PtrVectorType = std::vector<std::unique_ptr<TH2S>>;

  /// default constructor
  CalibPulser(PadSubset padSubset = PadSubset::ROC);

  /// default destructor
  ~CalibPulser() override = default;

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

  /// Reset temporary data and histogrms
  void resetData();

  /// set the adc range
  void setADCRange(int minADC, int maxADC)
  {
    mADCMin = minADC;
    mADCMax = maxADC;
    mNumberOfADCs = mADCMax - mADCMin + 1;
  }

  /// set the timeBin range
  void setTimeBinRange(int first, int last)
  {
    mFirstTimeBin = first;
    mLastTimeBin = last;
    //TODO: until automatic T0 calibration is done we use the same time range
    //      as for the time bin selection
    mXminT0 = mFirstTimeBin;
    mXmaxT0 = mLastTimeBin;
  }

  /// set noise and pedestal calibration objects
  void setPedestalAndNoise(const CalPad* pedestal, const CalPad* noise)
  {
    mPedestal = pedestal;
    mNoise = noise;
  }

  /// set T0 Binning
  void setT0Binning(int nbins, float min, float max)
  {
    mNbinsT0 = nbins;
    mXminT0 = min;
    mXmaxT0 = max;
  }

  /// set Width Binning
  void setWidthBinning(int nbins, float min, float max)
  {
    mNbinsWidth = nbins;
    mXminWidth = min;
    mXmaxWidth = max;
  }

  /// set Qtot Binning
  void setQtotBinning(int nbins, float min, float max)
  {
    mNbinsQtot = nbins;
    mXminQtot = min;
    mXmaxQtot = max;
  }

  /// Analyse the buffered pulser information
  void analyse();

  /// Get the pulser mean time calibration object
  /// \return pedestal calibration object
  const CalPad& getT0() const { return mT0; }

  /// Get the pulser pulse with calibration object
  /// \return pulse width calibration object
  const CalPad& getWidth() const { return mWidth; }

  /// Get the pulser total charge calibration object
  /// \return pulser total charge calibration object
  const CalPad& getQtot() const { return mQtot; }

  /// Dump the relevant data to file
  void dumpToFile(const std::string filename) final;

  /// Process the end of one raw reader
  void endReader() final;

  /// Dummy end event
  void endEvent() final{};

 private:
  // reference histogram ranges
  int mNbinsT0;     ///< Number of bins for T0 reference histogram
  float mXminT0;    ///< xmin   of T0 reference histogram
  float mXmaxT0;    ///< xmax   of T0 reference histogram
  int mNbinsQtot;   ///< Number of bins for Qtot reference histogram
  float mXminQtot;  ///< xmin   of Qtot reference histogram
  float mXmaxQtot;  ///< xmax   of Qtot reference histogram
  int mNbinsWidth;  ///< Number of bins for width reference histogram
  float mXminWidth; ///< xmin   of width reference histogram
  float mXmaxWidth; ///< xmax   of width reference histogram

  int mFirstTimeBin; ///< first time bin used in analysis
  int mLastTimeBin;  ///< first time bin used in analysis
  int mADCMin;       ///< minimum adc value
  int mADCMax;       ///< maximum adc value
  int mNumberOfADCs; ///< number of adc values (mADCMax-mADCMin+1)
  int mPeakIntMinus; ///< lower bound from maximum for the peak integration, mean and std dev. calc
  int mPeakIntPlus;  ///< upper bound from maximum for the peak integration, mean and std dev. calc
  int mMinimumQtot;  ///< minimal Qtot accepted as pulser signal
  CalPad mT0;        ///< CalDet object with pulser time information
  CalPad mWidth;     ///< CalDet object with pulser pulse width information
  CalPad mQtot;      ///< CalDet object with pulser Qtot information

  const CalPad* mPedestal; //!< Pedestal calibration object
  const CalPad* mNoise;    //!< Noise calibration object

  std::map<PadROCPos, VectorType> mPulserData; //!< ADC data to calculate pulser information

  PtrVectorType mT0Histograms;    //!< T0 histogramgs per ROC and pad
  PtrVectorType mWidthHistograms; //!< Width histogramgs per ROC and pad
  PtrVectorType mQtotHistograms;  //!< Qtot histogramgs per ROC and pad

  /// pulser data object
  struct PulserData {
    float mT0{0.f};
    float mWidth{0.f};
    float mQtot{0.f};
  };

  /// create or return a histogram for a specific ROC
  /// \param roc ROC
  /// \param rocVector vector of histograms to choose from
  /// \param nbins number of bins
  /// \param min minimum value of the histogram
  /// \param max maximum value of the histogram
  TH2S* getHistogram(ROC roc, CalibPulser::PtrVectorType& rocVector,
                     int nbins, float min, float max,
                     std::string_view type, bool create /*=kFALSE*/);

  /// return pointer to T0 histogram
  TH2S* getHistoT0(ROC roc, Bool_t force)
  {
    return getHistogram(roc, mT0Histograms, mNbinsT0, mXminT0, mXmaxT0, "T0", force);
  }

  /// return pointer to Qtot histogram
  TH2S* getHistoQtot(ROC roc, Bool_t force)
  {
    return getHistogram(roc, mQtotHistograms, mNbinsQtot, mXminQtot, mXmaxQtot, "Qtot", force);
  }

  /// return pointer to Q histogram
  TH2S* getHistoSigma(ROC roc, Bool_t force)
  {
    return getHistogram(roc, mWidthHistograms, mNbinsWidth, mXminWidth, mXmaxWidth, "Width", force);
  }

  /// process the adc values of one pad
  /// extract the T0, width, qmax and qtot
  /// \param adcData vector with ADC values per pad
  PulserData processPadData(const PadROCPos& padROCPos, const VectorType& adcData);

  /// dummy reset
  void resetEvent() final {}
};

} // namespace tpc

} // namespace o2
#endif
