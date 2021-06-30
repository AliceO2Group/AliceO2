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
#ifndef EMCALRAWFITTER_H_
#define EMCALRAWFITTER_H_

#include <iosfwd>
#include <array>
#include <optional>
#include <Rtypes.h>
#include <gsl/span>
#include "EMCALReconstruction/CaloFitResults.h"
#include "DataFormatsEMCAL/Constants.h"
#include "EMCALReconstruction/Bunch.h"

namespace o2
{

namespace emcal
{

/// \class EMCalRawFitter
/// \brief  Base class for extraction of signal amplitude and peak position
/// \author Hadi Hassan <hadi.hassan@cern.ch>, Oak Ridge National Laboratory
/// \since November 4th, 2019
///
/// Base class for extraction
/// of signal amplitude and peak position
/// from CALO Calorimeter RAW data.
/// It contains some utilities for preparing / selecting
/// signals suitable for signal extraction
class CaloRawFitter
{

 public:
  /**
   * \enum RawFitterError_t
   * \brief Error codes for failures in raw fitter procedure
   */
  enum class RawFitterError_t {
    SAMPLE_UNINITIALIZED, ///< Samples not initialized or length is 0
    FIT_ERROR,            ///< Fit procedure failed
    CHI2_ERROR,           ///< Chi2 cannot be determined (usually due to insufficient amount of samples)
    BUNCH_NOT_OK          ///< Bunch selection failed
  };

  /// \brief Create error message for a given error type
  /// \param fiterror Fit error type
  /// \return Error message connected to the error type
  static std::string createErrorMessage(RawFitterError_t fiterror);

  /// \brief Convert error type to numeric representation
  /// \param fiterror Fit error type
  /// \return Numeric representation of the raw fitter error
  static int getErrorNumber(RawFitterError_t fiterror);

  /// \brief Get the number of raw fit error types supported
  /// \return Number of error types (4)
  static constexpr int getNumberOfErrorTypes() noexcept { return 4; }

  /// \brief Constructor
  CaloRawFitter(const char* name, const char* nameshort);

  /// \brief Destructor
  virtual ~CaloRawFitter() = default;

  virtual CaloFitResults evaluate(const gsl::span<const Bunch> bunchvector,
                                  std::optional<unsigned int> altrocfg1,
                                  std::optional<unsigned int> altrocfg2) = 0;

  /// \brief Method to do the selection of what should possibly be fitted.
  /// \param bunchvector ALTRO bunches for the current channel
  /// \param altrocfg1 ALTRO config register 1 from RCU trailer
  /// \param altrocfg2 ALTRO config register 2 from RCU trailer
  /// \return Size of the sub-selected sample,
  /// \return index of the bunch with maximum signal,
  /// \return maximum signal,
  /// \return maximum aplitude,
  /// \return index of max aplitude in array,
  /// \return pedestal,
  /// \return first time bin,
  /// \return last time bin,
  std::tuple<int, int, float, short, short, float, int, int> preFitEvaluateSamples(const gsl::span<const Bunch> bunchvector,
                                                                                   std::optional<unsigned int> altrocfg1, std::optional<unsigned int> altrocfg2, int acut);

  /// \brief The require time range if the maximum ADC value is between min and max (timebin)
  void setTimeConstraint(int min, int max);

  void setIsZeroSuppressed(bool iszs = true) { mIsZerosupressed = iszs; }
  void setAmpCut(float cut) { mAmpCut = cut; }
  void setNsamplePed(int i) { mNsamplePed = i; }
  void setL1Phase(double phase) { mL1Phase = phase; }

  bool getIsZeroSuppressed() const { return mIsZerosupressed; }
  float getAmpCut() const { return mAmpCut; }
  int getNsamplePed() const { return mNsamplePed; }

  // access to array info
  double getReversed(const int i) const { return mReversed[i]; }
  const char* getAlgoName() const { return mName.c_str(); }
  const char* getAlgoAbbr() const { return mNameShort.c_str(); }
  FitAlgorithm getAlgo() const { return mAlgo; }

  /// \brief Get the maximum of a bunch array
  /// \return Maximum amplitute
  short maxAmp(const Bunch& bunch, int& maxindex) const;

  /// \brief Get maximum of array
  /// \return Maximum amplitute
  unsigned short maxAmp(const gsl::span<unsigned short> data) const;

  /// \brief A bunch is considered invalid if the maximum is in the first or last time-bin.
  bool checkBunchEdgesForMax(const Bunch& bunch) const;

  /// \brief Check if the index of the max ADC vaue is consistent with trigger.
  bool isInTimeRange(int maxindex, int maxtime, int mintime) const;

  /// \brief Time sample comes in reversed order, revers them back Subtract the baseline based on content of altrocfg1 and altrocfg2.
  /// \return Pedestal
  /// \return Array with revered and pedestal subtracted ADC signals
  std::tuple<float, std::array<double, constants::EMCAL_MAXTIMEBINS>> reverseAndSubtractPed(const Bunch& bunch,
                                                                                            std::optional<unsigned int> altrocfg1, std::optional<unsigned int> altrocfg2) const;

  /// \brief We select the bunch with the highest amplitude unless any time constraints is set.
  /// \return The index of the array with the maximum aplitude
  /// \return The bin where we have a maximum amp
  /// \return The maximum ADC signal
  std::tuple<short, short, short> selectBunch(const gsl::span<const Bunch>& bunchvector);

  /// \brief Selection of subset of data from one bunch that will be used for fitting or Peak finding.
  /// Go to the left and right of index of the maximum time bin
  /// Until the ADC value is less than cut, or derivative changes sign (data jump)
  /// \return The index of the jumb before the maximum
  /// \return The index of the jumb after the maximum
  std::tuple<int, int> selectSubarray(const gsl::span<double> data, short maxindex, int cut) const;

  /// \brief Pedestal evaluation if not zero suppressed
  float evaluatePedestal(const gsl::span<const uint16_t> data, std::optional<int> length) const;

  /// \brief Calculates the chi2 of the fit
  ///
  /// \param amp    - max amplitude;
  /// \param time   - time of max amplitude;
  /// \param first  - sample array indices to be used
  /// \param last   - sample array indices to be used
  /// \param adcErr - nominal error of amplitude measurement (one value for all channels) if adcErr<0 that mean adcErr=1.
  /// \param tau    - filter time response (in timebin units)
  ///
  /// \return chi2
  double calculateChi2(double amp, double time,
                       int first, int last,
                       double adcErr = 1,
                       double tau = 2.35) const;

 protected:
  std::array<double, constants::EMCAL_MAXTIMEBINS> mReversed; ///< Reversed sequence of samples (pedestalsubtracted)

  int mMinTimeIndex; ///< The timebin of the max signal value must be between fMinTimeIndex and fMaxTimeIndex
  int mMaxTimeIndex; ///< The timebin of the max signal value must be between fMinTimeIndex and fMaxTimeIndex

  float mAmpCut; ///< Max ADC - pedestal must be higher than this befor attemting to extract the amplitude

  int mNsamplePed; ///< Number of samples used for pedestal calculation (first in bunch)

  bool mIsZerosupressed; ///< Wether or not the data is zeros supressed, by default its assumed that the baseline is also subtracted if set to true

  std::string mName;      ///< Name of the algorithm
  std::string mNameShort; ///< Abbrevation for the name

  FitAlgorithm mAlgo; ///< Which algorithm to use

  double mL1Phase; ///< Phase of the ADC sampling clock relative to the LHC clock

  double mAmp; ///< The amplitude in entities of ADC counts

  ClassDefNV(CaloRawFitter, 1);
}; //CaloRawFitter

} // namespace emcal

} // namespace o2
#endif
