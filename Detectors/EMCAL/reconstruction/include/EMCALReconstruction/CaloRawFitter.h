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
#include <string_view>
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
    BUNCH_NOT_OK,         ///< Bunch selection failed
    LOW_SIGNAL            ///< No ADC value above threshold found
  };

  /// \brief Create error message for a given error type
  /// \param fiterror Fit error type
  /// \return Error message connected to the error type
  static std::string createErrorMessage(RawFitterError_t fiterror) { return getErrorTypeDescription(fiterror); }

  /// \brief Convert error type to numeric representation
  /// \param fiterror Fit error type
  /// \return Numeric representation of the raw fitter error
  static int getErrorNumber(RawFitterError_t fiterror);

  /// \brief Convert numeric representation of error type to RawFitterError_t
  ///
  /// Expect the error code provided to be a valid error code.
  ///
  /// \param fiterror Numeric representation of fit error
  /// \return Symbolic representation of the error code
  static RawFitterError_t intToErrorType(unsigned int fiterror);

  /// \brief Get the number of raw fit error types supported
  /// \return Number of error types (5)
  static constexpr int getNumberOfErrorTypes() noexcept { return 5; }

  /// \brief Get the name connected to the fit error type
  ///
  /// A single word descriptor i.e. used for object names
  /// is produced.
  ///
  /// \param errortype Error type raising the exception (symbolic representation)
  /// \return Name of the fit error type
  static const char* getErrorTypeName(RawFitterError_t fiterror);

  /// \brief Get the name connected to the fit error type
  ///
  /// A single word descriptor i.e. used for object names
  /// is produced.
  ///
  /// \param errortype Error type raising the exception (numeric representation)
  /// \return Name of the fit error type
  static const char* getErrorTypeName(unsigned int fiterror)
  {
    return getErrorTypeName(intToErrorType(fiterror));
  }

  /// \brief Get the title connected to the fit error type
  ///
  /// A short description i.e. used for bin labels or histogam titles
  /// is produced.
  ///
  /// \param errortype Error type raising the exception (symbolic representation)
  /// \return Title of the fit error type
  static const char* getErrorTypeTitle(RawFitterError_t fiterror);

  /// \brief Get the title connected to the fit error type
  ///
  /// A short description i.e. used for bin labels or histogam titles
  /// is produced.
  ///
  /// \param errortype Error type raising the exception (numeric representation)
  /// \return Title of the fit error type
  static const char* getErrorTypeTitle(unsigned int fiterror)
  {
    return getErrorTypeTitle(intToErrorType(fiterror));
  }

  /// \brief Get the description connected to the fit error type
  ///
  /// A detailed description i.e. used for error message on the stdout
  /// is produced.
  ///
  /// \param errortype Error type raising the exception (symbolic representation)
  /// \return Description connected to the fit error type
  static const char* getErrorTypeDescription(RawFitterError_t fiterror);

  /// \brief Get the description connected to the fit error type
  ///
  /// A detailed description i.e. used for error message on the stdout
  /// is produced.
  ///
  /// \param errortype Error type raising the exception (numeric representation)
  /// \return Description connected to the fit error type
  static const char* getErrorTypeDescription(unsigned int fiterror)
  {
    return getErrorTypeDescription(intToErrorType(fiterror));
  }

  /// \brief Constructor
  CaloRawFitter(const char* name, const char* nameshort);

  /// \brief Destructor
  virtual ~CaloRawFitter() = default;

  virtual CaloFitResults evaluate(const gsl::span<const Bunch> bunchvector) = 0;

  /// \brief Method to do the selection of what should possibly be fitted.
  /// \param bunchvector ALTRO bunches for the current channel
  /// \param adcThreshold ADC threshold applied in peak finding
  /// \return Size of the sub-selected sample,
  /// \return index of the bunch with maximum signal,
  /// \return maximum signal,
  /// \return maximum aplitude,
  /// \return index of max aplitude in array,
  /// \return pedestal,
  /// \return first time bin,
  /// \return last time bin,
  std::tuple<int, int, float, short, short, float, int, int> preFitEvaluateSamples(const gsl::span<const Bunch> bunchvector, int adcThreshold);

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
  const std::string_view getAlgoName() const { return mName.c_str(); }
  const std::string_view getAlgoAbbr() const { return mNameShort.c_str(); }

  /// \brief Get Type of the fit algorithm
  /// \return Fit algorithm type
  FitAlgorithm getAlgo() const { return mAlgo; }

  /// \brief Get the maximum amplitude and its index of a bunch array
  /// \return Maximum ADC value of the bunch
  /// \return Index of the max. ADC value of the bunch
  std::tuple<short, int> getMaxAmplitudeBunch(const Bunch& bunchx) const;

  /// \brief Get maximum of array
  /// \return Maximum amplitute
  unsigned short getMaxAmplitudeBunch(const gsl::span<unsigned short> data) const;

  /// \brief Check if the max. ADC value is at the edge of a bunch
  /// \param bunch The bunch to be checked
  /// \return True if the Max. ADC value is either the first or the last value of a bunch
  bool isMaxADCBunchEdge(const Bunch& bunch) const;

  /// \brief Check if the index of the max ADC vaue is within accepted time range
  /// \param indexMaxADC Index of the max. ADC value
  /// \param maxtime Max. time value of the accepted range
  /// \param mintime Min. time value of the accepted range
  /// \return True of the index of the max ADC is within the accepted time range, false otherwise
  bool isInTimeRange(int indexMaxADC, int maxtime, int mintime) const;

  /// \brief Time sample comes in reversed order, revers them back Subtract the baseline based on content of altrocfg1 and altrocfg2.
  /// \return Pedestal
  /// \return Array with revered and pedestal subtracted ADC signals
  std::tuple<float, std::array<double, constants::EMCAL_MAXTIMEBINS>> reverseAndSubtractPed(const Bunch& bunch) const;

  /// \brief We select the bunch with the highest amplitude unless any time constraints is set.
  /// \param bunchvector Bunches of the channel among which to select the maximium
  /// \return The index of the array with the maximum aplitude
  /// \return The bin where we have a maximum amp
  /// \return The maximum ADC signal
  /// \throw RawFitterError_t::BUNCH_NOT_OK ADC value is at a bunch edge
  std::tuple<short, short, short> selectMaximumBunch(const gsl::span<const Bunch>& bunchvector);

  /// \brief Find region constrainded by its closest minima around the main peak
  /// \param adcValues ADC values of the bunch
  /// \param indexMaxADC Index of the maximum ADC value
  /// \param threshold Min. ADC value accepted in peak region
  /// \return Index of the closest minimum before the peak
  /// \return Index of the closest minimum after the pea
  std::tuple<int, int> findPeakRegion(const gsl::span<double> adcValues, short indexMaxADC, int threshold) const;

  /// \brief Calculate the pedestal from the ADC values in a bunch
  /// \param data ADC values from which to calculate the pedestal
  /// \param length Optional bunch length
  /// \return Pedestal value
  double evaluatePedestal(const gsl::span<const uint16_t> data, std::optional<int> length) const;

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

  int getMinTimeIndex() const { return mMinTimeIndex; }
  int getMaxTimeIndex() const { return mMaxTimeIndex; }

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
}; // CaloRawFitter

/// \brief Stream operator for CaloRawFitter's RawFitterError
/// \param stream Stream to print on
/// \param error Error code to be printed
/// \return Stream after printing
std::ostream& operator<<(std::ostream& stream, const CaloRawFitter::RawFitterError_t error);

} // namespace emcal

} // namespace o2
#endif
