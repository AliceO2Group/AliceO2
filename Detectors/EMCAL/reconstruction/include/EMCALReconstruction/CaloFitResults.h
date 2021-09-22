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

#ifndef CALOFITRESULTS_H_
#define CALOFITRESULTS_H_

#include <iosfwd>
#include <Rtypes.h>
#include "DataFormatsEMCAL/Constants.h"

namespace o2
{

namespace emcal
{

/// \class CaloFitResults
/// \brief  Container class to hold results from fitting
/// \ingroup EMCALreconstruction
/// \author Hadi Hassan <hadi.hassan@cern.ch>, Oak Ridge National Laboratory
/// \since November 4th, 2019
///
/// Container class to hold results from fitting
/// as well as other methods for
/// raw data signals extraction. The class memebers
/// mChi2Sig, mNdfSig is only relevant if a fitting procedure is
/// Applied. mStatus holds information on wether or not
/// The signal was fitted sucessfully. mStatus might have a different meaning If other
/// procedures than A different meaning Fitting is applied
class CaloFitResults
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
    LOW_SIGNAL,           ///< No ADC value above threshold found
    NO_ERROR              ///< No raw fitter error
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

  /// \brief Default constructor
  CaloFitResults() = default;

  /// \brief copy constructor
  CaloFitResults(const CaloFitResults& fitresults) = default;

  /// \brief Assignment operator
  CaloFitResults& operator=(const CaloFitResults& source);

  /// \brief Constructor for recording all the fit  parameters
  explicit CaloFitResults(unsigned short maxSig,
                          float ped,
                          int fitStatus,
                          float amp,
                          double time,
                          int maxTimebin,
                          float chi,
                          unsigned short ndf,
                          unsigned short minSig = 0);

  /// \brief shorter interface when no fit is done
  explicit CaloFitResults(unsigned short maxSig,
                          float ped,
                          int fitStatus,
                          float amp,
                          int maxTimebin);

  /// \brief Fit results object in case the raw fit went into error
  /// \param fitError Error type of the raw fit
  explicit CaloFitResults(RawFitterError_t fitError);

  /// \brief minimum interface
  explicit CaloFitResults(int maxSig, int minSig);

  ~CaloFitResults() = default;

  void setMaxSig(unsigned short maxSig) { mMaxSig = maxSig; }
  void setPed(float ped) { mPed = ped; }
  void setMinSig(unsigned short minSig) { mMinSig = minSig; }
  void setStatus(int status) { mStatus = status; }
  void setTime(float time) { mTime = time; }
  void setAmp(float amp) { mAmpSig = amp; }
  void setMaxTimeBin(int timebin) { mMaxTimebin = timebin; }
  void setChi2(float chi2) { mChi2Sig = chi2; }
  void setNdf(unsigned short ndf) { mNdfSig = ndf; }

  bool isFitOK() const { return mFitError == RawFitterError_t::NO_ERROR; }
  RawFitterError_t getFitError() const { return mFitError; }
  unsigned short getMaxSig() const
  {
    checkFitError();
    return mMaxSig;
  }
  float getPed() const
  {
    checkFitError();
    return mPed;
  }
  unsigned short getMinSig() const
  {
    checkFitError();
    return mMinSig;
  }
  int getStatus() const
  {
    checkFitError();
    return mStatus;
  }
  float getAmp() const
  {
    checkFitError();
    return mAmpSig;
  }
  double getTime() const
  {
    checkFitError();
    return mTime;
  }
  int getMaxTimeBin() const
  {
    checkFitError();
    return mMaxTimebin;
  }
  float getChi2() const
  {
    checkFitError();
    return mChi2Sig;
  }
  unsigned short getNdf() const
  {
    checkFitError();
    return mNdfSig;
  }

 private:
  /// \brief Throw fit error in case the fit went into an eror status
  /// \throw RawFitterError_t in case the fit went into error and users try to access fit results nevertheless
  void checkFitError() const
  {
    if (mFitError != RawFitterError_t::NO_ERROR)
      throw mFitError;
  }

  RawFitterError_t mFitError = RawFitterError_t::NO_ERROR; ///< Error of the raw fitter process
  unsigned short mMaxSig = 0;                              ///< Maximum sample value ( 0 - 1023 )
  float mPed = -1;                                         ///< Pedestal
  int mStatus = -1;                                        ///< Sucess or failure of fitting pocedure
  float mAmpSig = -1;                                      ///< Amplitude in entities of ADC counts
  double mTime = -1;                                       ///< Peak/max time of signal in entities of sample intervals
  int mMaxTimebin = -1;                                    ///< Timebin with maximum ADC value
  float mChi2Sig = -1;                                     ///< Chi Square of fit
  unsigned short mNdfSig = 0;                              ///< Number of degrees of freedom of fit
  unsigned short mMinSig = 0;                              ///< Pedestal
};

} // namespace emcal

} // namespace o2
#endif
