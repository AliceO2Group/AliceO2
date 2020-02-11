// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
  /// \brief Default constructor
  CaloFitResults() = default;

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

  /// \brief minimum interface
  explicit CaloFitResults(int maxSig, int minSig);

  ~CaloFitResults() = default;

  unsigned short getMaxSig() const { return mMaxSig; }
  float getPed() const { return mPed; }
  unsigned short getMinSig() const { return mMinSig; }
  int getStatus() const { return mStatus; }
  float getAmp() const { return mAmpSig; }
  float getMaxTimeBin() const { return mMaxTimebin; }
  double getTime() const { return mTime; }
  int getMaxTimebin() const { return mMaxTimebin; }
  float getChi2() const { return mChi2Sig; }
  unsigned short getNdf() const { return mNdfSig; }

  void setTime(float time) { mTime = time; }
  void setAmp(float amp) { mAmpSig = amp; }

 private:
  unsigned short mMaxSig = 0; ///< Maximum sample value ( 0 - 1023 )
  float mPed = -1;            ///< Pedestal
  int mStatus = -1;           ///< Sucess or failure of fitting pocedure
  float mAmpSig = -1;         ///< Amplitude in entities of ADC counts
  double mTime = -1;          ///< Peak/max time of signal in entities of sample intervals
  int mMaxTimebin = -1;       ///< Timebin with maximum ADC value
  float mChi2Sig = -1;        ///< Chi Square of fit
  unsigned short mNdfSig = 0; ///< Number of degrees of freedom of fit
  unsigned short mMinSig = 0; ///< Pedestal
};

} // namespace emcal

} // namespace o2
#endif
