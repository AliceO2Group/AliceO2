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

/// \file CaloFitResults.cxx
/// \author Hadi Hassan (hadi.hassan@cern.ch)

#include "FairLogger.h"
#include "TMath.h"

#include "EMCALReconstruction/CaloFitResults.h"
#include "DataFormatsEMCAL/Constants.h"

using namespace o2::emcal;

std::string CaloFitResults::createErrorMessage(CaloFitResults::RawFitterError_t errorcode)
{
  switch (errorcode) {
    case RawFitterError_t::SAMPLE_UNINITIALIZED:
      return "Sample for fit not initialzied or bunch length is 0";
    case RawFitterError_t::FIT_ERROR:
      return "Fit of the raw bunch was not successful";
    case RawFitterError_t::CHI2_ERROR:
      return "Chi2 of the fit could not be determined";
    case RawFitterError_t::BUNCH_NOT_OK:
      return "Calo bunch could not be selected";
    case RawFitterError_t::LOW_SIGNAL:
      return "No ADC value above threshold found";
  };
  // Silence compiler warnings for false positives
  // can never enter here due to usage of enum class
  return "Unknown error code";
}

int CaloFitResults::getErrorNumber(CaloFitResults::RawFitterError_t fiterror)
{
  switch (fiterror) {
    case RawFitterError_t::SAMPLE_UNINITIALIZED:
      return 0;
    case RawFitterError_t::FIT_ERROR:
      return 1;
    case RawFitterError_t::CHI2_ERROR:
      return 2;
    case RawFitterError_t::BUNCH_NOT_OK:
      return 3;
    case RawFitterError_t::LOW_SIGNAL:
      return 4;
  };
  // Silence compiler warnings for false positives
  // can never enter here due to usage of enum class
  return -1;
}

CaloFitResults::CaloFitResults(unsigned short maxSig, float ped,
                               int fitstatus, float amp,
                               double time, int maxTimebin,
                               float chi, unsigned short ndf, unsigned short minSig) : mMaxSig(maxSig),
                                                                                       mPed(ped),
                                                                                       mStatus(fitstatus),
                                                                                       mAmpSig(amp),
                                                                                       mTime(time),
                                                                                       mMaxTimebin(maxTimebin),
                                                                                       mChi2Sig(chi),
                                                                                       mNdfSig(ndf),
                                                                                       mMinSig(minSig)
{
}

CaloFitResults::CaloFitResults(unsigned short maxSig, float ped,
                               int fitstatus, float amp,
                               int maxTimebin) : mMaxSig(maxSig),
                                                 mPed(ped),
                                                 mStatus(fitstatus),
                                                 mAmpSig(amp),
                                                 mTime(maxTimebin),
                                                 mMaxTimebin(maxTimebin)
{
}

CaloFitResults::CaloFitResults(int maxSig, int minSig) : mMaxSig(maxSig),
                                                         mMinSig(minSig)
{
}

CaloFitResults::CaloFitResults(RawFitterError_t fitError) : mFitError(fitError)
{
}