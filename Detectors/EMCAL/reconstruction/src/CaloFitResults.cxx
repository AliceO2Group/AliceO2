// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

///\brief Constructor, shorter interface when no fit is done
//_____________________________________________________________________
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

///
/// Constructor, minimal interface
//_____________________________________________________________________
CaloFitResults::CaloFitResults(int maxSig, int minSig) : mMaxSig(maxSig),
                                                         mMinSig(minSig)
{
}
