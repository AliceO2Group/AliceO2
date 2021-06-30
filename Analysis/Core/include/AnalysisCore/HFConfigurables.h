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

/// \file HFConfigurables.h
/// \brief Heavy-flavour candidate preselection configurables for HFTrackIndexSkimsCreator
///
/// \author Nima Zardoshti <nima.zardoshti@cern.ch>, CERN

#ifndef O2_ANALYSIS_HFCONFIGURABLES_H
#define O2_ANALYSIS_HFCONFIGURABLES_H

#include <Rtypes.h>

class HFTrackIndexSkimsCreatorConfigs
{
 public:
  HFTrackIndexSkimsCreatorConfigs() = default;
  ~HFTrackIndexSkimsCreatorConfigs() = default;

  // 2-prong cuts

  // D0ToPiK
  double mPtD0ToPiKMin = 4.; //original value 0.
  double mInvMassD0ToPiKMin = 1.46;
  double mInvMassD0ToPiKMax = 2.26;
  double mCPAD0ToPiKMin = 0.75;
  double mImpParProductD0ToPiKMax = -0.00005;
  // JpsiToEE
  double mPtJpsiToEEMin = 4.; //original value 0.
  double mInvMassJpsiToEEMin = 2.5;
  double mInvMassJpsiToEEMax = 4.1;
  double mCPAJpsiToEEMin = -2;
  double mImpParProductJpsiToEEMax = 1000.;

  // 3-prong cuts

  // DPlusToPiKPi
  double mPtDPlusToPiKPiMin = 4.;        //original value 0.
  double mInvMassDPlusToPiKPiMin = 1.75; //original value 1.7
  double mInvMassDPlusToPiKPiMax = 2.0;  //original value 2.05
  double mCPADPlusToPiKPiMin = 0.5;
  double mDecLenDPlusToPiKPiMin = 0.;
  // LcToPKPi
  double mPtLcToPKPiMin = 4.;        //original value 0.
  double mInvMassLcToPKPiMin = 2.15; //original value 2.1
  double mInvMassLcToPKPiMax = 2.45; //original value 2.5
  double mCPALcToPKPiMin = 0.5;
  double mDecLenLcToPKPiMin = 0.;
  // DsToPiKK
  double mPtDsToPiKKMin = 4.;        //original value 0.
  double mInvMassDsToPiKKMin = 1.75; //original value 1.7
  double mInvMassDsToPiKKMax = 2.15; //original value 2.2
  double mCPADsToPiKKMin = 0.5;
  double mDecLenDsToPiKKMin = 0.;
  // XicToPKPi
  double mPtXicToPKPiMin = 1.;        //
  double mInvMassXicToPKPiMin = 2.25; //
  double mInvMassXicToPKPiMax = 2.70; //
  double mCPAXicToPKPiMin = 0.5;
  double mDecLenXicToPKPiMin = 0.;

 private:
  ClassDef(HFTrackIndexSkimsCreatorConfigs, 1);
};

#endif
