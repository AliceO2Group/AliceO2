// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

// HF Configurable Classes
//
// Authors: Nima Zardoshti

#ifndef O2_ANALYSIS_HFCONFIGURABLES_H
#define O2_ANALYSIS_HFCONFIGURABLES_H

#include <TMath.h>

class HFTrackIndexSkimsCreatorConfigs
{
 public:
  HFTrackIndexSkimsCreatorConfigs() = default;
  ~HFTrackIndexSkimsCreatorConfigs() = default;

  // 2-prong cuts D0
  double mPtD0Min = 0.;
  double mInvMassD0Min = 1.46;
  double mInvMassD0Max = 2.26;
  double mCPAD0Min = 0.75;
  double mImpParProductD0Max = -0.00005;
  // 2-prong cuts Jpsi
  double mPtJpsiMin = 0.;
  double mInvMassJpsiMin = 2.5;
  double mInvMassJpsiMax = 4.1;
  double mCPAJpsiMin = -2;
  double mImpParProductJpsiMax = 1000.;
  // 3-prong cuts - D+
  double mPtDPlusMin = 1.;        //original value 0.
  double mInvMassDPlusMin = 1.75; //original value 1.7
  double mInvMassDPlusMax = 2.0;  //original value 2.05
  double mCPADPlusMin = 0.5;
  double mDecLenDPlusMin = 0.;
  // 3-prong cuts - Lc
  double mPtLcMin = 1.;        //original value 0.
  double mInvMassLcMin = 2.15; //original value 2.1
  double mInvMassLcMax = 2.45; //original value 2.5
  double mCPALcMin = 0.5;
  double mDecLenLcMin = 0.;
  // 3-prong cuts - Ds
  double mPtDsMin = 1.;        //original value 0.
  double mInvMassDsMin = 1.75; //original value 1.7
  double mInvMassDsMax = 2.15; //original value 2.2
  double mCPADsMin = 0.5;
  double mDecLenDsMin = 0.;

 private:
  ClassDef(HFTrackIndexSkimsCreatorConfigs, 1);
};

#endif
