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

/// @file   AlignableDetectorITS.h
/// @author ruben.shahoyan@cern.ch, michael.lettrich@cern.ch
/// @since  2021-02-01
/// @brief  ITS detector wrapper

#ifndef ALIGNABLEDETECTORITS_H
#define ALIGNABLEDETECTORITS_H

#include "Align/AlignableDetector.h"
#include "Align/utils.h"
#include "ReconstructionDataFormats/TrackParametrizationWithError.h"

namespace o2
{
namespace align
{

class Controller;

class AlignableDetectorITS : public AlignableDetector
{
 public:
  //
  enum ITSSel_t { kSPDNoSel,
                  kSPDBoth,
                  kSPDAny,
                  kSPD0,
                  kSPD1,
                  kNSPDSelTypes };
  //
  AlignableDetectorITS() = default;
  AlignableDetectorITS(Controller* ctr);
  ~AlignableDetectorITS() override = default;
  //
  void defineVolumes() override;
  //
  // RSTODO
  //  bool AcceptTrack(const AliESDtrack* trc, int trtype) const;

  void SetAddErrorLr(int ilr, double sigY, double sigZ);
  void SetSkipLr(int ilr);
  //
  void updatePointByTrackInfo(AlignmentPoint* pnt, const trackParam_t* t) const override;
  void setUseErrorParam(int v = 1) override;
  void SetITSSelPattern(int trtype, ITSSel_t sel) { fITSPatt[trtype] = sel; }
  void SetITSSelPatternColl(ITSSel_t sel = kSPDAny) { SetITSSelPattern(utils::Coll, sel); }
  void SetITSSelPatternCosm(ITSSel_t sel = kSPDNoSel) { SetITSSelPattern(utils::Cosm, sel); }

  int GetITSSelPattern(int tp) const { return fITSPatt[tp]; }
  int GetITSSelPatternColl() const { return fITSPatt[utils::Coll]; }
  int GetITSSelPatternCosm() const { return fITSPatt[utils::Cosm]; }
  //
  void Print(const Option_t* opt = "") const override;
  //
  static const char* GetITSPattName(int sel) { return sel < kNSPDSelTypes ? fgkHitsSel[sel] : nullptr; }
  //
 protected:
  //
  // -------- dummies --------
  AlignableDetectorITS(const AlignableDetectorITS&);
  AlignableDetectorITS& operator=(const AlignableDetectorITS&);
  //
 protected:
  //
  int fITSPatt[utils::NTrackTypes]; // ITS hits selection pattern for coll/cosm tracks
  //
  static const char* fgkHitsSel[kNSPDSelTypes]; // ITS selection names
  //
  ClassDefOverride(AlignableDetectorITS, 1);
};
} // namespace align
} // namespace o2
#endif
