// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

namespace o2
{
namespace align
{

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
  AlignableDetectorITS(const char* title = "");
  virtual ~AlignableDetectorITS();
  //
  virtual void defineVolumes();
  //
  bool AcceptTrack(const AliESDtrack* trc, int trtype) const;

  void SetAddErrorLr(int ilr, double sigY, double sigZ);
  void SetSkipLr(int ilr);
  //
  virtual void UpdatePointByTrackInfo(AlignmentPoint* pnt, const AliExternalTrackParam* t) const;
  virtual void setUseErrorParam(int v = 1);
  void SetITSSelPattern(int trtype, ITSSel_t sel) { fITSPatt[trtype] = sel; }
  void SetITSSelPatternColl(ITSSel_t sel = kSPDAny) { SetITSSelPattern(utils::kColl, sel); }
  void SetITSSelPatternCosm(ITSSel_t sel = kSPDNoSel) { SetITSSelPattern(utils::kCosm, sel); }

  int GetITSSelPattern(int tp) const { return fITSPatt[tp]; }
  int GetITSSelPatternColl() const { return fITSPatt[utils::kColl]; }
  int GetITSSelPatternCosm() const { return fITSPatt[utils::kCosm]; }
  //
  virtual void Print(const Option_t* opt = "") const;
  //
  static bool CheckHitPattern(const AliESDtrack* trc, int sel);
  static const char* GetITSPattName(int sel) { return sel < kNSPDSelTypes ? fgkHitsSel[sel] : 0; }
  //
 protected:
  //
  void GetErrorParamAngle(int layer, double tgl, double tgphitr, double& erry, double& errz) const;
  //
  // -------- dummies --------
  AlignableDetectorITS(const AlignableDetectorITS&);
  AlignableDetectorITS& operator=(const AlignableDetectorITS&);
  //
 protected:
  //
  int fITSPatt[utils::kNTrackTypes]; // ITS hits selection pattern for coll/cosm tracks
  //
  static const char* fgkHitsSel[kNSPDSelTypes]; // ITS selection names
  //
  ClassDef(AlignableDetectorITS, 1);
};
} // namespace align
} // namespace o2
#endif
