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
#include "ReconstructionDataFormats/BaseCluster.h"

namespace o2
{
namespace itsmft
{
class TopologyDictionary;
}

namespace align
{

class Controller;

class AlignableDetectorITS : public AlignableDetector
{
 public:
  //
  using ClusterD = o2::BaseCluster<double>;
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
  // virtual void initGeom() final;
  void defineVolumes() final;
  //
  // RSTODO
  //  bool AcceptTrack(const AliESDtrack* trc, int trtype) const;

  int processPoints(GIndex gid, bool inv) final;
  bool prepareDetectorData() final;

  void SetAddErrorLr(int ilr, double sigY, double sigZ);
  void SetSkipLr(int ilr);
  //
  void updatePointByTrackInfo(AlignmentPoint* pnt, const trackParam_t* t) const override;
  void setUseErrorParam(int v = 0) override;
  void SetITSSelPattern(int trtype, ITSSel_t sel) { fITSPatt[trtype] = sel; }
  void SetITSSelPatternColl(ITSSel_t sel = kSPDAny) { SetITSSelPattern(utils::Coll, sel); }
  void SetITSSelPatternCosm(ITSSel_t sel = kSPDNoSel) { SetITSSelPattern(utils::Cosm, sel); }

  int GetITSSelPattern(int tp) const { return fITSPatt[tp]; }
  int GetITSSelPatternColl() const { return fITSPatt[utils::Coll]; }
  int GetITSSelPatternCosm() const { return fITSPatt[utils::Cosm]; }
  //
  void setITSDictionary(const o2::itsmft::TopologyDictionary* d) { mITSDict = d; }
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
  std::vector<ClusterD> mITSClustersArray;
  const o2::itsmft::TopologyDictionary* mITSDict{nullptr}; // cluster patterns dictionary

  int fITSPatt[utils::NTrackTypes]; // ITS hits selection pattern for coll/cosm tracks
  //
  static const char* fgkHitsSel[kNSPDSelTypes]; // ITS selection names
  //
  ClassDefOverride(AlignableDetectorITS, 1);
};
} // namespace align
} // namespace o2
#endif
