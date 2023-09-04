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
  AlignableDetectorITS() = default; // RS FIXME do we need default c-tor?
  AlignableDetectorITS(Controller* ctr);
  ~AlignableDetectorITS() override = default;
  //
  // virtual void initGeom() final;
  void defineVolumes() final;
  //
  // RSTODO
  //  bool AcceptTrack(const AliESDtrack* trc, int trtype) const;

  int processPoints(GIndex gid, int npntCut, bool inv) final;
  bool prepareDetectorData() final;

  void SetAddErrorLr(int ilr, double sigY, double sigZ);
  void SetSkipLr(int ilr);
  //
  void updatePointByTrackInfo(AlignmentPoint* pnt, const trackParam_t* t) const override;
  void setUseErrorParam(int v = 0) override;
  //
  void setITSDictionary(const o2::itsmft::TopologyDictionary* d) { mITSDict = d; }
  //
  void Print(const Option_t* opt = "") const override;
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
  //
  ClassDefOverride(AlignableDetectorITS, 1);
};
} // namespace align
} // namespace o2
#endif
