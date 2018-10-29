// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
///
/// \file TrackerTraitsNV.h
/// \brief
///

#ifndef TRACKINGITSU_INCLUDE_TRACKERTRAITSNV_H_
#define TRACKINGITSU_INCLUDE_TRACKERTRAITSNV_H_

#include "ITStracking/Configuration.h"
#include "ITStracking/Definitions.h"
#include "ITStracking/TrackerTraits.h"

namespace o2
{
namespace ITS
{

class PrimaryVertexContext;

class TrackerTraitsNV : public TrackerTraits
{
 public:

  virtual ~TrackerTraitsNV() {}

  void computeLayerTracklets(PrimaryVertexContext*, int iteration = 0) final;
  void computeLayerCells(PrimaryVertexContext*, int iteration = 0) final;

};

}
}

#endif /* TRACKINGITSU_INCLUDE_TRACKERTRAITS_H_ */
