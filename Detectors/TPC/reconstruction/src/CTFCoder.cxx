// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   CTFCoder.cxx
/// \author ruben.shahoyan@cern.ch
/// \brief class for entropy encoding/decoding of TPC compressed clusters data

#include "TPCReconstruction/CTFCoder.h"

using namespace o2::tpc;

/// estimate size needed to store in the flat buffer the payload of the CompressedClusters (here only the counters are used)
size_t CTFCoder::estimateSize(CompressedClusters& c)
{
  const CompressedClustersCounters& h = c;
  size_t sz = 0;
  sz += alignSize(c.qTotU, h.nUnattachedClusters);
  sz += alignSize(c.qMaxU, h.nUnattachedClusters);
  sz += alignSize(c.flagsU, h.nUnattachedClusters);
  sz += alignSize(c.padDiffU, h.nUnattachedClusters);
  sz += alignSize(c.timeDiffU, h.nUnattachedClusters);
  sz += alignSize(c.sigmaPadU, h.nUnattachedClusters);
  sz += alignSize(c.sigmaTimeU, h.nUnattachedClusters);
  sz += alignSize(c.nSliceRowClusters, h.nSliceRows);

  if (h.nAttachedClusters) {
    sz += alignSize(c.qTotA, h.nAttachedClusters);
    sz += alignSize(c.qMaxA, h.nAttachedClusters);
    sz += alignSize(c.flagsA, h.nAttachedClusters);
    sz += alignSize(c.rowDiffA, h.nAttachedClustersReduced);
    sz += alignSize(c.sliceLegDiffA, h.nAttachedClustersReduced);
    sz += alignSize(c.padResA, h.nAttachedClustersReduced);
    sz += alignSize(c.timeResA, h.nAttachedClustersReduced);
    sz += alignSize(c.sigmaPadA, h.nAttachedClusters);
    sz += alignSize(c.sigmaTimeA, h.nAttachedClusters);

    sz += alignSize(c.qPtA, h.nTracks);
    sz += alignSize(c.rowA, h.nTracks);
    sz += alignSize(c.sliceA, h.nTracks);
    sz += alignSize(c.timeA, h.nTracks);
    sz += alignSize(c.padA, h.nTracks);

    sz += alignSize(c.nTrackClusters, h.nTracks);
  }
  return sz;
}

/// set addresses of the CompressedClusters fields to point on already reserved memory region
void CTFCoder::setCompClusAddresses(CompressedClusters& c, void*& buff)
{
  const CompressedClustersCounters& h = c;
  setAlignedPtr(buff, c.qTotU, h.nUnattachedClusters);
  setAlignedPtr(buff, c.qMaxU, h.nUnattachedClusters);
  setAlignedPtr(buff, c.flagsU, h.nUnattachedClusters);
  setAlignedPtr(buff, c.padDiffU, h.nUnattachedClusters);
  setAlignedPtr(buff, c.timeDiffU, h.nUnattachedClusters);
  setAlignedPtr(buff, c.sigmaPadU, h.nUnattachedClusters);
  setAlignedPtr(buff, c.sigmaTimeU, h.nUnattachedClusters);
  setAlignedPtr(buff, c.nSliceRowClusters, h.nSliceRows);

  if (h.nAttachedClusters) {

    setAlignedPtr(buff, c.qTotA, h.nAttachedClusters);
    setAlignedPtr(buff, c.qMaxA, h.nAttachedClusters);
    setAlignedPtr(buff, c.flagsA, h.nAttachedClusters);
    setAlignedPtr(buff, c.rowDiffA, h.nAttachedClustersReduced);
    setAlignedPtr(buff, c.sliceLegDiffA, h.nAttachedClustersReduced);
    setAlignedPtr(buff, c.padResA, h.nAttachedClustersReduced);
    setAlignedPtr(buff, c.timeResA, h.nAttachedClustersReduced);
    setAlignedPtr(buff, c.sigmaPadA, h.nAttachedClusters);
    setAlignedPtr(buff, c.sigmaTimeA, h.nAttachedClusters);

    setAlignedPtr(buff, c.qPtA, h.nTracks);
    setAlignedPtr(buff, c.rowA, h.nTracks);
    setAlignedPtr(buff, c.sliceA, h.nTracks);
    setAlignedPtr(buff, c.timeA, h.nTracks);
    setAlignedPtr(buff, c.padA, h.nTracks);

    setAlignedPtr(buff, c.nTrackClusters, h.nTracks);
  }
}
