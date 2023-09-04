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

/// \file   CTFCoder.cxx
/// \author ruben.shahoyan@cern.ch
/// \brief class for entropy encoding/decoding of TPC compressed clusters data

#include "TPCReconstruction/CTFCoder.h"
#include <fmt/format.h>

using namespace o2::tpc;

/// estimate size needed to store in the flat buffer the payload of the CompressedClusters (here only the counters are used)
size_t CTFCoder::estimateSize(CompressedClusters& c)
{
  const CompressedClustersCounters& header = c;
  size_t sz = 0;
  sz += alignSize(c.qTotU, header.nUnattachedClusters);
  sz += alignSize(c.qMaxU, header.nUnattachedClusters);
  sz += alignSize(c.flagsU, header.nUnattachedClusters);
  sz += alignSize(c.padDiffU, header.nUnattachedClusters);
  sz += alignSize(c.timeDiffU, header.nUnattachedClusters);
  sz += alignSize(c.sigmaPadU, header.nUnattachedClusters);
  sz += alignSize(c.sigmaTimeU, header.nUnattachedClusters);
  sz += alignSize(c.nSliceRowClusters, header.nSliceRows);

  if (header.nAttachedClusters) {
    sz += alignSize(c.qTotA, header.nAttachedClusters);
    sz += alignSize(c.qMaxA, header.nAttachedClusters);
    sz += alignSize(c.flagsA, header.nAttachedClusters);
    sz += alignSize(c.rowDiffA, header.nAttachedClustersReduced);
    sz += alignSize(c.sliceLegDiffA, header.nAttachedClustersReduced);
    sz += alignSize(c.padResA, header.nAttachedClustersReduced);
    sz += alignSize(c.timeResA, header.nAttachedClustersReduced);
    sz += alignSize(c.sigmaPadA, header.nAttachedClusters);
    sz += alignSize(c.sigmaTimeA, header.nAttachedClusters);

    sz += alignSize(c.qPtA, header.nTracks);
    sz += alignSize(c.rowA, header.nTracks);
    sz += alignSize(c.sliceA, header.nTracks);
    sz += alignSize(c.timeA, header.nTracks);
    sz += alignSize(c.padA, header.nTracks);

    sz += alignSize(c.nTrackClusters, header.nTracks);
  }
  return sz;
}

/// set addresses of the CompressedClusters fields to point on already reserved memory region
void CTFCoder::setCompClusAddresses(CompressedClusters& c, void*& buff)
{
  const CompressedClustersCounters& header = c;
  setAlignedPtr(buff, c.qTotU, header.nUnattachedClusters);
  setAlignedPtr(buff, c.qMaxU, header.nUnattachedClusters);
  setAlignedPtr(buff, c.flagsU, header.nUnattachedClusters);
  setAlignedPtr(buff, c.padDiffU, header.nUnattachedClusters);
  setAlignedPtr(buff, c.timeDiffU, header.nUnattachedClusters);
  setAlignedPtr(buff, c.sigmaPadU, header.nUnattachedClusters);
  setAlignedPtr(buff, c.sigmaTimeU, header.nUnattachedClusters);
  setAlignedPtr(buff, c.nSliceRowClusters, header.nSliceRows);

  if (header.nAttachedClusters) {

    setAlignedPtr(buff, c.qTotA, header.nAttachedClusters);
    setAlignedPtr(buff, c.qMaxA, header.nAttachedClusters);
    setAlignedPtr(buff, c.flagsA, header.nAttachedClusters);
    setAlignedPtr(buff, c.rowDiffA, header.nAttachedClustersReduced);
    setAlignedPtr(buff, c.sliceLegDiffA, header.nAttachedClustersReduced);
    setAlignedPtr(buff, c.padResA, header.nAttachedClustersReduced);
    setAlignedPtr(buff, c.timeResA, header.nAttachedClustersReduced);
    setAlignedPtr(buff, c.sigmaPadA, header.nAttachedClusters);
    setAlignedPtr(buff, c.sigmaTimeA, header.nAttachedClusters);

    setAlignedPtr(buff, c.qPtA, header.nTracks);
    setAlignedPtr(buff, c.rowA, header.nTracks);
    setAlignedPtr(buff, c.sliceA, header.nTracks);
    setAlignedPtr(buff, c.timeA, header.nTracks);
    setAlignedPtr(buff, c.padA, header.nTracks);

    setAlignedPtr(buff, c.nTrackClusters, header.nTracks);
  }
}

///________________________________
void CTFCoder::createCoders(const std::vector<char>& bufVec, o2::ctf::CTFCoderBase::OpType op)
{
  using namespace detail;
  const CTF::container_t ctf = CTF::getImage(bufVec.data());
  mCombineColumns = ctf.getHeader().flags & CTFHeader::CombinedColumns;
  LOG(info) << "TPC CTF Columns Combining " << (mCombineColumns ? "ON" : "OFF");

  const CompressedClusters cc; // just to get member types
  if (mCombineColumns) {
    buildCoder<combinedType_t<CTF::NBitsQTot, CTF::NBitsQMax>>(op, ctf, CTF::BLCqTotA);
  } else {
    buildCoder<std::remove_pointer_t<decltype(cc.qTotA)>>(op, ctf, CTF::BLCqTotA);
  }
  buildCoder<std::remove_pointer_t<decltype(cc.qMaxA)>>(op, ctf, CTF::BLCqMaxA);
  buildCoder<std::remove_pointer_t<decltype(cc.flagsA)>>(op, ctf, CTF::BLCflagsA);
  if (mCombineColumns) {
    buildCoder<combinedType_t<CTF::NBitsRowDiff, CTF::NBitsSliceLegDiff>>(op, ctf, CTF::BLCrowDiffA); // merged rowDiffA and sliceLegDiffA

  } else {
    buildCoder<std::remove_pointer_t<decltype(cc.rowDiffA)>>(op, ctf, CTF::BLCrowDiffA);
  }
  buildCoder<std::remove_pointer_t<decltype(cc.sliceLegDiffA)>>(op, ctf, CTF::BLCsliceLegDiffA);
  buildCoder<std::remove_pointer_t<decltype(cc.padResA)>>(op, ctf, CTF::BLCpadResA);
  buildCoder<std::remove_pointer_t<decltype(cc.timeResA)>>(op, ctf, CTF::BLCtimeResA);
  if (mCombineColumns) {
    buildCoder<combinedType_t<CTF::NBitsSigmaPad, CTF::NBitsSigmaTime>>(op, ctf, CTF::BLCsigmaPadA); // merged sigmaPadA and sigmaTimeA
  } else {
    buildCoder<std::remove_pointer_t<decltype(cc.sigmaPadA)>>(op, ctf, CTF::BLCsigmaPadA);
  }
  buildCoder<std::remove_pointer_t<decltype(cc.sigmaTimeA)>>(op, ctf, CTF::BLCsigmaTimeA);
  buildCoder<std::remove_pointer_t<decltype(cc.qPtA)>>(op, ctf, CTF::BLCqPtA);
  buildCoder<std::remove_pointer_t<decltype(cc.rowA)>>(op, ctf, CTF::BLCrowA);
  buildCoder<std::remove_pointer_t<decltype(cc.sliceA)>>(op, ctf, CTF::BLCsliceA);
  buildCoder<std::remove_pointer_t<decltype(cc.timeA)>>(op, ctf, CTF::BLCtimeA);
  buildCoder<std::remove_pointer_t<decltype(cc.padA)>>(op, ctf, CTF::BLCpadA);
  if (mCombineColumns) {
    buildCoder<combinedType_t<CTF::NBitsQTot, CTF::NBitsQMax>>(op, ctf, CTF::BLCqTotU); // merged qTotU and qMaxU
  } else {
    buildCoder<std::remove_pointer_t<decltype(cc.qTotU)>>(op, ctf, CTF::BLCqTotU);
  }
  buildCoder<std::remove_pointer_t<decltype(cc.qMaxU)>>(op, ctf, CTF::BLCqMaxU);
  buildCoder<std::remove_pointer_t<decltype(cc.flagsU)>>(op, ctf, CTF::BLCflagsU);
  buildCoder<std::remove_pointer_t<decltype(cc.padDiffU)>>(op, ctf, CTF::BLCpadDiffU);
  buildCoder<std::remove_pointer_t<decltype(cc.timeDiffU)>>(op, ctf, CTF::BLCtimeDiffU);
  if (mCombineColumns) {
    buildCoder<combinedType_t<CTF::NBitsSigmaPad, CTF::NBitsSigmaTime>>(op, ctf, CTF::BLCsigmaPadU); // merged sigmaPadU and sigmaTimeU
  } else {
    buildCoder<std::remove_pointer_t<decltype(cc.sigmaPadU)>>(op, ctf, CTF::BLCsigmaPadU);
  }
  buildCoder<std::remove_pointer_t<decltype(cc.sigmaTimeU)>>(op, ctf, CTF::BLCsigmaTimeU);
  buildCoder<std::remove_pointer_t<decltype(cc.nTrackClusters)>>(op, ctf, CTF::BLCnTrackClusters);
  buildCoder<std::remove_pointer_t<decltype(cc.nSliceRowClusters)>>(op, ctf, CTF::BLCnSliceRowClusters);
}

/// make sure loaded dictionaries (if any) are consistent with data
void CTFCoder::checkDataDictionaryConsistency(const CTFHeader& h)
{
  if (mCoders[0].has_value()) { // if external dictionary is provided (it will set , make sure its columns combining option is the same as
    if (mCombineColumns != (h.flags & CTFHeader::CombinedColumns)) {
      throw std::runtime_error(fmt::format("Mismatch in columns combining mode, Dictionary:{:s} CTFHeader:{:s}",
                                           mCombineColumns ? "ON" : "OFF", (h.flags & CTFHeader::CombinedColumns) ? "ON" : "OFF"));
    }
  } else {
    setCombineColumns(h.flags & CTFHeader::CombinedColumns);
    LOG(info) << "CTF with stored dictionaries, columns combining " << (mCombineColumns ? "ON" : "OFF");
  }
}

///________________________________
size_t CTFCoder::estimateCompressedSize(const CompressedClusters& ccl)
{
  using namespace detail;
  size_t sz = 0;

  // RS FIXME this is very crude estimate, instead, an empirical values should be used
  if (mCombineColumns) {
    sz += estimateBufferSize<combinedType_t<CTF::NBitsQTot, CTF::NBitsQMax>>(CTF::BLCqTotA, ccl.nAttachedClusters);
  } else {
    sz += estimateBufferSize(CTF::BLCqTotA, ccl.qTotA, ccl.qTotA + ccl.nAttachedClusters);
  }
  sz += estimateBufferSize(CTF::BLCqMaxA, ccl.qMaxA, ccl.qMaxA + (mCombineColumns ? 0 : ccl.nAttachedClusters));
  sz += estimateBufferSize(CTF::BLCflagsA, ccl.flagsA, ccl.flagsA + ccl.nAttachedClusters);

  if (mCombineColumns) {
    sz += estimateBufferSize<combinedType_t<CTF::NBitsRowDiff, CTF::NBitsSliceLegDiff>>(CTF::BLCrowDiffA, ccl.nAttachedClustersReduced);
  } else {
    sz += estimateBufferSize(CTF::BLCrowDiffA, ccl.rowDiffA, ccl.rowDiffA + ccl.nAttachedClustersReduced);
  }
  sz += estimateBufferSize(CTF::BLCsliceLegDiffA, ccl.sliceLegDiffA, ccl.sliceLegDiffA + (mCombineColumns ? 0 : ccl.nAttachedClustersReduced));
  sz += estimateBufferSize(CTF::BLCpadResA, ccl.padResA, ccl.padResA + ccl.nAttachedClustersReduced);
  sz += estimateBufferSize(CTF::BLCtimeResA, ccl.timeResA, ccl.timeResA + ccl.nAttachedClustersReduced);
  if (mCombineColumns) {
    sz += estimateBufferSize<combinedType_t<CTF::NBitsSigmaPad, CTF::NBitsSigmaTime>>(CTF::BLCsigmaPadA, ccl.nAttachedClusters);
  } else {
    sz += estimateBufferSize(CTF::BLCsigmaPadA, ccl.sigmaPadA, ccl.sigmaPadA + ccl.nAttachedClusters);
  }
  sz += estimateBufferSize(CTF::BLCsigmaTimeA, ccl.sigmaTimeA, ccl.sigmaTimeA + (mCombineColumns ? 0 : ccl.nAttachedClusters));
  sz += estimateBufferSize(CTF::BLCqPtA, ccl.qPtA, ccl.qPtA + ccl.nTracks);
  sz += estimateBufferSize(CTF::BLCrowA, ccl.rowA, ccl.rowA + ccl.nTracks);
  sz += estimateBufferSize(CTF::BLCsliceA, ccl.sliceA, ccl.sliceA + ccl.nTracks);
  sz += estimateBufferSize(CTF::BLCtimeA, ccl.timeA, ccl.timeA + ccl.nTracks);
  sz += estimateBufferSize(CTF::BLCpadA, ccl.padA, ccl.padA + ccl.nTracks);
  if (mCombineColumns) {
    sz += estimateBufferSize<combinedType_t<CTF::NBitsQTot, CTF::NBitsQMax>>(CTF::BLCqTotU, ccl.nUnattachedClusters);
  } else {
    sz += estimateBufferSize(CTF::BLCqTotU, ccl.qTotU, ccl.qTotU + ccl.nUnattachedClusters);
  }
  sz += estimateBufferSize(CTF::BLCqMaxU, ccl.qMaxU, ccl.qMaxU + (mCombineColumns ? 0 : ccl.nUnattachedClusters));
  sz += estimateBufferSize(CTF::BLCflagsU, ccl.flagsU, ccl.flagsU + ccl.nUnattachedClusters);
  sz += estimateBufferSize(CTF::BLCpadDiffU, ccl.padDiffU, ccl.padDiffU + ccl.nUnattachedClusters);
  sz += estimateBufferSize(CTF::BLCtimeDiffU, ccl.timeDiffU, ccl.timeDiffU + ccl.nUnattachedClusters);
  if (mCombineColumns) {
    sz += estimateBufferSize<combinedType_t<CTF::NBitsSigmaPad, CTF::NBitsSigmaTime>>(CTF::BLCsigmaPadU, ccl.nUnattachedClusters);
  } else {
    sz += estimateBufferSize(CTF::BLCsigmaPadU, ccl.sigmaPadU, ccl.sigmaPadU + ccl.nUnattachedClusters);
  }
  sz += estimateBufferSize(CTF::BLCsigmaTimeU, ccl.sigmaTimeU, ccl.sigmaTimeU + (mCombineColumns ? 0 : ccl.nUnattachedClusters));
  sz += estimateBufferSize(CTF::BLCnTrackClusters, ccl.nTrackClusters, ccl.nTrackClusters + ccl.nTracks);
  sz += estimateBufferSize(CTF::BLCnSliceRowClusters, ccl.nSliceRowClusters, ccl.nSliceRowClusters + ccl.nSliceRows);

  sz *= 2. / 3; // if needed, will be autoexpanded
  LOG(info) << "Estimated output size is " << sz << " bytes";
  return sz;
}
