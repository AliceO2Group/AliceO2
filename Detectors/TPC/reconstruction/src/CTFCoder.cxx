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
void CTFCoder::createCoders(const std::string& dictPath, o2::ctf::CTFCoderBase::OpType op)
{
  using namespace detail;

  bool mayFail = true; // RS FIXME if the dictionary file is not there, do not produce exception
  auto buff = readDictionaryFromFile<CTF>(dictPath, mayFail);
  if (!buff.size()) {
    if (mayFail) {
      return;
    }
    throw std::runtime_error("Failed to create CTF dictionaty");
  }
  const CTF::container_t* ctf = CTF::get(buff.data());
  mCombineColumns = ctf->getHeader().flags & CTFHeader::CombinedColumns;
  LOG(INFO) << "TPC CTF Columns Combining " << (mCombineColumns ? "ON" : "OFF");

  const CompressedClusters cc; // just to get member types
  if (mCombineColumns) {
    buildCoder<combinedType_t<CTF::NBitsQTot, CTF::NBitsQMax>>(op, *ctf, CTF::BLCqTotA);
  } else {
    buildCoder<std::remove_pointer_t<decltype(cc.qTotA)>>(op, *ctf, CTF::BLCqTotA);
  }
  buildCoder<std::remove_pointer_t<decltype(cc.qMaxA)>>(op, *ctf, CTF::BLCqMaxA);
  buildCoder<std::remove_pointer_t<decltype(cc.flagsA)>>(op, *ctf, CTF::BLCflagsA);
  if (mCombineColumns) {
    buildCoder<combinedType_t<CTF::NBitsRowDiff, CTF::NBitsSliceLegDiff>>(op, *ctf, CTF::BLCrowDiffA); // merged rowDiffA and sliceLegDiffA

  } else {
    buildCoder<std::remove_pointer_t<decltype(cc.rowDiffA)>>(op, *ctf, CTF::BLCrowDiffA);
  }
  buildCoder<std::remove_pointer_t<decltype(cc.sliceLegDiffA)>>(op, *ctf, CTF::BLCsliceLegDiffA);
  buildCoder<std::remove_pointer_t<decltype(cc.padResA)>>(op, *ctf, CTF::BLCpadResA);
  buildCoder<std::remove_pointer_t<decltype(cc.timeResA)>>(op, *ctf, CTF::BLCtimeResA);
  if (mCombineColumns) {
    buildCoder<combinedType_t<CTF::NBitsSigmaPad, CTF::NBitsSigmaTime>>(op, *ctf, CTF::BLCsigmaPadA); // merged sigmaPadA and sigmaTimeA
  } else {
    buildCoder<std::remove_pointer_t<decltype(cc.sigmaPadA)>>(op, *ctf, CTF::BLCsigmaPadA);
  }
  buildCoder<std::remove_pointer_t<decltype(cc.sigmaTimeA)>>(op, *ctf, CTF::BLCsigmaTimeA);
  buildCoder<std::remove_pointer_t<decltype(cc.qPtA)>>(op, *ctf, CTF::BLCqPtA);
  buildCoder<std::remove_pointer_t<decltype(cc.rowA)>>(op, *ctf, CTF::BLCrowA);
  buildCoder<std::remove_pointer_t<decltype(cc.sliceA)>>(op, *ctf, CTF::BLCsliceA);
  buildCoder<std::remove_pointer_t<decltype(cc.timeA)>>(op, *ctf, CTF::BLCtimeA);
  buildCoder<std::remove_pointer_t<decltype(cc.padA)>>(op, *ctf, CTF::BLCpadA);
  if (mCombineColumns) {
    buildCoder<combinedType_t<CTF::NBitsQTot, CTF::NBitsQMax>>(op, *ctf, CTF::BLCqTotU); // merged qTotU and qMaxU
  } else {
    buildCoder<std::remove_pointer_t<decltype(cc.qTotU)>>(op, *ctf, CTF::BLCqTotU);
  }
  buildCoder<std::remove_pointer_t<decltype(cc.qMaxU)>>(op, *ctf, CTF::BLCqMaxU);
  buildCoder<std::remove_pointer_t<decltype(cc.flagsU)>>(op, *ctf, CTF::BLCflagsU);
  buildCoder<std::remove_pointer_t<decltype(cc.padDiffU)>>(op, *ctf, CTF::BLCpadDiffU);
  buildCoder<std::remove_pointer_t<decltype(cc.timeDiffU)>>(op, *ctf, CTF::BLCtimeDiffU);
  if (mCombineColumns) {
    buildCoder<combinedType_t<CTF::NBitsSigmaPad, CTF::NBitsSigmaTime>>(op, *ctf, CTF::BLCsigmaPadU); // merged sigmaPadU and sigmaTimeU
  } else {
    buildCoder<std::remove_pointer_t<decltype(cc.sigmaPadU)>>(op, *ctf, CTF::BLCsigmaPadU);
  }
  buildCoder<std::remove_pointer_t<decltype(cc.sigmaTimeU)>>(op, *ctf, CTF::BLCsigmaTimeU);
  buildCoder<std::remove_pointer_t<decltype(cc.nTrackClusters)>>(op, *ctf, CTF::BLCnTrackClusters);
  buildCoder<std::remove_pointer_t<decltype(cc.nSliceRowClusters)>>(op, *ctf, CTF::BLCnSliceRowClusters);
}

/// make sure loaded dictionaries (if any) are consistent with data
void CTFCoder::checkDataDictionaryConsistency(const CTFHeader& h)
{
  if (mCoders[0]) { // if external dictionary is provided (it will set , make sure its columns combining option is the same as
    if (mCombineColumns != (h.flags & CTFHeader::CombinedColumns)) {
      throw std::runtime_error(fmt::format("Mismatch in columns combining mode, Dictionary:{:s} CTFHeader:{:s}",
                                           mCombineColumns ? "ON" : "OFF", (h.flags & CTFHeader::CombinedColumns) ? "ON" : "OFF"));
    }
  } else {
    setCombineColumns(h.flags & CTFHeader::CombinedColumns);
    LOG(INFO) << "CTF with stored dictionaries, columns combining " << (mCombineColumns ? "ON" : "OFF");
  }
}

///________________________________
size_t CTFCoder::estimateCompressedSize(const CompressedClusters& ccl)
{
  size_t sz = 0;
  // clang-format off
  // RS FIXME this is very crude estimate, instead, an empirical values should be used
#define ESTSIZE(slot, ptr, n) mCoders[int(slot)] ? \
    rans::calculateMaxBufferSize(n, reinterpret_cast<const o2::rans::LiteralEncoder64<std::remove_pointer<decltype(ptr)>::type>*>(mCoders[int(slot)].get())->getAlphabetRangeBits(), \
                                 sizeof(std::remove_pointer<decltype(ptr)>::type)) : n*sizeof(std::remove_pointer<decltype(ptr)>)
  sz += ESTSIZE(CTF::BLCqTotA,	           ccl.qTotA,  	          ccl.nAttachedClusters);
  sz += ESTSIZE(CTF::BLCqMaxA,	           ccl.qMaxA,  	          ccl.nAttachedClusters);
  sz += ESTSIZE(CTF::BLCflagsA,	           ccl.flagsA, 	          ccl.nAttachedClusters);
  sz += ESTSIZE(CTF::BLCrowDiffA,	   ccl.rowDiffA,	  ccl.nAttachedClustersReduced);
  sz += ESTSIZE(CTF::BLCsliceLegDiffA,     ccl.sliceLegDiffA,     ccl.nAttachedClustersReduced);
  sz += ESTSIZE(CTF::BLCpadResA,	   ccl.padResA,	          ccl.nAttachedClustersReduced);
  sz += ESTSIZE(CTF::BLCtimeResA,	   ccl.timeResA,	  ccl.nAttachedClustersReduced);
  sz += ESTSIZE(CTF::BLCsigmaPadA,	   ccl.sigmaPadA,	  ccl.nAttachedClusters);
  sz += ESTSIZE(CTF::BLCsigmaTimeA,	   ccl.sigmaTimeA,	  ccl.nAttachedClusters);
  sz += ESTSIZE(CTF::BLCqPtA, 	           ccl.qPtA,		  ccl.nTracks);
  sz += ESTSIZE(CTF::BLCrowA, 	           ccl.rowA,		  ccl.nTracks);
  sz += ESTSIZE(CTF::BLCsliceA,	           ccl.sliceA, 	          ccl.nTracks);
  sz += ESTSIZE(CTF::BLCtimeA,	           ccl.timeA,  	          ccl.nTracks);
  sz += ESTSIZE(CTF::BLCpadA, 	           ccl.padA,		  ccl.nTracks);
  sz += ESTSIZE(CTF::BLCqTotU,	           ccl.qTotU,  	          ccl.nUnattachedClusters);
  sz += ESTSIZE(CTF::BLCqMaxU,	           ccl.qMaxU,  	          ccl.nUnattachedClusters);
  sz += ESTSIZE(CTF::BLCflagsU,	           ccl.flagsU, 	          ccl.nUnattachedClusters);
  sz += ESTSIZE(CTF::BLCpadDiffU,	   ccl.padDiffU,	  ccl.nUnattachedClusters);
  sz += ESTSIZE(CTF::BLCtimeDiffU,	   ccl.timeDiffU,	  ccl.nUnattachedClusters);
  sz += ESTSIZE(CTF::BLCsigmaPadU,	   ccl.sigmaPadU,	  ccl.nUnattachedClusters);
  sz += ESTSIZE(CTF::BLCsigmaTimeU,	   ccl.sigmaTimeU,	  ccl.nUnattachedClusters);
  sz += ESTSIZE(CTF::BLCnTrackClusters,    ccl.nTrackClusters,    ccl.nTracks);
  sz += ESTSIZE(CTF::BLCnSliceRowClusters, ccl.nSliceRowClusters, ccl.nSliceRows);
  // clang-format on
  sz *= 2. / 3; // if needed, will be autoexpanded
  LOG(INFO) << "Estimated output size is " << sz << " bytes";
  return sz;
}
