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
  bool mayFail = true; // RS FIXME if the dictionary file is not there, do not produce exception
  auto buff = readDictionaryFromFile<CTF>(dictPath, o2::detectors::DetID::TPC, mayFail);
  if (!buff.size()) {
    if (mayFail) {
      return;
    }
    throw std::runtime_error("Failed to create CTF dictionaty");
  }
  const auto* ctf = CTF::get(buff.data());

  auto getFreq = [ctf](CTF::Slots slot) -> o2::rans::FrequencyTable {
    o2::rans::FrequencyTable ft;
    auto bl = ctf->getBlock(slot);
    auto md = ctf->getMetadata(slot);
    ft.addFrequencies(bl.getDict(), bl.getDict() + bl.getNDict(), md.min, md.max);
    return std::move(ft);
  };
  auto getProbBits = [ctf](CTF::Slots slot) -> int {
    return ctf->getMetadata(slot).probabilityBits;
  };

  CompressedClusters cc; // just to get member types
#define MAKECODER(part, slot) createCoder<std::remove_pointer<decltype(part)>::type>(op, getFreq(slot), getProbBits(slot), int(slot))
  // clang-format off
  MAKECODER(cc.qTotA,             CTF::BLCqTotA);
  MAKECODER(cc.qMaxA,             CTF::BLCqMaxA);
  MAKECODER(cc.flagsA,            CTF::BLCflagsA);
  MAKECODER(cc.rowDiffA,          CTF::BLCrowDiffA);
  MAKECODER(cc.sliceLegDiffA,     CTF::BLCsliceLegDiffA);
  MAKECODER(cc.padResA,           CTF::BLCpadResA);
  MAKECODER(cc.timeResA,          CTF::BLCtimeResA);
  MAKECODER(cc.sigmaPadA,         CTF::BLCsigmaPadA);
  MAKECODER(cc.sigmaTimeA,        CTF::BLCsigmaTimeA);
  MAKECODER(cc.qPtA,              CTF::BLCqPtA);
  MAKECODER(cc.rowA,              CTF::BLCrowA);
  MAKECODER(cc.sliceA,            CTF::BLCsliceA);
  MAKECODER(cc.timeA,             CTF::BLCtimeA);
  MAKECODER(cc.padA,              CTF::BLCpadA);
  MAKECODER(cc.qTotU,             CTF::BLCqTotU);
  MAKECODER(cc.qMaxU,             CTF::BLCqMaxU);
  MAKECODER(cc.flagsU,            CTF::BLCflagsU);
  MAKECODER(cc.padDiffU,          CTF::BLCpadDiffU);
  MAKECODER(cc.timeDiffU,         CTF::BLCtimeDiffU);
  MAKECODER(cc.sigmaPadU,         CTF::BLCsigmaPadU);
  MAKECODER(cc.sigmaTimeU,        CTF::BLCsigmaTimeU);
  MAKECODER(cc.nTrackClusters,    CTF::BLCnTrackClusters);
  MAKECODER(cc.nSliceRowClusters, CTF::BLCnSliceRowClusters);
  // clang-format on
}
