// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   CTFCoder.h
/// \author ruben.shahoyan@cern.ch
/// \brief class for entropy encoding/decoding of TPC compressed clusters data

#ifndef O2_TPC_CTFCODER_H
#define O2_TPC_CTFCODER_H

#include <algorithm>
#include <iterator>
#include <string>
#include "DataFormatsTPC/CTF.h"
#include "DataFormatsTPC/CompressedClusters.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "rANS/rans.h"

class TTree;

namespace o2
{
namespace tpc
{

class CTFCoder
{
 public:
  /// entropy-encode compressed clusters to flat buffer
  template <typename VEC>
  static void encode(VEC& buff, const CompressedClusters& ccl);

  template <typename VEC>
  static void decode(const CTF::base& ec, VEC& buff);

  static size_t constexpr Alignment = 16;
  static size_t estimateSize(CompressedClusters& c);
  static void setCompClusAddresses(CompressedClusters& c, void*& buff);

  template <size_t ALG = Alignment, typename T>
  static size_t alignSize(T*& var, size_t n = 1)
  {
    auto sz = sizeof(T) * n;
    auto res = sz % ALG;
    return res ? sz + (ALG - res) : sz;
  }

  template <size_t ALG = Alignment, typename T>
  static void setAlignedPtr(void*& ptr, T*& var, size_t n = 1)
  {
    auto sz = sizeof(T) * n;
    auto res = sz % ALG;
    var = reinterpret_cast<T*>(ptr);
    auto& ptrR = reinterpret_cast<size_t&>(ptr);
    ptrR += res ? sz + (ALG - res) : sz;
  }

  ClassDefNV(CTFCoder, 1);
};

/// entropy-encode clusters to buffer with CTF
template <typename VEC>
void CTFCoder::encode(VEC& buff, const CompressedClusters& ccl)
{
  using MD = o2::ctf::Metadata::OptStore;
  // what to do which each field: see o2::ctf::Metadata explanation
  constexpr MD optField[CTF::getNBlocks()] = {
    MD::EENCODE, //qTotA
    MD::EENCODE, //qMaxA
    MD::EENCODE, //flagsA
    MD::EENCODE, //rowDiffA
    MD::EENCODE, //sliceLegDiffA
    MD::EENCODE, //padResA
    MD::EENCODE, //timeResA
    MD::EENCODE, //sigmaPadA
    MD::EENCODE, //sigmaTimeA
    MD::EENCODE, //qPtA
    MD::EENCODE, //rowA
    MD::EENCODE, //sliceA
    MD::EENCODE, //timeA
    MD::EENCODE, //padA
    MD::EENCODE, //qTotU
    MD::EENCODE, //qMaxU
    MD::EENCODE, //flagsU
    MD::EENCODE, //padDiffU
    MD::EENCODE, //timeDiffU
    MD::EENCODE, //sigmaPadU
    MD::EENCODE, //sigmaTimeU
    MD::EENCODE, //nTrackClusters
    MD::EENCODE  //nSliceRowClusters
  };

  auto ec = CTF::create(buff);
  ec->setHeader(reinterpret_cast<const CompressedClustersCounters&>(ccl));
  ec->getANSHeader().majorVersion = 0;
  ec->getANSHeader().minorVersion = 1;
  // at every encoding the buffer might be autoexpanded, so we don't work with fixed pointer ec
#define ENCODE CTF::get(buff.data())->encode
  // clang-format off
  ENCODE(ccl.qTotA,             ccl.qTotA + ccl.nAttachedClusters,                CTF::BLCqTotA,             o2::rans::ProbabilityBits16Bit, optField[CTF::BLCqTotA],             &buff);
  ENCODE(ccl.qMaxA,             ccl.qMaxA + ccl.nAttachedClusters,                CTF::BLCqMaxA,             o2::rans::ProbabilityBits16Bit, optField[CTF::BLCqMaxA],             &buff);
  ENCODE(ccl.flagsA,            ccl.flagsA + ccl.nAttachedClusters,               CTF::BLCflagsA,            o2::rans::ProbabilityBits8Bit,  optField[CTF::BLCflagsA],            &buff);
  ENCODE(ccl.rowDiffA,          ccl.rowDiffA + ccl.nAttachedClustersReduced,      CTF::BLCrowDiffA,          o2::rans::ProbabilityBits8Bit,  optField[CTF::BLCrowDiffA],          &buff);
  ENCODE(ccl.sliceLegDiffA,     ccl.sliceLegDiffA + ccl.nAttachedClustersReduced, CTF::BLCsliceLegDiffA,     o2::rans::ProbabilityBits8Bit,  optField[CTF::BLCsliceLegDiffA],     &buff);
  ENCODE(ccl.padResA,           ccl.padResA + ccl.nAttachedClustersReduced,       CTF::BLCpadResA,           o2::rans::ProbabilityBits16Bit, optField[CTF::BLCpadResA],           &buff);
  ENCODE(ccl.timeResA,          ccl.timeResA + ccl.nAttachedClustersReduced,      CTF::BLCtimeResA,          o2::rans::ProbabilityBits25Bit, optField[CTF::BLCtimeResA],          &buff);
  ENCODE(ccl.sigmaPadA,         ccl.sigmaPadA + ccl.nAttachedClusters,            CTF::BLCsigmaPadA,         o2::rans::ProbabilityBits8Bit,  optField[CTF::BLCsigmaPadA],         &buff);
  ENCODE(ccl.sigmaTimeA,        ccl.sigmaTimeA + ccl.nAttachedClusters,           CTF::BLCsigmaTimeA,        o2::rans::ProbabilityBits8Bit,  optField[CTF::BLCsigmaTimeA],        &buff);  
  ENCODE(ccl.qPtA,              ccl.qPtA + ccl.nTracks,                           CTF::BLCqPtA,              o2::rans::ProbabilityBits8Bit,  optField[CTF::BLCqPtA],              &buff);
  ENCODE(ccl.rowA,              ccl.rowA + ccl.nTracks,                           CTF::BLCrowA,              o2::rans::ProbabilityBits8Bit,  optField[CTF::BLCrowA],              &buff);
  ENCODE(ccl.sliceA,            ccl.sliceA + ccl.nTracks,                         CTF::BLCsliceA,            o2::rans::ProbabilityBits8Bit,  optField[CTF::BLCsliceA],            &buff);
  ENCODE(ccl.timeA,             ccl.timeA + ccl.nTracks,                          CTF::BLCtimeA,             o2::rans::ProbabilityBits25Bit, optField[CTF::BLCtimeA],             &buff);
  ENCODE(ccl.padA,              ccl.padA + ccl.nTracks,                           CTF::BLCpadA,              o2::rans::ProbabilityBits16Bit, optField[CTF::BLCpadA],              &buff);
  ENCODE(ccl.qTotU,             ccl.qTotU + ccl.nUnattachedClusters,              CTF::BLCqTotU,             o2::rans::ProbabilityBits16Bit, optField[CTF::BLCqTotU],             &buff);
  ENCODE(ccl.qMaxU,             ccl.qMaxU + ccl.nUnattachedClusters,              CTF::BLCqMaxU,             o2::rans::ProbabilityBits16Bit, optField[CTF::BLCqMaxU],             &buff);
  ENCODE(ccl.flagsU,            ccl.flagsU + ccl.nUnattachedClusters,             CTF::BLCflagsU,            o2::rans::ProbabilityBits8Bit,  optField[CTF::BLCflagsU],            &buff);
  ENCODE(ccl.padDiffU,          ccl.padDiffU + ccl.nUnattachedClusters,           CTF::BLCpadDiffU,          o2::rans::ProbabilityBits16Bit, optField[CTF::BLCpadDiffU],          &buff);
  ENCODE(ccl.timeDiffU,         ccl.timeDiffU + ccl.nUnattachedClusters,          CTF::BLCtimeDiffU,         o2::rans::ProbabilityBits25Bit, optField[CTF::BLCtimeDiffU],         &buff);
  ENCODE(ccl.sigmaPadU,         ccl.sigmaPadU + ccl.nUnattachedClusters,          CTF::BLCsigmaPadU,         o2::rans::ProbabilityBits8Bit,  optField[CTF::BLCsigmaPadU],         &buff); 
  ENCODE(ccl.sigmaTimeU,        ccl.sigmaTimeU + ccl.nUnattachedClusters,         CTF::BLCsigmaTimeU,        o2::rans::ProbabilityBits8Bit,  optField[CTF::BLCsigmaTimeU],        &buff);
  ENCODE(ccl.nTrackClusters,    ccl.nTrackClusters + ccl.nTracks,                 CTF::BLCnTrackClusters,    o2::rans::ProbabilityBits16Bit, optField[CTF::BLCnTrackClusters],    &buff);
  ENCODE(ccl.nSliceRowClusters, ccl.nSliceRowClusters + ccl.nSliceRows,           CTF::BLCnSliceRowClusters, o2::rans::ProbabilityBits25Bit, optField[CTF::BLCnSliceRowClusters], &buff);
  // clang-format on
}

/// decode entropy-encoded bloks to TPC CompressedClusters into the externally provided vector (e.g. PMR vector from DPL)
template <typename VEC>
void CTFCoder::decode(const CTF::base& ec, VEC& buffVec)
{
  CompressedClusters cc;
  CompressedClustersCounters& ccCount = cc;
  ccCount = ec.getHeader(); // ec.getHeader is a saved copy of the CompressedClustersCounters
  CompressedClustersFlat* ccFlat = nullptr;
  size_t sizeCFlatBody = alignSize(ccFlat);
  size_t sz = sizeCFlatBody + estimateSize(cc);                                             // total size of the buffVec accounting for the alignment
  size_t vsz = sizeof(typename std::remove_reference<decltype(buffVec)>::type::value_type); // size of the element of the buffer
  buffVec.resize(sz / vsz);
  ccFlat = reinterpret_cast<CompressedClustersFlat*>(buffVec.data());                           // RS? do we need to align this pointer, or PMR vector will be already aligned?
  auto buff = reinterpret_cast<void*>(reinterpret_cast<char*>(buffVec.data()) + sizeCFlatBody); // will be the start of the CompressedClustersFlat payload

  setCompClusAddresses(cc, buff);
  ccFlat->set(sz, cc); // set offsets

  // decode encoded data directly to destination buff
  // clang-format off
  ec.decode(cc.qTotA,             CTF::BLCqTotA);
  ec.decode(cc.qMaxA,             CTF::BLCqMaxA);
  ec.decode(cc.flagsA,            CTF::BLCflagsA);
  ec.decode(cc.rowDiffA,          CTF::BLCrowDiffA);
  ec.decode(cc.sliceLegDiffA,     CTF::BLCsliceLegDiffA);
  ec.decode(cc.padResA,           CTF::BLCpadResA);
  ec.decode(cc.timeResA,          CTF::BLCtimeResA);
  ec.decode(cc.sigmaPadA,         CTF::BLCsigmaPadA);
  ec.decode(cc.sigmaTimeA,        CTF::BLCsigmaTimeA);
  ec.decode(cc.qPtA,              CTF::BLCqPtA);
  ec.decode(cc.rowA,              CTF::BLCrowA);
  ec.decode(cc.sliceA,            CTF::BLCsliceA);
  ec.decode(cc.timeA,             CTF::BLCtimeA);
  ec.decode(cc.padA,              CTF::BLCpadA);
  ec.decode(cc.qTotU,             CTF::BLCqTotU);
  ec.decode(cc.qMaxU,             CTF::BLCqMaxU);
  ec.decode(cc.flagsU,            CTF::BLCflagsU);
  ec.decode(cc.padDiffU,          CTF::BLCpadDiffU);
  ec.decode(cc.timeDiffU,         CTF::BLCtimeDiffU);
  ec.decode(cc.sigmaPadU,         CTF::BLCsigmaPadU);
  ec.decode(cc.sigmaTimeU,        CTF::BLCsigmaTimeU);
  ec.decode(cc.nTrackClusters,    CTF::BLCnTrackClusters);
  ec.decode(cc.nSliceRowClusters, CTF::BLCnSliceRowClusters);
  // clang-format on
}

} // namespace tpc
} // namespace o2

#endif // O2_TPC_CTFCODER_H
