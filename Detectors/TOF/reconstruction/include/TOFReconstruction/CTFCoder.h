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
/// \author fnoferin@cern.ch
/// \brief class for entropy encoding/decoding of TOF compressed infos data

#ifndef O2_TOF_CTFCODER_H
#define O2_TOF_CTFCODER_H

#include <algorithm>
#include <iterator>
#include <string>
#include "DataFormatsTOF/CTF.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "rANS/rans.h"
#include "TOFBase/Digit.h"

class TTree;

namespace o2
{
namespace tof
{

class CTFCoder
{
 public:
  /// entropy-encode clusters to buffer with CTF
  template <typename VEC>
  static void encode(VEC& buff, const gsl::span<const ReadoutWindowData>& rofRecVec, const gsl::span<const Digit>& cdigVec, const gsl::span<const unsigned char>& pattVec);

  /// entropy decode clusters from buffer with CTF
  template <typename VROF, typename VDIG, typename VPAT>
  static void decode(const CTF::base& ec, VROF& rofRecVec, VDIG& cdigVec, VPAT& pattVec);

 private:
  /// compres compact clusters to CompressedInfos
  static void compress(CompressedInfos& cc, const gsl::span<const ReadoutWindowData>& rofRecVec, const gsl::span<const Digit>& cdigVec, const gsl::span<const unsigned char>& pattVec);

  /// decompress CompressedInfos to compact clusters
  template <typename VROF, typename VDIG, typename VPAT>
  static void decompress(const CompressedInfos& cc, VROF& rofRecVec, VDIG& cdigVec, VPAT& pattVec);

  static void appendToTree(TTree& tree, o2::detectors::DetID id, CTF& ec);
  static void readFromTree(TTree& tree, int entry, o2::detectors::DetID id, std::vector<ReadoutWindowData>& rofRecVec, std::vector<Digit>& cdigVec, std::vector<unsigned char>& pattVec);

 protected:
  ClassDefNV(CTFCoder, 1);
};

} // namespace tof
} // namespace o2

#endif // O2_TOF_CTFCODER_H
