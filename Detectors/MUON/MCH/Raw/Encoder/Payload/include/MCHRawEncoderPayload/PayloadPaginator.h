// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_MCH_RAW_PAYLOAD_PAGINATOR_H
#define O2_MCH_RAW_PAYLOAD_PAGINATOR_H

namespace o2::raw
{
class RawFileWriter;
}

#include <string>
#include <gsl/span>
#include <set>
#include "MCHRawElecMap/Mapper.h"
#include <vector>

namespace o2::mch::raw
{

/// @brief Converts (DataBlockHeader,payload) pairs into RAW data (RDH,payload)
///
/// \nosubgrouping
class PayloadPaginator
{
 public:
  /// @param fw a RawFileWriter instance, that should be
  /// properly configured (once) _before_ calling the () operator
  /// @param outputFileName directory/basename of file to store the raw data
  /// @param filePerLink produce 1 file per link
  /// @param solar2feelink a mapper that converts a solarId value into
  /// a FeeLinkId object
  /// @param userLogic whether or not the format to emulate is UL
  /// @param chargeSumMode whether or not the format to emulate is in chargeSumMode
  PayloadPaginator(o2::raw::RawFileWriter& fw,
                   const std::string outputFileName,
                   bool filePerLink,
                   Solar2FeeLinkMapper solar2feelink,
                   bool userLogic,
                   bool chargeSumMode);

  /// Convert the buffer to raw data
  ///
  /// @param buffer a buffer of (DataBlockHeader,payload) MCH raw data
  /// (e.g. produced by a PayloadEncoder)
  void operator()(gsl::span<const std::byte> buffer);

 private:
  o2::raw::RawFileWriter& mRawFileWriter;
  Solar2FeeLinkMapper mSolar2FeeLink;
  std::string mOutputFileName;
  std::set<FeeLinkId> mFeeLinkIds{};
  uint16_t mExtraFeeIdMask{0};
  bool mFilePerLink{false};
};

/// helper function to wrap usage of PayloadPaginator class to
/// directly get memory representation of the buffer
/// (still creates a file as it's using RawFileWriter internally,
/// but that file is temporary)
std::vector<std::byte> paginate(gsl::span<const std::byte> buffer, bool userLogic, bool chargeSumMode,
                                const std::string& tmpfilename);
} // namespace o2::mch::raw
#endif
