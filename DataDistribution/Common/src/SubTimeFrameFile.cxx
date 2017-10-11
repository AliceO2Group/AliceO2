// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Common/SubTimeFrameFile.h"

#include <gsl/gsl_util>

namespace o2
{
namespace DataDistribution
{

////////////////////////////////////////////////////////////////////////////////
/// SubTimeFrameFileMeta
////////////////////////////////////////////////////////////////////////////////

const o2::header::DataDescription SubTimeFrameFileMeta::sDataDescFileSubTimeFrame{ "FILE_STF_META" };

std::ostream& operator<<(std::ostream& pStream, const SubTimeFrameFileMeta& pMeta)
{
  static_assert(std::is_standard_layout<SubTimeFrameFileMeta>::value,
                "SubTimeFrameFileMeta must be a std layout type.");

  // write DataHeader
  const o2::header::DataHeader lDataHeader = SubTimeFrameFileMeta::getDataHeader();
  pStream.write(reinterpret_cast<const char*>(&lDataHeader), sizeof(o2::header::DataHeader));
  // write the meta

  return pStream.write(reinterpret_cast<const char*>(&pMeta), sizeof(SubTimeFrameFileMeta));
}

////////////////////////////////////////////////////////////////////////////////
/// SubTimeFrameFileDataIndex
////////////////////////////////////////////////////////////////////////////////

const o2::header::DataDescription SubTimeFrameFileDataIndex::sDataDescFileStfDataIndex{ "FILE_STF_INDEX" };

std::ostream& operator<<(std::ostream& pStream, const SubTimeFrameFileDataIndex& pIndex)
{
  static_assert(std::is_standard_layout<SubTimeFrameFileDataIndex::DataIndexElem>::value,
                "SubTimeFrameFileDataIndex::DataIndexElem must be a std layout type.");

  // write DataHeader
  const o2::header::DataHeader lDataHeader = pIndex.getDataHeader();
  pStream.write(reinterpret_cast<const char*>(&lDataHeader), sizeof(o2::header::DataHeader));

  // write the index
  return pStream.write(reinterpret_cast<const char*>(pIndex.mDataIndex.data()),
                       pIndex.mDataIndex.size() * sizeof(SubTimeFrameFileDataIndex::DataIndexElem));
}
}
} /* o2::DataDistribution */
