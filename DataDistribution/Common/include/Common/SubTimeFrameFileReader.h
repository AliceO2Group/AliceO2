// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_SUBTIMEFRAME_FILE_READER_H_
#define ALICEO2_SUBTIMEFRAME_FILE_READER_H_

#include "Common/SubTimeFrameDataModel.h"
#include <Headers/DataHeader.h>

#include <boost/filesystem.hpp>
#include <fstream>
#include <vector>

class O2Device;

namespace o2
{
namespace DataDistribution
{

////////////////////////////////////////////////////////////////////////////////
/// SubTimeFrameFileReader
////////////////////////////////////////////////////////////////////////////////

class SubTimeFrameFileReader : public ISubTimeFrameVisitor
{
 public:
  SubTimeFrameFileReader() = delete;
  SubTimeFrameFileReader(boost::filesystem::path& pFileName);
  ~SubTimeFrameFileReader();

  ///
  /// Read a single TF from the file
  ///
  bool read(SubTimeFrame& pStf, std::uint64_t pStfId, FairMQChannel& pDstChan);

  ///
  /// Tell the current position of the file
  ///
  const std::uint64_t position() { return std::uint64_t(mFile.tellg()); }

  ///
  /// Tell the size of the file
  ///
  const std::uint64_t size() const { return mFileSize; }

 private:
  void visit(SubTimeFrame& pStf) override;

  std::ifstream mFile;
  std::uint64_t mFileSize;

  // helper to make sure written chunks are buffered, only allow pointers
  template <typename pointer,
            typename = std::enable_if_t<std::is_pointer<pointer>::value>>
  std::istream& buffered_read(pointer pPtr, std::streamsize pLen)
  {
    return mFile.read(reinterpret_cast<char*>(pPtr), pLen);
  }

  std::int64_t getHeaderStackSize();

  // vector of <hdr, fmqMsg> elements of a tf read from the file
  std::vector<SubTimeFrame::StfData> mStfData;
};
}
} /* o2::DataDistribution */

#endif /* ALICEO2_SUBTIMEFRAME_FILE_READER_H_ */
