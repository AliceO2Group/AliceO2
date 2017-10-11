// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_SUBTIMEFRAME_FILE_WRITER_H_
#define ALICEO2_SUBTIMEFRAME_FILE_WRITER_H_

#include "Common/SubTimeFrameDataModel.h"
#include "Common/SubTimeFrameFile.h"
#include <Headers/DataHeader.h>

#include <type_traits>
#include <boost/filesystem.hpp>
#include <fstream>
#include <vector>

class O2Device;

namespace o2
{
namespace DataDistribution
{

////////////////////////////////////////////////////////////////////////////////
/// SubTimeFrameFileWriter
////////////////////////////////////////////////////////////////////////////////

class SubTimeFrameFileWriter : public ISubTimeFrameConstVisitor
{
  static const constexpr char* sSidecarFieldSep = " ";
  static const constexpr char* sSidecarRecordSep = "\n";

 public:
  SubTimeFrameFileWriter() = delete;
  SubTimeFrameFileWriter(const boost::filesystem::path& pFileName, bool pWriteInfo = false);
  ~SubTimeFrameFileWriter();

  ///
  /// Writes a (Sub)TimeFrame
  ///
  std::uint64_t write(const SubTimeFrame& pStf);

  ///
  /// Tell current size of the file
  ///
  const std::uint64_t size() { return std::uint64_t(mFile.tellp()); }

 private:
  void visit(const SubTimeFrame& pStf) override;

  //
  //  workaround for buffered operation (performance):
  //   - provide a new, larger, buffer
  //   - always write less than 1024B at a time (hard limit in libstdc++)
  //
  static const std::streamsize sBuffSize = 256ul << 10; // 256 kiB
  static const std::streamsize sChunkSize = 512;
  std::ofstream mFile;

  bool mWriteInfo;
  std::ofstream mInfoFile;

  std::unique_ptr<char[]> mFileBuf;
  std::unique_ptr<char[]> mInfoFileBuf;

  // helper to make sure the written blocks are buffered
  template <
    typename pointer,
    typename std::enable_if<
      std::is_pointer<pointer>::value &&                      // pointers only
      (std::is_void<std::remove_pointer_t<pointer>>::value || // void* or standard layout!
       std::is_standard_layout<std::remove_pointer_t<pointer>>::value)>::type* = nullptr>
  void buffered_write(const pointer p, std::streamsize count);

  const std::uint64_t getSizeInFile() const;

  // vector of <headers, data> elements of a Stf to be written
  std::vector<const SubTimeFrame::StfData*> mStfData;
  SubTimeFrameFileDataIndex mStfDataIndex;
  std::uint64_t mStfSize = std::uint64_t(0); // meta + index + data (and all headers)
};
}
} /* o2::DataDistribution */

#endif /* ALICEO2_SUBTIMEFRAME_FILE_WRITER_H_ */
