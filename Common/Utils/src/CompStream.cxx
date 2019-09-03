// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   CompStream.cxx
/// @author Matthias Richter
/// @since  2018-08-28
/// @brief  Implementation of iostreams with compression filter

#include "CommonUtils/CompStream.h"
#include <boost/iostreams/device/file.hpp>
#include <boost/iostreams/stream.hpp>
#include <boost/iostreams/copy.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/filter/zlib.hpp>
#include <boost/iostreams/filter/bzip2.hpp>
//#include <boost/iostreams/filter/lzma.hpp>
#include <map>
#include <stdexcept>

namespace o2
{
namespace io
{

namespace comp_stream_helpers
{
template <typename T>
void pushDecompressor(T& stream, CompressionMethod method)
{
  switch (method) {
    case CompressionMethod::Gzip:
      stream.push(boost::iostreams::gzip_decompressor());
      break;
    case CompressionMethod::Zlib:
      stream.push(boost::iostreams::zlib_decompressor());
      break;
    case CompressionMethod::Lzma:
      throw std::runtime_error("lzma support not enabled");
      //stream.push(boost::iostreams::lzma_decompressor());
      break;
    case CompressionMethod::Bzip2:
      stream.push(boost::iostreams::bzip2_decompressor());
      break;
    case CompressionMethod::None:
      break;
    default:
      throw std::invalid_argument("missing implementation");
  }
}

template <typename T>
void pushCompressor(T& stream, CompressionMethod method)
{
  switch (method) {
    case CompressionMethod::Gzip:
      stream.push(boost::iostreams::gzip_compressor());
      break;
    case CompressionMethod::Zlib:
      stream.push(boost::iostreams::zlib_compressor());
      break;
    case CompressionMethod::Lzma:
      throw std::runtime_error("lzma support not enabled");
      //stream.push(boost::iostreams::lzma_compressor());
      break;
    case CompressionMethod::Bzip2:
      stream.push(boost::iostreams::bzip2_compressor());
      break;
    case CompressionMethod::None:
      break;
    default:
      throw std::invalid_argument("missing implementation");
  }
}

const std::map<std::string, CompressionMethod> Mapping = {
  {"none", CompressionMethod::None},
  {"gzip", CompressionMethod::Gzip},
  {"zlib", CompressionMethod::Zlib},
  {"bzip2", CompressionMethod::Bzip2},
  {"lzma", CompressionMethod::Lzma},
};

auto Method(std::string method)
{
  auto const& i = Mapping.find(method);
  if (i == Mapping.end()) {
    throw std::invalid_argument("invalid compression method: " + method);
  }
  return i->second;
}
} // namespace comp_stream_helpers

icomp_stream::icomp_stream(std::string filename, CompressionMethod method)
  : mStreamBuffer(), mInputFile(filename, std::ios_base::in | std::ios_base::binary), std::istream(&mStreamBuffer)
{
  comp_stream_helpers::pushDecompressor(mStreamBuffer, method);
  mStreamBuffer.push(mInputFile);
}

icomp_stream::icomp_stream(std::istream& backend, CompressionMethod method)
  : mStreamBuffer(), mInputFile(), std::istream(&mStreamBuffer)
{
  comp_stream_helpers::pushDecompressor(mStreamBuffer, method);
  mStreamBuffer.push(backend);
}

icomp_stream::icomp_stream(std::string filename, std::string method)
  : mStreamBuffer(), mInputFile(filename, std::ios_base::in | std::ios_base::binary), std::istream(&mStreamBuffer)
{
  comp_stream_helpers::pushDecompressor(mStreamBuffer, comp_stream_helpers::Method(method));
  mStreamBuffer.push(mInputFile);
}

icomp_stream::icomp_stream(std::istream& backend, std::string method)
  : mStreamBuffer(), mInputFile(), std::istream(&mStreamBuffer)
{
  comp_stream_helpers::pushDecompressor(mStreamBuffer, comp_stream_helpers::Method(method));
  mStreamBuffer.push(backend);
}

ocomp_stream::ocomp_stream(std::string filename, CompressionMethod method)
  : mStreamBuffer(), std::ostream(&mStreamBuffer)
{
  comp_stream_helpers::pushCompressor(mStreamBuffer, method);
  mStreamBuffer.push(boost::iostreams::file_sink(filename));
}

ocomp_stream::ocomp_stream(std::ostream& backend, CompressionMethod method)
  : mStreamBuffer(), std::ostream(&mStreamBuffer)
{
  comp_stream_helpers::pushCompressor(mStreamBuffer, method);
  mStreamBuffer.push(backend);
}

ocomp_stream::ocomp_stream(std::string filename, std::string method)
  : mStreamBuffer(), std::ostream(&mStreamBuffer)
{
  comp_stream_helpers::pushCompressor(mStreamBuffer, comp_stream_helpers::Method(method));
  mStreamBuffer.push(boost::iostreams::file_sink(filename));
}

ocomp_stream::ocomp_stream(std::ostream& backend, std::string method)
  : mStreamBuffer(), std::ostream(&mStreamBuffer)
{
  comp_stream_helpers::pushCompressor(mStreamBuffer, comp_stream_helpers::Method(method));
  mStreamBuffer.push(backend);
}
} // namespace io
} // namespace o2
