// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/* Local Variables:  */
/* mode: c++         */
/* End:              */

#ifndef COMPSTREAM_H
#define COMPSTREAM_H

/// @file   CompStream.h
/// @author Matthias Richter
/// @since  2018-08-28
/// @brief  Implementation of iostreams with compression filter

#include <istream>
#include <fstream>
#include <ostream>
#include <string>
#include <boost/iostreams/filtering_streambuf.hpp>

namespace o2
{
namespace io
{

enum class CompressionMethod {
  None,
  Gzip,
  Zlib,
  Bzip2,
  Lzma
};

/**
 * @class icomp_stream
 * An istream variant allowing to add compression filters.
 *
 * This stream can be used transparently as std::istream while providing decompression
 * directly on the backend stream/file.
 *
 * Implementation is based on boost::iostreams utilities, the filtered_streambuf
 * and compression filters. Currently supporting gzip, zlib, bzip2, and lzma algorithms.
 * The algorithm is specified either by enum or string to the constructor.
 */
class icomp_stream : public std::istream
{
 public:
  using streambuffer_t = boost::iostreams::filtering_streambuf<boost::iostreams::input>;

  /// constructor
  /// @param filename name of the file to read from
  /// @param method   compression method specified by enum
  icomp_stream(std::string filename, CompressionMethod method = CompressionMethod::None);

  /// constructor
  /// @param backend the stream to read data from
  /// @param method  compression method specified by enum
  icomp_stream(std::istream& backend, CompressionMethod method = CompressionMethod::None);

  /// constructor
  /// @param filename name of the file to read from
  /// @param method   compression method specified by string: gzip, zlib, bzip2, lzma
  icomp_stream(std::string filename, std::string method);

  /// constructor
  /// @param backend the stream to read data from
  /// @param method  compression method specified by string: gzip, zlib, bzip2, lzma
  icomp_stream(std::istream& backend, std::string method);

 private:
  streambuffer_t mStreamBuffer;
  std::ifstream mInputFile;
};

/**
 * @class ocomp_stream
 * An ostream variant allowing to add compression filters.
 *
 * This stream can be used transparently as std::ostream while providing compression
 * directly on the backend stream/file.
 *
 * Implementation is based on boost::iostreams utilities, the filtered_streambuf
 * and compression filters. Currently supporting gzip, zlib, bzip2, and lzma algorithms.
 * The algorithm is specified either by enum or string to the constructor.
 */
class ocomp_stream : public std::ostream
{
 public:
  using streambuffer_t = boost::iostreams::filtering_streambuf<boost::iostreams::output>;

  /// constructor
  /// @param filename name of the file to read from
  /// @param method   compression method specified by enum
  ocomp_stream(std::string filename, CompressionMethod method = CompressionMethod::None);

  /// constructor
  /// @param backend the stream to read data from
  /// @param method  compression method specified by enum
  ocomp_stream(std::ostream& backend, CompressionMethod method = CompressionMethod::None);

  /// constructor
  /// @param filename name of the file to read from
  /// @param method   compression method specified by string: gzip, zlib, bzip2, lzma
  ocomp_stream(std::string filename, std::string method);

  /// constructor
  /// @param backend the stream to read data from
  /// @param method  compression method specified by string: gzip, zlib, bzip2, lzma
  ocomp_stream(std::ostream& backend, std::string method);

 private:
  streambuffer_t mStreamBuffer;
};

} // namespace io
} // namespace o2
#endif // COMPSTREAM_H
