// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   testCompStream.cxx
/// @author Matthias Richter
/// @since  2018-08-28
/// @brief  unit tests for iostreams with compression filter

#include "CommonUtils/CompStream.h"
#include "CommonUtils/StringUtils.h"
#define BOOST_TEST_MODULE CompStream unit test
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <filesystem>
#include <iostream>
#include <iomanip>
#include <sstream>

BOOST_AUTO_TEST_CASE(test_compstream_filesink)
{
  const int range = 100;
  std::stringstream filename;
  filename << o2::utils::Str::create_unique_path(std::filesystem::temp_directory_path().native()) << "_testCompStream.gz";
  {
    o2::io::ocomp_stream stream(filename.str().c_str(), o2::io::CompressionMethod::Gzip);
    for (int i = 0; i < range; i++) {
      stream << i;
      if (i % 2) {
        stream << " ";
      } else {
        stream << std::endl;
      }
    }
    stream << std::endl;
    stream.flush();
  }

  {
    o2::io::icomp_stream stream(filename.str(), o2::io::CompressionMethod::Gzip);
    int val;
    int expected = 0;
    while (stream >> val) {
      BOOST_CHECK(val == expected);
      if (val != expected) {
        break;
      }
      ++expected;
    }
    BOOST_CHECK(expected == range);
  }

  std::filesystem::remove(filename.str());
}

BOOST_AUTO_TEST_CASE(test_compstream_methods)
{
  auto checker = [](o2::io::CompressionMethod method) {
    std::stringstream pipe;
    {
      o2::io::ocomp_stream stream(pipe, method);
      for (int i = 0; i < 4; i++) {
        stream << i;
        stream << " ";
      }
      stream << std::endl;
    }

    {
      o2::io::icomp_stream stream(pipe, method);
      int val;
      int expected = 0;
      while (stream >> val) {
        BOOST_CHECK(val == expected);
        if (val != expected) {
          return false;
        }
        ++expected;
      }
    }
    return true;
  };

  BOOST_REQUIRE(checker(o2::io::CompressionMethod::Gzip));
  BOOST_REQUIRE(checker(o2::io::CompressionMethod::Zlib));
  BOOST_REQUIRE(checker(o2::io::CompressionMethod::Bzip2));
  //BOOST_REQUIRE(checker(o2::io::CompressionMethod::Lzma));
}

BOOST_AUTO_TEST_CASE(test_compstream_methods_mapper)
{
  auto checker = [](o2::io::CompressionMethod outputmethod, std::string inputmethod) {
    std::stringstream pipe;
    {
      o2::io::ocomp_stream stream(pipe, outputmethod);
      for (int i = 0; i < 4; i++) {
        stream << i;
        stream << " ";
      }
      stream << std::endl;
    }

    {
      o2::io::icomp_stream stream(pipe, inputmethod);
      int val;
      int expected = 0;
      while (stream >> val) {
        BOOST_CHECK(val == expected);
        if (val != expected) {
          return false;
        }
        ++expected;
      }
    }
    return true;
  };

  BOOST_REQUIRE(checker(o2::io::CompressionMethod::Gzip, "gzip"));
  BOOST_REQUIRE(checker(o2::io::CompressionMethod::Zlib, "zlib"));
  BOOST_REQUIRE(checker(o2::io::CompressionMethod::Bzip2, "bzip2"));
  //BOOST_REQUIRE(checker(o2::io::CompressionMethod::Lzma, "lzma"));
}
