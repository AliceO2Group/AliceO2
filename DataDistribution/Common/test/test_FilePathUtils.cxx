// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#define BOOST_TEST_MODULE Test Utilities DataDistribution Common
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include <boost/filesystem.hpp>
#include <boost/range/iterator_range.hpp>

#include <iostream>
#include <memory>
// #include <except>

#include "Common/FilePathUtils.h"

using namespace o2::DataDistribution;
using namespace std::string_literals;
using namespace boost::filesystem;

//____________________________________________________________________________//

struct MySetup {
  MySetup()
  {
    std::cout << "Global setup:\n";

    mTmpPath = unique_path();
    std::cout << "Creting temp directory: " << mTmpPath.string() << '\n';

    if (!create_directory(mTmpPath)) {
      throw std::runtime_error("Can not create directory"s + mTmpPath.string());
    }
  }

  ~MySetup()
  {
    std::cout << "Global teardown\nDeleting temp directory and its content";

    remove_all(mTmpPath);
  }

  static path mTmpPath;
};

path MySetup::mTmpPath;

//____________________________________________________________________________//

BOOST_GLOBAL_FIXTURE(MySetup);

//____________________________________________________________________________//

BOOST_AUTO_TEST_CASE(GetNextSeqNameTest)
{

  BOOST_CHECK(FilePathUtils::getNextSeqName(MySetup::mTmpPath.string()) == "0"s);

  // create some dirs
  create_directory(MySetup::mTmpPath / "dir");
  BOOST_CHECK(FilePathUtils::getNextSeqName(MySetup::mTmpPath.string()) == "0"s);

  create_directory(MySetup::mTmpPath / "dir0");
  BOOST_CHECK(FilePathUtils::getNextSeqName(MySetup::mTmpPath.string()) == "dir1"s);

  create_directory(MySetup::mTmpPath / "dir9");
  BOOST_CHECK(FilePathUtils::getNextSeqName(MySetup::mTmpPath.string()) == "dir10"s);

  create_directory(MySetup::mTmpPath / "dir09");
  BOOST_CHECK(FilePathUtils::getNextSeqName(MySetup::mTmpPath.string()) == "dir10"s);

  create_directory(MySetup::mTmpPath / "dir0009");
  BOOST_CHECK(FilePathUtils::getNextSeqName(MySetup::mTmpPath.string()) == "dir0010"s);

  create_directory(MySetup::mTmpPath / "dir0012.data");
  BOOST_CHECK(FilePathUtils::getNextSeqName(MySetup::mTmpPath.string()) == "dir0013.data"s);

  create_directory(MySetup::mTmpPath / "0013.data");
  BOOST_CHECK(FilePathUtils::getNextSeqName(MySetup::mTmpPath.string()) == "0014.data"s);

  create_directory(MySetup::mTmpPath / "-000014.data");
  BOOST_CHECK(FilePathUtils::getNextSeqName(MySetup::mTmpPath.string()) == "-000015.data"s);

  create_directory(MySetup::mTmpPath / "000015");
  BOOST_CHECK(FilePathUtils::getNextSeqName(MySetup::mTmpPath.string()) == "000016"s);
}

BOOST_AUTO_TEST_CASE(GetNextSeqNameTest_noMatch)
{

  // remove all tests
  for (auto& entry : boost::make_iterator_range(directory_iterator(MySetup::mTmpPath), {}))
    remove_all(entry.path());

  // create some dirs
  create_directory(MySetup::mTmpPath / "dir");
  BOOST_CHECK(FilePathUtils::getNextSeqName(MySetup::mTmpPath.string()) == "0"s);

  create_directory(MySetup::mTmpPath / "-");
  BOOST_CHECK(FilePathUtils::getNextSeqName(MySetup::mTmpPath.string()) == "0"s);

  create_directory(MySetup::mTmpPath / "00005");
  BOOST_CHECK(FilePathUtils::getNextSeqName(MySetup::mTmpPath.string()) == "00006"s);
}
