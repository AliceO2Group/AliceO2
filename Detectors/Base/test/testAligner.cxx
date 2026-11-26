// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test Aligner
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include <stdexcept>
#include "DetectorsBase/Aligner.h"
#include "CommonUtils/ConfigurableParam.h"

using o2::base::Aligner;
using DetID = o2::detectors::DetID;

// Choice of the alignment object path per detector from align-geom.mAlignPaths
BOOST_AUTO_TEST_CASE(AlignmentPathDefault)
{
  BOOST_CHECK_EQUAL(Aligner::getAlignmentPath("", DetID::MCH), "MCH/Calib/Align");
  BOOST_CHECK_EQUAL(Aligner::getAlignmentPath("MID:MID/MisCalib/Align", DetID::MCH), "MCH/Calib/Align");
}

BOOST_AUTO_TEST_CASE(AlignmentPathOverride)
{
  const std::string paths = "MCH:MCH/MisCalib/Align,MID:MID/MisCalib/Align";
  BOOST_CHECK_EQUAL(Aligner::getAlignmentPath(paths, DetID::MCH), "MCH/MisCalib/Align");
  BOOST_CHECK_EQUAL(Aligner::getAlignmentPath(paths, DetID::MID), "MID/MisCalib/Align");
  BOOST_CHECK_EQUAL(Aligner::getAlignmentPath(paths, DetID::ITS), "ITS/Calib/Align");
  BOOST_CHECK_EQUAL(Aligner::getAlignmentPath(" MCH : MCH/MisCalib/Align ", DetID::MCH), "MCH/MisCalib/Align");
}

BOOST_AUTO_TEST_CASE(AlignmentPathInvalid)
{
  BOOST_CHECK_THROW(Aligner::getAlignmentPath("MCH", DetID::MCH), std::runtime_error);
  BOOST_CHECK_THROW(Aligner::getAlignmentPath("MCH:", DetID::MCH), std::runtime_error);
  BOOST_CHECK_THROW(Aligner::getAlignmentPath(":MCH/MisCalib/Align", DetID::MCH), std::runtime_error);
  BOOST_CHECK_THROW(Aligner::getAlignmentPath("XYZ:XYZ/MisCalib/Align", DetID::MCH), std::runtime_error);
  BOOST_CHECK_THROW(Aligner::getAlignmentPath("MCH:a/b/c,MCH:d/e/f", DetID::MCH), std::runtime_error);
  // an invalid entry is reported also when it is not the one asked for
  BOOST_CHECK_THROW(Aligner::getAlignmentPath("MCH:MCH/MisCalib/Align,XYZ:a/b/c", DetID::MCH), std::runtime_error);
}

BOOST_AUTO_TEST_CASE(AlignmentPathConfigKey)
{
  o2::conf::ConfigurableParam::updateFromString("align-geom.mAlignPaths=MCH:MCH/MisCalib/Align");
  BOOST_CHECK_EQUAL(Aligner::Instance().getAlignPaths(), "MCH:MCH/MisCalib/Align");
  BOOST_CHECK_EQUAL(Aligner::Instance().getAlignmentPath(DetID::MCH), "MCH/MisCalib/Align");
  BOOST_CHECK_EQUAL(Aligner::Instance().getAlignmentPath(DetID::TPC), "TPC/Calib/Align");
}
