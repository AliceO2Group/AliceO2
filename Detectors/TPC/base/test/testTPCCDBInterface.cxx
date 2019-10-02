// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test TPC CDBInterface class
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

// boost includes
#include <boost/range/combine.hpp>
#include <boost/test/unit_test.hpp>

// ROOT includes
#include "TFile.h"

// o2 includes
#include "TPCBase/CDBInterface.h"
#include "TPCBase/CDBInterface.h"
#include "TPCBase/CalArray.h"
#include "TPCBase/CalDet.h"
#include "TPCBase/Mapper.h"
#include "TPCBase/ParameterDetector.h"
#include "TPCBase/ParameterElectronics.h"
#include "TPCBase/ParameterGEM.h"
#include "TPCBase/ParameterGas.h"
#include "SimConfig/ConfigurableParam.h"

namespace o2
{
namespace tpc
{
//template <class T>
//void writeObject(T& obj, const std::string_view type, const std::string_view name, const int run)
//{
//  auto cdb = o2::ccdb::Manager::Instance();
//
//  auto id = new o2::ccdb::ConditionId(
//    "TPC/" + type +
//      "/" + name,
//    run, run, 1, 0);
//  auto md = new o2::ccdb::ConditionMetaData();
//  cdb->putObjectAny(&obj, *id, md);
//}

const std::string ccdbUrl = "file:///tmp/CCDBSnapshot";

/// \brief write a CalPad object to the CCDB
CalPad writeCalPadObject(const std::string_view name, const int run, const int dataOffset = 0)
{
  // ===| create and write test data |==========================================
  CalPad data(PadSubset::ROC);

  int iter = dataOffset;
  data.setName(name);
  for (auto& calArray : data.getData()) {
    for (auto& value : calArray.getData()) {
      value = iter++;
    }
  }

  o2::ccdb::CcdbApi ccdbApi;
  ccdbApi.init(ccdbUrl);
  std::map<std::string, std::string> metadata;
  ccdbApi.storeAsTFileAny<CalPad>(data, "Calib", metadata);

  writeObject(data, "Calib", name, run);

  return data;
}

/// \brief Check equivalence of two CalPad objects
void checkCalPadEqual(const CalPad& data, const CalPad& dataRead)
{
  auto& mapper = Mapper::instance();
  const auto numberOfPads = mapper.getPadsInSector() * 36;

  float sumROC = 0.f;

  int numberOfPadsROC = 0;

  for (auto const& arrays : boost::combine(data.getData(), dataRead.getData())) {
    for (auto const& val : boost::combine(arrays.get<0>().getData(), arrays.get<1>().getData())) {
      sumROC += (val.get<0>() - val.get<1>());
      ++numberOfPadsROC;
    }
  }

  BOOST_CHECK_EQUAL(data.getName(), dataRead.getName());
  BOOST_CHECK_CLOSE(sumROC, 0.f, 1.E-12);
  BOOST_CHECK_EQUAL(numberOfPadsROC, numberOfPads);
}

/// \brief Test reading pedestal object from the CDB using the TPC CDBInterface
BOOST_AUTO_TEST_CASE(CDBInterface_test_pedestals)
{
  const int run = 1;
  const int dataOffset = 0;
  const std::string_view type = "Pedestals";

  // ===| initialize CDB manager |==============================================
  auto cdb = o2::ccdb::Manager::Instance();
  cdb->setDefaultStorage("local://O2CDB");

  // ===| write test object |===================================================
  auto data = writeCalPadObject(type, run, dataOffset);

  // ===| TPC interface |=======================================================
  auto& tpcCDB = CDBInterface::instance();

  // ===| read object |=========================================================
  cdb->setRun(run);
  auto dataRead = tpcCDB.getPedestals();

  // ===| checks |==============================================================
  checkCalPadEqual(data, dataRead);
}

/// \brief Test reading noise object from the CDB using the TPC CDBInterface
BOOST_AUTO_TEST_CASE(CDBInterface_test_noise)
{
  const int run = 2;
  const int dataOffset = 1;
  const std::string_view type = "Noise";

  // ===| initialize CDB manager |==============================================
  auto cdb = o2::ccdb::Manager::Instance();
  cdb->setDefaultStorage("local://O2CDB");

  // ===| write test object |===================================================
  auto data = writeCalPadObject(type, run, dataOffset);

  // ===| TPC interface |=======================================================
  auto& tpcCDB = CDBInterface::instance();

  // ===| read object |=========================================================
  cdb->setRun(run);
  auto dataRead = tpcCDB.getNoise();

  // ===| checks |==============================================================
  checkCalPadEqual(data, dataRead);
}

/// \brief Test reading gain map object from the CDB using the TPC CDBInterface
BOOST_AUTO_TEST_CASE(CDBInterface_test_gainmap)
{
  const int run = 2;
  const int dataOffset = 1;
  const std::string_view type = "Gain";

  // ===| initialize CDB manager |==============================================
  auto cdb = o2::ccdb::Manager::Instance();
  cdb->setDefaultStorage("local://O2CDB");

  // ===| write test object |===================================================
  auto data = writeCalPadObject(type, run, dataOffset);

  // ===| TPC interface |=======================================================
  auto& tpcCDB = CDBInterface::instance();

  // ===| read object |=========================================================
  cdb->setRun(run);
  auto dataRead = tpcCDB.getGainMap();

  // ===| checks |==============================================================
  checkCalPadEqual(data, dataRead);
}

/// \brief Test reading ParameterDetector from the CDB using the TPC CDBInterface
BOOST_AUTO_TEST_CASE(CDBInterface_test_ParameterDetector)
{
  const int run = 3;
  const std::string_view name = "Detector";
  auto value = 100.3f;

  // ===| initialize CDB manager |==============================================
  auto cdb = o2::ccdb::Manager::Instance();
  cdb->setDefaultStorage("local://O2CDB");

  // ===| write test object |===================================================
  auto& data = ParameterDetector::Instance();
  //  o2::conf::ConfigurableParam::updateFromString(TString::Format("TPCDetParam.TPClength = %f", value).Data());

  // disabled for the moment since we cannot write these objects yet...
  //writeObject(data, "Parameter", name, run);

  // ===| TPC interface |=======================================================
  //  auto& tpcCDB = CDBInterface::instance();

  // ===| read object |=========================================================
  //  cdb->setRun(run);
  //  auto dataRead = tpcCDB.getParameterDetector();

  // ===| checks |==============================================================
  //  BOOST_CHECK_CLOSE(value, dataRead.getTPClength(), 1.E-12);
}

/// \brief Test reading ParameterElectronics from the CDB using the TPC CDBInterface
BOOST_AUTO_TEST_CASE(CDBInterface_test_ParameterElectronics)
{
  const int run = 3;
  const std::string_view name = "Electronics";
  auto value = 80.9f;

  // ===| initialize CDB manager |==============================================
  auto cdb = o2::ccdb::Manager::Instance();
  cdb->setDefaultStorage("local://O2CDB");

  // ===| write test object |===================================================
  auto& data = ParameterElectronics::Instance();
  //  o2::conf::ConfigurableParam::updateFromString(TString::Format("TPCEleParam.PeakingTime = %f", value).Data());

  // disabled for the moment since we cannot write these objects yet...
  //  writeObject(data, "Parameter", name, run);

  // ===| TPC interface |=======================================================
  //  auto& tpcCDB = CDBInterface::instance();

  // ===| read object |=========================================================
  //  cdb->setRun(run);
  //  auto dataRead = tpcCDB.getParameterElectronics();

  // ===| checks |==============================================================
  //  BOOST_CHECK_CLOSE(value, dataRead.getPeakingTime(), 1.E-12);
}

/// \brief Test reading ParameterGas from the CDB using the TPC CDBInterface
BOOST_AUTO_TEST_CASE(CDBInterface_test_ParameterGas)
{
  const int run = 3;
  const std::string_view name = "Gas";
  auto value = 1000.9434f;

  // ===| initialize CDB manager |==============================================
  auto cdb = o2::ccdb::Manager::Instance();
  cdb->setDefaultStorage("local://O2CDB");

  // ===| write test object |===================================================
  auto& data = ParameterGas::Instance();
  //  o2::conf::ConfigurableParam::updateFromString(TString::Format("TPCGasParam.DriftV = %f", value).Data());

  // disabled for the moment since we cannot write these objects yet...
  //  writeObject(data, "Parameter", name, run);

  // ===| TPC interface |=======================================================
  //  auto& tpcCDB = CDBInterface::instance();

  // ===| read object |=========================================================
  //  cdb->setRun(run);
  //  auto dataRead = tpcCDB.getParameterGas();

  // ===| checks |==============================================================
  //  BOOST_CHECK_CLOSE(value, dataRead.getVdrift(), 1.E-12);
}

/// \brief Test reading ParameterGEM from the CDB using the TPC CDBInterface
BOOST_AUTO_TEST_CASE(CDBInterface_test_ParameterGEM)
{
  const int run = 3;
  const std::string_view name = "GEM";
  auto value = 1.7382f;

  // ===| initialize CDB manager |==============================================
  auto cdb = o2::ccdb::Manager::Instance();
  cdb->setDefaultStorage("local://O2CDB");

  // ===| write test object |===================================================
  auto& data = ParameterGEM::Instance();
  //  o2::conf::ConfigurableParam::updateFromString(TString::Format("TPCGEMParam.TotalGainStack = %f", value).Data());

  // disabled for the moment since we cannot write these objects yet...
  //  writeObject(data, "Parameter", name, run);

  // ===| TPC interface |=======================================================
  //  auto& tpcCDB = CDBInterface::instance();

  // ===| read object |=========================================================
  //  cdb->setRun(run);
  //  auto dataRead = tpcCDB.getParameterGEM();

  // ===| checks |==============================================================
  //  BOOST_CHECK_CLOSE(value, dataRead.getCollectionEfficiency(0), 1.E-12);
}

/// \brief Test using the default parameters and initialize from file
BOOST_AUTO_TEST_CASE(CDBInterface_test_Default_ReadFromFile)
{
  // ===| TPC interface |=======================================================
  auto& tpcCDB = CDBInterface::instance();

  // ===| get default pedestals and noise |=====================================
  tpcCDB.setUseDefaults();

  // we need a copy here to do the comparison below
  auto pedestals = tpcCDB.getPedestals();
  auto noise = tpcCDB.getNoise();
  auto gainmap = tpcCDB.getGainMap();

  // check interface for defaults
  tpcCDB.getParameterDetector();
  tpcCDB.getParameterElectronics();
  tpcCDB.getParameterGas();
  tpcCDB.getParameterGEM();

  // ===| dump to file |========================================================
  auto f = TFile::Open("Calibration.root", "recreate");
  f->WriteObject(&pedestals, "Pedestals");
  f->WriteObject(&noise, "Noise");
  f->WriteObject(&gainmap, "Gain");
  delete f;

  // ===| read from file |======================================================
  tpcCDB.setUseDefaults(false);
  tpcCDB.resetLocalCalibration();
  tpcCDB.setPedestalsAndNoiseFromFile("Calibration.root");
  tpcCDB.setGainMapFromFile("Calibration.root");

  auto& pedestalsFromFile = tpcCDB.getPedestals();
  auto& noiseFromFile = tpcCDB.getNoise();
  auto& gainmapFromFile = tpcCDB.getGainMap();

  // ===| checks |==============================================================
  checkCalPadEqual(noise, noiseFromFile);
  checkCalPadEqual(pedestals, pedestalsFromFile);
  checkCalPadEqual(gainmap, gainmapFromFile);
}
} // namespace tpc
} // namespace o2
