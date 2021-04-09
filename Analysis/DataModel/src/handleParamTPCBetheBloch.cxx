// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

///
/// \file   handleParamTPCBetheBloch.cxx
/// \author Nicolo' Jacazio
/// \since  2020-06-22
/// \brief  A simple tool to produce Bethe Bloch parametrization objects for the TPC PID Response
///

#include "CCDB/CcdbApi.h"
#include <boost/program_options.hpp>
#include <FairLogger.h>
#include "TFile.h"
#include "AnalysisDataModel/PID/BetheBloch.h"
#include "AnalysisDataModel/PID/TPCReso.h"

using namespace o2::pid::tpc;
namespace bpo = boost::program_options;

bool initOptionsAndParse(bpo::options_description& options, int argc, char* argv[], bpo::variables_map& vm)
{
  options.add_options()(
    "url,u", bpo::value<std::string>()->default_value("http://ccdb-test.cern.ch:8080"), "URL of the CCDB database")(
    "ccdb-path,c", bpo::value<std::string>()->default_value("Analysis/PID/TPC"), "CCDB path for storage/retrieval")(
    "start,s", bpo::value<long>()->default_value(0), "Start timestamp of object validity")(
    "stop,S", bpo::value<long>()->default_value(4108971600000), "Stop timestamp of object validity")(
    "delete-previous,delete_previous,d", bpo::value<int>()->default_value(0), "Flag to delete previous versions of converter objects in the CCDB before uploading the new one so as to avoid proliferation on CCDB")(
    "save-to-file,file,f,o", bpo::value<std::string>()->default_value(""), "Option to save parametrization to file instead of uploading to ccdb")(
    "read-from-file,i", bpo::value<std::string>()->default_value(""), "Option to get parametrization from a file")(
    "exp-name,n", bpo::value<std::string>()->default_value("BetheBloch"), "Name of the parametrization object")(
    "reso-name,n", bpo::value<std::string>()->default_value("TPCReso"), "Name of the parametrization object")(
    "mode,m", bpo::value<unsigned int>()->default_value(1), "Working mode: 0 push 1 pull and test")(
    "p0", bpo::value<float>()->default_value(0.0320981), "Parameter 0 of the TPC expected value")(
    "p1", bpo::value<float>()->default_value(19.9768), "Parameter 1 of the TPC expected value")(
    "p2", bpo::value<float>()->default_value(2.52666e-16), "Parameter 2 of the TPC expected value")(
    "p3", bpo::value<float>()->default_value(2.72123), "Parameter 3 of the TPC expected value")(
    "p4", bpo::value<float>()->default_value(6.08092), "Parameter 4 of the TPC expected value")(
    "p5", bpo::value<float>()->default_value(50.f), "Parameter 5 of the TPC expected value")(
    "p6", bpo::value<float>()->default_value(2.3), "Parameter 6 of the TPC expected value")(
    "verbose,v", bpo::value<int>()->default_value(0), "Verbose level 0, 1")(
    "help,h", "Produce help message.");
  try {
    bpo::store(parse_command_line(argc, argv, options), vm);

    // help
    if (vm.count("help")) {
      LOG(INFO) << options;
      return false;
    }

    bpo::notify(vm);
  } catch (const bpo::error& e) {
    LOG(ERROR) << e.what() << "\n";
    LOG(ERROR) << "Error parsing command line arguments; Available options:";
    LOG(ERROR) << options;
    return false;
  }
  return true;
}

int main(int argc, char* argv[])
{
  bpo::options_description options("Allowed options");
  bpo::variables_map vm;
  if (!initOptionsAndParse(options, argc, argv, vm)) {
    return 1;
  }

  const unsigned int mode = vm["mode"].as<unsigned int>();
  const std::string path = vm["ccdb-path"].as<std::string>();
  std::map<std::string, std::string> metadata;
  std::map<std::string, std::string>* headers;
  o2::ccdb::CcdbApi api;
  const std::string url = vm["url"].as<std::string>();
  api.init(url);
  if (!api.isHostReachable()) {
    LOG(WARNING) << "CCDB host " << url << " is not reacheable, cannot go forward";
    return 1;
  }
  BetheBloch* bb = nullptr;
  TPCReso* reso = nullptr;
  const std::string exp_name = vm["exp-name"].as<std::string>();
  const std::string reso_name = vm["reso-name"].as<std::string>();
  if (mode == 0) { // Push mode
    LOG(INFO) << "Handling TPC parametrization in create mode";
    const std::string input_file_name = vm["read-from-file"].as<std::string>();
    if (!input_file_name.empty()) {
      TFile f(input_file_name.data(), "READ");
      if (!f.IsOpen()) {
        LOG(WARNING) << "Input file " << input_file_name << " is not reacheable, cannot get param from file";
      }
      f.GetObject(exp_name.c_str(), bb);
      f.GetObject(reso_name.c_str(), reso);
      f.Close();
    }
    if (!bb) {
      bb = new BetheBloch();
      const std::vector<float> bbparams = {vm["p0"].as<float>(), vm["p1"].as<float>(), vm["p2"].as<float>(), vm["p3"].as<float>(), vm["p4"].as<float>(), vm["p5"].as<float>(), vm["p6"].as<float>()};
      bb->SetParameters(bbparams);
    }
    if (!reso) {
      reso = new TPCReso();
      const std::vector<float> resoparams = {0.07, 0.0};
      reso->SetParameters(resoparams);
    }
    bb->Print();
    const std::string fname = vm["save-to-file"].as<std::string>();
    if (!fname.empty()) { // Saving it to file
      LOG(INFO) << "Saving parametrization to file " << fname;
      TFile f(fname.data(), "RECREATE");
      bb->Write();
      reso->Write();
      f.ls();
      f.Close();
    } else { // Saving it to CCDB
      LOG(INFO) << "Saving parametrization to CCDB " << path;

      long start = vm["start"].as<long>();
      long stop = vm["stop"].as<long>();

      if (vm["delete-previous"].as<int>()) {
        api.truncate(path);
      }
      api.storeAsTFileAny(bb, path + "/" + exp_name, metadata, start, stop);
      api.storeAsTFileAny(reso, path + "/" + reso_name, metadata, start, stop);
    }
  } else { // Pull and test mode
    LOG(INFO) << "Handling TPC parametrization in test mode";
    const float x[2] = {1, 1};
    BetheBloch* bb = api.retrieveFromTFileAny<BetheBloch>(path + "/" + exp_name, metadata, -1, headers);
    bb->Print();
    LOG(INFO) << "BetheBloch " << bb->operator()(x);
    TPCReso* reso = api.retrieveFromTFileAny<TPCReso>(path + "/" + reso_name, metadata, -1, headers);
    reso->Print();
    LOG(INFO) << "TPCReso " << reso->operator()(x);
  }

  return 0;
}
