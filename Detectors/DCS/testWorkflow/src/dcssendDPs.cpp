#include <string>
#include <zmq.hpp>
#include <zmq_addon.hpp>
#include <vector>
#include <iostream>
#include <fstream>
#include <chrono>

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

#include <thread>
#include <boost/program_options.hpp>

#include "DetectorsDCS/DataPointIdentifier.h"
#include "DetectorsDCS/DataPointValue.h"
#include "DetectorsDCS/DataPointCompositeObject.h"
#include "DetectorsDCS/DataPointCreator.h"

namespace bpo = boost::program_options;

int main(int argc, char** argv)
{
  using DPCOM = o2::dcs::DataPointCompositeObject;

  bpo::variables_map vm;
  bpo::options_description opt_general("Usage:\n  " + std::string(argv[0]));
  bpo::options_description opt_hidden("");
  bpo::options_description opt_all;
  bpo::positional_options_description opt_pos;

  try {
    auto add_option = opt_general.add_options();
    add_option("help,h", "Print this help message");
    add_option("rate,r", bpo::value<float>()->default_value(25.0f), "messages per second");
    add_option("size,s", bpo::value<int>()->default_value(1024 * 1024), "size per message");
    add_option("port,p", bpo::value<int>()->default_value(5556), "port to send the file");
    opt_all.add(opt_general).add(opt_hidden);
    bpo::store(bpo::command_line_parser(argc, argv).options(opt_all).positional(opt_pos).run(), vm);

    if (vm.count("help")) {
      std::cout << opt_general << std::endl;
      exit(0);
    }
    bpo::notify(vm);
  } catch (bpo::error& e) {
    std::cerr << "ERROR: " << e.what() << std::endl;
    std::cerr << opt_general << std::endl;
    exit(1);
  } catch (std::exception& e) {
    std::cerr << e.what() << ", application will now exit" << std::endl;
    exit(2);
  }

  // init --------------------
  zmq::context_t context(1);
  zmq::socket_t publisher(context, zmq::socket_type::push);

  int pub_port = vm["port"].as<int>();
  size_t sz = vm["size"].as<int>();

  publisher.bind("tcp://127.0.0.1:" + std::to_string(pub_port));

  std::chrono::duration<float> wait{1. / std::max(0.01f, vm["rate"].as<float>())};
  // end of init -------------

  auto timerStart = std::chrono::system_clock::now();
  auto timer0 = std::chrono::system_clock::now();
  //  std::cout << "will wait for " << wait << " s between injections\n"

  size_t trial = 0, datasize = 0;
  while (1) {
    std::this_thread::sleep_for(wait);

    // send -------------------------
    std::vector<zmq::message_t> data;

    std::vector<o2::dcs::DataPointCompositeObject> myVector;
    auto timeNow = std::chrono::high_resolution_clock::now();
    auto timeNowMs = std::chrono::duration_cast<std::chrono::milliseconds>(timeNow.time_since_epoch()).count(); // in ms
    auto timeNowS = std::chrono::duration_cast<std::chrono::seconds>(timeNow.time_since_epoch()).count();       // in ms

    size_t locSize = 0;
    do {
      myVector.emplace_back(o2::dcs::createDataPointCompositeObject("ADAPOS_LG/TEST_000100", 12345.6789, timeNowS, std::abs(timeNowMs - timeNowS * 1000)));
      locSize += myVector.size() * sizeof(DPCOM);
    } while (locSize < sz);

    // fill in filename
    data.push_back(zmq::message_t((void*)myVector.data(), myVector.size() * sizeof(DPCOM)));
    datasize += locSize;
    zmq::send_multipart(publisher, data);
    // end of send ------------------

    trial++;
    auto timerNow = std::chrono::system_clock::now();
    std::chrono::duration<float, std::ratio<1>> dur0 = timerNow - timer0;
    if (dur0.count() > 1) {
      timer0 = timerNow;
      std::chrono::duration<float, std::ratio<1>> durationStart = timerNow - timerStart;
      std::cout << trial << " messages sent in " << durationStart.count() << " s, total size: " << datasize << " bytes\n";
    }
  }
}
