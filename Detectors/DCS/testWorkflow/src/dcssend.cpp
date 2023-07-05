#include <string>
#include <zmq.hpp>
#include <zmq_addon.hpp>
#include <vector>
#include <iostream>
#include <fstream>
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

namespace bpo = boost::program_options;

int main(int argc, char** argv)
{
  bpo::variables_map vm;
  bpo::options_description opt_general("Usage:\n  " + std::string(argv[0]));
  bpo::options_description opt_hidden("");
  bpo::options_description opt_all;
  bpo::positional_options_description opt_pos;

  try {
    auto add_option = opt_general.add_options();
    add_option("help,h", "Print this help message");
    add_option("file,f", bpo::value<std::string>(), "file to send");
    add_option("file-port,o", bpo::value<int>()->default_value(5556), "port to send the file");
    add_option("ack-port,a", bpo::value<int>()->default_value(5557), "port to receive the acknowledgment");
    add_option("timeout,t", bpo::value<int>()->default_value(5), "timeout for acknowledgment");
    add_option("quit-on-ack,q", bpo::value<bool>()->default_value(false), "quit if acknowledgment is ok");
    opt_all.add(opt_general).add(opt_hidden);
    bpo::store(bpo::command_line_parser(argc, argv).options(opt_all).positional(opt_pos).run(), vm);

    if (vm.count("help") || vm.count("file") == 0) {
      std::cout << opt_general << std::endl;
      exit(0);
    }
    bpo::notify(vm);
  } catch (bpo::error& e) {
    std::cerr << "ERROR: " << e.what() << std::endl
              << std::endl;
    std::cerr << opt_general << std::endl;
    exit(1);
  } catch (std::exception& e) {
    std::cerr << e.what() << ", application will now exit" << std::endl;
    exit(2);
  }

  // init --------------------
  zmq::context_t context(1);
  zmq::socket_t publisher(context, zmq::socket_type::pub);
  zmq::socket_t collector(context, zmq::socket_type::pull);

  int recv_timeout = vm["timeout"].as<int>();

  const bool quitOnAck = vm["quit-on-ack"].as<bool>();

  collector.set(zmq::sockopt::rcvtimeo, recv_timeout * 1000);

  int pub_port = vm["file-port"].as<int>(), col_port = vm["ack-port"].as<int>();

  publisher.bind("tcp://127.0.0.1:" + std::to_string(pub_port));

  collector.bind("tcp://127.0.0.1:" + std::to_string(col_port));

  std::string filename = vm["file"].as<std::string>();

  // end of init -------------

  size_t trial = 0;
  while (1) {
    // 1s updates
    std::this_thread::sleep_for(std::chrono::seconds(1));

    // send -------------------------
    std::vector<zmq::message_t> data;

    // fill in filename
    data.push_back(zmq::message_t((void*)filename.data(), filename.size()));

    // load file into sstringbuffer
    std::ostringstream os;

    // read the file content
    std::string filepath = "./";
    std::ifstream detFile(filepath + filename, std::ifstream::in | std::ios::binary);
    os << detFile.rdbuf();

    // fill in the file content
    detFile.close();

    // fill in the file content
    data.push_back(zmq::message_t((void*)os.str().data(), os.str().size()));
    os.clear();

    std::cout << "send trial " << trial << "\n";
    // send it
    zmq::send_multipart(publisher, data);
    // end of send ------------------

    std::cout << "waiting for answer " << trial << "\n";
    // ack --------------------------
    zmq::message_t ack;
    std::string ans;
    auto rc = collector.recv(ack, zmq::recv_flags::none);

    ans.assign(ack.to_string());
    std::cout << ans << std::endl;
    // end of ack -------------------
    if (quitOnAck && (ans.find("ok") == (ans.size() - 2))) {
      break;
    }

    trial++;
  }
}
