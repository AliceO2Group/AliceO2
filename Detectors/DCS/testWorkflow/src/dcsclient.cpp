#include <string>
#include <zmq.hpp>
#include <zmq_addon.hpp>
#include <vector>
#include <iostream>
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

#include <fstream>
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
    add_option("file-port,o", bpo::value<int>()->default_value(5556), "port to receive the file");
    add_option("ack-port,a", bpo::value<int>()->default_value(5557), "port to send the acknowledgment");
    opt_all.add(opt_general).add(opt_hidden);
    bpo::store(bpo::command_line_parser(argc, argv).options(opt_all).positional(opt_pos).run(), vm);

    if (vm.count("help")) {
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

  zmq::context_t context(1);
  zmq::socket_t subscriber(context, zmq::socket_type::sub);
  zmq::socket_t answer(context, zmq::socket_type::push);

  int pub_port = vm["file-port"].as<int>(), col_port = vm["ack-port"].as<int>();

  subscriber.connect("tcp://127.0.0.1:" + std::to_string(pub_port));
  subscriber.set(zmq::sockopt::subscribe, "");

  answer.connect("tcp://127.0.0.1:" + std::to_string(col_port));

  int trial = 0;
  while (1) {
    std::cout << "waiting for message " << trial << "\n";
    std::vector<zmq::message_t> send_msgs;
    zmq::recv_multipart(subscriber, std::back_inserter(send_msgs));

    std::string filename = send_msgs[0].to_string();
    std::cout << filename << std::endl;

    std::ofstream detFile(filename, std::ios::out | std::ios::binary);
    detFile << send_msgs[1].to_string();

    // send ack
    std::cout << "sending ack. " << trial << "\n";
    std::string ackStr("All good");
    zmq::message_t ack(ackStr.size());
    memcpy(ack.data(), ackStr.data(), ackStr.size());
    std::cout << ack.to_string() << std::endl;
    answer.send(ack, zmq::send_flags::none);

    trial++;
  }
}
