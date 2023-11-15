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

#include <catch_amalgamated.hpp>
#include "Framework/ChannelSpecHelpers.h"
#include <sstream>

using namespace o2::framework;

TEST_CASE("TestChannelMethod")
{
  std::ostringstream oss;
  oss << ChannelSpecHelpers::methodAsString(ChannelMethod::Bind)
      << ChannelSpecHelpers::methodAsString(ChannelMethod::Connect);

  REQUIRE(oss.str() == "bindconnect");
  std::ostringstream oss2;
}

TEST_CASE("TestChannelType")
{
  std::ostringstream oss;
  oss << ChannelSpecHelpers::typeAsString(ChannelType::Pub)
      << ChannelSpecHelpers::typeAsString(ChannelType::Sub)
      << ChannelSpecHelpers::typeAsString(ChannelType::Push)
      << ChannelSpecHelpers::typeAsString(ChannelType::Pull);

  REQUIRE(oss.str() == "pubsubpushpull");
}

struct TestHandler : FairMQChannelConfigParser {
  void beginChannel() override
  {
    str << "@";
  }

  void endChannel() override
  {
    str << "*";
  }
  void error() override
  {
    hasError = true;
  }

  void property(std::string_view k, std::string_view v) override
  {
    str << k << "=" << v;
  }
  bool hasError;
  std::ostringstream str;
};

TEST_CASE("TestChannelParser")
{
  TestHandler h;
  ChannelSpecHelpers::parseChannelConfig("name=foo", h);
  REQUIRE(h.str.str() == "@name=foo*");
  TestHandler h1;
  ChannelSpecHelpers::parseChannelConfig("name=foo,bar=fur", h1);
  REQUIRE(h1.str.str() == "@name=foobar=fur*");
  TestHandler h2;
  ChannelSpecHelpers::parseChannelConfig("name=foo,bar=fur,abc=cdf;name=bar,foo=a", h2);
  REQUIRE(h2.str.str() == "@name=foobar=furabc=cdf*@name=barfoo=a*");
  TestHandler h3;
  ChannelSpecHelpers::parseChannelConfig("foo:bar=fur,abc=cdf;name=bar,foo=a", h3);
  REQUIRE(h3.str.str() == "@name=foobar=furabc=cdf*@name=barfoo=a*");

  // Test the OutputChannelSpecConfigParser::parseChannelConfig method
  OutputChannelSpecConfigParser p;
  ChannelSpecHelpers::parseChannelConfig("name=foo", p);
  REQUIRE(p.specs.size() == 1);
  REQUIRE(p.specs.back().name == "foo");

  OutputChannelSpecConfigParser p2;
  ChannelSpecHelpers::parseChannelConfig("foo:", p2);
  REQUIRE(p2.specs.size() == 1);
  REQUIRE(p2.specs.back().name == "foo");

  OutputChannelSpecConfigParser p3;
  ChannelSpecHelpers::parseChannelConfig("name=foo,method=bind,type=pub", p3);
  REQUIRE(p3.specs.size() == 1);
  REQUIRE(p3.specs.back().name == "foo");
  REQUIRE(p3.specs.back().method == ChannelMethod::Bind);
  REQUIRE(p3.specs.back().type == ChannelType::Pub);

  // Check the case for a channel with a protocol TPC
  OutputChannelSpecConfigParser p4;
  ChannelSpecHelpers::parseChannelConfig("name=foo,method=connect,type=sub,address=tcp://127.0.0.2:8080", p4);
  REQUIRE(p4.specs.size() == 1);
  REQUIRE(p4.specs.back().name == "foo");
  REQUIRE(p4.specs.back().method == ChannelMethod::Connect);
  REQUIRE(p4.specs.back().type == ChannelType::Sub);
  REQUIRE(p4.specs.back().hostname == "127.0.0.2");
  REQUIRE(p4.specs.back().port == 8080);
  REQUIRE(p4.specs.back().protocol == ChannelProtocol::Network);

  // Check the case for a channel with a protocol IPC
  OutputChannelSpecConfigParser p5;
  ChannelSpecHelpers::parseChannelConfig("name=foo,method=connect,type=sub,address=ipc://@some_ipc_file_8080", p5);
  REQUIRE(p5.specs.size() == 1);
  REQUIRE(p5.specs.back().name == "foo");
  REQUIRE(p5.specs.back().method == ChannelMethod::Connect);
  REQUIRE(p5.specs.back().type == ChannelType::Sub);
  REQUIRE(p5.specs.back().hostname == "@some_ipc_file_8080");
  REQUIRE(p5.specs.back().port == 0);
  REQUIRE(p5.specs.back().protocol == ChannelProtocol::IPC);
}
