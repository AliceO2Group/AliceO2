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

#include <cstdio>
#include <string>
#include <map>
#include <thread>

using map_t = std::map<std::string, std::string>;

void* threadfunc(void* p)
{
  map_t& m = *(map_t*)p;
  m["foo"] = "bar";
  return nullptr;
}

int main()
{
  map_t m;
  // Create a thread in C++11 mode, which executes threadfunc(&m)
  std::thread thread(threadfunc, &m);
  // Start
  printf("foo=%s\n", m["foo"].c_str());
  thread.join();
}
