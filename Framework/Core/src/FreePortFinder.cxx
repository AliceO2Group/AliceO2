// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "Framework/FreePortFinder.h"

#include <fairlogger/Logger.h>

#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <cstring>

namespace o2
{
namespace framework
{

FreePortFinder::FreePortFinder(unsigned short initialPort, unsigned short finalPort, unsigned short step)
  : mInitialPort{ initialPort },
    mFinalPort{ finalPort },
    mStep{ step },
    mSocket{ socket(AF_INET, SOCK_STREAM, 0) }
{
}

void FreePortFinder::scan()
{
  struct sockaddr_in addr;
  for (mPort = mInitialPort; mPort < mFinalPort; mPort += mStep) {
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(mPort);
    if (bind(mSocket, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
      LOG(WARN) << "Port range [" << mPort << ", " << mPort + mStep
                << "] already taken. Skipping";
      continue;
    }
    LOG(INFO) << "Using port range [" << mPort << ", " << mPort + mStep << "]";
    break;
  }
}

FreePortFinder::~FreePortFinder()
{
  close(mSocket);
}

unsigned short FreePortFinder::port() { return mPort + 1; }
unsigned short FreePortFinder::range() { return mStep; }

} // namespace framework
} // namespace o2
