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

///
/// \file    Location.cxx
/// \author  Julian Myrcha
///

#include "EventVisualisationDataConverter/Location.h"
#include <fairlogger/Logger.h>
#include <sys/socket.h>
#include <unistd.h>
#include <netdb.h>
#include <fcntl.h>
#include <poll.h>
#include <ctime>

using namespace std;

namespace o2::event_visualisation
{

int connect_with_timeout(const int socket, const struct sockaddr* addr, socklen_t addrlen, const unsigned int timeout_ms)
{
  int connection = 0;
  // Setting O_NONBLOCK
  int socket_flags_before;
  if ((socket_flags_before = fcntl(socket, F_GETFL, 0) < 0)) {
    return -1;
  }
  if (fcntl(socket, F_SETFL, socket_flags_before | O_NONBLOCK) < 0) {
    return -1;
  }
  do {
    if (connect(socket, addr, addrlen) < 0) {
      if ((errno != EWOULDBLOCK) && (errno != EINPROGRESS)) {
        connection = -1; // error
      } else {           // wait for complete
        // deadline 'timeout' ms from now
        timespec now; // NOLINT(*-pro-type-member-init)
        if (clock_gettime(CLOCK_MONOTONIC, &now) < 0) {
          connection = -1;
          break;
        }
        const timespec deadline = {.tv_sec = now.tv_sec,
                                   .tv_nsec = now.tv_nsec + timeout_ms * 1000000l};
        do {
          if (clock_gettime(CLOCK_MONOTONIC, &now) < 0) {
            connection = -1;
            break;
          }
          // compute remaining deadline
          const int ms_until_deadline = static_cast<int>((deadline.tv_sec - now.tv_sec) * 1000l + (deadline.tv_nsec - now.tv_nsec) / 1000000l);
          if (ms_until_deadline < 0) {
            connection = 0;
            break;
          }
          pollfd connectionPool[] = {{.fd = socket, .events = POLLOUT}};
          connection = poll(connectionPool, 1, ms_until_deadline);

          if (connection > 0) { // confirm the success
            int error = 0;
            socklen_t len = sizeof(error);
            if (getsockopt(socket, SOL_SOCKET, SO_ERROR, &error, &len) == 0) {
              errno = error;
            }
            if (error != 0) {
              connection = -1;
            }
          }
        } while (connection == -1 && errno == EINTR); // If interrupted, try again.
        if (connection == 0) {
          errno = ETIMEDOUT;
          connection = -1;
        }
      }
    }
  } while (false);
  // Restore socket state
  if (fcntl(socket, F_SETFL, socket_flags_before) < 0) {
    return -1;
  }
  return connection;
}

void Location::open()
{
  if (this->mToFile) {
    this->mOut = new std::ofstream(mFileName, std::ios::out | std::ios::binary);
  }
  if (this->mToSocket) {
    // resolve host name
    sockaddr_in serverAddress; // NOLINT(*-pro-type-member-init)
    serverAddress.sin_family = AF_INET;
    serverAddress.sin_port = htons(this->mPort); // Port number

    // ask once
    static auto server = gethostbyname(this->mHostName.c_str());
    if (server == nullptr) {
      LOGF(info, "Error no such host %s", this->mHostName.c_str());
      return;
    };

    bcopy((char*)server->h_addr,
          (char*)&serverAddress.sin_addr.s_addr,
          server->h_length);

    // Connect to the server
    this->mClientSocket = socket(AF_INET, SOCK_STREAM, 0);
    if (this->mClientSocket == -1) {
      LOGF(info, "Error creating socket");
      return;
    }

    if (connect_with_timeout(this->mClientSocket, (sockaddr*)&serverAddress,
                             sizeof(serverAddress), this->mTimeout) == -1) {
      LOGF(info, "Error connecting to %s:%d", this->mHostName.c_str(), this->mPort);
      ::close(this->mClientSocket);
      this->mClientSocket = -1;
      return;
    }
    try {
      char buf[256] = "SEND:";
      strncpy(buf + 6, this->mFileName.c_str(), sizeof(buf) - 7);
      strncpy(buf + sizeof(buf) - 6, "ALICE", 6);
      auto real = send(this->mClientSocket, buf, sizeof(buf), 0);
      if (real != sizeof(buf)) {
        throw real;
      }
    } catch (...) {
      ::close(this->mClientSocket);
      this->mClientSocket = -1;
      LOGF(info, "Error sending file name to %s:%d", this->mHostName.c_str(), this->mPort);
    }
  }
}

void Location::close()
{
  if (this->mToFile && this->mOut) {
    this->mOut->close();
    delete this->mOut;
    this->mOut = nullptr;
  }
  if (this->mToSocket && this->mClientSocket != -1) {
    ::close(this->mClientSocket);
    this->mClientSocket = -1;
  }
}

void Location::write(char* buf, std::streamsize size)
{
  if (size == 0) {
    return;
  }
  if (this->mToFile && this->mOut) {
    this->mOut->write(buf, size);
  }
  if (this->mToSocket && this->mClientSocket != -1) {
    LOGF(info, "Location::write() socket %s ++++++++++++++++++++++", fileName());
    try {
      auto real = send(this->mClientSocket, buf, size, 0);
      if (real != size) {
        throw real;
      }
    } catch (...) {
      ::close(this->mClientSocket);
      this->mClientSocket = -1;
      LOGF(info, "Error sending data to %s:%d", this->mHostName.c_str(), this->mPort);
    }
  }
}

} // namespace o2::event_visualisation