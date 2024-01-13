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
#include "DISocket.h"
#include <fairlogger/Logger.h>

#define ASIO_CATCH(customMessage)                                                                                               \
  catch (boost::system::system_error & err)                                                                                     \
  {                                                                                                                             \
    auto msg = std::string{err.what()};                                                                                         \
    auto code = std::to_string(err.code().value());                                                                             \
    throw std::runtime_error{"Exception in DataInspector (boost_code=" + code + ", boost_msg=" + msg + ") - " + customMessage}; \
  }

DIMessage::Header::Type DIMessage::Header::type() const
{
  return static_cast<DIMessage::Header::Type>(boost::endian::little_to_native(typeLE));
}

uint64_t DIMessage::Header::payloadSize() const
{
  return boost::endian::little_to_native(payloadSizeLE);
}

DIMessage::DIMessage(const DIMessage& other) noexcept : header(other.header)
{
  this->payload = new char[other.header.payloadSize()];
  std::memcpy(this->payload, other.payload, other.header.payloadSize());
}

DIMessage& DIMessage::operator=(const DIMessage& other) noexcept
{
  if (&other == this) {
    return *this;
  }

  this->header = Header{other.header};

  delete[] payload;
  this->payload = new char[other.header.payloadSize()];
  std::memcpy(this->payload, other.payload, other.header.payloadSize());

  return *this;
}

DIMessage::DIMessage(DIMessage&& other) noexcept : header(other.header), payload(other.payload)
{
  other.payload = nullptr;
}

DIMessage& DIMessage::operator=(DIMessage&& other) noexcept
{
  header = Header{other.header};
  delete[] payload;
  payload = other.payload;

  other.payload = nullptr;
  return *this;
}

DIMessage::~DIMessage()
{
  delete[] payload;
}

DISocket::DISocket(const std::string& address, int port) : ioContext(), socket(ioContext)
{
  try {
    auto ip_address = boost::asio::ip::address::from_string(address);
    socket.connect(boost::asio::ip::tcp::endpoint(ip_address, port));
  }
  ASIO_CATCH("DISocket::DISocket")
}

DISocket::~DISocket()
{
  socket.close();
}

bool DISocket::isMessageAvailable()
{
  return socket.available() >= sizeof(DIMessage::Header);
}

void DISocket::send(const DIMessage& message)
{
  try {
    socket.send(std::array<boost::asio::const_buffer, 2>{
      boost::asio::buffer(&message.header, sizeof(DIMessage::Header)),
      boost::asio::buffer(message.payload, message.header.payloadSize())});
  }
  ASIO_CATCH("DISocket::send")
}

DIMessage DISocket::receive()
{
  try {
    DIMessage message;
    socket.receive(boost::asio::buffer(&message.header, sizeof(DIMessage::Header)));

    if (message.header.payloadSize() > 0) {
      message.payload = new char[message.header.payloadSize()];
      socket.receive(boost::asio::buffer(message.payload, message.header.payloadSize()));
    }

    return message;
  }
  ASIO_CATCH("DISocket::receive")
  return {};
}
