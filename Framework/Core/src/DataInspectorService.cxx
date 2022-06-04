#include "Framework/DataInspectorService.h"
#include "Framework/ServiceSpec.h"
#include "Framework/ServiceRegistry.h"
#include "Framework/DeviceSpec.h"

namespace o2::framework
{
DataInspectorProxyService::DataInspectorProxyService(const std::string& deviceName, const std::string& address, int port) : deviceName(deviceName), socket(DISocket::connect(address, port))
{
  socket.send(DIMessage{DIMessage::Header::Type::DEVICE_ON, std::string(deviceName)});
}

DataInspectorProxyService::~DataInspectorProxyService()
{
  socket.send(DIMessage{DIMessage::Header::Type::DEVICE_OFF, std::string{deviceName}});
  socket.close();
}

std::unique_ptr<DataInspectorProxyService> DataInspectorProxyService::create(DeviceSpec const& spec, const std::string& address, int port)
{
  return std::make_unique<DataInspectorProxyService>(spec.name, address, port);
}

void DataInspectorProxyService::receive()
{
  if(socket.isReadyToReceive()) {
    DIMessage msg = socket.receive();
    handleMessage(msg);
  }
}

void DataInspectorProxyService::send(DIMessage&& msg)
{
  socket.send(std::move(msg));
}

void DataInspectorProxyService::handleMessage(DIMessage &msg)
{
  switch (msg.header.type) {
    case DIMessage::Header::Type::INSPECT_ON: {
      LOG(info) << "DIService - INSPECT ON";
      _isInspected = true;
      break;
    }
    case DIMessage::Header::Type::INSPECT_OFF: {
      LOG(info) << "DIService - INSPECT OFF";
      _isInspected = false;
      break;
    }
    default: {
      LOG(info) << "DIService - Wrong msg type: " << static_cast<uint32_t>(msg.header.type);
    }
  }
}
}