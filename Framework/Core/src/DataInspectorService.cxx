#include "Framework/DataInspectorService.h"
#include "Framework/ServiceSpec.h"
#include "Framework/ServiceRegistry.h"
#include "Framework/DeviceSpec.h"
#include "Framework/DIMessages.h"

namespace o2::framework
{
DIMessages::RegisterDevice createRegisterMessage(DeviceSpec const& spec, const std::string& id) {
  DIMessages::RegisterDevice msg;
  msg.name = spec.name;
  msg.analysisId = id;

  msg.specs.inputs = std::vector<DIMessages::RegisterDevice::Specs::Input>{};
  std::transform(spec.inputs.begin(), spec.inputs.end(), std::back_inserter(msg.specs.inputs), [](const InputRoute& input) -> DIMessages::RegisterDevice::Specs::Input{
    auto dataDescriptorMatcher = input.matcher.matcher.index() == 1;
    auto origin = dataDescriptorMatcher ? "" : std::get<ConcreteDataMatcher>(input.matcher.matcher).origin.str;
    auto description = dataDescriptorMatcher ? "" : std::get<ConcreteDataMatcher>(input.matcher.matcher).description.str;
    auto subSpec = dataDescriptorMatcher ? 0 : std::get<ConcreteDataMatcher>(input.matcher.matcher).subSpec;

    return DIMessages::RegisterDevice::Specs::Input{
      .binding = input.matcher.binding,
      .sourceChannel = input.sourceChannel,
      .timeslice = input.timeslice,
      .dataDescriptorMatcher = dataDescriptorMatcher,
      .origin = origin,
      .description = description,
      .subSpec = subSpec
    };
  });

  msg.specs.outputs = std::vector<DIMessages::RegisterDevice::Specs::Output>{};
  std::transform(spec.outputs.begin(), spec.outputs.end(), std::back_inserter(msg.specs.outputs), [](const OutputRoute& output) -> DIMessages::RegisterDevice::Specs::Output{
    auto index = output.matcher.matcher.index();
    auto origin = index == 0 ? std::get<ConcreteDataMatcher>(output.matcher.matcher).origin.str : std::get<ConcreteDataTypeMatcher>(output.matcher.matcher).origin.str;
    auto description = index == 0 ? std::get<ConcreteDataMatcher>(output.matcher.matcher).origin.str : std::get<ConcreteDataTypeMatcher>(output.matcher.matcher).origin.str;
    auto subSpec = index == 0 ? std::get<ConcreteDataMatcher>(output.matcher.matcher).subSpec : 0;

    return DIMessages::RegisterDevice::Specs::Output{
      .binding = output.matcher.binding.value,
      .channel = output.channel,
      .timeslice = output.timeslice,
      .maxTimeslices = output.maxTimeslices,
      .origin = origin,
      .description = description,
      .subSpec = subSpec
    };
  });

  msg.specs.forwards = std::vector<DIMessages::RegisterDevice::Specs::Forward>{};
  std::transform(spec.forwards.begin(), spec.forwards.end(), std::back_inserter(msg.specs.forwards), [](const ForwardRoute& forward) -> DIMessages::RegisterDevice::Specs::Forward{
    auto dataDescriptorMatcher = forward.matcher.matcher.index() == 1;
    auto origin = dataDescriptorMatcher ? "" : std::get<ConcreteDataMatcher>(forward.matcher.matcher).origin.str;
    auto description = dataDescriptorMatcher ? "" : std::get<ConcreteDataMatcher>(forward.matcher.matcher).description.str;
    auto subSpec = dataDescriptorMatcher ? 0 : std::get<ConcreteDataMatcher>(forward.matcher.matcher).subSpec;

    return DIMessages::RegisterDevice::Specs::Forward{
      .binding = forward.matcher.binding,
      .timeslice = forward.timeslice,
      .maxTimeslices = forward.maxTimeslices,
      .channel = forward.channel,
      .dataDescriptorMatcher = dataDescriptorMatcher,
      .origin = origin,
      .description = description,
      .subSpec = subSpec
    };
  });

  msg.specs.maxInputTimeslices = spec.maxInputTimeslices;
  msg.specs.inputTimesliceId = spec.inputTimesliceId;
  msg.specs.nSlots = spec.nSlots;
  msg.specs.rank = spec.rank;

  return msg;
}

DataInspectorProxyService::DataInspectorProxyService(DeviceSpec const& spec,
                                                     const std::string& address,
                                                     int port,
                                                     const std::string& id
                                                     ) : deviceName(spec.name),
                                                         socket(DISocket::connect(address, port)),
                                                         id(id)
{
  socket.send(DIMessage{DIMessage::Header::Type::DEVICE_ON, createRegisterMessage(spec, id)});
}

DataInspectorProxyService::~DataInspectorProxyService()
{
  socket.send(DIMessage{DIMessage::Header::Type::DEVICE_OFF, std::string{deviceName}});
  socket.close();
}

std::unique_ptr<DataInspectorProxyService> DataInspectorProxyService::create(DeviceSpec const& spec,
                                                                             const std::string& address,
                                                                             int port,
                                                                             const std::string& id)
{
  return std::make_unique<DataInspectorProxyService>(spec, address, port, id);
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