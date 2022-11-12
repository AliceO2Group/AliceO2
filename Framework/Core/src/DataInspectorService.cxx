#include "Framework/DataInspectorService.h"
#include "Framework/ServiceSpec.h"
#include "Framework/ServiceRegistryRef.h"
#include "Framework/DeviceSpec.h"
#include "Framework/DIMessages.h"
#include "Framework/ControlService.h"

namespace o2::framework
{
DIMessages::RegisterDevice createRegisterMessage(DeviceSpec const& spec, const std::string& runId) {
  DIMessages::RegisterDevice msg;
  msg.name = spec.name;
  msg.runId = runId;

  msg.specs.inputs = std::vector<DIMessages::RegisterDevice::Specs::Input>{};
  std::transform(spec.inputs.begin(), spec.inputs.end(), std::back_inserter(msg.specs.inputs), [](const InputRoute& input) -> DIMessages::RegisterDevice::Specs::Input{
    boost::optional<std::string> origin;
    boost::optional<std::string> description;
    boost::optional<uint32_t> subSpec;

    if (std::holds_alternative<ConcreteDataMatcher>(input.matcher.matcher)) {
      origin = std::get<ConcreteDataMatcher>(input.matcher.matcher).origin.str;
      description = std::get<ConcreteDataMatcher>(input.matcher.matcher).description.str;
      subSpec = std::get<ConcreteDataMatcher>(input.matcher.matcher).subSpec;
    }

    return DIMessages::RegisterDevice::Specs::Input{
      .binding = input.matcher.binding,
      .sourceChannel = input.sourceChannel,
      .timeslice = input.timeslice,
      .origin = origin,
      .description = description,
      .subSpec = subSpec
    };
  });

  msg.specs.outputs = std::vector<DIMessages::RegisterDevice::Specs::Output>{};
  std::transform(spec.outputs.begin(), spec.outputs.end(), std::back_inserter(msg.specs.outputs), [](const OutputRoute& output) -> DIMessages::RegisterDevice::Specs::Output{
    std::string origin;
    std::string description;
    boost::optional<uint32_t> subSpec;

    if (std::holds_alternative<ConcreteDataMatcher>(output.matcher.matcher)) {
      origin = std::get<ConcreteDataMatcher>(output.matcher.matcher).origin.str;
      description = std::get<ConcreteDataMatcher>(output.matcher.matcher).description.str;
      subSpec = std::get<ConcreteDataMatcher>(output.matcher.matcher).subSpec;
    } else {
      origin = std::get<ConcreteDataTypeMatcher>(output.matcher.matcher).origin.str;
      description = std::get<ConcreteDataTypeMatcher>(output.matcher.matcher).description.str;
    }

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
    boost::optional<std::string> origin;
    boost::optional<std::string> description;
    boost::optional<uint32_t> subSpec;

    if (std::holds_alternative<ConcreteDataMatcher>(forward.matcher.matcher)) {
      origin = std::get<ConcreteDataMatcher>(forward.matcher.matcher).origin.str;
      description = std::get<ConcreteDataMatcher>(forward.matcher.matcher).description.str;
      subSpec = std::get<ConcreteDataMatcher>(forward.matcher.matcher).subSpec;
    }

    return DIMessages::RegisterDevice::Specs::Forward{
      .binding = forward.matcher.binding,
      .timeslice = forward.timeslice,
      .maxTimeslices = forward.maxTimeslices,
      .channel = forward.channel,
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

DataInspectorProxyService::DataInspectorProxyService(ServiceRegistryRef serviceRegistry,
                                                     DeviceSpec const& spec,
                                                     const std::string& address,
                                                     int port,
                                                     const std::string& runId
                                                     ) : serviceRegistry(serviceRegistry),
                                                         deviceName(spec.name),
                                                         socket(address, port),
                                                         runId(runId)
{
  try{
    socket.send(DIMessage{DIMessage::Header::Type::DEVICE_ON, createRegisterMessage(spec, runId).toJson()});
  } catch(const std::runtime_error& error) {
    LOG(error) << error.what();
    terminate();
  }
}

DataInspectorProxyService::~DataInspectorProxyService()
{
  try {
    socket.send(DIMessage{DIMessage::Header::Type::DEVICE_OFF, std::string{deviceName}});
  } catch(const std::runtime_error& error) {
    LOG(error) << error.what();
    terminate();
  }
}

void DataInspectorProxyService::receive()
{
  try {
    if(socket.isMessageAvailable()) {
      DIMessage msg = socket.receive();
      handleMessage(msg);
    }
  } catch(const std::runtime_error& error) {
    LOG(error) << error.what();
    terminate();
  }
}

void DataInspectorProxyService::send(DIMessage&& msg)
{
  try {
    socket.send(std::move(msg));
  } catch(const std::runtime_error& error) {
    LOG(error) << error.what();
    terminate();
  }
}

void DataInspectorProxyService::handleMessage(const DIMessage &msg)
{
  switch (msg.header.type()) {
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
    case DIMessage::Header::Type::TERMINATE: {
      LOG(info) << "DIService - TERMINATE";
      terminate();
      break;
    }
    default: {
      LOG(info) << "DIService - Wrong msg type: " << static_cast<uint32_t>(msg.header.type());
    }
  }
}

void DataInspectorProxyService::terminate()
{
  serviceRegistry.get<ControlService>().readyToQuit(QuitRequest::All);
}
}