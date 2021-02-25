#include "Framework/DataInspector.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/DeviceSpec.h"
#include "Framework/OutputObjHeader.h"
#include "Framework/RawDeviceService.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <fcntl.h>
#include <iomanip>
#include <ios>
#include <iostream>
#include <iterator>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <utility>

#include <rapidjson/document.h>
#include <rapidjson/prettywriter.h>
#include <zmq.h>

using namespace rapidjson;

constexpr char PROXY_ADDRESS[]{"ipc:///tmp/proxy"};

class PushSocket
{
 public:
  PushSocket(const std::string& address)
  {
    context = zmq_ctx_new();
    if (context == nullptr) {
      throw std::runtime_error{"Failed to initialize ZeroMQ context"};
    }
    socket = zmq_socket(context, ZMQ_PUSH);
    if (socket == nullptr) {
      throw std::runtime_error{"Failed to initialize ZeroMQ socket"};
    }
    zmq_connect(socket, address.c_str());
  }

  PushSocket(PushSocket& other) = default;

  PushSocket(PushSocket&& other) noexcept
  {
    if (other.context != nullptr) {
      context = other.context;
      other.context = nullptr;
    }
    if (other.socket != nullptr) {
      socket = other.socket;
      other.socket = nullptr;
    }
  }

  ~PushSocket() noexcept
  {
    if (socket != nullptr) {
      zmq_close(socket);
    }
    if (context != nullptr) {
      zmq_ctx_destroy(context);
    }
  }

  void send(const std::string& message)
  {
    if (socket != nullptr) {
      zmq_send(socket, message.c_str(), message.length(), ZMQ_DONTWAIT);
    }
  }

 private:
  void* context{nullptr};
  void* socket{nullptr};
};

class stdout_capture
{
 public:
  stdout_capture() : is_capturing(false)
  {
    fds[READ] = 0;
    fds[WRITE] = 0;
    if (pipe(fds) == -1) {
      throw std::runtime_error{"pipe() error"};
    }
    old_fd = dup(fileno(stdout));
    if (old_fd == -1) {
      throw std::runtime_error{"dup() error"};
    }
  }

  ~stdout_capture()
  {
    if (is_capturing) {
      end();
    }
    if (old_fd > 0) {
      close(old_fd);
    }
    if (fds[READ] > 0) {
      close(fds[READ]);
    }
    if (fds[WRITE] > 0) {
      close(fds[WRITE]);
    }
  }

  void start()
  {
    if (is_capturing) {
      end();
    }
    fflush(stdout);
    dup2(fds[WRITE], fileno(stdout));
    is_capturing = true;
  }

  bool end()
  {
    if (!is_capturing) {
      return false;
    }
    fflush(stdout);
    dup2(old_fd, fileno(stdout));
    out.clear();

    std::string buffer;
    int size, nbytes;

    size = 1024;
    buffer.resize(size);

    while ((nbytes = read(fds[READ], &(*buffer.begin()), size)) >= size) {
      out << buffer;
    }
    if (nbytes > 0) {
      buffer.resize(nbytes);
      out << buffer;
    }
    is_capturing = false;
    return true;
  }

  std::string get() const
  {
    return out.str();
  }

 private:
  enum PIPES { READ,
               WRITE };
  int fds[2], old_fd;
  bool is_capturing, init;
  std::stringstream out;
};

/* Converts the `data` input a string of bytes as two hexadecimal numbers. Adds
   a 128 offset to deal with negative values. */
template <typename T>
static std::string asBytes(const T* data, uint32_t length)
{
  if (length == 0) {
    return "";
  }
  std::stringstream buffer;
  buffer << std::hex << std::setfill('0')
         << std::setw(2) << static_cast<unsigned>(data[0] + 128);
  for (uint32_t i = 1; i < length; i++) {
    buffer << ' ' << std::setw(2) << static_cast<unsigned>(data[i] + 128);
  }
  return buffer.str();
}

/* Replaces (in place) all occurences of `pattern` in `data` with `substitute`.
   `substitute` may be wider than `pattern`. */
static void replaceAll(
  std::string& data,
  const std::string& pattern,
  const std::string& substitute)
{
  std::string::size_type pos = data.find(pattern);
  while (pos != std::string::npos) {
    data.replace(pos, pattern.size(), substitute);
    pos = data.find(pattern, pos + substitute.size());
  }
}

namespace o2
{
namespace framework
{

bool isDataInspectorActive(FairMQDevice& device)
try {
  if (device.fConfig == nullptr) {
    return false;
  }
  return device.fConfig->GetProperty<bool>(INSPECTOR_ACTIVATION_PROPERTY);
} catch (...) {
  return false;
}

static FairMQParts copyMessage(FairMQParts& parts)
{
  FairMQParts partsCopy;
  for (auto& part : parts) {
    FairMQTransportFactory* transport = part->GetTransport();

    FairMQMessagePtr message(transport->CreateMessage());
    message->Copy(*part);

    partsCopy.AddPart(std::move(message));
  }
  return partsCopy;
}

void sendCopyToDataInspector(FairMQDevice& device, FairMQParts& parts, unsigned index)
{
  for (const auto& [name, channels] : device.fChannels) {
    if (name.find("to_DataInspector") != std::string::npos) {
      FairMQParts copy = copyMessage(parts);
      device.Send(copy, name, index);
      return;
    }
  }
}

/* Returns the name of an O2 device which is the source of a route in `routes`
   which matches with `matcher`. If no such route exists, return an empty
   string. */
static std::string findSenderByRoute(
  const std::vector<InputRoute>& routes,
  const InputSpec& matcher)
{
  for (const InputRoute& route : routes) {
    if (route.matcher == matcher) {
      std::string::size_type start = route.sourceChannel.find('_') + 1;
      std::string::size_type end = route.sourceChannel.find('_', start);
      return route.sourceChannel.substr(start, end - start);
    }
  }
  return "";
}

/* Callback which transforms each `DataRef` in `context` to a JSON object and
   sends it on the `socket`. The messages are sent separately. */
static void sendToProxy(std::shared_ptr<PushSocket> socket, ProcessingContext& context)
{
  DeviceSpec device = context.services().get<RawDeviceService>().spec();
  stdout_capture capture;
  for (const DataRef& ref : context.inputs()) {
    Document message;
    message.SetObject();
    Document::AllocatorType& alloc = message.GetAllocator();

    std::string sender = findSenderByRoute(device.inputs, *ref.spec);
    message.AddMember("sender", Value(sender.c_str(), alloc), alloc);

    const header::BaseHeader* baseHeader = header::BaseHeader::get(reinterpret_cast<const byte*>(ref.header));
    for (; baseHeader != nullptr; baseHeader = baseHeader->next()) {
      if (baseHeader->description == header::DataHeader::sHeaderType) {
        const auto* header = header::get<header::DataHeader*>(baseHeader->data());
        std::string origin = header->dataOrigin.as<std::string>();
        std::string description = header->dataDescription.as<std::string>();
        std::string method = header->payloadSerializationMethod.as<std::string>();
        std::string bytes = asBytes(ref.payload, header->payloadSize);

        message.AddMember("origin", Value(origin.c_str(), alloc), alloc);
        message.AddMember("description", Value(description.c_str(), alloc), alloc);
        message.AddMember("subSpecification", Value(header->subSpecification), alloc);
        message.AddMember("firstTForbit", Value(header->firstTForbit), alloc);
        message.AddMember("tfCounter", Value(header->tfCounter), alloc);
        message.AddMember("runNumber", Value(header->runNumber), alloc);
        message.AddMember("payloadSize", Value(header->payloadSize), alloc);
        message.AddMember("splitPayloadParts", Value(header->splitPayloadParts), alloc);
        message.AddMember("payloadSerialization", Value(method.c_str(), alloc), alloc);
        message.AddMember("payloadSplitIndex", Value(header->splitPayloadIndex), alloc);
        message.AddMember("payloadBytes", Value(bytes.c_str(), alloc), alloc);
        if (header->payloadSerializationMethod == header::gSerializationMethodROOT) {
          std::unique_ptr<TObject> object = DataRefUtils::as<TObject>(ref);

          capture.start();
          object->Dump();
          capture.end();

          std::string payload = capture.get();
          replaceAll(payload, "\n", "\\n");
          replaceAll(payload, "\t", "\\5");
          message.AddMember("payload", Value(payload.c_str(), alloc), alloc);
        }
      } else if (baseHeader->description == DataProcessingHeader::sHeaderType) {
        const auto* header = header::get<DataProcessingHeader*>(baseHeader->data());
        message.AddMember("startTime", Value(header->startTime), alloc);
        message.AddMember("duration", Value(header->duration), alloc);
        message.AddMember("creationTimer", Value(header->creation), alloc);
      } else if (baseHeader->description == OutputObjHeader::sHeaderType) {
        const auto* header = header::get<OutputObjHeader*>(baseHeader->data());
        message.AddMember("taskHash", Value(header->mTaskHash), alloc);
      }
    }
    StringBuffer buffer;
    Writer<StringBuffer> writer(buffer);
    message.Accept(writer);

    socket->send(std::string{buffer.GetString(), buffer.GetSize()});
  }
}

static inline bool isNonInternalDevice(const DataProcessorSpec& device)
{
  return device.name.find("internal") == std::string::npos;
}

static inline bool isNonEmptyOutput(const OutputSpec& output)
{
  return !output.binding.value.empty();
}

/* Transforms an `OutputSpec` into an `InputSpec`. For each output route,
   creates a new input route with the same values. */
static inline InputSpec asInputSpec(const OutputSpec& output)
{
  return std::visit([&output](auto&& matcher) {
    using T = std::decay_t<decltype(matcher)>;
    if constexpr (std::is_same_v<T, ConcreteDataMatcher>) {
      return InputSpec{output.binding.value, matcher, output.lifetime};
    } else {
      ConcreteDataMatcher matcher_{matcher.origin, matcher.description, 0};
      return InputSpec{output.binding.value, matcher_, output.lifetime};
    }
  },
                    output.matcher);
}

void addDataInspector(WorkflowSpec& workflow)
{
  DataProcessorSpec dataInspector{"DataInspector"};

  auto pusher = std::make_shared<PushSocket>(PROXY_ADDRESS);
  dataInspector.algorithm = AlgorithmSpec{
    [p{pusher}](ProcessingContext& context) mutable {
      sendToProxy(p, context);
    }};
  for (const DataProcessorSpec& device : workflow) {
    if (isNonInternalDevice(device)) {
      for (const OutputSpec& output : device.outputs) {
        if (isNonEmptyOutput(output)) {
          dataInspector.inputs.emplace_back(asInputSpec(output));
        }
      }
    }
  }
  workflow.emplace_back(std::move(dataInspector));
}

} // namespace framework
} // namespace o2
