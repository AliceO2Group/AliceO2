# O2Device

The O2Device is meant to build O2 specific behaviours on top of plain FairMQDevices.

# O2Device utilities

The header **O2Device/Utilities.h** contains utilities that enable consistent handling of 
data in the O2 format and some useful definitions.

```O2Message``` - currently alias of FairMQPArts, contains FairMQMessages in header-payload order

### addDataBlock()
Allows to consistently create O2 compatible messages for FairMQ transport.
Appends a header::Stack-data pair to message. Sets the data size in the DataHeader inside the stack based either on the message/container size.

```
template <typename T>
addDataBlock(O2Message& message, o2::header::Stack&& headerStack, T data);
```
#### arguments:
- headerStack: as the name implies.

- data:
  - ```FairMQMessagePtr``` (```std::unique_ptr<FairMQMessage>```)
  - STL-like container that uses a ```pmr::polymorphic_allocator``` as allocator. Note: this is only efficient if
    the memory resource used to construct the allocator is ```o2::memory_resource::ChannelResource```, otherwise an implicit copy occurs.

### forEach()
Executes a function on each data block within the message.

```
template<typename T>
T forEach(O2Message& message, T function);
```
#### arguments:
the function is a callable object (lambda, functor, std::function) and needs to take two arguents of type gsl::span, first one points to the header buffer, second one points to the payload buffer, e.g. ```[](span header, span payload) {}```.


#### basic example, inside a FairMQDevice:
```C++
// make sure we have the allocator associated with the transport appropriate for the selected channel:
auto outputChannelAllocator = o2::memory_resource::getTransportAllocator(GetChannel("dataOut").Transport());

//the data
using namespace boost::container::pmr;
std::vector<int, polymorphic_allocator<int>> dataVector(polymorphic_allocator<int>{outputChannelAllocator});
dataVector.reserve(10);
dataVector.push_back(1);
dataVector.push_back(2);

//create the message
O2message message;
addDataBlock(message,
             o2::header::Stack{outputChannelAllocator, DataHeader{<args>}},
             std::move(dataVector));

addDataBlock(message,
             o2::header::Stack{outputChannelAllocator, DataHeader{<args>}},
             NewMessageFor("dataOut",0,"some message string"));

Send(message,"dataOut");

//On the receiving end:
O2Message messageIn;
Receive(messageIn, "dataIn");
forEach(messageIn, [](span header, span payload) {
  auto dataHeader = get<DataHeader*>(header.data());
  if (dataHeader->dataDescription==gDataDescriptionMyDataType) {
    MyDataType* myData = reinterpret_cast<MyDataType*>(payload.data());
  }
});

```

