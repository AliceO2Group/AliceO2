/// @file testMessageList.cxx
/// @brief Unit test for message_list.h template class

//  TODO: this is for the moment only testing compilation
//  ideally, this test program is only compiled when the test
//  is going to be executed. By using add_executable in the cmake
//  configuartion, the program is always build (though not installed)

#include "message_list.h"
#include "memory-format.h"

#include <cstring> // memset
#include <iostream>

using namespace o2;
using namespace Format;

// a simple message type, just a pointer to some payload
using SimpleMsg_t = const uint8_t *;

// a simple header definition
struct SimpleHeader_t {
  uint32_t id;
  uint32_t specification;

  SimpleHeader_t() : id(0), specification(0) {}
  SimpleHeader_t(uint32_t _id, uint32_t _spec) : id(_id), specification(_spec) {}
};

// print operator for the simple header
std::ostream& operator<<(std::ostream& stream, SimpleHeader_t header) {
  stream << "Header ID: " << header.id << std::endl;
  stream << "Header Specification: " << std::hex << header.specification;

  return stream;
}

// more complex message type, some class which wraps around payload
// implements type conversion operator to return pointer to payload
// buffer
class TestMsg {
public:
  TestMsg() : mBuffer(nullptr), mBufferSize(0) {}
  TestMsg(const uint8_t* buffer, unsigned size) : mBuffer(new uint8_t[size]), mBufferSize(size) {
    memcpy(mBuffer, buffer, size);
  }
  ~TestMsg() {clear();}

  int alloc(int size) {
    // not yet implemented
    return 0;
  }

  uint8_t* get() const {
    return mBuffer;
  }

  operator uint8_t*() { return get();}

  void clear() {
    delete[] mBuffer;
    mBuffer = nullptr;
    mBufferSize = 0;
  }

private:
  uint8_t* mBuffer;
  unsigned mBufferSize;
};

// helper function to print entries of the message list by using
// iterator
template<typename ListType>
void print_list(ListType& list, typename ListType::HdrComparison hdrsel = typename ListType::HdrComparison()) {
  for (typename ListType::iterator it = list.begin(hdrsel);
       it != list.end();
       ++it) {
    // the iterator defines a conversion operator to the header type
    std::cout << static_cast<typename ListType::header_type>(it) << std::endl;
    // dereferencing of the iterator gives the payload
    std::cout << *it << std::endl;
  }
}

int main(int argc, char** argv)
{
  int iResult = 0;

  typedef messageList<SimpleMsg_t, SimpleHeader_t> SimpleList_t;
  SimpleList_t input;
  input.size();

  SimpleHeader_t hdr1(1, 0xdeadbeef);
  const char* payload1="payload1";
  SimpleMsg_t headerMsg1 = reinterpret_cast<SimpleMsg_t>(&hdr1);
  SimpleMsg_t payloadMsg1 = reinterpret_cast<SimpleMsg_t>(payload1);
  input.add(headerMsg1, payloadMsg1);

  SimpleHeader_t hdr2(2, 0xf00);
  const char* payload2="payload2";
  SimpleMsg_t headerMsg2 = reinterpret_cast<SimpleMsg_t>(&hdr2);
  SimpleMsg_t payloadMsg2 = reinterpret_cast<SimpleMsg_t>(payload2);
  input.add(headerMsg2, payloadMsg2);

  std::cout << "simple list:" << std::endl;
  print_list(input);
  std::cout << std::endl;

  typedef messageList<TestMsg, SimpleHeader_t> TestMsgList_t;
  TestMsgList_t msglist;
  TestMsg testHeaderMsg1((uint8_t*)&hdr1, sizeof(hdr1));
  TestMsg testPayloadMsg1((uint8_t*)payload1, strlen(payload1)+1);
  msglist.add(testHeaderMsg1, testPayloadMsg1);
  TestMsg testHeaderMsg2((uint8_t*)&hdr2, sizeof(hdr2));
  TestMsg testPayloadMsg2((uint8_t*)payload2, strlen(payload2)+1);
  msglist.add(testHeaderMsg2, testPayloadMsg2);

  std::cout << "message class list:" << std::endl;
  print_list(msglist, [](const TestMsgList_t::header_type& hdr){return hdr.specification==0xf00;} );
  std::cout << std::endl;

  return 0;
}
