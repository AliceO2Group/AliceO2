#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/smart_ptr/shared_ptr.hpp>
#include <boost/interprocess/containers/list.hpp>
#include <boost/interprocess/allocators/allocator.hpp>
#include <iostream>
#include <cstring>
#include <zmq.h>
#include <cstdlib> //std::system
#include <cstddef>
#include <cassert>
#include <utility>
#include <chrono>
#include <thread>
#include "../include/O2device/SharedMemory.h"

using namespace AliceO2;
namespace bi = boost::interprocess;

int main(int argc, char* argv[]) {

	void* zmqContext = zmq_ctx_new();
	void* sender = zmq_socket(zmqContext, ZMQ_PUSH);
	int rc = zmq_bind(sender, "tcp://*:2223");
	if (rc<0) printf("zmq connection err: %s\n",zmq_strerror(errno));

  //Construct managed shared memory
  SharedMemory::Manager segman(2000000);

  int nErrors = 0;
  size_t mMsgSize = 500;
  std::string shmPointerID;

  int niter = 1000000;
  int i = 0;
  for (i=0; i<niter; ++i) {
    auto id = segman.getUniqueID();
    auto blockPtr = segman.allocate(mMsgSize, id);
    //printf("allocated: %p",blockPtr);
    while (!blockPtr) {
      //printf("waiting\n");
      segman.waitForMemory();
      blockPtr = segman.allocate(mMsgSize, id);
    }

    std::string datastr{"some data "};
    datastr+=std::to_string(id);
    strcpy((char*)blockPtr->data(),datastr.c_str());

    //managed_shared_memory::handle_t handle = 1000;
    SharedMemory::HandleType handle = blockPtr->getHandle();
    //std::cout << " handle: " << handle << std::endl;
    rc = zmq_send(sender, &handle, sizeof(handle), 0);

    blockPtr.release();
    //if (rc<0) printf("send rc %i, error: %s\n", rc, zmq_strerror(errno));
    if (rc<0) ++nErrors;
  }

  zmq_send(sender, 0, 0, 0); //signal end
  printf("sent messages: %d, errors: %d\n",i,nErrors);
	zmq_close(sender);
	zmq_ctx_term(zmqContext);
	return 0;
}
