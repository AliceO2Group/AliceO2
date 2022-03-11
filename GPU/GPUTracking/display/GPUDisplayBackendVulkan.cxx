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

/// \file GPUDisplayBackendVulkan.cxx
/// \author David Rohr

#include <vulkan/vulkan.hpp>
#include <GLFW/glfw3.h>

#include "GPUCommonDef.h"
#include "GPUDisplayBackendVulkan.h"
#include "GPUDisplayShaders.h"
#include "GPUDisplay.h"
#include "GPUDisplayFrontendGlfw.h"

using namespace GPUCA_NAMESPACE::gpu;

extern "C" char _binary_shaders_display_shaders_vertex_vert_spv_start[];
extern "C" char _binary_shaders_display_shaders_vertex_vert_spv_end[];
static size_t _binary_shaders_display_shaders_vertex_vert_spv_len = _binary_shaders_display_shaders_vertex_vert_spv_end - _binary_shaders_display_shaders_vertex_vert_spv_start;
extern "C" char _binary_shaders_display_shaders_fragment_frag_spv_start[];
extern "C" char _binary_shaders_display_shaders_fragment_frag_spv_end[];
static size_t _binary_shaders_display_shaders_fragment_frag_spv_len = _binary_shaders_display_shaders_fragment_frag_spv_end - _binary_shaders_display_shaders_fragment_frag_spv_start;

namespace GPUCA_NAMESPACE::gpu
{
struct QueueFamiyIndices {
  uint32_t graphicsFamily;
};
struct SwapChainSupportDetails {
  VkSurfaceCapabilitiesKHR capabilities;
  std::vector<VkSurfaceFormatKHR> formats;
  std::vector<VkPresentModeKHR> presentModes;
};
struct VulkanBuffer {
  VkBuffer buffer;
  VkDeviceMemory memory;
  size_t size;
  bool deviceMemory;
};
} // namespace GPUCA_NAMESPACE::gpu

//#define CHKERR(cmd) {cmd;}
#define CHKERR(cmd)                                                                               \
  do {                                                                                            \
    auto tmp_internal_retVal = cmd;                                                               \
    if (tmp_internal_retVal < 0) {                                                                \
      GPUError("Vulkan Error %d: %s (%s: %d)", tmp_internal_retVal, "ERROR", __FILE__, __LINE__); \
      throw std::runtime_error("Vulkan Failure");                                                 \
    }                                                                                             \
  } while (false)

GPUDisplayBackendVulkan::GPUDisplayBackendVulkan()
{
  mQueueFamilyIndices = std::make_unique<QueueFamiyIndices>();
  mSwapChainDetails = std::make_unique<SwapChainSupportDetails>();
  mVBO.resize(GPUCA_NSLICES);
  mIndirectCommandBuffer.resize(1);
}
GPUDisplayBackendVulkan::~GPUDisplayBackendVulkan() = default;

int GPUDisplayBackendVulkan::ExtInit()
{
  return 0;
}

bool GPUDisplayBackendVulkan::CoreProfile()
{
  return false;
}

unsigned int GPUDisplayBackendVulkan::DepthBits()
{
  return 0;
}

void GPUDisplayBackendVulkan::createFB(GLfb& fb, bool tex, bool withDepth, bool msaa)
{
  fb.tex = tex;
  fb.depth = withDepth;
  fb.msaa = msaa;
  fb.created = true;
}

void GPUDisplayBackendVulkan::deleteFB(GLfb& fb)
{
  fb.created = false;
}

unsigned int GPUDisplayBackendVulkan::drawVertices(const vboList& v, const drawType tt)
{
  auto first = std::get<0>(v);
  auto count = std::get<1>(v);
  auto iSlice = std::get<2>(v);
  if (count == 0) {
    return 0;
  }
  if (mCommandBufferUpToDate[mImageIndex]) {
    return count;
  }

  vkCmdBindPipeline(mCommandBuffers[mImageIndex], VK_PIPELINE_BIND_POINT_GRAPHICS, mPipelines[tt]);
  if (mDisplay->cfgR().useGLIndirectDraw) {
    vkCmdDrawIndirect(mCommandBuffers[mImageIndex], mIndirectCommandBuffer[0].buffer, (mIndirectSliceOffset[iSlice] + first) * sizeof(DrawArraysIndirectCommand), count, sizeof(DrawArraysIndirectCommand));
  } else {
    for (unsigned int k = 0; k < count; k++) {
      vkCmdDraw(mCommandBuffers[mImageIndex], mDisplay->vertexBufferCount()[iSlice][first + k], 1, mDisplay->vertexBufferStart()[iSlice][first + k], 0);
    }
  }

  return count;
}

void GPUDisplayBackendVulkan::ActivateColor(std::array<float, 3>& color)
{
  writeToBuffer(mUniformBuffersCol[mImageIndex], sizeof(color), color.data());
  if (mCommandBufferUpToDate[mImageIndex]) {
    return;
  }
  vkCmdPushConstants(mCommandBuffers[mImageIndex], mPipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(color), color.data());
}

void GPUDisplayBackendVulkan::setQuality()
{
}

void GPUDisplayBackendVulkan::setDepthBuffer()
{
}

void GPUDisplayBackendVulkan::setFrameBuffer(int updateCurrent, unsigned int newID)
{
}

static int checkValidationLayerSupport(const std::vector<const char*>& validationLayers)
{
  uint32_t layerCount;
  vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
  std::vector<VkLayerProperties> availableLayers(layerCount);
  vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());
  for (const char* layerName : validationLayers) {
    bool layerFound = false;

    for (const auto& layerProperties : availableLayers) {
      if (strcmp(layerName, layerProperties.layerName) == 0) {
        layerFound = true;
        break;
      }
    }

    if (!layerFound) {
      return 1;
    }
  }
  return 0;
}

static VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger)
{
  auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
  if (func != nullptr) {
    return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
  } else {
    return VK_ERROR_EXTENSION_NOT_PRESENT;
  }
}

static void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator)
{
  auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
  if (func != nullptr) {
    func(instance, debugMessenger, pAllocator);
  }
}

template <class T>
static void setupDebugMessenger(VkInstance& instance, VkDebugUtilsMessengerEXT& debugMessenger, T& debugCallback)
{
  VkDebugUtilsMessengerCreateInfoEXT createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
  createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
  createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
  createInfo.pfnUserCallback = debugCallback;
  createInfo.pUserData = nullptr;
  if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS) {
    throw std::runtime_error("Error setting up debug messenger!");
  }
}

static VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats)
{
  for (const auto& availableFormat : availableFormats) {
    if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
      return availableFormat;
    }
  }
  return availableFormats[0];
}

static VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes)
{
  for (const auto& availablePresentMode : availablePresentModes) {
    if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
      return availablePresentMode;
    }
  }
  return VK_PRESENT_MODE_FIFO_KHR;
}

static VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities, GLFWwindow* window)
{
  if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
    return capabilities.currentExtent;
  } else {
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);

    VkExtent2D actualExtent = {(uint32_t)width, (uint32_t)height};
    actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
    actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);
    return actualExtent;
  }
}

static VkShaderModule createShaderModule(const char* code, size_t size, VkDevice device)
{
  VkShaderModuleCreateInfo createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  createInfo.codeSize = size;
  createInfo.pCode = reinterpret_cast<const uint32_t*>(code);
  VkShaderModule shaderModule;
  CHKERR(vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule));
  return shaderModule;
}

void GPUDisplayBackendVulkan::startFillCommandBuffer(VkCommandBuffer& commandBuffer, unsigned int imageIndex)
{
  vkResetCommandBuffer(commandBuffer, 0);

  VkCommandBufferBeginInfo beginInfo{};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags = 0;                  // Optional
  beginInfo.pInheritanceInfo = nullptr; // Optional
  CHKERR(vkBeginCommandBuffer(commandBuffer, &beginInfo));

  VkRenderPassBeginInfo renderPassInfo{};
  renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
  renderPassInfo.renderPass = mRenderPass;
  renderPassInfo.framebuffer = mFramebuffers[imageIndex];
  renderPassInfo.renderArea.offset = {0, 0};
  renderPassInfo.renderArea.extent = mExtent;

  VkClearValue clearColor = {{{0.0f, 0.0f, 0.0f, 1.0f}}};
  renderPassInfo.clearValueCount = 1;
  renderPassInfo.pClearValues = &clearColor;
  vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

  VkDeviceSize offsets[] = {0};
  vkCmdBindVertexBuffers(commandBuffer, 0, 1, &mVBO[0].buffer, offsets);
  vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, mPipelineLayout, 0, 1, &mDescriptorSets[mImageIndex], 0, nullptr);
}

void GPUDisplayBackendVulkan::endFillCommandBuffer(VkCommandBuffer& commandBuffer, unsigned int imageIndex)
{
  vkCmdEndRenderPass(commandBuffer);
  CHKERR(vkEndCommandBuffer(commandBuffer));
}

void GPUDisplayBackendVulkan::updateSwapChainDetails(const VkPhysicalDevice& device)
{
  vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, mSurface, &mSwapChainDetails->capabilities);
  uint32_t formatCount;
  vkGetPhysicalDeviceSurfaceFormatsKHR(device, mSurface, &formatCount, nullptr);
  mSwapChainDetails->formats.resize(formatCount);
  vkGetPhysicalDeviceSurfaceFormatsKHR(device, mSurface, &formatCount, mSwapChainDetails->formats.data());
  uint32_t presentModeCount;
  vkGetPhysicalDeviceSurfacePresentModesKHR(device, mSurface, &presentModeCount, nullptr);
  mSwapChainDetails->presentModes.resize(presentModeCount);
  vkGetPhysicalDeviceSurfacePresentModesKHR(device, mSurface, &presentModeCount, mSwapChainDetails->presentModes.data());
}

void GPUDisplayBackendVulkan::createDevice()
{
  VkApplicationInfo appInfo{};
  appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  appInfo.pApplicationName = "Hello Triangle";
  appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.pEngineName = "No Engine";
  appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.apiVersion = VK_API_VERSION_1_0;

  VkInstanceCreateInfo instanceCreateInfo{};
  instanceCreateInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  instanceCreateInfo.pApplicationInfo = &appInfo;

  uint32_t glfwExtensionCount = 0;
  const char** glfwExtensions;
  glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
  std::vector<const char*> reqInstanceExtensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

  const std::vector<const char*> reqValidationLayers = {
    "VK_LAYER_KHRONOS_validation"};
  auto debugCallback = [](VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData) -> VkBool32 {
    switch (messageSeverity) {
      case VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT:
        // GPUInfo("%s", pCallbackData->pMessage);
        break;
      case VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT:
        GPUWarning("%s", pCallbackData->pMessage);
        break;
      case VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT:
        GPUError("%s", pCallbackData->pMessage);
        break;
      case VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT:
      default:
        GPUInfo("%s", pCallbackData->pMessage);
        break;
    }
    return VK_FALSE;
  };
  if (mEnableValidationLayers) {
    if (checkValidationLayerSupport(reqValidationLayers)) {
      throw std::runtime_error("Requested validation layer support not available");
    }
    reqInstanceExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    instanceCreateInfo.enabledLayerCount = static_cast<uint32_t>(reqValidationLayers.size());
    instanceCreateInfo.ppEnabledLayerNames = reqValidationLayers.data();

  } else {
    instanceCreateInfo.enabledLayerCount = 0;
  }

  instanceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(reqInstanceExtensions.size());
  instanceCreateInfo.ppEnabledExtensionNames = reqInstanceExtensions.data();

  CHKERR(vkCreateInstance(&instanceCreateInfo, nullptr, &mInstance));
  if (mEnableValidationLayers) {
    setupDebugMessenger(mInstance, mDebugMessenger, debugCallback);
  }
  uint32_t instanceExtensionCount = 0;
  vkEnumerateInstanceExtensionProperties(nullptr, &instanceExtensionCount, nullptr);
  std::vector<VkExtensionProperties> extensions(instanceExtensionCount);
  vkEnumerateInstanceExtensionProperties(nullptr, &instanceExtensionCount, extensions.data());
  if (mEnableValidationLayers) {
    std::cout << "available extensions: " << instanceExtensionCount << "\n";
    for (const auto& extension : extensions) {
      std::cout << '\t' << extension.extensionName << '\n';
    }
  }

  /*VkWin32SurfaceCreateInfoKHR surfaceCreateInfo{};
  surfaceCreateInfo.sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
  surfaceCreateInfo.hwnd = glfwGetWin32Window(window);
  surfaceCreateInfo.hinstance = GetModuleHandle(nullptr);
  CHKERR(vkCreateWin32SurfaceKHR(instance, &surfaceCreateInfo, nullptr, &surface))*/
  CHKERR(glfwCreateWindowSurface(mInstance, ((GPUDisplayFrontendGlfw*)mDisplay->frontend())->Window(), nullptr, &mSurface));

  const std::vector<const char*> reqDeviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME};

  mPhysicalDevice = VK_NULL_HANDLE;
  uint32_t deviceCount = 0;
  vkEnumeratePhysicalDevices(mInstance, &deviceCount, nullptr);
  if (deviceCount == 0) {
    throw std::runtime_error("No Vulkan device present!");
  }
  std::vector<VkPhysicalDevice> devices(deviceCount);
  vkEnumeratePhysicalDevices(mInstance, &deviceCount, devices.data());
  for (const auto& device : devices) {
    VkPhysicalDeviceProperties deviceProperties;
    vkGetPhysicalDeviceProperties(device, &deviceProperties);
    VkPhysicalDeviceFeatures deviceFeatures;
    vkGetPhysicalDeviceFeatures(device, &deviceFeatures);
    if (deviceProperties.deviceType != VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ||
        !deviceFeatures.geometryShader) {
      continue;
    }

    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);
    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());
    bool found = false;
    for (unsigned int i = 0; i < queueFamilies.size(); i++) {
      if (!(queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
        continue;
      }
      VkBool32 presentSupport = false;
      vkGetPhysicalDeviceSurfaceSupportKHR(device, i, mSurface, &presentSupport);
      if (!presentSupport) {
        continue;
      }
      mQueueFamilyIndices->graphicsFamily = i;
      found = true;
      break;
    }
    if (!found) {
      GPUInfo("%s ignored due to missing queue properties", deviceProperties.deviceName);
      continue;
    }

    uint32_t deviceExtensionCount;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &deviceExtensionCount, nullptr);
    std::vector<VkExtensionProperties> availableExtensions(deviceExtensionCount);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &deviceExtensionCount, availableExtensions.data());
    unsigned int extensionsFound = 0;
    for (unsigned int i = 0; i < reqDeviceExtensions.size(); i++) {
      for (unsigned int j = 0; j < availableExtensions.size(); j++) {
        if (strcmp(reqDeviceExtensions[i], availableExtensions[j].extensionName) == 0) {
          extensionsFound++;
          break;
        }
      }
    }
    if (extensionsFound < reqDeviceExtensions.size()) {
      GPUInfo("%s ignored due to missing extensions", deviceProperties.deviceName);
      continue;
    }

    updateSwapChainDetails(device);
    if (mSwapChainDetails->formats.empty() || mSwapChainDetails->presentModes.empty()) {
      GPUInfo("%s ignored due to incompatible swap chain", deviceProperties.deviceName);
      continue;
    }

    mPhysicalDevice = device;
    GPUInfo("Using physicak Vulkan device %s", deviceProperties.deviceName);
    break;
  }

  if (mPhysicalDevice == VK_NULL_HANDLE) {
    throw std::runtime_error("All available Vulkan devices unsuited");
  }
  updateSwapChainDetails(mPhysicalDevice);
  uint32_t imageCount = mSwapChainDetails->capabilities.minImageCount + 1;
  if (mSwapChainDetails->capabilities.maxImageCount > 0 && imageCount > mSwapChainDetails->capabilities.maxImageCount) {
    imageCount = mSwapChainDetails->capabilities.maxImageCount;
  }
  mImageCount = imageCount;
  mFramesInFlight = mImageCount; // Simplifies reuse of command buffers

  VkDeviceQueueCreateInfo queueCreateInfo{};
  queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  queueCreateInfo.queueFamilyIndex = mQueueFamilyIndices->graphicsFamily;
  queueCreateInfo.queueCount = 1;
  float queuePriority = 1.0f;
  queueCreateInfo.pQueuePriorities = &queuePriority;
  VkPhysicalDeviceFeatures deviceFeatures;
  VkDeviceCreateInfo deviceCreateInfo{};
  deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
  deviceCreateInfo.queueCreateInfoCount = 1;
  deviceCreateInfo.pEnabledFeatures = &deviceFeatures;
  deviceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(reqDeviceExtensions.size());
  deviceCreateInfo.ppEnabledExtensionNames = reqDeviceExtensions.data();
  deviceCreateInfo.enabledLayerCount = instanceCreateInfo.enabledLayerCount;
  deviceCreateInfo.ppEnabledLayerNames = instanceCreateInfo.ppEnabledLayerNames;
  CHKERR(vkCreateDevice(mPhysicalDevice, &deviceCreateInfo, nullptr, &mDevice));
  vkGetDeviceQueue(mDevice, mQueueFamilyIndices->graphicsFamily, 0, &mGraphicsQueue);

  VkCommandPoolCreateInfo poolInfo{};
  poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  poolInfo.queueFamilyIndex = mQueueFamilyIndices->graphicsFamily;
  CHKERR(vkCreateCommandPool(mDevice, &poolInfo, nullptr, &mCommandPool));

  VkCommandBufferAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocInfo.commandPool = mCommandPool;
  allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandBufferCount = mFramesInFlight;
  mCommandBuffers.resize(mFramesInFlight);
  mCommandBufferUpToDate.resize(mFramesInFlight, false);
  CHKERR(vkAllocateCommandBuffers(mDevice, &allocInfo, mCommandBuffers.data()));

  VkSemaphoreCreateInfo semaphoreInfo{};
  semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
  VkFenceCreateInfo fenceInfo{};
  fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
  mImageAvailableSemaphore.resize(mFramesInFlight);
  mRenderFinishedSemaphore.resize(mFramesInFlight);
  mInFlightFence.resize(mFramesInFlight);
  mUniformBuffersMat.resize(mFramesInFlight);
  mUniformBuffersCol.resize(mFramesInFlight);
  for (unsigned int i = 0; i < mFramesInFlight; i++) {
    CHKERR(vkCreateSemaphore(mDevice, &semaphoreInfo, nullptr, &mImageAvailableSemaphore[i]));
    CHKERR(vkCreateSemaphore(mDevice, &semaphoreInfo, nullptr, &mRenderFinishedSemaphore[i]));
    CHKERR(vkCreateFence(mDevice, &fenceInfo, nullptr, &mInFlightFence[i]));
    mUniformBuffersMat[i] = createBuffer(sizeof(hmm_mat4), nullptr, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, false);
    mUniformBuffersCol[i] = createBuffer(sizeof(float) * 3, nullptr, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, false);
  }
}

void GPUDisplayBackendVulkan::clearDevice()
{
  for (unsigned int i = 0; i < mImageAvailableSemaphore.size(); i++) {
    vkDestroySemaphore(mDevice, mImageAvailableSemaphore[i], nullptr);
    vkDestroySemaphore(mDevice, mRenderFinishedSemaphore[i], nullptr);
    vkDestroyFence(mDevice, mInFlightFence[i], nullptr);
    clearBuffer(mUniformBuffersMat[i]);
    clearBuffer(mUniformBuffersCol[i]);
  }
  vkDestroyCommandPool(mDevice, mCommandPool, nullptr);
  vkDestroyDevice(mDevice, nullptr);
  vkDestroySurfaceKHR(mInstance, mSurface, nullptr);
  if (mEnableValidationLayers) {
    DestroyDebugUtilsMessengerEXT(mInstance, mDebugMessenger, nullptr);
  }
  vkDestroyInstance(mInstance, nullptr);
}

void GPUDisplayBackendVulkan::createSwapChain()
{
  updateSwapChainDetails(mPhysicalDevice);
  mSurfaceFormat = chooseSwapSurfaceFormat(mSwapChainDetails->formats);
  mPresentMode = chooseSwapPresentMode(mSwapChainDetails->presentModes);
  mExtent = chooseSwapExtent(mSwapChainDetails->capabilities, ((GPUDisplayFrontendGlfw*)mDisplay->frontend())->Window());

  VkSwapchainCreateInfoKHR swapCreateInfo{};
  swapCreateInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
  swapCreateInfo.surface = mSurface;
  swapCreateInfo.minImageCount = mImageCount;
  swapCreateInfo.imageFormat = mSurfaceFormat.format;
  swapCreateInfo.imageColorSpace = mSurfaceFormat.colorSpace;
  swapCreateInfo.imageExtent = mExtent;
  swapCreateInfo.imageArrayLayers = 1;
  swapCreateInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
  swapCreateInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
  swapCreateInfo.queueFamilyIndexCount = 0;     // Optional
  swapCreateInfo.pQueueFamilyIndices = nullptr; // Optional
  swapCreateInfo.preTransform = mSwapChainDetails->capabilities.currentTransform;
  swapCreateInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
  swapCreateInfo.presentMode = mPresentMode;
  swapCreateInfo.clipped = VK_TRUE;
  swapCreateInfo.oldSwapchain = VK_NULL_HANDLE;
  CHKERR(vkCreateSwapchainKHR(mDevice, &swapCreateInfo, nullptr, &mSwapChain));

  vkGetSwapchainImagesKHR(mDevice, mSwapChain, &mImageCount, nullptr);
  mImages.resize(mImageCount);
  vkGetSwapchainImagesKHR(mDevice, mSwapChain, &mImageCount, mImages.data());

  VkAttachmentDescription colorAttachment{};
  colorAttachment.format = mSurfaceFormat.format;
  colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
  colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

  VkAttachmentReference colorAttachmentRef{};
  colorAttachmentRef.attachment = 0;
  colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  VkSubpassDescription subpass{};
  subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
  subpass.colorAttachmentCount = 1;
  subpass.pColorAttachments = &colorAttachmentRef;
  VkSubpassDependency dependency{};
  dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
  dependency.dstSubpass = 0;
  dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  dependency.srcAccessMask = 0;
  dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  VkRenderPassCreateInfo renderPassInfo{};
  renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
  renderPassInfo.attachmentCount = 1;
  renderPassInfo.pAttachments = &colorAttachment;
  renderPassInfo.subpassCount = 1;
  renderPassInfo.pSubpasses = &subpass;
  renderPassInfo.dependencyCount = 1;
  renderPassInfo.pDependencies = &dependency;
  CHKERR(vkCreateRenderPass(mDevice, &renderPassInfo, nullptr, &mRenderPass));

  mImageViews.resize(mImages.size());
  mFramebuffers.resize(mImages.size());
  for (size_t i = 0; i < mImages.size(); i++) {
    VkImageViewCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    createInfo.image = mImages[i];
    createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    createInfo.format = mSurfaceFormat.format;
    createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
    createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
    createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
    createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
    createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    createInfo.subresourceRange.baseMipLevel = 0;
    createInfo.subresourceRange.levelCount = 1;
    createInfo.subresourceRange.baseArrayLayer = 0;
    createInfo.subresourceRange.layerCount = 1;
    CHKERR(vkCreateImageView(mDevice, &createInfo, nullptr, &mImageViews[i]));

    VkFramebufferCreateInfo framebufferInfo{};
    framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    framebufferInfo.renderPass = mRenderPass;
    framebufferInfo.attachmentCount = 1;
    framebufferInfo.pAttachments = &mImageViews[i];
    framebufferInfo.width = mExtent.width;
    framebufferInfo.height = mExtent.height;
    framebufferInfo.layers = 1;
    CHKERR(vkCreateFramebuffer(mDevice, &framebufferInfo, nullptr, &mFramebuffers[i]));
  }
}

void GPUDisplayBackendVulkan::createPipeline()
{
  VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
  vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
  vertShaderStageInfo.module = mModuleVertex;
  vertShaderStageInfo.pName = "main";
  VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
  fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
  fragShaderStageInfo.module = mModuleFragment;
  fragShaderStageInfo.pName = "main";
  VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

  VkVertexInputBindingDescription bindingDescription{};
  bindingDescription.binding = 0;
  bindingDescription.stride = 3 * sizeof(float);
  bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

  VkVertexInputAttributeDescription attributeDescriptions{};
  attributeDescriptions.binding = 0;
  attributeDescriptions.location = 0;
  attributeDescriptions.format = VK_FORMAT_R32G32B32_SFLOAT;
  attributeDescriptions.offset = 0;

  VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
  vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
  vertexInputInfo.vertexBindingDescriptionCount = 1;
  vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
  vertexInputInfo.vertexAttributeDescriptionCount = 1;
  vertexInputInfo.pVertexAttributeDescriptions = &attributeDescriptions;
  VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
  inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  // inputAssembly.topology
  inputAssembly.primitiveRestartEnable = VK_FALSE;

  VkViewport viewport{};
  viewport.x = 0.0f;
  viewport.y = 0.0f;
  viewport.width = (float)mExtent.width;
  viewport.height = (float)mExtent.height;
  viewport.minDepth = 0.0f;
  viewport.maxDepth = 1.0f;

  VkRect2D scissor{};
  scissor.offset = {0, 0};
  scissor.extent = mExtent;

  VkPipelineViewportStateCreateInfo viewportState{};
  viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
  viewportState.viewportCount = 1;
  viewportState.pViewports = &viewport;
  viewportState.scissorCount = 1;
  viewportState.pScissors = &scissor;

  VkPipelineRasterizationStateCreateInfo rasterizer{};
  rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
  rasterizer.depthClampEnable = VK_FALSE;
  rasterizer.rasterizerDiscardEnable = VK_FALSE;
  rasterizer.polygonMode = VK_POLYGON_MODE_FILL; // TODO: change me!
  rasterizer.lineWidth = 1.0f;                   // TODO: change me
  rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
  rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
  rasterizer.depthBiasEnable = VK_FALSE;
  rasterizer.depthBiasConstantFactor = 0.0f; // Optional
  rasterizer.depthBiasClamp = 0.0f;          // Optional
  rasterizer.depthBiasSlopeFactor = 0.0f;    // Optional

  VkPipelineMultisampleStateCreateInfo multisampling{};
  multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
  multisampling.sampleShadingEnable = VK_FALSE;
  multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
  multisampling.minSampleShading = 1.0f;          // Optional
  multisampling.pSampleMask = nullptr;            // Optional
  multisampling.alphaToCoverageEnable = VK_FALSE; // Optional
  multisampling.alphaToOneEnable = VK_FALSE;      // Optional

  VkPipelineColorBlendAttachmentState colorBlendAttachment{};
  colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
  colorBlendAttachment.blendEnable = VK_FALSE;

  VkPipelineColorBlendStateCreateInfo colorBlending{};
  colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  colorBlending.logicOpEnable = VK_FALSE;
  colorBlending.logicOp = VK_LOGIC_OP_COPY;
  colorBlending.attachmentCount = 1;
  colorBlending.pAttachments = &colorBlendAttachment;
  colorBlending.blendConstants[0] = 0.0f;
  colorBlending.blendConstants[1] = 0.0f;
  colorBlending.blendConstants[2] = 0.0f;
  colorBlending.blendConstants[3] = 0.0f;

  /*VkDynamicState dynamicStates[] = {
    VK_DYNAMIC_STATE_VIEWPORT,
    VK_DYNAMIC_STATE_LINE_WIDTH
  };
  VkPipelineDynamicStateCreateInfo dynamicState{};
  dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
  dynamicState.dynamicStateCount = 2;
  dynamicState.pDynamicStates = dynamicStates;*/

  VkPushConstantRange pushConstantRange{};
  pushConstantRange.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
  pushConstantRange.offset = 0;
  pushConstantRange.size = sizeof(float) * 3;
  VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
  pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipelineLayoutInfo.setLayoutCount = 1;
  pipelineLayoutInfo.pSetLayouts = &mUniformDescriptor;
  pipelineLayoutInfo.pushConstantRangeCount = 1;
  pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;
  CHKERR(vkCreatePipelineLayout(mDevice, &pipelineLayoutInfo, nullptr, &mPipelineLayout));

  VkGraphicsPipelineCreateInfo pipelineInfo{};
  pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
  pipelineInfo.stageCount = 2;
  pipelineInfo.pStages = shaderStages;
  pipelineInfo.pVertexInputState = &vertexInputInfo;
  pipelineInfo.pInputAssemblyState = &inputAssembly;
  pipelineInfo.pViewportState = &viewportState;
  pipelineInfo.pRasterizationState = &rasterizer;
  pipelineInfo.pMultisampleState = &multisampling;
  pipelineInfo.pDepthStencilState = nullptr; // Optional
  pipelineInfo.pColorBlendState = &colorBlending;
  pipelineInfo.pDynamicState = nullptr; // Optional
  pipelineInfo.layout = mPipelineLayout;
  pipelineInfo.renderPass = mRenderPass;
  pipelineInfo.subpass = 0;
  pipelineInfo.basePipelineHandle = VK_NULL_HANDLE; // Optional
  pipelineInfo.basePipelineIndex = -1;              // Optional

  mPipelines.resize(3);
  static constexpr VkPrimitiveTopology types[3] = {VK_PRIMITIVE_TOPOLOGY_POINT_LIST, VK_PRIMITIVE_TOPOLOGY_LINE_LIST, VK_PRIMITIVE_TOPOLOGY_LINE_STRIP};
  for (int i = 0; i < 3; i++) {
    inputAssembly.topology = types[i];
    CHKERR(vkCreateGraphicsPipelines(mDevice, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &mPipelines[i]));
  }
}

void GPUDisplayBackendVulkan::clearSwapChain()
{
  for (unsigned int i = 0; i < mImages.size(); i++) {
    vkDestroyFramebuffer(mDevice, mFramebuffers[i], nullptr);
    vkDestroyImageView(mDevice, mImageViews[i], nullptr);
  }
  vkDestroySwapchainKHR(mDevice, mSwapChain, nullptr);
}

void GPUDisplayBackendVulkan::clearPipeline()
{
  for (auto& pipeline : mPipelines) {
    vkDestroyPipeline(mDevice, pipeline, nullptr);
  }
  vkDestroyRenderPass(mDevice, mRenderPass, nullptr);
  vkDestroyPipelineLayout(mDevice, mPipelineLayout, nullptr);
}

void GPUDisplayBackendVulkan::createShaders()
{
  mModuleVertex = createShaderModule(_binary_shaders_display_shaders_vertex_vert_spv_start, _binary_shaders_display_shaders_vertex_vert_spv_len, mDevice);
  mModuleFragment = createShaderModule(_binary_shaders_display_shaders_fragment_frag_spv_start, _binary_shaders_display_shaders_fragment_frag_spv_len, mDevice);
}

void GPUDisplayBackendVulkan::clearShaders()
{
  vkDestroyShaderModule(mDevice, mModuleVertex, nullptr);
  vkDestroyShaderModule(mDevice, mModuleFragment, nullptr);
}

void GPUDisplayBackendVulkan::recreateSwapChain()
{
  vkDeviceWaitIdle(mDevice);
  clearPipeline();
  clearSwapChain();
  createSwapChain();
  createPipeline();
}

void GPUDisplayBackendVulkan::createUniformLayouts()
{
  VkDescriptorPoolSize poolSize{};
  poolSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  poolSize.descriptorCount = (uint32_t)mFramesInFlight * 2;
  VkDescriptorPoolCreateInfo poolInfo{};
  poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  poolInfo.poolSizeCount = 1;
  poolInfo.pPoolSizes = &poolSize;
  poolInfo.maxSets = (uint32_t)mFramesInFlight;
  CHKERR(vkCreateDescriptorPool(mDevice, &poolInfo, nullptr, &mDescriptorPool));

  VkDescriptorSetLayoutBinding uboLayoutBindingMat{};
  uboLayoutBindingMat.binding = 0;
  uboLayoutBindingMat.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  uboLayoutBindingMat.descriptorCount = 1;
  uboLayoutBindingMat.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
  uboLayoutBindingMat.pImmutableSamplers = nullptr; // Optional
  VkDescriptorSetLayoutBinding uboLayoutBindingCol = uboLayoutBindingMat;
  uboLayoutBindingCol.binding = 1;
  uboLayoutBindingCol.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
  VkDescriptorSetLayoutBinding bindings[2] = {uboLayoutBindingMat, uboLayoutBindingCol};

  VkDescriptorSetLayoutCreateInfo layoutInfo{};
  layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  layoutInfo.bindingCount = 2;
  layoutInfo.pBindings = bindings;
  CHKERR(vkCreateDescriptorSetLayout(mDevice, &layoutInfo, nullptr, &mUniformDescriptor));

  std::vector<VkDescriptorSetLayout> layouts(mFramesInFlight, mUniformDescriptor);
  VkDescriptorSetAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  allocInfo.descriptorPool = mDescriptorPool;
  allocInfo.descriptorSetCount = (uint32_t)mFramesInFlight;
  allocInfo.pSetLayouts = layouts.data();
  mDescriptorSets.resize(mFramesInFlight);
  CHKERR(vkAllocateDescriptorSets(mDevice, &allocInfo, mDescriptorSets.data()));

  for (unsigned int j = 0; j < 2; j++) {
    auto& mUniformBuffers = j ? mUniformBuffersCol : mUniformBuffersMat;
    for (unsigned int i = 0; i < mFramesInFlight; i++) {
      VkDescriptorBufferInfo bufferInfo{};
      bufferInfo.buffer = mUniformBuffers[i].buffer;
      bufferInfo.offset = 0;
      bufferInfo.range = mUniformBuffers[i].size;

      VkWriteDescriptorSet descriptorWrite{};
      descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      descriptorWrite.dstSet = mDescriptorSets[i];
      descriptorWrite.dstBinding = j;
      descriptorWrite.dstArrayElement = 0;
      descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
      descriptorWrite.descriptorCount = 1;
      descriptorWrite.pBufferInfo = &bufferInfo;
      descriptorWrite.pImageInfo = nullptr;       // Optional
      descriptorWrite.pTexelBufferView = nullptr; // Optional
      vkUpdateDescriptorSets(mDevice, 1, &descriptorWrite, 0, nullptr);
    }
  }
}

void GPUDisplayBackendVulkan::clearUniformLayouts()
{
  vkDestroyDescriptorSetLayout(mDevice, mUniformDescriptor, nullptr);
  vkDestroyDescriptorPool(mDevice, mDescriptorPool, nullptr);
}

int GPUDisplayBackendVulkan::InitBackend()
{
  std::cout << "Initializing Vulkan\n";

  mEnableValidationLayers = mDisplay->param() && mDisplay->param()->par.debugLevel >= 2;
  mFramesInFlight = 2;

  createDevice();
  createShaders();
  createUniformLayouts();
  createSwapChain();
  createPipeline();

  std::cout << "Vulkan initialized\n";
  return (0);
}

void GPUDisplayBackendVulkan::ExitBackend()
{
  std::cout << "Exiting Vulkan\n";
  vkDeviceWaitIdle(mDevice);
  clearVertexBuffers();
  clearPipeline();
  clearSwapChain();
  clearUniformLayouts();
  clearShaders();
  clearDevice();
  std::cout << "Vulkan destroyed\n";
}

void GPUDisplayBackendVulkan::resizeScene(unsigned int width, unsigned int height)
{
  if (mExtent.width == width && mExtent.height == height) {
    return;
  }
  recreateSwapChain();
  if (mExtent.width != width || mExtent.height != height) {
    // std::cout << "Unmatching window size: requested " << width << " x " << height << " - found " << mExtent.width << " x " << mExtent.height << "\n";
  }
  needRecordCommandBuffers();
}

void GPUDisplayBackendVulkan::clearScreen(bool colorOnly)
{
}

void GPUDisplayBackendVulkan::updateSettings()
{
}

static uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties, VkPhysicalDevice physDev)
{
  VkPhysicalDeviceMemoryProperties memProperties;
  vkGetPhysicalDeviceMemoryProperties(physDev, &memProperties);

  for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
    if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
      return i;
    }
  }

  throw std::runtime_error("failed to find suitable memory type!");
}

void GPUDisplayBackendVulkan::writeToBuffer(VulkanBuffer& buffer, size_t size, const void* srcData)
{
  if (!buffer.deviceMemory) {
    void* dstData;
    vkMapMemory(mDevice, buffer.memory, 0, buffer.size, 0, &dstData);
    memcpy(dstData, srcData, size);
    vkUnmapMemory(mDevice, buffer.memory);
  } else {
    auto tmp = createBuffer(size, srcData, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, false);

    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = mCommandPool;
    allocInfo.commandBufferCount = 1;
    VkCommandBuffer commandBuffer;
    vkAllocateCommandBuffers(mDevice, &allocInfo, &commandBuffer);
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(commandBuffer, &beginInfo);
    VkBufferCopy copyRegion{};
    copyRegion.size = size;
    vkCmdCopyBuffer(commandBuffer, tmp.buffer, buffer.buffer, 1, &copyRegion);
    vkEndCommandBuffer(commandBuffer);
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    vkQueueSubmit(mGraphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(mGraphicsQueue);
    vkFreeCommandBuffers(mDevice, mCommandPool, 1, &commandBuffer);

    clearBuffer(tmp);
  }
}

VulkanBuffer GPUDisplayBackendVulkan::createBuffer(size_t size, const void* srcData, VkBufferUsageFlags type, int deviceMemory)
{
  VkMemoryPropertyFlags properties = deviceMemory ? VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT : (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  if (deviceMemory == 1) {
    type |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  }

  VulkanBuffer buffer;
  VkBufferCreateInfo bufferInfo{};
  bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufferInfo.size = size;
  bufferInfo.usage = type;
  bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  CHKERR(vkCreateBuffer(mDevice, &bufferInfo, nullptr, &buffer.buffer));

  VkMemoryRequirements memRequirements;
  vkGetBufferMemoryRequirements(mDevice, buffer.buffer, &memRequirements);
  VkMemoryAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocInfo.allocationSize = memRequirements.size;
  allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties, mPhysicalDevice);
  CHKERR(vkAllocateMemory(mDevice, &allocInfo, nullptr, &buffer.memory));

  vkBindBufferMemory(mDevice, buffer.buffer, buffer.memory, 0);

  buffer.size = size;
  buffer.deviceMemory = deviceMemory;

  if (srcData != nullptr) {
    writeToBuffer(buffer, size, srcData);
  }

  return buffer;
}

void GPUDisplayBackendVulkan::clearBuffer(VulkanBuffer& buffer)
{
  vkDestroyBuffer(mDevice, buffer.buffer, nullptr);
  vkFreeMemory(mDevice, buffer.memory, nullptr);
}

void GPUDisplayBackendVulkan::clearVertexBuffers()
{
  for (unsigned int i = 0; i < mNVBOCreated; i++) {
    clearBuffer(mVBO[i]);
  }
  mNVBOCreated = 0;
  if (mIndirectCommandBufferCreated) {
    clearBuffer(mIndirectCommandBuffer[0]);
  }
  mIndirectCommandBufferCreated = false;
}

void GPUDisplayBackendVulkan::loadDataToGPU(size_t totalVertizes)
{
  vkDeviceWaitIdle(mDevice);
  clearVertexBuffers();
  mVBO[0] = createBuffer(totalVertizes * sizeof(mDisplay->vertexBuffer()[0][0]), mDisplay->vertexBuffer()[0].data());
  mNVBOCreated = 1;
  if (mDisplay->cfgR().useGLIndirectDraw) {
    fillIndirectCmdBuffer();
    mIndirectCommandBuffer[0] = createBuffer(mCmdBuffer.size() * sizeof(mCmdBuffer[0]), mCmdBuffer.data(), VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT);
    mIndirectCommandBufferCreated = true;
    mCmdBuffer.clear();
  }
  needRecordCommandBuffers();
}

void GPUDisplayBackendVulkan::prepareDraw()
{
  if (mDisplay->updateDrawCommands()) {
    needRecordCommandBuffers();
  }
  vkWaitForFences(mDevice, 1, &mInFlightFence[mCurrentFrame], VK_TRUE, UINT64_MAX);

  VkResult retVal = vkAcquireNextImageKHR(mDevice, mSwapChain, UINT64_MAX, mImageAvailableSemaphore[mCurrentFrame], VK_NULL_HANDLE, &mImageIndex);
  if (retVal == VK_ERROR_OUT_OF_DATE_KHR || retVal == VK_SUBOPTIMAL_KHR) {
    std::cout << "Pipeline out of data / suboptimal, recreating\n";
    recreateSwapChain();
    needRecordCommandBuffers();
    retVal = vkAcquireNextImageKHR(mDevice, mSwapChain, UINT64_MAX, mImageAvailableSemaphore[mCurrentFrame], VK_NULL_HANDLE, &mImageIndex);
  }
  CHKERR(retVal);
  vkResetFences(mDevice, 1, &mInFlightFence[mCurrentFrame]);

  if (!mCommandBufferUpToDate[mImageIndex]) {
    startFillCommandBuffer(mCommandBuffers[mImageIndex], mImageIndex);
  }
}

void GPUDisplayBackendVulkan::finishDraw()
{
  if (!mCommandBufferUpToDate[mImageIndex]) {
    endFillCommandBuffer(mCommandBuffers[mImageIndex], mImageIndex);
    mCommandBufferUpToDate[mImageIndex] = true;
  }

  VkSubmitInfo submitInfo{};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
  submitInfo.waitSemaphoreCount = 1;
  submitInfo.pWaitSemaphores = &mImageAvailableSemaphore[mCurrentFrame];
  submitInfo.pWaitDstStageMask = waitStages;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &mCommandBuffers[mImageIndex];
  submitInfo.signalSemaphoreCount = 1;
  submitInfo.pSignalSemaphores = &mRenderFinishedSemaphore[mCurrentFrame];
  CHKERR(vkQueueSubmit(mGraphicsQueue, 1, &submitInfo, mInFlightFence[mCurrentFrame]));

  VkPresentInfoKHR presentInfo{};
  presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
  presentInfo.waitSemaphoreCount = 1;
  presentInfo.pWaitSemaphores = &mRenderFinishedSemaphore[mCurrentFrame];
  presentInfo.swapchainCount = 1;
  presentInfo.pSwapchains = &mSwapChain;
  presentInfo.pImageIndices = &mImageIndex;
  presentInfo.pResults = nullptr;
  vkQueuePresentKHR(mGraphicsQueue, &presentInfo);
  mCurrentFrame = (mCurrentFrame + 1) % mFramesInFlight;
}

void GPUDisplayBackendVulkan::prepareText()
{
}

void GPUDisplayBackendVulkan::renderOffscreenBuffer(GLfb& buffer, GLfb& bufferNoMSAA, int mainBuffer)
{
}

void GPUDisplayBackendVulkan::setMatrices(const hmm_mat4& proj, const hmm_mat4& view)
{
  const hmm_mat4 modelViewProj = proj * view;
  writeToBuffer(mUniformBuffersMat[mImageIndex], sizeof(modelViewProj), &modelViewProj);
}

void GPUDisplayBackendVulkan::mixImages(GLfb& mixBuffer, float mixSlaveImage)
{
  {
    GPUWarning("Image mixing unsupported in Vulkan profile");
  }
}

void GPUDisplayBackendVulkan::readPixels(unsigned char* pixels, bool needBuffer, unsigned int width, unsigned int height)
{
}

void GPUDisplayBackendVulkan::pointSizeFactor(float factor)
{
}

void GPUDisplayBackendVulkan::lineWidthFactor(float factor)
{
}

void GPUDisplayBackendVulkan::needRecordCommandBuffers()
{
  std::fill(mCommandBufferUpToDate.begin(), mCommandBufferUpToDate.end(), false);
}
