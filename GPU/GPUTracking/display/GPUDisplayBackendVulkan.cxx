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

#include "GPUCommonDef.h"
#include "GPUDisplayBackendVulkan.h"
#include "GPUDisplay.h"

using namespace GPUCA_NAMESPACE::gpu;

#include "utils/qGetLdBinarySymbols.h"
QGET_LD_BINARY_SYMBOLS(shaders_display_shaders_vertex_vert_spv);
QGET_LD_BINARY_SYMBOLS(shaders_display_shaders_fragment_frag_spv);
QGET_LD_BINARY_SYMBOLS(shaders_display_shaders_vertexPoint_vert_spv);
QGET_LD_BINARY_SYMBOLS(shaders_display_shaders_fragmentText_frag_spv);
QGET_LD_BINARY_SYMBOLS(shaders_display_shaders_vertexText_vert_spv);

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
  size_t size = 0;
  bool deviceMemory;
};
struct VulkanImage {
  VkImage image;
  VkImageView view;
  VkDeviceMemory memory;
  unsigned int sizex, sizey;
  VkFormat format;
};
struct FontSymbolVulkan : public GPUDisplayBackend::FontSymbol {
  std::unique_ptr<char[]> data;
  float x0, x1, y0, y1;
};
struct TextDrawCommand {
  size_t firstVertex;
  size_t nVertices;
  float color[4];
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

// ---------------------------- VULKAN HELPERS ----------------------------

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

static VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats)
{
  for (const auto& availableFormat : availableFormats) {
    // Could use VK_FORMAT_B8G8R8A8_SRGB for sRGB, but we don't have photos anyway...
    if (availableFormat.format == VK_FORMAT_B8G8R8A8_UNORM && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
      return availableFormat;
    }
  }
  return availableFormats[0];
}

static VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes, VkPresentModeKHR desiredMode = VK_PRESENT_MODE_MAILBOX_KHR)
{
  for (const auto& availablePresentMode : availablePresentModes) {
    if (availablePresentMode == desiredMode) {
      return availablePresentMode;
    }
  }
  return VK_PRESENT_MODE_FIFO_KHR;
}

VkExtent2D GPUDisplayBackendVulkan::chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities)
{
  if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
    return capabilities.currentExtent;
  } else {
    int width, height;
    mDisplay->frontend()->getSize(width, height);
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

static void cmdImageMemoryBarrier(VkCommandBuffer cmdbuffer, VkImage image, VkAccessFlags srcAccessMask, VkAccessFlags dstAccessMask, VkImageLayout oldLayout, VkImageLayout newLayout, VkPipelineStageFlags srcStageMask, VkPipelineStageFlags dstStageMask, VkImageSubresourceRange subresourceRange)
{
  VkImageMemoryBarrier barrier{};
  barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  barrier.srcAccessMask = srcAccessMask;
  barrier.dstAccessMask = dstAccessMask;
  barrier.oldLayout = oldLayout;
  barrier.newLayout = newLayout;
  barrier.image = image;
  barrier.subresourceRange = subresourceRange;

  vkCmdPipelineBarrier(cmdbuffer, srcStageMask, dstStageMask, 0, 0, nullptr, 0, nullptr, 1, &barrier);
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

VkCommandBuffer GPUDisplayBackendVulkan::getSingleTimeCommandBuffer()
{
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
  return commandBuffer;
}

void GPUDisplayBackendVulkan::submitSingleTimeCommandBuffer(VkCommandBuffer commandBuffer)
{
  vkEndCommandBuffer(commandBuffer);
  VkSubmitInfo submitInfo{};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &commandBuffer;
  vkQueueSubmit(mGraphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
  vkQueueWaitIdle(mGraphicsQueue);
  vkFreeCommandBuffers(mDevice, mCommandPool, 1, &commandBuffer);
}

void GPUDisplayBackendVulkan::transitionImageLayout(VkCommandBuffer commandBuffer, VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout)
{
  VkImageMemoryBarrier barrier{};
  barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  barrier.oldLayout = oldLayout;
  barrier.newLayout = newLayout;
  barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.image = image;
  barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  barrier.subresourceRange.baseMipLevel = 0;
  barrier.subresourceRange.levelCount = 1;
  barrier.subresourceRange.baseArrayLayer = 0;
  barrier.subresourceRange.layerCount = 1;
  VkPipelineStageFlags sourceStage;
  VkPipelineStageFlags destinationStage;
  if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
  } else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
  } else {
    throw std::invalid_argument("unsupported layout transition!");
  }

  vkCmdPipelineBarrier(commandBuffer, sourceStage, destinationStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
}

static VkImageView createImageViewI(VkDevice device, VkImage image, VkFormat format, VkImageAspectFlags aspectFlags = VK_IMAGE_ASPECT_COLOR_BIT, uint32_t mipLevels = 1)
{
  VkImageViewCreateInfo viewInfo{};
  viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  viewInfo.image = image;
  viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
  viewInfo.format = format;
  viewInfo.subresourceRange.aspectMask = aspectFlags;
  viewInfo.subresourceRange.baseMipLevel = 0;
  viewInfo.subresourceRange.levelCount = mipLevels;
  viewInfo.subresourceRange.baseArrayLayer = 0;
  viewInfo.subresourceRange.layerCount = 1;
  VkImageView imageView;
  CHKERR(vkCreateImageView(device, &viewInfo, nullptr, &imageView));

  return imageView;
}

static void createImageI(VkDevice device, VkPhysicalDevice physicalDevice, VkImage& image, VkDeviceMemory& imageMemory, uint32_t width, uint32_t height, VkFormat format, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImageTiling tiling = VK_IMAGE_TILING_OPTIMAL, VkSampleCountFlagBits numSamples = VK_SAMPLE_COUNT_1_BIT, uint32_t mipLevels = 1)
{
  VkImageCreateInfo imageInfo{};
  imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  imageInfo.imageType = VK_IMAGE_TYPE_2D;
  imageInfo.extent.width = width;
  imageInfo.extent.height = height;
  imageInfo.extent.depth = 1;
  imageInfo.mipLevels = mipLevels;
  imageInfo.arrayLayers = 1;
  imageInfo.format = format;
  imageInfo.tiling = tiling;
  imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  imageInfo.usage = usage;
  imageInfo.samples = numSamples;
  imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  CHKERR(vkCreateImage(device, &imageInfo, nullptr, &image));

  VkMemoryRequirements memRequirements;
  vkGetImageMemoryRequirements(device, image, &memRequirements);

  VkMemoryAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocInfo.allocationSize = memRequirements.size;
  allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties, physicalDevice);
  CHKERR(vkAllocateMemory(device, &allocInfo, nullptr, &imageMemory));

  vkBindImageMemory(device, image, imageMemory, 0);
}

VkSampleCountFlagBits getMaxUsableSampleCount(VkPhysicalDeviceProperties& physicalDeviceProperties)
{
  VkSampleCountFlags counts = physicalDeviceProperties.limits.framebufferColorSampleCounts & physicalDeviceProperties.limits.framebufferDepthSampleCounts;
  if (counts & VK_SAMPLE_COUNT_64_BIT) {
    return VK_SAMPLE_COUNT_64_BIT;
  } else if (counts & VK_SAMPLE_COUNT_32_BIT) {
    return VK_SAMPLE_COUNT_32_BIT;
  } else if (counts & VK_SAMPLE_COUNT_16_BIT) {
    return VK_SAMPLE_COUNT_16_BIT;
  } else if (counts & VK_SAMPLE_COUNT_8_BIT) {
    return VK_SAMPLE_COUNT_8_BIT;
  } else if (counts & VK_SAMPLE_COUNT_4_BIT) {
    return VK_SAMPLE_COUNT_4_BIT;
  } else if (counts & VK_SAMPLE_COUNT_2_BIT) {
    return VK_SAMPLE_COUNT_2_BIT;
  }
  return VK_SAMPLE_COUNT_1_BIT;
}

static VkSampleCountFlagBits getMSAASamplesFlag(unsigned int msaa)
{
  if (msaa == 2) {
    return VK_SAMPLE_COUNT_2_BIT;
  } else if (msaa == 4) {
    return VK_SAMPLE_COUNT_4_BIT;
  } else if (msaa == 8) {
    return VK_SAMPLE_COUNT_8_BIT;
  } else if (msaa == 16) {
    return VK_SAMPLE_COUNT_16_BIT;
  } else if (msaa == 32) {
    return VK_SAMPLE_COUNT_32_BIT;
  } else if (msaa == 64) {
    return VK_SAMPLE_COUNT_64_BIT;
  }
  return VK_SAMPLE_COUNT_1_BIT;
}

// ---------------------------- VULKAN DEVICE MANAGEMENT ----------------------------

double GPUDisplayBackendVulkan::checkDevice(VkPhysicalDevice device, const std::vector<const char*>& reqDeviceExtensions)
{
  double score = -1.;
  VkPhysicalDeviceProperties deviceProperties;
  vkGetPhysicalDeviceProperties(device, &deviceProperties);
  VkPhysicalDeviceFeatures deviceFeatures;
  vkGetPhysicalDeviceFeatures(device, &deviceFeatures);
  VkPhysicalDeviceMemoryProperties memoryProperties;
  vkGetPhysicalDeviceMemoryProperties(device, &memoryProperties);
  if (!deviceFeatures.geometryShader || !deviceFeatures.wideLines || !deviceFeatures.largePoints) {
    return (-1);
  }

  uint32_t queueFamilyCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);
  std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
  vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());
  bool found = false;
  for (unsigned int i = 0; i < queueFamilies.size(); i++) {
    if (!(queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
      return (-1);
    }
    VkBool32 presentSupport = false;
    vkGetPhysicalDeviceSurfaceSupportKHR(device, i, mSurface, &presentSupport);
    if (!presentSupport) {
      return (-1);
    }
    mQueueFamilyIndices->graphicsFamily = i;
    found = true;
    break;
  }
  if (!found) {
    GPUInfo("%s ignored due to missing queue properties", deviceProperties.deviceName);
    return (-1);
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
    return (-1);
  }

  updateSwapChainDetails(device);
  if (mSwapChainDetails->formats.empty() || mSwapChainDetails->presentModes.empty()) {
    GPUInfo("%s ignored due to incompatible swap chain", deviceProperties.deviceName);
    return (-1);
  }

  score = 1;
  if (deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
    score += 1e12;
  } else if (deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) {
    score += 1e11;
  }

  for (unsigned int i = 0; i < memoryProperties.memoryHeapCount; i++) {
    if (memoryProperties.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
      score += memoryProperties.memoryHeaps[i].size;
    }
  }

  return score;
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

  const char** frontendExtensions;
  uint32_t frontendExtensionCount = mDisplay->frontend()->getReqVulkanExtensions(frontendExtensions);
  std::vector<const char*> reqInstanceExtensions(frontendExtensions, frontendExtensions + frontendExtensionCount);

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

  if (mDisplay->frontend()->getVulkanSurface(&mInstance, &mSurface)) {
    throw std::runtime_error("Frontend does not provide Vulkan surface");
  }

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
  double bestScore = -1.;
  for (unsigned int i = 0; i < devices.size(); i++) {
    double score = checkDevice(devices[i], reqDeviceExtensions);
    if (mDisplay->param()->par.debugLevel >= 2) {
      VkPhysicalDeviceProperties deviceProperties;
      vkGetPhysicalDeviceProperties(devices[i], &deviceProperties);
      GPUInfo("Available Vulkan device %d: %s - Score %f", i, deviceProperties.deviceName, score);
    }
    if (score > bestScore && score > 0) {
      mPhysicalDevice = devices[i];
      bestScore = score;
    }
  }

  if (mPhysicalDevice == VK_NULL_HANDLE) {
    throw std::runtime_error("All available Vulkan devices unsuited");
  }

  VkPhysicalDeviceProperties deviceProperties;
  vkGetPhysicalDeviceProperties(mPhysicalDevice, &deviceProperties);
  VkPhysicalDeviceFeatures deviceFeatures;
  vkGetPhysicalDeviceFeatures(mPhysicalDevice, &deviceFeatures);
  GPUInfo("Using physicak Vulkan device %s", deviceProperties.deviceName);
  mMaxMSAAsupported = getMaxUsableSampleCount(deviceProperties);

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
  mFontVertexBuffer.resize(mFramesInFlight);
  mCommandBuffers.resize(mFramesInFlight);
  mCommandBuffersText.resize(mFramesInFlight);
  mCommandBufferUpToDate.resize(mFramesInFlight, false);
  CHKERR(vkAllocateCommandBuffers(mDevice, &allocInfo, mCommandBuffers.data()));
  CHKERR(vkAllocateCommandBuffers(mDevice, &allocInfo, mCommandBuffersText.data()));

  VkSemaphoreCreateInfo semaphoreInfo{};
  semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
  VkFenceCreateInfo fenceInfo{};
  fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
  mImageAvailableSemaphore.resize(mFramesInFlight);
  mRenderFinishedSemaphore.resize(mFramesInFlight);
  mTextFinishedSemaphore.resize(mFramesInFlight);
  mInFlightFence.resize(mFramesInFlight);
  for (unsigned int i = 0; i < mFramesInFlight; i++) {
    CHKERR(vkCreateSemaphore(mDevice, &semaphoreInfo, nullptr, &mImageAvailableSemaphore[i]));
    CHKERR(vkCreateSemaphore(mDevice, &semaphoreInfo, nullptr, &mRenderFinishedSemaphore[i]));
    CHKERR(vkCreateSemaphore(mDevice, &semaphoreInfo, nullptr, &mTextFinishedSemaphore[i]));
    CHKERR(vkCreateFence(mDevice, &fenceInfo, nullptr, &mInFlightFence[i]));
  }
  for (int j = 0; j < 2; j++) {
    mUniformBuffersMat[j].resize(mFramesInFlight);
    mUniformBuffersCol[j].resize(mFramesInFlight);
    for (unsigned int i = 0; i < mFramesInFlight; i++) {
      mUniformBuffersMat[j][i] = createBuffer(sizeof(hmm_mat4), nullptr, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, false);
      mUniformBuffersCol[j][i] = createBuffer(sizeof(float) * 4, nullptr, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, false);
    }
  }
}

void GPUDisplayBackendVulkan::clearDevice()
{
  for (unsigned int i = 0; i < mImageAvailableSemaphore.size(); i++) {
    vkDestroySemaphore(mDevice, mImageAvailableSemaphore[i], nullptr);
    vkDestroySemaphore(mDevice, mRenderFinishedSemaphore[i], nullptr);
    vkDestroySemaphore(mDevice, mTextFinishedSemaphore[i], nullptr);
    vkDestroyFence(mDevice, mInFlightFence[i], nullptr);
    for (int j = 0; j < 2; j++) {
      clearBuffer(mUniformBuffersMat[j][i]);
      clearBuffer(mUniformBuffersCol[j][i]);
    }
  }
  vkDestroyCommandPool(mDevice, mCommandPool, nullptr);
  vkDestroyDevice(mDevice, nullptr);
  vkDestroySurfaceKHR(mInstance, mSurface, nullptr);
  if (mEnableValidationLayers) {
    DestroyDebugUtilsMessengerEXT(mInstance, mDebugMessenger, nullptr);
  }
  vkDestroyInstance(mInstance, nullptr);
}

// ---------------------------- VULKAN UNIFORM LAYOUTS ----------------------------

void GPUDisplayBackendVulkan::createUniformLayouts()
{
  std::array<VkDescriptorPoolSize, 2> poolSizes{};
  poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  poolSizes[0].descriptorCount = (uint32_t)mFramesInFlight * (2 * 2);
  poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  poolSizes[1].descriptorCount = (uint32_t)mFramesInFlight;
  VkDescriptorPoolCreateInfo poolInfo{};
  poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  poolInfo.poolSizeCount = poolSizes.size();
  poolInfo.pPoolSizes = poolSizes.data();
  poolInfo.maxSets = (uint32_t)mFramesInFlight * 2;
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
  VkDescriptorSetLayoutBinding samplerLayoutBinding{};
  samplerLayoutBinding.binding = 2;
  samplerLayoutBinding.descriptorCount = 1;
  samplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  samplerLayoutBinding.pImmutableSamplers = nullptr;
  samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
  VkDescriptorSetLayoutBinding bindings[3] = {uboLayoutBindingMat, uboLayoutBindingCol, samplerLayoutBinding};

  VkDescriptorSetLayoutCreateInfo layoutInfo{};
  layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  layoutInfo.bindingCount = 2;
  layoutInfo.pBindings = bindings;
  CHKERR(vkCreateDescriptorSetLayout(mDevice, &layoutInfo, nullptr, &mUniformDescriptor));
  layoutInfo.bindingCount = 3;
  CHKERR(vkCreateDescriptorSetLayout(mDevice, &layoutInfo, nullptr, &mUniformDescriptorText));

  VkDescriptorSetAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  allocInfo.descriptorPool = mDescriptorPool;
  allocInfo.descriptorSetCount = (uint32_t)mFramesInFlight;
  for (int j = 0; j < 2; j++) {
    mDescriptorSets[j].resize(mFramesInFlight);
    std::vector<VkDescriptorSetLayout> layouts(mFramesInFlight, j ? mUniformDescriptorText : mUniformDescriptor);
    allocInfo.pSetLayouts = layouts.data();
    CHKERR(vkAllocateDescriptorSets(mDevice, &allocInfo, mDescriptorSets[j].data()));

    for (int k = 0; k < 2; k++) {
      auto& mUniformBuffers = k ? mUniformBuffersCol[j] : mUniformBuffersMat[j];
      for (unsigned int i = 0; i < mFramesInFlight; i++) {
        VkDescriptorBufferInfo bufferInfo{};
        bufferInfo.buffer = mUniformBuffers[i].buffer;
        bufferInfo.offset = 0;
        bufferInfo.range = mUniformBuffers[i].size;

        VkWriteDescriptorSet descriptorWrite{};
        descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrite.dstSet = mDescriptorSets[j][i];
        descriptorWrite.dstBinding = k;
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
}

void GPUDisplayBackendVulkan::clearUniformLayouts()
{
  vkDestroyDescriptorSetLayout(mDevice, mUniformDescriptor, nullptr);
  vkDestroyDescriptorSetLayout(mDevice, mUniformDescriptorText, nullptr);
  vkDestroyDescriptorPool(mDevice, mDescriptorPool, nullptr);
}

// ---------------------------- VULKAN TEXTURE SAMPLER ----------------------------

void GPUDisplayBackendVulkan::createTextureSampler()
{
  VkSamplerCreateInfo samplerInfo{};
  samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
  samplerInfo.magFilter = VK_FILTER_LINEAR;
  samplerInfo.minFilter = VK_FILTER_LINEAR;
  samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  samplerInfo.compareEnable = VK_FALSE;
  samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
  samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
  samplerInfo.unnormalizedCoordinates = VK_FALSE;
  samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
  samplerInfo.mipLodBias = 0.0f;
  samplerInfo.minLod = 0.0f;
  samplerInfo.maxLod = 0.0f;
  CHKERR(vkCreateSampler(mDevice, &samplerInfo, nullptr, &mTextSampler));
}

void GPUDisplayBackendVulkan::clearTextureSampler()
{
  vkDestroySampler(mDevice, mTextSampler, nullptr);
}

// ---------------------------- VULKAN SWAPCHAIN MANAGEMENT ----------------------------

void GPUDisplayBackendVulkan::createSwapChain()
{
  mMSAASampleCount = getMSAASamplesFlag(std::min<unsigned int>(mMaxMSAAsupported, mDisplay->cfgR().drawQualityMSAA));
  mSwapchainImageReadable = mScreenshotRequested;

  updateSwapChainDetails(mPhysicalDevice);
  mSurfaceFormat = chooseSwapSurfaceFormat(mSwapChainDetails->formats);
  mPresentMode = chooseSwapPresentMode(mSwapChainDetails->presentModes, mDisplay->cfgR().drawQualityVSync ? VK_PRESENT_MODE_MAILBOX_KHR : VK_PRESENT_MODE_IMMEDIATE_KHR);
  mExtent = chooseSwapExtent(mSwapChainDetails->capabilities);

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
  if (mSwapchainImageReadable) {
    swapCreateInfo.imageUsage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
  }
  CHKERR(vkCreateSwapchainKHR(mDevice, &swapCreateInfo, nullptr, &mSwapChain));

  VkAttachmentDescription colorAttachment{};
  colorAttachment.format = mSurfaceFormat.format;
  colorAttachment.samples = mMSAASampleCount;
  colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
  VkAttachmentDescription colorAttachmentResolve{};
  colorAttachmentResolve.format = mSurfaceFormat.format;
  colorAttachmentResolve.samples = VK_SAMPLE_COUNT_1_BIT;
  colorAttachmentResolve.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  colorAttachmentResolve.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  colorAttachmentResolve.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  colorAttachmentResolve.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  colorAttachmentResolve.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  colorAttachmentResolve.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
  VkAttachmentReference colorAttachmentRef{};
  colorAttachmentRef.attachment = 0;
  colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  VkAttachmentReference colorAttachmentResolveRef{};
  colorAttachmentResolveRef.attachment = 1;
  colorAttachmentResolveRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
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

  std::vector<VkAttachmentDescription> attachments = {colorAttachment};
  if (mDisplay->cfgR().drawQualityMSAA) {
    attachments.emplace_back(colorAttachmentResolve);
    subpass.pResolveAttachments = &colorAttachmentResolveRef;
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  }

  VkRenderPassCreateInfo renderPassInfo{};
  renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
  renderPassInfo.attachmentCount = attachments.size();
  renderPassInfo.pAttachments = attachments.data();
  renderPassInfo.subpassCount = 1;
  renderPassInfo.pSubpasses = &subpass;
  renderPassInfo.dependencyCount = 1;
  renderPassInfo.pDependencies = &dependency;
  CHKERR(vkCreateRenderPass(mDevice, &renderPassInfo, nullptr, &mRenderPass));

  // Text overlay goes as extra rendering path
  renderPassInfo.attachmentCount = 1;
  renderPassInfo.pAttachments = &colorAttachment;
  colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
  colorAttachment.initialLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
  colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
  subpass.pResolveAttachments = nullptr;
  colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
  CHKERR(vkCreateRenderPass(mDevice, &renderPassInfo, nullptr, &mRenderPassText));

  vkGetSwapchainImagesKHR(mDevice, mSwapChain, &mImageCount, nullptr);
  mSwapChainImages.resize(mImageCount);
  vkGetSwapchainImagesKHR(mDevice, mSwapChain, &mImageCount, mSwapChainImages.data());
  mSwapChainImageViews.resize(mSwapChainImages.size());
  mFramebuffers.resize(mSwapChainImages.size());
  if (mDisplay->cfgR().drawQualityMSAA) {
    mFramebuffersText.resize(mSwapChainImages.size());
    mMSAAImages.resize(mSwapChainImages.size());
  }
  for (size_t i = 0; i < mSwapChainImages.size(); i++) {
    mSwapChainImageViews[i] = createImageViewI(mDevice, mSwapChainImages[i], mSurfaceFormat.format);
    std::vector<VkImageView> att;
    if (mDisplay->cfgR().drawQualityMSAA) {
      createImageI(mDevice, mPhysicalDevice, mMSAAImages[i].image, mMSAAImages[i].memory, mExtent.width, mExtent.height, mSurfaceFormat.format, VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_IMAGE_TILING_OPTIMAL, mMSAASampleCount);
      mMSAAImages[i].view = createImageViewI(mDevice, mMSAAImages[i].image, mSurfaceFormat.format, VK_IMAGE_ASPECT_COLOR_BIT, 1);
      att.emplace_back(mMSAAImages[i].view);
    }
    att.emplace_back(mSwapChainImageViews[i]);

    VkFramebufferCreateInfo framebufferInfo{};
    framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    framebufferInfo.renderPass = mRenderPass;
    framebufferInfo.attachmentCount = att.size();
    framebufferInfo.pAttachments = att.data();
    framebufferInfo.width = mExtent.width;
    framebufferInfo.height = mExtent.height;
    framebufferInfo.layers = 1;
    CHKERR(vkCreateFramebuffer(mDevice, &framebufferInfo, nullptr, &mFramebuffers[i]));

    if (mDisplay->cfgR().drawQualityMSAA) {
      framebufferInfo.attachmentCount = 1;
      framebufferInfo.pAttachments = &mSwapChainImageViews[i];
      framebufferInfo.renderPass = mRenderPassText;
      CHKERR(vkCreateFramebuffer(mDevice, &framebufferInfo, nullptr, &mFramebuffersText[i]));
    }
  }
}

void GPUDisplayBackendVulkan::clearSwapChain()
{
  for (unsigned int i = 0; i < mSwapChainImages.size(); i++) {
    vkDestroyFramebuffer(mDevice, mFramebuffers[i], nullptr);
    vkDestroyImageView(mDevice, mSwapChainImageViews[i], nullptr);
  }
  for (auto& img : mMSAAImages) {
    clearImage(img);
  }
  for (auto& fb : mFramebuffersText) {
    vkDestroyFramebuffer(mDevice, fb, nullptr);
  }
  mMSAAImages.resize(0);
  mFramebuffersText.resize(0);
  vkDestroySwapchainKHR(mDevice, mSwapChain, nullptr);
}

void GPUDisplayBackendVulkan::recreateSwapChain()
{
  vkDeviceWaitIdle(mDevice);
  clearPipeline();
  clearSwapChain();
  createSwapChain();
  createPipeline();
  needRecordCommandBuffers();
}

// ---------------------------- VULKAN PIPELINE ----------------------------

void GPUDisplayBackendVulkan::createPipeline()
{
  VkPipelineShaderStageCreateInfo shaderStages[2] = {VkPipelineShaderStageCreateInfo{}, VkPipelineShaderStageCreateInfo{}};
  VkPipelineShaderStageCreateInfo& vertShaderStageInfo = shaderStages[0];
  vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
  // vertShaderStageInfo.module // below
  vertShaderStageInfo.pName = "main";
  VkPipelineShaderStageCreateInfo& fragShaderStageInfo = shaderStages[1];
  fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
  // fragShaderStageInfo.module // below
  fragShaderStageInfo.pName = "main";

  VkVertexInputBindingDescription bindingDescription{};
  bindingDescription.binding = 0;
  // bindingDescription.stride // below
  bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

  VkVertexInputAttributeDescription attributeDescriptions{};
  attributeDescriptions.binding = 0;
  attributeDescriptions.location = 0;
  // attributeDescriptions.format // below
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
  rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
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
  // multisampling.rasterizationSamples // below
  multisampling.minSampleShading = 1.0f;          // Optional
  multisampling.pSampleMask = nullptr;            // Optional
  multisampling.alphaToCoverageEnable = VK_FALSE; // Optional
  multisampling.alphaToOneEnable = VK_FALSE;      // Optional

  VkPipelineColorBlendAttachmentState colorBlendAttachment{};
  colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
  // colorBlendAttachment.blendEnable // below
  colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
  colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
  colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
  colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
  colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
  colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
  colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;

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

  VkDynamicState dynamicStates[] = {
    VK_DYNAMIC_STATE_LINE_WIDTH};
  VkPipelineDynamicStateCreateInfo dynamicState{};
  dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
  dynamicState.dynamicStateCount = 1;
  dynamicState.pDynamicStates = dynamicStates;

  VkPushConstantRange pushConstantRanges[2] = {VkPushConstantRange{}, VkPushConstantRange{}};
  pushConstantRanges[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
  pushConstantRanges[0].offset = 0;
  pushConstantRanges[0].size = sizeof(float) * 4;
  pushConstantRanges[1].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
  pushConstantRanges[1].offset = pushConstantRanges[0].size;
  pushConstantRanges[1].size = sizeof(float);
  VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
  pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipelineLayoutInfo.setLayoutCount = 1;
  pipelineLayoutInfo.pSetLayouts = &mUniformDescriptor;
  pipelineLayoutInfo.pushConstantRangeCount = 2;
  pipelineLayoutInfo.pPushConstantRanges = pushConstantRanges;
  CHKERR(vkCreatePipelineLayout(mDevice, &pipelineLayoutInfo, nullptr, &mPipelineLayout));
  pipelineLayoutInfo.setLayoutCount = 1;
  pipelineLayoutInfo.pSetLayouts = &mUniformDescriptorText;
  CHKERR(vkCreatePipelineLayout(mDevice, &pipelineLayoutInfo, nullptr, &mPipelineLayoutText));

  VkGraphicsPipelineCreateInfo pipelineInfo{};
  pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
  pipelineInfo.stageCount = 2;
  pipelineInfo.pVertexInputState = &vertexInputInfo;
  pipelineInfo.pInputAssemblyState = &inputAssembly;
  pipelineInfo.pViewportState = &viewportState;
  pipelineInfo.pRasterizationState = &rasterizer;
  pipelineInfo.pMultisampleState = &multisampling;
  pipelineInfo.pDepthStencilState = nullptr; // Optional
  pipelineInfo.pColorBlendState = &colorBlending;
  pipelineInfo.pDynamicState = &dynamicState;
  // pipelineInfo.layout // below
  // pipelineInfo.renderPass // below
  pipelineInfo.subpass = 0;
  pipelineInfo.pStages = shaderStages;
  pipelineInfo.basePipelineHandle = VK_NULL_HANDLE; // Optional
  pipelineInfo.basePipelineIndex = -1;              // Optional

  mPipelines.resize(4);
  static constexpr VkPrimitiveTopology types[3] = {VK_PRIMITIVE_TOPOLOGY_POINT_LIST, VK_PRIMITIVE_TOPOLOGY_LINE_LIST, VK_PRIMITIVE_TOPOLOGY_LINE_STRIP};
  for (int i = 0; i < 4; i++) {
    if (i == 3) {
      bindingDescription.stride = 4 * sizeof(float);
      attributeDescriptions.format = VK_FORMAT_R32G32B32A32_SFLOAT;
      inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
      vertShaderStageInfo.module = mShaders["vertexText"];
      fragShaderStageInfo.module = mShaders["fragmentText"];
      pipelineInfo.layout = mPipelineLayoutText;
      pipelineInfo.renderPass = mRenderPassText;
      colorBlendAttachment.blendEnable = VK_TRUE;
      multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    } else {
      bindingDescription.stride = 3 * sizeof(float);
      attributeDescriptions.format = VK_FORMAT_R32G32B32_SFLOAT;
      inputAssembly.topology = types[i];
      vertShaderStageInfo.module = mShaders[types[i] == VK_PRIMITIVE_TOPOLOGY_POINT_LIST ? "vertexPoint" : "vertex"];
      fragShaderStageInfo.module = mShaders["fragment"];
      pipelineInfo.layout = mPipelineLayout;
      pipelineInfo.renderPass = mRenderPass;
      colorBlendAttachment.blendEnable = VK_TRUE;
      multisampling.rasterizationSamples = mMSAASampleCount;
    }

    CHKERR(vkCreateGraphicsPipelines(mDevice, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &mPipelines[i]));
  }
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
  vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, mPipelineLayout, 0, 1, &mDescriptorSets[0][mImageIndex], 0, nullptr);
}

void GPUDisplayBackendVulkan::endFillCommandBuffer(VkCommandBuffer& commandBuffer, unsigned int imageIndex)
{
  vkCmdEndRenderPass(commandBuffer);
  CHKERR(vkEndCommandBuffer(commandBuffer));
}

void GPUDisplayBackendVulkan::clearPipeline()
{
  for (auto& pipeline : mPipelines) {
    vkDestroyPipeline(mDevice, pipeline, nullptr);
  }
  vkDestroyRenderPass(mDevice, mRenderPass, nullptr);
  vkDestroyRenderPass(mDevice, mRenderPassText, nullptr);
  vkDestroyPipelineLayout(mDevice, mPipelineLayout, nullptr);
  vkDestroyPipelineLayout(mDevice, mPipelineLayoutText, nullptr);
}

// ---------------------------- VULKAN SHADERS ----------------------------

#define LOAD_SHADER(file, ext) \
  mShaders[#file] = createShaderModule(_binary_shaders_display_shaders_##file##_##ext##_spv_start, _binary_shaders_display_shaders_##file##_##ext##_spv_len, mDevice)

void GPUDisplayBackendVulkan::createShaders()
{
  LOAD_SHADER(vertex, vert);
  LOAD_SHADER(fragment, frag);
  LOAD_SHADER(vertexPoint, vert);
  LOAD_SHADER(vertexText, vert);
  LOAD_SHADER(fragmentText, frag);
}

void GPUDisplayBackendVulkan::clearShaders()
{
  for (auto& module : mShaders) {
    vkDestroyShaderModule(mDevice, module.second, nullptr);
  }
}

// ---------------------------- VULKAN BUFFERS ----------------------------

void GPUDisplayBackendVulkan::writeToBuffer(VulkanBuffer& buffer, size_t size, const void* srcData)
{
  if (!buffer.deviceMemory) {
    void* dstData;
    vkMapMemory(mDevice, buffer.memory, 0, buffer.size, 0, &dstData);
    memcpy(dstData, srcData, size);
    vkUnmapMemory(mDevice, buffer.memory);
  } else {
    auto tmp = createBuffer(size, srcData, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, false);

    VkCommandBuffer commandBuffer = getSingleTimeCommandBuffer();
    VkBufferCopy copyRegion{};
    copyRegion.size = size;
    vkCmdCopyBuffer(commandBuffer, tmp.buffer, buffer.buffer, 1, &copyRegion);
    submitSingleTimeCommandBuffer(commandBuffer);

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
  for (auto& buf : mFontVertexBuffer) {
    if (buf.size) {
      clearBuffer(buf);
    }
    buf.size = 0;
  }
}

// ---------------------------- VULKAN TEXTURES ----------------------------

void GPUDisplayBackendVulkan::writeToImage(VulkanImage& image, const void* srcData, size_t srcSize)
{
  auto tmp = createBuffer(srcSize, srcData, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, false);

  VkCommandBuffer commandBuffer = getSingleTimeCommandBuffer();
  transitionImageLayout(commandBuffer, image.image, image.format, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
  VkBufferImageCopy region{};
  region.bufferOffset = 0;
  region.bufferRowLength = 0;
  region.bufferImageHeight = 0;
  region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  region.imageSubresource.mipLevel = 0;
  region.imageSubresource.baseArrayLayer = 0;
  region.imageSubresource.layerCount = 1;
  region.imageOffset = {0, 0, 0};
  region.imageExtent = {image.sizex, image.sizey, 1};
  vkCmdCopyBufferToImage(commandBuffer, tmp.buffer, image.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
  transitionImageLayout(commandBuffer, image.image, image.format, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  submitSingleTimeCommandBuffer(commandBuffer);

  clearBuffer(tmp);
}

VulkanImage GPUDisplayBackendVulkan::createImage(unsigned int sizex, unsigned int sizey, const void* srcData, size_t srcSize, VkFormat format)
{
  VulkanImage image;
  createImageI(mDevice, mPhysicalDevice, image.image, image.memory, sizex, sizey, format, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_IMAGE_TILING_OPTIMAL, VK_SAMPLE_COUNT_1_BIT);

  image.view = createImageViewI(mDevice, image.image, format);

  image.sizex = sizex;
  image.sizey = sizey;
  image.format = format;

  if (srcData) {
    writeToImage(image, srcData, srcSize);
  }
  return image;
}

void GPUDisplayBackendVulkan::clearImage(VulkanImage& image)
{
  vkDestroyImageView(mDevice, image.view, nullptr);
  vkDestroyImage(mDevice, image.image, nullptr);
  vkFreeMemory(mDevice, image.memory, nullptr);
}

// ---------------------------- VULKAN INIT EXIT ----------------------------

int GPUDisplayBackendVulkan::InitBackendA()
{
  std::cout << "Initializing Vulkan\n";

  mEnableValidationLayers = mDisplay->param() && mDisplay->param()->par.debugLevel >= 2;
  mFramesInFlight = 2;

  createDevice();
  createShaders();
  createUniformLayouts();
  createTextureSampler();
  createSwapChain();
  createPipeline();

  std::cout << "Vulkan initialized\n";
  return (0);
}

void GPUDisplayBackendVulkan::ExitBackendA()
{
  std::cout << "Exiting Vulkan\n";
  vkDeviceWaitIdle(mDevice);
  if (mFontImage) {
    clearImage(*mFontImage);
  }
  clearVertexBuffers();
  clearPipeline();
  clearSwapChain();
  clearTextureSampler();
  clearUniformLayouts();
  clearShaders();
  clearDevice();
  std::cout << "Vulkan destroyed\n";
}

// ---------------------------- USER CODE ----------------------------

void GPUDisplayBackendVulkan::resizeScene(unsigned int width, unsigned int height)
{
  if (mExtent.width == width && mExtent.height == height) {
    return;
  }
  mMustUpdateSwapChain = true;
  /*if (mExtent.width != width || mExtent.height != height) {
    std::cout << "Unmatching window size: requested " << width << " x " << height << " - found " << mExtent.width << " x " << mExtent.height << "\n";
  }*/
}

void GPUDisplayBackendVulkan::clearScreen(bool colorOnly)
{
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

void GPUDisplayBackendVulkan::prepareDraw(const hmm_mat4& proj, const hmm_mat4& view, bool requestScreenshot)
{
  hasDrawnText = false;
  if (mDisplay->updateDrawCommands()) {
    needRecordCommandBuffers();
  }
  vkWaitForFences(mDevice, 1, &mInFlightFence[mCurrentFrame], VK_TRUE, UINT64_MAX);

  VkResult retVal = VK_SUCCESS;
  if (mDisplay->updateRenderPipeline() || (requestScreenshot && !mSwapchainImageReadable)) {
    mMustUpdateSwapChain = true;
  } else {
    retVal = vkAcquireNextImageKHR(mDevice, mSwapChain, UINT64_MAX, mImageAvailableSemaphore[mCurrentFrame], VK_NULL_HANDLE, &mImageIndex);
  }
  mScreenshotRequested = requestScreenshot;
  if (mMustUpdateSwapChain || retVal == VK_ERROR_OUT_OF_DATE_KHR || retVal == VK_SUBOPTIMAL_KHR) {
    if (!mMustUpdateSwapChain) {
      GPUInfo("Pipeline out of data / suboptimal, recreating");
    }
    recreateSwapChain();
    retVal = vkAcquireNextImageKHR(mDevice, mSwapChain, UINT64_MAX, mImageAvailableSemaphore[mCurrentFrame], VK_NULL_HANDLE, &mImageIndex);
  }
  CHKERR(retVal);
  mMustUpdateSwapChain = false;
  vkResetFences(mDevice, 1, &mInFlightFence[mCurrentFrame]);

  const hmm_mat4 modelViewProj = proj * view;
  writeToBuffer(mUniformBuffersMat[0][mImageIndex], sizeof(modelViewProj), &modelViewProj);

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

void GPUDisplayBackendVulkan::finishFrame()
{
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
  CHKERR(vkQueueSubmit(mGraphicsQueue, 1, &submitInfo, hasDrawnText ? VK_NULL_HANDLE : mInFlightFence[mCurrentFrame]));

  if (mScreenshotRequested) {
    vkDeviceWaitIdle(mDevice);
    readImageToPixels(mSwapChainImages[mImageIndex], mScreenshotPixels);
    mScreenshotRequested = false;
  }

  if (hasDrawnText) {
    submitInfo.pWaitSemaphores = &mRenderFinishedSemaphore[mCurrentFrame];
    submitInfo.pCommandBuffers = &mCommandBuffersText[mImageIndex];
    submitInfo.pSignalSemaphores = &mTextFinishedSemaphore[mCurrentFrame];
    CHKERR(vkQueueSubmit(mGraphicsQueue, 1, &submitInfo, mInFlightFence[mCurrentFrame]));
  }

  VkPresentInfoKHR presentInfo{};
  presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
  presentInfo.waitSemaphoreCount = 1;
  presentInfo.pWaitSemaphores = hasDrawnText ? &mTextFinishedSemaphore[mCurrentFrame] : &mRenderFinishedSemaphore[mCurrentFrame];
  presentInfo.swapchainCount = 1;
  presentInfo.pSwapchains = &mSwapChain;
  presentInfo.pImageIndices = &mImageIndex;
  presentInfo.pResults = nullptr;
  vkQueuePresentKHR(mGraphicsQueue, &presentInfo);
  mCurrentFrame = (mCurrentFrame + 1) % mFramesInFlight;
}

void GPUDisplayBackendVulkan::prepareText()
{
  hmm_mat4 proj = HMM_Orthographic(0.f, mDisplay->screenWidth(), 0.f, mDisplay->screenHeight(), -1, 1);
  writeToBuffer(mUniformBuffersMat[1][mImageIndex], sizeof(proj), &proj);

  mFontVertexBufferHost.clear();
  mTextDrawCommands.clear();
}

void GPUDisplayBackendVulkan::finishText()
{
  if (!hasDrawnText) {
    return;
  }

  vkResetCommandBuffer(mCommandBuffersText[mImageIndex], 0);

  VkCommandBufferBeginInfo beginInfo{};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags = 0;                  // Optional
  beginInfo.pInheritanceInfo = nullptr; // Optional
  CHKERR(vkBeginCommandBuffer(mCommandBuffersText[mImageIndex], &beginInfo));

  VkRenderPassBeginInfo renderPassInfo{};
  renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
  renderPassInfo.renderPass = mRenderPassText;
  renderPassInfo.framebuffer = mFramebuffersText.size() ? mFramebuffersText[mImageIndex] : mFramebuffers[mImageIndex];
  renderPassInfo.renderArea.offset = {0, 0};
  renderPassInfo.renderArea.extent = mExtent;
  renderPassInfo.clearValueCount = 0;
  vkCmdBeginRenderPass(mCommandBuffersText[mImageIndex], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

  if (mFontVertexBuffer[mImageIndex].size) {
    clearBuffer(mFontVertexBuffer[mImageIndex]);
  }
  mFontVertexBuffer[mImageIndex] = createBuffer(mFontVertexBufferHost.size() * sizeof(float), mFontVertexBufferHost.data(), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, false);

  vkCmdBindPipeline(mCommandBuffersText[mImageIndex], VK_PIPELINE_BIND_POINT_GRAPHICS, mPipelines[3]);
  vkCmdBindDescriptorSets(mCommandBuffersText[mImageIndex], VK_PIPELINE_BIND_POINT_GRAPHICS, mPipelineLayoutText, 0, 1, &mDescriptorSets[1][mImageIndex], 0, nullptr);
  VkDeviceSize offsets[] = {0};
  vkCmdBindVertexBuffers(mCommandBuffersText[mImageIndex], 0, 1, &mFontVertexBuffer[mImageIndex].buffer, offsets);

  for (const auto& cmd : mTextDrawCommands) {
    vkCmdPushConstants(mCommandBuffersText[mImageIndex], mPipelineLayoutText, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(cmd.color), cmd.color);
    vkCmdDraw(mCommandBuffersText[mImageIndex], cmd.nVertices, 1, cmd.firstVertex, 0);
  }

  mFontVertexBufferHost.clear();

  vkCmdEndRenderPass(mCommandBuffersText[mImageIndex]);
  CHKERR(vkEndCommandBuffer(mCommandBuffersText[mImageIndex]));
}

void GPUDisplayBackendVulkan::ActivateColor(std::array<float, 4>& color)
{
  if (mCommandBufferUpToDate[mImageIndex]) {
    return;
  }
  vkCmdPushConstants(mCommandBuffers[mImageIndex], mPipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(color), color.data());
}

void GPUDisplayBackendVulkan::pointSizeFactor(float factor)
{
  if (mCommandBufferUpToDate[mImageIndex]) {
    return;
  }
  float size = mDisplay->cfgL().pointSize * (mDisplay->cfgR().drawQualityDownsampleFSAA > 1 ? mDisplay->cfgR().drawQualityDownsampleFSAA : 1) * factor;
  vkCmdPushConstants(mCommandBuffers[mImageIndex], mPipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, sizeof(std::array<float, 4>), sizeof(size), &size);
}

void GPUDisplayBackendVulkan::lineWidthFactor(float factor)
{
  if (mCommandBufferUpToDate[mImageIndex]) {
    return;
  }
  vkCmdSetLineWidth(mCommandBuffers[mImageIndex], mDisplay->cfgL().lineWidth * (mDisplay->cfgR().drawQualityDownsampleFSAA > 1 ? mDisplay->cfgR().drawQualityDownsampleFSAA : 1) * factor);
}

void GPUDisplayBackendVulkan::needRecordCommandBuffers()
{
  std::fill(mCommandBufferUpToDate.begin(), mCommandBufferUpToDate.end(), false);
}

void GPUDisplayBackendVulkan::addFontSymbol(int symbol, int sizex, int sizey, int offsetx, int offsety, int advance, void* data)
{
  if (symbol != (int)mFontSymbols.size()) {
    throw std::runtime_error("Incorrect symbol ID");
  }
  mFontSymbols.emplace_back(FontSymbolVulkan{sizex, sizey, offsetx, offsety, advance, nullptr, 0.f, 0.f, 0.f, 0.f});
  auto& buffer = mFontSymbols.back().data;
  buffer.reset(new char[sizex * sizey]);
  memcpy(buffer.get(), data, sizex * sizey);
}

void GPUDisplayBackendVulkan::initializeTextDrawing()
{
  int maxSizeX = 0, maxSizeY = 0, maxBigX = 0, maxBigY = 0, maxRowY = 0;
  // Build a mega texture containing all fonts
  for (auto& symbol : mFontSymbols) {
    maxSizeX = std::max(maxSizeX, symbol.size[0]);
    maxSizeY = std::max(maxSizeY, symbol.size[1]);
  }
  unsigned int nn = ceil(std::sqrt(mFontSymbols.size()));
  int sizex = nn * maxSizeX;
  int sizey = nn * maxSizeY;
  std::unique_ptr<char[]> bigImage{new char[sizex * sizey]};
  memset(bigImage.get(), 0, sizex * sizey);
  int rowy = 0, colx = 0;
  for (unsigned int i = 0; i < mFontSymbols.size(); i++) {
    auto& s = mFontSymbols[i];
    if (colx + s.size[0] > sizex) {
      colx = 0;
      rowy += maxRowY;
      maxRowY = 0;
    }
    for (int k = 0; k < s.size[1]; k++) {
      for (int j = 0; j < s.size[0]; j++) {
        bigImage.get()[(colx + j) + (rowy + k) * sizex] = s.data.get()[j + k * s.size[0]];
      }
    }
    s.data.reset();
    s.x0 = colx;
    s.x1 = colx + s.size[0];
    s.y0 = rowy;
    s.y1 = rowy + s.size[1];
    maxBigX = std::max(maxBigX, colx + s.size[0]);
    maxBigY = std::max(maxBigY, rowy + s.size[1]);
    maxRowY = std::max(maxRowY, s.size[1]);
    colx += s.size[0];
  }
  if (maxBigX != sizex) {
    for (int y = 1; y < maxBigY; y++) {
      memcpy(bigImage.get() + y * maxBigX, bigImage.get() + y * sizex, maxBigX);
    }
  }
  sizex = maxBigX;
  sizey = maxBigY;
  for (unsigned int i = 0; i < mFontSymbols.size(); i++) {
    auto& s = mFontSymbols[i];
    s.x0 /= sizex;
    s.x1 /= sizex;
    s.y0 /= sizey;
    s.y1 /= sizey;
  }

  mFontImage = std::make_unique<VulkanImage>(createImage(sizex, sizey, bigImage.get(), sizex * sizey, VK_FORMAT_R8_UNORM));

  VkDescriptorImageInfo imageInfo{};
  imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  imageInfo.imageView = mFontImage->view;
  imageInfo.sampler = mTextSampler;
  for (unsigned int i = 0; i < mFramesInFlight; i++) {
    VkWriteDescriptorSet descriptorWrite{};
    descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrite.dstSet = mDescriptorSets[1][i];
    descriptorWrite.dstBinding = 2;
    descriptorWrite.dstArrayElement = 0;
    descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    descriptorWrite.descriptorCount = 1;
    descriptorWrite.pImageInfo = &imageInfo;
    vkUpdateDescriptorSets(mDevice, 1, &descriptorWrite, 0, nullptr);
  }
}

void GPUDisplayBackendVulkan::OpenGLPrint(const char* s, float x, float y, float* color, float scale)
{
  if (!mFreetypeInitialized || mDisplay->drawTextInCompatMode()) {
    return;
  }

  size_t firstVertex = mFontVertexBufferHost.size() / 4;

  float renderHeight = mDisplay->screenHeight() - 1;
  scale *= 0.25f; // Font size is 48 to have nice bitmap, scale to size 12

  for (const char* c = s; *c; c++) {
    if ((int)*c > (int)mFontSymbols.size()) {
      GPUError("Trying to draw unsupported symbol: %d > %d\n", (int)*c, (int)mFontSymbols.size());
      continue;
    }
    const FontSymbolVulkan& sym = mFontSymbols[*c];
    if (sym.size[0] && sym.size[1]) {
      hasDrawnText = true;
      float xpos = x + sym.offset[0] * scale;
      float ypos = y - (sym.size[1] - sym.offset[1]) * scale;
      float w = sym.size[0] * scale;
      float h = sym.size[1] * scale;
      float vertices[6][4] = {
        {xpos, renderHeight - ypos, sym.x0, sym.y1},
        {xpos, renderHeight - (ypos + h), sym.x0, sym.y0},
        {xpos + w, renderHeight - ypos, sym.x1, sym.y1},
        {xpos + w, renderHeight - ypos, sym.x1, sym.y1},
        {xpos, renderHeight - (ypos + h), sym.x0, sym.y0},
        {xpos + w, renderHeight - (ypos + h), sym.x1, sym.y0}};
      size_t oldSize = mFontVertexBufferHost.size();
      mFontVertexBufferHost.resize(oldSize + 4 * 6);
      memcpy(&mFontVertexBufferHost[oldSize], &vertices[0][0], sizeof(vertices));
    }
    x += (sym.advance >> 6) * scale; // shift is in 1/64th of a pixel
  }

  size_t nVertices = mFontVertexBufferHost.size() / 4 - firstVertex;

  if (nVertices) {
    auto& c = mTextDrawCommands;
    if (c.size() && c.back().color[0] == color[0] && c.back().color[1] == color[1] && c.back().color[2] == color[2] && c.back().color[3] == color[3]) {
      c.back().nVertices += nVertices;
    } else {
      c.emplace_back(TextDrawCommand{firstVertex, nVertices, {color[0], color[1], color[2], color[3]}});
    }
  }
}

void GPUDisplayBackendVulkan::readImageToPixels(VkImage image, std::vector<char>& pixels)
{
  pixels.resize(mExtent.width * mExtent.height * 4);

  VkImage dstImage;
  VkDeviceMemory dstImageMemory;
  createImageI(mDevice, mPhysicalDevice, dstImage, dstImageMemory, mExtent.width, mExtent.height, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, VK_IMAGE_TILING_LINEAR);
  VkCommandBuffer cmdBuffer = getSingleTimeCommandBuffer();

  VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  cmdImageMemoryBarrier(cmdBuffer, dstImage, 0, VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, range);
  cmdImageMemoryBarrier(cmdBuffer, image, VK_ACCESS_MEMORY_READ_BIT, VK_ACCESS_TRANSFER_READ_BIT, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, range);

  VkImageCopy imageCopyRegion{};
  imageCopyRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  imageCopyRegion.srcSubresource.layerCount = 1;
  imageCopyRegion.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  imageCopyRegion.dstSubresource.layerCount = 1;
  imageCopyRegion.extent.width = mExtent.width;
  imageCopyRegion.extent.height = mExtent.height;
  imageCopyRegion.extent.depth = 1;

  vkCmdCopyImage(cmdBuffer, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dstImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &imageCopyRegion);

  cmdImageMemoryBarrier(cmdBuffer, dstImage, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, range);
  cmdImageMemoryBarrier(cmdBuffer, image, VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_MEMORY_READ_BIT, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, range);
  submitSingleTimeCommandBuffer(cmdBuffer);

  VkImageSubresource subResource{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0};
  VkSubresourceLayout subResourceLayout;
  vkGetImageSubresourceLayout(mDevice, dstImage, &subResource, &subResourceLayout);
  const char* data;
  vkMapMemory(mDevice, dstImageMemory, 0, VK_WHOLE_SIZE, 0, (void**)&data);
  data += subResourceLayout.offset;
  memcpy(pixels.data(), data, pixels.size());
  vkUnmapMemory(mDevice, dstImageMemory);
  vkFreeMemory(mDevice, dstImageMemory, nullptr);
  vkDestroyImage(mDevice, dstImage, nullptr);
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

void GPUDisplayBackendVulkan::setQuality()
{
}

void GPUDisplayBackendVulkan::SetVSync(bool enable)
{
  recreateSwapChain();
}

void GPUDisplayBackendVulkan::setDepthBuffer()
{
}

void GPUDisplayBackendVulkan::setFrameBuffer(int updateCurrent, unsigned int newID)
{
}

void GPUDisplayBackendVulkan::renderOffscreenBuffer(GLfb& buffer, GLfb& bufferNoMSAA, int mainBuffer)
{
}

void GPUDisplayBackendVulkan::mixImages(GLfb& mixBuffer, float mixSlaveImage)
{
  {
    GPUWarning("Image mixing unsupported in Vulkan profile");
  }
}
