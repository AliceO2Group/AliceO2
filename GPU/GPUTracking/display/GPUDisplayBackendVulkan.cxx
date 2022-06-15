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
#include <mutex>

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

#include "GPUCommonDef.h"
#include "GPUDisplayBackendVulkan.h"
#include "GPUDisplay.h"

using namespace GPUCA_NAMESPACE::gpu;

#include "utils/qGetLdBinarySymbols.h"
QGET_LD_BINARY_SYMBOLS(shaders_shaders_vertex_vert_spv);
QGET_LD_BINARY_SYMBOLS(shaders_shaders_fragment_frag_spv);
QGET_LD_BINARY_SYMBOLS(shaders_shaders_vertexPoint_vert_spv);
QGET_LD_BINARY_SYMBOLS(shaders_shaders_vertexTexture_vert_spv);
QGET_LD_BINARY_SYMBOLS(shaders_shaders_fragmentTexture_frag_spv);
QGET_LD_BINARY_SYMBOLS(shaders_shaders_fragmentText_frag_spv);

//#define CHKERR(cmd) {cmd;}
#define CHKERR(cmd)                                                                                     \
  do {                                                                                                  \
    auto tmp_internal_retVal = cmd;                                                                     \
    if ((int)tmp_internal_retVal < 0) {                                                                 \
      GPUError("VULKAN ERROR: %d: %s (%s: %d)", (int)tmp_internal_retVal, "ERROR", __FILE__, __LINE__); \
      throw std::runtime_error("Vulkan Failure");                                                       \
    }                                                                                                   \
  } while (false)

GPUDisplayBackendVulkan::GPUDisplayBackendVulkan()
{
  mBackendType = TYPE_VULKAN;
  mBackendName = "Vulkan";
}
GPUDisplayBackendVulkan::~GPUDisplayBackendVulkan() = default;

// ---------------------------- VULKAN HELPERS ----------------------------

static int checkValidationLayerSupport(const std::vector<const char*>& validationLayers)
{
  std::vector<vk::LayerProperties> availableLayers = vk::enumerateInstanceLayerProperties();
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

static uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties, vk::PhysicalDevice physDev)
{
  vk::PhysicalDeviceMemoryProperties memProperties = physDev.getMemoryProperties();

  for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
    if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
      return i;
    }
  }

  throw std::runtime_error("failed to find suitable memory type!");
}

static vk::SurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats)
{
  for (const auto& availableFormat : availableFormats) {
    if (availableFormat.format == vk::Format::eB8G8R8A8Unorm && availableFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
      return availableFormat;
    }
  }
  return availableFormats[0];
}

static vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes, vk::PresentModeKHR desiredMode = vk::PresentModeKHR::eMailbox)
{
  for (const auto& availablePresentMode : availablePresentModes) {
    if (availablePresentMode == desiredMode) {
      return availablePresentMode;
    }
  }
  static bool errorShown = false;
  if (!errorShown) {
    errorShown = true;
    GPUError("VULKAN ERROR: Desired present mode not available, using FIFO mode");
  }
  return vk::PresentModeKHR::eFifo;
}

vk::Extent2D GPUDisplayBackendVulkan::chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities)
{
  if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
    return capabilities.currentExtent;
  } else {
    int width, height;
    mDisplay->frontend()->getSize(width, height);
    vk::Extent2D actualExtent = {(uint32_t)width, (uint32_t)height};
    actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
    actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);
    return actualExtent;
  }
}

static vk::ShaderModule createShaderModule(const char* code, size_t size, vk::Device device)
{
  vk::ShaderModuleCreateInfo createInfo{};
  createInfo.codeSize = size;
  createInfo.pCode = reinterpret_cast<const uint32_t*>(code);
  return device.createShaderModule(createInfo, nullptr);
}

static void cmdImageMemoryBarrier(vk::CommandBuffer cmdbuffer, vk::Image image, vk::AccessFlags srcAccessMask, vk::AccessFlags dstAccessMask, vk::ImageLayout oldLayout, vk::ImageLayout newLayout, vk::PipelineStageFlags srcStageMask, vk::PipelineStageFlags dstStageMask)
{
  vk::ImageSubresourceRange range{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
  vk::ImageMemoryBarrier barrier{};
  barrier.srcAccessMask = srcAccessMask;
  barrier.dstAccessMask = dstAccessMask;
  barrier.oldLayout = oldLayout;
  barrier.newLayout = newLayout;
  barrier.image = image;
  barrier.subresourceRange = range;
  cmdbuffer.pipelineBarrier(srcStageMask, dstStageMask, {}, 0, nullptr, 0, nullptr, 1, &barrier);
}

void GPUDisplayBackendVulkan::updateSwapChainDetails(const vk::PhysicalDevice& device)
{
  mSwapChainDetails.capabilities = device.getSurfaceCapabilitiesKHR(mSurface);
  mSwapChainDetails.formats = device.getSurfaceFormatsKHR(mSurface);
  mSwapChainDetails.presentModes = device.getSurfacePresentModesKHR(mSurface);
}

vk::CommandBuffer GPUDisplayBackendVulkan::getSingleTimeCommandBuffer()
{
  vk::CommandBufferAllocateInfo allocInfo{};
  allocInfo.level = vk::CommandBufferLevel::ePrimary;
  allocInfo.commandPool = mCommandPool;
  allocInfo.commandBufferCount = 1;
  vk::CommandBuffer commandBuffer = mDevice.allocateCommandBuffers(allocInfo)[0];
  vk::CommandBufferBeginInfo beginInfo{};
  beginInfo.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
  commandBuffer.begin(beginInfo);
  return commandBuffer;
}

void GPUDisplayBackendVulkan::submitSingleTimeCommandBuffer(vk::CommandBuffer commandBuffer)
{
  commandBuffer.end();
  vk::SubmitInfo submitInfo{};
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &commandBuffer;
  static std::mutex fenceMutex;
  {
    std::lock_guard<std::mutex> guard(fenceMutex);
    CHKERR(mGraphicsQueue.submit(1, &submitInfo, mSingleCommitFence));
    CHKERR(mDevice.waitForFences(1, &mSingleCommitFence, true, UINT64_MAX));
    CHKERR(mDevice.resetFences(1, &mSingleCommitFence));
  }
  mDevice.freeCommandBuffers(mCommandPool, 1, &commandBuffer);
}

static vk::ImageView createImageViewI(vk::Device device, vk::Image image, vk::Format format, vk::ImageAspectFlags aspectFlags = vk::ImageAspectFlagBits::eColor, uint32_t mipLevels = 1)
{
  vk::ImageViewCreateInfo viewInfo{};
  viewInfo.image = image;
  viewInfo.viewType = vk::ImageViewType::e2D;
  viewInfo.format = format;
  viewInfo.subresourceRange.aspectMask = aspectFlags;
  viewInfo.subresourceRange.baseMipLevel = 0;
  viewInfo.subresourceRange.levelCount = mipLevels;
  viewInfo.subresourceRange.baseArrayLayer = 0;
  viewInfo.subresourceRange.layerCount = 1;
  return device.createImageView(viewInfo, nullptr);
}

static void createImageI(vk::Device device, vk::PhysicalDevice physicalDevice, vk::Image& image, vk::DeviceMemory& imageMemory, uint32_t width, uint32_t height, vk::Format format, vk::ImageUsageFlags usage, vk::MemoryPropertyFlags properties, vk::ImageTiling tiling = vk::ImageTiling::eOptimal, vk::SampleCountFlagBits numSamples = vk::SampleCountFlagBits::e1, vk::ImageLayout layout = vk::ImageLayout::eUndefined, uint32_t mipLevels = 1)
{
  vk::ImageCreateInfo imageInfo{};
  imageInfo.imageType = vk::ImageType::e2D;
  imageInfo.extent.width = width;
  imageInfo.extent.height = height;
  imageInfo.extent.depth = 1;
  imageInfo.mipLevels = mipLevels;
  imageInfo.arrayLayers = 1;
  imageInfo.format = format;
  imageInfo.tiling = tiling;
  imageInfo.initialLayout = layout;
  imageInfo.usage = usage;
  imageInfo.samples = numSamples;
  imageInfo.sharingMode = vk::SharingMode::eExclusive;
  image = device.createImage(imageInfo);

  vk::MemoryRequirements memRequirements;
  memRequirements = device.getImageMemoryRequirements(image);

  vk::MemoryAllocateInfo allocInfo{};
  allocInfo.allocationSize = memRequirements.size;
  allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties, physicalDevice);
  imageMemory = device.allocateMemory(allocInfo, nullptr);

  device.bindImageMemory(image, imageMemory, 0);
}

static unsigned int getMaxUsableSampleCount(vk::PhysicalDeviceProperties& physicalDeviceProperties)
{
  vk::SampleCountFlags counts = physicalDeviceProperties.limits.framebufferColorSampleCounts & physicalDeviceProperties.limits.framebufferDepthSampleCounts;
  if (counts & vk::SampleCountFlagBits::e64) {
    return 64;
  } else if (counts & vk::SampleCountFlagBits::e32) {
    return 32;
  } else if (counts & vk::SampleCountFlagBits::e16) {
    return 16;
  } else if (counts & vk::SampleCountFlagBits::e8) {
    return 8;
  } else if (counts & vk::SampleCountFlagBits::e4) {
    return 4;
  } else if (counts & vk::SampleCountFlagBits::e2) {
    return 2;
  }
  return 1;
}

static vk::SampleCountFlagBits getMSAASamplesFlag(unsigned int msaa)
{
  if (msaa == 2) {
    return vk::SampleCountFlagBits::e2;
  } else if (msaa == 4) {
    return vk::SampleCountFlagBits::e4;
  } else if (msaa == 8) {
    return vk::SampleCountFlagBits::e8;
  } else if (msaa == 16) {
    return vk::SampleCountFlagBits::e16;
  } else if (msaa == 32) {
    return vk::SampleCountFlagBits::e32;
  } else if (msaa == 64) {
    return vk::SampleCountFlagBits::e64;
  }
  return vk::SampleCountFlagBits::e1;
}

template <class T, class S>
static inline void clearVector(T& v, S func, bool downsize = true)
{
  std::for_each(v.begin(), v.end(), func);
  if (downsize) {
    v.clear();
  }
}

// ---------------------------- VULKAN DEVICE MANAGEMENT ----------------------------

double GPUDisplayBackendVulkan::checkDevice(vk::PhysicalDevice device, const std::vector<const char*>& reqDeviceExtensions)
{
  double score = -1.;
  vk::PhysicalDeviceProperties deviceProperties = device.getProperties();
  vk::PhysicalDeviceFeatures deviceFeatures = device.getFeatures();
  vk::PhysicalDeviceMemoryProperties memoryProperties = device.getMemoryProperties();
  if (!deviceFeatures.geometryShader || !deviceFeatures.wideLines || !deviceFeatures.largePoints) {
    return (-1);
  }

  std::vector<vk::QueueFamilyProperties> queueFamilies = device.getQueueFamilyProperties();
  bool found = false;
  for (unsigned int i = 0; i < queueFamilies.size(); i++) {
    if (!(queueFamilies[i].queueFlags & vk::QueueFlagBits::eGraphics)) {
      return (-1);
    }
    vk::Bool32 presentSupport = device.getSurfaceSupportKHR(i, mSurface);
    if (!presentSupport) {
      return (-1);
    }
    mGraphicsFamily = i;
    found = true;
    break;
  }
  if (!found) {
    GPUInfo("%s ignored due to missing queue properties", deviceProperties.deviceName.data());
    return (-1);
  }

  std::vector<vk::ExtensionProperties> availableExtensions = device.enumerateDeviceExtensionProperties(nullptr);
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
    GPUInfo("%s ignored due to missing extensions", deviceProperties.deviceName.data());
    return (-1);
  }

  updateSwapChainDetails(device);
  if (mSwapChainDetails.formats.empty() || mSwapChainDetails.presentModes.empty()) {
    GPUInfo("%s ignored due to incompatible swap chain", deviceProperties.deviceName.data());
    return (-1);
  }

  score = 1;
  if (deviceProperties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu) {
    score += 1e12;
  } else if (deviceProperties.deviceType == vk::PhysicalDeviceType::eIntegratedGpu) {
    score += 1e11;
  }

  for (unsigned int i = 0; i < memoryProperties.memoryHeapCount; i++) {
    if (memoryProperties.memoryHeaps[i].flags & vk::MemoryHeapFlagBits::eDeviceLocal) {
      score += memoryProperties.memoryHeaps[i].size;
    }
  }

  return score;
}

void GPUDisplayBackendVulkan::createDevice()
{
  vk::ApplicationInfo appInfo{};
  appInfo.pApplicationName = "GPU CA Standalone display";
  appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.pEngineName = "GPU CI Standalone Engine";
  appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.apiVersion = VK_API_VERSION_1_0;

  vk::InstanceCreateInfo instanceCreateInfo;
  instanceCreateInfo.pApplicationInfo = &appInfo;

  const char** frontendExtensions;
  uint32_t frontendExtensionCount = mDisplay->frontend()->getReqVulkanExtensions(frontendExtensions);
  std::vector<const char*> reqInstanceExtensions(frontendExtensions, frontendExtensions + frontendExtensionCount);

  const std::vector<const char*> reqValidationLayers = {
    "VK_LAYER_KHRONOS_validation"};
  auto debugCallback = [](VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData) -> VkBool32 {
    static bool throwOnError = getenv("GPUCA_VULKAN_VALIDATION_THROW") && atoi(getenv("GPUCA_VULKAN_VALIDATION_THROW"));
    switch (messageSeverity) {
      case VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT:
        // GPUInfo("%s", pCallbackData->pMessage);
        break;
      case VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT:
        GPUWarning("%s", pCallbackData->pMessage);
        if (throwOnError) {
          throw std::logic_error("break_on_validation_warning");
        }
        break;
      case VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT:
        GPUError("%s", pCallbackData->pMessage);
        if (throwOnError) {
          throw std::logic_error("break_on_validation_error");
        }
        break;
      case VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT:
      default:
        GPUInfo("%s", pCallbackData->pMessage);
        break;
    }
    return false;
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

  mInstance = vk::createInstance(instanceCreateInfo, nullptr);
  mDLD = {mInstance, mDL.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr")};

  if (mEnableValidationLayers) {
    vk::DebugUtilsMessengerCreateInfoEXT createInfo{};
    createInfo.messageSeverity = vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose | vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning | vk::DebugUtilsMessageSeverityFlagBitsEXT::eError;
    createInfo.messageType = vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral | vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation | vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance;
    createInfo.pfnUserCallback = debugCallback;
    createInfo.pUserData = nullptr;
    mDebugMessenger = mInstance.createDebugUtilsMessengerEXT(createInfo, nullptr, mDLD);
  }
  std::vector<vk::ExtensionProperties> extensions = vk::enumerateInstanceExtensionProperties(nullptr);
  if (mDisplay->param()->par.debugLevel >= 3) {
    std::cout << "available instance extensions: " << extensions.size() << "\n";
    for (const auto& extension : extensions) {
      std::cout << '\t' << extension.extensionName << '\n';
    }
  }

  if (mDisplay->frontend()->getVulkanSurface(&mInstance, &mSurface)) {
    throw std::runtime_error("Frontend does not provide Vulkan surface");
  }

  const std::vector<const char*> reqDeviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME};

  mPhysicalDevice = VkPhysicalDevice(VK_NULL_HANDLE);
  std::vector<vk::PhysicalDevice> devices = mInstance.enumeratePhysicalDevices();
  if (devices.size() == 0) {
    throw std::runtime_error("No Vulkan device present!");
  }
  double bestScore = -1.;
  for (unsigned int i = 0; i < devices.size(); i++) {
    double score = checkDevice(devices[i], reqDeviceExtensions);
    if (mDisplay->param()->par.debugLevel >= 2) {
      vk::PhysicalDeviceProperties deviceProperties = devices[i].getProperties();
      GPUInfo("Available Vulkan device %d: %s - Score %f", i, deviceProperties.deviceName.data(), score);
    }
    if (score > bestScore && score > 0) {
      mPhysicalDevice = devices[i];
      bestScore = score;
    }
  }

  if (!mPhysicalDevice) {
    throw std::runtime_error("All available Vulkan devices unsuited");
  }

  updateSwapChainDetails(mPhysicalDevice);
  vk::PhysicalDeviceProperties deviceProperties = mPhysicalDevice.getProperties();
  vk::PhysicalDeviceFeatures deviceFeatures = mPhysicalDevice.getFeatures();
  vk::FormatProperties depth32FormatProperties = mPhysicalDevice.getFormatProperties(vk::Format::eD32Sfloat);
  vk::FormatProperties depth64FormatProperties = mPhysicalDevice.getFormatProperties(vk::Format::eD32SfloatS8Uint);
  vk::FormatProperties formatProperties = mPhysicalDevice.getFormatProperties(mSurfaceFormat.format);
  GPUInfo("Using physical Vulkan device %s", deviceProperties.deviceName.data());
  mMaxMSAAsupported = getMaxUsableSampleCount(deviceProperties);
  mZSupported = (bool)(depth32FormatProperties.optimalTilingFeatures & vk::FormatFeatureFlagBits::eDepthStencilAttachment);
  mStencilSupported = (bool)(depth64FormatProperties.optimalTilingFeatures & vk::FormatFeatureFlagBits::eDepthStencilAttachment);
  mCubicFilterSupported = (bool)(formatProperties.optimalTilingFeatures & vk::FormatFeatureFlagBits::eSampledImageFilterCubicEXT);
  bool mailboxSupported = std::find(mSwapChainDetails.presentModes.begin(), mSwapChainDetails.presentModes.end(), vk::PresentModeKHR::eMailbox) != mSwapChainDetails.presentModes.end();
  if (mDisplay->param()->par.debugLevel >= 2) {
    GPUInfo("Max MSAA: %d, 32 bit Z buffer %d, 32 bit Z buffer + stencil buffer %d, Cubic Filtering %d, Mailbox present mode %d\n", (int)mMaxMSAAsupported, (int)mZSupported, (int)mStencilSupported, (int)mCubicFilterSupported, (int)mailboxSupported);
  }

  vk::DeviceQueueCreateInfo queueCreateInfo{};
  queueCreateInfo.queueFamilyIndex = mGraphicsFamily;
  queueCreateInfo.queueCount = 1;
  float queuePriority = 1.0f;
  queueCreateInfo.pQueuePriorities = &queuePriority;
  vk::DeviceCreateInfo deviceCreateInfo{};
  deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
  deviceCreateInfo.queueCreateInfoCount = 1;
  deviceCreateInfo.pEnabledFeatures = &deviceFeatures;
  deviceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(reqDeviceExtensions.size());
  deviceCreateInfo.ppEnabledExtensionNames = reqDeviceExtensions.data();
  deviceCreateInfo.enabledLayerCount = instanceCreateInfo.enabledLayerCount;
  deviceCreateInfo.ppEnabledLayerNames = instanceCreateInfo.ppEnabledLayerNames;
  mDevice = mPhysicalDevice.createDevice(deviceCreateInfo, nullptr);
  mGraphicsQueue = mDevice.getQueue(mGraphicsFamily, 0);

  vk::CommandPoolCreateInfo poolInfo{};
  poolInfo.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
  poolInfo.queueFamilyIndex = mGraphicsFamily;
  mCommandPool = mDevice.createCommandPool(poolInfo, nullptr);
}

void GPUDisplayBackendVulkan::clearDevice()
{
  mDevice.destroyCommandPool(mCommandPool, nullptr);
  mDevice.destroy(nullptr);
  mInstance.destroySurfaceKHR(mSurface, nullptr);
  if (mEnableValidationLayers) {
    mInstance.destroyDebugUtilsMessengerEXT(mDebugMessenger, nullptr, mDLD);
  }
}

// ---------------------------- VULKAN COMMAND BUFFERS ----------------------------

void GPUDisplayBackendVulkan::createCommandBuffers()
{
  vk::CommandBufferAllocateInfo allocInfo{};
  allocInfo.commandPool = mCommandPool;
  allocInfo.level = vk::CommandBufferLevel::ePrimary;
  allocInfo.commandBufferCount = mFramesInFlight;
  mCommandBufferUpToDate.resize(mFramesInFlight, false);
  mCommandBuffers = mDevice.allocateCommandBuffers(allocInfo);
  mCommandBuffersText = mDevice.allocateCommandBuffers(allocInfo);
  mCommandBuffersTexture = mDevice.allocateCommandBuffers(allocInfo);
  mCommandBuffersDownsample = mDevice.allocateCommandBuffers(allocInfo);
  mCommandBuffersMix = mDevice.allocateCommandBuffers(allocInfo);
}

void GPUDisplayBackendVulkan::clearCommandBuffers()
{
  mDevice.freeCommandBuffers(mCommandPool, mCommandBuffers.size(), mCommandBuffers.data());
  mDevice.freeCommandBuffers(mCommandPool, mCommandBuffersText.size(), mCommandBuffersText.data());
  mDevice.freeCommandBuffers(mCommandPool, mCommandBuffersTexture.size(), mCommandBuffersTexture.data());
  mDevice.freeCommandBuffers(mCommandPool, mCommandBuffersDownsample.size(), mCommandBuffersDownsample.data());
  mDevice.freeCommandBuffers(mCommandPool, mCommandBuffersMix.size(), mCommandBuffersMix.data());
}

// ---------------------------- VULKAN SEMAPHORES AND FENCES ----------------------------

void GPUDisplayBackendVulkan::createSemaphoresAndFences()
{
  vk::SemaphoreCreateInfo semaphoreInfo{};
  vk::FenceCreateInfo fenceInfo{};
  fenceInfo.flags = vk::FenceCreateFlagBits::eSignaled;
  mImageAvailableSemaphore.resize(mFramesInFlight);
  mRenderFinishedSemaphore.resize(mFramesInFlight);
  mTextFinishedSemaphore.resize(mFramesInFlight);
  mMixFinishedSemaphore.resize(mFramesInFlight);
  mDownsampleFinishedSemaphore.resize(mFramesInFlight);
  mInFlightFence.resize(mFramesInFlight);
  for (unsigned int i = 0; i < mFramesInFlight; i++) {
    mImageAvailableSemaphore[i] = mDevice.createSemaphore(semaphoreInfo, nullptr);
    mRenderFinishedSemaphore[i] = mDevice.createSemaphore(semaphoreInfo, nullptr);
    mTextFinishedSemaphore[i] = mDevice.createSemaphore(semaphoreInfo, nullptr);
    mMixFinishedSemaphore[i] = mDevice.createSemaphore(semaphoreInfo, nullptr);
    mDownsampleFinishedSemaphore[i] = mDevice.createSemaphore(semaphoreInfo, nullptr);
    mInFlightFence[i] = mDevice.createFence(fenceInfo, nullptr);
  }
  fenceInfo.flags = {};
  mSingleCommitFence = mDevice.createFence(fenceInfo, nullptr);
}

void GPUDisplayBackendVulkan::clearSemaphoresAndFences()
{
  clearVector(mImageAvailableSemaphore, [&](auto& x) { mDevice.destroySemaphore(x, nullptr); });
  clearVector(mRenderFinishedSemaphore, [&](auto& x) { mDevice.destroySemaphore(x, nullptr); });
  clearVector(mTextFinishedSemaphore, [&](auto& x) { mDevice.destroySemaphore(x, nullptr); });
  clearVector(mMixFinishedSemaphore, [&](auto& x) { mDevice.destroySemaphore(x, nullptr); });
  clearVector(mDownsampleFinishedSemaphore, [&](auto& x) { mDevice.destroySemaphore(x, nullptr); });
  clearVector(mInFlightFence, [&](auto& x) { mDevice.destroyFence(x, nullptr); });
  mDevice.destroyFence(mSingleCommitFence, nullptr);
}

// ---------------------------- VULKAN UNIFORM LAYOUTS AND BUFFERS  ----------------------------

void GPUDisplayBackendVulkan::createUniformLayoutsAndBuffers()
{
  for (int j = 0; j < 3; j++) {
    mUniformBuffersMat[j].resize(mFramesInFlight);
    mUniformBuffersCol[j].resize(mFramesInFlight);
    for (unsigned int i = 0; i < mFramesInFlight; i++) {
      mUniformBuffersMat[j][i] = createBuffer(sizeof(hmm_mat4), nullptr, vk::BufferUsageFlagBits::eUniformBuffer, mDisplay->cfg().vulkan.uniformBuffersInDeviceMemory ? 2 : 0);
      mUniformBuffersCol[j][i] = createBuffer(sizeof(float) * 4, nullptr, vk::BufferUsageFlagBits::eUniformBuffer, mDisplay->cfg().vulkan.uniformBuffersInDeviceMemory ? 2 : 0);
    }
  }

  std::array<vk::DescriptorPoolSize, 2> poolSizes{};
  poolSizes[0].type = vk::DescriptorType::eUniformBuffer;
  poolSizes[0].descriptorCount = (uint32_t)mFramesInFlight * (2 * 3);
  poolSizes[1].type = vk::DescriptorType::eCombinedImageSampler;
  poolSizes[1].descriptorCount = (uint32_t)mFramesInFlight * 2;
  vk::DescriptorPoolCreateInfo poolInfo{};
  poolInfo.poolSizeCount = poolSizes.size();
  poolInfo.pPoolSizes = poolSizes.data();
  poolInfo.maxSets = (uint32_t)mFramesInFlight * 3;
  mDescriptorPool = mDevice.createDescriptorPool(poolInfo, nullptr);

  vk::DescriptorSetLayoutBinding uboLayoutBindingMat{};
  uboLayoutBindingMat.binding = 0;
  uboLayoutBindingMat.descriptorType = vk::DescriptorType::eUniformBuffer;
  uboLayoutBindingMat.descriptorCount = 1;
  uboLayoutBindingMat.stageFlags = vk::ShaderStageFlagBits::eVertex;
  vk::DescriptorSetLayoutBinding uboLayoutBindingCol = uboLayoutBindingMat;
  uboLayoutBindingCol.binding = 1;
  uboLayoutBindingCol.stageFlags = vk::ShaderStageFlagBits::eFragment;
  vk::DescriptorSetLayoutBinding samplerLayoutBinding{};
  samplerLayoutBinding.binding = 2;
  samplerLayoutBinding.descriptorCount = 1;
  samplerLayoutBinding.descriptorType = vk::DescriptorType::eCombinedImageSampler;
  samplerLayoutBinding.stageFlags = vk::ShaderStageFlagBits::eFragment;
  vk::DescriptorSetLayoutBinding bindings[3] = {uboLayoutBindingMat, uboLayoutBindingCol, samplerLayoutBinding};

  vk::DescriptorSetLayoutCreateInfo layoutInfo{};
  layoutInfo.bindingCount = 2;
  layoutInfo.pBindings = bindings;
  mUniformDescriptor = mDevice.createDescriptorSetLayout(layoutInfo, nullptr);
  layoutInfo.bindingCount = 3;
  mUniformDescriptorTexture = mDevice.createDescriptorSetLayout(layoutInfo, nullptr);

  vk::DescriptorSetAllocateInfo allocInfo{};
  allocInfo.descriptorPool = mDescriptorPool;
  allocInfo.descriptorSetCount = (uint32_t)mFramesInFlight;
  for (int j = 0; j < 3; j++) { // 0 = Render, 1 = Text, 2 = Texture
    std::vector<vk::DescriptorSetLayout> layouts(mFramesInFlight, j ? mUniformDescriptorTexture : mUniformDescriptor);
    allocInfo.pSetLayouts = layouts.data();
    mDescriptorSets[j] = mDevice.allocateDescriptorSets(allocInfo);

    for (int k = 0; k < 2; k++) {
      auto& mUniformBuffers = k ? mUniformBuffersCol[j] : mUniformBuffersMat[j];
      for (unsigned int i = 0; i < mFramesInFlight; i++) {
        vk::DescriptorBufferInfo bufferInfo{};
        bufferInfo.buffer = mUniformBuffers[i].buffer;
        bufferInfo.offset = 0;
        bufferInfo.range = mUniformBuffers[i].size;

        vk::WriteDescriptorSet descriptorWrite{};
        descriptorWrite.dstSet = mDescriptorSets[j][i];
        descriptorWrite.dstBinding = k;
        descriptorWrite.dstArrayElement = 0;
        descriptorWrite.descriptorType = vk::DescriptorType::eUniformBuffer;
        descriptorWrite.descriptorCount = 1;
        descriptorWrite.pBufferInfo = &bufferInfo;
        descriptorWrite.pImageInfo = nullptr;
        descriptorWrite.pTexelBufferView = nullptr;
        mDevice.updateDescriptorSets(1, &descriptorWrite, 0, nullptr);
      }
    }
  }

  if (mFontImage.sizex && mFontImage.sizey) {
    updateFontTextureDescriptor();
  }
}

void GPUDisplayBackendVulkan::clearUniformLayoutsAndBuffers()
{
  mDevice.destroyDescriptorSetLayout(mUniformDescriptor, nullptr);
  mDevice.destroyDescriptorSetLayout(mUniformDescriptorTexture, nullptr);
  mDevice.destroyDescriptorPool(mDescriptorPool, nullptr);
  for (int j = 0; j < 3; j++) {
    clearVector(mUniformBuffersMat[j], [&](auto& x) { clearBuffer(x); });
    clearVector(mUniformBuffersCol[j], [&](auto& x) { clearBuffer(x); });
  }
}

void GPUDisplayBackendVulkan::setMixDescriptor(int descriptorIndex, int imageIndex)
{
  vk::DescriptorImageInfo imageInfo{};
  imageInfo.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
  imageInfo.sampler = mTextureSampler;
  imageInfo.imageView = *mRenderTargetView[imageIndex + mImageCount];
  vk::WriteDescriptorSet descriptorWrite{};
  descriptorWrite.dstSet = mDescriptorSets[2][descriptorIndex];
  descriptorWrite.dstBinding = 2;
  descriptorWrite.dstArrayElement = 0;
  descriptorWrite.descriptorType = vk::DescriptorType::eCombinedImageSampler;
  descriptorWrite.descriptorCount = 1;
  descriptorWrite.pImageInfo = &imageInfo;
  mDevice.updateDescriptorSets(1, &descriptorWrite, 0, nullptr);
}

// ---------------------------- VULKAN TEXTURE SAMPLER ----------------------------

void GPUDisplayBackendVulkan::createTextureSampler()
{
  vk::SamplerCreateInfo samplerInfo{};
  samplerInfo.magFilter = vk::Filter::eLinear;
  samplerInfo.minFilter = vk::Filter::eLinear;
  samplerInfo.addressModeU = vk::SamplerAddressMode::eRepeat;
  samplerInfo.addressModeV = vk::SamplerAddressMode::eRepeat;
  samplerInfo.addressModeW = vk::SamplerAddressMode::eRepeat;
  samplerInfo.compareEnable = false;
  samplerInfo.compareOp = vk::CompareOp::eAlways;
  samplerInfo.borderColor = vk::BorderColor::eIntOpaqueBlack;
  samplerInfo.unnormalizedCoordinates = false;
  samplerInfo.mipmapMode = vk::SamplerMipmapMode::eLinear;
  samplerInfo.mipLodBias = 0.0f;
  samplerInfo.minLod = 0.0f;
  samplerInfo.maxLod = 0.0f;
  mTextureSampler = mDevice.createSampler(samplerInfo, nullptr);
}

void GPUDisplayBackendVulkan::clearTextureSampler()
{
  mDevice.destroySampler(mTextureSampler, nullptr);
}

// ---------------------------- VULKAN SWAPCHAIN MANAGEMENT ----------------------------

void GPUDisplayBackendVulkan::createSwapChain(bool forScreenshot, bool forMixing)
{
  mDownsampleFactor = getDownsampleFactor(forScreenshot);
  mDownsampleFSAA = mDownsampleFactor != 1;
  mSwapchainImageReadable = forScreenshot;

  updateSwapChainDetails(mPhysicalDevice);
  mSurfaceFormat = chooseSwapSurfaceFormat(mSwapChainDetails.formats);
  mPresentMode = chooseSwapPresentMode(mSwapChainDetails.presentModes, mDisplay->cfgR().drawQualityVSync ? vk::PresentModeKHR::eMailbox : vk::PresentModeKHR::eImmediate);
  vk::Extent2D extent = chooseSwapExtent(mSwapChainDetails.capabilities);

  uint32_t imageCount = mSwapChainDetails.capabilities.minImageCount + 1;
  if (mSwapChainDetails.capabilities.maxImageCount > 0 && imageCount > mSwapChainDetails.capabilities.maxImageCount) {
    imageCount = mSwapChainDetails.capabilities.maxImageCount;
  }

  mScreenWidth = extent.width;
  mScreenHeight = extent.height;
  mRenderWidth = mScreenWidth * mDownsampleFactor;
  mRenderHeight = mScreenHeight * mDownsampleFactor;

  vk::SwapchainCreateInfoKHR swapCreateInfo{};
  swapCreateInfo.surface = mSurface;
  swapCreateInfo.minImageCount = imageCount;
  swapCreateInfo.imageFormat = mSurfaceFormat.format;
  swapCreateInfo.imageColorSpace = mSurfaceFormat.colorSpace;
  swapCreateInfo.imageExtent = extent;
  swapCreateInfo.imageArrayLayers = 1;
  swapCreateInfo.imageUsage = vk::ImageUsageFlagBits::eColorAttachment;
  swapCreateInfo.imageSharingMode = vk::SharingMode::eExclusive;
  swapCreateInfo.queueFamilyIndexCount = 0;     // Optional
  swapCreateInfo.pQueueFamilyIndices = nullptr; // Optional
  swapCreateInfo.preTransform = mSwapChainDetails.capabilities.currentTransform;
  swapCreateInfo.compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque;
  swapCreateInfo.presentMode = mPresentMode;
  swapCreateInfo.clipped = true;
  swapCreateInfo.oldSwapchain = VkSwapchainKHR(VK_NULL_HANDLE);
  if (mSwapchainImageReadable) {
    swapCreateInfo.imageUsage |= vk::ImageUsageFlagBits::eTransferSrc;
  }
  if (mDownsampleFSAA) {
    swapCreateInfo.imageUsage |= vk::ImageUsageFlagBits::eTransferDst;
  }
  mSwapChain = mDevice.createSwapchainKHR(swapCreateInfo, nullptr);

  mSwapChainImages = mDevice.getSwapchainImagesKHR(mSwapChain);
  unsigned int oldFramesInFlight = mFramesInFlight;
  mImageCount = mSwapChainImages.size();
  mFramesInFlight = mDisplay->cfg().vulkan.nFramesInFlight == 0 ? mImageCount : mDisplay->cfg().vulkan.nFramesInFlight;
  mCommandBufferPerImage = mFramesInFlight == mImageCount;

  if (mFramesInFlight > oldFramesInFlight || !mCommandInfrastructureCreated) {
    if (mCommandInfrastructureCreated) {
      clearUniformLayoutsAndBuffers();
      clearCommandBuffers();
      clearSemaphoresAndFences();
    }
    createUniformLayoutsAndBuffers();
    createCommandBuffers();
    createSemaphoresAndFences();
    mCommandInfrastructureCreated = true;
  }

  mSwapChainImageViews.resize(mImageCount);
  for (unsigned int i = 0; i < mImageCount; i++) {
    mSwapChainImageViews[i] = createImageViewI(mDevice, mSwapChainImages[i], mSurfaceFormat.format);
  }
}

void GPUDisplayBackendVulkan::clearSwapChain()
{
  clearVector(mSwapChainImageViews, [&](auto& x) { mDevice.destroyImageView(x, nullptr); });
  mDevice.destroySwapchainKHR(mSwapChain, nullptr);
}

void GPUDisplayBackendVulkan::recreateRendering(bool forScreenshot, bool forMixing)
{
  mDevice.waitIdle();
  bool needUpdateSwapChain = mMustUpdateSwapChain || mDownsampleFactor != getDownsampleFactor(forScreenshot) || mSwapchainImageReadable != forScreenshot;
  bool needUpdateOffscreenBuffers = needUpdateSwapChain || mMSAASampleCount != getMSAASamplesFlag(std::min<unsigned int>(mMaxMSAAsupported, mDisplay->cfgR().drawQualityMSAA)) || mZActive != (mZSupported && mDisplay->cfgL().depthBuffer) || mMixingSupported != forMixing;
  clearPipeline();
  if (needUpdateOffscreenBuffers) {
    clearOffscreenBuffers();
    if (needUpdateSwapChain) {
      clearSwapChain();
      createSwapChain(forScreenshot, forMixing);
    }
    createOffscreenBuffers(forScreenshot, forMixing);
  }
  createPipeline();
  needRecordCommandBuffers();
}

// ---------------------------- VULKAN OFFSCREEN BUFFERS ----------------------------

void GPUDisplayBackendVulkan::createOffscreenBuffers(bool forScreenshot, bool forMixing)
{
  mMSAASampleCount = getMSAASamplesFlag(std::min<unsigned int>(mMaxMSAAsupported, mDisplay->cfgR().drawQualityMSAA));
  mZActive = mZSupported && mDisplay->cfgL().depthBuffer;
  mMixingSupported = forMixing;

  vk::AttachmentDescription colorAttachment{};
  colorAttachment.format = mSurfaceFormat.format;
  colorAttachment.samples = mMSAASampleCount;
  colorAttachment.loadOp = vk::AttachmentLoadOp::eClear;
  colorAttachment.storeOp = vk::AttachmentStoreOp::eStore;
  colorAttachment.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
  colorAttachment.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
  colorAttachment.initialLayout = vk::ImageLayout::eUndefined;
  colorAttachment.finalLayout = (mMSAASampleCount != vk::SampleCountFlagBits::e1 || mDownsampleFSAA) ? vk::ImageLayout::eColorAttachmentOptimal : vk::ImageLayout::ePresentSrcKHR;
  vk::AttachmentDescription depthAttachment{};
  depthAttachment.format = vk::Format::eD32Sfloat;
  depthAttachment.samples = mMSAASampleCount;
  depthAttachment.loadOp = vk::AttachmentLoadOp::eClear;
  depthAttachment.storeOp = vk::AttachmentStoreOp::eDontCare;
  depthAttachment.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
  depthAttachment.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
  depthAttachment.initialLayout = vk::ImageLayout::eUndefined;
  depthAttachment.finalLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal;
  vk::AttachmentDescription colorAttachmentResolve{};
  colorAttachmentResolve.format = mSurfaceFormat.format;
  colorAttachmentResolve.samples = vk::SampleCountFlagBits::e1;
  colorAttachmentResolve.loadOp = vk::AttachmentLoadOp::eDontCare;
  colorAttachmentResolve.storeOp = vk::AttachmentStoreOp::eStore;
  colorAttachmentResolve.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
  colorAttachmentResolve.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
  colorAttachmentResolve.initialLayout = vk::ImageLayout::eUndefined;
  colorAttachmentResolve.finalLayout = mDownsampleFSAA ? vk::ImageLayout::eColorAttachmentOptimal : vk::ImageLayout::ePresentSrcKHR;
  int nAttachments = 0;
  vk::AttachmentReference colorAttachmentRef{};
  colorAttachmentRef.attachment = nAttachments++;
  colorAttachmentRef.layout = vk::ImageLayout::eColorAttachmentOptimal;
  vk::AttachmentReference depthAttachmentRef{};
  // depthAttachmentRef.attachment // below
  depthAttachmentRef.layout = vk::ImageLayout::eDepthStencilAttachmentOptimal;
  vk::AttachmentReference colorAttachmentResolveRef{};
  // colorAttachmentResolveRef.attachment // below
  colorAttachmentResolveRef.layout = vk::ImageLayout::eColorAttachmentOptimal;
  vk::SubpassDescription subpass{};
  subpass.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
  subpass.colorAttachmentCount = 1;
  subpass.pColorAttachments = &colorAttachmentRef;
  vk::SubpassDependency dependency{};
  dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
  dependency.dstSubpass = 0;
  dependency.srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput | vk::PipelineStageFlagBits::eEarlyFragmentTests;
  dependency.srcAccessMask = {};
  dependency.dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput | vk::PipelineStageFlagBits::eEarlyFragmentTests;
  dependency.dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite | vk::AccessFlagBits::eDepthStencilAttachmentWrite;

  std::vector<vk::AttachmentDescription> attachments = {colorAttachment};
  if (mZActive) {
    attachments.emplace_back(depthAttachment);
    depthAttachmentRef.attachment = nAttachments++;
    subpass.pDepthStencilAttachment = &depthAttachmentRef;
  }
  if (mMSAASampleCount != vk::SampleCountFlagBits::e1) {
    attachments.emplace_back(colorAttachmentResolve);
    colorAttachmentResolveRef.attachment = nAttachments++;
    subpass.pResolveAttachments = &colorAttachmentResolveRef;
  }

  vk::RenderPassCreateInfo renderPassInfo{};
  renderPassInfo.attachmentCount = attachments.size();
  renderPassInfo.pAttachments = attachments.data();
  renderPassInfo.subpassCount = 1;
  renderPassInfo.pSubpasses = &subpass;
  renderPassInfo.dependencyCount = 1;
  renderPassInfo.pDependencies = &dependency;
  mRenderPass = mDevice.createRenderPass(renderPassInfo, nullptr);

  // Text overlay goes as extra rendering path
  renderPassInfo.attachmentCount = 1; // Remove depth and MSAA attachments
  renderPassInfo.pAttachments = &colorAttachment;
  subpass.pDepthStencilAttachment = nullptr;
  subpass.pResolveAttachments = nullptr;
  dependency.srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput; // Remove depth/stencil dependencies
  dependency.dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
  dependency.dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite;
  colorAttachment.loadOp = vk::AttachmentLoadOp::eLoad;            // Don't clear the frame buffer
  colorAttachment.initialLayout = vk::ImageLayout::ePresentSrcKHR; // Initial layout is not undefined after 1st pass
  colorAttachment.samples = vk::SampleCountFlagBits::e1;           // No MSAA for Text
  colorAttachment.finalLayout = vk::ImageLayout::ePresentSrcKHR;   // Might have been overwritten above for 1st pass in case of MSAA
  mRenderPassText = mDevice.createRenderPass(renderPassInfo, nullptr);

  if (mMixingSupported) {
    if (mDownsampleFSAA) {
      colorAttachment.initialLayout = vk::ImageLayout::eColorAttachmentOptimal;
      colorAttachment.finalLayout = mDownsampleFSAA ? vk::ImageLayout::eColorAttachmentOptimal : vk::ImageLayout::ePresentSrcKHR;
    }
    mRenderPassTexture = mDevice.createRenderPass(renderPassInfo, nullptr);
  }

  const unsigned int imageCountWithMixImages = mImageCount * (mMixingSupported ? 2 : 1);
  mRenderTargetView.resize(imageCountWithMixImages);
  mFramebuffers.resize(imageCountWithMixImages);
  if (mDownsampleFSAA) {
    mDownsampleImages.resize(imageCountWithMixImages);
  }
  if (mMSAASampleCount != vk::SampleCountFlagBits::e1) {
    mMSAAImages.resize(imageCountWithMixImages);
  }
  if (mZActive) {
    mZImages.resize(imageCountWithMixImages);
  }
  if (mMSAASampleCount != vk::SampleCountFlagBits::e1 || mZActive || mDownsampleFSAA) {
    mFramebuffersText.resize(mImageCount);
  }
  if (mMixingSupported) {
    if (mMSAASampleCount != vk::SampleCountFlagBits::e1 || mZActive || mDownsampleFSAA) {
      mFramebuffersTexture.resize(mImageCount);
    }
    if (!mDownsampleFSAA) {
      mMixImages.resize(mImageCount);
    }
  }
  for (unsigned int i = 0; i < imageCountWithMixImages; i++) {
    if (i < mImageCount) { // Main render chain
      // primary buffer mSwapChainImageViews[i] created as part of createSwapChain, not here
    } else if (!mDownsampleFSAA) { // for rendering to mixBuffer
      createImageI(mDevice, mPhysicalDevice, mMixImages[i - mImageCount].image, mMixImages[i - mImageCount].memory, mRenderWidth, mRenderHeight, mSurfaceFormat.format, vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled, vk::MemoryPropertyFlagBits::eDeviceLocal, vk::ImageTiling::eOptimal);
      mMixImages[i - mImageCount].view = createImageViewI(mDevice, mMixImages[i - mImageCount].image, mSurfaceFormat.format, vk::ImageAspectFlagBits::eColor, 1);
    }
    std::vector<vk::ImageView> att;
    if (mDownsampleFSAA) {
      vk::ImageUsageFlags usage = vk::ImageUsageFlagBits::eColorAttachment | (i >= mImageCount ? vk::ImageUsageFlagBits::eSampled : vk::ImageUsageFlagBits::eTransferSrc);
      createImageI(mDevice, mPhysicalDevice, mDownsampleImages[i].image, mDownsampleImages[i].memory, mRenderWidth, mRenderHeight, mSurfaceFormat.format, usage, vk::MemoryPropertyFlagBits::eDeviceLocal, vk::ImageTiling::eOptimal);
      mDownsampleImages[i].view = createImageViewI(mDevice, mDownsampleImages[i].image, mSurfaceFormat.format, vk::ImageAspectFlagBits::eColor, 1);
      mRenderTargetView[i] = &mDownsampleImages[i].view;
    } else {
      mRenderTargetView[i] = i < mImageCount ? &mSwapChainImageViews[i] : &mMixImages[i - mImageCount].view;
    }
    if (mMSAASampleCount != vk::SampleCountFlagBits::e1) { // First attachment is the render target, either the MSAA buffer or the framebuffer
      createImageI(mDevice, mPhysicalDevice, mMSAAImages[i].image, mMSAAImages[i].memory, mRenderWidth, mRenderHeight, mSurfaceFormat.format, vk::ImageUsageFlagBits::eColorAttachment, vk::MemoryPropertyFlagBits::eDeviceLocal, vk::ImageTiling::eOptimal, mMSAASampleCount);
      mMSAAImages[i].view = createImageViewI(mDevice, mMSAAImages[i].image, mSurfaceFormat.format, vk::ImageAspectFlagBits::eColor, 1);
      att.emplace_back(mMSAAImages[i].view);
    } else {
      att.emplace_back(*mRenderTargetView[i]);
    }
    if (mZActive) {
      createImageI(mDevice, mPhysicalDevice, mZImages[i].image, mZImages[i].memory, mRenderWidth, mRenderHeight, vk::Format::eD32Sfloat, vk::ImageUsageFlagBits::eDepthStencilAttachment, vk::MemoryPropertyFlagBits::eDeviceLocal, vk::ImageTiling::eOptimal, mMSAASampleCount);
      mZImages[i].view = createImageViewI(mDevice, mZImages[i].image, vk::Format::eD32Sfloat, vk::ImageAspectFlagBits::eDepth, 1);
      att.emplace_back(mZImages[i].view);
    }
    if (mMSAASampleCount != vk::SampleCountFlagBits::e1) { // If we use MSAA, we have to resolve to the framebuffer as the last target
      att.emplace_back(*mRenderTargetView[i]);
    }

    vk::FramebufferCreateInfo framebufferInfo{};
    framebufferInfo.renderPass = mRenderPass;
    framebufferInfo.attachmentCount = att.size();
    framebufferInfo.pAttachments = att.data();
    framebufferInfo.width = mRenderWidth;
    framebufferInfo.height = mRenderHeight;
    framebufferInfo.layers = 1;
    mFramebuffers[i] = mDevice.createFramebuffer(framebufferInfo, nullptr);

    if (i < mImageCount && mFramebuffersText.size()) {
      framebufferInfo.attachmentCount = 1;
      framebufferInfo.pAttachments = &mSwapChainImageViews[i];
      framebufferInfo.renderPass = mRenderPassText;
      framebufferInfo.width = mScreenWidth;
      framebufferInfo.height = mScreenHeight;
      mFramebuffersText[i] = mDevice.createFramebuffer(framebufferInfo, nullptr);
    }

    if (i >= mImageCount && mFramebuffersTexture.size()) {
      framebufferInfo.attachmentCount = 1;
      framebufferInfo.pAttachments = mRenderTargetView[i - mImageCount];
      framebufferInfo.renderPass = mRenderPassTexture;
      framebufferInfo.width = mRenderWidth;
      framebufferInfo.height = mRenderHeight;
      mFramebuffersTexture[i - mImageCount] = mDevice.createFramebuffer(framebufferInfo, nullptr);
    }
  }

  if (mMixingSupported) {
    float vertices[6][4] = {
      {0, (float)mRenderHeight, 0.0f, 1.0f},
      {0, 0, 0.0f, 0.0f},
      {(float)mRenderWidth, 0, 1.0f, 0.0f},
      {0, (float)mRenderHeight, 0.0f, 1.0f},
      {(float)mRenderWidth, 0, 1.0f, 0.0f},
      {(float)mRenderWidth, (float)mRenderHeight, 1.0f, 1.0f}};
    mMixingTextureVertexArray = createBuffer(sizeof(vertices), &vertices[0][0], vk::BufferUsageFlagBits::eVertexBuffer, 1);

    if (mCommandBufferPerImage) {
      for (unsigned int i = 0; i < mFramesInFlight; i++) {
        setMixDescriptor(i, i);
      }
    }
  }

  mFontVertexBuffer.resize(mFramesInFlight);
}

void GPUDisplayBackendVulkan::clearOffscreenBuffers()
{
  clearVector(mFramebuffers, [&](auto& x) { mDevice.destroyFramebuffer(x, nullptr); });
  clearVector(mMSAAImages, [&](auto& x) { clearImage(x); });
  clearVector(mDownsampleImages, [&](auto& x) { clearImage(x); });
  clearVector(mZImages, [&](auto& x) { clearImage(x); });
  clearVector(mMixImages, [&](auto& x) { clearImage(x); });
  clearVector(mFramebuffersText, [&](auto& x) { mDevice.destroyFramebuffer(x, nullptr); });
  clearVector(mFramebuffersTexture, [&](auto& x) { mDevice.destroyFramebuffer(x, nullptr); });
  mDevice.destroyRenderPass(mRenderPass, nullptr);
  mDevice.destroyRenderPass(mRenderPassText, nullptr);
  if (mMixingSupported) {
    mDevice.destroyRenderPass(mRenderPassTexture, nullptr);
    clearBuffer(mMixingTextureVertexArray);
  }
}

// ---------------------------- VULKAN PIPELINE ----------------------------

void GPUDisplayBackendVulkan::createPipeline()
{
  vk::PipelineShaderStageCreateInfo shaderStages[2] = {vk::PipelineShaderStageCreateInfo{}, vk::PipelineShaderStageCreateInfo{}};
  vk::PipelineShaderStageCreateInfo& vertShaderStageInfo = shaderStages[0];
  vertShaderStageInfo.stage = vk::ShaderStageFlagBits::eVertex;
  // vertShaderStageInfo.module // below
  vertShaderStageInfo.pName = "main";
  vk::PipelineShaderStageCreateInfo& fragShaderStageInfo = shaderStages[1];
  fragShaderStageInfo.stage = vk::ShaderStageFlagBits::eFragment;
  // fragShaderStageInfo.module // below
  fragShaderStageInfo.pName = "main";

  vk::VertexInputBindingDescription bindingDescription{};
  bindingDescription.binding = 0;
  // bindingDescription.stride // below
  bindingDescription.inputRate = vk::VertexInputRate::eVertex;

  vk::VertexInputAttributeDescription attributeDescriptions{};
  attributeDescriptions.binding = 0;
  attributeDescriptions.location = 0;
  // attributeDescriptions.format // below
  attributeDescriptions.offset = 0;

  vk::PipelineVertexInputStateCreateInfo vertexInputInfo{};
  vertexInputInfo.vertexBindingDescriptionCount = 1;
  vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
  vertexInputInfo.vertexAttributeDescriptionCount = 1;
  vertexInputInfo.pVertexAttributeDescriptions = &attributeDescriptions;
  vk::PipelineInputAssemblyStateCreateInfo inputAssembly{};
  // inputAssembly.topology // below
  inputAssembly.primitiveRestartEnable = false;

  vk::Viewport viewport{};
  viewport.x = 0.0f;
  viewport.y = 0.0f;
  // viewport.width // below
  // viewport.height // below
  viewport.minDepth = 0.0f;
  viewport.maxDepth = 1.0f;

  vk::Rect2D scissor{};
  scissor.offset = vk::Offset2D{0, 0};
  // scissor.extent // below

  vk::PipelineViewportStateCreateInfo viewportState{};
  viewportState.viewportCount = 1;
  viewportState.pViewports = &viewport;
  viewportState.scissorCount = 1;
  viewportState.pScissors = &scissor;

  vk::PipelineRasterizationStateCreateInfo rasterizer{};
  rasterizer.depthClampEnable = false;
  rasterizer.rasterizerDiscardEnable = false;
  rasterizer.polygonMode = vk::PolygonMode::eFill;
  rasterizer.lineWidth = mDisplay->cfgL().lineWidth;
  rasterizer.cullMode = vk::CullModeFlagBits::eBack;
  rasterizer.frontFace = vk::FrontFace::eClockwise;
  rasterizer.depthBiasEnable = false;
  rasterizer.depthBiasConstantFactor = 0.0f; // Optional
  rasterizer.depthBiasClamp = 0.0f;          // Optional
  rasterizer.depthBiasSlopeFactor = 0.0f;    // Optional

  vk::PipelineMultisampleStateCreateInfo multisampling{};
  multisampling.sampleShadingEnable = false;
  // multisampling.rasterizationSamples // below
  multisampling.minSampleShading = 1.0f;          // Optional
  multisampling.pSampleMask = nullptr;            // Optional
  multisampling.alphaToCoverageEnable = false;    // Optional
  multisampling.alphaToOneEnable = false;         // Optional

  vk::PipelineColorBlendAttachmentState colorBlendAttachment{};
  colorBlendAttachment.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
  // colorBlendAttachment.blendEnable // below
  colorBlendAttachment.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
  colorBlendAttachment.srcColorBlendFactor = vk::BlendFactor::eSrcAlpha;
  colorBlendAttachment.dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha;
  colorBlendAttachment.colorBlendOp = vk::BlendOp::eAdd;
  colorBlendAttachment.srcAlphaBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha;
  colorBlendAttachment.dstAlphaBlendFactor = vk::BlendFactor::eZero;
  colorBlendAttachment.alphaBlendOp = vk::BlendOp::eAdd;

  vk::PipelineColorBlendStateCreateInfo colorBlending{};
  colorBlending.logicOpEnable = false;
  colorBlending.logicOp = vk::LogicOp::eCopy;
  colorBlending.attachmentCount = 1;
  colorBlending.pAttachments = &colorBlendAttachment;
  colorBlending.blendConstants[0] = 0.0f;
  colorBlending.blendConstants[1] = 0.0f;
  colorBlending.blendConstants[2] = 0.0f;
  colorBlending.blendConstants[3] = 0.0f;

  vk::PipelineDepthStencilStateCreateInfo depthStencil{};
  depthStencil.depthTestEnable = true;
  depthStencil.depthWriteEnable = true;
  depthStencil.depthCompareOp = vk::CompareOp::eLess;
  depthStencil.depthBoundsTestEnable = false;
  depthStencil.stencilTestEnable = false;

  vk::DynamicState dynamicStates[] = {vk::DynamicState::eLineWidth};
  vk::PipelineDynamicStateCreateInfo dynamicState{};
  dynamicState.dynamicStateCount = 1;
  dynamicState.pDynamicStates = dynamicStates;

  vk::PushConstantRange pushConstantRanges[2] = {vk::PushConstantRange{}, vk::PushConstantRange{}};
  pushConstantRanges[0].stageFlags = vk::ShaderStageFlagBits::eFragment;
  pushConstantRanges[0].offset = 0;
  pushConstantRanges[0].size = sizeof(float) * 4;
  pushConstantRanges[1].stageFlags = vk::ShaderStageFlagBits::eVertex;
  pushConstantRanges[1].offset = pushConstantRanges[0].size;
  pushConstantRanges[1].size = sizeof(float);
  vk::PipelineLayoutCreateInfo pipelineLayoutInfo{};
  pipelineLayoutInfo.setLayoutCount = 1;
  pipelineLayoutInfo.pSetLayouts = &mUniformDescriptor;
  pipelineLayoutInfo.pushConstantRangeCount = 2;
  pipelineLayoutInfo.pPushConstantRanges = pushConstantRanges;
  mPipelineLayout = mDevice.createPipelineLayout(pipelineLayoutInfo, nullptr);
  pipelineLayoutInfo.setLayoutCount = 1;
  pipelineLayoutInfo.pSetLayouts = &mUniformDescriptorTexture;
  mPipelineLayoutTexture = mDevice.createPipelineLayout(pipelineLayoutInfo, nullptr);

  vk::GraphicsPipelineCreateInfo pipelineInfo{};
  pipelineInfo.stageCount = 2;
  pipelineInfo.pVertexInputState = &vertexInputInfo;
  pipelineInfo.pInputAssemblyState = &inputAssembly;
  pipelineInfo.pViewportState = &viewportState;
  pipelineInfo.pRasterizationState = &rasterizer;
  pipelineInfo.pMultisampleState = &multisampling;
  // pipelineInfo.pDepthStencilState // below
  pipelineInfo.pColorBlendState = &colorBlending;
  pipelineInfo.pDynamicState = &dynamicState;
  // pipelineInfo.layout // below
  // pipelineInfo.renderPass // below
  pipelineInfo.subpass = 0;
  pipelineInfo.pStages = shaderStages;
  pipelineInfo.basePipelineHandle = VkPipeline(VK_NULL_HANDLE); // Optional
  pipelineInfo.basePipelineIndex = -1;                          // Optional

  mPipelines.resize(mMixingSupported ? 5 : 4);
  static constexpr vk::PrimitiveTopology types[3] = {vk::PrimitiveTopology::ePointList, vk::PrimitiveTopology::eLineList, vk::PrimitiveTopology::eLineStrip};
  for (unsigned int i = 0; i < mPipelines.size(); i++) {
    if (i == 4) { // Texture rendering
      bindingDescription.stride = 4 * sizeof(float);
      attributeDescriptions.format = vk::Format::eR32G32B32A32Sfloat;
      inputAssembly.topology = vk::PrimitiveTopology::eTriangleList;
      vertShaderStageInfo.module = mShaders["vertexTexture"];
      fragShaderStageInfo.module = mShaders["fragmentTexture"];
      pipelineInfo.layout = mPipelineLayoutTexture;
      pipelineInfo.renderPass = mRenderPassTexture;
      pipelineInfo.pDepthStencilState = nullptr;
      colorBlendAttachment.blendEnable = true;
      multisampling.rasterizationSamples = vk::SampleCountFlagBits::e1;
      viewport.width = scissor.extent.width = mRenderWidth;
      viewport.height = scissor.extent.height = mRenderHeight;
    } else if (i == 3) { // Text rendering
      bindingDescription.stride = 4 * sizeof(float);
      attributeDescriptions.format = vk::Format::eR32G32B32A32Sfloat;
      inputAssembly.topology = vk::PrimitiveTopology::eTriangleList;
      vertShaderStageInfo.module = mShaders["vertexTexture"];
      fragShaderStageInfo.module = mShaders["fragmentText"];
      pipelineInfo.layout = mPipelineLayoutTexture;
      pipelineInfo.renderPass = mRenderPassText;
      pipelineInfo.pDepthStencilState = nullptr;
      colorBlendAttachment.blendEnable = true;
      multisampling.rasterizationSamples = vk::SampleCountFlagBits::e1;
      viewport.width = scissor.extent.width = mScreenWidth;
      viewport.height = scissor.extent.height = mScreenHeight;
    } else { // Point / line / line-strip rendering
      bindingDescription.stride = 3 * sizeof(float);
      attributeDescriptions.format = vk::Format::eR32G32B32Sfloat;
      inputAssembly.topology = types[i];
      vertShaderStageInfo.module = mShaders[types[i] == vk::PrimitiveTopology::ePointList ? "vertexPoint" : "vertex"];
      fragShaderStageInfo.module = mShaders["fragment"];
      pipelineInfo.layout = mPipelineLayout;
      pipelineInfo.renderPass = mRenderPass;
      pipelineInfo.pDepthStencilState = mZActive ? &depthStencil : nullptr;
      colorBlendAttachment.blendEnable = true;
      multisampling.rasterizationSamples = mMSAASampleCount;
      viewport.width = scissor.extent.width = mRenderWidth;
      viewport.height = scissor.extent.height = mRenderHeight;
    }

    CHKERR(mDevice.createGraphicsPipelines(VkPipelineCache(VK_NULL_HANDLE), 1, &pipelineInfo, nullptr, &mPipelines[i])); // TODO: multiple at once + cache?
  }
}

void GPUDisplayBackendVulkan::startFillCommandBuffer(vk::CommandBuffer& commandBuffer, unsigned int imageIndex, bool toMixBuffer)
{
  commandBuffer.reset({});

  vk::CommandBufferBeginInfo beginInfo{};
  beginInfo.flags = {};
  commandBuffer.begin(beginInfo);

  vk::ClearValue clearValues[2];
  clearValues[0].color = mDisplay->cfgL().invertColors ? vk::ClearColorValue{std::array<float, 4>{1.0f, 1.0f, 1.0f, 1.0f}} : vk::ClearColorValue{std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f}};
  clearValues[1].depthStencil = vk::ClearDepthStencilValue{{1.0f, 0}};

  vk::RenderPassBeginInfo renderPassInfo{};
  renderPassInfo.renderPass = mRenderPass;
  renderPassInfo.framebuffer = toMixBuffer ? mFramebuffers[imageIndex + mImageCount] : mFramebuffers[imageIndex];
  renderPassInfo.renderArea.offset = vk::Offset2D{0, 0};
  renderPassInfo.renderArea.extent = vk::Extent2D{mRenderWidth, mRenderHeight};
  renderPassInfo.clearValueCount = mZActive ? 2 : 1;
  renderPassInfo.pClearValues = clearValues;
  commandBuffer.beginRenderPass(&renderPassInfo, vk::SubpassContents::eInline);

  vk::DeviceSize offsets[] = {0};
  commandBuffer.bindVertexBuffers(0, 1, &mVBO.buffer, offsets);
  commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, mPipelineLayout, 0, 1, &mDescriptorSets[0][mCurrentBufferSet], 0, nullptr);
}

void GPUDisplayBackendVulkan::endFillCommandBuffer(vk::CommandBuffer& commandBuffer)
{
  commandBuffer.endRenderPass();
  commandBuffer.end();
}

void GPUDisplayBackendVulkan::clearPipeline()
{
  clearVector(mPipelines, [&](auto& x) { mDevice.destroyPipeline(x, nullptr); });
  mDevice.destroyPipelineLayout(mPipelineLayout, nullptr);
  mDevice.destroyPipelineLayout(mPipelineLayoutTexture, nullptr);
}

// ---------------------------- VULKAN SHADERS ----------------------------

#define LOAD_SHADER(file, ext) \
  mShaders[#file] = createShaderModule(_binary_shaders_shaders_##file##_##ext##_spv_start, _binary_shaders_shaders_##file##_##ext##_spv_len, mDevice)

void GPUDisplayBackendVulkan::createShaders()
{
  LOAD_SHADER(vertex, vert);
  LOAD_SHADER(fragment, frag);
  LOAD_SHADER(vertexPoint, vert);
  LOAD_SHADER(vertexTexture, vert);
  LOAD_SHADER(fragmentTexture, frag);
  LOAD_SHADER(fragmentText, frag);
}

void GPUDisplayBackendVulkan::clearShaders()
{
  clearVector(mShaders, [&](auto& x) { mDevice.destroyShaderModule(x.second, nullptr); });
}

// ---------------------------- VULKAN BUFFERS ----------------------------

void GPUDisplayBackendVulkan::writeToBuffer(VulkanBuffer& buffer, size_t size, const void* srcData)
{
  if (buffer.deviceMemory != 1) {
    void* dstData;
    CHKERR(mDevice.mapMemory(buffer.memory, 0, buffer.size, {}, &dstData));
    memcpy(dstData, srcData, size);
    mDevice.unmapMemory(buffer.memory);
  } else {
    auto tmp = createBuffer(size, srcData, vk::BufferUsageFlagBits::eTransferSrc, 0);

    vk::CommandBuffer commandBuffer = getSingleTimeCommandBuffer();
    vk::BufferCopy copyRegion{};
    copyRegion.size = size;
    commandBuffer.copyBuffer(tmp.buffer, buffer.buffer, 1, &copyRegion);
    submitSingleTimeCommandBuffer(commandBuffer);

    clearBuffer(tmp);
  }
}

GPUDisplayBackendVulkan::VulkanBuffer GPUDisplayBackendVulkan::createBuffer(size_t size, const void* srcData, vk::BufferUsageFlags type, int deviceMemory)
{
  vk::MemoryPropertyFlags properties;
  if (deviceMemory) {
    properties |= vk::MemoryPropertyFlagBits::eDeviceLocal;
  }
  if (deviceMemory == 0 || deviceMemory == 2) {
    properties |= (vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
  }
  if (deviceMemory == 1) {
    type |= vk::BufferUsageFlagBits::eTransferDst;
  }

  VulkanBuffer buffer;
  vk::BufferCreateInfo bufferInfo{};
  bufferInfo.size = size;
  bufferInfo.usage = type;
  bufferInfo.sharingMode = vk::SharingMode::eExclusive;
  buffer.buffer = mDevice.createBuffer(bufferInfo, nullptr);

  vk::MemoryRequirements memRequirements;
  memRequirements = mDevice.getBufferMemoryRequirements(buffer.buffer);
  vk::MemoryAllocateInfo allocInfo{};
  allocInfo.allocationSize = memRequirements.size;
  allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties, mPhysicalDevice);
  buffer.memory = mDevice.allocateMemory(allocInfo, nullptr);

  mDevice.bindBufferMemory(buffer.buffer, buffer.memory, 0);

  buffer.size = size;
  buffer.deviceMemory = deviceMemory;

  if (srcData != nullptr) {
    writeToBuffer(buffer, size, srcData);
  }

  return buffer;
}

void GPUDisplayBackendVulkan::clearBuffer(VulkanBuffer& buffer)
{
  mDevice.destroyBuffer(buffer.buffer, nullptr);
  mDevice.freeMemory(buffer.memory, nullptr);
}

void GPUDisplayBackendVulkan::clearVertexBuffers()
{
  if (mVBO.size) {
    clearBuffer(mVBO);
    mVBO.size = 0;
  }
  if (mIndirectCommandBuffer.size) {
    clearBuffer(mIndirectCommandBuffer);
    mIndirectCommandBuffer.size = 0;
  }
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
  auto tmp = createBuffer(srcSize, srcData, vk::BufferUsageFlagBits::eTransferSrc, 0);

  vk::CommandBuffer commandBuffer = getSingleTimeCommandBuffer();
  cmdImageMemoryBarrier(commandBuffer, image.image, {}, vk::AccessFlagBits::eTransferWrite, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eTransfer);
  vk::BufferImageCopy region{};
  region.bufferOffset = 0;
  region.bufferRowLength = 0;
  region.bufferImageHeight = 0;
  region.imageSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
  region.imageSubresource.mipLevel = 0;
  region.imageSubresource.baseArrayLayer = 0;
  region.imageSubresource.layerCount = 1;
  region.imageOffset = vk::Offset3D{0, 0, 0};
  region.imageExtent = vk::Extent3D{image.sizex, image.sizey, 1};
  commandBuffer.copyBufferToImage(tmp.buffer, image.image, vk::ImageLayout::eTransferDstOptimal, 1, &region);
  cmdImageMemoryBarrier(commandBuffer, image.image, vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal, vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eFragmentShader);
  submitSingleTimeCommandBuffer(commandBuffer);

  clearBuffer(tmp);
}

GPUDisplayBackendVulkan::VulkanImage GPUDisplayBackendVulkan::createImage(unsigned int sizex, unsigned int sizey, const void* srcData, size_t srcSize, vk::Format format)
{
  VulkanImage image;
  createImageI(mDevice, mPhysicalDevice, image.image, image.memory, sizex, sizey, format, vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled, vk::MemoryPropertyFlagBits::eDeviceLocal, vk::ImageTiling::eOptimal, vk::SampleCountFlagBits::e1);

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
  mDevice.destroyImageView(image.view, nullptr);
  mDevice.destroyImage(image.image, nullptr);
  mDevice.freeMemory(image.memory, nullptr);
}

// ---------------------------- VULKAN INIT EXIT ----------------------------

int GPUDisplayBackendVulkan::InitBackendA()
{
  mEnableValidationLayers = mDisplay->param() && mDisplay->param()->par.debugLevel >= 2;
  mFramesInFlight = 2;

  createDevice();
  createShaders();
  createTextureSampler();
  createSwapChain();
  createOffscreenBuffers();
  createPipeline();

  return (0);
}

void GPUDisplayBackendVulkan::ExitBackendA()
{
  mDevice.waitIdle();
  if (mFontImage.sizex && mFontImage.sizey) {
    clearImage(mFontImage);
    mFontImage.sizex = mFontImage.sizey = 0;
  }
  clearVertexBuffers();
  clearPipeline();
  clearOffscreenBuffers();
  clearSwapChain();
  if (mCommandInfrastructureCreated) {
    clearSemaphoresAndFences();
    clearCommandBuffers();
    clearUniformLayoutsAndBuffers();
  }
  clearTextureSampler();
  clearShaders();
  clearDevice();
}

// ---------------------------- USER CODE ----------------------------

void GPUDisplayBackendVulkan::resizeScene(unsigned int width, unsigned int height)
{
  if (mScreenWidth == width && mScreenHeight == height) {
    return;
  }
  updateSwapChainDetails(mPhysicalDevice);
  vk::Extent2D extent = chooseSwapExtent(mSwapChainDetails.capabilities);
  if (extent.width != mScreenWidth || extent.height != mScreenHeight) {
    mMustUpdateSwapChain = true;
  }
}

void GPUDisplayBackendVulkan::loadDataToGPU(size_t totalVertizes)
{
  mDevice.waitIdle();
  clearVertexBuffers();
  mVBO = createBuffer(totalVertizes * sizeof(mDisplay->vertexBuffer()[0][0]), mDisplay->vertexBuffer()[0].data(), vk::BufferUsageFlagBits::eVertexBuffer, 1);
  if (mDisplay->cfgR().useGLIndirectDraw) {
    fillIndirectCmdBuffer();
    mIndirectCommandBuffer = createBuffer(mCmdBuffer.size() * sizeof(mCmdBuffer[0]), mCmdBuffer.data(), vk::BufferUsageFlagBits::eIndirectBuffer, 1);
    mCmdBuffer.clear();
  }
  needRecordCommandBuffers();
}

unsigned int GPUDisplayBackendVulkan::drawVertices(const vboList& v, const drawType tt)
{
  auto first = std::get<0>(v);
  auto count = std::get<1>(v);
  auto iSlice = std::get<2>(v);
  if (count == 0) {
    return 0;
  }
  if (mCommandBufferUpToDate[mCurrentBufferSet]) {
    return count;
  }

  if (mCurrentCommandBufferLastPipeline != tt) {
    mCurrentCommandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, mPipelines[tt]);
    mCurrentCommandBufferLastPipeline = tt;
  }
  if (mDisplay->cfgR().useGLIndirectDraw) {
    mCurrentCommandBuffer.drawIndirect(mIndirectCommandBuffer.buffer, (mIndirectSliceOffset[iSlice] + first) * sizeof(DrawArraysIndirectCommand), count, sizeof(DrawArraysIndirectCommand));
  } else {
    for (unsigned int k = 0; k < count; k++) {
      mCurrentCommandBuffer.draw(mDisplay->vertexBufferCount()[iSlice][first + k], 1, mDisplay->vertexBufferStart()[iSlice][first + k], 0);
    }
  }

  return count;
}

void GPUDisplayBackendVulkan::prepareDraw(const hmm_mat4& proj, const hmm_mat4& view, bool requestScreenshot, bool toMixBuffer, float includeMixImage)
{
  if (mDisplay->updateDrawCommands() || toMixBuffer || includeMixImage > 0) {
    needRecordCommandBuffers();
  }

  if (includeMixImage == 0.f) {
    mCurrentFrame = (mCurrentFrame + 1) % mFramesInFlight;
    CHKERR(mDevice.waitForFences(1, &mInFlightFence[mCurrentFrame], true, UINT64_MAX));
    auto getImage = [&]() {
      vk::Fence fen = VkFence(VK_NULL_HANDLE);
      vk::Semaphore sem = VkSemaphore(VK_NULL_HANDLE);
      if (mCommandBufferPerImage) {
        fen = mInFlightFence[mCurrentFrame];
        CHKERR(mDevice.resetFences(1, &fen));
      } else {
        sem = mImageAvailableSemaphore[mCurrentFrame];
      }
      return mDevice.acquireNextImageKHR(mSwapChain, UINT64_MAX, sem, fen, &mCurrentImageIndex);
    };

    vk::Result retVal = vk::Result::eSuccess;
    bool mustUpdateRendering = mMustUpdateSwapChain;
    if (mDisplay->updateRenderPipeline() || (requestScreenshot && !mSwapchainImageReadable) || (toMixBuffer && !mMixingSupported) || mDownsampleFactor != getDownsampleFactor(requestScreenshot)) {
      mustUpdateRendering = true;
    } else if (!mMustUpdateSwapChain) {
      retVal = getImage();
    }
    if (mMustUpdateSwapChain || mustUpdateRendering || retVal == vk::Result::eErrorOutOfDateKHR || retVal == vk::Result::eSuboptimalKHR) {
      if (!mustUpdateRendering) {
        GPUInfo("Pipeline out of data / suboptimal, recreating");
      }
      recreateRendering(requestScreenshot, toMixBuffer);
      retVal = getImage();
    }
    CHKERR(retVal);
    if (mCommandBufferPerImage) {
      CHKERR(mDevice.waitForFences(1, &mInFlightFence[mCurrentFrame], true, UINT64_MAX));
    }
    CHKERR(mDevice.resetFences(1, &mInFlightFence[mCurrentFrame]));
    mMustUpdateSwapChain = false;
    mHasDrawnText = false;
    mCurrentBufferSet = mCommandBufferPerImage ? mCurrentImageIndex : mCurrentFrame;

    const hmm_mat4 modelViewProj = proj * view;
    writeToBuffer(mUniformBuffersMat[0][mCurrentBufferSet], sizeof(modelViewProj), &modelViewProj);
  }

  mCurrentCommandBuffer = toMixBuffer ? mCommandBuffersMix[mCurrentBufferSet] : mCommandBuffers[mCurrentBufferSet];
  mCurrentCommandBufferLastPipeline = -1;
  if (!mCommandBufferUpToDate[mCurrentBufferSet]) {
    startFillCommandBuffer(mCurrentCommandBuffer, mCurrentImageIndex, toMixBuffer);
  }
}

void GPUDisplayBackendVulkan::finishDraw(bool doScreenshot, bool toMixBuffer, float includeMixImage)
{
  if (!mCommandBufferUpToDate[mCurrentBufferSet]) {
    endFillCommandBuffer(mCurrentCommandBuffer);
    if (!toMixBuffer && includeMixImage == 0.f && mCommandBufferPerImage) {
      mCommandBufferUpToDate[mCurrentBufferSet] = true;
    }
  }
}

void GPUDisplayBackendVulkan::finishFrame(bool doScreenshot, bool toMixBuffer, float includeMixImage)
{
  vk::Semaphore* stageFinishedSemaphore = &mRenderFinishedSemaphore[mCurrentFrame];
  const vk::Fence noFence = VkFence(VK_NULL_HANDLE);

  vk::SubmitInfo submitInfo{};
  vk::PipelineStageFlags waitStages[] = {vk::PipelineStageFlagBits::eColorAttachmentOutput};
  submitInfo.pWaitSemaphores = includeMixImage > 0.f ? &mRenderFinishedSemaphore[mCurrentFrame] : (!mCommandBufferPerImage ? &mImageAvailableSemaphore[mCurrentFrame] : nullptr);
  submitInfo.waitSemaphoreCount = submitInfo.pWaitSemaphores != nullptr ? 1 : 0;
  submitInfo.pWaitDstStageMask = waitStages;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &mCurrentCommandBuffer;
  submitInfo.signalSemaphoreCount = 1;
  submitInfo.pSignalSemaphores = stageFinishedSemaphore;
  CHKERR(mGraphicsQueue.submit(1, &submitInfo, includeMixImage > 0 || toMixBuffer || mHasDrawnText || mDownsampleFSAA ? noFence : mInFlightFence[mCurrentFrame]));
  if (!toMixBuffer) {
    if (includeMixImage > 0.f) {
      mixImages(mCommandBuffersTexture[mCurrentBufferSet], includeMixImage);
      submitInfo.pWaitSemaphores = stageFinishedSemaphore;
      waitStages[0] = {vk::PipelineStageFlagBits::eColorAttachmentOutput};
      submitInfo.waitSemaphoreCount = 1;
      submitInfo.pCommandBuffers = &mCommandBuffersTexture[mCurrentBufferSet];
      stageFinishedSemaphore = &mMixFinishedSemaphore[mCurrentFrame];
      submitInfo.pSignalSemaphores = stageFinishedSemaphore;
      CHKERR(mGraphicsQueue.submit(1, &submitInfo, mHasDrawnText || mDownsampleFSAA ? noFence : mInFlightFence[mCurrentFrame]));
    }

    if (mDownsampleFSAA) {
      downsampleToFramebuffer(mCommandBuffersDownsample[mCurrentBufferSet]);
      submitInfo.pCommandBuffers = &mCommandBuffersDownsample[mCurrentBufferSet];
      submitInfo.pWaitSemaphores = stageFinishedSemaphore;
      waitStages[0] = {vk::PipelineStageFlagBits::eTransfer};
      submitInfo.waitSemaphoreCount = 1;
      stageFinishedSemaphore = &mDownsampleFinishedSemaphore[mCurrentFrame];
      submitInfo.pSignalSemaphores = stageFinishedSemaphore;
      CHKERR(mGraphicsQueue.submit(1, &submitInfo, mHasDrawnText ? noFence : mInFlightFence[mCurrentFrame]));
    }

    if (doScreenshot) {
      mDevice.waitIdle();
      if (mDisplay->cfgR().screenshotScaleFactor != 1) {
        readImageToPixels(mDownsampleImages[mCurrentImageIndex].image, vk::ImageLayout::eColorAttachmentOptimal, mScreenshotPixels);
      } else {
        readImageToPixels(mSwapChainImages[mCurrentImageIndex], vk::ImageLayout::ePresentSrcKHR, mScreenshotPixels);
      }
    }

    if (mHasDrawnText) {
      submitInfo.pWaitSemaphores = stageFinishedSemaphore;
      waitStages[0] = {vk::PipelineStageFlagBits::eColorAttachmentOutput};
      submitInfo.waitSemaphoreCount = 1;
      submitInfo.pCommandBuffers = &mCommandBuffersText[mCurrentBufferSet];
      stageFinishedSemaphore = &mTextFinishedSemaphore[mCurrentFrame];
      submitInfo.pSignalSemaphores = stageFinishedSemaphore;
      CHKERR(mGraphicsQueue.submit(1, &submitInfo, mInFlightFence[mCurrentFrame]));
    }

    vk::PresentInfoKHR presentInfo{};
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = stageFinishedSemaphore;
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = &mSwapChain;
    presentInfo.pImageIndices = &mCurrentImageIndex;
    presentInfo.pResults = nullptr;
    vk::Result retVal = mGraphicsQueue.presentKHR(&presentInfo);
    if (retVal == vk::Result::eErrorOutOfDateKHR) {
      mMustUpdateSwapChain = true;
    } else {
      CHKERR(retVal);
    }
  }
}

void GPUDisplayBackendVulkan::downsampleToFramebuffer(vk::CommandBuffer& commandBuffer)
{
  commandBuffer.reset({});
  vk::CommandBufferBeginInfo beginInfo{};
  beginInfo.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
  commandBuffer.begin(beginInfo);

  cmdImageMemoryBarrier(commandBuffer, mSwapChainImages[mCurrentImageIndex], {}, vk::AccessFlagBits::eTransferWrite, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eTransfer);
  cmdImageMemoryBarrier(commandBuffer, mDownsampleImages[mCurrentImageIndex].image, vk::AccessFlagBits::eMemoryRead, vk::AccessFlagBits::eTransferRead, vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eTransferSrcOptimal, vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eTransfer);

  vk::Offset3D blitSizeSrc;
  blitSizeSrc.x = mRenderWidth;
  blitSizeSrc.y = mRenderHeight;
  blitSizeSrc.z = 1;
  vk::Offset3D blitSizeDst;
  blitSizeDst.x = mScreenWidth;
  blitSizeDst.y = mScreenHeight;
  blitSizeDst.z = 1;
  vk::ImageBlit imageBlitRegion{};
  imageBlitRegion.srcSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
  imageBlitRegion.srcSubresource.layerCount = 1;
  imageBlitRegion.srcOffsets[1] = blitSizeSrc;
  imageBlitRegion.dstSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
  imageBlitRegion.dstSubresource.layerCount = 1;
  imageBlitRegion.dstOffsets[1] = blitSizeDst;
  commandBuffer.blitImage(mDownsampleImages[mCurrentImageIndex].image, vk::ImageLayout::eTransferSrcOptimal, mSwapChainImages[mCurrentImageIndex], vk::ImageLayout::eTransferDstOptimal, 1, &imageBlitRegion, vk::Filter::eLinear);

  cmdImageMemoryBarrier(commandBuffer, mSwapChainImages[mCurrentImageIndex], vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eMemoryRead, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::ePresentSrcKHR, vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eTransfer);
  cmdImageMemoryBarrier(commandBuffer, mDownsampleImages[mCurrentImageIndex].image, vk::AccessFlagBits::eTransferRead, vk::AccessFlagBits::eMemoryRead, vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal, vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eTransfer);

  commandBuffer.end();
}

void GPUDisplayBackendVulkan::prepareText()
{
  hmm_mat4 proj = HMM_Orthographic(0.f, mScreenWidth, 0.f, mScreenHeight, -1, 1);
  writeToBuffer(mUniformBuffersMat[1][mCurrentBufferSet], sizeof(proj), &proj);

  mFontVertexBufferHost.clear();
  mTextDrawCommands.clear();
}

void GPUDisplayBackendVulkan::finishText()
{
  if (!mHasDrawnText) {
    return;
  }

  mCommandBuffersText[mCurrentBufferSet].reset({});

  vk::CommandBufferBeginInfo beginInfo{};
  beginInfo.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
  mCommandBuffersText[mCurrentBufferSet].begin(beginInfo);

  vk::RenderPassBeginInfo renderPassInfo{};
  renderPassInfo.renderPass = mRenderPassText;
  renderPassInfo.framebuffer = mFramebuffersText.size() ? mFramebuffersText[mCurrentImageIndex] : mFramebuffers[mCurrentImageIndex];
  renderPassInfo.renderArea.offset = vk::Offset2D{0, 0};
  renderPassInfo.renderArea.extent = vk::Extent2D{mScreenWidth, mScreenHeight};
  renderPassInfo.clearValueCount = 0;
  mCommandBuffersText[mCurrentBufferSet].beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);

  if (mFontVertexBuffer[mCurrentBufferSet].size) {
    clearBuffer(mFontVertexBuffer[mCurrentBufferSet]);
  }
  mFontVertexBuffer[mCurrentBufferSet] = createBuffer(mFontVertexBufferHost.size() * sizeof(float), mFontVertexBufferHost.data(), vk::BufferUsageFlagBits::eVertexBuffer, 0);

  mCommandBuffersText[mCurrentBufferSet].bindPipeline(vk::PipelineBindPoint::eGraphics, mPipelines[3]);
  mCommandBuffersText[mCurrentBufferSet].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, mPipelineLayoutTexture, 0, 1, &mDescriptorSets[1][mCurrentBufferSet], 0, nullptr);
  vk::DeviceSize offsets[] = {0};
  mCommandBuffersText[mCurrentBufferSet].bindVertexBuffers(0, 1, &mFontVertexBuffer[mCurrentBufferSet].buffer, offsets);

  for (const auto& cmd : mTextDrawCommands) {
    mCommandBuffersText[mCurrentBufferSet].pushConstants(mPipelineLayoutTexture, vk::ShaderStageFlagBits::eFragment, 0, sizeof(cmd.color), cmd.color);
    mCommandBuffersText[mCurrentBufferSet].draw(cmd.nVertices, 1, cmd.firstVertex, 0);
  }

  mFontVertexBufferHost.clear();

  mCommandBuffersText[mCurrentBufferSet].endRenderPass();
  mCommandBuffersText[mCurrentBufferSet].end();
}

void GPUDisplayBackendVulkan::mixImages(vk::CommandBuffer commandBuffer, float mixSlaveImage)
{
  hmm_mat4 proj = HMM_Orthographic(0.f, mRenderWidth, 0.f, mRenderHeight, -1, 1);
  writeToBuffer(mUniformBuffersMat[2][mCurrentBufferSet], sizeof(proj), &proj);

  commandBuffer.reset({});
  vk::CommandBufferBeginInfo beginInfo{};
  beginInfo.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
  commandBuffer.begin(beginInfo);

  vk::Image& image = mDownsampleFSAA ? mDownsampleImages[mCurrentImageIndex + mImageCount].image : mMixImages[mCurrentImageIndex].image;
  vk::ImageLayout srcLayout = mDownsampleFSAA ? vk::ImageLayout::eColorAttachmentOptimal : vk::ImageLayout::ePresentSrcKHR;
  cmdImageMemoryBarrier(commandBuffer, image, {}, vk::AccessFlagBits::eMemoryRead, srcLayout, vk::ImageLayout::eShaderReadOnlyOptimal, vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eFragmentShader);

  vk::RenderPassBeginInfo renderPassInfo{};
  renderPassInfo.renderPass = mRenderPassTexture;
  renderPassInfo.framebuffer = mFramebuffersTexture.size() ? mFramebuffersTexture[mCurrentImageIndex] : mFramebuffers[mCurrentImageIndex];
  renderPassInfo.renderArea.offset = vk::Offset2D{0, 0};
  renderPassInfo.renderArea.extent = vk::Extent2D{mRenderWidth, mRenderHeight};
  renderPassInfo.clearValueCount = 0;
  commandBuffer.beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);

  commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, mPipelines[4]);
  if (!mCommandBufferPerImage) {
    setMixDescriptor(mCurrentBufferSet, mCurrentImageIndex);
  }
  commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, mPipelineLayoutTexture, 0, 1, &mDescriptorSets[2][mCurrentBufferSet], 0, nullptr);
  vk::DeviceSize offsets[] = {0};
  commandBuffer.bindVertexBuffers(0, 1, &mMixingTextureVertexArray.buffer, offsets);

  commandBuffer.pushConstants(mPipelineLayoutTexture, vk::ShaderStageFlagBits::eFragment, 0, sizeof(mixSlaveImage), &mixSlaveImage);
  commandBuffer.draw(6, 1, 0, 0);

  commandBuffer.endRenderPass();
  commandBuffer.end();
}

void GPUDisplayBackendVulkan::ActivateColor(std::array<float, 4>& color)
{
  if (mCommandBufferUpToDate[mCurrentBufferSet]) {
    return;
  }
  mCurrentCommandBuffer.pushConstants(mPipelineLayout, vk::ShaderStageFlagBits::eFragment, 0, sizeof(color), color.data());
}

void GPUDisplayBackendVulkan::pointSizeFactor(float factor)
{
  if (mCommandBufferUpToDate[mCurrentBufferSet]) {
    return;
  }
  float size = mDisplay->cfgL().pointSize * mDownsampleFactor * factor;
  mCurrentCommandBuffer.pushConstants(mPipelineLayout, vk::ShaderStageFlagBits::eVertex, sizeof(std::array<float, 4>), sizeof(size), &size);
}

void GPUDisplayBackendVulkan::lineWidthFactor(float factor)
{
  if (mCommandBufferUpToDate[mCurrentBufferSet]) {
    return;
  }
  mCurrentCommandBuffer.setLineWidth(mDisplay->cfgL().lineWidth * mDownsampleFactor * factor);
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
  mFontSymbols.emplace_back(FontSymbolVulkan{{{sizex, sizey}, {offsetx, offsety}, advance}, nullptr, 0.f, 0.f, 0.f, 0.f});
  auto& buffer = mFontSymbols.back().data;
  if (sizex && sizey) {
    buffer.reset(new char[sizex * sizey]);
    memcpy(buffer.get(), data, sizex * sizey);
  }
}

void GPUDisplayBackendVulkan::initializeTextDrawing()
{
  int maxSizeX = 0, maxSizeY = 0, maxBigX = 0, maxBigY = 0, maxRowY = 0;
  bool smooth = smoothFont();
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
        char val = s.data.get()[j + k * s.size[0]];
        if (!smooth) {
          val = val < 0 ? 0xFF : 0;
        }
        bigImage.get()[(colx + j) + (rowy + k) * sizex] = val;
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
      memmove(bigImage.get() + y * maxBigX, bigImage.get() + y * sizex, maxBigX);
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

  mFontImage = createImage(sizex, sizey, bigImage.get(), sizex * sizey, vk::Format::eR8Unorm);
  updateFontTextureDescriptor();
}

void GPUDisplayBackendVulkan::updateFontTextureDescriptor()
{
  vk::DescriptorImageInfo imageInfo{};
  imageInfo.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
  imageInfo.imageView = mFontImage.view;
  imageInfo.sampler = mTextureSampler;
  for (unsigned int i = 0; i < mFramesInFlight; i++) {
    vk::WriteDescriptorSet descriptorWrite{};
    descriptorWrite.dstSet = mDescriptorSets[1][i];
    descriptorWrite.dstBinding = 2;
    descriptorWrite.dstArrayElement = 0;
    descriptorWrite.descriptorType = vk::DescriptorType::eCombinedImageSampler;
    descriptorWrite.descriptorCount = 1;
    descriptorWrite.pImageInfo = &imageInfo;
    mDevice.updateDescriptorSets(1, &descriptorWrite, 0, nullptr);
  }
}

void GPUDisplayBackendVulkan::OpenGLPrint(const char* s, float x, float y, float* color, float scale)
{
  if (!mFreetypeInitialized || mDisplay->drawTextInCompatMode()) {
    return;
  }

  size_t firstVertex = mFontVertexBufferHost.size() / 4;
  if (smoothFont()) {
    scale *= 0.25f; // Font size is 48 to have nice bitmap, scale to size 12
  }

  for (const char* c = s; *c; c++) {
    if ((int)*c > (int)mFontSymbols.size()) {
      GPUError("Trying to draw unsupported symbol: %d > %d\n", (int)*c, (int)mFontSymbols.size());
      continue;
    }
    const FontSymbolVulkan& sym = mFontSymbols[*c];
    if (sym.size[0] && sym.size[1]) {
      mHasDrawnText = true;
      float xpos = x + sym.offset[0] * scale;
      float ypos = y - (sym.size[1] - sym.offset[1]) * scale;
      float w = sym.size[0] * scale;
      float h = sym.size[1] * scale;
      float vertices[6][4] = {
        {xpos, mScreenHeight - 1 - ypos, sym.x0, sym.y1},
        {xpos, mScreenHeight - 1 - (ypos + h), sym.x0, sym.y0},
        {xpos + w, mScreenHeight - 1 - ypos, sym.x1, sym.y1},
        {xpos + w, mScreenHeight - 1 - ypos, sym.x1, sym.y1},
        {xpos, mScreenHeight - 1 - (ypos + h), sym.x0, sym.y0},
        {xpos + w, mScreenHeight - 1 - (ypos + h), sym.x1, sym.y0}};
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

void GPUDisplayBackendVulkan::readImageToPixels(vk::Image image, vk::ImageLayout layout, std::vector<char>& pixels)
{
  unsigned int width = mScreenWidth * mDisplay->cfgR().screenshotScaleFactor;
  unsigned int height = mScreenHeight * mDisplay->cfgR().screenshotScaleFactor;
  static constexpr int bytesPerPixel = 4;
  pixels.resize(width * height * bytesPerPixel);

  vk::Image dstImage, dstImage2, src2;
  vk::DeviceMemory dstImageMemory, dstImageMemory2;
  createImageI(mDevice, mPhysicalDevice, dstImage, dstImageMemory, width, height, vk::Format::eR8G8B8A8Unorm, vk::ImageUsageFlagBits::eTransferDst, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, vk::ImageTiling::eLinear);
  vk::CommandBuffer cmdBuffer = getSingleTimeCommandBuffer();
  cmdImageMemoryBarrier(cmdBuffer, image, vk::AccessFlagBits::eMemoryRead, vk::AccessFlagBits::eTransferRead, layout, vk::ImageLayout::eTransferSrcOptimal, vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eTransfer);
  if (mDisplay->cfgR().screenshotScaleFactor != 1) {
    createImageI(mDevice, mPhysicalDevice, dstImage2, dstImageMemory2, width, height, mSurfaceFormat.format, vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst, vk::MemoryPropertyFlagBits::eDeviceLocal, vk::ImageTiling::eOptimal);
    cmdImageMemoryBarrier(cmdBuffer, dstImage2, {}, vk::AccessFlagBits::eTransferWrite, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eTransfer);
    vk::Offset3D blitSizeSrc = {(int)mRenderWidth, (int)mRenderHeight, 1};
    vk::Offset3D blitSizeDst = {(int)width, (int)height, 1};
    vk::ImageBlit imageBlitRegion{};
    imageBlitRegion.srcSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
    imageBlitRegion.srcSubresource.layerCount = 1;
    imageBlitRegion.srcOffsets[1] = blitSizeSrc;
    imageBlitRegion.dstSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
    imageBlitRegion.dstSubresource.layerCount = 1;
    imageBlitRegion.dstOffsets[1] = blitSizeDst;
    cmdBuffer.blitImage(image, vk::ImageLayout::eTransferSrcOptimal, dstImage2, vk::ImageLayout::eTransferDstOptimal, 1, &imageBlitRegion, vk::Filter::eLinear);
    src2 = dstImage2;
    cmdImageMemoryBarrier(cmdBuffer, dstImage2, vk::AccessFlagBits::eMemoryRead, vk::AccessFlagBits::eTransferRead, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eTransferSrcOptimal, vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eTransfer);
  } else {
    src2 = image;
  }

  cmdImageMemoryBarrier(cmdBuffer, dstImage, {}, vk::AccessFlagBits::eTransferWrite, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eTransfer);
  vk::ImageCopy imageCopyRegion{};
  imageCopyRegion.srcSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
  imageCopyRegion.srcSubresource.layerCount = 1;
  imageCopyRegion.dstSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
  imageCopyRegion.dstSubresource.layerCount = 1;
  imageCopyRegion.extent.width = width;
  imageCopyRegion.extent.height = height;
  imageCopyRegion.extent.depth = 1;
  cmdBuffer.copyImage(src2, vk::ImageLayout::eTransferSrcOptimal, dstImage, vk::ImageLayout::eTransferDstOptimal, 1, &imageCopyRegion);

  cmdImageMemoryBarrier(cmdBuffer, dstImage, vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eMemoryRead, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eGeneral, vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eTransfer);
  cmdImageMemoryBarrier(cmdBuffer, image, vk::AccessFlagBits::eTransferRead, vk::AccessFlagBits::eMemoryRead, vk::ImageLayout::eTransferSrcOptimal, layout, vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eTransfer);
  submitSingleTimeCommandBuffer(cmdBuffer);

  vk::ImageSubresource subResource{vk::ImageAspectFlagBits::eColor, 0, 0};
  vk::SubresourceLayout subResourceLayout = mDevice.getImageSubresourceLayout(dstImage, subResource);
  const char* data;
  CHKERR(mDevice.mapMemory(dstImageMemory, 0, VK_WHOLE_SIZE, {}, (void**)&data));
  data += subResourceLayout.offset;
  for (unsigned int i = 0; i < height; i++) {
    memcpy(pixels.data() + i * width * bytesPerPixel, data + (height - i - 1) * width * bytesPerPixel, width * bytesPerPixel);
  }
  mDevice.unmapMemory(dstImageMemory);
  mDevice.freeMemory(dstImageMemory, nullptr);
  mDevice.destroyImage(dstImage, nullptr);
  if (mDisplay->cfgR().screenshotScaleFactor != 1) {
    mDevice.freeMemory(dstImageMemory2, nullptr);
    mDevice.destroyImage(dstImage2, nullptr);
  }
}

unsigned int GPUDisplayBackendVulkan::DepthBits()
{
  return 32;
}

bool GPUDisplayBackendVulkan::backendNeedRedraw()
{
  return !mCommandBufferUpToDate[mCurrentBufferSet];
}
