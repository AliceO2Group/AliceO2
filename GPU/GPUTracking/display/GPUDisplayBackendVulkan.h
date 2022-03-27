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

/// \file GPUDisplayBackendVulkan.h
/// \author David Rohr

#ifndef GPUDISPLAYBACKENDVULKAN_H
#define GPUDISPLAYBACKENDVULKAN_H

#include "GPUDisplayBackend.h"
#include <vulkan/vulkan.hpp>

#include <vector>
#include <unordered_map>
#include <utils/vecpod.h>

namespace GPUCA_NAMESPACE
{
namespace gpu
{

class GPUDisplayBackendVulkan : public GPUDisplayBackend
{
 public:
  GPUDisplayBackendVulkan();
  ~GPUDisplayBackendVulkan();

  unsigned int DepthBits() override;

 protected:
  struct SwapChainSupportDetails {
    vk::SurfaceCapabilitiesKHR capabilities;
    std::vector<vk::SurfaceFormatKHR> formats;
    std::vector<vk::PresentModeKHR> presentModes;
  };
  struct VulkanBuffer {
    vk::Buffer buffer;
    vk::DeviceMemory memory;
    size_t size = 0;
    bool deviceMemory;
  };
  struct VulkanImage {
    vk::Image image;
    vk::ImageView view;
    vk::DeviceMemory memory;
    unsigned int sizex = 0, sizey = 0;
    vk::Format format;
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

  unsigned int drawVertices(const vboList& v, const drawType t) override;
  void ActivateColor(std::array<float, 4>& color) override;
  void SetVSync(bool enable) override { mMustUpdateSwapChain = true; };
  void setDepthBuffer() override{};
  bool backendNeedRedraw() override;
  int InitBackendA() override;
  void ExitBackendA() override;
  void loadDataToGPU(size_t totalVertizes) override;
  void prepareDraw(const hmm_mat4& proj, const hmm_mat4& view, bool requestScreenshot, bool toMixBuffer, float includeMixImage) override;
  void finishDraw(bool doScreenshot, bool toMixBuffer, float includeMixImage) override;
  void finishFrame(bool doScreenshot, bool toMixBuffer, float includeMixImage) override;
  void prepareText() override;
  void finishText() override;
  void mixImages(vk::CommandBuffer cmdBuffer, float mixSlaveImage);
  void pointSizeFactor(float factor) override;
  void lineWidthFactor(float factor) override;
  backendTypes backendType() const override { return TYPE_VULKAN; }
  void resizeScene(unsigned int width, unsigned int height) override;
  float getYFactor() const override { return -1.0f; }

  double checkDevice(vk::PhysicalDevice device, const std::vector<const char*>& reqDeviceExtensions);
  vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities);
  void transitionImageLayout(vk::CommandBuffer commandBuffer, vk::Image image, vk::Format format, vk::ImageLayout oldLayout, vk::ImageLayout newLayout);
  VulkanBuffer createBuffer(size_t size, const void* srcData = nullptr, vk::BufferUsageFlags type = vk::BufferUsageFlagBits::eVertexBuffer, int deviceMemory = 1);
  void writeToBuffer(VulkanBuffer& buffer, size_t size, const void* srcData);
  void clearBuffer(VulkanBuffer& buffer);
  VulkanImage createImage(unsigned int sizex, unsigned int sizey, const void* srcData = nullptr, size_t srcSize = 0, vk::Format format = vk::Format::eR8G8B8A8Srgb);
  void writeToImage(VulkanImage& image, const void* srcData, size_t srcSize);
  void clearImage(VulkanImage& image);
  void clearVertexBuffers();

  void startFillCommandBuffer(vk::CommandBuffer& commandBuffer, unsigned int imageIndex, bool toMixBuffer = false);
  void endFillCommandBuffer(vk::CommandBuffer& commandBuffer, unsigned int imageIndex);
  vk::CommandBuffer getSingleTimeCommandBuffer();
  void submitSingleTimeCommandBuffer(vk::CommandBuffer commandBuffer);
  void readImageToPixels(vk::Image image, vk::ImageLayout layout, std::vector<char>& pixels);
  void downsampleToFramebuffer(vk::CommandBuffer& commandBuffer);

  void updateSwapChainDetails(const vk::PhysicalDevice& device);
  void updateFontTextureDescriptor();
  void createDevice();
  void createShaders();
  void createTextureSampler();
  void createCommandBuffers();
  void createSemaphoresAndFences();
  void createUniformLayoutsAndBuffers();
  void createSwapChain(bool forScreenshot = false, bool forMixing = false);
  void createOffscreenBuffers(bool forScreenshot = false, bool forMixing = false);
  void createPipeline();
  void clearDevice();
  void clearShaders();
  void clearTextureSampler();
  void clearCommandBuffers();
  void clearSemaphoresAndFences();
  void clearUniformLayoutsAndBuffers();
  void clearOffscreenBuffers();
  void clearSwapChain();
  void clearPipeline();
  void recreateRendering(bool forScreenshot = false, bool forMixing = false);
  void needRecordCommandBuffers();

  void addFontSymbol(int symbol, int sizex, int sizey, int offsetx, int offsety, int advance, void* data) override;
  void initializeTextDrawing() override;
  void OpenGLPrint(const char* s, float x, float y, float* color, float scale) override;

  unsigned int mGraphicsFamily;
  SwapChainSupportDetails mSwapChainDetails;
  bool mEnableValidationLayers = false;

  vk::Instance mInstance;
  vk::DynamicLoader mDL;
  vk::DispatchLoaderDynamic mDLD;
  vk::DebugUtilsMessengerEXT mDebugMessenger;
  vk::PhysicalDevice mPhysicalDevice;
  vk::Device mDevice;
  vk::Queue mGraphicsQueue;
  vk::SurfaceKHR mSurface;
  vk::SurfaceFormatKHR mSurfaceFormat;
  vk::PresentModeKHR mPresentMode;
  vk::SwapchainKHR mSwapChain;
  bool mMustUpdateSwapChain = false;
  std::vector<vk::Image> mSwapChainImages;
  std::vector<vk::ImageView> mSwapChainImageViews;
  std::vector<vk::ImageView*> mRenderTargetView;
  std::vector<VulkanImage> mMSAAImages;
  std::vector<VulkanImage> mDownsampleImages;
  std::vector<VulkanImage> mZImages;
  std::vector<VulkanImage> mMixImages;
  std::unordered_map<std::string, vk::ShaderModule> mShaders;
  vk::PipelineLayout mPipelineLayout;
  vk::PipelineLayout mPipelineLayoutTexture;
  vk::RenderPass mRenderPass;
  vk::RenderPass mRenderPassText;
  vk::RenderPass mRenderPassTexture;
  std::vector<vk::Pipeline> mPipelines;
  std::vector<vk::Framebuffer> mFramebuffers;
  std::vector<vk::Framebuffer> mFramebuffersText;
  std::vector<vk::Framebuffer> mFramebuffersTexture;
  vk::CommandPool mCommandPool;

  bool mCommandInfrastructureCreated = false;
  unsigned int mImageCount = 0;
  unsigned int mFramesInFlight = 0;
  int mCurrentFrame = 0;
  uint32_t mCurrentImageIndex = 0;
  vk::CommandBuffer mCurrentCommandBuffer;
  int mCurrentCommandBufferLastPipeline = -1;

  std::vector<vk::CommandBuffer> mCommandBuffers;
  std::vector<vk::CommandBuffer> mCommandBuffersDownsample;
  std::vector<vk::CommandBuffer> mCommandBuffersText;
  std::vector<vk::CommandBuffer> mCommandBuffersTexture;
  std::vector<vk::CommandBuffer> mCommandBuffersMix;
  std::vector<bool> mCommandBufferUpToDate;
  std::vector<vk::Semaphore> mImageAvailableSemaphore;
  std::vector<vk::Semaphore> mRenderFinishedSemaphore;
  std::vector<vk::Semaphore> mTextFinishedSemaphore;
  std::vector<vk::Semaphore> mMixFinishedSemaphore;
  std::vector<vk::Semaphore> mDownsampleFinishedSemaphore;
  std::vector<vk::Fence> mInFlightFence;
  std::vector<VulkanBuffer> mUniformBuffersMat[3];
  std::vector<VulkanBuffer> mUniformBuffersCol[3];
  std::vector<vk::DescriptorSet> mDescriptorSets[3];
  vk::DescriptorSetLayout mUniformDescriptor;
  vk::DescriptorSetLayout mUniformDescriptorTexture;
  vk::DescriptorPool mDescriptorPool;

  VulkanBuffer mVBO;
  VulkanBuffer mIndirectCommandBuffer;

  std::vector<FontSymbolVulkan> mFontSymbols;
  VulkanImage mFontImage;
  std::vector<VulkanBuffer> mFontVertexBuffer;
  std::vector<TextDrawCommand> mTextDrawCommands;
  vk::CommandBuffer mTmpTextCommandBuffer;
  vk::Sampler mTextureSampler;
  vecpod<float> mFontVertexBufferHost;
  bool mHasDrawnText = false;

  bool mSwapchainImageReadable = false;
  vk::SampleCountFlagBits mMSAASampleCount = vk::SampleCountFlagBits::e16;
  unsigned int mMaxMSAAsupported = 0;
  bool mZActive = false;
  bool mZSupported = false;
  bool mDownsampleFSAA = false;
  bool mMixingSupported = 0;

  VulkanBuffer mMixingTextureVertexArray;

  vk::Fence mSingleCommitFence;
};
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
