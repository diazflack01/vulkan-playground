// vulkan_guide.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <vector>
#include <functional>
#include <deque>
#include <unordered_map>
#include <chrono>

#include <vk_types.h>

#include <vk_mem_alloc.h>
#include <vk_mesh.h>
#include <glm/glm.hpp>

struct Texture {
    AllocatedImage image;
    VkImageView imageView;
};

struct GPUSceneData {
	glm::vec4 fogColor; // w is for exponent
	glm::vec4 fogDistances; //x for min, y for max, zw unused.
	glm::vec4 ambientColor;
	glm::vec4 sunlightDirection; //w for sun power
	glm::vec4 sunlightColor;
};

struct GPUCameraData {
	glm::mat4 view;
	glm::mat4 proj;
	glm::mat4 viewproj;
};

struct GPUObjectData{
	glm::mat4 modelMatrix;
};

struct UploadContext {
    VkFence _uploadFence;
    VkCommandPool _commandPool;
    VkCommandBuffer _commandBuffer;
};

// Per frame context
struct FrameData {
	VkCommandPool commandPool;
    VkCommandBuffer mainCommandBuffer;

	VkFence renderFence;
	VkSemaphore presentSemaphore;
	VkSemaphore renderSemaphore;

	//buffer that holds a single GPUCameraData to use when rendering
	AllocatedBuffer cameraBuffer;
	AllocatedBuffer objectBuffer;

	VkDescriptorSet globalDescriptor;
	VkDescriptorSet objectDescriptor;
};

struct Material {
    VkDescriptorSet textureSet{VK_NULL_HANDLE}; //texture defaulted to null
	VkPipeline pipeline;
	VkPipelineLayout pipelineLayout;
};

struct RenderObject {
	Mesh* mesh;

	Material* material;

	glm::mat4 transformMatrix;
};

struct MeshPushConstants {
	glm::vec4 data;
	glm::mat4 render_matrix;
};

struct DeletionQueue
{
	std::deque<std::function<void()>> deletors;

	void push_function(std::function<void()>&& function) {
		deletors.push_back(function);
	}

	void flush() {
		// reverse iterate the deletion queue to execute all the functions
		for (auto it = deletors.rbegin(); it != deletors.rend(); it++) {
			(*it)(); //call the function
		}

		deletors.clear();
	}
};

class PipelineBuilder {
public:

    std::vector<VkPipelineShaderStageCreateInfo> _shaderStages;
    VkPipelineVertexInputStateCreateInfo _vertexInputInfo;
    VkPipelineInputAssemblyStateCreateInfo _inputAssembly;
    VkViewport _viewport;
    VkRect2D _scissor;
    VkPipelineRasterizationStateCreateInfo _rasterizer;
    VkPipelineColorBlendAttachmentState _colorBlendAttachment;
    VkPipelineMultisampleStateCreateInfo _multisampling;
	VkPipelineDepthStencilStateCreateInfo _depthStencil;
    VkPipelineLayout _pipelineLayout;

    VkPipeline build_pipeline(VkDevice device, VkRenderPass pass);
};

class VulkanEngine {
public:

	bool _isInitialized{ false };
	size_t _frameNumber {0};
    size_t _lastFrameNumberReported {0};
    size_t _lastFps{0};
    std::chrono::time_point<std::chrono::high_resolution_clock> _lastFpsReportTime{std::chrono::high_resolution_clock::now()};

	VkExtent2D _windowExtent{ 1700 , 900 };

	struct SDL_Window* _window{ nullptr };

	//initializes everything in the engine
	void init();

	//shuts down the engine
	void cleanup();

	//draw loop
	void draw();

	//run main loop
	void run();

	VkInstance _instance; // Vulkan library handle
	VkDebugUtilsMessengerEXT _debug_messenger; // Vulkan debug output handle
	VkPhysicalDevice _chosenGPU; // GPU chosen as the default device
	VkDevice _device; // Vulkan device for commands
	VkSurfaceKHR _surface; // Vulkan window surface

    VkSwapchainKHR _swapchain; // from other articles

    // image format expected by the windowing system
    VkFormat _swapchainImageFormat;

    //array of images from the swapchain
    std::vector<VkImage> _swapchainImages;

    //array of image-views from the swapchain
    std::vector<VkImageView> _swapchainImageViews;

    VkQueue _graphicsQueue; //queue we will submit to
    uint32_t _graphicsQueueFamily; //family of that queue

    VkCommandPool _commandPool; //the command pool for our commands
    VkCommandBuffer _mainCommandBuffer; //the buffer we will record into

    VkRenderPass _renderPass;

    std::vector<VkFramebuffer> _framebuffers;


    VkPipelineLayout _trianglePipelineLayout;
    VkPipeline _trianglePipeline;
	VkPipeline _coloredTrianglePipeline;

	bool useColoredTrianglePipeline{false};

	DeletionQueue _mainDeletionQueue;

	VmaAllocator _allocator;

	VkPipeline _meshPipeline;
	Mesh _triangleMesh;

	VkPipelineLayout _meshPipelineLayout;

	Mesh _monkeyMesh;
	Mesh _wolfMesh;
	Mesh _maleHumanMesh;

	VkImageView _depthImageView;
	AllocatedImage _depthImage;

	//the format for the depth image
	VkFormat _depthFormat;

	//default array of renderable objects
	std::vector<RenderObject> _renderables;

	std::unordered_map<std::string,Material> _materials;
	std::unordered_map<std::string,Mesh> _meshes;

	glm::vec3 _camPos{0.0f, -6.f, -10.0f};

	// Double buffering
	static constexpr size_t FRAME_OVERLAP = 2;
	std::array<FrameData, FRAME_OVERLAP> _frames;

	VkDescriptorSetLayout _globalSetLayout;
	VkDescriptorSetLayout _objectSetLayout;
    VkDescriptorSetLayout _singleTextureSetLayout;
	VkDescriptorPool _descriptorPool;

	GPUSceneData _sceneParameters;
	AllocatedBuffer _sceneParameterBuffer;

	VkPhysicalDeviceProperties _gpuProperties;

    UploadContext _uploadContext;

    void immediate_submit(std::function<void(VkCommandBuffer cmd)>&& function);

    AllocatedBuffer create_buffer(size_t allocSize, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage);

    std::unordered_map<std::string, Texture> _loadedTextures;

    void load_images();

private:
	void init_vulkan();

    void init_swapchain();

    void init_commands();

    void init_default_renderpass();

    void init_framebuffers();

    void init_sync_structures();

	bool load_shader_module(const char* filePath, VkShaderModule* outShaderModule);

    void init_pipelines();

    void init_imgui();

	void load_meshes();

	void upload_mesh(Mesh& mesh);

	//create material and add it to the map
	Material* create_material(VkPipeline pipeline, VkPipelineLayout layout,const std::string& name);

	//returns nullptr if it can't be found
	Material* get_material(const std::string& name);

	//returns nullptr if it can't be found
	Mesh* get_mesh(const std::string& name);

	//our draw function
	void draw_objects(VkCommandBuffer cmd,RenderObject* first, int count);

	void init_scene();

	FrameData& get_current_frame();

	void init_descriptors();

	size_t pad_uniform_buffer_size(size_t originalSize);

    bool use_gpu_only_memory_for_mesh_buffers = true;
};
