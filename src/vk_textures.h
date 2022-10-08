#pragma once

class VulkanEngine;
struct AllocatedImage;

namespace vkutil {

bool load_image_from_file(VulkanEngine& engine, const char* file, AllocatedImage& outImage);

}
