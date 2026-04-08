// Test BEVPoolV2 Plugin Registration
// Based on Phase 2 sparse_log2_quant test

#include <iostream>
#include <cstring>
#include "NvInfer.h"
#include "NvInferPlugin.h"

// Explicit init function from the plugin library
extern "C" void forceInitBEVPoolV2Plugin();

int main() {
    std::cout << "=== BEVPoolV2 Plugin Registration Test ===" << std::endl;

    // Initialize plugin library
    forceInitBEVPoolV2Plugin();
    std::cout << "[OK] forceInitBEVPoolV2Plugin() called" << std::endl;

    // Get plugin registry
    auto registry = getPluginRegistry();
    if (!registry) {
        std::cerr << "[FAIL] Failed to get plugin registry" << std::endl;
        return 1;
    }
    std::cout << "[OK] Plugin registry obtained" << std::endl;

    // Find BEVPoolV2 plugin
    const char* pluginName = "BEVPoolV2";
    const char* pluginVersion = "1";
    
    auto creator = registry->getPluginCreator(pluginName, pluginVersion);
    if (!creator) {
        std::cerr << "[FAIL] Plugin '" << pluginName << "' version '" << pluginVersion 
                  << "' not found in registry" << std::endl;
        return 1;
    }
    std::cout << "[OK] Plugin '" << pluginName << "' version '" << pluginVersion 
              << "' found in registry" << std::endl;

    // Check plugin fields
    auto fieldNames = creator->getFieldNames();
    if (!fieldNames) {
        std::cerr << "[FAIL] Failed to get field names" << std::endl;
        return 1;
    }
    
    std::cout << "[OK] Plugin fields (" << fieldNames->nbFields << "):";
    for (int i = 0; i < fieldNames->nbFields; ++i) {
        std::cout << " " << fieldNames->fields[i].name;
    }
    std::cout << std::endl;

    // Create plugin instance with valid data
    int B = 1, D = 118, H = 128, W = 128;
    nvinfer1::PluginField fields[] = {
        {"B", &B, nvinfer1::PluginFieldType::kINT32, 1},
        {"D", &D, nvinfer1::PluginFieldType::kINT32, 1},
        {"H", &H, nvinfer1::PluginFieldType::kINT32, 1},
        {"W", &W, nvinfer1::PluginFieldType::kINT32, 1},
    };
    nvinfer1::PluginFieldCollection fc;
    fc.nbFields = 4;
    fc.fields = fields;
    
    auto plugin = creator->createPlugin("bev_pool_test", &fc);
    if (!plugin) {
        std::cerr << "[FAIL] Failed to create plugin instance" << std::endl;
        return 1;
    }
    std::cout << "[OK] Plugin instance created successfully" << std::endl;

    // Test serialization
    size_t serialSize = plugin->getSerializationSize();
    std::cout << "[OK] Serialization size: " << serialSize << " bytes" << std::endl;
    
    void* serialData = malloc(serialSize);
    plugin->serialize(serialData);
    std::cout << "[OK] Plugin serialized successfully" << std::endl;

    // Test deserialization
    auto plugin2 = creator->deserializePlugin("bev_pool_test2", serialData, serialSize);
    if (!plugin2) {
        std::cerr << "[FAIL] Failed to deserialize plugin" << std::endl;
        free(serialData);
        return 1;
    }
    std::cout << "[OK] Plugin deserialized successfully" << std::endl;

    // Cleanup
    free(serialData);
    plugin->destroy();
    plugin2->destroy();

    std::cout << "\n[SUCCESS] All tests passed!" << std::endl;
    return 0;
}
