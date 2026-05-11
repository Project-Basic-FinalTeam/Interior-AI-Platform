#pragma once  // 중복 포함 방지 (필수!)
#include <entt/entt.hpp>
#include <string>

struct Transform {
    float x, y, z;
};

struct AssetInfo {
    std::string id;
    std::string type;
};

class InteriorRegistry {
public:
    entt::registry registry;

    entt::entity AddFurniture(std::string id, float x, float y, float z) {
        auto entity = registry.create();
        registry.emplace<Transform>(entity, x, y, z);
        registry.emplace<AssetInfo>(entity, id, "generated");
        return entity;
    }
};