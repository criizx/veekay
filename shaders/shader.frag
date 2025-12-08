#version 450

layout (location = 0) in vec3 f_position;
layout (location = 1) in vec3 f_normal;
layout (location = 2) in vec2 f_uv;

layout (location = 0) out vec4 final_color;

layout (binding = 0, std140) uniform SceneUniforms {
    mat4 view_projection;
    vec3 camera_position;
    float _pad0;

    vec3 ambient_color;
    float ambient_intensity;

    vec3 directional_direction;
    float _pad1;
    vec3 directional_color;
    float directional_intensity;

    vec3 spotlight_position;
    float spotlight_inner_cutoff;
    vec3 spotlight_direction;
    float spotlight_outer_cutoff;
    vec3 spotlight_color;
    float spotlight_intensity;
};

layout (binding = 1, std140) uniform ModelUniforms {
    mat4 model;
    vec3 albedo_color;
    float _pad2;
};

void main() {
    vec3 normal = normalize(f_normal);
    vec3 view_dir = normalize(camera_position - f_position);

    vec3 ambient = ambient_color * ambient_intensity;

    vec3 light_dir = normalize(-directional_direction);
    float diff = max(dot(normal, light_dir), 0.0);
    vec3 directional = directional_color * directional_intensity * diff;

    vec3 halfway_dir = normalize(light_dir + view_dir);
    float spec = pow(max(dot(normal, halfway_dir), 0.0), 32.0);
    vec3 specular_directional = directional_color * spec * 0.5;

    vec3 spotlight_dir = normalize(spotlight_position - f_position);

    float theta = dot(spotlight_dir, normalize(-spotlight_direction));

    float epsilon = spotlight_inner_cutoff - spotlight_outer_cutoff;
    float spotlight_intensity_factor = clamp((theta - spotlight_outer_cutoff) / epsilon, 0.0, 1.0);

    float spotlight_diff = max(dot(normal, spotlight_dir), 0.0);

    float distance = length(spotlight_position - f_position);
    float attenuation = 1.0 / (distance * distance);

    vec3 spotlight = spotlight_color * spotlight_intensity * spotlight_diff *
                     spotlight_intensity_factor * attenuation;

    vec3 spotlight_halfway = normalize(spotlight_dir + view_dir);
    float spotlight_spec = pow(max(dot(normal, spotlight_halfway), 0.0), 32.0);
    vec3 specular_spotlight = spotlight_color * spotlight_spec *
                              spotlight_intensity_factor * attenuation * 0.5;

    vec3 result = (ambient + directional + specular_directional +
                   spotlight + specular_spotlight) * albedo_color;

    final_color = vec4(result, 1.0);
}