
file(GLOB sources "src/*.cpp" "include/rtxdi/*")

add_library(rtxdi-runtime STATIC EXCLUDE_FROM_ALL ${sources})
target_include_directories(rtxdi-runtime PUBLIC include)
set_target_properties(rtxdi-runtime PROPERTIES FOLDER "RTXDI SDK")

# Dependencies for the resampling compile tests
file(GLOB shader_dependencies "${CMAKE_CURRENT_SOURCE_DIR}/include/rtxdi/*")
