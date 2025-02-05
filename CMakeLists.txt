
set(sources
	"Include/Rtxdi/ImportanceSamplingContext.h"
	"Include/Rtxdi/RtxdiParameters.h"
	"Include/Rtxdi/RtxdiTypes.h"
	"Include/Rtxdi/RtxdiUtils.h"
	"Include/Rtxdi/DI/BoilingFilter.hlsli"
	"Include/Rtxdi/DI/InitialSampling.hlsli"
	"Include/Rtxdi/DI/PairwiseStreaming.hlsli"
	"Include/Rtxdi/DI/Reservoir.hlsli"
	"Include/Rtxdi/DI/ReSTIRDI.h"
	"Include/Rtxdi/DI/ReSTIRDIParameters.h"
	"Include/Rtxdi/DI/SpatialResampling.hlsli"
	"Include/Rtxdi/DI/SpatioTemporalResampling.hlsli"
	"Include/Rtxdi/DI/TemporalResampling.hlsli"
	"Include/Rtxdi/GI/BoilingFilter.hlsli"
	"Include/Rtxdi/GI/JacobianMath.hlsli"
	"Include/Rtxdi/GI/Reservoir.hlsli"
	"Include/Rtxdi/GI/ReSTIRGI.h"
	"Include/Rtxdi/GI/ReSTIRGIParameters.h"
	"Include/Rtxdi/GI/SpatialResampling.hlsli"
	"Include/Rtxdi/GI/SpatioTemporalResampling.hlsli"
	"Include/Rtxdi/GI/TemporalResampling.hlsli"
	"Include/Rtxdi/LightSampling/LocalLightSelection.hlsli"
	"Include/Rtxdi/LightSampling/PresamplingFunctions.hlsli"
	"Include/Rtxdi/LightSampling/RISBuffer.hlsli"
	"Include/Rtxdi/LightSampling/RISBufferSegmentAllocator.h"
	"Include/Rtxdi/LightSampling/RISBufferSegmentParameters.h"
	"Include/Rtxdi/LightSampling/UniformSampling.hlsli"
	"Include/Rtxdi/ReGIR/ReGIR.h"
	"Include/Rtxdi/ReGIR/ReGIRParameters.h"
	"Include/Rtxdi/ReGIR/ReGIRSampling.hlsli"
	"Include/Rtxdi/Utils/BoilingFilter.hlsli"
	"Include/Rtxdi/Utils/Checkerboard.hlsli"
	"Include/Rtxdi/Utils/Math.hlsli"
	"Include/Rtxdi/Utils/ReservoirAddressing.hlsli"
	"Source/ImportanceSamplingContext.cpp"
	"Source/ReGIR.cpp"
	"Source/ReSTIRDI.cpp"
	"Source/ReSTIRGI.cpp"
	"Source/RISBufferSegmentAllocator.cpp"
	"Source/RtxdiUtils.cpp")

# Organize MSVS filters (the little folders in the solution explorer) to match the folder structure
foreach(source IN LISTS sources)
    get_filename_component(source_path "${source}" PATH)
    string(REPLACE "/" "\\" source_path_msvc "${source_path}")
    source_group("${source_path_msvc}" FILES "${source}")
endforeach()

add_library(Rtxdi STATIC EXCLUDE_FROM_ALL ${sources})
target_include_directories(Rtxdi PUBLIC "include")
set_target_properties(Rtxdi PROPERTIES FOLDER "RTXDI SDK")

# Dependencies for the resampling compile tests
file(GLOB shader_dependencies "${CMAKE_CURRENT_SOURCE_DIR}/include/Rtxdi/*")
