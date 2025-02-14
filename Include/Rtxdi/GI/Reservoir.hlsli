/***************************************************************************
 # Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/

#ifndef RTXDI_GI_RESERVOIR_HLSLI
#define RTXDI_GI_RESERVOIR_HLSLI

#include "Rtxdi/GI/ReSTIRGIParameters.h"
#include "Rtxdi/Utils/ReservoirAddressing.hlsli"

// Define this macro to 0 if your shader needs read-only access to the reservoirs, 
// to avoid compile errors in the RTXDI_StoreDIReservoir function
#ifndef RTXDI_ENABLE_STORE_RESERVOIR
#define RTXDI_ENABLE_STORE_RESERVOIR 1
#endif

#ifndef RTXDI_GI_RESERVOIR_BUFFER
#error "RTXDI_GI_RESERVOIR_BUFFER must be defined to point to a RWStructuredBuffer<RTXDI_PackedGIReservoir> type resource"
#endif

// This structure represents a indirect lighting reservoir that stores the radiance and weight
// as well as its the position where the radiane come from.
struct RTXDI_GIReservoir
{
    // postion of the 2nd bounce surface.
    float3 position;

    // normal vector of the 2nd bounce surface.
    float3 normal;

    // incoming radiance from the 2nd bounce surface.
    float3 radiance;

    // Overloaded: represents RIS weight sum during streaming,
    // then reservoir weight (inverse PDF) after FinalizeResampling
    float weightSum;

    // Number of samples considered for this reservoir
    uint M;

    // Number of frames the chosen sample has survived.
    uint age;
};

// Encoding helper constants for RTXDI_PackedGIReservoir
static const uint RTXDI_PackedGIReservoir_MShift = 0;
static const uint RTXDI_PackedGIReservoir_MaxM = 0x0ff;

static const uint RTXDI_PackedGIReservoir_AgeShift = 8;
static const uint RTXDI_PackedGIReservoir_MaxAge = 0x0ff;

// "misc data" only exists in the packed form of GI reservoir and stored into a gap field of the packed form.
// RTXDI SDK doesn't look into this field at all and when it stores a packed GI reservoir, the field is always filled with zero.
// Application can use this field to store anything.
static const uint RTXDI_PackedGIReservoir_MiscDataMask = 0xffff0000;

// Converts a GIReservoir into its packed form.
// This function should be used only when the application needs to store data with the given argument.
// It can be retrieved when unpacking the GIReservoir, but RTXDI SDK doesn't use the filed at all. 
RTXDI_PackedGIReservoir RTXDI_PackGIReservoir(const RTXDI_GIReservoir reservoir, const uint miscData)
{
    RTXDI_PackedGIReservoir data;

    data.position = reservoir.position;
    data.packed_normal = RTXDI_EncodeNormalizedVectorToSnorm2x16(reservoir.normal);

    data.packed_miscData_age_M =
        (miscData & RTXDI_PackedGIReservoir_MiscDataMask)
        | (min(reservoir.age, RTXDI_PackedGIReservoir_MaxAge) << RTXDI_PackedGIReservoir_AgeShift)
        | (min(reservoir.M, RTXDI_PackedGIReservoir_MaxM) << RTXDI_PackedGIReservoir_MShift);

    data.weight = reservoir.weightSum;
    data.packed_radiance = RTXDI_EncodeRGBToLogLuv(reservoir.radiance);
    data.unused = 0;

    return data;
}

// Converts a PackedGIReservoir into its unpacked form.
// This function should be used only when the application wants to retrieve the misc data stored in the gap field of the packed form.
RTXDI_GIReservoir RTXDI_UnpackGIReservoir(RTXDI_PackedGIReservoir data, out uint miscData)
{
    RTXDI_GIReservoir res;

    res.position = data.position;
    res.normal = RTXDI_DecodeNormalizedVectorFromSnorm2x16(data.packed_normal);

    res.radiance = RTXDI_DecodeLogLuvToRGB(data.packed_radiance);

    res.weightSum = data.weight;

    res.M = (data.packed_miscData_age_M >> RTXDI_PackedGIReservoir_MShift) & RTXDI_PackedGIReservoir_MaxM;

    res.age = (data.packed_miscData_age_M >> RTXDI_PackedGIReservoir_AgeShift) & RTXDI_PackedGIReservoir_MaxAge;

    miscData = data.packed_miscData_age_M & RTXDI_PackedGIReservoir_MiscDataMask;

    return res;
}

// Converts a PackedGIReservoir into its unpacked form.
RTXDI_GIReservoir RTXDI_UnpackGIReservoir(RTXDI_PackedGIReservoir data)
{
    uint miscFlags; // unused;
    return RTXDI_UnpackGIReservoir(data, miscFlags);
}

RTXDI_GIReservoir RTXDI_LoadGIReservoir(
    RTXDI_ReservoirBufferParameters reservoirParams,
    uint2 reservoirPosition,
    uint reservoirArrayIndex)
{
    uint pointer = RTXDI_ReservoirPositionToPointer(reservoirParams, reservoirPosition, reservoirArrayIndex);
    return RTXDI_UnpackGIReservoir(RTXDI_GI_RESERVOIR_BUFFER[pointer]);
}

RTXDI_GIReservoir RTXDI_LoadGIReservoir(
    RTXDI_ReservoirBufferParameters reservoirParams,
    uint2 reservoirPosition,
    uint reservoirArrayIndex,
    out uint miscFlags)
{
    uint pointer = RTXDI_ReservoirPositionToPointer(reservoirParams, reservoirPosition, reservoirArrayIndex);
    return RTXDI_UnpackGIReservoir(RTXDI_GI_RESERVOIR_BUFFER[pointer], miscFlags);
}

#if RTXDI_ENABLE_STORE_RESERVOIR

void RTXDI_StorePackedGIReservoir(
    const RTXDI_PackedGIReservoir packedGIReservoir,
    RTXDI_ReservoirBufferParameters reservoirParams,
    uint2 reservoirPosition,
    uint reservoirArrayIndex)
{
    uint pointer = RTXDI_ReservoirPositionToPointer(reservoirParams, reservoirPosition, reservoirArrayIndex);
    RTXDI_GI_RESERVOIR_BUFFER[pointer] = packedGIReservoir;
}

void RTXDI_StoreGIReservoir(
    const RTXDI_GIReservoir reservoir,
    RTXDI_ReservoirBufferParameters reservoirParams,
    uint2 reservoirPosition,
    uint reservoirArrayIndex)
{
    RTXDI_StorePackedGIReservoir(
        RTXDI_PackGIReservoir(reservoir, 0), reservoirParams, reservoirPosition, reservoirArrayIndex);
}

void RTXDI_StoreGIReservoir(
    const RTXDI_GIReservoir reservoir,
    const uint miscFlags,
    RTXDI_ReservoirBufferParameters reservoirParams,
    uint2 reservoirPosition,
    uint reservoirArrayIndex)
{
    RTXDI_StorePackedGIReservoir(
        RTXDI_PackGIReservoir(reservoir, miscFlags), reservoirParams, reservoirPosition, reservoirArrayIndex);
}

#endif // RTXDI_ENABLE_STORE_RESERVOIR

RTXDI_GIReservoir RTXDI_EmptyGIReservoir()
{
    RTXDI_GIReservoir s;

    s.position = float3(0.0, 0.0, 0.0);
    s.normal = float3(0.0, 0.0, 0.0);
    s.radiance = float3(0.0, 0.0, 0.0);
    s.weightSum = 0.0;
    s.M = 0;
    s.age = 0;

    return s;
}

// Creates a GI reservoir from a raw light sample.
// Note: the original sample PDF can be embedded into sampleRadiance, in which case the samplePdf parameter should be set to 1.0.
RTXDI_GIReservoir RTXDI_MakeGIReservoir(
    const float3 samplePos,
    const float3 sampleNormal,
    const float3 sampleRadiance,
    const float samplePdf)
{
    RTXDI_GIReservoir reservoir;
    reservoir.position = samplePos;
    reservoir.normal = sampleNormal;
    reservoir.radiance = sampleRadiance;
    reservoir.weightSum = samplePdf > 0.0 ? 1.0 / samplePdf : 0.0;
    reservoir.M = 1;
    reservoir.age = 0;
    return reservoir;
}

// Adds `newReservoir` into `reservoir`, returns true if the new reservoir's sample was selected.
// This function assumes the newReservoir has been normalized, so its weightSum means "1/g * 1/M * \sum{g/p}"
// and the targetPdf is a conversion factor from the newReservoir's space to the reservoir's space (integrand).
bool RTXDI_CombineGIReservoirs(
    inout RTXDI_GIReservoir reservoir,
    const RTXDI_GIReservoir newReservoir,
    float random,
    float targetPdf)
{
    // What's the current weight (times any prior-step RIS normalization factor)
    const float risWeight = targetPdf * newReservoir.weightSum * newReservoir.M;

    // Our *effective* candidate pool is the sum of our candidates plus those of our neighbors
    reservoir.M += newReservoir.M;

    // Update the weight sum
    reservoir.weightSum += risWeight;

    // Decide if we will randomly pick this sample
    bool selectSample = (random * reservoir.weightSum <= risWeight);

    if (selectSample)
    {
        reservoir.position = newReservoir.position;
        reservoir.normal = newReservoir.normal;
        reservoir.radiance = newReservoir.radiance;
        reservoir.age = newReservoir.age;
    }

    return selectSample;
}

// Performs normalization of the reservoir after streaming.
void RTXDI_FinalizeGIResampling(
    inout RTXDI_GIReservoir reservoir,
    float normalizationNumerator,
    float normalizationDenominator)
{
    reservoir.weightSum = (normalizationDenominator == 0.0) ? 0.0 : (reservoir.weightSum * normalizationNumerator) / normalizationDenominator;
}

bool RTXDI_IsValidGIReservoir(const RTXDI_GIReservoir reservoir)
{
    return reservoir.M != 0;
}

#endif // RTXDI_GI_RESERVOIR_HLSLI
