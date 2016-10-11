/*
 *  WaveletSerializationTypes.h
 *  
 *
 *  Created by Diego Rossinelli on 3/27/13.
 *  Extended by Panos Hadjidoukas.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#ifndef _WAVELETSERIALIZATIONTYPES_H_
#define _WAVELETSERIALIZATIONTYPES_H_ 1

#pragma once

struct BlockMetadata { int idcompression, subid, ix, iy, iz; }  __attribute__((packed));
struct HeaderLUT { size_t aggregate_bytes; int nchunks; }  __attribute__((packed));
struct CompressedBlock{ size_t start, extent; int subid; }  __attribute__((packed));

#endif
