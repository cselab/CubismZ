//
//  HDF5SliceDumper.h
//  Cubism
//
//  Created by Fabian Wermelinger 09/27/2016
//  Copyright 2016 ETH Zurich. All rights reserved.
//
#ifndef HDF5SLICEDUMPER_H_QI4Y9HO7
#define HDF5SLICEDUMPER_H_QI4Y9HO7

#include <cassert>
#include <iostream>
#include <vector>
#include <sstream>

#ifdef _USE_HDF_
#include <hdf5.h>
#endif

#ifdef _FLOAT_PRECISION_
typedef float Real;
#else
typedef double Real;
#endif

#ifdef _FLOAT_PRECISION_
#define HDF_REAL H5T_NATIVE_FLOAT
#else
#define HDF_REAL H5T_NATIVE_DOUBLE
#endif

#include "BlockInfo.h"
#include "ArgumentParser.h"

template <typename TGrid>
struct Slice
{
    typedef TGrid GridType;

    int id;
    int dir;
    int idx;
    int width, height;
    bool valid;
    Slice() : id(-1), dir(-1), idx(-1), width(0), height(0), valid(false) {}

    template <typename TSlice>
    static std::vector<TSlice> getSlices(ArgumentParser& parser, const TGrid& grid)
    {
        typedef typename TGrid::BlockType B;
        int Dim[3];
        Dim[0] = grid.getBlocksPerDimension(0)*B::sizeX;
        Dim[1] = grid.getBlocksPerDimension(1)*B::sizeY;
        Dim[2] = grid.getBlocksPerDimension(2)*B::sizeZ;

        std::vector<TSlice> slices(0);
        const size_t nSlices = parser("nslices").asInt(0);
        for (size_t i = 0; i < nSlices; ++i)
        {
            TSlice thisOne;
            thisOne.id = i+1;

            std::ostringstream identifier;
            identifier << "slice" << i+1;
            // fetch direction
            const std::string sDir = identifier.str() + "_direction";
            if (parser.check(sDir)) thisOne.dir = parser(sDir).asInt(0);
            const bool bDirOK = (thisOne.dir >= 0 && thisOne.dir < 3);
            assert(bDirOK);

            // compute index
            const std::string sIndex = identifier.str() + "_index";
            const std::string sFrac  = identifier.str() + "_fraction";
            if (parser.check(sIndex)) thisOne.idx = parser(sIndex).asInt(0);
            else if (parser.check(sFrac))
            {
                const double fraction = parser(sFrac).asDouble(0.0);
                const int idx = static_cast<int>(Dim[thisOne.dir] * fraction);
                thisOne.idx = (fraction == 1.0) ? Dim[thisOne.dir]-1 : idx;
            }
            const bool bIdxOK = (thisOne.idx >= 0 && thisOne.idx < Dim[thisOne.dir]);
            assert(bIdxOK);

            if (bDirOK && bIdxOK) thisOne.valid = true;
            else
            {
                std::cerr << "Slice: WARNING: Ill defined slice \"" << identifier.str() << "\"... Skipping this one" << std::endl;
                thisOne.valid = false;
                slices.push_back(thisOne);
                continue;
            }

            // define slice layout
            if (thisOne.dir == 0)
            {
                thisOne.width  = Dim[2];
                thisOne.height = Dim[1];
            }
            else if (thisOne.dir == 1)
            {
                thisOne.width  = Dim[2];
                thisOne.height = Dim[0];
            }
            else if (thisOne.dir == 2)
            {
                thisOne.width  = Dim[0];
                thisOne.height = Dim[1];
            }
            slices.push_back(thisOne);
        }
        return slices;
    }
};


template <typename TSlice>
class SliceProcessor
{
private:
    const bool m_verbose;
    void (*m_writer)(const TSlice&, const typename TSlice::GridType&, const int, const Real, const std::string, const std::string);
    std::vector<TSlice> m_slices;

    inline void _process(const TSlice& slice, const typename TSlice::GridType& grid, const int stepID, const Real t, const std::string fname, const std::string path)
    {
        if (m_writer && slice.valid)
            m_writer(slice, grid, stepID, t, fname, path);
        if (m_verbose && !m_writer)
            std::cerr << "SliceProcessor: WARNING: No functor defined... Skipping slice" << slice.id << std::endl;
    }

public:
    SliceProcessor(ArgumentParser& parser, const typename TSlice::GridType& grid, const bool verbose=true, void (*f)(const TSlice&, const typename TSlice::GridType&, const int, const Real, const std::string, const std::string) = NULL) :
        m_verbose(verbose), m_writer(f)
    {
        m_slices = TSlice::template getSlices<TSlice>(parser, grid);
    }
    ~SliceProcessor() {}

    inline void setFunctor(void (*f)(const TSlice&, const typename TSlice::GridType&, const int, const Real, const std::string, const std::string))
    {
        m_writer = f;
    }

    inline void process(const int sliceID, const typename TSlice::GridType& grid, const int stepID, const Real t, const std::string fname, const std::string dpath=".")
    {
        // TODO: (fabianw@mavt.ethz.ch; Thu 29 Sep 2016 02:59:54 PM CEST)
        // WARNING: sliceID starts from zero, eventhough slice numbering in the
        // .conf file starts from 1 (to be consistent with earlier
        // implementation of sensors)
        if (sliceID >= 0 && sliceID < static_cast<int>(m_slices.size()))
            _process(m_slices[sliceID], grid, stepID, t, fname, dpath);
        else
            if (m_verbose)
                std::cerr << "SliceProcessor: WARNING: Requesting undefined slice... Skipping slice" << sliceID << std::endl;
    }

    inline void process_all(const typename TSlice::GridType& grid, const int stepID, const Real t, const std::string fname, const std::string dpath=".")
    {
        for (size_t i = 0; i < m_slices.size(); ++i)
            _process(m_slices[i], grid, stepID, t, fname, dpath);
    }

    void showSlices()
    {
        if (m_verbose)
        {
            std::cout << "Got n = " << m_slices.size() << " slices:" << std::endl;
            for (size_t i = 0; i < m_slices.size(); ++i)
            {
                std::cout << "Slice ID = " << m_slices[i].id << std::endl;
                std::cout << "\tSlice index:     " << m_slices[i].idx << std::endl;
                std::cout << "\tSlice direction: " << m_slices[i].dir << std::endl;
                std::cout << "\tSlice dimension: (" << m_slices[i].width << ", " << m_slices[i].height << ")" << std::endl;
                std::cout << "\tSlice valid:     " << m_slices[i].valid << std::endl;
                std::cout << std::endl;
            }
        }
    }
};


namespace SliceExtractor
{
    template <typename TBlock, typename TStreamer>
    void YZ(const int ix, const int width, std::vector<BlockInfo>& bInfo, Real * const data)
    {
        const unsigned int NCHANNELS = TStreamer::NCHANNELS;

#pragma omp parallel for
        for(int i = 0; i < (int)bInfo.size(); ++i)
        {
            BlockInfo& info = bInfo[i];
            const unsigned int idx[3] = {info.index[0], info.index[1], info.index[2]};
            TBlock& b = *(TBlock*)info.ptrBlock;
            TStreamer streamer(b);

            for(unsigned int iz=0; iz<TBlock::sizeZ; ++iz)
                for(unsigned int iy=0; iy<TBlock::sizeY; ++iy)
                {
                    Real output[NCHANNELS];
                    for(unsigned int k=0; k<NCHANNELS; ++k)
                        output[k] = 0;

                    streamer.operate(ix, iy, iz, (Real*)output);

                    const unsigned int gy = idx[1]*TBlock::sizeY + iy;
                    const unsigned int gz = idx[2]*TBlock::sizeZ + iz;

                    Real * const ptr = data + NCHANNELS*(gz + width * gy);

                    for(unsigned int k=0; k<NCHANNELS; ++k)
                        ptr[k] = output[k];
                }
        }
    }

    template <typename TBlock, typename TStreamer>
    void XZ(const int iy, const int width, std::vector<BlockInfo>& bInfo, Real * const data)
    {
        const unsigned int NCHANNELS = TStreamer::NCHANNELS;

#pragma omp parallel for
        for(int i = 0; i < (int)bInfo.size(); ++i)
        {
            BlockInfo& info = bInfo[i];
            const unsigned int idx[3] = {info.index[0], info.index[1], info.index[2]};
            TBlock& b = *(TBlock*)info.ptrBlock;
            TStreamer streamer(b);

            for(unsigned int iz=0; iz<TBlock::sizeZ; ++iz)
                for(unsigned int ix=0; ix<TBlock::sizeX; ++ix)
                {
                    Real output[NCHANNELS];
                    for(unsigned int k=0; k<NCHANNELS; ++k)
                        output[k] = 0;

                    streamer.operate(ix, iy, iz, (Real*)output);

                    const unsigned int gx = idx[0]*TBlock::sizeX + ix;
                    const unsigned int gz = idx[2]*TBlock::sizeZ + iz;

                    Real * const ptr = data + NCHANNELS*(gz + width * gx);

                    for(unsigned int k=0; k<NCHANNELS; ++k)
                        ptr[k] = output[k];
                }
        }
    }

    template <typename TBlock, typename TStreamer>
    void YX(const int iz, const int width, std::vector<BlockInfo>& bInfo, Real * const data)
    {
        const unsigned int NCHANNELS = TStreamer::NCHANNELS;

#pragma omp parallel for
        for(int i = 0; i < (int)bInfo.size(); ++i)
        {
            BlockInfo& info = bInfo[i];
            const unsigned int idx[3] = {info.index[0], info.index[1], info.index[2]};
            TBlock& b = *(TBlock*)info.ptrBlock;
            TStreamer streamer(b);

            for(unsigned int iy=0; iy<TBlock::sizeY; ++iy)
                for(unsigned int ix=0; ix<TBlock::sizeX; ++ix)
                {
                    Real output[NCHANNELS];
                    for(unsigned int k=0; k<NCHANNELS; ++k)
                        output[k] = 0;

                    streamer.operate(ix, iy, iz, (Real*)output);

                    const unsigned int gx = idx[0]*TBlock::sizeX + ix;
                    const unsigned int gy = idx[1]*TBlock::sizeY + iy;

                    Real * const ptr = data + NCHANNELS*(gx + width * gy);

                    for(unsigned int k=0; k<NCHANNELS; ++k)
                        ptr[k] = output[k];
                }
        }
    }
}

template<typename TSlice, typename TStreamer>
void DumpSliceHDF5(const TSlice& slice, const typename TSlice::GridType& grid, const int stepID, const Real t, const std::string fname, const std::string dpath=".")
{
#ifdef _USE_HDF_
    typedef typename TSlice::GridType::BlockType B;

    static const unsigned int NCHANNELS = TStreamer::NCHANNELS;
    const unsigned int width = slice.width;
    const unsigned int height = slice.height;

    std::cout << "Writing HDF5 Slice file\n";
    std::cout << "Allocating " << (width * height * NCHANNELS)/(1024.*1024.) << "MB of HDF5 slice data" << std::endl;;

    Real * array_all = new Real[width * height * NCHANNELS];

    std::vector<BlockInfo> bInfo_local = grid.getBlocksInfo();
    std::vector<BlockInfo> bInfo_slice;
    for (size_t i = 0; i < bInfo_local.size(); ++i)
    {
        const int start = bInfo_local[i].index[slice.dir] * _BLOCKSIZE_;
        if (start <= slice.idx && slice.idx < (start+_BLOCKSIZE_))
            bInfo_slice.push_back(bInfo_local[i]);
    }

    ostringstream filename;
    filename << dpath << "/" << fname << "_slice" << slice.id;

    herr_t status;
    hid_t file_id, dataset_id, fspace_id, fapl_id, mspace_id;

    hsize_t count[3] = {height, width, NCHANNELS};
    hsize_t dims[3] = {height, width, NCHANNELS};
    hsize_t offset[3] = {0, 0, 0};

    H5open();
    fapl_id = H5Pcreate(H5P_FILE_ACCESS);
    file_id = H5Fcreate((filename.str()+".h5").c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, fapl_id);
    status = H5Pclose(fapl_id);

    if (0 == slice.dir)
        SliceExtractor::YZ<B,TStreamer>(slice.idx%_BLOCKSIZE_, width, bInfo_slice, array_all);
    else if (1 == slice.dir)
        SliceExtractor::XZ<B,TStreamer>(slice.idx%_BLOCKSIZE_, width, bInfo_slice, array_all);
    else if (2 == slice.dir)
        SliceExtractor::YX<B,TStreamer>(slice.idx%_BLOCKSIZE_, width, bInfo_slice, array_all);

    fapl_id = H5Pcreate(H5P_DATASET_XFER);
    fspace_id = H5Screate_simple(3, dims, NULL);
#ifndef _ON_FERMI_
    dataset_id = H5Dcreate(file_id, "data", HDF_REAL, fspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
#else
    dataset_id = H5Dcreate2(file_id, "data", HDF_REAL, fspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
#endif

    fspace_id = H5Dget_space(dataset_id);
    H5Sselect_hyperslab(fspace_id, H5S_SELECT_SET, offset, NULL, count, NULL);
    mspace_id = H5Screate_simple(3, count, NULL);
    status = H5Dwrite(dataset_id, HDF_REAL, mspace_id, fspace_id, fapl_id, array_all);

    status = H5Sclose(mspace_id);
    status = H5Sclose(fspace_id);
    status = H5Dclose(dataset_id);
    status = H5Pclose(fapl_id);
    status = H5Fclose(file_id);
    H5close();

    delete [] array_all;

    // writing xmf wrapper
    {
        FILE *xmf = 0;
        xmf = fopen((filename.str()+".xmf").c_str(), "w");
        fprintf(xmf, "<?xml version=\"1.0\" ?>\n");
        fprintf(xmf, "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n");
        fprintf(xmf, "<Xdmf Version=\"2.0\">\n");
        fprintf(xmf, " <Domain>\n");
        fprintf(xmf, "   <Grid GridType=\"Uniform\">\n");
        fprintf(xmf, "     <Time Value=\"%e\"/>\n", t);
        fprintf(xmf, "     <Topology TopologyType=\"3DCoRectMesh\" Dimensions=\"1 %d %d\"/>\n", height, width);
        fprintf(xmf, "     <Geometry GeometryType=\"ORIGIN_DXDYDZ\">\n");
        fprintf(xmf, "       <DataItem Name=\"Origin\" Dimensions=\"3\" NumberType=\"Float\" Precision=\"4\" Format=\"XML\">\n");
        fprintf(xmf, "        %e %e %e\n", 0., 0., 0.);
        fprintf(xmf, "       </DataItem>\n");
        fprintf(xmf, "       <DataItem Name=\"Spacing\" Dimensions=\"3\" NumberType=\"Float\" Precision=\"4\" Format=\"XML\">\n");
        fprintf(xmf, "        %e %e %e\n", 1.,1.,1.);
        fprintf(xmf, "       </DataItem>\n");
        fprintf(xmf, "     </Geometry>\n");
        fprintf(xmf, "     <Attribute Name=\"data\" AttributeType=\"%s\" Center=\"Node\">\n", TStreamer::getAttributeName());
        fprintf(xmf, "       <DataItem Dimensions=\"1 %d %d %d\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">\n", height, width, NCHANNELS);
        fprintf(xmf, "        %s:/data\n",(filename.str()+".h5").c_str());
        fprintf(xmf, "       </DataItem>\n");
        fprintf(xmf, "     </Attribute>\n");
        fprintf(xmf, "   </Grid>\n");
        fprintf(xmf, " </Domain>\n");
        fprintf(xmf, "</Xdmf>\n");
        fclose(xmf);
    }
#else
#warning USE OF HDF WAS DISABLED AT COMPILE TIME
#endif
}

#endif /* HDF5SLICEDUMPER_H_QI4Y9HO7 */
