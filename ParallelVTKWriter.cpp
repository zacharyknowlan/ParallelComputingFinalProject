#include <mpi.h>
#include <string>
#include "mfem.hpp"
#include "MicromorphicIntegrator.hpp"
#include "VTKWriter.hpp"
#include <chrono>
#include <iostream>

int main(int argc, char** argv) 
{
    // Initialize MPI environment and get rank variables
    MPI_Init(&argc, &argv);
    int WorldRank, WorldSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &WorldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &WorldSize);
    MPI_Request req_data, req_np, req_nc, req_size;

    // Create mesh and VTKwriter
    std::string MeshFile = "../UnitSquare.msh";
    auto mesh = mfem::Mesh(MeshFile.c_str(), 1, 1);
    auto writer = VTKWriter(mesh);
    
    // Misc. variables
    int dim = mesh.SpaceDimension();
    mfem::H1_FECollection *u_ec, *phi_ec;
    mfem::FiniteElementSpace *u_space, *phi_space;
    mfem::GridFunction u, phi;
    mfem::BlockVector x;
    int np, nc, size;

    // The 2nd to last rank needs to have a copy of u_space
    if (WorldSize >= 3  && WorldRank == (WorldSize-2))
    {
        u_ec = new mfem::H1_FECollection(3, dim, mfem::BasisType::GaussLobatto);
        u_space = new mfem::FiniteElementSpace(&mesh, u_ec, dim);
        u.SetSpace(u_space);
        MPI_Irecv(u.GetData(), u.Size(), MPI_DOUBLE, WorldRank+1, 1, MPI_COMM_WORLD, &req_data);
    }

    // Each rank other than rank 0 needs to recieve the precomputed data from writer
    if (WorldRank > 0)
    {
        MPI_Irecv(&np, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, &req_np);
        MPI_Irecv(&nc, 1, MPI_INT, 0, 3, MPI_COMM_WORLD, &req_nc);
        MPI_Irecv(&size, 1, MPI_INT, 0, 4, MPI_COMM_WORLD, &req_size);
    }
    else if (WorldRank == 0)
    {
        // Precompute np, nc, and size on rank 0
        writer.SetupWriteParameters();
        np = writer.GetNumPoints();
        nc = writer.GetNumElements();
        size = writer.GetElementDataSize();

        if (WorldSize > 1) // If not serial send to other ranks
        {    
            for (int ii=1; ii<WorldSize; ii++)
            {
                MPI_Send(&np, 1, MPI_INT, ii, 2, MPI_COMM_WORLD);
                MPI_Send(&nc, 1, MPI_INT, ii, 3, MPI_COMM_WORLD);
                MPI_Send(&size, 1, MPI_INT, ii, 4, MPI_COMM_WORLD);
            }
        }
    }

    // Create a mock solution on rank WorldSize-1
    // This is done because this rank always writes either the displacment or 
    // micro displacement gradient values and thus avoids storing aditional copies 
    // of the finite element spaces.
    if (WorldRank == (WorldSize-1))
    {
        u_ec = new mfem::H1_FECollection(3, dim, mfem::BasisType::GaussLobatto);
        u_space = new mfem::FiniteElementSpace(&mesh, u_ec, dim);

        phi_ec = new mfem::H1_FECollection(2, dim, mfem::BasisType::GaussLobatto);
        phi_space = new mfem::FiniteElementSpace(&mesh, phi_ec, dim*dim);

        mfem::Array<int> block_offsets(3); 
        block_offsets[0] = 0;
        block_offsets[1] = u_space->GetVSize();
        block_offsets[2] = phi_space->GetVSize();
        block_offsets.PartialSum();

        // Print total degrees of freedom
        std::cout << "Total DOF: " << block_offsets[2] << "\n";

        x.Update(block_offsets);
        x.GetBlock(0) = 4.; // Fake values for all displacement components
        x.GetBlock(1) = 8.; // Fake values for all micro displacement gradient values
        x.SyncFromBlocks();

        u.MakeRef(u_space, x.GetBlock(0), 0);
        phi.MakeRef(phi_space, x.GetBlock(1), 0);

        // Send u GridFunction data to the second to last rank if WorldSize >= 3
        if (WorldSize >= 3)
        {
            MPI_Send(u.GetData(), u.Size(), MPI_DOUBLE, WorldRank-1, 1, MPI_COMM_WORLD);
        }
    }

    // All ranks must wait for the solution to continue
    MPI_Barrier(MPI_COMM_WORLD);

    // Start file writing timer
    std::chrono::time_point<std::chrono::system_clock> start;
    if (WorldRank == 0) {start = std::chrono::system_clock::now();}

    bool RankWritePoints = false;
    bool RankWriteElements = false;
    bool RankWriteElementTypes = false;
    bool RankWriteElementMaterials = false;
    bool RankWriteField1 = false; // Field1 = displacements u
    bool RankWriteField2 = false; // Field2 = micro displacement gradient phi

    // Divide up the work depending on the number of ranks 
    writer.DetermineRankWork(WorldSize, WorldRank, RankWritePoints, RankWriteElements, 
                            RankWriteElementTypes, RankWriteElementMaterials, RankWriteField1,
                            RankWriteField2);
    if (WorldRank > 0) {writer.SetValuesForRank(np, nc, size);}

    // Get the data for the rank and format it into a string
    if (RankWritePoints) {writer.WriteHeader(); writer.WriteNodes();}
    if (RankWriteElements) {writer.WriteElements();}
    if (RankWriteElementTypes) {writer.WriteElementTypes();}
    if (RankWriteElementMaterials) {writer.WriteElementMaterials();}
    if (RankWriteField1) {writer.WriteVectorField(u, "u");}   
    if (RankWriteField2) {writer.WriteTensorField(phi, "phi");}
    std::string RankData = writer.GetData();
    int RankDataSize = static_cast<int>(RankData.size());

    // Get offset for this rank
    std::vector<int> WorldDataSizes(WorldSize);
    MPI_Allgather(&RankDataSize, 1, MPI_INT, WorldDataSizes.data(), 1, MPI_INT, MPI_COMM_WORLD);
    int offset = std::accumulate(WorldDataSizes.begin(), WorldDataSizes.begin() + WorldRank, 0);

    // Create the output file and write to it
    std::string filename = "../result" + std::to_string(WorldSize) + ".vtk";
    MPI_File file;
    MPI_File_open(MPI_COMM_WORLD, filename.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &file);
    MPI_File_write_at(file, offset, RankData.c_str(), RankDataSize, MPI_CHAR, MPI_STATUS_IGNORE);
    MPI_File_close(&file);
    MPI_Barrier(MPI_COMM_WORLD); // Make sure all ranks get here before ending timer

    if (WorldRank == 0) 
    {
        auto end = std::chrono::system_clock::now(); // End timer
        std::chrono::duration<double> elapsed_time = end - start;
        std::cout << "WorldSize: " << WorldSize << "\n";
        std::cout << "File Write Time: " << elapsed_time.count() << "\n";
    }

    if (WorldSize >= 3  && WorldRank >= (WorldSize-2))
    {
       delete u_ec, phi_ec;
       delete u_space, phi_space;
    }

    MPI_Finalize();

    return 0;
}