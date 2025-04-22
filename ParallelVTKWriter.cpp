#include <mpi.h>
#include <vector>
#include <iostream>
#include <sstream>
#include <string>
#include <numeric>
#include "mfem.hpp"
#include "MicromorphicIntegrator.hpp"

int main(int argc, char** argv) 
{
    MPI_Init(&argc, &argv);

    int WorldRank, WorldSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &WorldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &WorldSize);
    MPI_Request req_data, req_np, req_nc, req_size;

    // Creating mesh and finite element spaces on each rank is very inefficient 
    // but is the only option within the time horizon of this project
    std::string MeshFile = "../UnitSquare.msh";
    auto mesh = mfem::Mesh(MeshFile.c_str(), 1, 1);
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

    // Each rank less than than rank WorldSize-2 recieves the data
    if (WorldRank < (WorldSize-2))
    {
        MPI_Irecv(&np, 1, MPI_INT, (WorldSize-2), (2+WorldRank), MPI_COMM_WORLD, &req_np);
        MPI_Irecv(&nc, 1, MPI_INT, (WorldSize-2), (2+WorldRank*WorldSize), MPI_COMM_WORLD, &req_nc);
        MPI_Irecv(&size, 1, MPI_INT, (WorldSize-2), (2+2*WorldRank*WorldSize), MPI_COMM_WORLD, &req_size);
    }
    else if (WorldSize == 1 || (WorldSize == 2 && WorldRank == (WorldSize-1)) || WorldRank == (WorldSize-2))
    {
        // Precompute np, nc, and size on Rank WorldSize-2
        mfem::RefinedGeometry *RefG;
        np = nc = size = 0;
        for (int i = 0; i < mesh.GetNE(); i++)
        {
            mfem::Geometry::Type geom = mesh.GetElementBaseGeometry(i);
            int nv = mfem::Geometries.GetVertices(geom)->GetNPoints();
            RefG = mfem::GlobGeometryRefiner.Refine(geom, 0, 1);
            np += RefG->RefPts.GetNPoints();
            nc += RefG->RefGeoms.Size() / nv;
            size += (RefG->RefGeoms.Size() / nv) * (nv + 1);
        }
    }
    if (WorldRank < (WorldSize-2))
    {
        // Send the compute values for np, nc, and size to the relevant ranks
        for (int ii=0; ii<WorldRank; ii++)
        {
            MPI_Send(&np, 1, MPI_INT, ii, (2+ii), MPI_COMM_WORLD);
            MPI_Send(&nc, 1, MPI_INT, ii, (2+ii*WorldSize), MPI_COMM_WORLD);
            MPI_Send(&size, 1, MPI_INT, ii, (2+2*ii*WorldSize), MPI_COMM_WORLD);
        }
    }

    // The last rank solves the problem
    if (WorldRank == (WorldSize-1))
    {
        double Mu = 454545.45454545453;
        double Lambda = 113636.36363636363;
        double ls = 0.1;

        u_ec = new mfem::H1_FECollection(3, dim, mfem::BasisType::GaussLobatto);
        u_space = new mfem::FiniteElementSpace(&mesh, u_ec, dim);

        phi_ec = new mfem::H1_FECollection(2, dim, mfem::BasisType::GaussLobatto);
        phi_space = new mfem::FiniteElementSpace(&mesh, phi_ec, dim*dim);

        auto MuAlpha = mfem::ConstantCoefficient(2.*Mu);
        auto LambdaAlpha = mfem::ConstantCoefficient(2.*Lambda);
        auto MuBeta = mfem::ConstantCoefficient(Mu);
        auto LambdaBeta = mfem::ConstantCoefficient(Lambda);
        auto MuGamma = mfem::ConstantCoefficient(Mu);
        auto LambdaGamma = mfem::ConstantCoefficient(Lambda);
        auto MuA = mfem::ConstantCoefficient(ls*ls*Mu);
        auto LambdaA = mfem::ConstantCoefficient(ls*ls*Lambda);

        mfem::Array<int> block_offsets(3); 
        block_offsets[0] = 0;
        block_offsets[1] = u_space->GetVSize();
        block_offsets[2] = phi_space->GetVSize();
        block_offsets.PartialSum();

        x.Update(block_offsets);
        x.GetBlock(0) = 0.;
        x.GetBlock(1) = 0.;
        x.SyncFromBlocks();

        auto b = mfem::BlockVector(block_offsets);
        b.GetBlock(0) = 0.;
        b.GetBlock(1) = 0.;
        b.SyncFromBlocks();
        
        auto negative_x = mfem::BlockVector(block_offsets);
        negative_x.GetBlock(0) = 0.;
        negative_x.GetBlock(1) = 0.;
        negative_x.SyncFromBlocks();

        mfem::Array<int> uBCDOFs, phiBCDOFs, phi0BCDOFs;
        mfem::Array<int> LeftEdge({0, 0, 0, 1, 0});

        phi_space->GetEssentialTrueDofs(LeftEdge, phi0BCDOFs, 0);
        for (int ii=0; ii<phi0BCDOFs.Size(); ii++)
        {
            x.GetBlock(1)[phi0BCDOFs[ii]] = 0.001;
            negative_x.GetBlock(1)[phi0BCDOFs[ii]] = -0.001;
        }
        x.SyncFromBlocks();
        negative_x.SyncFromBlocks();

        u_space->GetEssentialTrueDofs(LeftEdge, uBCDOFs);
        phi_space->GetEssentialTrueDofs(LeftEdge, phiBCDOFs);

        mfem::Array<int> GlobalBCDOFs(uBCDOFs.Size() + phiBCDOFs.Size());
        int offset = uBCDOFs.Size();
        int u_size = u_space->GetVSize();
        for (int ii=0; ii<uBCDOFs.Size(); ii++) {GlobalBCDOFs[ii] = uBCDOFs[ii];}
        for (int ii=0; ii<phiBCDOFs.Size(); ii++) {GlobalBCDOFs[ii + offset] = u_size + phiBCDOFs[ii];}

        auto xForceComponent = mfem::Vector({0., 1e3, 0., 0., 0.});
        auto BoundaryForce = mfem::VectorArrayCoefficient(dim);
        BoundaryForce.Set(0, new mfem::PWConstCoefficient(xForceComponent));
        BoundaryForce.Set(1, new mfem::ConstantCoefficient(0.));

        auto b1 = mfem::LinearForm(u_space);
        b1.MakeRef(u_space, b.GetBlock(0), 0);
        b1.AddBoundaryIntegrator(new mfem::VectorBoundaryLFIntegrator(BoundaryForce));
        b1.Assemble();
        b1.SyncAliasMemory(b);

        auto a1 = mfem::BilinearForm(u_space);
        a1.AddDomainIntegrator(new MicromorphicIntegrator(&LambdaAlpha, &MuAlpha, Block::TopLeft)); 
        a1.SetAssemblyLevel(mfem::AssemblyLevel::PARTIAL);
        a1.Assemble();

        auto a2 = mfem::MixedBilinearForm(phi_space, u_space);
        a2.AddDomainIntegrator(new MicromorphicIntegrator(&LambdaBeta, &MuBeta, Block::TopRight)); 
        a2.SetAssemblyLevel(mfem::AssemblyLevel::PARTIAL);
        a2.Assemble();

        auto a3 = mfem::MixedBilinearForm(u_space, phi_space);
        a3.AddDomainIntegrator(new MicromorphicIntegrator(&LambdaBeta, &MuBeta, Block::BottomLeft));
        a3.SetAssemblyLevel(mfem::AssemblyLevel::PARTIAL);
        a3.Assemble();

        auto a4 = mfem::BilinearForm(phi_space);
        a4.AddDomainIntegrator(new MicromorphicIntegrator(&LambdaGamma, &MuGamma, &LambdaA, &MuA, Block::BottomRight));
        a4.SetAssemblyLevel(mfem::AssemblyLevel::PARTIAL);
        a4.Assemble();

        auto A = mfem::BlockOperator(block_offsets);
        A.SetBlock(0, 0, &a1);
        A.SetBlock(0, 1, &a2);
        A.SetBlock(1, 0, &a3);
        A.SetBlock(1, 1, &a4);

        A.AddMult(negative_x, b); 

        for (int ii=0; ii<uBCDOFs.Size(); ii++) {b.GetBlock(0)[uBCDOFs[ii]] = 0.;}
        for (int ii=0; ii<phiBCDOFs.Size(); ii++) {b.GetBlock(1)[phiBCDOFs[ii]] = 0.;}
        for (int ii=0; ii<phi0BCDOFs.Size(); ii++) {b.GetBlock(1)[phi0BCDOFs[ii]] = 0.001;}
        b.SyncFromBlocks();

        mfem::ConstrainedOperator a1_constrained(&a1, uBCDOFs);
        mfem::RectangularConstrainedOperator a2_constrained(&a2, phiBCDOFs, uBCDOFs);
        mfem::RectangularConstrainedOperator a3_constrained(&a3, uBCDOFs, phiBCDOFs);
        mfem::ConstrainedOperator a4_constrained(&a4, phiBCDOFs);

        auto A_constrained = mfem::BlockOperator(block_offsets);
        A_constrained.SetBlock(0, 0, &a1_constrained);
        A_constrained.SetBlock(0, 1, &a2_constrained);
        A_constrained.SetBlock(1, 0, &a3_constrained);
        A_constrained.SetBlock(1, 1, &a4_constrained);

        auto a1_diag = mfem::Vector(u_space->GetVSize());
        auto a4_diag = mfem::Vector(phi_space->GetVSize());
        auto GlobalDiag = mfem::Vector(block_offsets[2]);

        a1.AssembleDiagonal(a1_diag);
        a4.AssembleDiagonal(a4_diag);

        for (int ii=0; ii<a1_diag.Size(); ii++) {GlobalDiag[ii] = a1_diag[ii];}
        for (int ii=0; ii<a4_diag.Size(); ii++) {GlobalDiag[ii+block_offsets[1]] = a4_diag[ii];}

        auto preconditioner = mfem::OperatorJacobiSmoother(GlobalDiag, GlobalBCDOFs, 1.0);
        auto solver = mfem::GMRESSolver();
        solver.SetOperator(A_constrained);
        solver.SetKDim(100);
        solver.SetPreconditioner(preconditioner);
        solver.iterative_mode = false;
        solver.SetRelTol(1e-4);
        solver.SetAbsTol(1e-8);
        solver.SetMaxIter(6000);
        solver.SetPrintLevel(0);
        solver.Mult(b, x);

        u.MakeRef(u_space, x.GetBlock(0), 0);
        phi.MakeRef(phi_space, x.GetBlock(1), 0);

        // Send u GridFunction to the second to last rank
        if (WorldSize >= 3)
        {
            MPI_Send(u.GetData(), u.Size(), MPI_DOUBLE, WorldRank-1, 1, MPI_COMM_WORLD);
        }
    }

    // All ranks must wait for the solution to continue
    MPI_Barrier(MPI_COMM_WORLD);

    bool RankWritePoints = false;
    bool RankWriteCells = false;
    bool RankWriteCellType = false;
    bool RankWriteCellMaterial = false;
    bool RankWriteField1 = false; // Field1 = displacements u
    bool RankWriteField2 = false; // Field2 = micro displacement gradient phi

    // Divide up the work depending on the number of ranks 
    // NOTE: errors are not properly thrown to avoid deadlocks
    switch (WorldSize)
    {
        case 1:
            RankWritePoints = true;
            RankWriteCells = true;
            RankWriteCellType = true;
            RankWriteCellMaterial = true;
            RankWriteField1 = true;
            RankWriteField2 = true;
            break;
        case 2:
            switch (WorldRank)
            {
                case 0:
                    RankWritePoints = true;
                    RankWriteCells = true;
                    RankWriteCellType = true;
                    RankWriteCellMaterial = true;
                    break;
                case 1: 
                    RankWriteField1 = true;
                    RankWriteField2 = true;
                    break;
            }
            break;
        case 3:
            switch (WorldRank)
            {
                case 0:
                    RankWritePoints = true;
                    RankWriteCells = true;
                    break;
                case 1:
                    RankWriteCellType = true;
                    RankWriteCellMaterial = true;
                    RankWriteField1 = true;
                    break;
                case 2:
                    RankWriteField2 = true;
                    break;
            }
            break;
        case 4:
            switch (WorldRank)
            {
                case 0:
                    RankWritePoints = true;
                    break;
                case 1:
                    RankWriteCells = true;
                    RankWriteCellType = true;
                    RankWriteCellMaterial = true;
                    break;
                case 2:
                    RankWriteField1 = true;
                    break;
                case 3:
                    RankWriteField2 = true;
                    break;
            }
            break;
        case 5:
            switch (WorldRank)
            {
                case 0:
                    RankWritePoints = true;
                    break;
                case 1:
                    RankWriteCells = true;
                    break;
                case 2:
                    RankWriteCellType = true;
                    RankWriteCellMaterial = true;
                    break;
                case 3:
                    RankWriteField1 = true;
                    break;
                case 4:
                    RankWriteField2 = true;
                    break;
            }
            break;
        case 6:
            switch (WorldRank)
            {
                case 0:
                    RankWritePoints = true;
                    break;
                case 1:
                    RankWriteCells = true;
                    break;
                case 2:
                    RankWriteCellType = true;
                    break;
                case 3:
                    RankWriteCellMaterial = true;
                    break;
                case 4:
                    RankWriteField1 = true;
                    break;
                case 5:
                    RankWriteField2 = true;
                    break;
            }
            break;
    }

    // Get the data for the rank and format it into a string
    std::ostringstream oss;
    if (RankWritePoints)
    {
        mfem::RefinedGeometry *RefG;
        mfem::DenseMatrix pmat;

        oss << "# vtk DataFile Version 3.0\n" << "Generated by VTKWriter\n";
        oss << "ASCII\n" << "DATASET UNSTRUCTURED_GRID\n";
        oss << "POINTS " << np << " double\n";
        for (int i = 0; i < mesh.GetNE(); i++)
        {
            RefG = mfem::GlobGeometryRefiner.Refine(mesh.GetElementBaseGeometry(i), 0, 1);
            mesh.GetElementTransformation(i)->Transform(RefG->RefPts, pmat);

            for (int j = 0; j < pmat.Width(); j++)
            {
                oss << pmat(0, j) << ' ';
                if (pmat.Height() > 1)
                {
                    oss << pmat(1, j) << ' ';
                    if (pmat.Height() > 2)
                    {
                        oss << pmat(2, j);
                    }
                    else
                    {
                        oss << 0.0;
                    }
                }
                else
                {
                    oss << 0.0 << ' ' << 0.0;
                }
                oss << '\n';
            }
        }
    }
    if (RankWriteCells)
    {
        mfem::RefinedGeometry *RefG;
        oss << "CELLS " << nc << ' ' << size << '\n';
        np = 0;
        for (int i = 0; i < mesh.GetNE(); i++)
        {
           mfem::Geometry::Type geom = mesh.GetElementBaseGeometry(i);
           int nv = mfem::Geometries.GetVertices(geom)->GetNPoints();
           RefG = mfem::GlobGeometryRefiner.Refine(geom, 0, 1);
           mfem::Array<int> &RG = RefG->RefGeoms;
     
           for (int j = 0; j < RG.Size(); )
           {
              oss << nv;
              for (int k = 0; k < nv; k++, j++)
              {
                 oss << ' ' << np + RG[j];
              }
              oss << '\n';
           }
           np += RefG->RefPts.GetNPoints();
        }
    }
    if (RankWriteCellType)
    {
        mfem::RefinedGeometry *RefG;
        oss << "CELL_TYPES " << nc << '\n';
        for (int i = 0; i < mesh.GetNE(); i++)
        {
           mfem::Geometry::Type geom = mesh.GetElementBaseGeometry(i);
           int nv = mfem::Geometries.GetVertices(geom)->GetNPoints();
           RefG = mfem::GlobGeometryRefiner.Refine(geom, 0, 1);
           mfem::Array<int> &RG = RefG->RefGeoms;
           int vtk_cell_type = mfem::VTKGeometry::Map[geom];
     
           for (int j = 0; j < RG.Size(); j += nv)
           {
              oss << vtk_cell_type << '\n';
           }
        }
    }
    if (RankWriteCellMaterial)
    {
        mfem::RefinedGeometry *RefG;
        oss << "CELL_DATA " << nc << '\n'
        << "SCALARS material int\n"
        << "LOOKUP_TABLE default\n";
        for (int i = 0; i < mesh.GetNE(); i++)
        {
            mfem::Geometry::Type geom = mesh.GetElementBaseGeometry(i);
            int nv = mfem::Geometries.GetVertices(geom)->GetNPoints();
            RefG = mfem::GlobGeometryRefiner.Refine(geom, 0, 1);
            int attr = mesh.GetAttribute(i);
            for (int j = 0; j < RefG->RefGeoms.Size(); j += nv)
            {
                oss << attr << '\n';
            }
        }
    }
    if (RankWriteField1)
    {   
        mfem::RefinedGeometry *RefG;
        mfem::DenseMatrix vval, pmat;
        std::string field_name = "u";

        oss << "POINT_DATA " << np << "\n";
        oss << "VECTORS " << field_name << " double\n";
        for (int i = 0; i < mesh.GetNE(); i++)
        {
            RefG = mfem::GlobGeometryRefiner.Refine(mesh.GetElementBaseGeometry(i), 0, 1);
            mfem::ElementTransformation* T = mesh.GetElementTransformation(i);
            u.GetVectorValues(*T, RefG->RefPts, vval, &pmat);
            for (int j = 0; j < vval.Width(); j++)
            {
                oss << vval(0, j) << ' ' << vval(1, j) << ' ';
                if (vval.Height() == 2)
                {
                    oss << 0.0;
                }
                else
                {
                    oss << vval(2, j);
                }
                oss << '\n';
            }
        }
    }
    if (RankWriteField2)
    {
        mfem::RefinedGeometry *RefG;
        mfem::DenseMatrix vval, pmat;
        std::string field_name = "phi";

        oss << "FIELD FieldData 1\n";
        oss << field_name << ' ' << 9 << ' ' << np << " double\n"; 

        for (int i = 0; i < mesh.GetNE(); i++)
        {
            RefG = mfem::GlobGeometryRefiner.Refine(mesh.GetElementBaseGeometry(i), 0, 1);
            mfem::ElementTransformation* T = mesh.GetElementTransformation(i);
            phi.GetVectorValues(*T, RefG->RefPts, vval, &pmat);
            
            if (mesh.SpaceDimension() == 2)
            {
                for (int j=0; j<vval.Width(); j++)
                {
                    oss << vval(0,j) << ' ' << vval(1,j) << ' ' << 0. << ' ';
                    oss << vval(2,j) << ' ' << vval(3,j) << ' ' << 0. << ' ';
                    oss << 0. << ' ' << 0. << ' ' << 0. << '\n';
                }
            }
            else if (mesh.SpaceDimension() == 3)
            {
                for (int j=0; j<vval.Width(); j++)
                {
                    oss << vval(0,j) << ' ' << vval(1,j) << ' ' << vval(2,j) << ' ';
                    oss << vval(3,j) << ' ' << vval(4,j) << ' ' << vval(5,j) << ' ';
                    oss << vval(6,j) << ' ' << vval(7,j) << ' ' << vval(8,j) << '\n';
                }
            }
        }
    }
    std::string RankData = oss.str();
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

    if (WorldSize >= 3  && WorldRank >= (WorldSize-2))
    {
       delete u_ec, phi_ec;
       delete u_space, phi_space;
    }

    MPI_Finalize();

    return 0;
}