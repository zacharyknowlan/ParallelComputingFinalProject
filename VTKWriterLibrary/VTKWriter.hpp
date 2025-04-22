#ifndef VTK_WRITER_FOR_MFEM
#define VTK_WRITER_FOR_MFEM

#include <sstream>
#include <string>
#include "mfem.hpp"

class VTKWriter
{
    private:
        std::ostringstream oss;
        mfem::Mesh &mesh;
        int num_points, num_elements, element_data_size;

    public:

        VTKWriter(mfem::Mesh& m) : mesh(m) {}

        // If SetupWriteParameters is not called for a rank, this can be used
        void SetValuesForRank(int np, int nc, int size); 

        // Setup the required parameters for file writing
        void SetupWriteParameters();

        // Return the number of points belonging to *this
        inline int GetNumPoints() {return num_points;}

        // Return the number of elements belonging to *this
        inline int GetNumElements() {return num_elements;}

        // Return the element data size belonging to *this
        inline int GetElementDataSize() {return element_data_size;}

        // Determine the work to be done by each rank
        void DetermineRankWork(int WorldSize, int WorldRank, bool& RankWritePoints, bool& RankWriteElements, 
                                bool& RankWriteElementTypes, bool& RankWriteElementMaterials, bool& RankWriteField1,
                                bool& RankWriteField2);

        // Write the required VTK header information
        void WriteHeader();
        
        // Write the nodes field as mfem::RefinedGeometry
        void WriteNodes();

        // Write the elements corresponding to nodes defined in WriteNodes
        void WriteElements();

        // Write the element types (triangles or quadrilaterals)
        void WriteElementTypes();

        // Write the element materials (from mesh attributes)
        void WriteElementMaterials();

        // Write a vector field (used for displacements)
        void WriteVectorField(mfem::GridFunction& gridfunc, const std::string& field_name);

        // Write a Tensor field (used for micro displacement gradient)
        void WriteTensorField(mfem::GridFunction& gridfunc, const std::string& field_name);

        // Get the rank's data as a string
        const std::string GetData() const {return oss.str();}
};


#endif