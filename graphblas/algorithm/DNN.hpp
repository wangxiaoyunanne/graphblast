#ifndef GRAPHBLAS_ALGORITHM_DNN_HPP_
#define GRAPHBLAS_ALGORITHM_DNN_HPP_
#include <limits>
#include <vector>
#include <string> 
//#include "graphblas/algorithm/test_gdnn.hpp"
#include "graphblas/backend/cuda/util.hpp"

namespace graphblas {
namespace algorithm {
Matrix <float> DNN (Matrix <float> *Y, 
                    Matrix <float> *W, 
                    Descriptor*          desc)
{
    Index Y_nrows, Y_ncols ; 
    CHECK(Y->nrows(&Y_nrows));  
    CHECK(Y->ncols(&Y_ncols));

    Matrix <float> result (Y_nrows,Y_ncols); 
    mxm<float,float,float,float > (&result,GrB_NULL, GrB_NULL,PlusMultipliesSemiring<float>() ,&Y,&W, desc) ;
    //add reul here. 


    return result;
}



}// namspace algorithm
} //namespace graphblas

