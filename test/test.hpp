#ifndef GRB_TEST_HPP
#define GRB_TEST_HPP

#include <vector>
#include <iostream>

template <typename T>
void BOOST_ASSERT_FLOAT( const T lhs,
                         const T rhs,
                         const T tol=0.001 )
{
  if (rhs==0)
    BOOST_ASSERT( fabs(lhs)<tol );
  else if (lhs == 0)
    BOOST_ASSERT( fabs(rhs)<tol );
  else
    BOOST_ASSERT( fabs(lhs-rhs)<tol );
}

template <typename T>
bool assert_float( T lhs,
                   T rhs,
                   T tol = 0.001 )
{
  if (rhs==0)
    return fabs(lhs)<tol;
  else if (lhs == 0)
    return fabs(rhs)<tol;
  else
    return fabs(lhs-rhs)<tol;
}

template <typename T, typename S, typename L>
void BOOST_ASSERT_LIST( const T* lhs, 
                        const S* rhs, 
                        L length=5 )
{
  for( L i=0; i<length; i++ )
    BOOST_ASSERT( lhs[i] == rhs[i] );
}

template <typename T, typename S, typename L>
void BOOST_ASSERT_LIST( const std::vector<T>& lhs, 
                        const S* rhs, 
                        L length=5 )
{
  //length = lhs.size();
  for( L i=0; i<length; i++ )
    BOOST_ASSERT( lhs[i] == rhs[i] );
}

template <typename T, typename S, typename L>
void BOOST_ASSERT_LIST(const std::vector<T>& lhs,
                       const std::vector<S>& rhs, 
                       L                     length = 5) {
  int flag = 0;
  for (L i = 0; i < length; i++) {
    if (lhs[i] != rhs[i] && flag == 0) {
      std::cout << "\nINCORRECT: [" << (unsigned long) i << "]: ";
      std::cout << rhs[i] << " != " << lhs[i] << "\nresult[...";

      for (size_t j = (i >= 5) ? i - 5 : 0; (j < i + 5) && (j < length); j++)
        std::cout << rhs[j] << ", ";
      std::cout << "...]\nlhs[...";

      for (size_t j = (i >= 5) ? i - 5 : 0; (j < i + 5) && (j < length); j++)
        std::cout << lhs[j] << ", ";
      std::cout << "...]";
    }
    if (lhs[i] != rhs[i])
      flag++;
  }
  std::cout << "\n";
  if (flag == 0)
    std::cout << "CORRECT\n";
  else
    std::cout << flag << " errors occurred.\n";
}

template <typename T, typename S, typename L>
void BOOST_ASSERT_LIST(const T*              lhs, 
                       const std::vector<S>& rhs,
                       L                     length = 5) {
  int flag = 0;
  for (L i = 0; i < length; i++) {
    if (lhs[i] != rhs[i] && flag == 0) {
      std::cout << "\nINCORRECT: [" << (unsigned long) i << "]: ";
      std::cout << rhs[i] << " != " << lhs[i] << "\nresult[...";

      for (size_t j = (i >= 5) ? i - 5 : 0; (j < i + 5) && (j < length); j++)
        std::cout << rhs[j] << ", ";
      std::cout << "...]\nlhs[...";

      for (size_t j = (i >= 5) ? i - 5 : 0; (j < i + 5) && (j < length); j++)
        std::cout << lhs[j] << ", ";
      std::cout << "...]";
    }
    if (lhs[i] != rhs[i])
      flag++;
  }
  std::cout << "\n";
  if (flag == 0)
    std::cout << "CORRECT\n";
  else
    std::cout << flag << " errors occurred.\n";
}

template <typename T, typename S, typename L>
void BOOST_ASSERT_LIST_FLOAT( const T* lhs, 
                              const std::vector<S>& rhs,
                              L length=5 )
{
  int flag = 0;
  //length = rhs.size();
  for( L i=0; i<length; i++ )
  {
    if( !assert_float(lhs[i], rhs[i]) && flag==0 )
    {
      std::cout << "\nINCORRECT: [" << (unsigned long) i << "]: ";
      std::cout << rhs[i] << " != " << lhs[i] << "\nresult[...";

      for( size_t j = (i >= 5) ? i - 5 : 0; (j < i + 5) && (j < length); j++ )
        std::cout << rhs[j] << ", ";
      std::cout << "...]\nlhs[...";

      for( size_t j = (i >= 5) ? i - 5 : 0; (j < i + 5) && (j < length); j++ )
        std::cout << lhs[j] << ", ";
      std::cout << "...]";
    }
    if( !assert_float(lhs[i], rhs[i]) )
      flag += 1;
  }
  std::cout << "\n";
  if( flag==0 )
    std::cout << "CORRECT\n";
  else
    std::cout << flag << " errors occured.\n";
}
#endif  // GRB_TEST_HPP
