#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <iostream>

#include "cudaprint.h"

using namespace std;

int main(int argc, char** argv)
{

  cout << "Hello World!!!" << endl;
  
  cudaPrint();

  return 0;
}
