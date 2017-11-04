using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ANN.Utils
{
    public static class AdvancedMath
    {
        public static bool MatrixMultiplication(double[][] A, double[][] B, out double[][] C)
        {
            if (A[0].Length != B.Length)
            {
                Console.WriteLine("Incorrect Parameters: C(m,n) = A(m,k) * B(k,n)");
                Console.WriteLine("\tGiven: C(m,n) = A({0},{1}) * B({2},{3})", A.Length, A[0].Length, B.Length, B[0].Length);
                C = new double[1][];
                return false;
            }
            int M = A.Count();
            int N = B[0].Count();
            int K = B.Count();
            C = new double[M][];


            /*Parallel.ForEach(A, () => 0.0d,
                (x, loopstate, partialresult)=>
                {
                    Parallel.For(0, K, index =>
                    {

                    });
                    Console.WriteLine("{0}", x);
                    return 1.0d;
                });*/
            return true;
        }

        public static double[,] MultiplyMatrix(double[,] a, double[,] b)
        {
            double[,] c = new double[a.GetLength(0), b.GetLength(1)];
            if (a.GetLength(1) == b.GetLength(0))
            {
                for (int i = 0; i < c.GetLength(0); i++)
                {
                    for (int j = 0; j < c.GetLength(1); j++)
                    {
                        c[i, j] = 0;
                        for (int k = 0; k < a.GetLength(1); k++) // OR k<b.GetLength(0)
                            c[i, j] = c[i, j] + a[i, k] * b[k, j];
                    }
                }
            }
            else
            {
                Console.WriteLine("\n Number of columns in First Matrix should be equal to Number of rows in Second Matrix.");
                Console.WriteLine("\n Please re-enter correct dimensions.");
                Environment.Exit(-1);
            }
            return c;
        }
    }
}
