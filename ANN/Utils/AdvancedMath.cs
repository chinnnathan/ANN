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
                Console.WriteLine("Incorrect Parameters: C(m,n) = A(m,k) * B(k,n)");
                Console.WriteLine("\tGiven: C(m,n) = A({0},{1}) * B({2},{3})", a.Length, a.GetLength(0), b.Length, b.GetLength(0));
                Environment.Exit(-1);
            }
            return c;
        }

        public static double[][] JMultiplyMatrix(double[,] a, double[,] b)
        {
            double[][] c = new double[a.GetLength(0)][];
            if (a.GetLength(1) == b.GetLength(0))
            {
                for (int i = 0; i < c.GetLength(0); i++)
                {
                    c[i] = new double[b.GetLength(1)];
                    for (int j = 0; j < c[i].Length; j++)
                    {
                        c[i][j] = 0;
                        for (int k = 0; k < a.GetLength(1); k++) // OR k<b.GetLength(0)
                            c[i][j] = c[i][j] + a[i, k] * b[k, j];
                    }
                }
            }
            else
            {
                Console.WriteLine("Incorrect Parameters: C(m,n) = A(m,k) * B(k,n)");
                Console.WriteLine("\tGiven: C(m,n) = A({0},{1}) * B({2},{3})", a.Length, a.GetLength(0), b.Length, b.GetLength(0));
                Environment.Exit(-1);
            }
            return c;
        }

        public static double[][] JSubtractMatrix(double[,]a, double[,]b)
        {
            double[][] c = new double[a.GetLength(0)][];
            if (a.GetLength(0) == b.GetLength(0) && a.GetLength(1) == b.GetLength(1))
                Parallel.For(0, a.GetLength(0), ai =>
                {
                    c[ai] = new double[a.GetLength(1)];
                    for (int i = 0; i < a.GetLength(1); i++)
                        c[ai][i] = a[ai, i] - b[ai, i];
                });
            else if (a.GetLength(1) == b.GetLength(0))
                Parallel.For(0, a.GetLength(0), ai =>
                {
                    c[ai] = new double[a.GetLength(1)];
                    for (int i = 0; i < a.GetLength(1); i++)
                        c[ai][i] = a[ai, i] - b[i, 0];
                });
            return c;
        }

        private static readonly Random random = new Random();
        public static double GetRandomRange(double min, double max)
        {
            var next = random.NextDouble();
            return min + (next * (max - min));
        }

        public static double[] GetRandomRanges(int length, double min, double max)
        {
            double[] ret = new double[length];
            for(int i = 0; i < length; i++)
            {
                ret[i] = GetRandomRange(min, max);
            }
            return ret;
        }

        public static double[,] ConvertToMatrix(double[] A, int L, int W)
        {
            if (L == -1)
                L = A.Length; //Vertical Matrix
            if (W == -1)
                W = A.Length; //Horizontal Matrix
            double[,] B = new double[L,W];
            Parallel.For(0, L, y =>
            {
                Parallel.For(0, W, x =>
                {
                    //B[y, x] = A[(y * W) + x];
                    try
                    {
                        B[y, x] = A[(x * L) + y];
                    }
                    catch
                    {
                        Console.WriteLine("B[{0},{1}] = A[{1}*{2}+{0}] -> L = {3}", y, x, W, L);
                    }
                });
            });
            return B;
        }
    }
}
