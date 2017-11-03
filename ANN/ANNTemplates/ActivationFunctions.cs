using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ANN.ANNTemplates
{
    class ActivationFunctions
    {
        public static double Tanh(double input)
        {
            if (input < -20.0) return -1.0F; // approximation is correct to 30 decimals
            else if (input > 20.0) return 1.0F;
            else return (float)Math.Tanh(input);
        }

        public static double SoftMax(double input, double[] other)
        {
            double retval = 0;
            double denominator = 0;
            object lockobject = new object();
            Parallel.ForEach(
                other,
                () => 0.0d,
                (n, loopState, partialResult) =>
                {
                    return Math.Exp(n) + partialResult;
                },
                (localPartialSum) =>
                {
                    lock (lockobject)
                    {
                        denominator += localPartialSum;
                    }
                });

            try
            {
                retval = Math.Exp(input) / denominator;
            }
            catch
            {
                retval = -999;
            }
            return retval;
        }
    }
}
