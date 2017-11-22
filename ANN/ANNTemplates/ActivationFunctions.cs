using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ANN.ANNTemplates
{
    public class ActivationFunctions
    {
        public delegate double Del(double input);
        public delegate double Del1(double input, double[] other);
        public delegate double Del2(double target, double actual);

        public static double Tanh(double input)
        {
            if (input < -20.0) return -1.0F; // approximation is correct to 30 decimals
            else if (input > 20.0) return 1.0F;
            else return (float)Math.Tanh(input);
        }

        public static double SumSquaredError(double target, double actual)
        {
            return 0.5 * Math.Pow(target - actual, 2);
        }

        public static double SumSquaredError(double[] errors)
        {
            double error = 0;
            object lockobject = new object();
            Parallel.ForEach(
                errors,
                () => 0.0d,
                (x, loopstate, partialresult) =>
                {
                    return Math.Pow(x,2) + partialresult;
                },
                (localPartialSum) =>
                {
                    lock (lockobject)
                    {
                        error += localPartialSum;
                    }
                }
            );

            return 0.5 * error;
        }

        public static double Distance(double target, double actual)
        {
            return target - actual;
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

        //need a placeholder activation function for input
        public static double Input(double input)
        {
            return input;
        }

        public static double Context(double input)
        {
            return input;
        }
    }
}
