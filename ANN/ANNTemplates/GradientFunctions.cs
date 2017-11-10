using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ANN.ANNTemplates
{
    public static class GradientFunctions
    {
        public static double Tanh(double input)
        {
            if ((input < -15.0) || (input > 15.0)) return 0.0F; // approximation is correct to 30 decimals
            else if (input == 0) return 1.0F;
            else return (Math.Tanh(input) * (1 - Math.Tanh(input)));
        }

        public static double Logistic(double input)
        {
            return input * (1 - input);
        }

        public static double SoftMax(double target, double actual)
        {
            return actual - target;
        }

        public static double SumSquaredError(double target, double actual)
        {
            return (actual - target);
        }
    }
}
